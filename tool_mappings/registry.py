"""tool_mappings/registry.py — Thread-safe store and resolver for MappingDefinitions.

Resolution order for a given model name:
  1. User-defined mappings (~/.mlxr/tool_mappings.json), in definition order
  2. Built-in mappings (builtin_mappings.json), in definition order
Within each tier the first glob that matches (case-insensitive) wins.
"""
from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Optional

from .schema import MappingDefinition

log = logging.getLogger("mlxr")

BUILTIN_MAPPINGS_PATH = Path(__file__).parent / "builtin_mappings.json"
USER_MAPPINGS_PATH = Path.home() / ".mlxr" / "tool_mappings.json"

# File envelope schema
_ENVELOPE_VERSION = "1"


def _load_file(path: Path) -> list[MappingDefinition]:
    """Load and validate all MappingDefinitions from a JSON or YAML file."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("tool_mappings: cannot read %s: %s", path, exc)
        return []

    # Parse — try YAML first (optional dep), fall back to JSON
    try:
        raw: dict = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
            raw = yaml.safe_load(text)
        except Exception as exc:
            log.warning("tool_mappings: cannot parse %s: %s", path, exc)
            return []

    if not isinstance(raw, dict):
        log.warning("tool_mappings: %s root must be a JSON object", path)
        return []

    items = raw.get("mappings", [])
    if not isinstance(items, list):
        log.warning("tool_mappings: %s 'mappings' must be an array", path)
        return []

    definitions: list[MappingDefinition] = []
    for entry in items:
        try:
            defn = MappingDefinition.from_dict(entry)
            errs = defn.validate()
            if errs:
                log.warning("tool_mappings: %s id=%r has errors: %s", path, entry.get("id"), errs)
                continue
            definitions.append(defn)
        except Exception as exc:
            log.warning("tool_mappings: skipping malformed entry in %s: %s", path, exc)

    return definitions


def _save_file(path: Path, definitions: list[MappingDefinition]) -> None:
    """Serialise definitions to path as a versioned JSON envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "version": _ENVELOPE_VERSION,
        "mappings": [d.to_dict() for d in definitions],
    }
    path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")


class MappingRegistry:
    """Thread-safe registry of MappingDefinitions.

    Typical usage::

        registry = MappingRegistry()
        mapping = registry.resolve("mlx-community/Qwen3-4B-4bit")
        if mapping:
            parser = MappedToolCallParser(enabled=True, mapping=mapping)
        else:
            parser = ToolCallParser(enabled=True)   # legacy fallback
    """

    def __init__(
        self,
        builtin_path: Path = BUILTIN_MAPPINGS_PATH,
        user_path: Path = USER_MAPPINGS_PATH,
    ) -> None:
        self._builtin_path = builtin_path
        self._user_path = user_path
        self._lock = Lock()
        self._builtin: list[MappingDefinition] = []
        self._user: list[MappingDefinition] = []
        self._load()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        builtin = _load_file(self._builtin_path)
        for d in builtin:
            d.is_builtin = True
        user = _load_file(self._user_path)
        with self._lock:
            self._builtin = builtin
            self._user = user
        log.info(
            "tool_mappings: loaded %d builtin + %d user mapping(s)",
            len(builtin), len(user),
        )

    def _snapshot(self) -> tuple[list[MappingDefinition], list[MappingDefinition]]:
        """Return copies of both lists without holding the lock."""
        with self._lock:
            return list(self._user), list(self._builtin)

    # ── Public read API ───────────────────────────────────────────────────────

    def resolve(self, model_name: str) -> Optional[MappingDefinition]:
        """Return the best MappingDefinition for *model_name*, or None.

        Matching is case-insensitive fnmatch against the full repo-id string.
        User mappings are checked first; built-ins are the fallback.
        Returns None when no glob matches — callers should fall back to the
        legacy ToolCallParser.
        """
        key = model_name.lower()
        user, builtin = self._snapshot()
        for defn in (*user, *builtin):
            for glob in defn.match_globs:
                if fnmatch.fnmatch(key, glob.lower()):
                    return defn
        return None

    def all_mappings(self) -> list[dict]:
        """Return all mappings (user then builtin) serialised as dicts.

        Each dict includes an ``is_builtin`` boolean for the UI.
        """
        user, builtin = self._snapshot()
        result = []
        for defn in user:
            d = defn.to_dict()
            d["is_builtin"] = False
            result.append(d)
        for defn in builtin:
            d = defn.to_dict()
            d["is_builtin"] = True
            result.append(d)
        return result

    def get_by_id(self, mapping_id: str) -> Optional[MappingDefinition]:
        """Fetch a specific mapping by its id (user takes precedence)."""
        user, builtin = self._snapshot()
        for defn in (*user, *builtin):
            if defn.id == mapping_id:
                return defn
        return None

    def resolve_info(self, model_name: str) -> dict:
        """Return resolution metadata for /api/tool-mappings/resolve/<model>."""
        key = model_name.lower()
        user, builtin = self._snapshot()
        for tier_name, tier in (("user", user), ("builtin", builtin)):
            for defn in tier:
                for glob in defn.match_globs:
                    if fnmatch.fnmatch(key, glob.lower()):
                        return {
                            "resolved": True,
                            "mapping_id": defn.id,
                            "mapping_name": defn.name,
                            "matched_glob": glob,
                            "tier": tier_name,
                        }
        return {"resolved": False}

    # ── Mutation API ──────────────────────────────────────────────────────────

    def save_user_mapping(self, defn: MappingDefinition) -> None:
        """Persist a user mapping (upsert by id). Thread-safe."""
        defn.is_builtin = False
        with self._lock:
            updated = [d for d in self._user if d.id != defn.id]
            updated.append(defn)
            self._user = updated
            snapshot = list(updated)
        _save_file(self._user_path, snapshot)
        log.info("tool_mappings: saved user mapping id=%r", defn.id)

    def delete_user_mapping(self, mapping_id: str) -> bool:
        """Remove a user mapping by id. Returns True if it existed."""
        with self._lock:
            before = len(self._user)
            self._user = [d for d in self._user if d.id != mapping_id]
            removed = len(self._user) < before
            snapshot = list(self._user)
        if removed:
            _save_file(self._user_path, snapshot)
            log.info("tool_mappings: deleted user mapping id=%r", mapping_id)
        return removed

    def import_from_file(self, path: Path) -> list[MappingDefinition]:
        """Load definitions from a JSON/YAML file and merge into user store.

        Existing user mappings with the same id are overwritten.
        Returns the list of successfully imported definitions.
        """
        incoming = _load_file(path)
        for defn in incoming:
            self.save_user_mapping(defn)
        return incoming

    def import_from_text(self, text: str, filename: str = "upload.json") -> list[MappingDefinition]:
        """Import from raw text (for API uploads that don't land on disk first)."""
        import tempfile
        suffix = ".yaml" if filename.endswith((".yaml", ".yml")) else ".json"
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix,
                                        encoding="utf-8", delete=False) as fh:
            fh.write(text)
            tmp = Path(fh.name)
        try:
            return self.import_from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def export_to_json(self, mapping_ids: Optional[list[str]] = None) -> str:
        """Return all (or selected) user mappings serialised as a JSON string."""
        user, _ = self._snapshot()
        if mapping_ids is not None:
            user = [d for d in user if d.id in mapping_ids]
        envelope = {
            "version": _ENVELOPE_VERSION,
            "mappings": [d.to_dict() for d in user],
        }
        return json.dumps(envelope, indent=2, ensure_ascii=False)

    def reload(self) -> None:
        """Re-read both files from disk. Safe to call from any thread."""
        self._load()
