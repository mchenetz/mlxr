"""tool_mappings/schema.py — Data model for tool-call format mapping definitions.

A MappingDefinition describes, for a given model family:
  - How to detect tool-call blocks in the token stream (detection)
  - How to parse the body once delimited (body_format)
  - How to encode the result for the wire format (output)
  - Glob patterns that identify which models use this format (match_globs)
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Leaf sub-structures ───────────────────────────────────────────────────────

@dataclass
class TagPair:
    """An (open, close) delimiter pair used to detect tool-call blocks."""
    open: str
    close: str

    @classmethod
    def from_dict(cls, d: dict) -> "TagPair":
        return cls(open=d["open"], close=d["close"])

    def to_dict(self) -> dict:
        return {"open": self.open, "close": self.close}


@dataclass
class BareJsonSignal:
    """Key-presence heuristic for bare-JSON tool call detection."""
    required_keys: list[str] = field(default_factory=lambda: ["name"])
    argument_keys: list[str] = field(default_factory=lambda: ["arguments", "parameters"])

    @classmethod
    def from_dict(cls, d: dict) -> "BareJsonSignal":
        return cls(
            required_keys=d.get("required_keys", ["name"]),
            argument_keys=d.get("argument_keys", ["arguments", "parameters"]),
        )

    def to_dict(self) -> dict:
        return {"required_keys": self.required_keys, "argument_keys": self.argument_keys}


@dataclass
class Detection:
    """How to detect that a token stream contains a tool call in this format.

    strategy:
      - "tag_pair"              — look for open/close tag delimiters
      - "bare_json"             — look for a raw JSON object with key signals
      - "tag_pair_and_bare_json" — try tag pairs first; fall back to bare-JSON flush
    """
    strategy: str  # "tag_pair" | "bare_json" | "tag_pair_and_bare_json"
    tag_pairs: list[TagPair] = field(default_factory=list)
    bare_json_signal: Optional[BareJsonSignal] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(
            strategy=d.get("strategy", "tag_pair"),
            tag_pairs=[TagPair.from_dict(tp) for tp in d.get("tag_pairs", [])],
            bare_json_signal=BareJsonSignal.from_dict(d["bare_json_signal"])
            if "bare_json_signal" in d else None,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"strategy": self.strategy}
        if self.tag_pairs:
            out["tag_pairs"] = [tp.to_dict() for tp in self.tag_pairs]
        if self.bare_json_signal:
            out["bare_json_signal"] = self.bare_json_signal.to_dict()
        return out


@dataclass
class BodyFormat:
    """How to parse the captured body text into (name, args) pairs.

    type values:
      - "json_object"       — body is a JSON object (or array of objects)
      - "deepseek_sep"      — DeepSeek-style: type<sep>name\\n```json\\n{...}\\n```
      - "xml_function_tag"  — Qwen3 XML: <function=NAME><parameter=KEY>VALUE</parameter>
      - "fenced_json"       — markdown-fenced JSON, strip fence then treat as json_object
    """
    type: str = "json_object"
    json_name_key: str = "name"
    json_args_keys: list[str] = field(default_factory=lambda: ["arguments", "parameters"])
    json_nested_function_key: Optional[str] = None   # e.g. "function" for OpenAI-nested bodies
    deepseek_separator: Optional[str] = None
    xml_function_open_re: Optional[str] = None
    xml_parameter_open_re: Optional[str] = None
    fenced_languages: list[str] = field(default_factory=lambda: ["json", "xml", ""])

    @classmethod
    def from_dict(cls, d: dict) -> "BodyFormat":
        return cls(
            type=d.get("type", "json_object"),
            json_name_key=d.get("json_name_key", "name"),
            json_args_keys=d.get("json_args_keys", ["arguments", "parameters"]),
            json_nested_function_key=d.get("json_nested_function_key"),
            deepseek_separator=d.get("deepseek_separator"),
            xml_function_open_re=d.get("xml_function_open_re"),
            xml_parameter_open_re=d.get("xml_parameter_open_re"),
            fenced_languages=d.get("fenced_languages", ["json", "xml", ""]),
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"type": self.type}
        if self.json_name_key != "name":
            out["json_name_key"] = self.json_name_key
        if self.json_args_keys != ["arguments", "parameters"]:
            out["json_args_keys"] = self.json_args_keys
        if self.json_nested_function_key is not None:
            out["json_nested_function_key"] = self.json_nested_function_key
        if self.deepseek_separator is not None:
            out["deepseek_separator"] = self.deepseek_separator
        if self.xml_function_open_re is not None:
            out["xml_function_open_re"] = self.xml_function_open_re
        if self.xml_parameter_open_re is not None:
            out["xml_parameter_open_re"] = self.xml_parameter_open_re
        if self.fenced_languages != ["json", "xml", ""]:
            out["fenced_languages"] = self.fenced_languages
        return out


@dataclass
class OutputConfig:
    """How to encode extracted tool calls for the wire format."""
    arguments_encoding: str = "json_string"  # "json_string" (OpenAI) | "json_object" (Anthropic)
    id_prefix: str = "call_"

    @classmethod
    def from_dict(cls, d: dict) -> "OutputConfig":
        return cls(
            arguments_encoding=d.get("arguments_encoding", "json_string"),
            id_prefix=d.get("id_prefix", "call_"),
        )

    def to_dict(self) -> dict:
        return {"arguments_encoding": self.arguments_encoding, "id_prefix": self.id_prefix}


# ── Top-level definition ──────────────────────────────────────────────────────

@dataclass
class MappingDefinition:
    """Complete description of a model family's tool-call wire format.

    Fields
    ------
    id           Unique slug (e.g. "qwen3-hermes", "deepseek-v3")
    name         Human-readable display name
    match_globs  fnmatch globs against the lowercased HF repo-id; first match wins
    detection    How to detect tool-call blocks in the token stream
    body_format  How to parse the captured body text
    output       How to encode the result for the OpenAI/Anthropic wire format
    description  Optional free-form description shown in the UI
    version      Semver string for this definition
    probe_config Optional hints for the AI mapping creator (probe tool, system prompt)
    is_builtin   Set by the registry; not serialised to user files
    """
    id: str
    name: str
    match_globs: list[str]
    detection: Detection
    body_format: BodyFormat = field(default_factory=BodyFormat)
    output: OutputConfig = field(default_factory=OutputConfig)
    description: str = ""
    version: str = "1.0.0"
    probe_config: dict = field(default_factory=dict)
    is_builtin: bool = field(default=False, compare=False)

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "MappingDefinition":
        """Deserialise from a plain dict (JSON/YAML origin)."""
        if "id" not in d or "name" not in d:
            raise ValueError("MappingDefinition requires 'id' and 'name'")
        if "detection" not in d:
            raise ValueError("MappingDefinition requires 'detection'")
        return cls(
            id=d["id"],
            name=d["name"],
            match_globs=d.get("match_globs", []),
            detection=Detection.from_dict(d["detection"]),
            body_format=BodyFormat.from_dict(d.get("body_format", {})),
            output=OutputConfig.from_dict(d.get("output", {})),
            description=d.get("description", ""),
            version=d.get("version", "1.0.0"),
            probe_config=d.get("probe_config", {}),
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (excludes is_builtin)."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "match_globs": self.match_globs,
            "detection": self.detection.to_dict(),
            "body_format": self.body_format.to_dict(),
            "output": self.output.to_dict(),
            **({"probe_config": self.probe_config} if self.probe_config else {}),
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of validation error messages; empty list means valid."""
        errors: list[str] = []
        if not self.id:
            errors.append("id must not be empty")
        elif not re.match(r"^[\w\-]+$", self.id):
            errors.append("id must contain only word chars and hyphens")
        if not self.name:
            errors.append("name must not be empty")
        if not self.match_globs:
            errors.append("match_globs must not be empty")
        strat = self.detection.strategy
        if strat not in ("tag_pair", "bare_json", "tag_pair_and_bare_json"):
            errors.append(f"detection.strategy '{strat}' is not recognised")
        if strat in ("tag_pair", "tag_pair_and_bare_json") and not self.detection.tag_pairs:
            errors.append("detection.tag_pairs is required for strategy 'tag_pair'")
        if strat in ("bare_json", "tag_pair_and_bare_json") and self.detection.bare_json_signal is None:
            errors.append("detection.bare_json_signal is required for strategy 'bare_json'")
        if self.body_format.type not in ("json_object", "deepseek_sep", "xml_function_tag", "fenced_json"):
            errors.append(f"body_format.type '{self.body_format.type}' is not recognised")
        if self.output.arguments_encoding not in ("json_string", "json_object"):
            errors.append(f"output.arguments_encoding '{self.output.arguments_encoding}' is not recognised")
        # Validate any XML regexes compile
        for attr in ("xml_function_open_re", "xml_parameter_open_re"):
            pat = getattr(self.body_format, attr)
            if pat:
                try:
                    re.compile(pat)
                except re.error as exc:
                    errors.append(f"body_format.{attr} is not a valid regex: {exc}")
        return errors

    def make_id(self) -> str:
        """Generate a collision-resistant id from the name."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return slug or f"mapping-{uuid.uuid4().hex[:8]}"
