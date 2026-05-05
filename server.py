"""MLXr — management server for Apple MLX model engine.

Exposes a JSON API and serves a dashboard for loading MLX language models,
running streaming inference, and inspecting host + engine state.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator, Optional

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

log = logging.getLogger("mlxr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Suppress uvicorn access-log spam from the dashboard's polling loops.
# /api/status and /api/hf/downloads are hit every 2-3 seconds; they clutter
# the log and make real events (tool calls, errors, model loads) hard to find.
_POLL_PATHS = frozenset(["/api/status", "/api/hf/downloads"])

class _PollFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in _POLL_PATHS)

logging.getLogger("uvicorn.access").addFilter(_PollFilter())

STATIC_DIR = Path(__file__).parent / "static"
SETTINGS_PATH = Path(os.environ.get("MLXR_SETTINGS_PATH", str(Path.home() / ".mlxr" / "settings.json")))

# Only these packages may be upgraded via the dashboard. Prevents the API from
# being abused to `pip install` arbitrary things.
ALLOWED_UPGRADE_PACKAGES = ("mlx", "mlx-lm", "huggingface_hub", "transformers")
PACKAGE_TO_MODULE = {
    "mlx": "mlx",
    "mlx-lm": "mlx_lm",
    "huggingface_hub": "huggingface_hub",
    "transformers": "transformers",
}
# run.sh watches for this exit code and re-launches the server.
RESTART_EXIT_CODE = 42

SUGGESTED_MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
]


MLXR_MAX_MODELS = max(1, int(os.environ.get("MLXR_MAX_MODELS", "1")))


@dataclass
class LoadedModel:
    name: str
    loaded_at: float
    model: Any
    tokenizer: Any
    context_length: int = 32768   # auto-detected from model config at load time
    generations: int = 0
    total_tokens: int = 0
    last_used: float = field(default_factory=time.time)
    # Per-generation perf metrics updated after each inference call.
    last_ttft: Optional[float] = None   # Time To First Token (seconds)
    last_tps: Optional[float] = None    # tokens / second (generation throughput)
    is_vlm: bool = False


def _detect_context_length(tokenizer: Any, model_name: str) -> int:
    """Return the model's context window size.

    Priority:
    1. tokenizer.model_max_length — set correctly by most modern tokenizers
       but some (e.g. very old or debug tokenizers) set it to sys.maxsize.
    2. config.json in the HF cache — check max_position_embeddings at
       top level and inside text_config (VLMs nest it there).
    3. Fallback: 32768 (conservative but safe for all current Apple Silicon).
    """
    # 1. Tokenizer attribute
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and isinstance(tok_max, int) and tok_max < 10_000_000:
        return tok_max

    # 2. HF cache config.json
    try:
        from huggingface_hub import try_to_load_from_cache
        config_path = try_to_load_from_cache(model_name, "config.json")
        if config_path:
            import json as _json
            cfg = _json.loads(open(config_path).read())
            for source in (cfg, cfg.get("text_config", {})):
                for key in ("max_position_embeddings", "n_positions", "seq_length"):
                    val = source.get(key)
                    if val and isinstance(val, int):
                        return val
    except Exception as e:
        log.debug("context length detection failed: %s", e)

    return 32768


def _clear_mlx_cache() -> None:
    """Ask MLX to release Metal buffer cache. Silently ignored if MLX is not installed."""
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass


# ---- VLM helpers ---------------------------------------------------------

_VLM_CONFIG_KEYS = frozenset([
    "vision_config", "visual_config", "image_token_id",
    "num_image_tokens", "pixel_values_videos", "visual_token_id",
    "image_seq_length", "vision_tower",
])


def _is_vlm_model(name: str) -> bool:
    """Check model config.json for VLM indicators. Returns False on any error."""
    try:
        from huggingface_hub import try_to_load_from_cache
        cfg_path = try_to_load_from_cache(name, "config.json")
        if cfg_path:
            cfg = json.loads(open(cfg_path).read())
            return bool(_VLM_CONFIG_KEYS & set(cfg.keys()))
    except Exception:
        pass
    return False


def _load_llm(name: str) -> LoadedModel:
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(name)
    auto_ctx = _detect_context_length(tokenizer, name)
    saved_ctx = settings.get_model(name).get("context_length")
    ctx = int(saved_ctx) if saved_ctx else auto_ctx
    log.info("Context length for %s: %d%s", name, ctx, " (overridden)" if saved_ctx else " (auto)")
    return LoadedModel(name=name, loaded_at=time.time(), model=model, tokenizer=tokenizer, context_length=ctx)


def _load_vlm(name: str) -> LoadedModel:
    try:
        from mlx_vlm import load as vlm_load
    except ImportError:
        raise RuntimeError(
            f"{name} appears to be a VLM but mlx-vlm is not installed. "
            "Run: pip install mlx-vlm"
        )
    model, processor = vlm_load(name)
    tokenizer = getattr(processor, "tokenizer", processor)
    auto_ctx = _detect_context_length(tokenizer, name)
    saved_ctx = settings.get_model(name).get("context_length")
    ctx = int(saved_ctx) if saved_ctx else auto_ctx
    log.info("VLM context length for %s: %d", name, ctx)
    return LoadedModel(
        name=name, loaded_at=time.time(),
        model=model, tokenizer=processor,   # store full processor as tokenizer
        context_length=ctx, is_vlm=True,
    )


class EnginePool:
    """LRU pool of loaded MLX models.

    MLXR_MAX_MODELS controls capacity (default 1, preserving backward compat).
    A single _gen_lock serialises ALL inference across ALL models — MLX/Metal
    crashes with concurrent eval() calls.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        # Single gen lock shared across all models — Metal requirement.
        self._gen_lock = Lock()
        # canonical name → LoadedModel, insertion order = load order
        self._models: dict[str, LoadedModel] = {}
        self._loading: Optional[str] = None

    # ---- properties (backward compat) ------------------------------------

    @property
    def gen_lock(self) -> Lock:
        return self._gen_lock

    @property
    def current(self) -> Optional[LoadedModel]:
        """Most recently used model (first in reverse-insertion order)."""
        with self._lock:
            if not self._models:
                return None
            # Return the model with the highest last_used timestamp.
            return max(self._models.values(), key=lambda m: m.last_used)

    @property
    def loading(self) -> Optional[str]:
        return self._loading

    # ---- public API ------------------------------------------------------

    def get(self, name: str) -> Optional[LoadedModel]:
        """Look up by canonical name or alias."""
        if not name:
            return None
        with self._lock:
            if name in self._models:
                return self._models[name]
            # Check aliases
            for m in self._models.values():
                saved = settings.get_model(m.name)
                if saved.get("alias") == name:
                    return m
        return None

    def loaded_models(self) -> list[LoadedModel]:
        """All loaded models, newest-used first."""
        with self._lock:
            return sorted(self._models.values(), key=lambda m: m.last_used, reverse=True)

    def load(self, name: str) -> LoadedModel:
        """Load a model into the pool, evicting LRU if at capacity."""
        with self._lock:
            if self._loading:
                raise RuntimeError(f"Another load is in progress: {self._loading}")
            if name in self._models:
                # Already loaded — bump last_used and return.
                m = self._models[name]
                m.last_used = time.time()
                return m
            self._loading = name

        try:
            # Evict LRU models until we're below capacity.
            with self._lock:
                while len(self._models) >= MLXR_MAX_MODELS:
                    lru_name = min(self._models, key=lambda k: self._models[k].last_used)
                    log.info("Pool full — evicting LRU model %s", lru_name)
                    del self._models[lru_name]

            log.info("Loading model %s", name)
            t0 = time.time()
            is_vlm = _is_vlm_model(name)
            if is_vlm:
                log.info("Detected VLM architecture for %s", name)
                loaded = _load_vlm(name)
            else:
                loaded = _load_llm(name)
            log.info("Loaded %s in %.1fs", name, time.time() - t0)

            with self._lock:
                self._models[name] = loaded
                self._loading = None
            return loaded
        except Exception:
            with self._lock:
                self._loading = None
            raise

    def unload(self, name: Optional[str] = None) -> bool:
        """Unload by canonical name/alias, or the LRU model if name is None."""
        with self._lock:
            if not self._models:
                return False
            if name is None:
                # Unload LRU
                target = min(self._models.values(), key=lambda m: m.last_used)
                del self._models[target.name]
            else:
                # Find by name or alias
                found_key = None
                if name in self._models:
                    found_key = name
                else:
                    for k, m in self._models.items():
                        saved = settings.get_model(m.name)
                        if saved.get("alias") == name:
                            found_key = k
                            break
                if found_key is None:
                    return False
                del self._models[found_key]
        _clear_mlx_cache()
        return True


class Settings:
    """Per-model generation defaults + engine preferences, persisted to JSON.

    File layout:
      { "models": { "<repo-id>": { "system": "...", "temperature": 0.7, ... } },
        "general": { ... reserved ... } }
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"models": {}, "general": {}}
        try:
            return json.loads(self.path.read_text())
        except Exception as e:
            log.warning("settings read failed (%s), starting empty", e)
            return {"models": {}, "general": {}}

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def get_model(self, repo_id: str) -> dict:
        with self._lock:
            return dict(self._data.get("models", {}).get(repo_id, {}))

    def set_model(self, repo_id: str, values: dict) -> dict:
        with self._lock:
            models = self._data.setdefault("models", {})
            # Merge rather than replace, so clients can PATCH a single field.
            merged = {**models.get(repo_id, {}), **values}
            # Drop Nones so defaults cleanly revert.
            merged = {k: v for k, v in merged.items() if v is not None}
            models[repo_id] = merged
            self._save_locked()
            return dict(merged)

    def delete_model(self, repo_id: str) -> bool:
        with self._lock:
            removed = self._data.get("models", {}).pop(repo_id, None)
            if removed is not None:
                self._save_locked()
            return removed is not None

    def autoload_name(self) -> Optional[str]:
        with self._lock:
            for repo_id, cfg in self._data.get("models", {}).items():
                if cfg.get("autoload"):
                    return repo_id
        return None


@dataclass
class DownloadJob:
    repo_id: str
    status: str = "queued"  # queued | downloading | done | error | cancelled
    started_at: float = 0.0
    finished_at: float = 0.0
    bytes_downloaded: int = 0
    total_bytes: int = 0
    files_total: int = 0
    files_done: int = 0
    error: Optional[str] = None
    local_dir: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "files_total": self.files_total,
            "files_done": self.files_done,
            "error": self.error,
            "local_dir": self.local_dir,
            "percent": (self.bytes_downloaded / self.total_bytes * 100.0) if self.total_bytes else None,
        }


class HFManager:
    """HuggingFace browse / download / cache operations."""

    def __init__(self) -> None:
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = Lock()

    # ---- search --------------------------------------------------------

    def search(self, query: str, author: Optional[str], limit: int) -> list[dict]:
        from huggingface_hub import HfApi

        api = HfApi()
        # MLX-compatible models are typically published by the mlx-community org,
        # but we also allow library="mlx" and free-form search.
        kwargs: dict[str, Any] = {"limit": limit, "sort": "downloads"}
        if query:
            kwargs["search"] = query
        if author:
            kwargs["author"] = author
        results = []
        try:
            try:
                # hf_hub <1.0 used `direction=-1` for descending; 1.x removed it
                # and sorts descending by default for known sort keys.
                iterator = api.list_models(**kwargs, direction=-1)
            except TypeError:
                iterator = api.list_models(**kwargs)
            for m in iterator:
                tags = list(getattr(m, "tags", []) or [])
                results.append({
                    "id": m.modelId if hasattr(m, "modelId") else m.id,
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                    "last_modified": str(getattr(m, "lastModified", "") or getattr(m, "last_modified", "") or ""),
                    "tags": tags,
                    "pipeline_tag": getattr(m, "pipeline_tag", None),
                })
        except Exception as e:
            log.warning("HF list_models failed: %s", e)
            raise
        return results

    # ---- downloads -----------------------------------------------------

    def start_download(self, repo_id: str) -> DownloadJob:
        with self._lock:
            existing = self._jobs.get(repo_id)
            if existing and existing.status in ("queued", "downloading"):
                return existing
            job = DownloadJob(repo_id=repo_id, status="queued", started_at=time.time())
            self._jobs[repo_id] = job

        t = threading.Thread(target=self._run_download, args=(job,), daemon=True)
        t.start()
        return job

    def _run_download(self, job: DownloadJob) -> None:
        try:
            from huggingface_hub import HfApi, snapshot_download

            job.status = "downloading"

            # Determine total size up front so the UI can show a progress bar.
            try:
                info = HfApi().model_info(job.repo_id, files_metadata=True)
                siblings = getattr(info, "siblings", []) or []
                job.files_total = len(siblings)
                job.total_bytes = sum(int(getattr(s, "size", 0) or 0) for s in siblings)
            except Exception as e:
                log.warning("model_info failed for %s: %s", job.repo_id, e)

            stop_event = threading.Event()
            poll = threading.Thread(target=self._poll_progress, args=(job, stop_event), daemon=True)
            poll.start()
            try:
                local_dir = snapshot_download(
                    repo_id=job.repo_id,
                    # Avoid blowing up memory for tokenizer-less repos; MLX models are small-ish.
                    allow_patterns=None,
                )
            finally:
                stop_event.set()
                poll.join(timeout=1.0)

            job.local_dir = str(local_dir)
            # Final size read
            try:
                job.bytes_downloaded = _dir_size(Path(local_dir))
                job.files_done = _file_count(Path(local_dir))
            except Exception:
                pass
            job.status = "done"
            job.finished_at = time.time()
            log.info("downloaded %s -> %s", job.repo_id, job.local_dir)
        except Exception as e:
            log.exception("download failed for %s", job.repo_id)
            job.status = "error"
            job.error = str(e)
            job.finished_at = time.time()

    def _poll_progress(self, job: DownloadJob, stop: threading.Event) -> None:
        from huggingface_hub import try_to_load_from_cache  # noqa: F401
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_root = Path(HF_HUB_CACHE) / f"models--{job.repo_id.replace('/', '--')}"
        while not stop.is_set():
            try:
                if cache_root.exists():
                    job.bytes_downloaded = _dir_size(cache_root)
                    job.files_done = _file_count(cache_root)
            except Exception:
                pass
            stop.wait(0.75)

    def jobs(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    # ---- cache ---------------------------------------------------------

    def cache(self) -> dict:
        from huggingface_hub import scan_cache_dir

        try:
            info = scan_cache_dir()
        except Exception as e:
            return {"size_on_disk": 0, "repos": [], "error": str(e)}

        repos = []
        for repo in info.repos:
            revisions = [
                {
                    "commit_hash": r.commit_hash,
                    "size_on_disk": r.size_on_disk,
                    "last_modified": r.last_modified,
                    "nb_files": r.nb_files,
                    "refs": sorted(list(r.refs)) if r.refs else [],
                }
                for r in repo.revisions
            ]
            repos.append({
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "size_on_disk": repo.size_on_disk,
                "nb_files": repo.nb_files,
                "last_accessed": repo.last_accessed,
                "last_modified": repo.last_modified,
                "repo_path": str(repo.repo_path),
                "revisions": revisions,
            })
        repos.sort(key=lambda r: r["size_on_disk"], reverse=True)
        return {"size_on_disk": info.size_on_disk, "repos": repos}

    def delete_repo(self, repo_id: str) -> dict:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        revisions = []
        for repo in info.repos:
            if repo.repo_id == repo_id and repo.repo_type == "model":
                revisions.extend(r.commit_hash for r in repo.revisions)
        if not revisions:
            raise ValueError(f"{repo_id!r} not found in cache")
        strategy = info.delete_revisions(*revisions)
        freed = strategy.expected_freed_size
        strategy.execute()
        return {"ok": True, "freed_bytes": freed, "revisions": len(revisions)}


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and not p.is_symlink())


# ────────────────────────────────────────────────────────────────────
# Tiered KV-cache  (RAM hot tier  →  SSD cold tier)
# ────────────────────────────────────────────────────────────────────
KVC_ENABLED          = os.environ.get("MLXR_KVC_ENABLED", "1") not in ("0", "false", "no")
KVC_MAX_RAM_ENTRIES  = int(os.environ.get("MLXR_KVC_RAM_ENTRIES", "8"))
KVC_MAX_DISK_ENTRIES = int(os.environ.get("MLXR_KVC_DISK_ENTRIES", "32"))
KVC_CACHE_DIR        = Path(os.environ.get("MLXR_KVC_DIR",
                             str(Path.home() / ".mlxr" / "kvcache")))
KVC_MIN_PREFIX_LEN   = int(os.environ.get("MLXR_KVC_MIN_PREFIX", "64"))


def _kvc_model_key(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _kvc_prefix_hash(token_ids: list, length: int) -> str:
    buf = b"".join(int(t).to_bytes(4, "little") for t in token_ids[:length])
    return hashlib.sha256(buf).hexdigest()[:32]


def _kvc_serialize(cache: list, path: Path) -> bool:
    """Save a KV-cache to a compressed NPZ file. Returns True on success."""
    try:
        import numpy as np
        import mlx.core as mx
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict = {}
        n = 0
        for i, layer in enumerate(cache):
            k  = getattr(layer, "keys",   None)
            v  = getattr(layer, "values", None)
            off = getattr(layer, "offset", None)
            if k is None or v is None or not off:
                continue
            mx.eval(k, v)
            ks = k[..., :off, :]
            vs = v[..., :off, :]
            mx.eval(ks, vs)
            arrays[f"l{i}k"] = np.array(ks)
            arrays[f"l{i}v"] = np.array(vs)
            arrays[f"l{i}o"] = np.array([off], dtype=np.int64)
            n += 1
        if not n:
            return False
        np.savez_compressed(str(path), **arrays)
        log.info("kvc: saved %d layers (%.1f MB) → %s",
                 n, path.stat().st_size / 1e6, path.name)
        return True
    except Exception as e:
        log.warning("kvc: serialize failed: %s", e)
        return False


def _kvc_deserialize(path: Path, model: Any) -> Optional[list]:
    """Restore a KV-cache from an NPZ file. Returns None on failure."""
    try:
        import numpy as np
        import mlx.core as mx
        if not path.exists() or not hasattr(model, "make_cache"):
            return None
        data = dict(np.load(str(path)))
        cache = model.make_cache()
        n = 0
        for i, layer in enumerate(cache):
            if f"l{i}k" not in data:
                continue
            layer.keys   = mx.array(data[f"l{i}k"])
            layer.values = mx.array(data[f"l{i}v"])
            if hasattr(layer, "offset"):
                layer.offset = int(data[f"l{i}o"][0])
            n += 1
        if not n:
            return None
        log.info("kvc: restored %d layers from %s", n, path.name)
        return cache
    except Exception as e:
        log.warning("kvc: deserialize failed (%s): %s", path.name, e)
        return None


class KVCacheManager:
    """Two-tier KV-cache for prompt-prefix reuse.

    **Hot tier (RAM):** in-process dict of { hash → (cache_list, token_len, ts) }.
    **Cold tier (SSD):** compressed NPZ files under KVC_CACHE_DIR.

    Keys are SHA-256 hashes of token-ID prefixes at power-of-2 lengths.
    On a new request, the manager scans from longest to shortest candidate
    prefix and returns the first (longest) hit, together with the number of
    tokens that are already computed so the caller can skip them.
    """

    def __init__(self) -> None:
        self._lock  = Lock()
        self._ram:  dict[str, dict[str, tuple]] = {}   # mkey→{h→(cache,len,ts)}
        self._disk: dict[str, dict[str, dict]]  = {}   # mkey→{h→{"path","tok_len","ts"}}
        self._load_disk_index()

    # ── public ──────────────────────────────────────────────────────

    def find(self, model_name: str, model: Any, token_ids: list) -> Optional[tuple]:
        """Return (cache_list, prefix_token_len) or None."""
        if not KVC_ENABLED or len(token_ids) < KVC_MIN_PREFIX_LEN:
            return None
        mkey = _kvc_model_key(model_name)
        with self._lock:
            ram  = dict(self._ram.get(mkey,  {}))
            disk = dict(self._disk.get(mkey, {}))

        for exp in range(14, 5, -1):           # 16384 → 64
            clen = 1 << exp
            if clen > len(token_ids) - 1 or clen < KVC_MIN_PREFIX_LEN:
                continue
            h = _kvc_prefix_hash(token_ids, clen)
            if h in ram:
                cache, tl, _ = ram[h]
                with self._lock:
                    t = self._ram.get(mkey, {})
                    if h in t:
                        t[h] = (cache, tl, time.time())
                log.info("kvc: RAM hit — %d tokens for %s", tl, model_name)
                return cache, tl
            if h in disk:
                cache = _kvc_deserialize(disk[h]["path"], model)
                if cache is not None:
                    tl = disk[h]["tok_len"]
                    self._put_ram(mkey, h, cache, tl)
                    log.info("kvc: disk→RAM hit — %d tokens for %s", tl, model_name)
                    return cache, tl
        return None

    def store(self, model_name: str, token_ids: list, cache: list) -> None:
        """Store cache checkpoints at every power-of-2 prefix length."""
        if not KVC_ENABLED or len(token_ids) < KVC_MIN_PREFIX_LEN:
            return
        mkey = _kvc_model_key(model_name)
        for exp in range(6, 15):              # 64 → 16384
            clen = 1 << exp
            if clen >= len(token_ids):
                break
            h = _kvc_prefix_hash(token_ids, clen)
            self._put_ram(mkey, h, cache, clen)
        # Full-length entry
        h = _kvc_prefix_hash(token_ids, len(token_ids))
        self._put_ram(mkey, h, cache, len(token_ids))

    def clear(self, model_name: Optional[str] = None) -> dict:
        with self._lock:
            if model_name:
                mk = _kvc_model_key(model_name)
                rn = len(self._ram.pop(mk, {}))
                dn = len(self._disk.pop(mk, {}))
                shutil.rmtree(KVC_CACHE_DIR / mk, ignore_errors=True)
            else:
                rn = sum(len(v) for v in self._ram.values())
                dn = sum(len(v) for v in self._disk.values())
                self._ram.clear(); self._disk.clear()
                shutil.rmtree(KVC_CACHE_DIR, ignore_errors=True)
        return {"ram_cleared": rn, "disk_cleared": dn}

    def stats(self) -> dict:
        with self._lock:
            rn = sum(len(v) for v in self._ram.values())
            dn = sum(len(v) for v in self._disk.values())
        db = sum(f.stat().st_size for f in KVC_CACHE_DIR.rglob("*.npz")
                 if f.exists()) if KVC_CACHE_DIR.exists() else 0
        return {
            "enabled": KVC_ENABLED, "ram_entries": rn, "ram_max": KVC_MAX_RAM_ENTRIES,
            "disk_entries": dn, "disk_max": KVC_MAX_DISK_ENTRIES,
            "disk_bytes": db, "cache_dir": str(KVC_CACHE_DIR),
        }

    # ── internal ────────────────────────────────────────────────────

    def _put_ram(self, mkey: str, h: str, cache: list, tl: int) -> None:
        with self._lock:
            tier = self._ram.setdefault(mkey, {})
            tier[h] = (cache, tl, time.time())
            while len(tier) > KVC_MAX_RAM_ENTRIES:
                old_h = min(tier, key=lambda x: tier[x][2])
                old_c, old_l, _ = tier.pop(old_h)
                threading.Thread(
                    target=self._spill, args=(mkey, old_h, old_c, old_l), daemon=True
                ).start()

    def _spill(self, mkey: str, h: str, cache: list, tl: int) -> None:
        with self._lock:
            disk = self._disk.get(mkey, {})
            if len(disk) >= KVC_MAX_DISK_ENTRIES:
                old_h = min(disk, key=lambda x: disk[x]["ts"])
                try:
                    disk.pop(old_h)["path"].unlink(missing_ok=True)
                except Exception:
                    pass
        path = KVC_CACHE_DIR / mkey / f"{h}.npz"
        if _kvc_serialize(cache, path):
            with self._lock:
                self._disk.setdefault(mkey, {})[h] = {"path": path, "tok_len": tl, "ts": time.time()}
            self._save_meta(mkey)

    def _load_disk_index(self) -> None:
        if not KVC_CACHE_DIR.exists():
            return
        for d in KVC_CACHE_DIR.iterdir():
            if not d.is_dir():
                continue
            mkey = d.name
            meta_p = d / "_meta.json"
            if meta_p.exists():
                try:
                    for h, info in json.loads(meta_p.read_text()).items():
                        p = d / f"{h}.npz"
                        if p.exists():
                            self._disk.setdefault(mkey, {})[h] = {
                                "path": p, "tok_len": info.get("tok_len", 0), "ts": info.get("ts", 0.0)
                            }
                except Exception as e:
                    log.debug("kvc: meta error %s: %s", d.name, e)

    def _save_meta(self, mkey: str) -> None:
        d = KVC_CACHE_DIR / mkey
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            meta = {h: {"tok_len": v["tok_len"], "ts": v["ts"]}
                    for h, v in self._disk.get(mkey, {}).items()}
        (d / "_meta.json").write_text(json.dumps(meta, indent=2))


kvc = KVCacheManager()

engine = EnginePool()
hf = HFManager()
settings = Settings(SETTINGS_PATH)
app = FastAPI(title="MLXr", version="0.1.0")

# CORS — allow any browser/frontend origin so OpenWebUI, LibreChat, custom UIs,
# and Jupyter notebooks can call the API directly without a reverse-proxy.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API-key gate on /v1/ routes.
# Set MLXR_API_KEY in the environment to require clients to send
#   Authorization: Bearer <key>
# When the env var is not set, any value (or no header) is accepted, which
# preserves the existing "no auth required" behaviour.
_API_KEY: Optional[str] = os.environ.get("MLXR_API_KEY") or None


async def _idle_unload_task() -> None:
    """Background task: auto-unload models that have been idle past their TTL.

    Checks every 60 seconds. The TTL (``idle_timeout_minutes``) is a per-model
    setting stored in ~/.mlxr/settings.json. A value of None means 'never
    auto-unload', preserving backward-compatible behaviour for all existing
    model configs.
    """
    while True:
        await asyncio.sleep(60)
        for m in engine.loaded_models():
            saved = settings.get_model(m.name)
            timeout_min = saved.get("idle_timeout_minutes")
            if timeout_min and (time.time() - m.last_used) >= timeout_min * 60:
                log.info(
                    "Auto-unloading %s: idle %.1f min >= TTL %d min",
                    m.name, (time.time() - m.last_used) / 60, timeout_min,
                )
                engine.unload(m.name)


@app.on_event("startup")
async def _autoload_on_start() -> None:
    name = settings.autoload_name()
    if name:
        log.info("Autoloading %s per settings", name)

        def _go():
            try:
                engine.load(name)
            except Exception as e:
                log.warning("autoload failed: %s", e)

        threading.Thread(target=_go, daemon=True).start()

    # Background task — runs for the lifetime of the server.
    asyncio.create_task(_idle_unload_task())


# ---- models --------------------------------------------------------------


class LoadRequest(BaseModel):
    name: str = Field(..., description="HuggingFace repo id or local path of an MLX model.")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = Field(default=None, ge=1, le=131072)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    system: Optional[str] = None
    stream: bool = True


# Hard fallbacks used when neither the request nor saved settings specify a value.
DEFAULT_GEN: dict[str, Any] = {
    "max_tokens": 4096,   # raised from 512 — tool-call arguments (file writes etc.) need headroom
    "temperature": 0.7,
    "top_p": 0.95,
    "system": None,
}
# When tools are active and the client/user haven't set a token limit, use a
# higher floor so that tool-call arguments (e.g. write_file with a full file
# body) aren't silently truncated mid-JSON.
DEFAULT_TOOLS_MAX_TOKENS = 16384


def _resolve_gen(cur: LoadedModel, req: GenerateRequest) -> GenerateRequest:
    """Return a new GenerateRequest with saved-per-model + hard defaults applied."""
    saved = settings.get_model(cur.name)
    payload = req.model_dump()
    for key, fallback in DEFAULT_GEN.items():
        if payload.get(key) is None:
            payload[key] = saved.get(key, fallback)
    # Re-validate so any saved garbage is caught here, not deeper in MLX.
    return GenerateRequest(**payload)


# ---- helpers -------------------------------------------------------------


def _host_stats() -> dict:
    vm = psutil.virtual_memory()
    stats: dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "mem_total": vm.total,
        "mem_used": vm.used,
        "mem_percent": vm.percent,
    }
    try:
        import mlx.core as mx

        active = mx.metal.get_active_memory()
        peak = mx.metal.get_peak_memory()
        cache = mx.metal.get_cache_memory()
        stats["mlx"] = {
            "active_bytes": int(active),
            "peak_bytes": int(peak),
            "cache_bytes": int(cache),
        }
    except Exception as e:
        stats["mlx"] = {"error": str(e)}
    return stats


def _resolve_model(name: Optional[str]) -> Optional[LoadedModel]:
    """Return a loaded model by name/alias, or the most-recently-used model."""
    if name:
        return engine.get(name)
    return engine.current


def _pool_model_state(m: LoadedModel) -> dict:
    """Return state dict for a single pool entry (used by /api/models/pool)."""
    return {
        "loaded": True,
        "name": m.name,
        "loaded_at": m.loaded_at,
        "uptime_seconds": time.time() - m.loaded_at,
        "generations": m.generations,
        "total_tokens": m.total_tokens,
        "last_used": m.last_used,
        "context_length": m.context_length,
        "last_ttft": m.last_ttft,
        "last_tps": m.last_tps,
        "is_vlm": m.is_vlm,
    }


def _model_state() -> dict:
    """Return state of the most-recently-used model for backward compat."""
    cur = engine.current
    if not cur:
        return {"loaded": False, "loading": engine.loading}
    return {
        "loaded": True,
        "loading": engine.loading,
        "name": cur.name,
        "loaded_at": cur.loaded_at,
        "uptime_seconds": time.time() - cur.loaded_at,
        "generations": cur.generations,
        "total_tokens": cur.total_tokens,
        "last_used": cur.last_used,
        "context_length": cur.context_length,
        "last_ttft": cur.last_ttft,
        "last_tps": cur.last_tps,
        "is_vlm": cur.is_vlm,
    }


class ThinkStripper:
    """Streaming-safe filter that drops reasoning / chain-of-thought blocks.

    Reasoning models (Qwen3, DeepSeek-R1, GLM-4.5, etc.) emit chain-of-thought
    inside wrapper tags before the real answer. OpenAI-compatible clients have
    no way to separate reasoning from content, so we strip the block by default.

    Works across streamed chunks: if a tag straddles the boundary, we hold
    back just enough tail in an internal buffer until we know whether it's a
    tag or plain text. Multiple tag families are recognized simultaneously so
    the stripper works regardless of which reasoning-model family is loaded.
    """

    # Each entry: (open, close). All checked on every iteration; first match wins.
    TAG_PAIRS: tuple[tuple[str, str], ...] = (
        ("<think>", "</think>"),
        ("<thinking>", "</thinking>"),
        ("<reasoning>", "</reasoning>"),
        ("<thought>", "</thought>"),
        ("<|thinking|>", "<|/thinking|>"),
        ("<|reasoning_start|>", "<|reasoning_end|>"),
    )

    # Bare special tokens that sometimes leak through the decoder — strip on
    # sight. These are never meaningful content in a chat response.
    STRAY_TOKENS: tuple[str, ...] = (
        # ChatML (Qwen, Mistral-Instruct, etc.)
        "<|im_start|>",
        "<|im_end|>",
        "<|im_sep|>",
        "<|endoftext|>",
        # Llama-3 / Llama-3.1
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        # Qwen3 mask / turn tokens that leak from the A3B-4bit quant
        "<|mask_start|>",
        "<|mask_end|>",
        "<|turn_start|>",
        "<|turn_end|>",
        # DeepSeek conversation tokens
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
        "<\uff5cend\u2581of\u2581sentence\uff5c>",
        "<\uff5cUser\uff5c>",
        "<\uff5cAssistant\uff5c>",
        # DeepSeek-V3 tool-call outer container tokens.  The inner per-call
        # wrappers (<｜tool▁call▁begin｜> / <｜tool▁call▁end｜>) are consumed
        # by ToolCallParser, so only the outer container markers are stripped here.
        "<\uff5ctool\u2581calls\u2581begin\uff5c>",
        "<\uff5ctool\u2581calls\u2581end\uff5c>",
    )

    def __init__(self, enabled: bool = True, starts_in_think: bool = False) -> None:
        self.enabled = enabled
        self._buffer = ""
        # Reasoning models (Qwen3, DeepSeek-R1) often have the opening tag
        # injected at the end of the prompt by the chat template — so the
        # *stream* begins inside the think block and we only ever see a close
        # tag. Callers pass ``starts_in_think=True`` in that case.
        self._in_think = starts_in_think
        # When starts_in_think is True, we don't know which close tag to
        # expect — so match any known close tag.
        self._active_close: Optional[str] = None

    # Longest open tag length — we hold back this-many-minus-one chars at
    # the tail of the buffer when looking for a potential opening tag, to
    # avoid emitting half of one.
    _MAX_OPEN_LEN = max(len(o) for o, _ in TAG_PAIRS)
    _MAX_CLOSE_LEN = max(len(c) for _, c in TAG_PAIRS)

    def feed(self, chunk: str) -> str:
        if not self.enabled or not chunk:
            return chunk
        self._buffer += chunk
        out: list[str] = []
        while self._buffer:
            if self._in_think:
                # Find earliest close tag (either the known one if we have it,
                # or any of the known close tags if we don't).
                closes = (
                    [self._active_close] if self._active_close
                    else [c for _, c in self.TAG_PAIRS]
                )
                earliest_idx = -1
                earliest_close = None
                for c in closes:
                    i = self._buffer.find(c)
                    if i >= 0 and (earliest_idx < 0 or i < earliest_idx):
                        earliest_idx = i
                        earliest_close = c
                if earliest_idx >= 0 and earliest_close:
                    self._buffer = self._buffer[earliest_idx + len(earliest_close):]
                    self._in_think = False
                    self._active_close = None
                    continue
                # Might be a partial close tag at the tail — hold back the
                # last (max_close_len - 1) chars, drop the rest.
                keep = self._MAX_CLOSE_LEN - 1
                if len(self._buffer) > keep:
                    self._buffer = self._buffer[-keep:]
                break
            else:
                # Find earliest open tag across all known pairs…
                earliest_idx = -1
                earliest_open = None
                earliest_close = None
                is_open = True
                for o, c in self.TAG_PAIRS:
                    i = self._buffer.find(o)
                    if i >= 0 and (earliest_idx < 0 or i < earliest_idx):
                        earliest_idx = i
                        earliest_open = o
                        earliest_close = c
                        is_open = True
                # …AND earliest stray *close* tag (some models emit a bare
                # ``</think>`` after the real answer — we silently drop those
                # so they don't leak into OpenAI-client output).
                for _, c in self.TAG_PAIRS:
                    i = self._buffer.find(c)
                    if i >= 0 and (earliest_idx < 0 or i < earliest_idx):
                        earliest_idx = i
                        earliest_open = c  # we'll advance past this close tag
                        earliest_close = None
                        is_open = False
                # …AND any known stray special token (``<|im_end|>`` etc.).
                for tok in self.STRAY_TOKENS:
                    i = self._buffer.find(tok)
                    if i >= 0 and (earliest_idx < 0 or i < earliest_idx):
                        earliest_idx = i
                        earliest_open = tok
                        earliest_close = None
                        is_open = False
                if earliest_idx >= 0 and earliest_open:
                    if earliest_idx:
                        out.append(self._buffer[:earliest_idx])
                    self._buffer = self._buffer[earliest_idx + len(earliest_open):]
                    if is_open:
                        self._in_think = True
                        self._active_close = earliest_close
                    # else: stray close tag — dropped silently, stay outside think.
                    continue
                # No tag — emit everything except a possible partial tag at
                # the tail (use the longest of open/close/stray-token lengths).
                keep = max(
                    self._MAX_OPEN_LEN,
                    self._MAX_CLOSE_LEN,
                    max((len(t) for t in self.STRAY_TOKENS), default=0),
                ) - 1
                if len(self._buffer) > keep:
                    out.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                break
        return "".join(out)

    def flush(self) -> str:
        if not self.enabled:
            return ""
        if self._in_think:
            # Unclosed think block — drop it.
            self._buffer = ""
            self._in_think = False
            self._active_close = None
            return ""
        tail = self._buffer
        self._buffer = ""
        return tail

    def strip(self, text: str) -> str:
        """One-shot strip for non-streaming text."""
        return self.feed(text) + self.flush()


class ToolCallParser:
    """Extract tool-call blocks from a token stream.

    Different instruct-model families emit tool calls with different wrapper
    syntax. We recognize the common ones simultaneously so the parser works
    regardless of which tokenizer the loaded model was trained with:

    * ``<tool_call>{...}</tool_call>`` — Qwen2.5 / Qwen3 / Hermes
    * ``<|tool_call|>{...}<|/tool_call|>`` — some Qwen3 variants
    * ``<function_call>{...}</function_call>`` — earlier Qwen / some fine-tunes
    * ``<tool_call_begin>{...}<tool_call_end>`` — DeepSeek-V3 / R1 tool format
    * ``[TOOL_CALLS][{...}]`` — Mistral tool format
    """

    # (open, close). All checked on every iteration; earliest match wins.
    TAG_PAIRS: tuple[tuple[str, str], ...] = (
        # Qwen2.5 / Qwen3 / Hermes-3 / NousResearch / Gemma 3
        ("<tool_call>", "</tool_call>"),
        # Phi-4-mini
        ("<|tool_call|>", "<|/tool_call|>"),
        # Earlier Qwen / some fine-tunes
        ("<function_call>", "</function_call>"),
        # Generic ASCII variants (kept for forward-compat)
        ("<tool_call_begin>", "<tool_call_end>"),
        ("<|tool_call_begin|>", "<|tool_call_end|>"),
        # Mistral (no real close tag — close tag never appears; flush() handles it)
        ("[TOOL_CALLS]", "[/TOOL_CALLS]"),
        # DeepSeek-V2/V3 individual call wrappers (Unicode fullwidth | + ▁ tokens).
        # The outer <｜tool▁calls▁begin｜>/<｜tool▁calls▁end｜> container is stripped
        # by ThinkStripper.STRAY_TOKENS so we only need to match the inner pair here.
        ("<\uff5ctool\u2581call\u2581begin\uff5c>", "<\uff5ctool\u2581call\u2581end\uff5c>"),
    )

    # DeepSeek-V3 body separator between call-type and function name.
    _DS_SEP = "<\uff5ctool\u2581sep\uff5c>"

    _MAX_OPEN_LEN = max(len(o) for o, _ in TAG_PAIRS)
    _MAX_CLOSE_LEN = max(len(c) for _, c in TAG_PAIRS)

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._buffer = ""
        self._tool_buffer = ""
        self._in_tool = False
        self._active_close: Optional[str] = None

    def feed(self, chunk: str) -> tuple[str, list[dict]]:
        """Feed a chunk, return (visible_content, newly_completed_tool_calls)."""
        if not self.enabled:
            # When parsing is off, treat input as plain content.
            return (chunk or ""), []
        if not chunk:
            return "", []
        self._buffer += chunk
        content_out: list[str] = []
        tools_out: list[dict] = []
        while self._buffer:
            if self._in_tool and self._active_close:
                idx = self._buffer.find(self._active_close)
                if idx >= 0:
                    self._tool_buffer += self._buffer[:idx]
                    self._buffer = self._buffer[idx + len(self._active_close):]
                    # Split on Mistral-style arrays: [TOOL_CALLS] can hold
                    # a list of calls in one block.
                    for tool in self._parse_tools(self._tool_buffer):
                        tools_out.append(tool)
                    self._tool_buffer = ""
                    self._in_tool = False
                    self._active_close = None
                    continue
                keep = len(self._active_close) - 1
                if len(self._buffer) > keep:
                    self._tool_buffer += self._buffer[:-keep]
                    self._buffer = self._buffer[-keep:]
                break
            else:
                earliest_idx = -1
                earliest_open = None
                earliest_close = None
                for o, c in self.TAG_PAIRS:
                    i = self._buffer.find(o)
                    if i >= 0 and (earliest_idx < 0 or i < earliest_idx):
                        earliest_idx = i
                        earliest_open = o
                        earliest_close = c
                if earliest_idx >= 0 and earliest_open:
                    if earliest_idx > 0:
                        content_out.append(self._buffer[:earliest_idx])
                    self._buffer = self._buffer[earliest_idx + len(earliest_open):]
                    self._in_tool = True
                    self._active_close = earliest_close
                    continue
                keep = self._MAX_OPEN_LEN - 1
                if len(self._buffer) > keep:
                    content_out.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                break
        return "".join(content_out), tools_out

    def flush(self) -> tuple[str, list[dict]]:
        if not self.enabled:
            tail = self._buffer
            self._buffer = ""
            return tail, []
        if self._in_tool:
            tools = self._parse_tools(self._tool_buffer + self._buffer)
            self._tool_buffer = ""
            self._buffer = ""
            self._in_tool = False
            self._active_close = None
            return "", tools
        tail = self._buffer
        self._buffer = ""
        return tail, []

    @classmethod
    def _parse_tools(cls, text: str) -> list[dict]:
        """Parse a tool-call body. Tries multiple formats in order:

        1. JSON object or array (Qwen2.5 / Hermes / Mistral).
        2. ```json ...``` fenced JSON (DeepSeek-V3).
        3. XML-style ``<function=NAME><parameter=K>V</parameter>…</function>``
           (newer Qwen3.x variants and several agentic fine-tunes — this is the
           format that broke OpenCode integration with Qwen3.6-35B-A3B-4bit).
        """
        text = text.strip()
        if not text:
            return []

        # DeepSeek-V3: body is "function<｜tool▁sep｜>NAME\n```json\n{…}\n```"
        if cls._DS_SEP in text:
            return cls._parse_deepseek_body(text)

        # Fenced code block — unwrap first.
        if text.startswith("```"):
            stripped = text.strip("`")
            if "\n" in stripped:
                first, rest = stripped.split("\n", 1)
                if first.strip().lower() in ("json", "xml", ""):
                    stripped = rest
            text = stripped.strip()

        # Try JSON first.
        try:
            data = json.loads(text)
        except Exception:
            data = None

        if data is not None:
            items = data if isinstance(data, list) else [data]
            results: list[dict] = []
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                tool = cls._make_tool(entry)
                if tool:
                    results.append(tool)
            if results:
                return results

        # Fall back to XML-style.
        xml_tools = cls._parse_xml_tools(text)
        if xml_tools:
            return xml_tools

        log.warning("tool_call body parse failed (tried JSON + XML) — raw: %r", text[:300])
        return []

    @classmethod
    def _parse_deepseek_body(cls, text: str) -> list[dict]:
        """Parse a DeepSeek-V3/V2.5 individual tool-call body.

        Format (after the outer <｜tool▁call▁begin｜> is stripped by the tag
        pair matcher):

            function<｜tool▁sep｜>FUNCTION_NAME
            ```json
            {"key": "value"}
            ```

        ``function`` is a literal type discriminator; the name and args follow
        the separator.  Multiple calls are handled by the outer loop in
        ToolCallParser.feed() which fires once per <｜tool▁call▁begin｜>…
        <｜tool▁call▁end｜> pair.
        """
        sep_idx = text.find(cls._DS_SEP)
        if sep_idx < 0:
            return []
        rest = text[sep_idx + len(cls._DS_SEP):]
        nl_idx = rest.find("\n")
        if nl_idx < 0:
            fn_name = rest.strip()
            fn_body = ""
        else:
            fn_name = rest[:nl_idx].strip()
            fn_body = rest[nl_idx + 1:].strip()
        # Strip optional fenced code block wrapper (```json … ```)
        if fn_body.startswith("```"):
            lines = fn_body.splitlines()
            inner: list[str] = []
            for line in lines[1:]:          # skip the ```json opener
                if line.strip().startswith("```"):
                    break
                inner.append(line)
            fn_body = "\n".join(inner).strip()
        if not fn_name:
            return []
        try:
            args = json.loads(fn_body) if fn_body else {}
        except Exception:
            log.warning("DeepSeek tool body JSON parse failed: %r", fn_body[:200])
            args = {}
        return [{
            "id": f"call_{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }]

    @classmethod
    def try_extract_raw_json(cls, content: str) -> list[dict]:
        """Last-resort heuristic for models that emit bare JSON tool calls
        with no wrapper tags — most notably Llama 3.2's default chat template,
        which outputs ``{"name": "…", "parameters": {…}}`` directly.

        We only convert to a tool call when ALL of these are true:
        * The entire content (trimmed) is a single JSON object.
        * The object has a "name" string key.
        * The object has an "arguments" or "parameters" dict key.

        This is strict enough to avoid false-positives for plain-text
        responses that happen to contain JSON.
        """
        stripped = content.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return []
        try:
            data = json.loads(stripped)
        except Exception:
            return []
        if not isinstance(data, dict):
            return []
        if not isinstance(data.get("name"), str):
            return []
        has_args = "arguments" in data or "parameters" in data
        if not has_args:
            return []
        tool = cls._make_tool(data)
        return [tool] if tool else []

    # XML-style:
    #   <function=NAME>
    #     <parameter=KEY>\nVALUE\n</parameter>
    #     <parameter=KEY2>\nVALUE2\n</parameter>
    #   </function>
    # We're lenient about whitespace and attribute quoting.
    #
    # IMPORTANT: we use a stack-based approach rather than regex `.*?` (non-greedy)
    # because file content may legitimately contain the literal strings
    # "</parameter>" or "</function>" (e.g. XML files, HTML, Jinja templates).
    # Non-greedy matching would terminate early on the first such occurrence and
    # silently truncate the content. The manual parser below handles nesting depth.

    _XML_FUNCTION_OPEN_RE = __import__("re").compile(
        r"<function\s*=\s*[\"']?([^\s\"'>]+)[\"']?\s*>",
    )
    _XML_PARAMETER_OPEN_RE = __import__("re").compile(
        r"<parameter\s*=\s*[\"']?([^\s\"'>]+)[\"']?\s*>",
    )

    @classmethod
    def _parse_xml_tools(cls, text: str) -> list[dict]:
        """Parse Qwen3-style XML tool calls using a depth-tracking approach.

        Handles the case where file content contains the literal strings
        </parameter> or </function> without prematurely terminating the match.
        """
        results: list[dict] = []
        pos = 0
        while pos < len(text):
            fn_m = cls._XML_FUNCTION_OPEN_RE.search(text, pos)
            if not fn_m:
                break
            fn_name = fn_m.group(1).strip()
            body_start = fn_m.end()

            # Find the matching </function> by tracking open/close depth.
            body, body_end = cls._find_closing(text, body_start, "<function", "</function>")
            if body is None:
                # Unclosed tag — nothing more to parse.
                break
            pos = body_end

            params: dict[str, Any] = {}
            param_pos = 0
            while param_pos < len(body):
                pm = cls._XML_PARAMETER_OPEN_RE.search(body, param_pos)
                if not pm:
                    break
                key = pm.group(1).strip()
                val_start = pm.end()
                raw_val, val_end = cls._find_closing(body, val_start, "<parameter", "</parameter>")
                if raw_val is None:
                    break
                param_pos = val_end

                # Preserve raw value without stripping — leading/trailing
                # whitespace in file content is significant. Only strip for
                # type-detection; store the original.
                stripped = raw_val.strip()
                if stripped and (stripped[0] in "{[\"" or stripped in ("true", "false", "null") or cls._looks_numeric(stripped)):
                    try:
                        params[key] = json.loads(stripped)
                        continue
                    except Exception:
                        pass
                # Store with only a single leading/trailing newline stripped
                # (the model typically wraps values in \n…\n).
                params[key] = raw_val.lstrip("\n").rstrip("\n")

            if not fn_name:
                continue
            results.append({
                "id": f"call_{uuid.uuid4().hex[:20]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(params, ensure_ascii=False),
                },
            })
        return results

    @staticmethod
    def _find_closing(text: str, start: int, open_tag_prefix: str, close_tag: str) -> tuple[Optional[str], int]:
        """Return (body, end_pos) where body is the text between start and the
        matching close_tag, respecting nesting of open_tag_prefix.

        Returns (None, start) if the close_tag is never found.
        """
        depth = 1
        pos = start
        close_len = len(close_tag)
        while pos < len(text):
            next_close = text.find(close_tag, pos)
            next_open = text.find(open_tag_prefix, pos)
            if next_close < 0:
                return None, start  # unclosed
            # If there's a nested open before the next close, go deeper.
            if next_open >= 0 and next_open < next_close:
                depth += 1
                pos = next_open + len(open_tag_prefix)
            else:
                depth -= 1
                if depth == 0:
                    return text[start:next_close], next_close + close_len
                pos = next_close + close_len
        return None, start

    @staticmethod
    def _looks_numeric(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def _make_tool(data: dict) -> Optional[dict]:
        # Templates vary: some use "arguments", some "parameters", some nest
        # under "function": {"name": ..., "arguments": ...}.
        if "function" in data and isinstance(data["function"], dict):
            fn = data["function"]
            name = fn.get("name")
            args = fn.get("arguments") if fn.get("arguments") is not None else fn.get("parameters", {})
        else:
            name = data.get("name")
            args = data.get("arguments") if data.get("arguments") is not None else data.get("parameters", {})
        if not name:
            return None
        # OpenAI's streaming delta expects arguments as a JSON *string*.
        args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return {
            "id": f"call_{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        }


def _make_iterator(cur: LoadedModel, prompt: str, req: GenerateRequest,
                   images=(), prompt_cache: Optional[list] = None):
    """Build a stream_generate iterator using the best available sampler.

    Tries to pass advanced sampling params (min_p, repetition_penalty) to
    make_sampler and falls back gracefully when an older mlx-lm doesn't
    accept them.  Always falls back to the bare temp= kwarg path when
    make_sampler itself is unavailable.

    When cur.is_vlm and images are provided, delegates to _make_vlm_iterator.
    """
    if cur.is_vlm and images:
        return _make_vlm_iterator(cur, prompt, req, list(images))

    from mlx_lm import stream_generate

    gen_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": req.max_tokens,
    }
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache
    try:
        from mlx_lm.sample_utils import make_sampler

        sampler_kw: dict[str, Any] = {
            "temp": req.temperature,
            "top_p": req.top_p,
        }
        if req.min_p is not None:
            sampler_kw["min_p"] = req.min_p
        if req.repetition_penalty is not None:
            sampler_kw["repetition_penalty"] = req.repetition_penalty
        try:
            sampler = make_sampler(**sampler_kw)
        except TypeError:
            # Older mlx-lm: drop advanced params and retry.
            sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
        try:
            return stream_generate(cur.model, cur.tokenizer, **gen_kwargs, sampler=sampler)
        except TypeError:
            gen_kwargs.pop("prompt_cache", None)
            try:
                return stream_generate(cur.model, cur.tokenizer, **gen_kwargs, sampler=sampler)
            except TypeError:
                return stream_generate(cur.model, cur.tokenizer, **gen_kwargs, temp=req.temperature)
    except ImportError:
        try:
            return stream_generate(cur.model, cur.tokenizer, **gen_kwargs, temp=req.temperature)
        except TypeError:
            gen_kwargs.pop("prompt_cache", None)
            return stream_generate(cur.model, cur.tokenizer, **gen_kwargs, temp=req.temperature)


def _make_vlm_iterator(cur: LoadedModel, prompt: str, req: GenerateRequest, images: list):
    """Build a VLM stream iterator using mlx_vlm.stream_generate."""
    try:
        from mlx_vlm import stream_generate as vlm_stream
        from mlx_vlm.utils import load_image
    except ImportError:
        log.warning("mlx-vlm not installed — falling back to text-only generation")
        return _make_iterator(cur, prompt, req)

    img_objs = []
    for url in images:
        try:
            img_objs.append(load_image(url))
        except Exception as e:
            log.warning("Failed to load image %s: %s", url, e)

    if not img_objs:
        return _make_iterator(cur, prompt, req)

    return vlm_stream(
        cur.model, cur.tokenizer, prompt,
        image=img_objs[0] if len(img_objs) == 1 else img_objs,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )


def _render_vlm_prompt(cur: LoadedModel, messages: list[dict], images: list) -> str:
    """Build a VLM-formatted prompt string."""
    try:
        from mlx_vlm.prompt_utils import apply_chat_template as vlm_tmpl
        # Extract last user text for the VLM prompt
        user_text = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            ""
        )
        return vlm_tmpl(cur.tokenizer, cur.model.config, user_text, num_images=len(images))
    except Exception as e:
        log.warning("VLM apply_chat_template failed (%s), falling back to regular template", e)
        return _render_chat(cur.tokenizer, messages)


def _extract_images(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Strip image_url content parts out of messages, return (clean_msgs, urls)."""
    clean, urls = [], []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts, img_urls = [], []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif p.get("type") == "image_url":
                        u = p.get("image_url") or {}
                        url = u.get("url", "") if isinstance(u, dict) else str(u)
                        if url:
                            img_urls.append(url)
            urls.extend(img_urls)
            clean.append({**m, "content": "".join(parts)})
        else:
            clean.append(m)
    return clean, urls


def _strip_thinking_enabled(cur: LoadedModel) -> bool:
    saved = settings.get_model(cur.name)
    # Default ON — mirrors what most proxies do. Users can opt out per model.
    return bool(saved.get("strip_thinking", True))


def _render_prompt(tokenizer: Any, prompt: str, system: Optional[str]) -> str:
    """Use the tokenizer's chat template when available; otherwise fall back to raw prompt."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        log.warning("chat_template failed, using raw prompt: %s", e)
    return prompt


# ---- routes --------------------------------------------------------------


MIN_PYTHON = (3, 10)


def _engine_versions() -> dict:
    # Use importlib.metadata rather than importing each module and reading
    # __version__.  importlib.metadata reads from the installed .dist-info
    # directory on disk, so it reflects what pip just wrote *without* needing
    # a process restart — critical for the "check versions after upgrade" flow.
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    # (module-attr-key → PyPI package name)
    _PKG_MAP = {
        "mlx":             "mlx",
        "mlx_lm":          "mlx-lm",
        "huggingface_hub": "huggingface-hub",
        "transformers":    "transformers",
    }
    out: dict[str, Any] = {}
    for mod_key, pkg_name in _PKG_MAP.items():
        try:
            out[mod_key] = _pkg_version(pkg_name)
        except PackageNotFoundError:
            out[mod_key] = "not installed"
        except Exception as e:
            out[mod_key] = f"not installed ({e.__class__.__name__})"
    out["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    out["python_too_old"] = sys.version_info < MIN_PYTHON
    out["python_min"] = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    return out


@app.get("/health")
def health() -> dict:
    """Standard health-check used by Docker, Homebrew services, and IDE integrations."""
    cur = engine.current
    return {
        "status": "ok",
        "model": cur.name if cur else None,
        "loading": engine.loading,
    }


@app.get("/api/engine/version")
def api_engine_version() -> dict:
    return _engine_versions()


class UpgradeRequest(BaseModel):
    packages: list[str] = Field(default_factory=lambda: list(ALLOWED_UPGRADE_PACKAGES))


def _pypi_latest(pkg: str) -> Optional[str]:
    try:
        req = urllib.request.Request(
            f"https://pypi.org/pypi/{pkg}/json",
            headers={"User-Agent": "MLXr/0.1"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.load(r)
        return data.get("info", {}).get("version")
    except Exception as e:
        log.warning("PyPI lookup for %s failed: %s", pkg, e)
        return None


def _version_tuple(v: str) -> tuple:
    parts = []
    for chunk in v.split("."):
        n = ""
        for ch in chunk:
            if ch.isdigit():
                n += ch
            else:
                break
        parts.append(int(n) if n else 0)
    return tuple(parts)


@app.get("/api/engine/check")
async def api_engine_check() -> dict:
    installed = _engine_versions()
    latest_list = await asyncio.gather(
        *[asyncio.to_thread(_pypi_latest, p) for p in ALLOWED_UPGRADE_PACKAGES]
    )
    result: dict[str, Any] = {}
    any_update = False
    for pkg, latest in zip(ALLOWED_UPGRADE_PACKAGES, latest_list):
        mod = PACKAGE_TO_MODULE[pkg]
        inst = installed.get(mod, "unknown")
        update = False
        if latest and not inst.startswith("not installed") and inst != "unknown":
            try:
                update = _version_tuple(latest) > _version_tuple(inst)
            except Exception:
                update = inst != latest
        any_update = any_update or update
        missing = inst.startswith("not installed")
        result[pkg] = {
            "installed": inst,
            "latest": latest,
            "update_available": update,
            "missing": missing,
        }
        any_update = any_update or update
    any_missing = any(v["missing"] for v in result.values())
    return {
        "packages": result,
        "update_available": any_update,
        "install_available": any_missing,
        "python": installed.get("python"),
        "python_too_old": bool(installed.get("python_too_old")),
        "python_min": installed.get("python_min"),
    }


@app.post("/api/engine/upgrade")
async def api_engine_upgrade(req: UpgradeRequest):
    requested = [p for p in req.packages if p in ALLOWED_UPGRADE_PACKAGES]
    if not requested:
        raise HTTPException(
            status_code=400,
            detail=f"No valid packages. Allowed: {', '.join(ALLOWED_UPGRADE_PACKAGES)}",
        )
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *requested]
    return StreamingResponse(
        _stream_subprocess(cmd),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_subprocess(cmd: list[str]) -> AsyncIterator[bytes]:
    yield f"event: start\ndata: {json.dumps({'cmd': cmd})}\n\n".encode()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n".encode()
        return
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\n")
        yield f"data: {json.dumps({'line': text})}\n\n".encode()
    rc = await proc.wait()
    yield f"event: done\ndata: {json.dumps({'returncode': rc})}\n\n".encode()


@app.post("/api/engine/restart")
def api_engine_restart() -> dict:
    """Exit with a special code that run.sh interprets as 'restart me'.

    If the server wasn't launched via run.sh, the process just exits and the
    user must relaunch — documented in the response so the UI can tell them.
    """

    def _exit_soon() -> None:
        time.sleep(0.3)
        os._exit(RESTART_EXIT_CODE)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {
        "ok": True,
        "exit_code": RESTART_EXIT_CODE,
        "managed_by_run_sh": os.environ.get("MLXR_MANAGED") == "1",
    }


@app.get("/api/debug/recent_chats")
def api_debug_recent_chats() -> dict:
    """Return the last N /v1/chat/completions invocations with prompt tail
    and raw-output preview. Useful for diagnosing 'why isn't the tool call
    firing' without having to dig through the server log."""
    with _RECENT_CHATS_LOCK:
        return {"recent": list(reversed(_RECENT_CHATS))}


@app.get("/api/status")
def api_status() -> dict:
    return {
        "host": _host_stats(),
        "model": _model_state(),
        "suggested": SUGGESTED_MODELS,
        "versions": _engine_versions(),
    }


@app.get("/api/models")
def api_models() -> dict:
    return {"current": _model_state(), "suggested": SUGGESTED_MODELS}


@app.post("/api/models/load")
async def api_load(req: LoadRequest) -> dict:
    try:
        loaded = await asyncio.to_thread(engine.load, req.name)
    except Exception as e:
        log.exception("load failed")
        msg = str(e)
        if "not supported" in msg.lower() or "unknown model type" in msg.lower():
            msg = f"{msg} — this architecture is newer than your installed mlx-lm. Run: pip install --upgrade mlx mlx-lm"
        raise HTTPException(status_code=500, detail=f"load failed: {msg}")
    return {"ok": True, "name": loaded.name, "loaded_at": loaded.loaded_at}


class UnloadRequest(BaseModel):
    name: Optional[str] = None
    model_config = {"extra": "ignore"}


@app.post("/api/models/unload")
def api_unload(req: Optional[UnloadRequest] = None) -> dict:
    name = req.name if req else None
    return {"ok": engine.unload(name)}


@app.get("/api/models/pool")
def api_models_pool() -> dict:
    """Return all loaded models in the pool, newest-used first."""
    return {"models": [_pool_model_state(m) for m in engine.loaded_models()]}


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    cur = engine.current
    if not cur:
        raise HTTPException(status_code=409, detail="No model loaded.")

    req = _resolve_gen(cur, req)
    rendered = _render_prompt(cur.tokenizer, req.prompt, req.system)

    if req.stream:
        return StreamingResponse(
            _stream_generation(cur, rendered, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, tokens, _fr = await asyncio.to_thread(_generate_blocking, cur, rendered, req)
    cur.generations += 1
    cur.total_tokens += tokens
    cur.last_used = time.time()
    return {"text": text, "tokens": tokens}


def _generate_blocking(
    cur: LoadedModel, rendered: str, req: GenerateRequest,
    starts_in_think: bool = False,
    images: list = (),
) -> tuple[str, int, str]:
    """Run blocking generation. Returns (text, token_count, finish_reason).

    Also updates cur.last_ttft and cur.last_tps so the dashboard can show
    per-generation performance without needing a separate stats endpoint.
    """
    # ── KV-cache lookup ──────────────────────────────────────────────
    prompt_cache: Optional[list] = None
    if KVC_ENABLED and not cur.is_vlm:
        try:
            _tok = getattr(cur.tokenizer, "tokenizer", cur.tokenizer)
            _ids = _tok.encode(rendered)
            _hit = kvc.find(cur.name, cur.model, _ids)
            if _hit:
                prompt_cache, _skip = _hit
                log.info("kvc: blocking — resuming from %d cached tokens", _skip)
        except Exception as _e:
            log.debug("kvc: lookup error: %s", _e)
        if prompt_cache is None and hasattr(cur.model, "make_cache"):
            try:
                prompt_cache = cur.model.make_cache()
            except Exception:
                pass

    # Build iterator before acquiring the lock — stream_generate is lazy.
    iterator = _make_iterator(cur, rendered, req, images, prompt_cache=prompt_cache)

    parts: list[str] = []
    token_count = 0
    finish_reason = "stop"
    t_first_token: Optional[float] = None

    t0 = time.time()
    with engine.gen_lock:
        waited = time.time() - t0
        if waited > 0.1:
            log.info("MLX (blocking) queued for %.1fs before starting", waited)
        t_gen_start = time.time()
        for c in iterator:
            piece = getattr(c, "text", c) if not isinstance(c, str) else c
            if piece:
                if t_first_token is None:
                    t_first_token = time.time()
                parts.append(piece)
            token_count += 1
            fr = getattr(c, "finish_reason", None)
            if fr:
                finish_reason = fr
        t_gen_end = time.time()

    # Store perf metrics on the LoadedModel for dashboard display.
    if t_first_token is not None:
        cur.last_ttft = t_first_token - t_gen_start
    elapsed = t_gen_end - t_gen_start
    cur.last_tps = token_count / elapsed if elapsed > 0 else None

    # ── KV-cache store ────────────────────────────────────────────────
    if prompt_cache is not None and not cur.is_vlm:
        try:
            _tok = getattr(cur.tokenizer, "tokenizer", cur.tokenizer)
            _ids = _tok.encode(rendered)
            kvc.store(cur.name, _ids, prompt_cache)
        except Exception as _e:
            log.debug("kvc: store error: %s", _e)

    text = "".join(parts)
    if text:
        text = ThinkStripper(
            enabled=_strip_thinking_enabled(cur),
            starts_in_think=starts_in_think,
        ).strip(text)
    return text, token_count, finish_reason


async def _stream_generation(cur: LoadedModel, rendered: str, req: GenerateRequest) -> AsyncIterator[bytes]:
    """Stream tokens via Server-Sent Events by running the blocking MLX generator in a thread."""
    from mlx_lm import stream_generate

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sentinel = object()

    def worker():
        token_count = 0
        t0 = time.time()
        with engine.gen_lock:
            waited = time.time() - t0
            if waited > 0.1:
                log.info("MLX (stream) queued for %.1fs before starting", waited)
            try:
                try:
                    from mlx_lm.sample_utils import make_sampler

                    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
                    iterator = stream_generate(cur.model, cur.tokenizer, prompt=rendered, max_tokens=req.max_tokens, sampler=sampler)
                except Exception:
                    iterator = stream_generate(cur.model, cur.tokenizer, prompt=rendered, max_tokens=req.max_tokens, temp=req.temperature)
                for chunk in iterator:
                    piece = getattr(chunk, "text", chunk) if not isinstance(chunk, str) else chunk
                    if piece:
                        token_count += 1
                        asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put({"error": str(e)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put((sentinel, token_count)), loop)

    asyncio.get_running_loop().run_in_executor(None, worker)

    yield b": mlxr stream open\n\n"
    produced = 0
    stripper = ThinkStripper(
        enabled=_strip_thinking_enabled(cur),
        starts_in_think=_prompt_starts_in_think(rendered),
    )
    while True:
        item = await queue.get()
        if isinstance(item, tuple) and item and item[0] is sentinel:
            produced = item[1]
            break
        if isinstance(item, dict) and "error" in item:
            yield f"event: error\ndata: {json.dumps(item)}\n\n".encode()
            break
        visible = stripper.feed(item)
        if visible:
            yield f"data: {json.dumps({'delta': visible})}\n\n".encode()
    tail = stripper.flush()
    if tail:
        yield f"data: {json.dumps({'delta': tail})}\n\n".encode()

    cur.generations += 1
    cur.total_tokens += produced
    cur.last_used = time.time()
    yield f"event: done\ndata: {json.dumps({'tokens': produced})}\n\n".encode()


# ---- OpenAI-compatible endpoints ----------------------------------------
# Lets AI clients (OpenAI SDK, Cursor, Continue, OpenWebUI, LibreChat, Raycast,
# etc.) connect to MLXr as if it were an OpenAI-compatible server.


class OAIToolCall(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    # {"name": str, "arguments": str (JSON-encoded)}
    function: dict


class OAIMessage(BaseModel):
    role: str
    # Content is None for assistant messages that only contain tool_calls.
    # OpenAI allows content to be a string OR an array of content-part objects
    # (e.g. [{"type": "text", "text": "..."}]). Zed's agent mode sends arrays.
    # We accept Any here and normalise to str in the handler.
    content: Optional[Any] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[OAIToolCall]] = None

    def text_content(self) -> Optional[str]:
        """Return content as a plain string regardless of whether the client
        sent a str or an OpenAI-style content-parts array."""
        if self.content is None:
            return None
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            # Extract text from content-parts: [{"type": "text", "text": "..."}, ...]
            parts = []
            for part in self.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text") or "")
                    elif part.get("type") == "image_url":
                        pass  # silently skip images — model can't see them
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(self.content)


class OAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[OAIMessage]
    # Accept both max_tokens (legacy) and max_completion_tokens (OpenAI v2).
    # No bounds — the model's context window is the real limit.
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = None  # pass-through; model clamps internally
    top_p: Optional[float] = None
    stream: bool = False
    stream_options: Optional[Any] = None  # {"include_usage": bool}
    # Advanced sampling params — forwarded to mlx-lm make_sampler when supported.
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    # Accepted for OpenAI API compatibility but not forwarded (mlx-lm lacks support).
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    # Tool use. `tools` follows OpenAI's function-tool schema:
    #   [{"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON schema>}}]
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None  # "auto" | "none" | "required" | {type, function}
    # JSON / structured-output mode.
    # {"type": "json_object"} — injects a system instruction and strips non-JSON
    #   prefix/suffix.  Full schema-constrained generation is not yet supported.
    # {"type": "text"} — default, pass-through.
    response_format: Optional[dict] = None

    model_config = {"extra": "ignore"}  # silently drop unknown fields (n, logprobs, etc.)

    def effective_max_tokens(self) -> Optional[int]:
        """Prefer max_completion_tokens if set, fall back to max_tokens."""
        return self.max_completion_tokens or self.max_tokens

    def include_usage(self) -> bool:
        """Whether the client asked for a usage chunk at end of stream."""
        if isinstance(self.stream_options, dict):
            return bool(self.stream_options.get("include_usage"))
        return False


@app.get("/v1/models")
def v1_models() -> dict:
    data = []
    for m in engine.loaded_models():
        saved = settings.get_model(m.name)
        alias = saved.get("alias")
        # Always list the canonical HuggingFace repo-id.
        data.append({
            "id": m.name,
            "object": "model",
            "created": int(m.loaded_at),
            "owned_by": "mlxr",
            "context_length": m.context_length,  # non-standard but useful
        })
        # Also expose the alias so clients that have it hardcoded can find the model.
        if alias and alias != m.name:
            data.append({
                "id": alias,
                "object": "model",
                "created": int(m.loaded_at),
                "owned_by": "mlxr",
                "context_length": m.context_length,
            })
    return {"object": "list", "data": data}


class OAICompletionRequest(BaseModel):
    """OpenAI /v1/completions (legacy text-completion) request schema."""
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    stream: bool = False
    stream_options: Optional[Any] = None
    suffix: Optional[str] = None   # accepted, not used
    model_config = {"extra": "ignore"}

    def effective_max_tokens(self) -> Optional[int]:
        return self.max_completion_tokens or self.max_tokens

    def include_usage(self) -> bool:
        if isinstance(self.stream_options, dict):
            return bool(self.stream_options.get("include_usage"))
        return False


@app.post("/v1/completions")
async def v1_completions(req: OAICompletionRequest):
    """Legacy raw-text completion endpoint (not chat).

    Passes the prompt straight to the model without any chat template, which
    is what the OpenAI spec describes.  Useful for Aider's –-model=openai/…
    mode and other tools that use the older completions API.
    """
    cur = _resolve_model(req.model)
    if not cur:
        return _oai_error(503, "no_model_loaded", "No model is loaded in MLXr. Load one from the dashboard first.")

    saved = settings.get_model(cur.name)
    model_id = req.model or cur.name
    completion_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    client_max = req.effective_max_tokens()
    saved_max = saved.get("max_tokens")
    resolved_max = client_max or saved_max or DEFAULT_GEN["max_tokens"]

    gen_req = GenerateRequest(
        prompt=req.prompt,
        max_tokens=resolved_max,
        temperature=req.temperature if req.temperature is not None else saved.get("temperature", DEFAULT_GEN["temperature"]),
        top_p=req.top_p if req.top_p is not None else saved.get("top_p", DEFAULT_GEN["top_p"]),
        min_p=req.min_p,
        repetition_penalty=req.repetition_penalty,
        stream=req.stream,
    )

    if req.stream:
        return StreamingResponse(
            _oai_stream_completions(cur, req.prompt, gen_req, model_id, completion_id, created, req.include_usage()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, tokens, finish_reason = await asyncio.to_thread(
        _generate_blocking, cur, req.prompt, gen_req, False,
    )
    cur.generations += 1
    cur.total_tokens += tokens
    cur.last_used = time.time()
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "system_fingerprint": None,
        "choices": [{
            "text": text or "",
            "index": 0,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": tokens,
            "total_tokens": tokens,
        },
    }


async def _oai_stream_completions(
    cur: LoadedModel, prompt: str, req: GenerateRequest, model_id: str,
    completion_id: str, created: int, include_usage: bool = False,
) -> AsyncIterator[bytes]:
    """Stream text.completion.chunk events for /v1/completions."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sentinel = object()

    def cmpl_chunk(text: str, finish_reason: Optional[str] = None) -> bytes:
        payload = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "system_fingerprint": None,
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    def worker():
        t0 = time.time()
        with engine.gen_lock:
            waited = time.time() - t0
            if waited > 0.1:
                log.info("MLX (completions stream) queued for %.1fs", waited)
            t_gen_start = time.time()
            asyncio.run_coroutine_threadsafe(queue.put({"__gen_start__": t_gen_start}), loop)
            # ── KV-cache lookup (in worker thread, holds gen_lock) ────
            _prompt_cache: Optional[list] = None
            if KVC_ENABLED and not cur.is_vlm:
                try:
                    _tok = getattr(cur.tokenizer, "tokenizer", cur.tokenizer)
                    _ids = _tok.encode(prompt)
                    _hit = kvc.find(cur.name, cur.model, _ids)
                    if _hit:
                        _prompt_cache, _skip = _hit
                        log.info("kvc: completions stream — resuming from %d cached tokens", _skip)
                except Exception as _e:
                    log.debug("kvc: completions stream lookup error: %s", _e)
                if _prompt_cache is None and hasattr(cur.model, "make_cache"):
                    try:
                        _prompt_cache = cur.model.make_cache()
                    except Exception:
                        pass
            try:
                iterator = _make_iterator(cur, prompt, req, prompt_cache=_prompt_cache)
                gen_finish_reason = "stop"
                for c in iterator:
                    piece = getattr(c, "text", c) if not isinstance(c, str) else c
                    if piece:
                        asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
                    fr = getattr(c, "finish_reason", None)
                    if fr:
                        gen_finish_reason = fr
                asyncio.run_coroutine_threadsafe(queue.put({"__finish_reason__": gen_finish_reason}), loop)
                # ── KV-cache store ──────────────────────────────────────
                if _prompt_cache is not None and not cur.is_vlm:
                    try:
                        _tok = getattr(cur.tokenizer, "tokenizer", cur.tokenizer)
                        _ids = _tok.encode(prompt)
                        kvc.store(cur.name, _ids, _prompt_cache)
                    except Exception as _e:
                        log.debug("kvc: completions stream store error: %s", _e)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put({"__error__": str(e)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

    loop.run_in_executor(None, worker)

    finish_reason = "stop"
    token_count = 0
    t_gen_start: Optional[float] = None
    t_first_token: Optional[float] = None

    stripper = ThinkStripper(
        enabled=_strip_thinking_enabled(cur),
        starts_in_think=False,
    )

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, dict) and "__gen_start__" in item:
            t_gen_start = item["__gen_start__"]
            continue
        if isinstance(item, dict) and "__finish_reason__" in item:
            finish_reason = item["__finish_reason__"]
            continue
        if isinstance(item, dict) and "__error__" in item:
            log.warning("completions: stream error: %s", item["__error__"])
            yield cmpl_chunk("", finish_reason="stop")
            finish_reason = "stop"
            break
        token_count += 1
        if t_first_token is None:
            t_first_token = time.time()
        visible = stripper.feed(item)
        if visible:
            yield cmpl_chunk(visible)

    tail = stripper.flush()
    if tail:
        yield cmpl_chunk(tail)

    yield cmpl_chunk("", finish_reason=finish_reason)

    if include_usage:
        usage_payload = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "system_fingerprint": None,
            "choices": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": token_count,
                "total_tokens": token_count,
            },
        }
        yield f"data: {json.dumps(usage_payload)}\n\n".encode()

    yield b"data: [DONE]\n\n"

    cur.generations += 1
    cur.total_tokens += token_count
    cur.last_used = time.time()
    t_end = time.time()
    if t_first_token and t_gen_start:
        cur.last_ttft = t_first_token - t_gen_start
    if t_gen_start:
        elapsed = t_end - t_gen_start
        cur.last_tps = token_count / elapsed if elapsed > 0 else None


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: OAIChatRequest):
    cur = _resolve_model(req.model)
    if not cur:
        # Use OpenAI's error envelope so SDK clients surface the message cleanly.
        return _oai_error(503, "no_model_loaded", "No model is loaded in MLXr. Load one from the dashboard first.")

    # Build the message list in the shape chat templates expect. Keep
    # tool_call_id/tool_calls so tool-turn conversations round-trip correctly.
    #
    # CRITICAL: OpenAI's spec encodes ``tool_calls[*].function.arguments`` as a
    # JSON **string**, but Qwen3's chat template (and most others) iterates the
    # arguments as a **dict** (``{% for k,v in arguments|items %}``). If we pass
    # the string straight through, the template raises on ``str.items()`` and
    # every multi-turn conversation after a tool call produces garbage. Parse
    # the JSON back to a dict before templating.
    saved = settings.get_model(cur.name)
    messages: list[dict] = []
    for m in req.messages:
        entry: dict[str, Any] = {"role": m.role, "content": m.text_content() if m.content is not None else ""}
        if m.name:
            entry["name"] = m.name
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            normalized_calls = []
            for tc in m.tool_calls:
                tc_dict = tc.model_dump(exclude_none=True)
                fn = tc_dict.get("function") or {}
                args = fn.get("arguments")
                # OpenAI sends arguments as a JSON-encoded string; the chat
                # template needs a dict. Parse it back, but tolerate
                # already-dict values (some non-OpenAI clients send dicts)
                # and unparseable strings (pass them through as {"_raw": str}
                # so the template at least doesn't crash).
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except Exception:
                        log.warning("assistant tool_call arguments not valid JSON: %r", args[:200])
                        fn["arguments"] = {"_raw": args}
                elif args is None:
                    fn["arguments"] = {}
                tc_dict["function"] = fn
                normalized_calls.append(tc_dict)
            entry["tool_calls"] = normalized_calls
            # When an assistant message carries tool_calls, content is usually
            # empty; keep it as an empty string so templates don't crash.
        messages.append(entry)

    # Extract image URLs from multimodal content parts.
    messages, image_urls = _extract_images(messages)

    # Inject saved system prompt if the client didn't send one.
    if not any(m["role"] == "system" for m in messages) and saved.get("system"):
        messages.insert(0, {"role": "system", "content": saved["system"]})

    # JSON mode — response_format: {"type": "json_object"}.
    # Injects a short system instruction so that models that don't natively
    # support the JSON mode flag still produce well-formed JSON output.
    # We append (not prepend) so the instruction is close to the generation
    # boundary and not buried under a long user system prompt.
    if req.response_format and req.response_format.get("type") == "json_object":
        json_instr = (
            "You MUST respond with valid JSON only. "
            "Do not include any explanation, markdown fences, or text outside the JSON object."
        )
        sys_msgs = [i for i, m in enumerate(messages) if m["role"] == "system"]
        if sys_msgs:
            messages[sys_msgs[-1]]["content"] += "\n\n" + json_instr
        else:
            messages.append({"role": "system", "content": json_instr})
        log.info("chat: JSON mode active — injected response_format instruction")

    # If the caller passed tool_choice="none", suppress tools entirely so
    # the template doesn't advertise any.
    tools_for_template = req.tools if (req.tools and req.tool_choice != "none") else None

    # Enforce tool_choice="required" and tool_choice={"function":{"name":"X"}}.
    #
    # Most open-source chat templates (including Qwen3's) don't have a native
    # tool_choice concept — the IMPORTANT reminder allows the model to skip
    # tool calls entirely. We enforce it by injecting a system instruction that
    # overrides that allowance. This is appended AFTER any existing system
    # message so it appears closest to the generation point and isn't buried.
    if tools_for_template and req.tool_choice and req.tool_choice not in ("auto", "none"):
        if req.tool_choice == "required":
            tool_names_str = ", ".join(
                t.get("function", {}).get("name", "") for t in tools_for_template
            )
            enforcement = (
                f"You MUST respond by calling one of the available tools "
                f"({tool_names_str}). Do NOT answer in plain text — a tool call "
                f"is required for this turn."
            )
        elif isinstance(req.tool_choice, dict):
            forced_name = (req.tool_choice.get("function") or {}).get("name", "")
            if forced_name:
                enforcement = (
                    f"You MUST call the '{forced_name}' tool on this turn. "
                    f"Do not answer in plain text."
                )
            else:
                enforcement = None
        else:
            enforcement = None

        if enforcement:
            # Insert immediately before the final user turn so it's in scope.
            # If there's already a trailing system message we update it; otherwise
            # we append a fresh one.
            if messages and messages[-1]["role"] == "system":
                messages[-1]["content"] = messages[-1]["content"] + "\n\n" + enforcement
            else:
                messages.append({"role": "system", "content": enforcement})
            log.info("chat: tool_choice=%r — injected enforcement instruction", req.tool_choice)

    # enable_thinking resolution: saved setting → auto (False if tools else True).
    saved_enable_thinking = saved.get("enable_thinking")
    if saved_enable_thinking is None:
        enable_thinking = not bool(tools_for_template)
    else:
        enable_thinking = bool(saved_enable_thinking)

    if cur.is_vlm and image_urls:
        prompt = _render_vlm_prompt(cur, messages, image_urls)
    else:
        prompt = _render_chat(
            cur.tokenizer, messages, tools=tools_for_template, enable_thinking=enable_thinking,
        )
    starts_in_think = _prompt_starts_in_think(prompt)
    # Resolve model_id: prefer the alias (if set) so responses echo back the
    # same name the client used, preserving round-trip compatibility.
    alias = saved.get("alias")
    model_id = req.model or alias or cur.name
    tools_active = bool(tools_for_template)

    tool_names = [t.get("function", {}).get("name") for t in (tools_for_template or [])]
    entry_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    log.info(
        "chat: id=%s model=%s msgs=%d tools=%d enable_thinking=%s starts_in_think=%s stream=%s",
        entry_id,
        model_id,
        len(messages),
        len(tools_for_template or []),
        enable_thinking,
        starts_in_think,
        req.stream,
    )
    if tools_for_template:
        log.info("chat: id=%s tool_names=%s", entry_id, tool_names)
    # Log the tail of the rendered prompt so it's visible in the server log
    # whether the template actually advertised tools / opened a <think> block.
    prompt_tail = prompt[-600:] if len(prompt) > 600 else prompt
    log.info("chat: id=%s prompt_tail=%r", entry_id, prompt_tail)

    _record_chat({
        "id": entry_id,
        "created": int(time.time()),
        "model": model_id,
        "messages": len(messages),
        "tools": tool_names,
        "enable_thinking": enable_thinking,
        "starts_in_think": starts_in_think,
        "stream": req.stream,
        "prompt_tail": prompt_tail,
        "prompt_length": len(prompt),
        "output_preview": None,
        "tool_calls_emitted": 0,
        "finish_reason": None,
    })

    # Token budget: client value > saved model setting > default.
    # When tools are active and neither the client nor the user set a limit,
    # escalate to DEFAULT_TOOLS_MAX_TOKENS so tool-call arguments (e.g. a
    # write_file body) are never truncated mid-JSON by a conservative default.
    client_max = req.effective_max_tokens()
    saved_max = saved.get("max_tokens")
    if client_max is not None:
        resolved_max_tokens = client_max
    elif saved_max is not None:
        resolved_max_tokens = saved_max
    elif tools_active:
        resolved_max_tokens = DEFAULT_TOOLS_MAX_TOKENS
    else:
        resolved_max_tokens = DEFAULT_GEN["max_tokens"]

    gen_req = GenerateRequest(
        prompt="",  # not used for chat-template path
        max_tokens=resolved_max_tokens,
        temperature=req.temperature if req.temperature is not None else saved.get("temperature", DEFAULT_GEN["temperature"]),
        top_p=req.top_p if req.top_p is not None else saved.get("top_p", DEFAULT_GEN["top_p"]),
        min_p=req.min_p,
        repetition_penalty=req.repetition_penalty,
        stream=req.stream,
    )

    if req.stream:
        return StreamingResponse(
            _oai_stream_chat(
                cur, prompt, gen_req, model_id,
                tools_active=tools_active, starts_in_think=starts_in_think,
                entry_id=entry_id, include_usage=req.include_usage(),
                images=image_urls,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, tokens, gen_finish_reason = await asyncio.to_thread(
        _generate_blocking, cur, prompt, gen_req, starts_in_think, image_urls,
    )
    # Record what the model actually produced for diagnostics.
    _update_chat(entry_id, {
        "output_preview": (text or "")[:800],
        "tokens": tokens,
    })
    log.info("chat: id=%s blocking output preview=%r", entry_id, (text or "")[:400])
    if gen_finish_reason == "length" and tools_active:
        log.warning(
            "chat: id=%s hit max_tokens=%d (blocking) while tools were active — "
            "tool-call arguments may be truncated.",
            entry_id, gen_req.max_tokens,
        )
    cur.generations += 1
    cur.total_tokens += tokens
    cur.last_used = time.time()

    # Extract tool calls from the completed text if tools were requested.
    tool_calls: list[dict] = []
    content_text = text
    if tools_active:
        parser = ToolCallParser(enabled=True)
        content, tools_in_stream = parser.feed(text)
        tail_content, tail_tools = parser.flush()
        content_text = content + tail_content
        tool_calls = tools_in_stream + tail_tools
        # Raw-JSON fallback: Llama 3.x and some other models emit a bare JSON
        # object with no wrapper tags.  Only fires when the parser found no
        # tagged tool calls and the entire output looks like a JSON tool call.
        if not tool_calls and content_text:
            fallback = ToolCallParser.try_extract_raw_json(content_text)
            if fallback:
                tool_calls = fallback
                content_text = ""
                log.info(
                    "chat: id=%s raw-JSON tool call detected (Llama/bare-JSON fallback)",
                    entry_id,
                )

    # Build the assistant message per OpenAI spec.
    # content is null when: (a) tool_calls are present with no text, or
    # (b) the model generated nothing visible (all output was stripped thinking
    # blocks). Returning "" in case (b) confuses clients — null is cleaner.
    cleaned = content_text.strip()
    message: dict[str, Any] = {
        "role": "assistant",
        "content": cleaned if cleaned else None,
        "refusal": None,  # required field in spec; null when not refusing
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = "tool_calls" if tool_calls else gen_finish_reason
    _update_chat(entry_id, {
        "tool_calls_emitted": len(tool_calls),
        "finish_reason": finish_reason,
    })
    if tool_calls:
        log.info("chat: id=%s parsed %d tool_call(s) from blocking output", entry_id, len(tool_calls))
    return {
        "id": entry_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,  # required field in spec; null when not requested
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": tokens,
            "total_tokens": tokens,
        },
    }


def _oai_error(status: int, code: str, message: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "mlxr_error", "code": code}},
    )


def _render_chat(
    tokenizer: Any,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Apply the tokenizer's chat template.

    ``enable_thinking`` is a Qwen-family template flag (ignored by non-Qwen
    templates). Default: ``False`` when tools are present (Qwen's own
    recommendation — reasoning mode and tool use together often makes the
    model narrate instead of actually emitting a tool_call), else ``True``.

    Gracefully degrades when the tokenizer rejects specific kwargs: some
    templates don't accept ``enable_thinking``, and we must NOT silently fall
    through to plain-join in that case — it would produce garbage output.
    """
    if enable_thinking is None:
        enable_thinking = not bool(tools)

    if getattr(tokenizer, "chat_template", None):
        base_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if tools:
            base_kwargs["tools"] = tools

        # Attempt 1: full kwargs.
        try:
            return tokenizer.apply_chat_template(
                messages, **base_kwargs, enable_thinking=enable_thinking,
            )
        except TypeError as e:
            log.info("chat_template rejected enable_thinking kwarg: %s — retrying without", e)
        except Exception as e:
            # Might be the template raising on a field inside a message. Try
            # once more without enable_thinking in case that's the culprit.
            log.warning("chat_template raised with enable_thinking: %s — retrying without", e)

        # Attempt 2: without enable_thinking.
        try:
            return tokenizer.apply_chat_template(messages, **base_kwargs)
        except Exception as e:
            log.warning("chat_template failed without enable_thinking: %s — retrying without tools", e)

        # Attempt 3: without tools (last-ditch — we'll lose tool-calling but
        # at least get valid chat formatting).
        if tools:
            try:
                minimal = {k: v for k, v in base_kwargs.items() if k != "tools"}
                result = tokenizer.apply_chat_template(messages, **minimal)
                log.warning("chat_template couldn't accept tools — tools won't be advertised")
                return result
            except Exception as e:
                log.warning("chat_template failed even without tools: %s", e)

    # Plain-join fallback: only reached if the tokenizer has no template
    # at all or every attempt raised. Signals a serious config problem.
    log.error("No working chat_template — using plain-join fallback (model output will likely be garbage)")
    out = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        out.append(f"{m['role']}: {content}")
    return "\n".join(out) + "\nassistant:"


# Ring buffer of recent /v1/chat/completions calls for debugging.
_RECENT_CHATS: list[dict] = []
_RECENT_CHATS_MAX = 10
_RECENT_CHATS_LOCK = Lock()


def _record_chat(entry: dict) -> None:
    with _RECENT_CHATS_LOCK:
        _RECENT_CHATS.append(entry)
        if len(_RECENT_CHATS) > _RECENT_CHATS_MAX:
            del _RECENT_CHATS[: len(_RECENT_CHATS) - _RECENT_CHATS_MAX]


def _update_chat(entry_id: str, updates: dict) -> None:
    with _RECENT_CHATS_LOCK:
        for e in _RECENT_CHATS:
            if e.get("id") == entry_id:
                e.update(updates)
                return


def _prompt_starts_in_think(prompt: str) -> bool:
    """Detect whether the chat template left the stream starting inside a
    ``<think>`` block (i.e. the template appended ``<think>`` to the prompt)."""
    tail = prompt.rstrip()
    return tail.endswith("<think>")


async def _oai_stream_chat(
    cur: LoadedModel, prompt: str, req: GenerateRequest, model_id: str,
    tools_active: bool = False, starts_in_think: bool = False,
    entry_id: Optional[str] = None, include_usage: bool = False,
    images: list = (),
) -> AsyncIterator[bytes]:
    """Stream chat.completion.chunk events in OpenAI's SSE format."""
    from mlx_lm import stream_generate

    completion_id = entry_id or f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    # Buffer raw (pre-strip, pre-parse) output so we can log a preview at end-of-stream.
    raw_output_capture: list[str] = []
    raw_output_cap = 2000  # chars

    def chunk(delta: dict, finish_reason: Optional[str] = None) -> bytes:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "system_fingerprint": None,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    # role announcement, per OpenAI spec
    yield chunk({"role": "assistant"})

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sentinel = object()
    token_count = 0

    def worker():
        t0 = time.time()
        with engine.gen_lock:
            waited = time.time() - t0
            if waited > 0.1:
                log.info("MLX (chat stream) queued for %.1fs before starting", waited)
            # Announce the moment generation actually starts so the consumer
            # can compute Time To First Token without clock-skew from queuing.
            t_gen_start = time.time()
            asyncio.run_coroutine_threadsafe(
                queue.put({"__gen_start__": t_gen_start}), loop
            )
            # ── KV-cache lookup (in worker thread, holds gen_lock) ────
            _prompt_cache: Optional[list] = None
            if KVC_ENABLED and not cur.is_vlm:
                try:
                    _tok = getattr(cur.tokenizer, "tokenizer", cur.tokenizer)
                    _ids = _tok.encode(prompt)
                    _hit = kvc.find(cur.name, cur.model, _ids)
                    if _hit:
                        _prompt_cache, _skip = _hit
                        log.info("kvc: stream — resuming from %d cached tokens", _skip)
                except Exception as _e:
                    log.debug("kvc: stream lookup error: %s", _e)
                if _prompt_cache is None and hasattr(cur.model, "make_cache"):
                    try:
                        _prompt_cache = cur.model.make_cache()
                    except Exception:
                        pass
            try:
                iterator = _make_iterator(cur, prompt, req, images, prompt_cache=_prompt_cache)
                gen_finish_reason = "stop"
                for c in iterator:
                    piece = getattr(c, "text", c) if not isinstance(c, str) else c
                    if piece:
                        asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
                    # mlx-lm sets finish_reason on the final GenerationResponse.
                    fr = getattr(c, "finish_reason", None)
                    if fr:
                        gen_finish_reason = fr
                # Send finish_reason to the consumer via a sentinel dict.
                asyncio.run_coroutine_threadsafe(
                    queue.put({"__finish_reason__": gen_finish_reason}), loop
                )
                # ── KV-cache store ──────────────────────────────────────
                if _prompt_cache is not None and not cur.is_vlm:
                    try:
                        _tok = getattr(cur.tokenizer, "tokenizer", cur.tokenizer)
                        _ids = _tok.encode(prompt)
                        kvc.store(cur.name, _ids, _prompt_cache)
                    except Exception as _e:
                        log.debug("kvc: stream store error: %s", _e)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put({"__error__": str(e)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

    loop.run_in_executor(None, worker)

    stripper = ThinkStripper(
        enabled=_strip_thinking_enabled(cur),
        starts_in_think=starts_in_think,
    )
    tool_parser = ToolCallParser(enabled=tools_active)
    tool_index = 0
    tools_emitted = 0
    finish_reason = "stop"
    # TTFT tracking: populated from __gen_start__ worker message + first token.
    t_gen_start: Optional[float] = None
    t_first_token: Optional[float] = None

    # Raw-JSON tool-call buffer — for models that emit bare JSON with no
    # wrapper tags (Llama 3.x default chat template, Gemma 3 IT).
    # When tools are active and the first visible character is '{', we hold
    # back content until either we confirm it's NOT a JSON tool call (content
    # starts with a non-'{' char, OR a recognised tag is seen, OR buffer
    # overflows) and then stream normally, or the stream ends and we attempt
    # a JSON parse.  For all wrapper-tag models this is a ~zero-overhead
    # no-op: the first visible char is '<' (tag opener), so rj_streaming
    # switches to True on the very first content chunk.
    rj_buf: list[str] = []         # pending content while in detect/buffer mode
    rj_streaming = not tools_active  # True → emit content chunks directly
    RJ_BUF_MAX = 4096              # give up buffering beyond this many chars

    def emit_tool(tool: dict) -> bytes:
        """Emit a complete tool_call in one delta chunk. OpenAI clients that
        expect per-arg-chunk deltas still concatenate empty strings, so sending
        the full payload once is interoperable."""
        nonlocal tool_index
        delta = {
            "tool_calls": [
                {
                    "index": tool_index,
                    "id": tool["id"],
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "arguments": tool["function"]["arguments"],
                    },
                }
            ]
        }
        tool_index += 1
        return chunk(delta)

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, dict) and "__gen_start__" in item:
            t_gen_start = item["__gen_start__"]
            continue
        if isinstance(item, dict) and "__finish_reason__" in item:
            finish_reason = item["__finish_reason__"]  # "stop" or "length" from mlx-lm
            continue
        if isinstance(item, dict) and "__error__" in item:
            log.warning("chat: id=%s stream error: %s", completion_id, item["__error__"])
            # "stop" is the only safe finish_reason for unexpected errors;
            # "error" is not a valid OpenAI spec value and breaks strict clients.
            yield chunk({}, finish_reason="stop")
            finish_reason = "stop"
            break
        token_count += 1
        if t_first_token is None:
            t_first_token = time.time()
        # Capture for diagnostics, up to cap.
        if sum(len(s) for s in raw_output_capture) < raw_output_cap:
            raw_output_capture.append(item)
        visible = stripper.feed(item)
        if not visible:
            continue
        content, new_tools = tool_parser.feed(visible)
        # If the tag-parser found a tagged tool call, switch to streaming mode
        # (we know it's not a bare-JSON model) and flush any pending buffer.
        if new_tools and not rj_streaming:
            rj_streaming = True
            if rj_buf:
                yield chunk({"content": "".join(rj_buf)})
                rj_buf.clear()
        if content:
            if rj_streaming:
                yield chunk({"content": content})
            else:
                # Decide on streaming vs. buffering using the first visible char.
                combined = "".join(rj_buf) + content
                first_visible = combined.lstrip()
                if first_visible and first_visible[0] != "{":
                    # Not a JSON tool call — stream from now on.
                    rj_streaming = True
                    yield chunk({"content": combined})
                    rj_buf.clear()
                else:
                    rj_buf.append(content)
                    if len("".join(rj_buf)) > RJ_BUF_MAX:
                        # Buffer too large — treat as plain content.
                        rj_streaming = True
                        yield chunk({"content": "".join(rj_buf)})
                        rj_buf.clear()
        for t in new_tools:
            yield emit_tool(t)
            tools_emitted += 1

    # Flush both stages.
    stripper_tail = stripper.flush()
    if stripper_tail:
        content, new_tools = tool_parser.feed(stripper_tail)
        if new_tools and not rj_streaming:
            rj_streaming = True
            if rj_buf:
                yield chunk({"content": "".join(rj_buf)})
                rj_buf.clear()
        if content:
            if rj_streaming:
                yield chunk({"content": content})
            else:
                rj_buf.append(content)
        for t in new_tools:
            yield emit_tool(t)
            tools_emitted += 1

    parser_tail, trailing_tools = tool_parser.flush()
    if parser_tail:
        if rj_streaming:
            yield chunk({"content": parser_tail})
        else:
            rj_buf.append(parser_tail)
    for t in trailing_tools:
        if not rj_streaming:
            rj_streaming = True
            if rj_buf:
                yield chunk({"content": "".join(rj_buf)})
                rj_buf.clear()
        yield emit_tool(t)
        tools_emitted += 1

    # Raw-JSON fallback for bare-JSON models (Llama 3.x etc.).
    if rj_buf and not rj_streaming and tools_emitted == 0:
        full_content = "".join(rj_buf)
        rj_buf.clear()
        fallback = ToolCallParser.try_extract_raw_json(full_content)
        if fallback:
            log.info(
                "chat: id=%s raw-JSON tool call detected in stream (Llama/bare-JSON fallback)",
                completion_id,
            )
            for t in fallback:
                yield emit_tool(t)
                tools_emitted += 1
        else:
            # It was just plain content that happened to start with '{'.
            yield chunk({"content": full_content})

    if tools_emitted and finish_reason == "stop":
        finish_reason = "tool_calls"

    # Warn when the context window was exhausted mid tool-call — this means
    # arguments were truncated and the tool call will likely be malformed.
    if finish_reason == "length" and tools_active:
        log.warning(
            "chat: id=%s hit max_tokens=%d while tools were active — tool-call "
            "arguments may be truncated. Consider raising max_tokens in the "
            "dashboard or sending a higher max_completion_tokens from the client.",
            completion_id, req.max_tokens,
        )

    yield chunk({}, finish_reason=finish_reason)

    # Optional usage chunk: spec says emit before [DONE] when include_usage=true.
    # choices must be an empty array in this chunk per spec.
    if include_usage:
        usage_payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "system_fingerprint": None,
            "choices": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": token_count,
                "total_tokens": token_count,
            },
        }
        yield f"data: {json.dumps(usage_payload)}\n\n".encode()

    yield b"data: [DONE]\n\n"

    cur.generations += 1
    cur.total_tokens += token_count
    cur.last_used = time.time()

    # Compute and store TTFT / throughput on the model so /api/status exposes them.
    t_stream_end = time.time()
    if t_first_token is not None and t_gen_start is not None:
        cur.last_ttft = t_first_token - t_gen_start
    if t_gen_start is not None:
        elapsed = t_stream_end - t_gen_start
        cur.last_tps = token_count / elapsed if elapsed > 0 else None

    raw_preview = "".join(raw_output_capture)[:800]
    log.info(
        "chat: id=%s stream done tokens=%d tool_calls=%d finish=%s ttft=%.3fs tps=%.1f raw_preview=%r",
        completion_id, token_count, tools_emitted, finish_reason,
        cur.last_ttft or 0, cur.last_tps or 0, raw_preview,
    )
    _update_chat(completion_id, {
        "output_preview": raw_preview,
        "tokens": token_count,
        "tool_calls_emitted": tools_emitted,
        "finish_reason": finish_reason,
        "ttft": cur.last_ttft,
        "tps": cur.last_tps,
    })


# ---- settings ------------------------------------------------------------


class ModelSettingsBody(BaseModel):
    system: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=131072)
    autoload: Optional[bool] = None
    # Strip <think>...</think> reasoning blocks from responses (default on).
    strip_thinking: Optional[bool] = None
    # Qwen-family ``enable_thinking`` chat-template flag. ``None`` = auto
    # (False when tools are present, True otherwise).
    enable_thinking: Optional[bool] = None
    # Auto-unload the model after this many minutes of inactivity. None = never.
    idle_timeout_minutes: Optional[int] = Field(default=None, ge=1, le=10080)
    # Friendly alias exposed through /v1/models and accepted in request.model.
    # Example: set alias="gpt-4o" so existing configs that hardcode that name
    # don't need changing.
    alias: Optional[str] = None


@app.get("/api/settings")
def api_settings_get() -> dict:
    return {"settings": settings.snapshot(), "path": str(settings.path), "defaults": DEFAULT_GEN}


@app.get("/api/settings/models/{repo_id:path}")
def api_settings_model_get(repo_id: str) -> dict:
    return {"name": repo_id, "settings": settings.get_model(repo_id)}


@app.put("/api/settings/models/{repo_id:path}")
def api_settings_model_put(repo_id: str, body: ModelSettingsBody) -> dict:
    values = body.model_dump(exclude_none=True)
    # If a field was explicitly set to null in the JSON body, pydantic's
    # exclude_none drops it — which means "revert to default". That's the
    # semantics we want. An empty dict means "no changes".
    saved = settings.set_model(repo_id, values)
    # If autoload was turned on for this repo, turn it off for all others —
    # we only support a single autoload model.
    if values.get("autoload") is True:
        snap = settings.snapshot().get("models", {})
        for other_id in list(snap.keys()):
            if other_id != repo_id and snap[other_id].get("autoload"):
                settings.set_model(other_id, {"autoload": False})
    return {"ok": True, "name": repo_id, "settings": saved}


@app.delete("/api/settings/models/{repo_id:path}")
def api_settings_model_delete(repo_id: str) -> dict:
    removed = settings.delete_model(repo_id)
    return {"ok": True, "removed": removed}


# ---- huggingface ---------------------------------------------------------


class HFDownloadRequest(BaseModel):
    name: str


class HFDeleteRequest(BaseModel):
    name: str


@app.get("/api/hf/search")
async def api_hf_search(q: str = "", author: str = "mlx-community", limit: int = 25) -> dict:
    limit = max(1, min(limit, 100))
    try:
        results = await asyncio.to_thread(hf.search, q, author or None, limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HF search failed: {e}")
    return {"query": q, "author": author, "results": results}


@app.post("/api/hf/download")
def api_hf_download(req: HFDownloadRequest) -> dict:
    job = hf.start_download(req.name.strip())
    return {"ok": True, "job": job.to_dict()}


@app.get("/api/hf/downloads")
def api_hf_downloads() -> dict:
    return {"jobs": hf.jobs()}


@app.get("/api/hf/cache")
async def api_hf_cache() -> dict:
    return await asyncio.to_thread(hf.cache)


@app.post("/api/hf/delete")
async def api_hf_delete(req: HFDeleteRequest) -> dict:
    try:
        return await asyncio.to_thread(hf.delete_repo, req.name.strip())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"delete failed: {e}")


# ---- Anthropic Messages API (/v1/messages) --------------------------------
# Maps the Anthropic Messages API wire format to MLXr's existing chat pipeline.
# Clients that speak Anthropic (Claude Code, some agent frameworks, Cursor's
# native mode) can connect without an adapter layer.


class _AnthrTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None  # JSON schema of tool parameters


class _AnthrToolChoice(BaseModel):
    type: str = "auto"  # "auto" | "any" | "tool"
    name: Optional[str] = None


class AnthropicRequest(BaseModel):
    model: Optional[str] = None
    messages: list[dict]
    max_tokens: int = Field(default=4096, ge=1)
    system: Optional[Any] = None   # str or list of content blocks
    tools: Optional[list[_AnthrTool]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None    # accepted, ignored (mlx-lm has top_k but not via make_sampler yet)
    stream: bool = False
    metadata: Optional[dict] = None
    stop_sequences: Optional[list[str]] = None
    model_config = {"extra": "ignore"}


def _anth_content_to_str(content: Any) -> str:
    """Collapse Anthropic content (str or list of blocks) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    # tool_result content may itself be a list or str
                    inner = block.get("content", "")
                    parts.append(_anth_content_to_str(inner))
                # image / document blocks: silently skip
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _anth_messages_to_oai(messages: list[dict], system: Any) -> list[dict]:
    """Convert Anthropic message list + system to OpenAI-style message list."""
    result: list[dict] = []
    # Anthropic puts the system prompt as a top-level field, not in messages.
    if system:
        sys_text = _anth_content_to_str(system)
        if sys_text:
            result.append({"role": "system", "content": sys_text})

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            tool_results: list[dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    # Anthropic tool_use → OpenAI tool_calls
                    tc = {
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:20]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                    tool_calls.append(tc)
                elif btype == "tool_result":
                    # Anthropic tool_result → OpenAI tool role message
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _anth_content_to_str(block.get("content", "")),
                    })
            if tool_results:
                result.extend(tool_results)
            elif tool_calls:
                entry: dict[str, Any] = {
                    "role": role,
                    "content": "".join(text_parts) or None,
                    "tool_calls": tool_calls,
                }
                result.append(entry)
            elif text_parts:
                result.append({"role": role, "content": "".join(text_parts)})
    return result


def _anth_tools_to_oai(tools: list[_AnthrTool]) -> list[dict]:
    """Convert Anthropic tool list to OpenAI function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _oai_finish_to_anth_stop(finish_reason: str) -> str:
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}.get(
        finish_reason, "end_turn"
    )


def _build_anth_response(
    msg_id: str, model_id: str, content_blocks: list[dict],
    stop_reason: str, input_tokens: int, output_tokens: int,
) -> dict:
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model_id,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


@app.post("/v1/messages")
async def v1_messages(req: AnthropicRequest):
    """Anthropic Messages API compatibility endpoint.

    Accepts the Anthropic SDK wire format and maps it to MLXr's chat pipeline.
    Enables Claude Code, some agent frameworks, and Cursor's native Anthropic
    mode to use locally-hosted MLX models without an adapter.
    """
    cur = _resolve_model(req.model)
    if not cur:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=529,
            content={"type": "error", "error": {
                "type": "overloaded_error",
                "message": "No model is loaded in MLXr. Load one from the dashboard first.",
            }},
        )

    saved = settings.get_model(cur.name)
    alias = saved.get("alias")
    model_id = req.model or alias or cur.name
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Convert Anthropic format to OpenAI messages.
    messages = _anth_messages_to_oai(req.messages, req.system)
    oai_tools: Optional[list[dict]] = _anth_tools_to_oai(req.tools) if req.tools else None

    # tool_choice mapping: Anthropic → OAI
    tool_choice: Any = "auto"
    if req.tool_choice:
        tc = req.tool_choice if isinstance(req.tool_choice, dict) else {}
        tc_type = tc.get("type", "auto")
        if tc_type == "any":
            tool_choice = "required"
        elif tc_type == "tool":
            tool_choice = {"type": "function", "function": {"name": tc.get("name", "")}}
        elif tc_type == "none":
            tool_choice = "none"
            oai_tools = None

    # Build an internal OAIChatRequest and reuse the existing pipeline.
    fake_req = OAIChatRequest(
        model=model_id,
        messages=[OAIMessage(**m) for m in messages],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        tools=oai_tools,
        tool_choice=tool_choice,
        stream=req.stream,
    )

    if not any(m["role"] == "system" for m in messages) and saved.get("system"):
        messages.insert(0, {"role": "system", "content": saved["system"]})

    enable_thinking = not bool(oai_tools)
    saved_et = saved.get("enable_thinking")
    if saved_et is not None:
        enable_thinking = bool(saved_et)

    prompt = _render_chat(cur.tokenizer, messages, tools=oai_tools, enable_thinking=enable_thinking)
    starts_in_think = _prompt_starts_in_think(prompt)

    client_max = req.max_tokens
    saved_max = saved.get("max_tokens")
    resolved_max = client_max or saved_max or (DEFAULT_TOOLS_MAX_TOKENS if oai_tools else DEFAULT_GEN["max_tokens"])

    gen_req = GenerateRequest(
        prompt="",
        max_tokens=resolved_max,
        temperature=req.temperature if req.temperature is not None else saved.get("temperature", DEFAULT_GEN["temperature"]),
        top_p=req.top_p if req.top_p is not None else saved.get("top_p", DEFAULT_GEN["top_p"]),
        stream=req.stream,
    )

    log.info("messages: id=%s model=%s msgs=%d tools=%d stream=%s",
             msg_id, model_id, len(messages), len(oai_tools or []), req.stream)

    if req.stream:
        return StreamingResponse(
            _anth_stream(cur, prompt, gen_req, model_id, msg_id,
                         bool(oai_tools), starts_in_think),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, tokens, finish_reason = await asyncio.to_thread(
        _generate_blocking, cur, prompt, gen_req, starts_in_think,
    )
    cur.generations += 1
    cur.total_tokens += tokens
    cur.last_used = time.time()

    # Parse tool calls from the raw output.
    content_blocks: list[dict] = []
    tool_calls: list[dict] = []
    if oai_tools:
        parser = ToolCallParser(enabled=True)
        content_txt, tls = parser.feed(text)
        tail_txt, tail_tls = parser.flush()
        content_txt = content_txt + tail_txt
        tool_calls = tls + tail_tls
        if not tool_calls and content_txt:
            fallback = ToolCallParser.try_extract_raw_json(content_txt)
            if fallback:
                tool_calls = fallback
                content_txt = ""
    else:
        content_txt = text

    cleaned = (content_txt or "").strip()
    if cleaned:
        content_blocks.append({"type": "text", "text": cleaned})
    for tc in tool_calls:
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments", "{}"))
        except Exception:
            inp = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:20]}"),
            "name": fn.get("name", ""),
            "input": inp,
        })

    stop_reason = _oai_finish_to_anth_stop("tool_calls" if tool_calls else finish_reason)
    return _build_anth_response(msg_id, model_id, content_blocks, stop_reason, 0, tokens)


async def _anth_stream(
    cur: LoadedModel, prompt: str, req: GenerateRequest,
    model_id: str, msg_id: str, tools_active: bool, starts_in_think: bool,
) -> AsyncIterator[bytes]:
    """Stream Anthropic SSE events for /v1/messages."""

    def sse(event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    # message_start
    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model_id,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        },
    })
    yield sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    yield sse("ping", {"type": "ping"})

    # Re-use the OpenAI streaming helper and translate deltas.
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sentinel = object()

    def worker():
        t0 = time.time()
        with engine.gen_lock:
            waited = time.time() - t0
            if waited > 0.1:
                log.info("MLX (anth stream) queued for %.1fs", waited)
            try:
                iterator = _make_iterator(cur, prompt, req)
                for c in iterator:
                    piece = getattr(c, "text", c) if not isinstance(c, str) else c
                    if piece:
                        asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
                    fr = getattr(c, "finish_reason", None)
                    if fr:
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"__finish_reason__": fr}), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put({"__error__": str(e)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

    loop.run_in_executor(None, worker)

    stripper = ThinkStripper(
        enabled=_strip_thinking_enabled(cur),
        starts_in_think=starts_in_think,
    )
    tool_parser = ToolCallParser(enabled=tools_active)
    finish_reason = "stop"
    token_count = 0
    text_block_idx = 0
    tool_blocks: list[dict] = []   # completed tool calls to emit at end

    # Buffer for bare-JSON tool detection (same pattern as _oai_stream_chat).
    rj_buf: list[str] = []
    rj_streaming = not tools_active
    RJ_BUF_MAX = 4096

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, dict) and "__finish_reason__" in item:
            finish_reason = item["__finish_reason__"]
            continue
        if isinstance(item, dict) and "__error__" in item:
            log.warning("anth stream error: %s", item["__error__"])
            break
        token_count += 1
        visible = stripper.feed(item)
        if not visible:
            continue
        content, new_tools = tool_parser.feed(visible)
        if new_tools and not rj_streaming:
            rj_streaming = True
            if rj_buf:
                txt = "".join(rj_buf); rj_buf.clear()
                yield sse("content_block_delta", {
                    "type": "content_block_delta", "index": text_block_idx,
                    "delta": {"type": "text_delta", "text": txt},
                })
        if content:
            if rj_streaming:
                yield sse("content_block_delta", {
                    "type": "content_block_delta", "index": text_block_idx,
                    "delta": {"type": "text_delta", "text": content},
                })
            else:
                combined = "".join(rj_buf) + content
                fv = combined.lstrip()
                if fv and fv[0] != "{":
                    rj_streaming = True
                    yield sse("content_block_delta", {
                        "type": "content_block_delta", "index": text_block_idx,
                        "delta": {"type": "text_delta", "text": combined},
                    })
                    rj_buf.clear()
                else:
                    rj_buf.append(content)
                    if len("".join(rj_buf)) > RJ_BUF_MAX:
                        rj_streaming = True
                        combined = "".join(rj_buf); rj_buf.clear()
                        yield sse("content_block_delta", {
                            "type": "content_block_delta", "index": text_block_idx,
                            "delta": {"type": "text_delta", "text": combined},
                        })
        tool_blocks.extend(new_tools)

    # Flush remaining
    tail_str = stripper.flush()
    if tail_str:
        ct, nt = tool_parser.feed(tail_str)
        if ct:
            if rj_streaming:
                yield sse("content_block_delta", {
                    "type": "content_block_delta", "index": text_block_idx,
                    "delta": {"type": "text_delta", "text": ct},
                })
            else:
                rj_buf.append(ct)
        tool_blocks.extend(nt)
    pt, trailing = tool_parser.flush()
    if pt:
        if rj_streaming:
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": text_block_idx,
                "delta": {"type": "text_delta", "text": pt},
            })
        else:
            rj_buf.append(pt)
    tool_blocks.extend(trailing)

    # Bare-JSON fallback.
    if rj_buf and not rj_streaming and not tool_blocks:
        full = "".join(rj_buf); rj_buf.clear()
        fb = ToolCallParser.try_extract_raw_json(full)
        if fb:
            tool_blocks.extend(fb)
        else:
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": text_block_idx,
                "delta": {"type": "text_delta", "text": full},
            })

    # Close text block.
    yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_idx})

    # Emit tool_use blocks (Anthropic format).
    for i, tc in enumerate(tool_blocks, start=text_block_idx + 1):
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments", "{}"))
        except Exception:
            inp = {}
        yield sse("content_block_start", {
            "type": "content_block_start", "index": i,
            "content_block": {
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:20]}"),
                "name": fn.get("name", ""),
                "input": {},
            },
        })
        yield sse("content_block_delta", {
            "type": "content_block_delta", "index": i,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(inp)},
        })
        yield sse("content_block_stop", {"type": "content_block_stop", "index": i})

    stop_reason = _oai_finish_to_anth_stop(
        "tool_calls" if tool_blocks else finish_reason
    )
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": token_count},
    })
    yield sse("message_stop", {"type": "message_stop"})

    cur.generations += 1
    cur.total_tokens += token_count
    cur.last_used = time.time()


# ---- Embeddings API (/v1/embeddings) --------------------------------------
# Serves text-embedding models loaded through mlx-lm (BGE, nomic-embed, etc.)
# using a best-effort strategy:
#   1. model.encode() — BERT/encoder models loaded via mlx-lm
#   2. mean-pool of embed_tokens — generative LLMs (less accurate but functional)
# Normalises all embeddings to unit length (L2) before returning.


class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Any  # str | list[str] | list[int] (token IDs — not supported, ignored)
    encoding_format: str = "float"   # "float" | "base64"
    dimensions: Optional[int] = None  # truncate if set
    model_config = {"extra": "ignore"}


def _embed_texts(cur: LoadedModel, texts: list[str], encoding_format: str = "float") -> list:
    """Compute normalised embeddings for a list of texts.

    Returns a list of embedding vectors (float lists) or base64 strings.
    Raises ValueError if the model architecture is not supported.
    """
    import mlx.core as mx

    model = cur.model
    tokenizer = cur.tokenizer
    results = []

    for text in texts:
        # Tokenise — use padding/truncation when available.
        try:
            toks = tokenizer(
                text, return_tensors="mlx",
                padding=True, truncation=True, max_length=512,
            )
        except Exception:
            toks = tokenizer(text, return_tensors="mlx")

        input_ids = toks["input_ids"]

        # Strategy 1: dedicated encode() method (some mlx-lm encoder models).
        if hasattr(model, "encode"):
            emb = model.encode(input_ids)
            if emb.ndim > 1:
                emb = emb[0]
        else:
            # Strategy 2: mean-pool the token embedding table (works for any
            # generative LLM — less precise but gives useful dense vectors).
            embed_fn = (
                getattr(getattr(model, "model", None), "embed_tokens", None)
                or getattr(model, "embed_tokens", None)
                or getattr(getattr(model, "model", None), "embed", None)
            )
            if embed_fn is None:
                raise ValueError(
                    "This model does not expose an embedding layer. "
                    "Load a dedicated embedding model (BGE, nomic-embed, etc.)."
                )
            hidden = embed_fn(input_ids)   # (1, seq_len, dim)
            mx.eval(hidden)
            mask = toks.get("attention_mask")
            if mask is not None:
                m_f = mask.astype(mx.float32)[:, :, None]
                emb = (hidden * m_f).sum(axis=1) / mx.clip(m_f.sum(axis=1), 1e-9, None)
            else:
                emb = hidden.mean(axis=1)
            emb = emb[0]  # (dim,)

        # L2 normalise.
        mx.eval(emb)
        norm = mx.sqrt((emb * emb).sum())
        emb = emb / (norm + 1e-8)
        mx.eval(emb)

        vec = emb.tolist()
        if encoding_format == "base64":
            import base64
            import struct
            raw = struct.pack(f"{len(vec)}f", *vec)
            results.append(base64.b64encode(raw).decode())
        else:
            results.append(vec)

    return results


@app.post("/v1/embeddings")
async def v1_embeddings(req: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint.

    Works best with a dedicated embedding model (BGE, nomic-embed-text, etc.)
    loaded through mlx-lm.  Falls back to mean-pooling the token embedding
    table for generative LLMs, which provides useful-but-not-fine-tuned vectors.
    """
    cur = _resolve_model(req.model)
    if not cur:
        return _oai_error(503, "no_model_loaded", "No model loaded. Load one from the dashboard.")

    # Normalise input to a list of strings.
    inp = req.input
    if isinstance(inp, str):
        texts = [inp]
    elif isinstance(inp, list) and inp and isinstance(inp[0], int):
        # Token-ID lists: decode back to string (best-effort).
        try:
            texts = [cur.tokenizer.decode(inp)]
        except Exception:
            return _oai_error(400, "unsupported_input", "Token-ID input lists are not supported; send text strings.")
    elif isinstance(inp, list):
        texts = [str(t) for t in inp]
    else:
        return _oai_error(400, "invalid_input", "input must be a string or list of strings.")

    try:
        vecs = await asyncio.to_thread(_embed_texts, cur, texts, req.encoding_format)
    except ValueError as e:
        return _oai_error(422, "embedding_unsupported", str(e))
    except Exception as e:
        log.exception("embeddings failed")
        return _oai_error(500, "embedding_error", f"Embedding failed: {e}")

    saved = settings.get_model(cur.name)
    model_id = req.model or saved.get("alias") or cur.name
    data = [
        {"object": "embedding", "index": i, "embedding": v}
        for i, v in enumerate(vecs)
    ]
    total_tokens = sum(len(cur.tokenizer.encode(t)) for t in texts)
    return {
        "object": "list",
        "data": data,
        "model": model_id,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


# ---- Rerank API (/v1/rerank) -----------------------------------------------
# Cohere/Jina-compatible reranking endpoint using embedding cosine similarity.


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: list[Any]  # list[str] or list[{"text": str}]
    top_n: Optional[int] = None
    return_documents: bool = True
    model_config = {"extra": "ignore"}


@app.post("/v1/rerank")
async def v1_rerank(req: RerankRequest):
    """Cohere/Jina-compatible reranking via embedding cosine similarity.

    Works with any loaded model. Uses the same _embed_texts backend as
    /v1/embeddings.
    """
    cur = _resolve_model(req.model)
    if not cur:
        return _oai_error(503, "no_model_loaded", "No model loaded.")

    # Normalise documents to strings.
    docs = []
    for d in req.documents:
        if isinstance(d, str):
            docs.append(d)
        elif isinstance(d, dict):
            docs.append(d.get("text", str(d)))
        else:
            docs.append(str(d))

    # Embed query + all docs together.
    try:
        all_texts = [req.query] + docs
        vecs = await asyncio.to_thread(_embed_texts, cur, all_texts, "float")
    except ValueError as e:
        return _oai_error(422, "embedding_unsupported", str(e))
    except Exception as e:
        log.exception("rerank embed failed")
        return _oai_error(500, "rerank_error", str(e))

    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)

    q_vec = vecs[0]
    scored = [(i, cosine(q_vec, vecs[i + 1])) for i in range(len(docs))]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_n = req.top_n or len(docs)
    results = []
    for rank, (idx, score) in enumerate(scored[:top_n]):
        r: dict[str, Any] = {"index": idx, "relevance_score": score}
        if req.return_documents:
            r["document"] = {"text": docs[idx]}
        results.append(r)

    saved = settings.get_model(cur.name)
    model_id = req.model or saved.get("alias") or cur.name
    return {
        "id": f"rerank-{uuid.uuid4().hex[:16]}",
        "model": model_id,
        "results": results,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


# ---- Benchmark API (/api/benchmark) ----------------------------------------
# Run N timed inference passes and return throughput stats via SSE so the UI
# can show a live progress bar + per-run timings.


class BenchmarkRequest(BaseModel):
    prompt: str = "Explain in detail how Apple Silicon M-series chips achieve high efficiency."
    max_tokens: int = Field(default=150, ge=10, le=2048)
    runs: int = Field(default=3, ge=1, le=10)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)


@app.post("/api/benchmark")
async def api_benchmark(req: BenchmarkRequest):
    """Run N timed generations and stream per-run results + a summary."""
    cur = engine.current
    if not cur:
        raise HTTPException(status_code=409, detail="No model loaded.")
    return StreamingResponse(
        _benchmark_stream(cur, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _benchmark_stream(cur: LoadedModel, req: BenchmarkRequest) -> AsyncIterator[bytes]:
    """Stream benchmark progress events then a summary."""
    saved = settings.get_model(cur.name)
    sys_prompt = saved.get("system")
    rendered = _render_prompt(cur.tokenizer, req.prompt, sys_prompt)
    gen_req = GenerateRequest(
        prompt=rendered,
        max_tokens=req.max_tokens,
        temperature=req.temperature if req.temperature is not None else 0.0,
        top_p=saved.get("top_p", 0.95),
        stream=False,
    )

    yield f"event: start\ndata: {json.dumps({'runs': req.runs, 'max_tokens': req.max_tokens})}\n\n".encode()

    ttfts: list[float] = []
    tpss: list[float] = []
    token_counts: list[int] = []

    for i in range(req.runs):
        yield f"event: run_start\ndata: {json.dumps({'run': i + 1, 'of': req.runs})}\n\n".encode()

        t_wall_start = time.time()
        try:
            _text, tokens, _fr = await asyncio.to_thread(
                _generate_blocking, cur, rendered, gen_req, False,
            )
            ttft = cur.last_ttft
            tps = cur.last_tps
        except Exception as e:
            yield f"event: run_error\ndata: {json.dumps({'run': i + 1, 'error': str(e)})}\n\n".encode()
            continue

        wall = time.time() - t_wall_start
        ttfts.append(ttft or 0.0)
        tpss.append(tps or 0.0)
        token_counts.append(tokens)
        yield f"event: run_done\ndata: {json.dumps({'run': i + 1, 'tokens': tokens, 'wall_s': round(wall, 3), 'ttft_ms': round((ttft or 0) * 1000, 1), 'tps': round(tps or 0, 1)})}\n\n".encode()

    if ttfts:
        summary = {
            "runs": len(ttfts),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts) * 1000, 1),
            "min_ttft_ms": round(min(ttfts) * 1000, 1),
            "max_ttft_ms": round(max(ttfts) * 1000, 1),
            "avg_tps": round(sum(tpss) / len(tpss), 1),
            "max_tps": round(max(tpss), 1),
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
        }
    else:
        summary = {"runs": 0, "error": "all runs failed"}
    yield f"event: summary\ndata: {json.dumps(summary)}\n\n".encode()

    cur.last_used = time.time()


# ---- static dashboard ----------------------------------------------------


@app.middleware("http")
async def _request_logger(request, call_next):
    """Log incoming /v1/ requests and enforce optional API-key auth.

    API-key gate: if MLXR_API_KEY is set in the environment, every /v1/ request
    must carry a matching ``Authorization: Bearer <key>`` header (or an
    ``x-api-key: <key>`` header for Anthropic-SDK compatibility). Dashboard
    routes (/api/, static assets) are never gated.
    """
    path = request.url.path

    # ── API-key check ────────────────────────────────────────────────────────
    if _API_KEY and path.startswith("/v1/"):
        auth_header = request.headers.get("authorization", "")
        xapi = request.headers.get("x-api-key", "")
        provided = ""
        if auth_header.lower().startswith("bearer "):
            provided = auth_header[7:].strip()
        elif xapi:
            provided = xapi.strip()
        if provided != _API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": {
                    "message": "Invalid API key. Set MLXR_API_KEY on the server to the same value as your client's key.",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }},
            )

    if path.startswith("/v1/"):
        body = await request.body()
        log.info("v1 request: %s %s body=%r", request.method, path, body[:1000] if body else b"")
        # Stash the body so Starlette can re-read it (body stream is consumed).
        from starlette.datastructures import Headers
        from starlette.requests import Request as _Req
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        request = _Req(request.scope, receive)
    response = await call_next(request)
    if path.startswith("/v1/") and response.status_code >= 400:
        log.warning("v1 response: %s %s → HTTP %d", request.method, path, response.status_code)
    # Disable caching for dashboard assets.
    if not path.startswith("/api") and not path.startswith("/v1"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@app.get("/api/kvcache/stats")
def api_kvcache_stats() -> dict:
    """Return KV-cache tier statistics for the dashboard."""
    return kvc.stats()


class KVClearRequest(BaseModel):
    model: Optional[str] = None   # if set, clear only this model's cache


@app.post("/api/kvcache/clear")
async def api_kvcache_clear(req: KVClearRequest) -> dict:
    """Clear KV-cache entries from RAM and disk."""
    result = await asyncio.to_thread(kvc.clear, req.model)
    log.info("kvc: cleared %s — %s", req.model or "all", result)
    return {"ok": True, **result}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
