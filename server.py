"""MLXr — management server for Apple MLX model engine.

Exposes a JSON API and serves a dashboard for loading MLX language models,
running streaming inference, and inspecting host + engine state.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
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


class Engine:
    """Thread-safe holder for the currently-loaded MLX model."""

    def __init__(self) -> None:
        self._lock = Lock()
        # Serializes all MLX inference calls. MLX / Apple's Metal driver can
        # crash (SIGSEGV in AGXMetalG17X) when two threads run eval() on the
        # same model concurrently — which happens as soon as a second HTTP
        # request arrives mid-generation. Keep inference strictly one-at-a-time.
        self._gen_lock = Lock()
        self._current: Optional[LoadedModel] = None
        self._loading: Optional[str] = None

    @property
    def gen_lock(self) -> Lock:
        return self._gen_lock

    @property
    def current(self) -> Optional[LoadedModel]:
        return self._current

    @property
    def loading(self) -> Optional[str]:
        return self._loading

    def load(self, name: str) -> LoadedModel:
        # mlx_lm is imported lazily so the server starts even if MLX is missing.
        from mlx_lm import load as mlx_load

        with self._lock:
            if self._loading:
                raise RuntimeError(f"Another load is in progress: {self._loading}")
            if self._current and self._current.name == name:
                return self._current
            self._loading = name

        try:
            log.info("Loading model %s", name)
            t0 = time.time()
            model, tokenizer = mlx_load(name)
            log.info("Loaded %s in %.1fs", name, time.time() - t0)
            # Detect context window; honour per-model override from settings.
            auto_ctx = _detect_context_length(tokenizer, name)
            saved_ctx = settings.get_model(name).get("context_length")
            ctx = int(saved_ctx) if saved_ctx else auto_ctx
            log.info("Context length for %s: %d%s", name, ctx,
                     " (overridden in settings)" if saved_ctx else " (auto-detected)")
            loaded = LoadedModel(
                name=name, loaded_at=time.time(),
                model=model, tokenizer=tokenizer, context_length=ctx,
            )
            with self._lock:
                self._current = loaded
                self._loading = None
            return loaded
        except Exception:
            with self._lock:
                self._loading = None
            raise

    def unload(self) -> bool:
        with self._lock:
            if not self._current:
                return False
            self._current = None
        # Let MLX reclaim buffers.
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except Exception:
            pass
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


engine = Engine()
hf = HFManager()
settings = Settings(SETTINGS_PATH)
app = FastAPI(title="MLXr", version="0.1.0")


@app.on_event("startup")
async def _autoload_on_start() -> None:
    name = settings.autoload_name()
    if not name:
        return
    log.info("Autoloading %s per settings", name)

    def _go():
        try:
            engine.load(name)
        except Exception as e:
            log.warning("autoload failed: %s", e)

    threading.Thread(target=_go, daemon=True).start()


# ---- models --------------------------------------------------------------


class LoadRequest(BaseModel):
    name: str = Field(..., description="HuggingFace repo id or local path of an MLX model.")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = Field(default=None, ge=1, le=131072)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
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


def _model_state() -> dict:
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
        # DeepSeek
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
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
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call|>", "<|/tool_call|>"),
        ("<function_call>", "</function_call>"),
        ("<tool_call_begin>", "<tool_call_end>"),
        ("<|tool_call_begin|>", "<|tool_call_end|>"),
        ("[TOOL_CALLS]", "[/TOOL_CALLS]"),
    )

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
    out: dict[str, Any] = {}
    for mod in ("mlx", "mlx_lm", "huggingface_hub", "transformers"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:
            out[mod] = f"not installed ({e.__class__.__name__})"
    out["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    out["python_too_old"] = sys.version_info < MIN_PYTHON
    out["python_min"] = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    return out


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


@app.post("/api/models/unload")
def api_unload() -> dict:
    return {"ok": engine.unload()}


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
) -> tuple[str, int, str]:
    """Run blocking generation. Returns (text, token_count, finish_reason)."""
    from mlx_lm import stream_generate

    parts: list[str] = []
    token_count = 0
    finish_reason = "stop"

    try:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
        iterator = stream_generate(
            cur.model, cur.tokenizer, prompt=rendered,
            max_tokens=req.max_tokens, sampler=sampler,
        )
    except Exception:
        iterator = stream_generate(
            cur.model, cur.tokenizer, prompt=rendered,
            max_tokens=req.max_tokens, temp=req.temperature,
        )

    t0 = time.time()
    with engine.gen_lock:
        waited = time.time() - t0
        if waited > 0.1:
            log.info("MLX (blocking) queued for %.1fs before starting", waited)
        for c in iterator:
            piece = getattr(c, "text", c) if not isinstance(c, str) else c
            if piece:
                parts.append(piece)
            token_count += 1
            fr = getattr(c, "finish_reason", None)
            if fr:
                finish_reason = fr

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
    # Tool use. `tools` follows OpenAI's function-tool schema:
    #   [{"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON schema>}}]
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None  # "auto" | "none" | "required" | {type, function}

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
    cur = engine.current
    data = []
    if cur:
        data.append({
            "id": cur.name,
            "object": "model",
            "created": int(cur.loaded_at),
            "owned_by": "mlxr",
            "context_length": cur.context_length,  # non-standard but useful for clients
        })
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: OAIChatRequest):
    cur = engine.current
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

    # Inject saved system prompt if the client didn't send one.
    if not any(m["role"] == "system" for m in messages) and saved.get("system"):
        messages.insert(0, {"role": "system", "content": saved["system"]})

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

    prompt = _render_chat(
        cur.tokenizer, messages, tools=tools_for_template, enable_thinking=enable_thinking,
    )
    starts_in_think = _prompt_starts_in_think(prompt)
    model_id = req.model or cur.name
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
        stream=req.stream,
    )

    if req.stream:
        return StreamingResponse(
            _oai_stream_chat(
                cur, prompt, gen_req, model_id,
                tools_active=tools_active, starts_in_think=starts_in_think,
                entry_id=entry_id, include_usage=req.include_usage(),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, tokens, gen_finish_reason = await asyncio.to_thread(
        _generate_blocking, cur, prompt, gen_req, starts_in_think,
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
            try:
                try:
                    from mlx_lm.sample_utils import make_sampler

                    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
                    iterator = stream_generate(
                        cur.model, cur.tokenizer, prompt=prompt,
                        max_tokens=req.max_tokens, sampler=sampler,
                    )
                except Exception:
                    iterator = stream_generate(
                        cur.model, cur.tokenizer, prompt=prompt,
                        max_tokens=req.max_tokens, temp=req.temperature,
                    )
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
        # Capture for diagnostics, up to cap.
        if sum(len(s) for s in raw_output_capture) < raw_output_cap:
            raw_output_capture.append(item)
        visible = stripper.feed(item)
        if not visible:
            continue
        content, new_tools = tool_parser.feed(visible)
        if content:
            yield chunk({"content": content})
        for t in new_tools:
            yield emit_tool(t)
            tools_emitted += 1

    # Flush both stages.
    stripper_tail = stripper.flush()
    if stripper_tail:
        content, new_tools = tool_parser.feed(stripper_tail)
        if content:
            yield chunk({"content": content})
        for t in new_tools:
            yield emit_tool(t)
            tools_emitted += 1
    parser_tail, trailing_tools = tool_parser.flush()
    if parser_tail:
        yield chunk({"content": parser_tail})
    for t in trailing_tools:
        yield emit_tool(t)
        tools_emitted += 1

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

    raw_preview = "".join(raw_output_capture)[:800]
    log.info(
        "chat: id=%s stream done tokens=%d tool_calls=%d finish=%s raw_preview=%r",
        completion_id, token_count, tools_emitted, finish_reason, raw_preview,
    )
    _update_chat(completion_id, {
        "output_preview": raw_preview,
        "tokens": token_count,
        "tool_calls_emitted": tools_emitted,
        "finish_reason": finish_reason,
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


# ---- static dashboard ----------------------------------------------------


@app.middleware("http")
async def _request_logger(request, call_next):
    """Log incoming /v1/ requests and any error responses so client issues are
    visible in the server log without needing a separate proxy."""
    path = request.url.path
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


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
