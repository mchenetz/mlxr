"""tool_mappings/converter.py — MappedToolCallParser

Drop-in replacement for the legacy ToolCallParser in server.py.

When a MappingDefinition is provided the parser uses its detection/body_format
config to parse tool calls.  When mapping=None it falls back to the legacy
ToolCallParser so all existing behaviour is preserved with zero regression risk.

Public API (identical to ToolCallParser):
    feed(chunk: str) -> tuple[str, list[dict]]
    flush()          -> tuple[str, list[dict]]
    try_extract_raw_json(content, mapping=None) -> list[dict]   (classmethod)
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .schema import MappingDefinition

log = logging.getLogger("mlxr")


class MappedToolCallParser:
    """Streaming tool-call extractor driven by a MappingDefinition.

    When *mapping* is None this class silently delegates every call to an
    internal instance of the legacy ToolCallParser, preserving all existing
    behaviour.
    """

    def __init__(self, enabled: bool = True, mapping: Optional["MappingDefinition"] = None) -> None:
        self.enabled = enabled
        self._mapping = mapping

        if mapping is None:
            # Delegate to legacy parser — zero regression.
            from server import ToolCallParser as _Legacy  # type: ignore
            self._legacy: Optional[Any] = _Legacy(enabled=enabled)
        else:
            self._legacy = None

        # Streaming state (only used when mapping is set)
        self._buffer = ""
        self._tool_buffer = ""
        self._in_tool = False
        self._active_close: Optional[str] = None

        # Pre-compute lengths for efficient lookahead
        if mapping and mapping.detection.tag_pairs:
            self._max_open = max(len(tp.open) for tp in mapping.detection.tag_pairs)
            self._max_close = max(len(tp.close) for tp in mapping.detection.tag_pairs)
        else:
            self._max_open = 0
            self._max_close = 0

        # Compile XML regexes whenever patterns are specified (regardless of body_format.type
        # so that json_object mappings can still fall back to XML parsing).
        _default_fn = r"<function\s*=\s*[\"']?(?P<name>[^\s\"'>]+)[\"']?\s*>"
        _default_pm = r"<parameter\s*=\s*[\"']?(?P<key>[^\s\"'>]+)[\"']?\s*>"
        if mapping and mapping.body_format.type == "xml_function_tag":
            fn_pat = mapping.body_format.xml_function_open_re or _default_fn
            pm_pat = mapping.body_format.xml_parameter_open_re or _default_pm
            self._fn_re = re.compile(fn_pat)
            self._pm_re = re.compile(pm_pat)
        elif mapping and (mapping.body_format.xml_function_open_re or mapping.body_format.xml_parameter_open_re):
            # Custom patterns provided even for non-xml type — compile them for fallback use.
            self._fn_re = re.compile(mapping.body_format.xml_function_open_re or _default_fn)
            self._pm_re = re.compile(mapping.body_format.xml_parameter_open_re or _default_pm)
        else:
            # Always have default XML regexes available as last-resort fallback.
            self._fn_re = re.compile(_default_fn)
            self._pm_re = re.compile(_default_pm)

    # ── Public streaming API ──────────────────────────────────────────────────

    def feed(self, chunk: str) -> tuple[str, list[dict]]:
        """Feed a text chunk; return (visible_content, completed_tool_calls)."""
        if self._legacy is not None:
            return self._legacy.feed(chunk)

        if not self.enabled:
            return (chunk or ""), []
        if not chunk:
            return "", []

        mapping = self._mapping
        assert mapping is not None
        strat = mapping.detection.strategy

        if strat == "bare_json":
            # Bare-JSON models: pass chunks through as visible content so the
            # caller's rj_buf (in the server streaming loop) can accumulate them.
            # Extraction happens in try_extract_raw_json after flush — no internal
            # buffering needed here.
            return chunk, []

        # tag_pair or tag_pair_and_bare_json — streaming tag scanner
        self._buffer += chunk
        content_out: list[str] = []
        tools_out: list[dict] = []

        while self._buffer:
            if self._in_tool and self._active_close:
                idx = self._buffer.find(self._active_close)
                if idx >= 0:
                    self._tool_buffer += self._buffer[:idx]
                    self._buffer = self._buffer[idx + len(self._active_close):]
                    tools_out.extend(self._parse_body(self._tool_buffer))
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
                earliest_open: Optional[str] = None
                earliest_close: Optional[str] = None
                for tp in mapping.detection.tag_pairs:
                    i = self._buffer.find(tp.open)
                    if i >= 0 and (earliest_idx < 0 or i < earliest_idx):
                        earliest_idx = i
                        earliest_open = tp.open
                        earliest_close = tp.close
                if earliest_idx >= 0 and earliest_open:
                    if earliest_idx > 0:
                        content_out.append(self._buffer[:earliest_idx])
                    self._buffer = self._buffer[earliest_idx + len(earliest_open):]
                    self._in_tool = True
                    self._active_close = earliest_close
                    continue
                keep = self._max_open - 1
                if keep > 0 and len(self._buffer) > keep:
                    content_out.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                break

        return "".join(content_out), tools_out

    def flush(self) -> tuple[str, list[dict]]:
        """Flush any remaining buffered content."""
        if self._legacy is not None:
            return self._legacy.flush()

        if not self.enabled:
            tail = self._buffer
            self._buffer = ""
            return tail, []

        mapping = self._mapping
        assert mapping is not None
        strat = mapping.detection.strategy

        if strat == "bare_json":
            tail = self._buffer
            self._buffer = ""
            return tail, []

        # tag_pair / tag_pair_and_bare_json
        if self._in_tool:
            # Mistral-style: close tag never appears; parse whatever we have.
            tools = self._parse_body(self._tool_buffer + self._buffer)
            self._tool_buffer = ""
            self._buffer = ""
            self._in_tool = False
            self._active_close = None
            return "", tools

        tail = self._buffer
        self._buffer = ""
        return tail, []

    # ── Class-method fallback ────────────────────────────────────────────────

    @classmethod
    def try_extract_raw_json(
        cls,
        content: str,
        mapping: Optional["MappingDefinition"] = None,
    ) -> list[dict]:
        """Last-resort bare-JSON extractor.

        When a mapping is provided its bare_json_signal config is used.
        Otherwise falls back to the legacy heuristic (name + arguments/parameters).
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

        if mapping and mapping.detection.bare_json_signal:
            sig = mapping.detection.bare_json_signal
            # All required keys must be present
            if not all(k in data for k in sig.required_keys):
                return []
            # At least one argument key must be present
            has_args = any(k in data for k in sig.argument_keys)
            if not has_args:
                return []
        else:
            # Legacy heuristic
            if not isinstance(data.get("name"), str):
                return []
            if "arguments" not in data and "parameters" not in data:
                return []

        tool = cls._make_tool_from_dict(data, mapping)
        return [tool] if tool else []

    # ── Body parsing ─────────────────────────────────────────────────────────

    def _parse_body(self, text: str) -> list[dict]:
        """Dispatch to the appropriate parser based on body_format.type."""
        mapping = self._mapping
        assert mapping is not None
        fmt = mapping.body_format.type

        text = text.strip()
        if not text:
            return []

        if fmt == "deepseek_sep":
            return self._parse_deepseek(text)

        # Strip fences for fenced_json or any body that opens with ```
        if fmt == "fenced_json" or text.startswith("```"):
            text = self._strip_fence(text, mapping.body_format.fenced_languages)

        # Always try JSON first — even for xml_function_tag mappings, because many
        # Qwen3 models emit JSON inside <tool_call> tags despite the "XML" template.
        try:
            data = json.loads(text)
        except Exception:
            data = None

        if data is not None:
            items = data if isinstance(data, list) else [data]
            results: list[dict] = []
            for entry in items:
                if isinstance(entry, dict):
                    tool = self._make_tool_from_dict(entry, mapping)
                    if tool:
                        results.append(tool)
            if results:
                return results

        # Try XML — covers xml_function_tag format AND json_object fallback for
        # models that switched to XML (e.g. newer Qwen3 fine-tunes).
        xml_tools = self._parse_xml(text)
        if xml_tools:
            return xml_tools

        log.warning("tool_mappings: body parse failed (mapping=%s) — raw: %r",
                    mapping.id, text[:300])
        return []

    def _parse_deepseek(self, text: str) -> list[dict]:
        """Parse a DeepSeek-style body: type<sep>NAME\\n```json\\n{...}\\n```"""
        mapping = self._mapping
        assert mapping is not None
        sep = mapping.body_format.deepseek_separator
        if not sep or sep not in text:
            return []
        rest = text[text.find(sep) + len(sep):]
        nl = rest.find("\n")
        fn_name = rest[:nl].strip() if nl >= 0 else rest.strip()
        fn_body = rest[nl + 1:].strip() if nl >= 0 else ""
        fn_body = self._strip_fence(fn_body, mapping.body_format.fenced_languages)
        if not fn_name:
            return []
        try:
            args = json.loads(fn_body) if fn_body else {}
        except Exception:
            log.warning("tool_mappings: DeepSeek body JSON parse failed: %r", fn_body[:200])
            args = {}
        return [self._make_tool(fn_name, args)]

    def _parse_xml(self, text: str) -> list[dict]:
        """Parse <function=NAME><parameter=KEY>VALUE</parameter></function> XML."""
        fn_re = self._fn_re
        pm_re = self._pm_re
        if not fn_re or not pm_re:
            return []

        results: list[dict] = []
        pos = 0
        while pos < len(text):
            fn_m = fn_re.search(text, pos)
            if not fn_m:
                break
            fn_name = (fn_m.group("name") if "name" in fn_m.groupdict() else fn_m.group(1)).strip()
            body_start = fn_m.end()
            body, body_end = self._find_closing(text, body_start, "<function", "</function>")
            if body is None:
                break
            pos = body_end

            params: dict[str, Any] = {}
            pp = 0
            while pp < len(body):
                pm = pm_re.search(body, pp)
                if not pm:
                    break
                key = (pm.group("key") if "key" in pm.groupdict() else pm.group(1)).strip()
                vs = pm.end()
                raw_val, ve = self._find_closing(body, vs, "<parameter", "</parameter>")
                if raw_val is None:
                    break
                pp = ve
                stripped_v = raw_val.strip()
                if stripped_v and (stripped_v[0] in '{["'
                                   or stripped_v in ("true", "false", "null")
                                   or self._looks_numeric(stripped_v)):
                    try:
                        params[key] = json.loads(stripped_v)
                        continue
                    except Exception:
                        pass
                params[key] = raw_val.lstrip("\n").rstrip("\n")

            if fn_name:
                results.append(self._make_tool(fn_name, params))
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_fence(text: str, languages: list[str]) -> str:
        """Remove markdown code fences if present."""
        t = text.strip()
        if not t.startswith("```"):
            return t
        lines = t.splitlines()
        first_line = lines[0][3:].strip().lower() if lines else ""
        if first_line in [l.lower() for l in languages]:
            inner = lines[1:]
            # Drop trailing ```
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            return "\n".join(inner).strip()
        return t

    @staticmethod
    def _find_closing(text: str, start: int,
                      open_prefix: str, close_tag: str) -> tuple[Optional[str], int]:
        """Depth-tracking close-tag finder (same logic as legacy ToolCallParser)."""
        depth = 1
        pos = start
        cl = len(close_tag)
        while pos < len(text):
            nc = text.find(close_tag, pos)
            no = text.find(open_prefix, pos)
            if nc < 0:
                return None, start
            if no >= 0 and no < nc:
                depth += 1
                pos = no + len(open_prefix)
            else:
                depth -= 1
                if depth == 0:
                    return text[start:nc], nc + cl
                pos = nc + cl
        return None, start

    @staticmethod
    def _looks_numeric(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _make_tool(self, name: str, args: Any) -> dict:
        """Produce a normalised OpenAI tool-call dict."""
        mapping = self._mapping
        prefix = mapping.output.id_prefix if mapping else "call_"
        encoding = mapping.output.arguments_encoding if mapping else "json_string"
        if encoding == "json_string":
            args_val = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        else:
            args_val = args if isinstance(args, dict) else (
                json.loads(args) if isinstance(args, str) else {}
            )
        return {
            "id": f"{prefix}{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {"name": name, "arguments": args_val},
        }

    @classmethod
    def _make_tool_from_dict(
        cls,
        data: dict,
        mapping: Optional["MappingDefinition"],
    ) -> Optional[dict]:
        """Extract name + args from a parsed JSON dict using mapping config."""
        if mapping:
            bf = mapping.body_format
            # Handle nested function key (e.g. OpenAI-style {"function": {...}})
            if bf.json_nested_function_key and bf.json_nested_function_key in data:
                inner = data[bf.json_nested_function_key]
                if isinstance(inner, dict):
                    data = inner
            name = data.get(bf.json_name_key)
            args = None
            for k in bf.json_args_keys:
                if k in data:
                    args = data[k]
                    break
        else:
            # Legacy fallback
            if "function" in data and isinstance(data["function"], dict):
                fn = data["function"]
                name = fn.get("name")
                args = fn.get("arguments") if fn.get("arguments") is not None else fn.get("parameters", {})
            else:
                name = data.get("name")
                args = data.get("arguments") if data.get("arguments") is not None else data.get("parameters", {})

        if not name:
            return None

        prefix = mapping.output.id_prefix if mapping else "call_"
        encoding = mapping.output.arguments_encoding if mapping else "json_string"
        if args is None:
            args = {}
        if encoding == "json_string":
            args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        else:
            args_str = args  # type: ignore[assignment]

        return {
            "id": f"{prefix}{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        }
