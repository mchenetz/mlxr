"""tool_mappings/ai_creator.py — AI-assisted mapping creation.

Flow
----
1. run_probe(model_name, generate_fn)
   Forces the loaded model to emit a tool call for a known test tool and
   returns the raw unstripped model output.

2. derive_mapping(model_name, raw_output, generate_fn, ...)
   a. Heuristic fast-path — scan raw_output for known tag-pairs / key signals.
      If a builtin match is found, clone it with a model-specific glob.
   b. AI extraction path — feed raw output + probe tool schema back to the
      model and ask it to produce a MappingDefinition JSON.  Retry × 2.

Both steps require a *generate_fn* callable with signature:
    generate_fn(messages: list[dict], tools: list[dict] | None,
                tool_choice: str | None, max_tokens: int) -> str
That function runs a blocking inference and returns the raw text.
The server wires this up as a closure inside the /api/tool-mappings/probe handler.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Callable, Optional

from .schema import BareJsonSignal, BodyFormat, Detection, MappingDefinition, OutputConfig, TagPair

log = logging.getLogger("mlxr")

# ── Probe tool ────────────────────────────────────────────────────────────────

PROBE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
}

PROBE_MESSAGES: list[dict] = [
    {
        "role": "user",
        "content": (
            "What is the weather in San Francisco in celsius? "
            "You MUST call the get_weather tool."
        ),
    }
]

# ── AI extraction prompt ──────────────────────────────────────────────────────

_EXTRACTION_PROMPT_TEMPLATE = """\
You are a tool-call format analyser. Below is the raw text output produced by a \
language model when it was required to call a tool named "get_weather" with \
arguments {{"location": "San Francisco", "unit": "celsius"}}.

RAW MODEL OUTPUT:
<output>
{raw_output}
</output>

The tool schema was:
<tool_schema>
{tool_schema}
</tool_schema>

Analyse the output and produce a single JSON object describing the \
MappingDefinition for this model's tool-call format. Output ONLY the JSON — \
no markdown fences, no explanation, no extra text.

The JSON must follow this schema exactly:
{{
  "id": "<short-slug>",
  "name": "<human name>",
  "version": "1.0.0",
  "description": "<brief description>",
  "match_globs": ["*<model-family>*"],
  "detection": {{
    "strategy": "<tag_pair|bare_json|tag_pair_and_bare_json>",
    "tag_pairs": [{{"open": "<open>", "close": "<close>"}}],
    "bare_json_signal": {{"required_keys": ["name"], "argument_keys": ["arguments","parameters"]}}
  }},
  "body_format": {{
    "type": "<json_object|deepseek_sep|xml_function_tag|fenced_json>",
    "json_name_key": "name",
    "json_args_keys": ["arguments","parameters"]
  }},
  "output": {{"arguments_encoding": "json_string", "id_prefix": "call_"}}
}}

Rules:
- If there are wrapper tags, use strategy "tag_pair" and list them in tag_pairs.
- If the tool call appears as a raw JSON object with no wrapper tags, use \
strategy "bare_json" and omit tag_pairs.
- If there are wrapper tags AND the model might also emit bare JSON, use \
"tag_pair_and_bare_json".
- For DeepSeek-style bodies (type<sep>name\\n```json\\n{{}}\\n```), use \
body_format.type "deepseek_sep" and set deepseek_separator.
- For XML <function=NAME><parameter=KEY> format, use body_format.type \
"xml_function_tag".
- match_globs should capture the model family based on "{model_slug}".
"""


# ── Public API ────────────────────────────────────────────────────────────────

GenerateFn = Callable[[list[dict], Optional[list[dict]], Optional[str], int], str]


def run_probe(model_name: str, generate_fn: GenerateFn) -> str:
    """Send a forced tool-call probe to the loaded model; return raw output.

    Args:
        model_name:   HF repo-id of the loaded model (for logging).
        generate_fn:  Blocking callable — see module docstring for signature.

    Returns:
        The raw, unstripped model output text.
    """
    log.info("tool_mappings: running probe for model=%s", model_name)
    try:
        raw = generate_fn(
            PROBE_MESSAGES,
            [PROBE_TOOL],
            "required",
            512,
        )
    except Exception as exc:
        raise RuntimeError(f"Probe generation failed: {exc}") from exc

    log.info("tool_mappings: probe raw output (%d chars): %r", len(raw), raw[:200])
    return raw


def derive_mapping(
    model_name: str,
    raw_output: str,
    generate_fn: GenerateFn,
    existing_mapping_ids: list[str] = (),
    builtins: Optional[list[MappingDefinition]] = None,
) -> tuple[MappingDefinition, str]:
    """Analyse *raw_output* and derive a MappingDefinition.

    Returns:
        (definition, confidence) where confidence is "heuristic" or "ai".

    Raises:
        RuntimeError if both heuristic and AI paths fail.
    """
    # ── Step 1: heuristic fast-path ────────────────────────────────────────
    if builtins is None:
        from .registry import BUILTIN_MAPPINGS_PATH, _load_file
        builtins = _load_file(BUILTIN_MAPPINGS_PATH)

    heuristic = _heuristic_detect(model_name, raw_output, builtins)
    if heuristic is not None:
        log.info("tool_mappings: heuristic match → id=%s", heuristic.id)
        return heuristic, "heuristic"

    # ── Step 2: AI extraction path ─────────────────────────────────────────
    log.info("tool_mappings: no heuristic match — trying AI extraction")
    model_slug = _model_slug(model_name)
    prompt_text = _EXTRACTION_PROMPT_TEMPLATE.format(
        raw_output=raw_output[:2000],
        tool_schema=json.dumps(PROBE_TOOL, indent=2),
        model_slug=model_slug,
    )

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            ai_text = generate_fn(
                [{"role": "user", "content": prompt_text}],
                None,
                None,
                1024,
            )
            defn = _parse_ai_response(ai_text, model_name, existing_mapping_ids)
            log.info("tool_mappings: AI extraction success on attempt %d", attempt + 1)
            return defn, "ai"
        except Exception as exc:
            last_exc = exc
            log.warning("tool_mappings: AI extraction attempt %d failed: %s", attempt + 1, exc)

    raise RuntimeError(
        f"Could not derive a mapping for {model_name!r}. "
        f"Last error: {last_exc}. "
        "You can define one manually in the Tool-Call Mappings panel."
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _heuristic_detect(
    model_name: str,
    raw_output: str,
    builtins: list[MappingDefinition],
) -> Optional[MappingDefinition]:
    """Check raw_output against known tag-pairs and bare-JSON signals.

    When multiple builtins use the same wrapper tag (e.g. qwen3-hermes-json
    and qwen3-xml both wrap with <tool_call>…</tool_call>), inspect the body
    between the tags to pick the one whose body_format actually matches.

    Returns a cloned definition with model-specific match_globs, or None.
    """
    # 1. Scan for tag pairs — collect *all* matching builtins, then pick best
    matches: list[tuple[MappingDefinition, str, str]] = []  # (defn, open, body)
    for defn in builtins:
        for tp in defn.detection.tag_pairs:
            i = raw_output.find(tp.open)
            if i < 0:
                continue
            body_start = i + len(tp.open)
            j = raw_output.find(tp.close, body_start)
            body = raw_output[body_start:j].strip() if j >= 0 else raw_output[body_start:].strip()
            matches.append((defn, tp.open, body))
            break  # one tag-pair per builtin is enough

    if matches:
        body = matches[0][2]
        body_is_xml = "<function" in body and "<parameter" in body
        body_is_json = body.startswith("{") or body.startswith("[")

        # Prefer a builtin whose body_format matches what we actually see
        if body_is_xml:
            for defn, tag, _ in matches:
                if defn.body_format.type == "xml_function_tag":
                    log.info("tool_mappings: heuristic found tag %r with XML body (builtin=%s)",
                             tag, defn.id)
                    return _clone_for_model(defn, model_name)
        if body_is_json:
            for defn, tag, _ in matches:
                if defn.body_format.type in ("json_object", "fenced_json"):
                    log.info("tool_mappings: heuristic found tag %r with JSON body (builtin=%s)",
                             tag, defn.id)
                    return _clone_for_model(defn, model_name)

        # Fall back to the first match (legacy behaviour)
        defn, tag, _ = matches[0]
        log.info("tool_mappings: heuristic found tag %r (builtin=%s, body type unclear)",
                 tag, defn.id)
        return _clone_for_model(defn, model_name)

    # 2. Bare-JSON signal
    stripped = raw_output.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and isinstance(data.get("name"), str):
                has_args = "arguments" in data or "parameters" in data
                if has_args:
                    # Find the bare_json builtin
                    for defn in builtins:
                        if defn.detection.strategy in ("bare_json", "tag_pair_and_bare_json"):
                            return _clone_for_model(defn, model_name)
        except Exception:
            pass

    return None


def _clone_for_model(
    original: MappingDefinition,
    model_name: str,
) -> MappingDefinition:
    """Clone a builtin definition with a model-specific glob and new id."""
    slug = _model_slug(model_name)
    new_id = f"user-{slug}"
    new_glob = f"*{slug}*"

    # Deep-clone via dict round-trip
    d = original.to_dict()
    d["id"] = new_id
    d["name"] = f"{original.name} ({slug})"
    d["match_globs"] = [new_glob]
    d["description"] = (
        f"Auto-detected from probe output. Based on builtin '{original.id}'."
    )
    cloned = MappingDefinition.from_dict(d)
    cloned.is_builtin = False
    return cloned


def _parse_ai_response(
    ai_text: str,
    model_name: str,
    existing_ids: list[str],
) -> MappingDefinition:
    """Parse the model's JSON response into a validated MappingDefinition."""
    # Strip any accidental markdown fences
    text = ai_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = []
        for line in lines[1:]:
            if line.strip().startswith("```"):
                break
            inner.append(line)
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI response is not valid JSON: {exc}\nResponse: {text[:500]}") from exc

    # Ensure a unique id
    base_id = data.get("id") or f"user-{_model_slug(model_name)}"
    uid = base_id
    counter = 1
    while uid in existing_ids:
        uid = f"{base_id}-{counter}"
        counter += 1
    data["id"] = uid

    defn = MappingDefinition.from_dict(data)
    errs = defn.validate()
    if errs:
        raise ValueError(f"AI-generated mapping has validation errors: {errs}")
    return defn


def _model_slug(model_name: str) -> str:
    """Produce a short lowercase slug from a HF repo-id."""
    # Take the repo name part after the last '/'
    part = model_name.rsplit("/", 1)[-1].lower()
    # Keep only alphanumeric and hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", part).strip("-")
    # Trim to a reasonable length
    return slug[:40] or "model"
