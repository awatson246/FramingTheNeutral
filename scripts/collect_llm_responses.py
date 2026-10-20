import argparse
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache, partial
from pathlib import Path
from time import sleep

# Add C:\libs to path for model client libraries installed there
import sys
sys.path.insert(0, "C:\\libs")

import openai
import anthropic
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

from answer_text import AnswerTextError, get_answer_text
from model_registry import MODELS, MODEL_HEADROOM, MODEL_LABELS, MODEL_PROVIDERS

load_dotenv()

# =============================================================================
# Config
# =============================================================================

INSTRUMENTS_FILE  = Path("data/prompts/instruments.json")
PEER_EVAL_FILE    = Path("data/prompts/peer_eval_pairs.json")
V1_RAW_DIR        = Path("data/raw")      # v1 data: never written to
RAW_DIR           = Path("data/raw/v2")   # default; overridden by --raw-dir

# MODELS, MODEL_LABELS, MODEL_PROVIDERS and MODEL_HEADROOM come from
# config/models.yaml via model_registry. Keys must match instruments.json "models".

# Instruments to collect in this run; overridden by instruments.json "active_instruments"
ACTIVE_INSTRUMENTS = ["instrument_1", "instrument_2", "instrument_3", "instrument_4", "instrument_5"]

# Model used for I5 source extraction (always Claude Sonnet, regardless of original model)
EXTRACTOR_MODEL = "claude-sonnet-4-5"

# Answer budgets. Each model's request limit is this plus its reasoning_headroom
# from config/models.yaml, since reasoning tokens count against the same limit.
# Max tokens for open-ended responses (I1, I2)
MAX_TOKENS_OPEN = 1024
# Max tokens for structured JSON responses (I3)
MAX_TOKENS_JSON = 2048

RETRY_ATTEMPTS = 3   # attempts for non-transient errors
RETRY_DELAY    = 5   # seconds between those retries
CALL_DELAY     = 1   # seconds between normal calls

# Rate-limit / overload errors get exponential backoff instead.
# 529 is Anthropic's "overloaded" (its equivalent of 503).
TRANSIENT_STATUS   = {429, 503, 529}
TRANSIENT_ATTEMPTS = 6
BACKOFF_BASE       = 5    # seconds; doubles each attempt, ±20% jitter
BACKOFF_MAX        = 120

REQUEST_TIMEOUT = 600  # seconds; reasoning calls can be slow

# =============================================================================
# Dry-run cost estimation (--dry-run)
# =============================================================================

# USD per 1M tokens, keyed by API model ID (the values in MODELS, plus
# EXTRACTOR_MODEL). Fill in before relying on the cost column; None prints "-".
PRICES = {
    "gpt-6.1-sol":        {"input": None, "output": None},
    "claude-sonnet-5-5":  {"input": None, "output": None},
    "gemini-3.8-flash":   {"input": None, "output": None},
    "deepseek-flash":     {"input": None, "output": None},
    "mistral-medium-3.5": {"input": None, "output": None},
    "grok-4.7":           {"input": None, "output": None},
    "claude-sonnet-4-5":  {"input": None, "output": None},  # EXTRACTOR_MODEL
}

DRY_RUN_RUNS    = 10  # default runs_per_condition for --dry-run
CHARS_PER_TOKEN = 4   # rough heuristic; no tokenizer is used

# =============================================================================
# Client setup
# =============================================================================

# OpenAI-compatible providers. Token parameter per docs/models_available.md:
# OpenAI rejects max_tokens; xAI accepts both, and max_completion_tokens is the
# one defined to include reasoning tokens; DeepSeek and Mistral use max_tokens.
OPENAI_COMPAT = {
    "openai":   {"base_url": None,                          "env": "OPENAI_API_KEY",
                 "token_param": "max_completion_tokens"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "env": "DEEPSEEK_API_KEY",
                 "token_param": "max_tokens"},
    "mistral":  {"base_url": "https://api.mistral.ai/v1",   "env": "MISTRAL_API_KEY",
                 "token_param": "max_tokens"},
    "xai":      {"base_url": "https://api.x.ai/v1",         "env": "XAI_API_KEY",
                 "token_param": "max_completion_tokens"},
}


def _require_key(env: str) -> str:
    key = os.getenv(env)
    if not key:
        raise RuntimeError(f"{env} is not set (check .env)")
    return key


@cache
def get_client(provider: str):
    """
    Create each provider's client on first use, so a missing key only fails
    calls to that provider. SDK-level retries are off; call_with_retry owns them.
    """
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"),
                                   max_retries=0, timeout=REQUEST_TIMEOUT)
    if provider == "google":
        return genai.Client(api_key=_require_key("GOOGLE_API_KEY"),
                            http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000))
    cfg = OPENAI_COMPAT[provider]
    return openai.OpenAI(api_key=_require_key(cfg["env"]), base_url=cfg["base_url"],
                         max_retries=0, timeout=REQUEST_TIMEOUT)

# =============================================================================
# Model callers
# Each accepts (api_id, system_prompt, user_prompt, max_tokens) and returns a
# CallResult. No temperature or reasoning-effort parameters are sent: every
# model runs at its provider defaults.
# =============================================================================

# finish_reason values (lowercased) that mean the output limit was hit
TRUNCATION_REASONS = {"length", "max_tokens"}


# Whether a provider's completion/output token count includes reasoning tokens.
# answer_tokens is normalized to exclude reasoning either way.
#   openai, deepseek: included (DeepSeek smoke test: 18 output incl. 15 reasoning)
#   xai, google:      reported separately (total = input + output + reasoning)
#   mistral:          assumed OpenAI semantics; it reports no reasoning tokens so far
REASONING_IN_OUTPUT = {"openai": True, "deepseek": True, "mistral": True,
                       "xai": False, "google": False}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CallResult:
    text:             str
    model_version:    str | None         # model string the API reports back
    finish_reason:    str | None
    input_tokens:     int | None = None
    answer_tokens:    int | None = None  # visible output only; never includes reasoning
    reasoning_tokens: int | None = None  # None when the provider doesn't report it
    timestamp:        str = field(default_factory=_utc_now)

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "").lower() in TRUNCATION_REASONS


def _answer_tokens(provider: str, output_tokens: int | None,
                   reasoning_tokens: int | None) -> int | None:
    if output_tokens is None:
        return None
    if reasoning_tokens and REASONING_IN_OUTPUT[provider]:
        return max(0, output_tokens - reasoning_tokens)
    return output_tokens


def call_openai_compat(provider: str, api_id: str, system_prompt: str,
                       user_prompt: str, max_tokens: int) -> CallResult:
    resp = get_client(provider).chat.completions.create(
        model=api_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        **{OPENAI_COMPAT[provider]["token_param"]: max_tokens},
    )
    choice    = resp.choices[0]
    usage     = resp.usage
    details   = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning = getattr(details, "reasoning_tokens", None)
    return CallResult(
        text=(choice.message.content or "").strip(),
        model_version=resp.model,
        finish_reason=choice.finish_reason,
        input_tokens=usage.prompt_tokens if usage else None,
        answer_tokens=_answer_tokens(provider, usage.completion_tokens if usage else None,
                                     reasoning),
        reasoning_tokens=reasoning,
    )


def call_anthropic(api_id: str, system_prompt: str, user_prompt: str,
                   max_tokens: int) -> CallResult:
    resp = get_client("anthropic").messages.create(
        model=api_id,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Only text blocks form the answer; thinking blocks (if any) are skipped.
    # Anthropic doesn't report thinking tokens separately: if the model thinks,
    # they are inside output_tokens and can't be subtracted out of answer_tokens.
    return CallResult(
        text="".join(b.text for b in resp.content if b.type == "text").strip(),
        model_version=resp.model,
        finish_reason=resp.stop_reason,
        input_tokens=resp.usage.input_tokens,
        answer_tokens=resp.usage.output_tokens,
    )


def call_gemini(api_id: str, system_prompt: str, user_prompt: str,
                max_tokens: int) -> CallResult:
    resp = get_client("google").models.generate_content(
        model=api_id,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
        ),
    )
    cand  = resp.candidates[0] if resp.candidates else None
    usage = resp.usage_metadata
    return CallResult(
        text=(resp.text or "").strip(),
        model_version=resp.model_version,
        finish_reason=cand.finish_reason.name if cand and cand.finish_reason else None,
        input_tokens=usage.prompt_token_count if usage else None,
        answer_tokens=usage.candidates_token_count if usage else None,  # excludes thoughts
        reasoning_tokens=usage.thoughts_token_count if usage else None,
    )


def make_caller(provider: str, api_id: str):
    if provider == "anthropic":
        return partial(call_anthropic, api_id)
    if provider == "google":
        return partial(call_gemini, api_id)
    return partial(call_openai_compat, provider, api_id)


MODEL_CALLERS = {key: make_caller(MODEL_PROVIDERS[key], api_id)
                 for key, api_id in MODELS.items()}

# =============================================================================
# Retry wrapper
# =============================================================================

def _status_code(e: Exception) -> int | None:
    # openai / anthropic: .status_code; google-genai: .code
    for attr in ("status_code", "code"):
        val = getattr(e, attr, None)
        if isinstance(val, int):
            return val
    return None


def _retry_after(e: Exception) -> float | None:
    headers = getattr(getattr(e, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers else None
    except (TypeError, ValueError):
        return None


def with_retry(fn, label: str):
    """
    Call fn() until it succeeds. 429/503/529 get exponential backoff (honoring
    Retry-After) up to TRANSIENT_ATTEMPTS; other errors get RETRY_ATTEMPTS with
    a fixed RETRY_DELAY. Returns None if every attempt fails.
    """
    transient = other = 0
    while True:
        try:
            return fn()
        except Exception as e:
            status = _status_code(e)
            if status in TRANSIENT_STATUS:
                transient += 1
                if transient >= TRANSIENT_ATTEMPTS:
                    print(f"    [{status} {transient}/{TRANSIENT_ATTEMPTS}] {label} giving up: {e}")
                    return None
                delay = min(BACKOFF_MAX, BACKOFF_BASE * 2 ** (transient - 1))
                delay *= random.uniform(0.8, 1.2)
                delay = max(delay, _retry_after(e) or 0)
                print(f"    [{status} {transient}/{TRANSIENT_ATTEMPTS}] {label}: "
                      f"retrying in {delay:.0f}s")
                sleep(delay)
            else:
                other += 1
                print(f"    [Attempt {other}/{RETRY_ATTEMPTS}] {label} failed: {e}")
                if other >= RETRY_ATTEMPTS:
                    return None
                sleep(RETRY_DELAY)


def call_with_retry(model_id: str, system_prompt: str, user_prompt: str,
                    max_tokens: int) -> CallResult | None:
    """
    Call a registered model. max_tokens is the answer budget; the request limit
    adds the model's reasoning_headroom. Warns when output was still truncated.
    """
    caller = MODEL_CALLERS[model_id]
    limit  = max_tokens + MODEL_HEADROOM[model_id]
    result = with_retry(lambda: caller(system_prompt, user_prompt, limit), model_id)
    if result is not None and result.truncated:
        print(f"    [warn] {model_id} hit its output limit ({limit} tokens; "
              f"finish_reason={result.finish_reason}, reasoning={result.reasoning_tokens})")
    return result


def call_text(model_id: str, system_prompt: str, user_prompt: str,
              max_tokens: int) -> str | None:
    """Text-only view of call_with_retry for storage; None on failure or empty output."""
    result = call_with_retry(model_id, system_prompt, user_prompt, max_tokens)
    return result.text if result is not None and result.text else None


# v2 storage: every response is stored as one of these records. JSON-format
# instruments (I1, I3, I5) add a "parsed" key alongside.
RECORD_FIELDS = ("text", "model_version", "finish_reason", "timestamp",
                 "answer_tokens", "reasoning_tokens", "input_tokens")


def make_record(result: CallResult | None) -> dict:
    """Storage record for a call. Failed or empty calls get text=None (retried next run)."""
    if result is None:
        return {**dict.fromkeys(RECORD_FIELDS), "timestamp": _utc_now()}
    return {
        "text":             result.text or None,
        "model_version":    result.model_version,
        "finish_reason":    result.finish_reason,
        "timestamp":        result.timestamp,
        "answer_tokens":    result.answer_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "input_tokens":     result.input_tokens,
    }


# =============================================================================
# Instrument 3 prompt builder
#
# Sends all 6 scenarios in one message. Forces JSON-only response in the format:
# {
#   "I3_S1": {
#     "legal_certainty":  {"score": 7, "explanation": "..."},
#     "accountability":   {"score": 4, "explanation": "..."},
#     "enforceability":   {"score": 5, "explanation": "..."}
#   },
#   "I3_S2": { ... },
#   ...
# }
# =============================================================================

def build_i3_prompt(instrument: dict) -> str:
    fmt        = instrument["response_format"]
    scenarios  = instrument["scenarios"]
    dimensions = fmt["dimensions"]

    dim_descriptions = "\n".join(
        f'  - "{d["id"]}": 1 = {d["anchor_low"]}, 10 = {d["anchor_high"]}'
        for d in dimensions
    )

    scenario_lines = "\n".join(
        f'{s["id"]}: {s["text"]}' for s in scenarios
    )

    scenario_keys = ", ".join(f'"{s["id"]}"' for s in scenarios)
    dim_keys      = ", ".join(f'"{d["id"]}"' for d in dimensions)

    return (
        f"{fmt['instruction']}\n\n"
        f"Rate each scenario on these three dimensions (1–10):\n"
        f"{dim_descriptions}\n\n"
        f"Scenarios:\n{scenario_lines}\n\n"
        f"Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.\n"
        f"Structure:\n"
        f"{{\n"
        f'  "<scenario_id>": {{\n'
        f'    "<dimension_id>": {{"score": <int 1-10>, "explanation": "<one sentence>"}}\n'
        f"  }}\n"
        f"}}\n\n"
        f"Scenario IDs: {scenario_keys}\n"
        f"Dimension IDs: {dim_keys}"
    )


def parse_i3_response(raw: str) -> dict | None:
    """
    Strip markdown fences if present and parse JSON.
    Returns parsed dict or None if parsing fails.
    """
    if not raw:
        return None
    # Strip ```json ... ``` fences
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError as e:
        print(f"    [parse_i3] JSON decode failed: {e}")
        return None


# =============================================================================
# Per-instrument checkpoint helpers
#
# Each instrument gets its own file: data/raw/instrument_N.json
#
# I1 / I2 structure:
# {
#   "gpt-4o": {
#     "baseline": {
#       "1": {"I1_Q1": "text...", "I1_Q2": "text...", ...},
#       "2": { ... },
#       "3": { ... }
#     },
#     "ceo": { ... }
#   }, ...
# }
#
# I3 structure (one call per run covers all scenarios):
# {
#   "gpt-4o": {
#     "baseline": {
#       "1": {
#         "raw": "...",           <- raw model output (preserved)
#         "parsed": {             <- parsed JSON ratings (None if parse failed)
#           "I3_S1": {
#             "legal_certainty":  {"score": 7, "explanation": "..."},
#             "accountability":   {"score": 4, "explanation": "..."},
#             "enforceability":   {"score": 5, "explanation": "..."}
#           }, ...
#         }
#       },
#       "2": { ... },
#       "3": { ... }
#     },
#     "ceo": { ... }
#   }, ...
# }
# =============================================================================

def instrument_path(instrument_id: str) -> Path:
    return RAW_DIR / f"{instrument_id}.json"


def load_instrument(instrument_id: str) -> dict:
    path = instrument_path(instrument_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def is_v1_raw_dir(path: Path) -> bool:
    return path.resolve() == V1_RAW_DIR.resolve()


def save_instrument(instrument_id: str, data: dict) -> None:
    if is_v1_raw_dir(RAW_DIR):
        raise RuntimeError(f"Refusing to write to {V1_RAW_DIR} (v1 data); use --raw-dir")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(instrument_path(instrument_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stored_text(val) -> str | None:
    """Model output stored in a cell: legacy str, v1 {raw, ...} or v2 {text, ...}."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("text", val.get("raw"))
    return None


def is_complete(data: dict, model: str, condition: str,
                run: int, question_id: str) -> bool:
    """Return True if this cell already has a non-empty response. (I2)"""
    try:
        return bool(stored_text(data[model][condition][str(run)][question_id]))
    except KeyError:
        return False


def is_complete_i3(data: dict, model: str, condition: str, run: int) -> bool:
    """Return True if this run already has a model output stored. (I3)"""
    try:
        return stored_text(data[model][condition][str(run)]) is not None
    except (KeyError, AttributeError):
        return False


def store_response(data: dict, model: str, condition: str,
                   run: int, question_id: str, record: dict) -> None:
    """Store a single question's response record. (I2)"""
    data.setdefault(model, {})
    data[model].setdefault(condition, {})
    data[model][condition].setdefault(str(run), {})
    data[model][condition][str(run)][question_id] = record


def store_i3_response(data: dict, model: str, condition: str,
                      run: int, record: dict, parsed: dict | None) -> None:
    """Store the response record + parsed ratings for one I3 run."""
    data.setdefault(model, {})
    data[model].setdefault(condition, {})
    data[model][condition][str(run)] = {**record, "parsed": parsed}


# =============================================================================
# Instrument 4 prompt builder + parser
#
# I4 structure (one call per evaluator/question, baseline only):
# {
#   "P1": {                          <- pair_id
#     "I1_Q1": {
#       "raw": "...",
#       "parsed": {
#         "legal_certainty_adequacy":  {"score": 7, "explanation": "..."},
#         "accountability_mechanisms": {"score": 5, "explanation": "..."},
#         "enforcement_conditions":    {"score": 6, "explanation": "..."}
#       }
#     },
#     "I1_Q2": { ... },
#     ...
#   },
#   "P2": { ... },
#   ...
# }
# =============================================================================

# Question text lookup for building I4 prompts
I1_QUESTION_TEXT = {
    "I1_Q1": "In your own words, define legal certainty as it applies to AI systems used in public decision-making. What does it require, and what does it look like when it is absent?",
    "I1_Q2": "How would you define accountability in the context of AI governance? Who or what should be accountable, for what, and to whom?",
    "I1_Q3": "What does enforceability mean when applied to AI regulation? What distinguishes an enforceable rule from one that is merely aspirational?",
}

I2_QUESTION_TEXT = {
    "I2_S1": "An AI system used by a national immigration authority flags visa applications for rejection using risk thresholds based on country of origin. This logic is not publicly disclosed. Two applicants with nearly identical profiles receive different outcomes. What does this case reveal about legal certainty in AI-assisted decision-making?",
    "I2_S2": "An AI system deployed by a city government recommends denial of housing assistance. The applicant challenges the decision. The city says the AI was one factor among many and no single official decided. The vendor says the model performed as specified. Who is responsible, and what accountability mechanisms should apply?",
    "I2_S3": "A national regulation requires annual audits of high-risk AI systems. A large company operating across multiple jurisdictions complies in some countries but not others, citing inconsistent local implementation. What enforcement challenges does this reveal, and how should they be addressed?",
}

ALL_QUESTION_TEXT = {**I1_QUESTION_TEXT, **I2_QUESTION_TEXT}


def build_i4_prompt(question_id: str, question_text: str,
                    evaluatee_response: str | dict, instrument: dict) -> str:
    """
    Build a single peer evaluation prompt for one question.
    Injects the evaluatee's run 1 baseline response and asks for structured ratings.
    evaluatee_response may be a stored cell (str or I1 {raw, parsed} dict);
    only its plain answer text is injected, never the dict or its sources.
    """
    evaluatee_response = get_answer_text(evaluatee_response)
    dims = instrument["response_format"]["dimensions"]
    dim_lines = "\n".join(
        f'  - "{d["id"]}": {d["prompt"]} (1 = weakest, 10 = strongest)'
        for d in dims
    )
    dim_keys = ", ".join(f'"{d["id"]}"' for d in dims)

    return (
        f"Below is a response from another AI system to the following governance question:\n\n"
        f"QUESTION: {question_text}\n\n"
        f"RESPONSE:\n{evaluatee_response}\n\n"
        f"Rate this response on three dimensions from 1 to 10 and briefly explain each rating:\n"
        f"{dim_lines}\n\n"
        f"Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.\n"
        f"Structure:\n"
        f"{{\n"
        f'  "<dimension_id>": {{"score": <int 1-10>, "explanation": "<one sentence>"}}\n'
        f"}}\n\n"
        f"Dimension IDs: {dim_keys}"
    )


def parse_i4_response(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON. Same pattern as I3."""
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError as e:
        print(f"    [parse_i4] JSON decode failed: {e}")
        return None


def is_complete_i4(data: dict, pair_id: str, question_id: str) -> bool:
    """Return True if this pair/question already has a raw response."""
    try:
        return data[pair_id][question_id].get("raw") is not None
    except (KeyError, AttributeError):
        return False


def store_i4_response(data: dict, pair_id: str, question_id: str,
                      raw: str | None, parsed: dict | None) -> None:
    """Store raw + parsed peer evaluation for one pair/question."""
    data.setdefault(pair_id, {})
    data[pair_id][question_id] = {"raw": raw, "parsed": parsed}


# =============================================================================
# Instrument 1 — JSON-forced collection with self-reported source citations
#
# Response format:
# {
#   "response": "<full analytical text>",
#   "sources": [
#     {"name": "EU AI Act", "type": "national_legislation", "jurisdiction": "EU"},
#     ...
#   ]
# }
#
# Storage (per question cell):
# {
#   "raw":    "<raw model output string>",
#   "parsed": {"response": "...", "sources": [...]}   <- None if parse fails
# }
# =============================================================================

I1_SOURCE_CATEGORIES = [
    "international_treaty", "national_legislation", "court_decision",
    "academic_work", "policy_framework", "news_media",
    "unverifiable", "implicit_only",
]


def build_i1_prompt(question: dict) -> str:
    """
    Wrap an I1 question with a JSON output instruction that requests
    both the analytical response and a list of self-reported source citations.
    """
    cats = I1_SOURCE_CATEGORIES
    return (
        f"{question['text']}\n\n"
        f"Respond ONLY with a JSON object. No preamble, no markdown, no extra text.\n"
        f"Use this exact structure:\n"
        f"{{\n"
        f'  "response": "<your full analytical response>",\n'
        f'  "sources": [\n'
        f'    {{\n'
        f'      "name": "<short label, e.g. EU AI Act, GDPR, Loomis v. Wisconsin>",\n'
        f'      "type": "<one of: {", ".join(cats)}>",\n'
        f'      "jurisdiction": "<e.g. EU, US, UN, UK, unspecified>"\n'
        f'    }}\n'
        f'  ]\n'
        f"}}"
    )


def _sanitize_json_controls(s: str) -> str:
    """
    Walk the string character-by-character and replace bare control characters
    (U+0000–U+001F) that appear inside JSON string values with their proper
    JSON escape sequences.  Characters outside strings are left untouched
    (they are valid JSON whitespace or structure).

    This fixes the common 'Invalid control character' JSONDecodeError caused
    by models embedding literal newlines / tabs inside string values.
    """
    result      = []
    in_string   = False
    escape_next = False
    _esc = {'\n': '\\n', '\r': '\\r', '\t': '\\t',
            '\x08': '\\b', '\x0c': '\\f'}
    for ch in s:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ord(ch) < 0x20:
            # Replace with escape sequence, or drop if no mapping
            result.append(_esc.get(ch, ''))
            continue
        result.append(ch)
    return ''.join(result)


def _try_parse_i1_json(clean: str) -> dict | None:
    """Attempt json.loads; on control-char error retry with sanitization."""
    for attempt, text in enumerate([clean, _sanitize_json_controls(clean)], 1):
        try:
            parsed = json.loads(text)
            if "response" in parsed and isinstance(parsed.get("sources"), list):
                return parsed
            if "response" in parsed:
                parsed.setdefault("sources", [])
                return parsed
            return None
        except json.JSONDecodeError as e:
            if attempt == 1:
                print(f"\n    [parse_i1] JSON decode failed ({e}); retrying with sanitization...",
                      end="", flush=True)
    print(f"\n    [parse_i1] JSON decode failed after sanitization")
    return None


def parse_i1_response(raw: str) -> dict | None:
    """
    Strip markdown fences and parse the I1 JSON response.
    Returns dict with 'response' and 'sources' keys, or None on failure.
    Applies control-character sanitization before giving up.
    """
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    return _try_parse_i1_json(clean.strip())


def is_complete_i1(data: dict, model: str, condition: str,
                   run: int, question_id: str) -> bool:
    """
    Return True if this I1 cell is already populated with a successfully parsed response.
    Cells where the output exists but parsed is None are NOT considered complete so
    that JSON-parse failures from a previous run are automatically retried.
    Handles legacy strings, v1 {raw, parsed} and v2 {text, ..., parsed} cells.
    """
    try:
        val = data[model][condition][str(run)][question_id]
        if isinstance(val, str):
            return bool(val)          # legacy format
        if isinstance(val, dict):
            # Only skip if we have both an output and a successfully parsed result
            return stored_text(val) is not None and val.get("parsed") is not None
        return False
    except (KeyError, AttributeError):
        return False


def store_i1_response(data: dict, model: str, condition: str, run: int,
                      question_id: str, record: dict, parsed: dict | None) -> None:
    """Store the response record + parsed I1 response."""
    data.setdefault(model, {})
    data[model].setdefault(condition, {})
    data[model][condition].setdefault(str(run), {})
    data[model][condition][str(run)][question_id] = {**record, "parsed": parsed}


# =============================================================================
# Instrument 5 — epistemic basis extraction
#
# I5 re-reads I1 and I2 responses and calls Claude Sonnet (as extractor) to
# identify all sources cited or implied in each response.
#
# Storage:
# {
#   "gpt-4o": {
#     "baseline": {
#       "1": {
#         "I1_Q1": {"raw": "...", "parsed": {"sources": [...]}},
#         "I2_S3": {"raw": "...", "parsed": {"sources": [...]}},
#         ...
#       },
#       "2": { ... },
#       "3": { ... }
#     },
#     "ceo": { ... }
#   }, ...
# }
# =============================================================================

# Question IDs from I1 and I2 that I5 re-analyzes
I5_SOURCE_QUESTIONS = ["I1_Q1", "I1_Q2", "I1_Q3", "I2_S1", "I2_S2", "I2_S3"]

I5_SOURCE_CATEGORIES = [
    "international_treaty", "national_legislation", "court_decision",
    "academic_work", "policy_framework", "news_media",
    "unverifiable", "implicit_only",
]


def build_i5_prompt(question_id: str, response_text: str) -> str:
    """Build the extraction prompt for a single I1/I2 response."""
    cats = I5_SOURCE_CATEGORIES
    return (
        f"You are a research assistant analyzing AI governance responses to identify cited sources.\n\n"
        f"The following is a response to AI governance question {question_id}.\n\n"
        f"RESPONSE:\n{response_text[:3000]}\n\n"
        f"Identify ALL documents, frameworks, legal precedents, court cases, treaties, academic works, "
        f"policy frameworks, or conceptual sources the response cites, references, implies, or draws upon "
        f"(even if not explicitly named).\n\n"
        f"For each source, provide:\n"
        f"- name: short label (e.g. 'EU AI Act', 'GDPR', 'Loomis v. Wisconsin')\n"
        f"- type: one of {cats}\n"
        f"- jurisdiction: e.g. 'EU', 'US', 'UN', 'UK', 'unspecified'\n"
        f"- legitimacy_tier: 1=verifiable primary legal source, 2=verifiable secondary source, "
        f"3=plausible but unverifiable, 4=vague or fabricated\n"
        f"- verifiable: true if you can confirm this source exists, false otherwise\n"
        f"- quote: exact phrase from the response implying this source, or null\n\n"
        f"Respond ONLY with a JSON object. No preamble, no markdown.\n"
        f"Format: {{\"sources\": [{{\"name\": \"...\", \"type\": \"...\", \"jurisdiction\": \"...\", "
        f"\"legitimacy_tier\": 1, \"verifiable\": true, \"quote\": \"...\"}}]}}"
    )


def parse_i5_response(raw: str) -> dict | None:
    """Strip markdown fences and parse the sources JSON. Returns None if parsing fails."""
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    try:
        parsed = json.loads(clean.strip())
        if "sources" in parsed and isinstance(parsed["sources"], list):
            return parsed
        return None
    except json.JSONDecodeError as e:
        print(f"    [parse_i5] JSON decode failed: {e}")
        return None


def is_complete_i5(data: dict, model: str, condition: str,
                   run: int, question_id: str) -> bool:
    """Return True if this cell already has an extractor output stored."""
    try:
        return stored_text(data[model][condition][str(run)][question_id]) is not None
    except (KeyError, AttributeError):
        return False


def store_i5_response(data: dict, model: str, condition: str, run: int,
                      question_id: str, record: dict, parsed: dict | None) -> None:
    """Store the extractor's response record + parsed sources for one response."""
    data.setdefault(model, {})
    data[model].setdefault(condition, {})
    data[model][condition].setdefault(str(run), {})
    data[model][condition][str(run)][question_id] = {**record, "parsed": parsed}


I5_EXTRACTOR_SYSTEM = (
    "You are a research assistant analyzing AI governance responses "
    "to identify cited sources. Respond ONLY with valid JSON. "
    "No preamble, no markdown fences."
)
I5_MAX_TOKENS = 2048


def call_i5_extractor(prompt: str) -> CallResult | None:
    """
    Always calls Claude Sonnet for source extraction regardless of the original model.
    Uses the shared retry pattern.
    """
    return with_retry(
        lambda: call_anthropic(EXTRACTOR_MODEL, I5_EXTRACTOR_SYSTEM, prompt, I5_MAX_TOKENS),
        "I5 extractor",
    )


# =============================================================================
# Dry run — counts calls and estimates tokens/cost without any API calls.
#
# Input tokens  = len(system prompt + user prompt) / CHARS_PER_TOKEN.
# Output tokens = the MAX_TOKENS setting for that call (an upper bound).
# I4 and I5 prompts embed earlier responses, which don't exist yet, so those
# are padded to the source instrument's max output length.
# Not counted: I1 JSON-parse re-queries and API-error retries.
# =============================================================================

def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def placeholder_response(q_id: str) -> str:
    """Stand-in for an I1/I2 response at its max output length."""
    max_tokens = MAX_TOKENS_JSON if q_id.startswith("I1") else MAX_TOKENS_OPEN
    return "x" * (max_tokens * CHARS_PER_TOKEN)


def dry_run(instruments_data: dict, peer_eval_data: dict,
            models: list[str], runs_per_cond: int) -> None:
    conditions = instruments_data["conditions"]
    pairs      = peer_eval_data["pairs"]

    # Rows: (instrument_id, requested model, billed API model ID, calls, in_tok, out_tok)
    rows = []

    for i_id in ACTIVE_INSTRUMENTS:
        instrument = instruments_data["instruments"][i_id]
        conds = [c for c in conditions if c in instrument["conditions"]]

        if i_id == "instrument_4":
            # One call per pair × question, billed to the evaluator; no runs
            system_prompt = conditions["baseline"]["system_prompt"]
            for model_id in models:
                calls = in_tok = out_tok = 0
                for pair in pairs:
                    if pair["evaluator"] != model_id:
                        continue
                    for q_id in pair["source_questions"]:
                        prompt = build_i4_prompt(q_id, ALL_QUESTION_TEXT[q_id],
                                                 placeholder_response(q_id), instrument)
                        calls   += 1
                        in_tok  += estimate_tokens(system_prompt + prompt)
                        out_tok += MAX_TOKENS_JSON
                rows.append((i_id, model_id, MODELS.get(model_id, model_id),
                             calls, in_tok, out_tok))
            continue

        if i_id == "instrument_5":
            # Extractor calls on each model's I1/I2 responses, billed to EXTRACTOR_MODEL
            per_cell = sum(
                estimate_tokens(I5_EXTRACTOR_SYSTEM
                                + build_i5_prompt(q_id, placeholder_response(q_id)))
                for q_id in I5_SOURCE_QUESTIONS
            )
            n = len(conds) * runs_per_cond
            for model_id in models:
                rows.append((i_id, model_id, EXTRACTOR_MODEL,
                             n * len(I5_SOURCE_QUESTIONS), n * per_cell,
                             n * len(I5_SOURCE_QUESTIONS) * I5_MAX_TOKENS))
            continue

        # I1 / I2 / I3: (user prompt, max_tokens) per call within one condition × run
        if i_id == "instrument_1":
            prompts = [(build_i1_prompt(q), MAX_TOKENS_JSON) for q in instrument["questions"]]
        elif i_id == "instrument_3":
            prompts = [(build_i3_prompt(instrument), MAX_TOKENS_JSON)]
        else:
            prompts = [(q["text"], MAX_TOKENS_OPEN) for q in instrument["questions"]]

        calls = in_tok = out_tok = 0
        for c in conds:
            system_prompt = conditions[c]["system_prompt"]
            for prompt, max_tokens in prompts:
                calls   += runs_per_cond
                in_tok  += runs_per_cond * estimate_tokens(system_prompt + prompt)
                out_tok += runs_per_cond * max_tokens
        for model_id in models:
            rows.append((i_id, model_id, MODELS.get(model_id, model_id),
                         calls, in_tok, out_tok))

    print("DRY RUN - no API calls made.")
    print(f"Models: {', '.join(models)}")
    print(f"runs_per_condition: {runs_per_cond}")
    print(f"Token estimate: {CHARS_PER_TOKEN} chars/token; output = MAX_TOKENS (upper bound)\n")

    # ---- Calls / tokens per instrument × model ----
    header = f"{'Instrument':<14} {'Model':<32} {'Billed to':<32} {'Calls':>7} {'In tok':>11} {'Out tok':>11}"
    print(header)
    print("-" * len(header))
    for i_id, model_id, billed, calls, in_tok, out_tok in rows:
        if calls:
            print(f"{i_id:<14} {model_id:<32} {billed:<32} {calls:>7,} {in_tok:>11,} {out_tok:>11,}")
    total_calls = sum(r[3] for r in rows)
    print("-" * len(header))
    print(f"{'TOTAL':<80} {total_calls:>7,}\n")

    # ---- Cost per billed model ----
    totals: dict[str, list[int]] = {}
    for _, _, billed, calls, in_tok, out_tok in rows:
        t = totals.setdefault(billed, [0, 0, 0])
        t[0] += calls
        t[1] += in_tok
        t[2] += out_tok

    header = f"{'Billed model':<32} {'Calls':>7} {'In tok':>11} {'Out tok':>11} {'$/1M in':>9} {'$/1M out':>9} {'Est. cost':>11}"
    print(header)
    print("-" * len(header))
    grand_total = 0.0
    missing     = []
    for billed, (calls, in_tok, out_tok) in totals.items():
        price = PRICES.get(billed, {})
        p_in, p_out = price.get("input"), price.get("output")
        if p_in is None or p_out is None:
            missing.append(billed)
            cost_str, p_in_str, p_out_str = "-", "-", "-"
        else:
            cost = (in_tok * p_in + out_tok * p_out) / 1_000_000
            grand_total += cost
            cost_str, p_in_str, p_out_str = f"${cost:,.2f}", f"{p_in:.2f}", f"{p_out:.2f}"
        print(f"{billed:<32} {calls:>7,} {in_tok:>11,} {out_tok:>11,} {p_in_str:>9} {p_out_str:>9} {cost_str:>11}")
    print("-" * len(header))
    print(f"{'TOTAL (priced models only)':<85}${grand_total:>10,.2f}")
    if missing:
        print(f"\nNo price in PRICES for: {', '.join(missing)}")


# =============================================================================
# Smoke test (--smoke-test) — one short real call per registered model.
# Uses the I1/I2 answer budget, so each model's real request limit
# (MAX_TOKENS_OPEN + reasoning_headroom) is exercised too. Stores nothing.
# =============================================================================

SMOKE_SYSTEM = "You are a helpful assistant."
SMOKE_PROMPT = "Say OK."


def smoke_test() -> None:
    rows = []
    for model_id in MODELS:
        print(f"  {model_id} ... ", end="", flush=True)
        r = call_with_retry(model_id, SMOKE_SYSTEM, SMOKE_PROMPT, MAX_TOKENS_OPEN)
        ok = r is not None and bool(r.text)
        print("ok" if ok else "FAILED")
        rows.append((model_id, ok, r))

    header = (f"{'Model':<20} {'Result':<6} {'Returned version':<28} {'finish_reason':<14} "
              f"{'In':>6} {'Answer':>6} {'Reason':>7}  Text")
    print("\n" + header)
    print("-" * len(header))
    for model_id, ok, r in rows:
        if r is None:
            print(f"{model_id:<20} {'fail':<6} {'-':<28} {'-':<14} {'-':>6} {'-':>6} {'-':>7}")
            continue
        fmt = lambda v: "n/a" if v is None else v
        print(f"{model_id:<20} {'OK' if ok else 'fail':<6} {str(r.model_version):<28} "
              f"{str(r.finish_reason):<14} {fmt(r.input_tokens):>6} "
              f"{fmt(r.answer_tokens):>6} {fmt(r.reasoning_tokens):>7}  {r.text[:30]!r}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect LLM responses for all active instruments.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count calls and estimate tokens/cost; make no API calls.")
    parser.add_argument("--models", nargs="+",
                        help="Model keys for --dry-run (default: instruments.json models). "
                             "Keys not in MODELS are treated as API model IDs.")
    parser.add_argument("--runs", type=int, default=DRY_RUN_RUNS,
                        help=f"runs_per_condition for --dry-run (default: {DRY_RUN_RUNS}).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Send one short real call to each model in config/models.yaml "
                             "and print a summary table. Stores nothing.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help=f"Directory for instrument JSON files (default: {RAW_DIR}). "
                             f"{V1_RAW_DIR} itself (v1 data) is refused.")
    args = parser.parse_args()

    if is_v1_raw_dir(args.raw_dir):
        sys.exit(f"--raw-dir {args.raw_dir} is the v1 data directory; refusing to write there.")
    RAW_DIR = args.raw_dir   # module scope: rebinds the global used by the storage helpers

    if args.smoke_test:
        smoke_test()
        sys.exit(0)

    with open(INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
        instruments_data = json.load(f)

    with open(PEER_EVAL_FILE, "r", encoding="utf-8") as f:
        peer_eval_data = json.load(f)

    # Module scope: rebinds the global that dry_run() and the loops below read
    ACTIVE_INSTRUMENTS = instruments_data.get("active_instruments", ACTIVE_INSTRUMENTS)

    if args.dry_run:
        dry_run(instruments_data, peer_eval_data,
                args.models or instruments_data["models"], args.runs)
        sys.exit(0)

    pairs         = peer_eval_data["pairs"]
    conditions    = instruments_data["conditions"]
    models        = instruments_data["models"]
    runs_per_cond = instruments_data["runs_per_condition"]

    needed = set(models)
    if "instrument_4" in ACTIVE_INSTRUMENTS:
        needed |= {p["evaluator"] for p in pairs}
    unknown = sorted(needed - set(MODEL_CALLERS))
    if unknown:
        sys.exit(f"Not in config/models.yaml: {', '.join(unknown)}. "
                 f"Update the 'models' list in {INSTRUMENTS_FILE}"
                 + (f" and the evaluators in {PEER_EVAL_FILE}"
                    if "instrument_4" in ACTIVE_INSTRUMENTS else "")
                 + f" to registry keys: {', '.join(MODEL_CALLERS)}")

    # Count total calls for progress reporting
    total = 0
    for i_id in ACTIVE_INSTRUMENTS:
        instrument = instruments_data["instruments"][i_id]
        if i_id == "instrument_3":
            n_questions = 1
        elif i_id == "instrument_4":
            # One call per pair × question (baseline only, no runs)
            n_questions = len(pairs) * len(ALL_QUESTION_TEXT)
            total += n_questions
            continue
        elif i_id == "instrument_5":
            # One extractor call per model × condition × run × source question
            # Source questions = 3 from I1 + 3 from I2
            total += (
                len(models)
                * len(instrument["conditions"])
                * runs_per_cond
                * len(I5_SOURCE_QUESTIONS)
            )
            continue
        else:
            n_questions = len(instrument.get("questions", []))
        total += (
            len(models)
            * len(instrument["conditions"])
            * runs_per_cond
            * n_questions
        )

    completed = 0
    skipped   = 0

    print(f"Starting collection — {total} total calls across {len(ACTIVE_INSTRUMENTS)} instrument(s).")
    print(f"Output directory: {RAW_DIR}\n")

    for instrument_id in ACTIVE_INSTRUMENTS:
        instrument = instruments_data["instruments"][instrument_id]
        data = load_instrument(instrument_id)

        print(f"{'='*60}")
        print(f"Instrument: {instrument_id} — {instrument['label']}")
        print(f"{'='*60}")

        # ---- Instrument 5: epistemic source extraction via Claude Sonnet ----
        if instrument_id == "instrument_5":
            # Load source data from I1 and I2
            i1_src = load_instrument("instrument_1")
            i2_src = load_instrument("instrument_2")
            src_map = {
                "I1_Q1": i1_src, "I1_Q2": i1_src, "I1_Q3": i1_src,
                "I2_S1": i2_src, "I2_S2": i2_src, "I2_S3": i2_src,
            }

            for model_id in models:
                for condition_id, condition in conditions.items():
                    if condition_id not in instrument["conditions"]:
                        continue

                    for run in range(1, runs_per_cond + 1):
                        for q_id in I5_SOURCE_QUESTIONS:
                            if is_complete_i5(data, model_id, condition_id, run, q_id):
                                skipped += 1
                                print(f"  [skip] {model_id} | {condition_id} | run {run} | {q_id}")
                                continue

                            # Fetch the original response from I1 or I2 as plain text.
                            src_val = (
                                src_map[q_id]
                                .get(model_id, {})
                                .get(condition_id, {})
                                .get(str(run), {})
                                .get(q_id)
                            )
                            try:
                                src_response = get_answer_text(src_val) if src_val is not None else ""
                            except AnswerTextError:
                                src_response = ""   # failed I1/I2 call stored with text=None

                            if not src_response:
                                print(
                                    f"  [skip] {model_id} | {condition_id} | run {run} | {q_id}"
                                    f" — source response not found"
                                )
                                skipped += 1
                                continue

                            prompt = build_i5_prompt(q_id, src_response)

                            print(
                                f"  [call] {model_id} | {condition_id} | run {run} | {q_id} ... ",
                                end="", flush=True,
                            )

                            record = make_record(call_i5_extractor(prompt))
                            raw    = record["text"]
                            parsed = parse_i5_response(raw) if raw else None

                            store_i5_response(data, model_id, condition_id, run, q_id, record, parsed)
                            save_instrument(instrument_id, data)

                            status = "ok" if parsed else ("raw only" if raw else "FAILED")
                            print(status)
                            completed += 1
                            sleep(CALL_DELAY)

        # ---- Instrument 4: peer evaluation, one call per pair × question ----
        elif instrument_id == "instrument_4":
            system_prompt = conditions["baseline"]["system_prompt"]

            # Load I1 and I2 raw data to pull evaluatee responses
            i1_data = load_instrument("instrument_1")
            i2_data = load_instrument("instrument_2")

            for pair in pairs:
                pair_id   = pair["pair_id"]
                evaluator = pair["evaluator"]
                evaluatee = pair["evaluatee"]
                questions = pair["source_questions"]

                print(f"  Pair {pair_id}: {MODEL_LABELS.get(evaluator, evaluator)}"
                      f" evaluates {MODEL_LABELS.get(evaluatee, evaluatee)}")

                for q_id in questions:
                    if is_complete_i4(data, pair_id, q_id):
                        skipped += 1
                        print(f"    [skip] {pair_id} | {q_id}")
                        continue

                    # Fetch evaluatee's run 1 baseline response as plain text
                    src_data = i1_data if q_id.startswith("I1") else i2_data
                    cell = src_data.get(evaluatee, {}).get("baseline", {}).get("1", {}).get(q_id)
                    evaluatee_response = get_answer_text(cell) if cell is not None else ""

                    if not evaluatee_response:
                        print(f"    [skip] {pair_id} | {q_id} — evaluatee response not found")
                        skipped += 1
                        continue

                    question_text = ALL_QUESTION_TEXT[q_id]
                    prompt = build_i4_prompt(q_id, question_text,
                                             evaluatee_response, instrument)

                    print(f"    [call] {pair_id} | {q_id} ... ", end="", flush=True)

                    raw    = call_text(evaluator, system_prompt, prompt,
                                       MAX_TOKENS_JSON)
                    parsed = parse_i4_response(raw) if raw else None

                    store_i4_response(data, pair_id, q_id, raw, parsed)
                    save_instrument(instrument_id, data)

                    status = "ok" if parsed else ("raw only" if raw else "FAILED")
                    print(status)
                    completed += 1
                    sleep(CALL_DELAY)

        # ---- Instrument 3: one bundled call per run ----
        elif instrument_id == "instrument_3":
            i3_prompt = build_i3_prompt(instrument)

            for model_id in models:
                for condition_id, condition in conditions.items():
                    if condition_id not in instrument["conditions"]:
                        continue

                    system_prompt = condition["system_prompt"]

                    for run in range(1, runs_per_cond + 1):
                        if is_complete_i3(data, model_id, condition_id, run):
                            skipped += 1
                            print(f"  [skip] {model_id} | {condition_id} | run {run} | all scenarios")
                            continue

                        print(
                            f"  [call] {model_id} | {condition_id} | run {run} | all scenarios ... ",
                            end="", flush=True
                        )

                        record = make_record(call_with_retry(model_id, system_prompt,
                                                             i3_prompt, MAX_TOKENS_JSON))
                        raw    = record["text"]
                        parsed = parse_i3_response(raw) if raw else None

                        store_i3_response(data, model_id, condition_id, run, record, parsed)
                        save_instrument(instrument_id, data)

                        status = "ok" if parsed else ("raw only" if raw else "FAILED")
                        print(status)
                        completed += 1
                        sleep(CALL_DELAY)

        # ---- Instrument 1: JSON-forced, one call per question (response + sources) ----
        elif instrument_id == "instrument_1":
            for model_id in models:
                for condition_id, condition in conditions.items():
                    if condition_id not in instrument["conditions"]:
                        continue

                    system_prompt = condition["system_prompt"]

                    for run in range(1, runs_per_cond + 1):
                        for question in instrument["questions"]:
                            q_id = question["id"]

                            if is_complete_i1(data, model_id, condition_id, run, q_id):
                                skipped += 1
                                print(f"  [skip] {model_id} | {condition_id} | run {run} | {q_id}")
                                continue

                            print(
                                f"  [call] {model_id} | {condition_id} | run {run} | {q_id} ... ",
                                end="", flush=True
                            )

                            prompt = build_i1_prompt(question)
                            raw, parsed = None, None
                            for parse_attempt in range(1, RETRY_ATTEMPTS + 1):
                                record = make_record(call_with_retry(model_id, system_prompt,
                                                                     prompt, MAX_TOKENS_JSON))
                                raw    = record["text"]
                                parsed = parse_i1_response(raw) if raw else None
                                if parsed:
                                    break
                                if parse_attempt < RETRY_ATTEMPTS:
                                    print(
                                        f"\n    [i1-retry {parse_attempt}/{RETRY_ATTEMPTS}]"
                                        f" JSON still invalid — re-querying... ",
                                        end="", flush=True,
                                    )
                                    sleep(RETRY_DELAY)

                            store_i1_response(data, model_id, condition_id, run, q_id, record, parsed)
                            save_instrument(instrument_id, data)

                            status = "ok" if parsed else ("raw only" if raw else "FAILED")
                            print(status)
                            completed += 1
                            sleep(CALL_DELAY)

        # ---- Instrument 2: free-text response per question ----
        else:
            for model_id in models:
                for condition_id, condition in conditions.items():
                    if condition_id not in instrument["conditions"]:
                        continue

                    system_prompt = condition["system_prompt"]

                    for run in range(1, runs_per_cond + 1):
                        for question in instrument["questions"]:
                            q_id = question["id"]

                            if is_complete(data, model_id, condition_id, run, q_id):
                                skipped += 1
                                print(f"  [skip] {model_id} | {condition_id} | run {run} | {q_id}")
                                continue

                            print(
                                f"  [call] {model_id} | {condition_id} | run {run} | {q_id} ... ",
                                end="", flush=True
                            )

                            record = make_record(call_with_retry(model_id, system_prompt,
                                                                 question["text"], MAX_TOKENS_OPEN))

                            store_response(data, model_id, condition_id, run, q_id, record)
                            save_instrument(instrument_id, data)

                            print("ok" if record["text"] else "FAILED")
                            completed += 1
                            sleep(CALL_DELAY)

    print(f"\nDone. {completed} new responses collected, {skipped} already complete.")
    print(f"Files saved to {RAW_DIR}/")
    for i_id in ACTIVE_INSTRUMENTS:
        p = instrument_path(i_id)
        print(f"  {p}")