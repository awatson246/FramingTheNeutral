"""
Check which candidate model IDs are callable on each provider and which request
parameters they accept. Prints every provider's list-models output, then sends
1-token smoke tests and writes a summary table to docs/models_available.md.

Usage:
    python scripts/check_models.py
    python scripts/check_models.py --only gpt-6.1-sol grok-4.7
        (re-tests only those candidates; other rows are kept from the
         existing docs/models_available.md. list-models always re-runs.)

Notes:
- "Accepted" means the API returned 200 with that parameter set. Some providers
  silently ignore parameters they don't support, so accepted != honored.
- Reasoning detection is heuristic (reasoning-token counts, reasoning content,
  or "temperature rejected + effort accepted"); see the Evidence column.
- Mistral and xAI are queried through their OpenAI-compatible endpoints.
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from time import sleep

# Add C:\libs to path for model client libraries installed there
sys.path.insert(0, "C:\\libs")

import anthropic
import openai
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config
# =============================================================================

OUTPUT_FILE = Path("docs/models_available.md")

CANDIDATES = [
    "gpt-6.1-sol",
    "claude-sonnet-5-5",
    "gemini-3.8-flash",
    "deepseek-flash",
    "mistral-medium-3.5",
    "grok-4.7",
]

# provider -> (base_url, env var) for OpenAI-compatible APIs
OPENAI_COMPAT = {
    "openai":   (None,                          "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "mistral":  ("https://api.mistral.ai/v1",   "MISTRAL_API_KEY"),
    "xai":      ("https://api.x.ai/v1",         "XAI_API_KEY"),
}

PROVIDER_ENV = {
    **{p: env for p, (_, env) in OPENAI_COMPAT.items()},
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
}

PROVIDER_ORDER = ["openai", "anthropic", "google", "deepseek", "mistral", "xai"]

# Candidate ID prefix -> provider
PREFIXES = [
    ("gpt",       "openai"),
    ("claude",    "anthropic"),
    ("gemini",    "google"),
    ("deepseek",  "deepseek"),
    ("mistral",   "mistral"),
    ("magistral", "mistral"),
    ("grok",      "xai"),
]

TIMEOUT   = 60
SMOKE_MSG = "Hi"

# Reasoning models spend output tokens on thinking, so 1 can be too few
REASONING_MAX_TOKENS = 64

# Retries for 503 / "high demand" errors
OVERLOAD_RETRIES = 3
OVERLOAD_DELAY   = 10  # seconds
OVERLOAD_MARKERS = ("503", "UNAVAILABLE", "high demand", "overloaded")

# =============================================================================
# Helpers
# =============================================================================

def provider_for(model_id: str) -> str | None:
    for prefix, provider in PREFIXES:
        if model_id.startswith(prefix):
            return provider
    return None


def short_error(e: Exception) -> str:
    msg = str(e).replace("\n", " ").replace("|", "/")
    return msg[:160] + ("..." if len(msg) > 160 else "")


def is_overloaded(err: str) -> bool:
    return any(m in err for m in OVERLOAD_MARKERS)


def probe(fn):
    """
    Run fn(); return (True, result) or (False, short error string).
    Overload errors (503 / high demand) are retried OVERLOAD_RETRIES times.
    """
    for attempt in range(OVERLOAD_RETRIES + 1):
        try:
            return True, fn()
        except Exception as e:
            err = short_error(e)
            if not is_overloaded(err) or attempt == OVERLOAD_RETRIES:
                return False, err
            print(f"[503, retry {attempt + 1}/{OVERLOAD_RETRIES} in {OVERLOAD_DELAY}s] ",
                  end="", flush=True)
            sleep(OVERLOAD_DELAY)


def yn(ok: bool, err: str = "") -> str:
    if ok:
        return "yes"
    return "unknown (503)" if is_overloaded(err) else "no"


def make_client(provider: str):
    key = os.getenv(PROVIDER_ENV[provider])
    if not key:
        return None
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=key, max_retries=0, timeout=TIMEOUT)
    if provider == "google":
        return genai.Client(api_key=key)
    base_url, _ = OPENAI_COMPAT[provider]
    return openai.OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=TIMEOUT)


def list_models(provider: str, client) -> list[str]:
    if provider == "google":
        return sorted(m.name.removeprefix("models/") for m in client.models.list())
    # Anthropic and OpenAI-compatible SDKs both page through .data with .id
    return sorted(m.id for m in client.models.list())


def empty_result(model_id: str, provider: str | None) -> dict:
    return {
        "model": model_id, "provider": provider or "?", "listed": "?",
        "callable": "no", "returned": "", "reasoning": "?", "evidence": "",
        "temperature": "", "max_tokens": "", "max_completion_tokens": "",
        "effort": "", "notes": "",
    }

# =============================================================================
# Provider-specific smoke tests
# Each fills in and returns a result dict.
# =============================================================================

def check_openai_compat(client, model_id: str, res: dict) -> dict:
    msgs = [{"role": "user", "content": SMOKE_MSG}]

    def call(**kw):
        return client.chat.completions.create(model=model_id, messages=msgs, **kw)

    notes = []

    mt_ok,  mt_r  = probe(lambda: call(max_tokens=1))
    mct_ok, mct_r = probe(lambda: call(max_completion_tokens=1))
    mct_limit = 1
    if not mct_ok:
        # Reasoning models can fail at 1 token; retry with room to think
        notes.append(f"max_completion_tokens=1: {mct_r}")
        mct_ok, mct_r = probe(lambda: call(max_completion_tokens=REASONING_MAX_TOKENS))
        mct_limit = REASONING_MAX_TOKENS
    if not mt_ok:
        notes.insert(0, f"max_tokens: {mt_r}")
    if not mct_ok:
        notes.append(f"max_completion_tokens={mct_limit}: {mct_r}")

    res["max_tokens"] = yn(mt_ok, mt_r if not mt_ok else "")
    res["max_completion_tokens"] = (
        yn(mct_ok, mct_r if not mct_ok else "")
        + (f" (needs >= {mct_limit})" if mct_ok and mct_limit > 1 else "")
    )

    if not (mt_ok or mct_ok):
        if is_overloaded(mt_r) or is_overloaded(mct_r):
            res["callable"] = "unknown (503)"
        res["notes"] = "; ".join(notes)
        return res

    resp = mt_r if mt_ok else mct_r
    tp   = {"max_tokens": 1} if mt_ok else {"max_completion_tokens": mct_limit}
    res["callable"] = "yes"
    res["returned"] = resp.model or ""

    temp_ok, temp_r = probe(lambda: call(temperature=0.5, **tp))
    eff_ok,  eff_r  = probe(lambda: call(reasoning_effort="low", **tp))
    res["temperature"] = yn(temp_ok, "" if temp_ok else temp_r)
    res["effort"]      = "yes (reasoning_effort)" if eff_ok else yn(False, eff_r)

    # Reasoning signals
    evidence = []
    details = getattr(resp.usage, "completion_tokens_details", None) if resp.usage else None
    if details is not None and (getattr(details, "reasoning_tokens", 0) or 0) > 0:
        evidence.append("reasoning_tokens>0")
    msg_extra = (resp.choices[0].message.model_extra or {}) if resp.choices else {}
    if msg_extra.get("reasoning_content") or msg_extra.get("reasoning"):
        evidence.append("reasoning_content")
    if evidence:
        res["reasoning"] = "yes"
    elif not temp_ok and eff_ok:
        res["reasoning"] = "likely"
        evidence.append("temperature rejected, effort accepted")
    else:
        res["reasoning"] = "no evidence"
    if not mt_ok and "max_completion_tokens" in mt_r:
        evidence.append("max_tokens rejected in favor of max_completion_tokens")
    res["evidence"] = ", ".join(evidence)

    if not temp_ok:
        notes.append(f"temperature: {temp_r}")
    if not eff_ok:
        notes.append(f"effort: {eff_r}")
    res["notes"] = "; ".join(notes)
    return res


def check_anthropic(client, model_id: str, res: dict) -> dict:
    msgs = [{"role": "user", "content": SMOKE_MSG}]

    def call(**kw):
        return client.messages.create(model=model_id, max_tokens=1, messages=msgs, **kw)

    ok, resp = probe(call)
    res["max_tokens"]            = yn(ok, "" if ok else resp)
    res["max_completion_tokens"] = "n/a"
    if not ok:
        if is_overloaded(resp):
            res["callable"] = "unknown (503)"
        res["notes"] = resp
        return res

    res["callable"] = "yes"
    res["returned"] = resp.model

    temp_ok, temp_r = probe(lambda: call(temperature=0.5))
    # Effort is passed via extra_body so this works regardless of SDK version
    eff_ok, eff_r = probe(lambda: call(extra_body={"output_config": {"effort": "low"}}))
    res["temperature"] = yn(temp_ok, "" if temp_ok else temp_r)
    res["effort"]      = "yes (output_config.effort)" if eff_ok else yn(False, eff_r)

    if eff_ok:
        res["reasoning"] = "likely"
        res["evidence"]  = "effort accepted (extended thinking not probed: needs budget >= 1024)"
    else:
        res["reasoning"] = "no evidence"

    notes = []
    if not temp_ok:
        notes.append(f"temperature: {temp_r}")
    if not eff_ok:
        notes.append(f"effort: {eff_r}")
    res["notes"] = "; ".join(notes)
    return res


def check_google(client, model_id: str, res: dict) -> dict:
    def call(**cfg):
        return client.models.generate_content(
            model=model_id,
            contents=SMOKE_MSG,
            config=genai_types.GenerateContentConfig(max_output_tokens=1, **cfg),
        )

    ok, resp = probe(call)
    res["max_tokens"]            = "n/a (max_output_tokens)"
    res["max_completion_tokens"] = "n/a"
    if not ok:
        if is_overloaded(resp):
            res["callable"] = "unknown (503)"
        res["notes"] = resp
        return res

    res["callable"] = "yes"
    res["returned"] = resp.model_version or ""

    temp_ok, temp_r = probe(lambda: call(temperature=0.5))
    lvl_ok, lvl_r = probe(lambda: call(
        thinking_config=genai_types.ThinkingConfig(thinking_level="low")))
    bud_ok, bud_r = probe(lambda: call(
        thinking_config=genai_types.ThinkingConfig(thinking_budget=128)))
    res["temperature"] = yn(temp_ok, "" if temp_ok else temp_r)
    effort = [
        f"{name}: {yn(okk, '' if okk else err)}"
        for name, okk, err in [("thinking_level", lvl_ok, lvl_r),
                               ("thinking_budget", bud_ok, bud_r)]
    ]
    if lvl_ok or bud_ok:
        res["effort"] = "yes (" + ", ".join(effort) + ")"
    elif is_overloaded(lvl_r) or is_overloaded(bud_r):
        res["effort"] = "unknown (503) (" + ", ".join(effort) + ")"
    else:
        res["effort"] = "no"

    evidence = []
    usage = resp.usage_metadata
    if usage is not None and (usage.thoughts_token_count or 0) > 0:
        evidence.append("thoughts_token_count>0")
    info_ok, info = probe(lambda: client.models.get(model=model_id))
    if info_ok and getattr(info, "thinking", None):
        evidence.append("models.get: thinking=True")
    if evidence:
        res["reasoning"] = "yes"
    elif lvl_ok or bud_ok:
        res["reasoning"] = "likely"
        evidence.append("thinking config accepted")
    else:
        res["reasoning"] = "no evidence"
    res["evidence"] = ", ".join(evidence)

    notes = []
    if not temp_ok:
        notes.append(f"temperature: {temp_r}")
    if not lvl_ok:
        notes.append(f"thinking_level: {lvl_r}")
    if not bud_ok:
        notes.append(f"thinking_budget: {bud_r}")
    res["notes"] = "; ".join(notes)
    return res


def check_candidate(model_id: str, clients: dict, listed: dict) -> dict:
    provider = provider_for(model_id)
    res = empty_result(model_id, provider)
    if provider is None:
        res["notes"] = "unknown provider prefix"
        return res

    client = clients.get(provider)
    if client is None:
        res["notes"] = f"{PROVIDER_ENV[provider]} not set"
        return res

    if provider in listed:
        res["listed"] = yn(model_id in listed[provider])

    if provider == "anthropic":
        return check_anthropic(client, model_id, res)
    if provider == "google":
        return check_google(client, model_id, res)
    return check_openai_compat(client, model_id, res)

# =============================================================================
# Output
# =============================================================================

COLUMNS = [
    ("model",                 "Candidate ID"),
    ("provider",              "Provider"),
    ("listed",                "In list-models"),
    ("callable",              "Callable"),
    ("returned",              "Returned model/version"),
    ("reasoning",             "Reasoning model"),
    ("evidence",              "Evidence"),
    ("temperature",           "temperature"),
    ("max_tokens",            "max_tokens"),
    ("max_completion_tokens", "max_completion_tokens"),
    ("effort",                "Reasoning effort"),
    ("notes",                 "Notes / errors"),
]


def load_existing_rows() -> dict[str, dict]:
    """Parse the results table in OUTPUT_FILE into {candidate ID: row dict}."""
    if not OUTPUT_FILE.exists():
        return {}
    rows = {}
    for line in OUTPUT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| Candidate ID"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == len(COLUMNS):
            row = {k: v for (k, _), v in zip(COLUMNS, cells)}
            rows[row["model"]] = row
    return rows


def write_markdown(results: list[dict], listed: dict, list_errors: dict) -> None:
    lines = [
        "# Models available",
        "",
        f"Generated by `scripts/check_models.py` on {date.today().isoformat()}.",
        "",
        "Each candidate got 1-token smoke tests. Parameter columns show whether the API "
        "accepted a request with that parameter set; some providers silently ignore "
        "parameters they don't support, so accepted does not mean honored. "
        "Reasoning detection is heuristic; see Evidence.",
        "",
        "| " + " | ".join(h for _, h in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for r in results:
        lines.append("| " + " | ".join(str(r[k]) for k, _ in COLUMNS) + " |")

    lines += ["", "## list-models output", ""]
    for provider in PROVIDER_ORDER:
        if provider in list_errors:
            lines.append(f"- **{provider}**: {list_errors[provider]}")
            continue
        ids = listed.get(provider, [])
        lines += [
            f"<details><summary><b>{provider}</b> ({len(ids)} models)</summary>",
            "",
            *[f"- `{i}`" for i in ids],
            "",
            "</details>",
            "",
        ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model availability and parameters.")
    parser.add_argument("--only", nargs="+", metavar="MODEL_ID",
                        help="Re-test only these candidates; keep other rows from the existing file.")
    args = parser.parse_args()

    to_test = args.only or CANDIDATES
    unknown = [m for m in to_test if m not in CANDIDATES]
    if unknown:
        sys.exit(f"Not in CANDIDATES: {', '.join(unknown)}")
    existing = load_existing_rows() if args.only else {}

    clients     = {}
    listed      = {}
    list_errors = {}

    for provider in PROVIDER_ORDER:
        print(f"{'='*60}\n{provider}\n{'='*60}")
        client = make_client(provider)
        if client is None:
            list_errors[provider] = f"{PROVIDER_ENV[provider]} not set"
            print(f"  skipped: {list_errors[provider]}\n")
            continue
        clients[provider] = client

        ok, ids = probe(lambda: list_models(provider, client))
        if not ok:
            list_errors[provider] = f"list-models failed: {ids}"
            print(f"  {list_errors[provider]}\n")
            continue
        listed[provider] = ids
        for i in ids:
            print(f"  {i}")
        print()

    print(f"{'='*60}\nSmoke tests\n{'='*60}")
    results = []
    for model_id in CANDIDATES:
        if model_id not in to_test:
            if model_id in existing:
                results.append(existing[model_id])
                print(f"  {model_id}: kept from existing file")
            else:
                print(f"  {model_id}: not re-tested and no existing row; omitted")
            continue
        print(f"  {model_id} ... ", end="", flush=True)
        r = check_candidate(model_id, clients, listed)
        results.append(r)
        print(f"callable={r['callable']} returned={r['returned'] or '-'} "
              f"reasoning={r['reasoning']}")

    write_markdown(results, listed, list_errors)
    print(f"\nWrote {OUTPUT_FILE}")
