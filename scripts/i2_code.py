"""
I2 criterion coding, step 3: code every I2 answer against its own model's I1 criteria,
with two LLM coders.

Criteria: data/processed/i2_criteria_v1.json (the version reviewed and hand-coded
against in manual_review.json). Answers: all 108 I2 answers, plain text via
get_answer_text; Mistral from data/raw/mistral_i2_recollect.json. Each answer is
coded against the criteria of the same model, condition and run
(I1_Q1 -> I2_S1, I1_Q2 -> I2_S2, I1_Q3 -> I2_S3).

One call per answer per coder, all its criteria at once, JSON output. Coders are
not told which model wrote the answer or under which condition.
  claude-sonnet   claude-sonnet-4-5-20250929, temperature 0
  gpt-6.1-sol     provider defaults; max_completion_tokens 4096 + 16384 reasoning headroom

Codes per criterion: met / not_addressed / contradicted. Met and contradicted need
a verbatim evidence quote, checked in code against the answer:
  exact       quote is a substring of the answer as stored
  normalized  substring after collapsing whitespace, straightening quotes/dashes
              and lowercasing
  flag        "" | "typography" (normalized only) | "not_found" | "no_evidence"

Output: data/processed/i2_codes.json
  {"meta": {...},
   "codes": {coder: {model: {condition: {run: {I2_Sx: {
       "answer_id", "criteria_cell_id", "raw",
       "parsed": {"codes": [{n, code, evidence}]},
       "quote_check": [{n, exact, normalized, flag}], "flags": [...], "meta": {...}}}}}}}}
Re-running fills in only cells without a parsed result. Reads data/raw/ only.

Usage:
    python scripts/i2_code.py --dry-run   # call counts, no API calls
    python scripts/i2_code.py             # real run (both coders in parallel)
"""

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import anthropic
import openai
from dotenv import load_dotenv

from answer_text import get_answer_text, _strip_fences
from i2_criteria import CONDITIONS, INSTRUMENTS_FILE, LABEL, MAPPING, MODELS, RUNS, normalize

load_dotenv(".env")

CRITERIA_FILE  = Path("data/processed/i2_criteria_v1.json")
I2_FILE        = Path("data/raw/instrument_2.json")
RECOLLECT_FILE = Path("data/raw/mistral_i2_recollect.json")
OUT_FILE       = Path("data/processed/i2_codes.json")

CLAUDE_MODEL       = "claude-sonnet-4-5-20250929"
CLAUDE_MAX_TOKENS  = 4096
GPT_MODEL          = "gpt-6.1-sol"
GPT_ANSWER_TOKENS  = 4096
GPT_REASONING_HEADROOM = 16384
CODERS         = {"claude-sonnet": CLAUDE_MODEL, "gpt-6.1-sol": GPT_MODEL}
CODES          = ["met", "not_addressed", "contradicted"]
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5

I2_TO_I1 = {i2: i1 for i1, (i2, _) in MAPPING.items()}

SYSTEM_PROMPT = ("You are a research assistant coding AI governance texts. "
                 "Respond ONLY with valid JSON. No preamble, no markdown fences.")

USER_TEMPLATE = """Code an answer to a governance scenario against a list of criteria for {label}.

SCENARIO:
<<<
{scenario}
>>>

ANSWER:
<<<
{answer}
>>>

CRITERIA:
{criteria}

For each criterion, give one code:
Met: the answer applies the requirement to the case (names it as violated, required, or part of the remedy). Not addressed: the answer doesn't engage with it. Contradicted: the answer proposes or accepts something the requirement rules out. One passage may satisfy several criteria.

For "met" and "contradicted", give "evidence": a quote from the answer copied exactly, character for character (same spelling, capitalization, punctuation and markdown), as one contiguous span with no ellipses or added words. Keep it short: the passage that shows the code. For "not_addressed", give "evidence": "".

Code every criterion, once each, by its number. Return JSON in exactly this form:
{{"codes": [{{"n": 1, "code": "met" | "not_addressed" | "contradicted", "evidence": "..."}}]}}"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Coders: each returns (raw text, call metadata); raises on failure
# =============================================================================

_anthropic_client = None
_openai_client = None


def code_claude(prompt: str) -> tuple[str, dict]:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = _anthropic_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
        system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}],
        # anthropic 1.x dropped the temperature keyword; Sonnet 4.5 still accepts it
        extra_body={"temperature": 0})
    meta = {"model_version": resp.model, "finish_reason": resp.stop_reason,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens}
    return resp.content[0].text.strip(), meta


def code_gpt(prompt: str) -> tuple[str, dict]:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    limit = GPT_ANSWER_TOKENS + GPT_REASONING_HEADROOM
    resp = _openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": prompt}],
        max_completion_tokens=limit,     # provider defaults otherwise
    )
    choice  = resp.choices[0]
    usage   = resp.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    meta = {"model_version": resp.model, "finish_reason": choice.finish_reason,
            "max_completion_tokens": limit,
            "input_tokens": usage.prompt_tokens if usage else None,
            # OpenAI counts reasoning inside completion_tokens
            "output_tokens": usage.completion_tokens if usage else None,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None)}
    if not choice.message.content:
        raise RuntimeError(f"empty content (finish_reason={choice.finish_reason})")
    return choice.message.content.strip(), meta


CODER_CALLS = {"claude-sonnet": code_claude, "gpt-6.1-sol": code_gpt}


def call_coder(coder: str, prompt: str) -> tuple[str | None, dict]:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            raw, meta = CODER_CALLS[coder](prompt)
            return raw, {**meta, "attempt": attempt, "timestamp": utc_now()}
        except Exception as e:
            print(f"    [{coder} attempt {attempt}/{RETRY_ATTEMPTS}] failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                sleep(RETRY_DELAY)
    return None, {"error": "all attempts failed", "timestamp": utc_now()}


# =============================================================================
# Parsing and checks
# =============================================================================

def parse(raw: str, ns: list[int]) -> tuple[dict | None, list]:
    """Parsed codes in criterion order, plus flags for missing / extra / invalid entries."""
    try:
        obj = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None, ["parse_failed"]
    items = obj.get("codes") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return None, ["parse_failed"]
    by_n, flags = {}, []
    for it in items:
        if not (isinstance(it, dict) and isinstance(it.get("n"), int)
                and it.get("code") in CODES and isinstance(it.get("evidence", ""), str)):
            flags.append("invalid_entry")
            continue
        if it["n"] not in ns:
            flags.append(f"unknown_n:{it['n']}")
        elif it["n"] in by_n:
            flags.append(f"duplicate_n:{it['n']}")
        else:
            by_n[it["n"]] = {"n": it["n"], "code": it["code"], "evidence": it.get("evidence", "")}
    flags += [f"missing_n:{n}" for n in ns if n not in by_n]
    return {"codes": [by_n[n] for n in ns if n in by_n]}, flags


def check_quotes(parsed: dict, answer: str) -> tuple[list, list]:
    norm_ans = normalize(answer)
    checks, flags = [], []
    for c in parsed["codes"]:
        if c["code"] == "not_addressed":
            continue
        q = c["evidence"]
        exact = bool(q.strip()) and q in answer
        normed = bool(q.strip()) and normalize(q) in norm_ans
        flag = ("no_evidence" if not q.strip() else "" if exact
                else "typography" if normed else "not_found")
        checks.append({"n": c["n"], "exact": exact, "normalized": normed, "flag": flag})
        if flag:
            flags.append(f"{flag}:{c['n']}")
    return checks, flags


# =============================================================================
# Main
# =============================================================================

def build_items() -> list[dict]:
    instruments = load(INSTRUMENTS_FILE)
    scenarios = {q["id"]: q["text"] for q in instruments["instruments"]["instrument_2"]["questions"]}
    crit = load(CRITERIA_FILE)["criteria"]
    i2, recollect = load(I2_FILE), load(RECOLLECT_FILE)
    items = []
    for m in MODELS:
        src = recollect if m == "mistral-large" else i2
        for c in CONDITIONS:
            for r in RUNS:
                for s in scenarios:
                    q = I2_TO_I1[s]
                    criteria = crit[m][c][r][q]["parsed"]["criteria"]
                    answer = get_answer_text(src[m][c][r][s])
                    prompt = USER_TEMPLATE.format(
                        label=LABEL[MAPPING[q][1]], scenario=scenarios[s], answer=answer,
                        criteria="\n".join(f"{k['n']}. {k['criterion']}" for k in criteria))
                    items.append({"key": (m, c, r, s), "answer_id": f"{m}:{c}:{r}:{s}",
                                  "criteria_cell_id": f"{m}:{c}:{r}:{q}",
                                  "ns": [k["n"] for k in criteria],
                                  "answer": answer, "prompt": prompt})
    return items


def stored(out: dict, coder: str, key: tuple) -> dict:
    m, c, r, s = key
    return out["codes"].get(coder, {}).get(m, {}).get(c, {}).get(r, {}).get(s, {})


def run_coder(coder: str, todo: list[dict], out: dict, lock: threading.Lock) -> None:
    for k, it in enumerate(todo, 1):
        raw, meta = call_coder(coder, it["prompt"])
        flags, parsed, checks = [], None, []
        if raw is None:
            flags.append("call_failed")
        else:
            parsed, pflags = parse(raw, it["ns"])
            flags += pflags
            if parsed:
                checks, qflags = check_quotes(parsed, it["answer"])
                flags += qflags
        if meta.get("finish_reason") in ("max_tokens", "length"):
            flags.append("truncated")
        m, c, r, s = it["key"]
        with lock:
            (out["codes"].setdefault(coder, {}).setdefault(m, {}).setdefault(c, {})
             .setdefault(r, {}))[s] = {
                "answer_id": it["answer_id"], "criteria_cell_id": it["criteria_cell_id"],
                "raw": raw, "parsed": parsed, "quote_check": checks, "flags": flags,
                "meta": meta}
            OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{coder} {k}/{len(todo)}] {it['answer_id']}"
              + (f"  FLAGS {flags}" if flags else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = build_items()
    out = load(OUT_FILE) if OUT_FILE.exists() else {"meta": {}, "codes": {}}
    todo = {coder: [it for it in items if not stored(out, coder, it["key"]).get("parsed")]
            for coder in CODERS}

    n_crit = sum(len(it["ns"]) for it in items)
    print(f"{len(items)} answers, {n_crit} criterion codes per coder "
          f"({n_crit / len(items):.2f} per answer)")
    for coder, t in todo.items():
        print(f"  {coder:14s} {CODERS[coder]:28s} {len(t)} calls to make")
    print(f"Total: {sum(map(len, todo.values()))} calls")
    longest = max(items, key=lambda it: len(it["prompt"]))
    print(f"Longest prompt: {len(longest['prompt']):,} chars ({longest['answer_id']})")
    if args.dry_run:
        raise SystemExit(0)

    out["meta"] = {"criteria_file": str(CRITERIA_FILE), "answers": [str(I2_FILE), str(RECOLLECT_FILE)],
                   "coders": {"claude-sonnet": {"model": CLAUDE_MODEL, "temperature": 0,
                                                "max_tokens": CLAUDE_MAX_TOKENS},
                              "gpt-6.1-sol": {"model": GPT_MODEL, "params": "provider defaults",
                                              "max_completion_tokens":
                                                  GPT_ANSWER_TOKENS + GPT_REASONING_HEADROOM}},
                   "system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE,
                   "blind_to": ["model", "condition"], "updated": utc_now()}

    lock = threading.Lock()
    threads = [threading.Thread(target=run_coder, args=(coder, t, out, lock))
               for coder, t in todo.items() if t]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"Saved {OUT_FILE}")
