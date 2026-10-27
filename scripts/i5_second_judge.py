"""
Second source judge for I5: re-tier the already-extracted sources with gpt-6.1-sol.

Judge only, no re-extraction. One call per I5 response (6 models x 2 conditions x
3 runs x 6 questions = 216), all of its sources at once, JSON output. The judge
is not told which model wrote the response or the condition.

Input per call, matching what the original extractor (build_i5_prompt in
collect_llm_responses.py) saw: the question ID and the plain answer text
(get_answer_text) cut to its first 3,000 characters. Per source the judge sees
name, jurisdiction and quote only; the extractor's type, legitimacy_tier and
verifiable are withheld (type values such as "unverifiable" nearly fix the tier).
Tier and verifiable definitions are the extractor's, word for word.

Judge: gpt-6.1-sol, provider defaults; max_completion_tokens 4096 + 16384
reasoning headroom.

Output: data/processed/i5_second_judge.json
  {"meta": {...},
   "judgments": {model: {condition: {run: {question_id: {
       "raw", "parsed": {"sources": [{i, legitimacy_tier, verifiable}]},
       "n_sources", "flags": [...], "meta": {...}}}}}}}
  i is the 1-based position of the source in instrument_5.json's sources list.
Re-running fills in only cells without a parsed result. Reads data/raw/ only.

Usage:
    python scripts/i5_second_judge.py --dry-run   # call counts, no API calls
    python scripts/i5_second_judge.py             # real run
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import openai
from dotenv import load_dotenv

from answer_text import get_answer_text, _strip_fences

load_dotenv(".env")

I1_FILE  = Path("data/raw/instrument_1.json")
I2_FILE  = Path("data/raw/instrument_2.json")
I5_FILE  = Path("data/raw/instrument_5.json")
OUT_FILE = Path("data/processed/i5_second_judge.json")

JUDGE_MODEL       = "gpt-6.1-sol"
ANSWER_TOKENS     = 4096
REASONING_HEADROOM = 16384
TEXT_CHARS        = 3000      # as in build_i5_prompt
WORKERS           = 4
RETRY_ATTEMPTS    = 3
RETRY_DELAY       = 5

MODELS     = ["gpt-4o", "claude-sonnet", "deepseek-v3", "mistral-large",
              "gemini-3.1-flash-lite-preview", "grok-4.7"]
CONDITIONS = ["baseline", "ceo"]
RUNS       = ["1", "2", "3"]
QUESTIONS  = ["I1_Q1", "I1_Q2", "I1_Q3", "I2_S1", "I2_S2", "I2_S3"]

# Word for word from build_i5_prompt
TIER_DEF       = ("- legitimacy_tier: 1=verifiable primary legal source, 2=verifiable secondary source, "
                  "3=plausible but unverifiable, 4=vague or fabricated")
VERIFIABLE_DEF = "- verifiable: true if you can confirm this source exists, false otherwise"

SYSTEM_PROMPT = ("You are a research assistant assessing the sources cited in AI governance "
                 "responses. Respond ONLY with valid JSON. No preamble, no markdown fences.")

USER_TEMPLATE = """The following is a response to AI governance question {question_id}.

RESPONSE:
{response}

The response was found to cite, reference, imply, or draw upon the sources below.

SOURCES:
{sources}

For each source, provide:
{tier_def}
{verifiable_def}

Rate every source, once each, by its number. Respond ONLY with a JSON object. No preamble, no markdown.
Format: {{"sources": [{{"i": 1, "legitimacy_tier": 1, "verifiable": true}}]}}"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_line(i: int, s: dict) -> str:
    quote = s.get("quote")
    return (f"{i}. name: {s.get('name', '')}; jurisdiction: {s.get('jurisdiction', '')}; "
            f"quote: {json.dumps(quote, ensure_ascii=False) if quote else 'null'}")


def build_items() -> list[dict]:
    i1, i2, i5 = load(I1_FILE), load(I2_FILE), load(I5_FILE)
    items = []
    for m in MODELS:
        for c in CONDITIONS:
            for r in RUNS:
                for q in QUESTIONS:
                    src = (i1 if q.startswith("I1") else i2)[m][c][r][q]
                    sources = i5[m][c][r][q]["parsed"]["sources"]
                    prompt = USER_TEMPLATE.format(
                        question_id=q, response=get_answer_text(src)[:TEXT_CHARS],
                        sources="\n".join(source_line(k, s) for k, s in enumerate(sources, 1)),
                        tier_def=TIER_DEF, verifiable_def=VERIFIABLE_DEF)
                    items.append({"key": (m, c, r, q), "n_sources": len(sources), "prompt": prompt})
    return items


_client = None


def call_judge(prompt: str) -> tuple[str | None, dict]:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    limit = ANSWER_TOKENS + REASONING_HEADROOM
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = _client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user",   "content": prompt}],
                max_completion_tokens=limit,      # provider defaults otherwise
            )
            choice  = resp.choices[0]
            usage   = resp.usage
            details = getattr(usage, "completion_tokens_details", None) if usage else None
            meta = {"model_version": resp.model, "finish_reason": choice.finish_reason,
                    "max_completion_tokens": limit,
                    "input_tokens": usage.prompt_tokens if usage else None,
                    # OpenAI counts reasoning inside completion_tokens
                    "output_tokens": usage.completion_tokens if usage else None,
                    "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                    "attempt": attempt, "timestamp": utc_now()}
            if not choice.message.content:
                raise RuntimeError(f"empty content (finish_reason={choice.finish_reason})")
            return choice.message.content.strip(), meta
        except Exception as e:
            print(f"    [attempt {attempt}/{RETRY_ATTEMPTS}] failed: {e}", flush=True)
            if attempt < RETRY_ATTEMPTS:
                sleep(RETRY_DELAY)
    return None, {"error": "all attempts failed", "timestamp": utc_now()}


def parse(raw: str, n: int) -> tuple[dict | None, list]:
    try:
        obj = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None, ["parse_failed"]
    items = obj.get("sources") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return None, ["parse_failed"]
    by_i, flags = {}, []
    for it in items:
        if not (isinstance(it, dict) and isinstance(it.get("i"), int)
                and it.get("legitimacy_tier") in (1, 2, 3, 4)
                and isinstance(it.get("verifiable"), bool)):
            flags.append("invalid_entry")
            continue
        if not 1 <= it["i"] <= n:
            flags.append(f"unknown_i:{it['i']}")
        elif it["i"] in by_i:
            flags.append(f"duplicate_i:{it['i']}")
        else:
            by_i[it["i"]] = {"i": it["i"], "legitimacy_tier": it["legitimacy_tier"],
                             "verifiable": it["verifiable"]}
    flags += [f"missing_i:{i}" for i in range(1, n + 1) if i not in by_i]
    return {"sources": [by_i[i] for i in sorted(by_i)]}, flags


def stored(out: dict, key: tuple) -> dict:
    m, c, r, q = key
    return out["judgments"].get(m, {}).get(c, {}).get(r, {}).get(q, {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = build_items()
    out = load(OUT_FILE) if OUT_FILE.exists() else {"meta": {}, "judgments": {}}
    todo = [it for it in items if not stored(out, it["key"]).get("parsed")]
    print(f"{len(items)} responses, {sum(it['n_sources'] for it in items)} sources "
          f"(min {min(it['n_sources'] for it in items)}, max {max(it['n_sources'] for it in items)} per response)")
    print(f"{len(todo)} calls to make ({JUDGE_MODEL})")
    longest = max(items, key=lambda it: len(it["prompt"]))
    print(f"Longest prompt: {len(longest['prompt']):,} chars ({':'.join(longest['key'])})")
    if args.dry_run:
        raise SystemExit(0)

    out["meta"] = {"judge_model": JUDGE_MODEL, "params": "provider defaults",
                   "max_completion_tokens": ANSWER_TOKENS + REASONING_HEADROOM,
                   "system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE,
                   "tier_def": TIER_DEF, "verifiable_def": VERIFIABLE_DEF,
                   "response_text": f"get_answer_text, first {TEXT_CHARS} chars",
                   "source_fields_shown": ["name", "jurisdiction", "quote"],
                   "blind_to": ["model", "condition"], "updated": utc_now()}
    lock = threading.Lock()
    done = [0]

    def work(it: dict) -> None:
        raw, meta = call_judge(it["prompt"])
        flags, parsed = [], None
        if raw is None:
            flags.append("call_failed")
        else:
            parsed, pflags = parse(raw, it["n_sources"])
            flags += pflags
        if meta.get("finish_reason") == "length":
            flags.append("truncated")
        m, c, r, q = it["key"]
        with lock:
            (out["judgments"].setdefault(m, {}).setdefault(c, {}).setdefault(r, {}))[q] = {
                "raw": raw, "parsed": parsed, "n_sources": it["n_sources"],
                "flags": flags, "meta": meta}
            OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            done[0] += 1
            print(f"[{done[0]}/{len(todo)}] {m} {c} r{r} {q}"
                  + (f"  FLAGS {flags}" if flags else ""), flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, todo))
    print(f"Saved {OUT_FILE}")
