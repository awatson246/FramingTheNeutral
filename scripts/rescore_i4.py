"""
Instrument 4 re-scoring and Grok pairs, with pinned evaluator model IDs.

Uses the collection script's own I4 pieces unchanged: build_i4_prompt (plain
answer text via get_answer_text), parse_i4_response, call_with_retry, the
baseline system prompt and MAX_TOKENS_JSON. Evaluatee answers are run 1 baseline.

Outputs (never writes instrument_4.json or any other data file):
  data/raw/instrument_4_rescore.json  P1-P5 (peer_eval_pairs.json), the three I1
                                      questions re-scored on clean text; plus, where
                                      mistral-large is the evaluatee, its I2 answers
  data/raw/instrument_4_grok.json     G1-G10 (peer_eval_pairs_grok.json), all six questions

Both files use instrument_4.json's structure, {pair_id: {question_id: {raw, parsed}}},
and each cell adds "meta": evaluator model ID, returned version, provider,
finish_reason, token counts, timestamp.

Mistral I2 answers as evaluatee are scored twice, stored separately:
  <pair_id>                           original instrument_2.json answer (truncated at 1,024)
  <pair_id>__mistral_i2_recollect     data/raw/mistral_i2_recollect.json answer
Re-running fills in only cells without a stored output.

Usage:
    python scripts/rescore_i4.py --dry-run          # call counts, no API calls
    python scripts/rescore_i4.py --smoke-openrouter # one tiny DeepSeek call via OpenRouter
    python scripts/rescore_i4.py                    # real run
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from time import sleep

import openai

import collect_llm_responses as clr
from answer_text import get_answer_text

PAIRS_FILE      = Path("data/prompts/peer_eval_pairs.json")
GROK_PAIRS_FILE = Path("data/prompts/peer_eval_pairs_grok.json")
RECOLLECT_FILE  = clr.RAW_DIR / "mistral_i2_recollect.json"
RESCORE_FILE    = clr.RAW_DIR / "instrument_4_rescore.json"
GROK_FILE       = clr.RAW_DIR / "instrument_4_grok.json"
PROTECTED       = clr.RAW_DIR / "instrument_4.json"

I1_QUESTIONS = ["I1_Q1", "I1_Q2", "I1_Q3"]
I2_QUESTIONS = ["I2_S1", "I2_S2", "I2_S3"]
RECOLLECT_SUFFIX = "__mistral_i2_recollect"

# Pinned evaluator model IDs, keyed by the model keys used in the data files
EVALUATOR_MODELS = {
    "claude-sonnet":                 "claude-sonnet-4-5-20250929",
    "gpt-4o":                        "gpt-4o-2024-08-06",
    "mistral-large":                 "mistral-large-2512",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
    "deepseek-v3":                   "deepseek/deepseek-v3.2",   # via OpenRouter
    "grok-4.7":                      "grok-4.7",
}

# =============================================================================
# Pinned evaluator callers
# Same parameters as the original callers in collect_llm_responses.py; only the
# model ID differs. Each records call metadata in clr.LAST_CALL_META.
# =============================================================================

_openrouter_client = None


def get_openrouter_client() -> openai.OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = openai.OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                                           base_url="https://openrouter.ai/api/v1")
    return _openrouter_client


def _record_openai_style(resp, **extra) -> str:
    """Shared metadata capture for OpenAI-compatible responses; raises on empty output."""
    choice  = resp.choices[0]
    usage   = resp.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    clr.LAST_CALL_META.clear()
    clr.LAST_CALL_META.update({
        "model_version":    resp.model,
        "finish_reason":    choice.finish_reason,
        "input_tokens":     usage.prompt_tokens if usage else None,
        "output_tokens":    usage.completion_tokens if usage else None,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        **extra,
    })
    if not choice.message.content:
        # Like the original callers (.strip() on None raises): lets call_with_retry retry
        raise RuntimeError(f"empty content (finish_reason={choice.finish_reason})")
    return choice.message.content.strip()


def _messages(system_prompt: str, user_prompt: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}]


def eval_gpt4o(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    resp = clr.openai_client.chat.completions.create(
        model=EVALUATOR_MODELS["gpt-4o"],
        messages=_messages(system_prompt, user_prompt),
        max_tokens=max_tokens,
        temperature=1,          # as in clr.call_openai
    )
    return _record_openai_style(resp)


def eval_deepseek_openrouter(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    resp = get_openrouter_client().chat.completions.create(
        model=EVALUATOR_MODELS["deepseek-v3"],
        messages=_messages(system_prompt, user_prompt),
        max_tokens=max_tokens,
        # Non-thinking mode, matching how deepseek-chat behaved
        extra_body={"reasoning": {"enabled": False}},
    )
    return _record_openai_style(resp, provider=(resp.model_extra or {}).get("provider"))


def eval_claude(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    resp = clr.anthropic_client.messages.create(
        model=EVALUATOR_MODELS["claude-sonnet"],
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    clr.LAST_CALL_META.clear()
    clr.LAST_CALL_META.update({
        "model_version": resp.model,
        "finish_reason": resp.stop_reason,
        "input_tokens":  resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    })
    return resp.content[0].text.strip()


def eval_mistral(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    resp = clr.mistral_client.chat.complete(
        model=EVALUATOR_MODELS["mistral-large"],
        messages=_messages(system_prompt, user_prompt),
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    clr.LAST_CALL_META.clear()
    clr.LAST_CALL_META.update({
        "model_version": resp.model,
        "finish_reason": str(choice.finish_reason),
        "input_tokens":  resp.usage.prompt_tokens if resp.usage else None,
        "output_tokens": resp.usage.completion_tokens if resp.usage else None,
    })
    return choice.message.content.strip()


def eval_gemini(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    # As in clr.call_gemini: no output limit is passed
    model = clr.genai.GenerativeModel(
        model_name=EVALUATOR_MODELS["gemini-3.1-flash-lite-preview"],
        system_instruction=system_prompt,
    )
    resp  = model.generate_content(user_prompt)
    usage = getattr(resp, "usage_metadata", None)
    clr.LAST_CALL_META.clear()
    clr.LAST_CALL_META.update({
        "model_version": (getattr(resp, "model_version", None)
                          or getattr(getattr(resp, "_result", None), "model_version", None)),
        "finish_reason": resp.candidates[0].finish_reason.name if resp.candidates else None,
        "input_tokens":  getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "reasoning_tokens": getattr(usage, "thoughts_token_count", None),
    })
    return resp.text.strip()


EVALUATOR_CALLERS = {
    "claude-sonnet":                 eval_claude,
    "gpt-4o":                        eval_gpt4o,
    "mistral-large":                 eval_mistral,
    "gemini-3.1-flash-lite-preview": eval_gemini,
    "deepseek-v3":                   eval_deepseek_openrouter,
    "grok-4.7":                      clr.call_xai,   # already grok-4.7, with reasoning headroom
}

# =============================================================================
# Jobs
# =============================================================================

def load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_output(path: Path, data: dict) -> None:
    assert path in (RESCORE_FILE, GROK_FILE) and path != PROTECTED
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_jobs() -> list[dict]:
    """Every (output file, pair key, evaluator, evaluatee, question, answer source) to score."""
    jobs = []

    def add(out, pair, q_ids):
        for q in q_ids:
            jobs.append(dict(out=out, pair_key=pair["pair_id"], evaluator=pair["evaluator"],
                             evaluatee=pair["evaluatee"], q_id=q, source="original"))
            if pair["evaluatee"] == "mistral-large" and q in I2_QUESTIONS:
                jobs.append(dict(out=out, pair_key=pair["pair_id"] + RECOLLECT_SUFFIX,
                                 evaluator=pair["evaluator"], evaluatee=pair["evaluatee"],
                                 q_id=q, source="recollect"))

    for pair in load_json(PAIRS_FILE, {})["pairs"]:
        q_ids = list(I1_QUESTIONS)
        if pair["evaluatee"] == "mistral-large":
            q_ids += I2_QUESTIONS      # original + re-collected Mistral I2, for comparison
        add(RESCORE_FILE, pair, q_ids)

    for pair in load_json(GROK_PAIRS_FILE, {})["pairs"]:
        add(GROK_FILE, pair, pair["source_questions"])
    return jobs


def evaluatee_text(job: dict, i1: dict, i2: dict, recollect: dict) -> str:
    q = job["q_id"]
    src = recollect if job["source"] == "recollect" else (i1 if q.startswith("I1") else i2)
    cell = src.get(job["evaluatee"], {}).get("baseline", {}).get("1", {}).get(q)
    return get_answer_text(cell) if cell is not None else ""


def is_done(outputs: dict, job: dict) -> bool:
    cell = outputs[job["out"]].get(job["pair_key"], {}).get(job["q_id"])
    return isinstance(cell, dict) and cell.get("raw") is not None


def dry_run(jobs: list[dict], outputs: dict) -> None:
    todo = [j for j in jobs if not is_done(outputs, j)]
    print("DRY RUN - no API calls made.\n")
    print(f"{'Evaluator':<32} {'Pinned model':<30} {'Rescore':>8} {'Grok':>6} {'Total':>6}")
    print("-" * 86)
    for ev, model in EVALUATOR_MODELS.items():
        r = sum(1 for j in todo if j["evaluator"] == ev and j["out"] == RESCORE_FILE)
        g = sum(1 for j in todo if j["evaluator"] == ev and j["out"] == GROK_FILE)
        print(f"{ev:<32} {model:<30} {r:>8} {g:>6} {r + g:>6}")
    print("-" * 86)
    r = sum(j["out"] == RESCORE_FILE for j in todo)
    print(f"{'TOTAL':<63} {r:>8} {len(todo) - r:>6} {len(todo):>6}")
    print(f"\nOf which Mistral re-collected I2 (stored under *{RECOLLECT_SUFFIX}): "
          f"{sum(j['source'] == 'recollect' for j in todo)}")
    print(f"Already done (skipped): {len(jobs) - len(todo)}")


def smoke_openrouter() -> None:
    text = eval_deepseek_openrouter("Reply briefly.", "Say OK.", 16)
    m = clr.LAST_CALL_META
    print(f"text={text!r}\nreturned model={m.get('model_version')} | provider={m.get('provider')} | "
          f"finish={m.get('finish_reason')} | reasoning_tokens={m.get('reasoning_tokens')} | "
          f"in/out={m.get('input_tokens')}/{m.get('output_tokens')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I4 re-scoring and Grok pairs with pinned evaluators.")
    parser.add_argument("--dry-run", action="store_true", help="Print call counts; no API calls.")
    parser.add_argument("--smoke-openrouter", action="store_true",
                        help="One tiny call to deepseek/deepseek-v3.2 via OpenRouter.")
    args = parser.parse_args()

    if args.smoke_openrouter:
        smoke_openrouter()
        raise SystemExit(0)

    jobs    = build_jobs()
    outputs = {RESCORE_FILE: load_json(RESCORE_FILE, {}), GROK_FILE: load_json(GROK_FILE, {})}

    if args.dry_run:
        dry_run(jobs, outputs)
        raise SystemExit(0)

    # Route evaluator calls through the collection script's retry wrapper
    for ev, fn in EVALUATOR_CALLERS.items():
        clr.MODEL_CALLERS[f"eval:{ev}"] = fn

    with open(clr.INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
        instruments_data = json.load(f)
    instrument    = instruments_data["instruments"]["instrument_4"]
    system_prompt = instruments_data["conditions"]["baseline"]["system_prompt"]
    i1, i2    = clr.load_instrument("instrument_1"), clr.load_instrument("instrument_2")
    recollect = load_json(RECOLLECT_FILE, {})

    counts = Counter()
    for job in jobs:
        label = f"{job['out'].name} | {job['pair_key']} | {job['q_id']}"
        if is_done(outputs, job):
            counts["skipped"] += 1
            continue
        answer = evaluatee_text(job, i1, i2, recollect)
        if not answer:
            print(f"  [skip] {label} — evaluatee answer not found")
            counts["no_answer"] += 1
            continue

        prompt = clr.build_i4_prompt(job["q_id"], clr.ALL_QUESTION_TEXT[job["q_id"]],
                                     answer, instrument)
        print(f"  [call] {label} ({job['evaluator']}) ... ", end="", flush=True)
        clr.LAST_CALL_META.clear()
        raw    = clr.call_with_retry(f"eval:{job['evaluator']}", system_prompt, prompt,
                                     clr.MAX_TOKENS_JSON)
        parsed = clr.parse_i4_response(raw) if raw else None

        meta = {"evaluator": job["evaluator"],
                "evaluator_model": EVALUATOR_MODELS[job["evaluator"]],
                "evaluatee": job["evaluatee"],
                "evaluatee_answer": ("mistral_i2_recollect.json" if job["source"] == "recollect"
                                     else ("instrument_1.json" if job["q_id"].startswith("I1")
                                           else "instrument_2.json")) + " run 1 baseline",
                "timestamp": clr.utc_now(),
                **clr.LAST_CALL_META}
        data = outputs[job["out"]]
        data.setdefault(job["pair_key"], {})[job["q_id"]] = {"raw": raw, "parsed": parsed, "meta": meta}
        save_output(job["out"], data)

        status = "ok" if parsed else ("raw only" if raw else "FAILED")
        counts[status] += 1
        print(f"{status} (version={meta.get('model_version')}, finish={meta.get('finish_reason')})")
        sleep(clr.CALL_DELAY)

    print(f"\nDone: {dict(counts)}")
    print(f"Saved {RESCORE_FILE} and {GROK_FILE}")
