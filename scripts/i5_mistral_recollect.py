"""
I5 on Mistral Large's re-collected I2 answers (data/raw/mistral_i2_recollect.json):
18 answers (2 conditions x 3 runs x I2_S1-S3).

Step 1, extractor: exactly as the original I5 collection in collect_llm_responses.py
(build_i5_prompt with its 3,000-character cut, call_i5_extractor, parse_i5_response;
claude-sonnet-4-5-20250929, provider default temperature, max_tokens 2048).
Step 2, second judge: exactly as i5_second_judge.py (gpt-6.1-sol, provider
defaults, same prompt; name, jurisdiction and quote of each extracted source).

Outputs (instrument_5.json and i5_second_judge.json are never written):
  data/processed/i5_mistral_i2_recollect.json
      {"meta", "extractions": {"mistral-large": {condition: {run: {I2_Sx: {raw, parsed, meta}}}}}}
  data/processed/i5_second_judge_mistral_i2_recollect.json
      {"meta", "judgments": {"mistral-large": {condition: {run: {I2_Sx: {...as i5_second_judge}}}}}}
data_sources.load_i5 / load_i5_judge merge these in for MISTRAL_I2_SOURCE=recollect.
Re-running fills in only cells without a parsed result.

Usage:
    python scripts/i5_mistral_recollect.py --dry-run   # call counts, no API calls
    python scripts/i5_mistral_recollect.py             # real run (extractor, then judge)
"""

import argparse
import json
from datetime import datetime, timezone

import collect_llm_responses as clr
import i5_second_judge as sj
from answer_text import get_answer_text
from data_sources import (I2_QS, I5_RECOLLECT_FILE, JUDGE_RECOLLECT_FILE, MISTRAL,
                          RECOLLECT_FILE)

CONDITIONS = ["baseline", "ceo"]
RUNS       = ["1", "2", "3"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_or(path, default: dict) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def cells() -> list[tuple]:
    return [(c, r, q) for c in CONDITIONS for r in RUNS for q in I2_QS]


def get(d: dict, c: str, r: str, q: str) -> dict:
    return d.get(MISTRAL, {}).get(c, {}).get(r, {}).get(q, {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    answers = json.loads(RECOLLECT_FILE.read_text(encoding="utf-8"))[MISTRAL]
    texts = {(c, r, q): get_answer_text(answers[c][r][q]) for c, r, q in cells()}
    ext = load_or(I5_RECOLLECT_FILE, {"meta": {}, "extractions": {}})
    jud = load_or(JUDGE_RECOLLECT_FILE, {"meta": {}, "judgments": {}})
    todo_ext = [k for k in cells() if not get(ext["extractions"], *k).get("parsed")]
    todo_jud = [k for k in cells() if not get(jud["judgments"], *k).get("parsed")]

    lens = [len(t) for t in texts.values()]
    print(f"{len(texts)} re-collected Mistral I2 answers; {sum(l > sj.TEXT_CHARS for l in lens)} "
          f"longer than {sj.TEXT_CHARS} chars (cut as in build_i5_prompt); "
          f"length min {min(lens)}, max {max(lens)}")
    print(f"Extractor ({clr.EXTRACTOR_MODEL}): {len(todo_ext)} calls")
    print(f"Second judge ({sj.JUDGE_MODEL}):   {len(todo_jud)} calls "
          f"(after the extractor; one per answer with all its sources)")
    print(f"Total: {len(todo_ext) + len(todo_jud)} calls")
    if args.dry_run:
        raise SystemExit(0)

    # ---- step 1: extractor ----
    ext["meta"] = {"extractor_model": clr.EXTRACTOR_MODEL, "prompt": "collect_llm_responses.build_i5_prompt",
                   "caller": "collect_llm_responses.call_i5_extractor", "answers": str(RECOLLECT_FILE),
                   "updated": utc_now()}
    for k, (c, r, q) in enumerate(todo_ext, 1):
        raw = clr.call_i5_extractor(clr.build_i5_prompt(q, texts[(c, r, q)]))
        meta = {**clr.LAST_CALL_META, "timestamp": utc_now()}
        clr.LAST_CALL_META.clear()
        parsed = clr.parse_i5_response(raw) if raw else None
        ext["extractions"].setdefault(MISTRAL, {}).setdefault(c, {}).setdefault(r, {})[q] = {
            "raw": raw, "parsed": parsed, "meta": meta}
        save(I5_RECOLLECT_FILE, ext)
        n = len(parsed["sources"]) if parsed else "-"
        print(f"[extract {k}/{len(todo_ext)}] {c} r{r} {q}: {n} sources"
              + ("" if parsed else "  FAILED" if raw is None else "  PARSE FAILED"), flush=True)

    # ---- step 2: second judge ----
    jud["meta"] = {"judge_model": sj.JUDGE_MODEL, "params": "provider defaults",
                   "max_completion_tokens": sj.ANSWER_TOKENS + sj.REASONING_HEADROOM,
                   "system_prompt": sj.SYSTEM_PROMPT, "user_template": sj.USER_TEMPLATE,
                   "tier_def": sj.TIER_DEF, "verifiable_def": sj.VERIFIABLE_DEF,
                   "response_text": f"get_answer_text, first {sj.TEXT_CHARS} chars",
                   "source_fields_shown": ["name", "jurisdiction", "quote"],
                   "extractions": str(I5_RECOLLECT_FILE), "blind_to": ["model", "condition"],
                   "updated": utc_now()}
    for k, (c, r, q) in enumerate(todo_jud, 1):
        cell = get(ext["extractions"], c, r, q)
        if not cell.get("parsed"):
            print(f"[judge {k}/{len(todo_jud)}] {c} r{r} {q}: no extraction, skipped")
            continue
        sources = cell["parsed"]["sources"]
        prompt = sj.USER_TEMPLATE.format(
            question_id=q, response=texts[(c, r, q)][:sj.TEXT_CHARS],
            sources="\n".join(sj.source_line(i, s) for i, s in enumerate(sources, 1)),
            tier_def=sj.TIER_DEF, verifiable_def=sj.VERIFIABLE_DEF)
        raw, meta = sj.call_judge(prompt)
        flags, parsed = [], None
        if raw is None:
            flags.append("call_failed")
        else:
            parsed, pflags = sj.parse(raw, len(sources))
            flags += pflags
        if meta.get("finish_reason") == "length":
            flags.append("truncated")
        jud["judgments"].setdefault(MISTRAL, {}).setdefault(c, {}).setdefault(r, {})[q] = {
            "raw": raw, "parsed": parsed, "n_sources": len(sources), "flags": flags, "meta": meta}
        save(JUDGE_RECOLLECT_FILE, jud)
        print(f"[judge {k}/{len(todo_jud)}] {c} r{r} {q}" + (f"  FLAGS {flags}" if flags else ""),
              flush=True)
    print(f"Saved {I5_RECOLLECT_FILE} and {JUDGE_RECOLLECT_FILE}")
