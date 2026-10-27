"""
I2 criterion coding, step 1: extract each model's stated criteria from its I1 definitions.

For every I1 cell (model x condition x run x question) the extractor lists the
requirements the definition states for that question's dimension: at most 15 short
criteria, each with the verbatim phrase from the definition it comes from. Answer
text only (get_answer_text), no sources.

Extraction rules (v2, from the manual review in data/processed/manual_review.json):
  cap       at most 15 criteria (v1: 10)
  negation  a positive requirement implied by a negative statement ("a rule that
            cannot be proved ... remains advice") is allowed; the extractor marks
            it implied and quotes the negative wording (v1: not allowed)
  grain     one requirement per criterion: bundled requirements are split, and
            criteria that split or restate a single requirement are merged
v1 output (10 criteria, no implied) is kept in data/processed/i2_criteria_v1.json;
the hand codes in manual_review.json were made against it.

Extractor: claude-sonnet-4-5-20250929, temperature 0, JSON output.

Each verbatim phrase is checked in code against its definition:
  exact       phrase is a substring of the definition as stored
  normalized  substring after collapsing whitespace, straightening quotes/dashes
              and lowercasing
and given one flag:
  ""          exact match, not implied
  typography  matches only after normalization
  implied     extractor marked it implied and the phrase is found (exact or normalized)
  rewritten   phrase not found even after normalization

Mapping for the later I2 coding, from the tripod_dimension of each question in
data/prompts/instruments.json (checked at start-up):
  I1_Q1 -> I2_S1 (legal certainty)
  I1_Q2 -> I2_S2 (accountability)
  I1_Q3 -> I2_S3 (enforceability)

Output: data/processed/i2_criteria.json
  {"meta": {...}, "mapping": {...},
   "criteria": {model: {condition: {run: {I1_Qx: {
       "i2_question", "dimension", "raw",
       "parsed": {"criteria": [{n, criterion, phrase, implied}]},
       "phrase_check": [{n, exact, normalized, flag}], "flags": [...], "meta": {...}}}}}}}
Re-running fills in only cells without a parsed result, and refuses to add to a
file written under different rules. Reads data/raw/ only.

Usage:
    python scripts/i2_criteria.py --dry-run   # cell count and definition lengths, no API calls
    python scripts/i2_criteria.py             # real run
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import sleep

import anthropic
from dotenv import load_dotenv

from answer_text import get_answer_text, _strip_fences

load_dotenv(".env")

I1_FILE          = Path("data/raw/instrument_1.json")
INSTRUMENTS_FILE = Path("data/prompts/instruments.json")
OUT_FILE         = Path("data/processed/i2_criteria.json")

EXTRACTOR_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS      = 4096   # v1: 2048; room for 15 criteria with phrases
MAX_CRITERIA    = 15
RULES_VERSION   = "v2"
RETRY_ATTEMPTS  = 3
RETRY_DELAY     = 5

MODELS     = ["gpt-4o", "claude-sonnet", "deepseek-v3", "mistral-large",
              "gemini-3.1-flash-lite-preview", "grok-4.7"]
CONDITIONS = ["baseline", "ceo"]
RUNS       = ["1", "2", "3"]
MAPPING    = {"I1_Q1": ("I2_S1", "legal_certainty"),
              "I1_Q2": ("I2_S2", "accountability"),
              "I1_Q3": ("I2_S3", "enforceability")}
LABEL      = {"legal_certainty": "legal certainty", "accountability": "accountability",
              "enforceability": "enforceability"}

SYSTEM_PROMPT = ("You are a research assistant coding AI governance texts. "
                 "Respond ONLY with valid JSON. No preamble, no markdown fences.")

USER_TEMPLATE = """Below is a definition of {label} that was written in answer to this question:

QUESTION: {question}

DEFINITION:
<<<
{definition}
>>>

List the requirements this definition states for {label}: what it says {label} requires, consists of, or needs in order to be present.

Rules:
- Only what the definition states. Add nothing and do not use outside knowledge.
- One requirement per criterion: something that must be present or true, and that a text could meet or fail on its own.
  - Split bundles: if one sentence or list item names several distinct requirements, give each its own criterion.
  - Merge splits: if two criteria could not be met independently (one requirement described in two parts), or a sentence restates requirements listed elsewhere, list that requirement once.
- Negative statements: if the definition says that without X, {label} is absent or fails (e.g. "a rule that cannot be enforced is merely aspirational"), you may list X as a requirement and set "implied": true. Quote the negative wording as the phrase. Do not turn consequences of absence (e.g. "this erodes public trust") into requirements. Every other criterion has "implied": false.
- Write each criterion as a short phrase (at most about 12 words).
- At most {max_n} criteria. If the definition states more, keep the {max_n} it treats as most central.
- Do not list examples or cited sources, unless the definition presents them as a requirement.
- For each criterion give "phrase": the words in the definition it comes from, copied exactly, character for character (same spelling, capitalization, punctuation and markdown), as one contiguous span with no ellipses or added words. Keep the phrase short: the clause that states the requirement, not the whole paragraph.

Return JSON in exactly this form:
{{"criteria": [{{"criterion": "...", "phrase": "...", "implied": false}}]}}"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_mapping(instruments: dict) -> list[str]:
    """Differences between MAPPING and the tripod_dimension fields in instruments.json."""
    qs = {i: {q["id"]: q for q in instruments["instruments"][i]["questions"]}
          for i in ("instrument_1", "instrument_2")}
    diffs = []
    for i1, (i2, dim) in MAPPING.items():
        d1 = qs["instrument_1"][i1]["tripod_dimension"]
        d2 = qs["instrument_2"][i2]["tripod_dimension"]
        if not (d1 == d2 == dim):
            diffs.append(f"{i1} ({d1}) -> {i2} ({d2}); expected {dim}")
    return diffs


def normalize(s: str) -> str:
    s = (s.replace("‘", "'").replace("’", "'").replace("“", '"')
          .replace("”", '"').replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", s).strip().lower()


def parse(raw: str) -> dict | None:
    try:
        obj = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None
    crit = obj.get("criteria") if isinstance(obj, dict) else None
    if not isinstance(crit, list) or not all(
            isinstance(c, dict) and isinstance(c.get("criterion"), str)
            and isinstance(c.get("phrase"), str) for c in crit):
        return None
    return {"criteria": [{"n": k, "criterion": c["criterion"].strip(), "phrase": c["phrase"],
                          "implied": c.get("implied") is True}
                         for k, c in enumerate(crit, 1)]}


def phrase_flag(exact: bool, normed: bool, implied: bool) -> str:
    if not normed:
        return "rewritten"
    if implied:
        return "implied"
    return "" if exact else "typography"


def check_phrases(parsed: dict, definition: str) -> tuple[list, list]:
    norm_def = normalize(definition)
    checks, flags = [], []
    for c in parsed["criteria"]:
        exact = bool(c["phrase"]) and c["phrase"] in definition
        normed = bool(c["phrase"].strip()) and normalize(c["phrase"]) in norm_def
        flag = phrase_flag(exact, normed, c.get("implied", False))
        checks.append({"n": c["n"], "exact": exact, "normalized": normed, "flag": flag})
        if flag:
            flags.append(f"{flag}:{c['n']}")
    if len(parsed["criteria"]) > MAX_CRITERIA:
        flags.append(f"too_many_criteria:{len(parsed['criteria'])}")
    if not parsed["criteria"]:
        flags.append("no_criteria")
    return checks, flags


def call_extractor(client: anthropic.Anthropic, prompt: str) -> tuple[str | None, dict]:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = client.messages.create(
                model=EXTRACTOR_MODEL, max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}],
                # anthropic 1.x dropped the temperature keyword; Sonnet 4.5 still accepts it
                extra_body={"temperature": 0})
            meta = {"model_version": resp.model, "finish_reason": resp.stop_reason,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens, "timestamp": utc_now()}
            return resp.content[0].text.strip(), meta
        except Exception as e:
            print(f"    [Attempt {attempt}/{RETRY_ATTEMPTS}] extractor failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                sleep(RETRY_DELAY)
    return None, {"error": "all attempts failed", "timestamp": utc_now()}


def load_out() -> dict:
    if OUT_FILE.exists():
        return json.loads(OUT_FILE.read_text(encoding="utf-8"))
    return {"meta": {}, "mapping": {}, "criteria": {}}


def save_out(out: dict) -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    instruments = json.loads(INSTRUMENTS_FILE.read_text(encoding="utf-8"))
    diffs = check_mapping(instruments)
    print("Mapping check:", "matches instruments.json" if not diffs else "; ".join(diffs))
    questions = {q["id"]: q["text"] for q in instruments["instruments"]["instrument_1"]["questions"]}
    i1 = json.loads(I1_FILE.read_text(encoding="utf-8"))

    cells = [(m, c, r, q) for m in MODELS for c in CONDITIONS for r in RUNS for q in MAPPING]
    texts = {cell: get_answer_text(i1[cell[0]][cell[1]][cell[2]][cell[3]]) for cell in cells}
    print(f"{len(cells)} cells; definition length (chars) "
          f"mean {mean(map(len, texts.values())):.0f}, "
          f"min {min(map(len, texts.values()))}, max {max(map(len, texts.values()))}")

    out = load_out()
    if out["criteria"] and out["meta"].get("rules_version") != RULES_VERSION:
        raise SystemExit(f"{OUT_FILE} was written under rules "
                         f"{out['meta'].get('rules_version', 'v1')}, not {RULES_VERSION}; "
                         f"move it aside before re-running.")
    todo = [cell for cell in cells
            if not (out["criteria"].get(cell[0], {}).get(cell[1], {})
                    .get(cell[2], {}).get(cell[3], {}).get("parsed"))]
    print(f"{len(todo)} cells to extract")
    if args.dry_run:
        raise SystemExit(0)

    out["meta"] = {"rules_version": RULES_VERSION,
                   "extractor_model": EXTRACTOR_MODEL, "temperature": 0,
                   "max_tokens": MAX_TOKENS, "max_criteria": MAX_CRITERIA,
                   "system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE,
                   "source": str(I1_FILE), "answer_text": "get_answer_text (no sources)",
                   "updated": utc_now()}
    out["mapping"] = {i1q: {"i2_question": i2q, "dimension": d} for i1q, (i2q, d) in MAPPING.items()}

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    for k, (m, c, r, q) in enumerate(todo, 1):
        i2q, dim = MAPPING[q]
        prompt = USER_TEMPLATE.format(label=LABEL[dim], question=questions[q],
                                      definition=texts[(m, c, r, q)], max_n=MAX_CRITERIA)
        raw, meta = call_extractor(client, prompt)
        parsed = parse(raw) if raw else None
        flags = []
        if raw is None:
            flags.append("call_failed")
        elif parsed is None:
            flags.append("parse_failed")
        if meta.get("finish_reason") == "max_tokens":
            flags.append("max_tokens")
        checks = []
        if parsed:
            checks, more = check_phrases(parsed, texts[(m, c, r, q)])
            flags += more
        (out["criteria"].setdefault(m, {}).setdefault(c, {}).setdefault(r, {}))[q] = {
            "i2_question": i2q, "dimension": dim, "raw": raw, "parsed": parsed,
            "phrase_check": checks, "flags": flags, "meta": meta}
        save_out(out)
        n = len(parsed["criteria"]) if parsed else "-"
        print(f"[{k}/{len(todo)}] {m} {c} r{r} {q}: {n} criteria"
              + (f"  FLAGS {flags}" if flags else ""))
    print(f"Saved {OUT_FILE}")
