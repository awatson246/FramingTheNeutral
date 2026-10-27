"""
Manual review pack for the I2 criterion coding. No API calls; reads data/ only.

criteria_review: extracted criteria next to their I1 definitions, for checking the
extraction by hand: claude-sonnet baseline run 1 (I1_Q1-Q3), grok-4.7 baseline
run 1 (I1_Q1-Q3), and grok-4.7 baseline run 3 I1_Q3 (rewritten negation).
Phrase flags come from the phrase checks in i2_criteria.json:
  ""            exact substring of the definition
  "typography"  matches only after normalization (quotes, dashes, whitespace, case)
  "rewritten"   no match even after normalization

coding_sample: 6 I2 answers to code by hand against the same model's criteria
(same condition and run; I1_Q1 -> I2_S1, I1_Q2 -> I2_S2, I1_Q3 -> I2_S3).
One answer per model, each scenario twice, 3 baseline and 3 CEO, run at random
(random.Random(7)). Mistral answers come from mistral_i2_recollect.json.

Output: data/processed/manual_review_pack.json (schema review_pack_v1)

Usage:
    python scripts/make_review_pack.py
"""

import json
import random
from pathlib import Path

from answer_text import get_answer_text
from data_sources import load_i2
from i2_criteria import (CONDITIONS, I1_FILE, INSTRUMENTS_FILE, MAPPING, MODELS,
                         OUT_FILE as CRITERIA_FILE, RUNS)

OUT_FILE      = Path("data/processed/manual_review_pack.json")
SEED           = 7

REVIEW_CELLS = ([("claude-sonnet", "baseline", "1", q) for q in MAPPING]
                + [("grok-4.7", "baseline", "1", q) for q in MAPPING]
                + [("grok-4.7", "baseline", "3", "I1_Q3")])
I2_TO_I1 = {i2: i1 for i1, (i2, _) in MAPPING.items()}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def phrase_flag(check: dict) -> str:
    if "flag" in check:                     # v2 criteria files store the flag
        return check["flag"]
    if check["exact"]:
        return ""
    return "typography" if check["normalized"] else "rewritten"


def criteria_review(crit: dict, i1: dict) -> list[dict]:
    out = []
    for m, c, r, q in REVIEW_CELLS:
        cell = crit[m][c][r][q]
        checks = {k["n"]: k for k in cell["phrase_check"]}
        out.append({
            "cell_id": f"{m}:{c}:{r}:{q}", "model": m, "condition": c, "run": int(r),
            "question_id": q, "dimension": cell["dimension"],
            "definition_text": get_answer_text(i1[m][c][r][q]),
            "criteria": [{"n": k["n"], "criterion": k["criterion"], "phrase": k["phrase"],
                          "flag": phrase_flag(checks[k["n"]])}
                         for k in cell["parsed"]["criteria"]],
        })
    return out


def coding_sample(crit: dict, i2: dict, scenarios: dict) -> list[dict]:
    rng = random.Random(SEED)
    scen = [s for s in scenarios for _ in range(2)]                 # each scenario twice
    conds = [c for c in CONDITIONS for _ in range(len(MODELS) // 2)]  # 3 baseline, 3 ceo
    rng.shuffle(scen)
    rng.shuffle(conds)
    out = []
    for m, s, c in zip(MODELS, scen, conds):
        r = rng.choice(RUNS)
        q = I2_TO_I1[s]
        src = i2
        out.append({
            "answer_id": f"{m}:{c}:{r}:{s}", "model": m, "condition": c, "run": int(r),
            "scenario_id": s, "dimension": MAPPING[q][1],
            "scenario_text": scenarios[s],
            "answer_text": get_answer_text(src[m][c][r][s]),
            "criteria_cell_id": f"{m}:{c}:{r}:{q}",
            "criteria": [{"n": k["n"], "criterion": k["criterion"]}
                         for k in crit[m][c][r][q]["parsed"]["criteria"]],
        })
    return out


if __name__ == "__main__":
    instruments = load(INSTRUMENTS_FILE)
    scenarios = {q["id"]: q["text"] for q in instruments["instruments"]["instrument_2"]["questions"]}
    crit = load(CRITERIA_FILE)["criteria"]

    pack = {"schema": "review_pack_v1",
            "criteria_review": criteria_review(crit, load(I1_FILE)),
            "coding_sample": coding_sample(crit, load_i2(), scenarios)}
    OUT_FILE.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {OUT_FILE} ({OUT_FILE.stat().st_size:,} bytes)")
    for a in pack["coding_sample"]:
        print(" ", a["answer_id"])
