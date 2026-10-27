"""
Which Mistral Large I2 answers the analyses use: one switch for every script.

MISTRAL_I2_SOURCE (environment variable, default "recollect"):
  recollect  data/raw/mistral_i2_recollect.json (mistral-large-2512, untruncated);
             primary since r6
  original   Mistral's answers in data/raw/instrument_2.json (truncated at 1,024
             tokens); sensitivity variant
Only mistral-large's I2 answers differ between the two; every other model and
instrument is the same.

Loaders:
  load_i2(source)        instrument_2 with Mistral's answers from the chosen source
  load_i5(source)        instrument_5 with Mistral's I2_S1-S3 extractions from
                         data/processed/i5_mistral_i2_recollect.json under "recollect"
  load_i5_judge(source)  second-judge tiers (i5_second_judge.json) with Mistral's
                         I2_S1-S3 cells from i5_second_judge_mistral_i2_recollect.json
                         under "recollect"
data/raw/ files are only read.
"""

import copy
import json
import os
from pathlib import Path

SOURCES  = ("recollect", "original")
DEFAULT  = "recollect"
MISTRAL  = "mistral-large"
I2_QS    = ["I2_S1", "I2_S2", "I2_S3"]

I2_FILE               = Path("data/raw/instrument_2.json")
RECOLLECT_FILE        = Path("data/raw/mistral_i2_recollect.json")
I5_FILE               = Path("data/raw/instrument_5.json")
I5_RECOLLECT_FILE     = Path("data/processed/i5_mistral_i2_recollect.json")
JUDGE_FILE            = Path("data/processed/i5_second_judge.json")
JUDGE_RECOLLECT_FILE  = Path("data/processed/i5_second_judge_mistral_i2_recollect.json")

LABELS = {"recollect": "Mistral I2 re-collected", "original": "Mistral I2 original"}


def mistral_i2_source(source: str | None = None) -> str:
    s = source or os.getenv("MISTRAL_I2_SOURCE", DEFAULT)
    if s not in SOURCES:
        raise ValueError(f"MISTRAL_I2_SOURCE must be one of {SOURCES}, got {s!r}")
    return s


def other_source(source: str | None = None) -> str:
    return "original" if mistral_i2_source(source) == "recollect" else "recollect"


def require_source(source: str, what: str) -> None:
    """For outputs that exist for one source only (e.g. the I2 criterion codes)."""
    if mistral_i2_source() != source:
        raise SystemExit(f"{what} exists for MISTRAL_I2_SOURCE={source} only; "
                         f"got {mistral_i2_source()}.")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_mistral_i2(base: dict, repl: dict) -> dict:
    """Copy of base with mistral-large's I2_S1-S3 cells taken from repl (same layout)."""
    out = copy.deepcopy(base)
    for c, runs in repl[MISTRAL].items():
        for r, qs in runs.items():
            for q in I2_QS:
                out[MISTRAL][c][r][q] = qs[q]
    return out


def load_i2(source: str | None = None) -> dict:
    i2 = _load(I2_FILE)
    if mistral_i2_source(source) == "original":
        return i2
    return {**i2, MISTRAL: _load(RECOLLECT_FILE)[MISTRAL]}


def load_i5(source: str | None = None) -> dict:
    i5 = _load(I5_FILE)
    if mistral_i2_source(source) == "original":
        return i5
    return _replace_mistral_i2(i5, _load(I5_RECOLLECT_FILE)["extractions"])


def load_i5_judge(source: str | None = None) -> dict:
    """{"meta", "judgments"} with Mistral's I2 cells swapped in under "recollect"."""
    j = _load(JUDGE_FILE)
    if mistral_i2_source(source) == "original":
        return j
    return {**j, "judgments": _replace_mistral_i2(j["judgments"],
                                                  _load(JUDGE_RECOLLECT_FILE)["judgments"])}
