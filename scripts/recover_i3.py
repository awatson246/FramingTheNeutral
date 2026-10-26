"""
Recover I3 scores that failed to parse at collection time, from the stored raw text.

Only extracts scores that are written in the raw output; nothing is inferred.
  tolerant_json  raw failed json.loads; parsed after stripping markdown fences and
                 trailing commas before } or ] (json.loads with strict=False)
  key_typo       a scenario is missing exactly one expected dimension key and has
                 exactly one unexpected key that is a close spelling match
                 (difflib ratio >= 0.8); its written score is assigned to the
                 missing dimension. Flagged separately so it can be excluded.

Also re-parses every stored cell with the tolerant parser and checks it agrees with
the existing parsed scores, as a guard on the parser.

Reads data/raw/instrument_3.json (never written). Writes
data/processed/i3_recovered.json: one record per recovered score, with the raw
snippet it came from.

Usage:
    python scripts/recover_i3.py
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

I3_FILE  = Path("data/raw/instrument_3.json")
OUT_FILE = Path("data/processed/i3_recovered.json")
SCENARIOS = [f"I3_S{i}" for i in range(1, 7)]
DIMS      = ["legal_certainty", "accountability", "enforceability"]
TYPO_RATIO = 0.8


def tolerant_loads(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    clean = re.sub(r",(\s*[}\]])", r"\1", clean)       # trailing commas
    return json.loads(clean.strip(), strict=False)


def score_of(block: dict, key: str):
    v = (block or {}).get(key)
    s = v.get("score") if isinstance(v, dict) else None
    return s if isinstance(s, (int, float)) and not isinstance(s, bool) else None


def snippet(raw: str, scenario: str, key: str) -> str:
    """The raw text where this scenario's key and score are written."""
    start = raw.find(f'"{scenario}"')
    i = raw.find(f'"{key}"', start)
    m = re.search(r'"score"\s*:\s*-?\d+(\.\d+)?', raw[i:i + 200]) if i >= 0 else None
    return raw[i:i + m.end()] if m else raw[i:i + 80]


if __name__ == "__main__":
    data = json.loads(I3_FILE.read_text(encoding="utf-8"))
    records, still_missing, disagreements, checked = [], [], [], 0

    for model, conds in data.items():
        for cond, runs in conds.items():
            for run, cell in runs.items():
                raw, parsed = cell.get("raw") or "", cell.get("parsed")
                try:
                    tolerant = tolerant_loads(raw)
                except json.JSONDecodeError as e:
                    tolerant = None
                    tolerant_error = str(e)

                for s in SCENARIOS:
                    for d in DIMS:
                        existing = score_of((parsed or {}).get(s), d)
                        if existing is not None:
                            checked += 1
                            if tolerant is not None and score_of(tolerant.get(s), d) != existing:
                                disagreements.append((model, cond, run, s, d))
                            continue

                        base = dict(model=model, condition=cond, run=int(run), scenario=s, dimension=d)
                        found = score_of((tolerant or {}).get(s), d)
                        if found is not None and parsed is None:
                            records.append({**base, "score": found, "method": "tolerant_json",
                                            "raw_snippet": snippet(raw, s, d)})
                            continue

                        # Misspelled dimension key: one missing expected key, one close unknown key
                        block = (tolerant or parsed or {}).get(s) or {}
                        missing = [x for x in DIMS if score_of(block, x) is None]
                        unknown = [k for k in block if k not in DIMS and score_of(block, k) is not None]
                        if missing == [d] and len(unknown) == 1:
                            ratio = SequenceMatcher(None, unknown[0], d).ratio()
                            if ratio >= TYPO_RATIO:
                                records.append({**base, "score": score_of(block, unknown[0]),
                                                "method": "key_typo", "raw_key": unknown[0],
                                                "key_similarity": round(ratio, 3),
                                                "raw_snippet": snippet(raw, s, unknown[0])})
                                continue
                        still_missing.append({**base, "reason": "no written score found"
                                              if tolerant is not None else f"unparseable: {tolerant_error}"})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = {"source": str(I3_FILE),
           "methods": {"tolerant_json": "fences and trailing commas removed, then json.loads",
                       "key_typo": f"misspelled dimension key (similarity >= {TYPO_RATIO})"},
           "recovered": records, "still_missing": still_missing}
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Parser check: {checked} existing scores re-parsed, {len(disagreements)} disagreements"
          + (f": {disagreements[:5]}" if disagreements else ""))
    print(f"Recovered {len(records)} scores; still missing {len(still_missing)}")
    by = {}
    for r in records:
        by.setdefault((r["model"], r["condition"], r["run"], r["method"]), []).append(r)
    for (m, c, run, meth), rs in by.items():
        print(f"  {m} | {c} | run {run} | {meth}: {len(rs)} scores")
        for r in rs[:2] + (rs[-1:] if len(rs) > 2 else []):
            print(f"      {r['scenario']}/{r['dimension']} = {r['score']}  <- {r['raw_snippet']!r}")
    for r in still_missing:
        print("  still missing:", r)
    print(f"Saved {OUT_FILE}")
