"""
Model-level I4 asymmetry: mean I4 score a model gave its peers minus its own mean
I3 baseline score, per dimension (same mapping and definition as build_elp in
plot_response_results.py), plus the mean over the three dimensions.

I3: baseline scores from data/raw/instrument_3.json plus recovered scores
(data/processed/i3_recovered.json, via stats_i3_ceo.load_strata), pooled over all
scenario x run cells as build_elp does.

I4 peer scores, primary data:
  original evaluators: data/raw/instrument_4.json pairs P1-P5 (peer_eval_pairs.json)
  grok-4.7:            data/raw/instrument_4_grok.json G1-G5 (Grok as evaluator)
  Mistral as evaluatee (P3, G4), I2 questions: MISTRAL_I2_SOURCE (data_sources.py;
    default re-collected: the <pair>__mistral_i2_recollect cells of
    instrument_4_rescore.json / instrument_4_grok.json; original: the P3/G4 cells)
Sensitivity variants (each replaces or adds to the primary data on its own):
  (a) + G6-G10: also the original models' ratings of Grok
  (b) the other Mistral I2 source (default: original, truncated answers)
  (c) re-scored I1: P1-P5 I1 cells from data/raw/instrument_4_rescore.json

Across the six models, per dimension: mean, SD, two-sided one-sample t-test vs 0,
t-based 95% CI, model-only bootstrap CI (model_level.py), and how many models have
asymmetry > 0. Also reproduces the old build_elp asymmetry for the five original
models (unrecovered I3, original I4) as a check.

Writes results/stats/i4_asymmetry_model_level.csv and .md, and the I4 rows of
results/stats/sensitivity_summary.csv. No API calls; reads data/ only.

Usage:
    python scripts/stats_i4_asymmetry.py
"""

import csv
import json
from pathlib import Path

import numpy as np

import stats_i3_ceo as i3
from data_sources import LABELS, mistral_i2_source, other_source
from model_level import model_level_inference, update_summary

I4_FILE        = Path("data/raw/instrument_4.json")
I4_GROK_FILE   = Path("data/raw/instrument_4_grok.json")
I4_RESCORE     = Path("data/raw/instrument_4_rescore.json")
PAIRS_FILE     = Path("data/prompts/peer_eval_pairs.json")
GROK_PAIRS     = Path("data/prompts/peer_eval_pairs_grok.json")
OLD_ELP        = Path("results/instrument_4/elp_profiles.json")
OUT_DIR        = Path("results/stats")

MODELS  = i3.MODELS
DIMS    = i3.DIMS
ALL     = i3.ALL
I4_MAP  = {"legal_certainty_adequacy": "legal_certainty",
           "accountability_mechanisms": "accountability",
           "enforcement_conditions":    "enforceability"}
I1_QS   = ["I1_Q1", "I1_Q2", "I1_Q3"]
I2_QS   = ["I2_S1", "I2_S2", "I2_S3"]
SUFFIX  = "__mistral_i2_recollect"
PRIMARY_SOURCE = mistral_i2_source()
VARIANTS = {"primary": "primary",
            "a":       "(a) + G6-G10",
            "b":       f"(b) {LABELS[other_source()]}",
            "c":       "(c) re-scored I1"}
LABEL   = {"legal_certainty": "legal certainty", "accountability": "accountability",
           "enforceability": "enforceability", ALL: "all three"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def i3_means(strata: dict) -> dict:
    """Pooled mean of all baseline scenario x run scores per model and dimension (build_elp)."""
    out = {}
    for m in MODELS:
        cells = np.vstack([strata[m]["baseline"][s] for s in i3.SCENARIOS])
        out[m] = dict(zip(DIMS, np.nanmean(cells, axis=0)))
    return out


def peer_cells(variant: str, i4: dict, grok: dict, rescore: dict,
               pairs: list[dict], grok_pairs: list[dict]) -> dict:
    """{evaluator: [I4 cell, ...]} for one data variant."""
    by_eval = {m: [] for m in MODELS}

    # re-collected Mistral I2 cells: the primary source, or the other one under (b)
    use_recollect = (PRIMARY_SOURCE == "recollect") != (variant == "b")

    def take(pair_id: str, src: dict, evaluator: str, evaluatee: str):
        for q, cell in src.get(pair_id, {}).items():
            if use_recollect and evaluatee == "mistral-large" and q in I2_QS:
                rec = (rescore if pair_id.startswith("P") else grok).get(pair_id + SUFFIX, {})
                cell = rec.get(q, cell)
            if variant == "c" and pair_id.startswith("P") and q in I1_QS:
                cell = rescore.get(pair_id, {}).get(q, cell)
            by_eval[evaluator].append(cell)

    for p in pairs:                                            # P1-P5, original evaluators
        take(p["pair_id"], i4, p["evaluator"], p["evaluatee"])
    for p in grok_pairs:
        if p["evaluator"] == "grok-4.7" or variant == "a":     # G1-G5; G6-G10 only in (a)
            take(p["pair_id"], grok, p["evaluator"], p["evaluatee"])
    return by_eval


def peer_means(cells_by_eval: dict) -> tuple[dict, dict]:
    means, counts = {}, {}
    for m, cells in cells_by_eval.items():
        vals = {d: [] for d in DIMS}
        for c in cells:
            for i4d, i3d in I4_MAP.items():
                s = ((c or {}).get("parsed") or {}).get(i4d, {}).get("score")
                if isinstance(s, (int, float)):
                    vals[i3d].append(s)
        means[m] = {d: (float(np.mean(v)) if v else None) for d, v in vals.items()}
        counts[m] = len(cells)
    return means, counts


def asymmetry_rows(variant: str, i3m: dict, pm: dict, counts: dict) -> tuple[list, list]:
    per_model, across = [], []
    asym = {m: {d: pm[m][d] - i3m[m][d] for d in DIMS} for m in MODELS}
    for m in MODELS:
        asym[m][ALL] = float(np.mean([asym[m][d] for d in DIMS]))
        for d in DIMS + [ALL]:
            per_model.append(dict(
                analysis="per_model", variant=VARIANTS[variant], model=m, dimension=d,
                i3_mean=float(i3m[m][d]) if d != ALL else float(np.mean([i3m[m][x] for x in DIMS])),
                i4_peer_mean=pm[m][d] if d != ALL else float(np.mean([pm[m][x] for x in DIMS])),
                asymmetry=asym[m][d], n_i4_cells=counts[m]))
    for d in DIMS + [ALL]:
        vals = [asym[m][d] for m in MODELS]
        across.append(dict(analysis="across_models", variant=VARIANTS[variant],
                           model=f"{len(MODELS)} models", dimension=d,
                           **model_level_inference(vals), k_positive=int(sum(v > 0 for v in vals))))
    return per_model, across


def reproduce_old(i4: dict, pairs: list[dict]) -> dict:
    """Old build_elp asymmetry: unrecovered I3 baseline, original I4, five original models."""
    data = load(i3.I3_FILE)
    strata, _ = i3.load_strata(data, [])
    i3m = i3_means(strata)
    by_eval = {m: [] for m in MODELS}
    for p in pairs:
        by_eval[p["evaluator"]] += list(i4.get(p["pair_id"], {}).values())
    pm, _ = peer_means(by_eval)
    return {m: {d: pm[m][d] - i3m[m][d] for d in DIMS} for m in MODELS[:5]}


if __name__ == "__main__":
    i4, grok, rescore = load(I4_FILE), load(I4_GROK_FILE), load(I4_RESCORE)
    pairs, grok_pairs = load(PAIRS_FILE)["pairs"], load(GROK_PAIRS)["pairs"]
    recovered = load(i3.RECOVERED_FILE)["recovered"]
    strata, _ = i3.load_strata(load(i3.I3_FILE), recovered)
    i3m = i3_means(strata)

    per_model, across = [], []
    for v in VARIANTS:
        pm, counts = peer_means(peer_cells(v, i4, grok, rescore, pairs, grok_pairs))
        pr, ac = asymmetry_rows(v, i3m, pm, counts)
        per_model += pr
        across += ac

    # Check against the old ELP numbers
    old = load(OLD_ELP)
    rep = reproduce_old(i4, pairs)
    max_dev = max(abs(rep[m][d] - old[m]["asymmetry"][d]) for m in rep for d in DIMS)

    # ---- outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["analysis", "variant", "model", "dimension", "i3_mean", "i4_peer_mean", "asymmetry",
            "n_i4_cells", "n_models", "mean", "sd", "t", "df", "p_t", "t_ci_low", "t_ci_high",
            "model_ci_low", "model_ci_high", "model_resamples_possible", "k_positive"]
    with open(OUT_DIR / "i4_asymmetry_model_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_model + across:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    f3 = lambda x: f"{x:+.3f}"
    iv = lambda lo, hi: f"[{f3(lo)}, {f3(hi)}]"
    pget = lambda v, m, d: next(r for r in per_model if r["variant"] == VARIANTS[v]
                                and r["model"] == m and r["dimension"] == d)
    md = ["# I4 asymmetry, model level", "",
          "Generated by `scripts/stats_i4_asymmetry.py`. asymmetry = mean I4 score given to peers "
          "minus own mean I3 baseline score (build_elp mapping). I3 includes recovered scores. "
          "Primary: P1-P5 (instrument_4.json) for the five original evaluators, G1-G5 for "
          f"grok-4.7, {LABELS[PRIMARY_SOURCE]} answers. Across models: n = 6, one-sample t-test vs 0 "
          "(two-sided), t-based 95% CI, model-only bootstrap 95% CI (10,000, seed 42).", "",
          "## Per model (primary)", "",
          "| Model | I4 cells | " + " | ".join(f"{LABEL[d]}: I3 / I4 / asym" for d in DIMS) + " | all three asym |",
          "|---|---|" + "---|" * (len(DIMS) + 1)]
    for m in MODELS:
        cells = [pget("primary", m, d) for d in DIMS]
        md.append(f"| {m} | {cells[0]['n_i4_cells']} | " + " | ".join(
            f"{c['i3_mean']:.3f} / {c['i4_peer_mean']:.3f} / {f3(c['asymmetry'])}" for c in cells)
            + f" | {f3(pget('primary', m, ALL)['asymmetry'])} |")
    md += ["", "## Across models", "",
           "| Variant | Dimension | mean | SD | t | df | p (two-sided) | 95% CI (t) | 95% CI (models) | models > 0 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for a in across:
        md.append(f"| {a['variant']} | {LABEL[a['dimension']]} | {f3(a['mean'])} | {a['sd']:.3f} | "
                  f"{a['t']:.3f} | {a['df']} | {a['p_t']:.2e} | {iv(a['t_ci_low'], a['t_ci_high'])} | "
                  f"{iv(a['model_ci_low'], a['model_ci_high'])} | {a['k_positive']}/{a['n_models']} |")
    md += ["", "## Check: old build_elp asymmetry reproduced", "",
           f"Five original models, unrecovered I3, original I4: max absolute deviation from "
           f"results/instrument_4/elp_profiles.json = {max_dev:.4f} (that file rounds to 3 decimals).", ""]
    (OUT_DIR / "i4_asymmetry_model_level.md").write_text("\n".join(md), encoding="utf-8")

    update_summary([{**a, "test": f"I4 asymmetry {LABEL[a['dimension']]}"
                     + ("" if a["variant"] == "primary" else f" {a['variant']}")} for a in across])

    print((OUT_DIR / "i4_asymmetry_model_level.md").read_text(encoding="utf-8"))
    print(f"Saved {OUT_DIR / 'i4_asymmetry_model_level.csv'} and {OUT_DIR / 'i4_asymmetry_model_level.md'}")
