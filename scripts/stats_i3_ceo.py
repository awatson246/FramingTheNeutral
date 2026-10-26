"""
Model-level I3 CEO test: CEO-condition minus baseline scores, per model and dimension.

Point estimates follow the H2 test in plot_response_results.py: per scenario,
mean CEO score over runs minus mean baseline score over runs, averaged over the
six scenarios (unparsed runs and missing scores are skipped, as extract_i3_scores
does). The "all" dimension is the mean of the three dimension deltas.

Bootstrap (10,000 resamples, seed 42): within a model, scenario x run cells are
resampled with replacement separately for each condition (baseline and CEO runs
are independent, and cell counts can differ between conditions). Each resample's
delta is mean(CEO cells) - mean(baseline cells), using the same resampled cells
for every dimension. At the full sample this equals the H2 estimate whenever cells
are balanced; it is reported as delta_cells. Number of distinct resamples possible:
C(2n_b - 1, n_b) * C(2n_c - 1, n_c).

Across models, per dimension: mean delta with a cluster-bootstrap 95% CI (resample
models, then cells within each), and a two-sided sign test on models with delta
> 0 vs < 0 (ties excluded). Also reproduces the original H2 one-sample t-tests on
the five original models.

Writes results/stats/i3_ceo_model_level.csv and results/stats/i3_ceo_model_level.md.
No API calls.

Usage:
    python scripts/stats_i3_ceo.py
"""

import csv
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

I3_FILE   = Path("data/raw/instrument_3.json")
OUT_DIR   = Path("results/stats")
MODELS    = ["gpt-4o", "claude-sonnet", "deepseek-v3", "mistral-large",
             "gemini-3.1-flash-lite-preview", "grok-4.7"]
ORIGINAL_FIVE = MODELS[:5]
SCENARIOS = [f"I3_S{i}" for i in range(1, 7)]
DIMS      = ["legal_certainty", "accountability", "enforceability"]
ALL       = "all_dimensions"
N_BOOT    = 10_000
SEED      = 42


def load_cells(data: dict, model: str, cond: str) -> list[dict]:
    """One cell per (scenario, parsed run): {"scenario", dim: score or None}."""
    cells = []
    for run, cell in sorted(data.get(model, {}).get(cond, {}).items(), key=lambda kv: int(kv[0])):
        parsed = cell.get("parsed")
        if not parsed:
            continue
        for s in SCENARIOS:
            row = {"scenario": s}
            for d in DIMS:
                v = (parsed.get(s) or {}).get(d, {}).get("score")
                row[d] = v if isinstance(v, (int, float)) else None
            cells.append(row)
    return cells


def h2_delta(base: list[dict], ceo: list[dict], dim: str) -> float:
    """H2 estimator: mean over scenarios of (mean CEO - mean baseline) for one dimension."""
    deltas = []
    for s in SCENARIOS:
        b = [c[dim] for c in base if c["scenario"] == s and c[dim] is not None]
        c_ = [c[dim] for c in ceo if c["scenario"] == s and c[dim] is not None]
        if b and c_:
            deltas.append(np.mean(c_) - np.mean(b))
    return float(np.mean(deltas))


def scenario_mean(cells: list[dict], dim: str) -> float:
    return float(np.mean([np.mean([c[dim] for c in cells if c["scenario"] == s and c[dim] is not None])
                          for s in SCENARIOS
                          if any(c["scenario"] == s and c[dim] is not None for c in cells)]))


def as_arrays(cells: list[dict]) -> np.ndarray:
    return np.array([[np.nan if c[d] is None else c[d] for d in DIMS] for c in cells], dtype=float)


def cell_deltas(base: np.ndarray, ceo: np.ndarray) -> np.ndarray:
    """Per-dimension mean(CEO) - mean(baseline) over the given cells, plus their mean."""
    d = np.nanmean(ceo, axis=0) - np.nanmean(base, axis=0)
    return np.append(d, d.mean())


def resample(rng, arr: np.ndarray) -> np.ndarray:
    return arr[rng.integers(0, len(arr), size=len(arr))]


def model_bootstrap(base: np.ndarray, ceo: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return np.array([cell_deltas(resample(rng, base), resample(rng, ceo)) for _ in range(N_BOOT)])


def cluster_bootstrap(arrays: dict) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    keys = list(arrays)
    out = np.empty((N_BOOT, len(DIMS) + 1))
    for i in range(N_BOOT):
        picked = rng.integers(0, len(keys), size=len(keys))
        out[i] = np.mean([cell_deltas(resample(rng, arrays[keys[k]][0]),
                                      resample(rng, arrays[keys[k]][1])) for k in picked], axis=0)
    return out


def ci(boots: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


if __name__ == "__main__":
    data = json.loads(I3_FILE.read_text(encoding="utf-8"))
    rows, per_model, arrays = [], {}, {}

    for m in MODELS:
        base, ceo = load_cells(data, m, "baseline"), load_cells(data, m, "ceo")
        b_arr, c_arr = as_arrays(base), as_arrays(ceo)
        arrays[m] = (b_arr, c_arr)
        boots = model_bootstrap(b_arr, c_arr)
        full  = cell_deltas(b_arr, c_arr)
        n_b, n_c = len(base), len(ceo)
        possible = comb(2 * n_b - 1, n_b) * comb(2 * n_c - 1, n_c)
        deltas = {d: h2_delta(base, ceo, d) for d in DIMS}
        deltas[ALL] = float(np.mean(list(deltas.values())))
        per_model[m] = deltas
        for j, d in enumerate(DIMS + [ALL]):
            lo, hi = ci(boots[:, j])
            rows.append(dict(
                analysis="per_model", model=m, dimension=d,
                baseline_mean=(scenario_mean(base, d) if d != ALL
                               else float(np.mean([scenario_mean(base, x) for x in DIMS]))),
                ceo_mean=(scenario_mean(ceo, d) if d != ALL
                          else float(np.mean([scenario_mean(ceo, x) for x in DIMS]))),
                delta=deltas[d], delta_cells=float(full[j]), ci_low=lo, ci_high=hi,
                n_baseline_cells=n_b, n_ceo_cells=n_c,
                n_missing_scores=int(np.isnan(b_arr[:, j]).sum() + np.isnan(c_arr[:, j]).sum())
                                 if d != ALL else int(np.isnan(b_arr).sum() + np.isnan(c_arr).sum()),
                distinct_resamples_possible=possible,
                distinct_boot_values=len(np.unique(boots[:, j].round(10))),
                n_boot=N_BOOT))

    cboots = cluster_bootstrap(arrays)
    for j, d in enumerate(DIMS + [ALL]):
        vals = [per_model[m][d] for m in MODELS]
        k_pos, k_neg = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
        n_nz = k_pos + k_neg
        p = stats.binomtest(k_pos, n_nz, 0.5, alternative="two-sided").pvalue if n_nz else float("nan")
        lo, hi = ci(cboots[:, j])
        rows.append(dict(analysis="across_models", model=f"{len(MODELS)} models", dimension=d,
                         baseline_mean=float(np.mean([r["baseline_mean"] for r in rows
                                                      if r["analysis"] == "per_model" and r["dimension"] == d])),
                         ceo_mean=float(np.mean([r["ceo_mean"] for r in rows
                                                 if r["analysis"] == "per_model" and r["dimension"] == d])),
                         delta=float(np.mean(vals)), ci_low=lo, ci_high=hi, n_boot=N_BOOT,
                         sign_k_negative=k_neg, sign_k_positive=k_pos, sign_ties=len(vals) - n_nz,
                         sign_p_two_sided=float(p)))

    # Reproduction of the original H2 one-sample t-tests (five original models)
    for d in DIMS:
        arr = np.array([per_model[m][d] for m in ORIGINAL_FIVE])
        t, p = stats.ttest_1samp(arr, 0)
        rows.append(dict(analysis="h2_ttest_original_five", model="5 models", dimension=d,
                         delta=float(arr.mean()), t=float(t), p_t=float(p)))

    # ---- outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["analysis", "model", "dimension", "baseline_mean", "ceo_mean", "delta", "delta_cells",
            "ci_low", "ci_high", "n_baseline_cells", "n_ceo_cells", "n_missing_scores",
            "distinct_resamples_possible", "distinct_boot_values", "n_boot",
            "sign_k_negative", "sign_k_positive", "sign_ties", "sign_p_two_sided", "t", "p_t"]
    with open(OUT_DIR / "i3_ceo_model_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    f3 = lambda v: f"{v:+.3f}"
    LABEL = {"legal_certainty": "Legal certainty", "accountability": "Accountability",
             "enforceability": "Enforceability", ALL: "All three"}
    get = lambda m, d: next(r for r in rows if r["analysis"] == "per_model" and r["model"] == m
                            and r["dimension"] == d)
    across = {r["dimension"]: r for r in rows if r["analysis"] == "across_models"}
    md = ["# I3 CEO vs baseline, model level", "",
          "Generated by `scripts/stats_i3_ceo.py`. delta = CEO - baseline (H2 estimator: mean over "
          "scenarios of per-scenario run means). CIs: bootstrap, 10,000 resamples, seed 42; cells "
          "(scenario x run) resampled within each condition; CIs are for delta_cells.", "",
          "## Per model: delta [95% CI]", "",
          "| Model | cells (b/c) | " + " | ".join(LABEL[d] for d in DIMS + [ALL]) + " |",
          "|---|---|" + "---|" * (len(DIMS) + 1)]
    for m in MODELS:
        r0 = get(m, DIMS[0])
        md.append(f"| {m} | {r0['n_baseline_cells']}/{r0['n_ceo_cells']} | " + " | ".join(
            f"{f3(get(m, d)['delta'])} [{f3(get(m, d)['ci_low'])}, {f3(get(m, d)['ci_high'])}]"
            for d in DIMS + [ALL]) + " |")
    md += ["", "## Across models", "",
           "| Dimension | mean delta | 95% CI (cluster) | models < 0 | models > 0 | ties | sign test p (two-sided) |",
           "|---|---|---|---|---|---|---|"]
    for d in DIMS + [ALL]:
        a = across[d]
        md.append(f"| {LABEL[d]} | {f3(a['delta'])} | [{f3(a['ci_low'])}, {f3(a['ci_high'])}] | "
                  f"{a['sign_k_negative']} | {a['sign_k_positive']} | {a['sign_ties']} | "
                  f"{a['sign_p_two_sided']:.4f} |")
    md += ["", "## Accountability", "",
           "| Model | baseline | CEO | delta | 95% CI |", "|---|---|---|---|---|"]
    for m in MODELS:
        r = get(m, "accountability")
        md.append(f"| {m} | {r['baseline_mean']:.3f} | {r['ceo_mean']:.3f} | {f3(r['delta'])} | "
                  f"[{f3(r['ci_low'])}, {f3(r['ci_high'])}] |")
    a = across["accountability"]
    md.append(f"| **{a['model']} (mean)** | {a['baseline_mean']:.3f} | {a['ceo_mean']:.3f} | "
              f"{f3(a['delta'])} | [{f3(a['ci_low'])}, {f3(a['ci_high'])}] |")
    md += ["", f"Sign test: {a['sign_k_negative']} < 0, {a['sign_k_positive']} > 0, "
               f"{a['sign_ties']} ties; two-sided p = {a['sign_p_two_sided']:.4f}.", "",
           "## Original H2 one-sample t-tests (five original models)", "",
           "| Dimension | mean delta | t | p |", "|---|---|---|---|"]
    for r in rows:
        if r["analysis"] == "h2_ttest_original_five":
            md.append(f"| {LABEL[r['dimension']]} | {f3(r['delta'])} | {r['t']:.3f} | {r['p_t']:.4f} |")
    (OUT_DIR / "i3_ceo_model_level.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved {OUT_DIR / 'i3_ceo_model_level.csv'} and {OUT_DIR / 'i3_ceo_model_level.md'}")
