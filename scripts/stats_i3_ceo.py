"""
Model-level I3 CEO test: CEO-condition minus baseline scores, per model and dimension.

Data: data/raw/instrument_3.json parsed scores, plus scores recovered from the raw
text by scripts/recover_i3.py (data/processed/i3_recovered.json) for cells that
failed to parse. --no-recovery uses parsed scores only; --exclude-key-typo drops
recoveries made by fixing a misspelled dimension key.

Point estimates follow the H2 test in plot_response_results.py: per scenario,
mean CEO score over runs minus mean baseline score over runs, averaged over the six
scenarios. The "all" dimension is the mean of the three dimension deltas.

Bootstrap, 10,000 resamples, seed 42:
  primary (stratified): within each model, runs are resampled with replacement
      separately within every scenario x condition stratum, and the H2 delta is
      recomputed. Cluster level: resample models, then stratified resampling inside
      each. Possible distinct resamples per model: product over strata of C(2n-1, n).
  secondary (cell): scenario x run cells resampled within each condition, delta =
      mean(CEO cells) - mean(baseline cells) (= H2 estimate when cells are balanced).

Across models, per dimension: mean delta, cluster CIs, and a two-sided sign test on
models with delta > 0 vs < 0 (ties excluded). Also reproduces the original H2
one-sample t-tests on the five original models.

Writes results/stats/i3_ceo_model_level.csv and results/stats/i3_ceo_model_level.md.
No API calls.

Usage:
    python scripts/stats_i3_ceo.py [--no-recovery] [--exclude-key-typo]
"""

import argparse
import csv
import json
from collections import Counter
from math import comb, prod
from pathlib import Path

import numpy as np
from scipy import stats

I3_FILE        = Path("data/raw/instrument_3.json")
RECOVERED_FILE = Path("data/processed/i3_recovered.json")
OUT_DIR        = Path("results/stats")
MODELS    = ["gpt-4o", "claude-sonnet", "deepseek-v3", "mistral-large",
             "gemini-3.1-flash-lite-preview", "grok-4.7"]
ORIGINAL_FIVE = MODELS[:5]
CONDS     = ["baseline", "ceo"]
SCENARIOS = [f"I3_S{i}" for i in range(1, 7)]
DIMS      = ["legal_certainty", "accountability", "enforceability"]
ALL       = "all_dimensions"
N_BOOT    = 10_000
SEED      = 42


def load_strata(data: dict, recovered: list[dict]) -> tuple[dict, Counter]:
    """
    strata[model][cond][scenario] = array (n_runs, 3) of scores (NaN where missing).
    A run is included if it has any score, parsed or recovered.
    """
    rec = {(r["model"], r["condition"], str(r["run"]), r["scenario"], r["dimension"]): r["score"]
           for r in recovered}
    n_recovered = Counter(r["model"] for r in recovered)
    strata = {}
    for m in MODELS:
        strata[m] = {}
        for c in CONDS:
            runs = data.get(m, {}).get(c, {})
            per_scenario = {s: [] for s in SCENARIOS}
            for run in sorted(runs, key=int):
                parsed = runs[run].get("parsed") or {}
                rows = {}
                for s in SCENARIOS:
                    row = []
                    for d in DIMS:
                        v = ((parsed.get(s) or {}).get(d) or {}).get("score")
                        if not isinstance(v, (int, float)):
                            v = rec.get((m, c, run, s, d))
                        row.append(np.nan if v is None else float(v))
                    rows[s] = row
                if any(not np.isnan(x) for row in rows.values() for x in row):
                    for s in SCENARIOS:
                        per_scenario[s].append(rows[s])
            strata[m][c] = {s: np.array(v, dtype=float).reshape(-1, len(DIMS))
                            for s, v in per_scenario.items()}
    return strata, n_recovered


def h2_delta(st: dict) -> np.ndarray:
    """Per-dimension mean over scenarios of (mean CEO - mean baseline), plus their mean."""
    d = np.nanmean([np.nanmean(st["ceo"][s], axis=0) - np.nanmean(st["baseline"][s], axis=0)
                    for s in SCENARIOS], axis=0)
    return np.append(d, d.mean())


def condition_means(st: dict, cond: str) -> np.ndarray:
    d = np.nanmean([np.nanmean(st[cond][s], axis=0) for s in SCENARIOS], axis=0)
    return np.append(d, d.mean())


def stratified_resample(rng, st: dict) -> dict:
    return {c: {s: a[rng.integers(0, len(a), size=len(a))] for s, a in st[c].items()} for c in CONDS}


def cells(st: dict, cond: str) -> np.ndarray:
    return np.vstack([st[cond][s] for s in SCENARIOS])


def cell_delta(base: np.ndarray, ceo: np.ndarray) -> np.ndarray:
    d = np.nanmean(ceo, axis=0) - np.nanmean(base, axis=0)
    return np.append(d, d.mean())


def resample_rows(rng, a: np.ndarray) -> np.ndarray:
    return a[rng.integers(0, len(a), size=len(a))]


def boot_model(st: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    strat = np.array([h2_delta(stratified_resample(rng, st)) for _ in range(N_BOOT)])
    rng = np.random.default_rng(SEED)
    b, c = cells(st, "baseline"), cells(st, "ceo")
    cell = np.array([cell_delta(resample_rows(rng, b), resample_rows(rng, c)) for _ in range(N_BOOT)])
    return strat, cell


def boot_cluster(strata: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = list(strata)
    rng = np.random.default_rng(SEED)
    strat = np.empty((N_BOOT, len(DIMS) + 1))
    for i in range(N_BOOT):
        picked = rng.integers(0, len(keys), size=len(keys))
        strat[i] = np.mean([h2_delta(stratified_resample(rng, strata[keys[k]])) for k in picked], axis=0)
    rng = np.random.default_rng(SEED)
    cell = np.empty((N_BOOT, len(DIMS) + 1))
    for i in range(N_BOOT):
        picked = rng.integers(0, len(keys), size=len(keys))
        cell[i] = np.mean([cell_delta(resample_rows(rng, cells(strata[keys[k]], "baseline")),
                                      resample_rows(rng, cells(strata[keys[k]], "ceo")))
                           for k in picked], axis=0)
    return strat, cell


def ci(boots: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def possible_resamples(st: dict) -> tuple[int, int]:
    n = lambda a: len(a)
    strat = prod(comb(2 * n(a) - 1, n(a)) for c in CONDS for a in st[c].values() if n(a))
    nb, nc = len(cells(st, "baseline")), len(cells(st, "ceo"))
    return strat, comb(2 * nb - 1, nb) * comb(2 * nc - 1, nc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-level I3 CEO vs baseline statistics.")
    parser.add_argument("--no-recovery", action="store_true",
                        help="Use parsed scores only (ignore data/processed/i3_recovered.json).")
    parser.add_argument("--exclude-key-typo", action="store_true",
                        help="Drop recoveries made by fixing a misspelled dimension key.")
    args = parser.parse_args()

    data = json.loads(I3_FILE.read_text(encoding="utf-8"))
    recovered = []
    if not args.no_recovery and RECOVERED_FILE.exists():
        recovered = json.loads(RECOVERED_FILE.read_text(encoding="utf-8"))["recovered"]
        if args.exclude_key_typo:
            recovered = [r for r in recovered if r["method"] != "key_typo"]
    strata, n_recovered = load_strata(data, recovered)
    recovery_note = ("parsed scores only" if args.no_recovery else
                     f"parsed scores + {len(recovered)} recovered from raw text "
                     f"({RECOVERED_FILE}{', key_typo excluded' if args.exclude_key_typo else ''})")

    LABELS = DIMS + [ALL]
    rows, per_model = [], {}
    for m in MODELS:
        st = strata[m]
        point = h2_delta(st)
        strat, cell = boot_model(st)
        full_cell = cell_delta(cells(st, "baseline"), cells(st, "ceo"))
        base_m, ceo_m = condition_means(st, "baseline"), condition_means(st, "ceo")
        p_strat, p_cell = possible_resamples(st)
        per_model[m] = dict(zip(LABELS, point))
        runs_b = {len(a) for a in st["baseline"].values()}
        runs_c = {len(a) for a in st["ceo"].values()}
        for j, d in enumerate(LABELS):
            lo, hi = ci(strat[:, j])
            clo, chi = ci(cell[:, j])
            miss = (np.isnan(cells(st, "baseline")).sum() + np.isnan(cells(st, "ceo")).sum()) if d == ALL else \
                   (np.isnan(cells(st, "baseline")[:, j]).sum() + np.isnan(cells(st, "ceo")[:, j]).sum())
            rows.append(dict(
                analysis="per_model", model=m, dimension=d,
                baseline_mean=float(base_m[j]), ceo_mean=float(ceo_m[j]),
                delta=float(point[j]), ci_low=lo, ci_high=hi,
                delta_cells=float(full_cell[j]), cell_ci_low=clo, cell_ci_high=chi,
                runs_baseline="/".join(map(str, sorted(runs_b))),
                runs_ceo="/".join(map(str, sorted(runs_c))),
                n_recovered_scores=n_recovered.get(m, 0), n_missing_scores=int(miss),
                strat_resamples_possible=p_strat, cell_resamples_possible=p_cell,
                strat_distinct_boot_values=len(np.unique(strat[:, j].round(10))),
                n_boot=N_BOOT))

    cstrat, ccell = boot_cluster(strata)
    for j, d in enumerate(LABELS):
        vals = [per_model[m][d] for m in MODELS]
        k_pos, k_neg = sum(v > 1e-12 for v in vals), sum(v < -1e-12 for v in vals)
        n_nz = k_pos + k_neg
        p = stats.binomtest(k_pos, n_nz, 0.5, alternative="two-sided").pvalue if n_nz else float("nan")
        lo, hi = ci(cstrat[:, j])
        clo, chi = ci(ccell[:, j])
        mrows = [r for r in rows if r["analysis"] == "per_model" and r["dimension"] == d]
        rows.append(dict(analysis="across_models", model=f"{len(MODELS)} models", dimension=d,
                         baseline_mean=float(np.mean([r["baseline_mean"] for r in mrows])),
                         ceo_mean=float(np.mean([r["ceo_mean"] for r in mrows])),
                         delta=float(np.mean(vals)), ci_low=lo, ci_high=hi,
                         cell_ci_low=clo, cell_ci_high=chi, n_boot=N_BOOT,
                         sign_k_negative=k_neg, sign_k_positive=k_pos,
                         sign_ties=len(vals) - n_nz, sign_p_two_sided=float(p)))

    for d in DIMS:
        arr = np.array([per_model[m][d] for m in ORIGINAL_FIVE])
        t, p = stats.ttest_1samp(arr, 0)
        rows.append(dict(analysis="h2_ttest_original_five", model="5 models", dimension=d,
                         delta=float(arr.mean()), t=float(t), p_t=float(p)))

    # ---- outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["analysis", "model", "dimension", "baseline_mean", "ceo_mean", "delta",
            "ci_low", "ci_high", "delta_cells", "cell_ci_low", "cell_ci_high",
            "runs_baseline", "runs_ceo", "n_recovered_scores", "n_missing_scores",
            "strat_resamples_possible", "cell_resamples_possible", "strat_distinct_boot_values",
            "n_boot", "sign_k_negative", "sign_k_positive", "sign_ties", "sign_p_two_sided", "t", "p_t"]
    with open(OUT_DIR / "i3_ceo_model_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    f3 = lambda v: f"{v:+.3f}"
    iv = lambda lo, hi: f"[{f3(lo)}, {f3(hi)}]"
    LABEL = {"legal_certainty": "Legal certainty", "accountability": "Accountability",
             "enforceability": "Enforceability", ALL: "All three"}
    get = lambda m, d: next(r for r in rows if r["analysis"] == "per_model"
                            and r["model"] == m and r["dimension"] == d)
    across = {r["dimension"]: r for r in rows if r["analysis"] == "across_models"}
    md = ["# I3 CEO vs baseline, model level", "",
          f"Generated by `scripts/stats_i3_ceo.py`. Data: {recovery_note}.",
          "delta = CEO - baseline (H2 estimator: mean over scenarios of per-scenario run means). "
          "Primary CI: stratified bootstrap (runs resampled within each scenario x condition), "
          "10,000 resamples, seed 42. Secondary CI: cell bootstrap (scenario x run cells "
          "resampled within condition).", "",
          "## Data per model", "",
          "| Model | runs per scenario (b/c) | recovered scores | missing scores |", "|---|---|---|---|"]
    for m in MODELS:
        r0 = get(m, ALL)
        md.append(f"| {m} | {r0['runs_baseline']}/{r0['runs_ceo']} | {r0['n_recovered_scores']} | "
                  f"{r0['n_missing_scores']} |")
    md += ["", "## Per model: delta [stratified 95% CI]", "",
           "| Model | " + " | ".join(LABEL[d] for d in LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for m in MODELS:
        md.append(f"| {m} | " + " | ".join(
            f"{f3(get(m, d)['delta'])} {iv(get(m, d)['ci_low'], get(m, d)['ci_high'])}" for d in LABELS) + " |")
    md += ["", "## Across models", "",
           "| Dimension | mean delta | 95% CI stratified (cluster) | 95% CI cell (cluster) | "
           "models < 0 | models > 0 | ties | sign test p (two-sided) |", "|---|---|---|---|---|---|---|---|"]
    for d in LABELS:
        a = across[d]
        md.append(f"| {LABEL[d]} | {f3(a['delta'])} | {iv(a['ci_low'], a['ci_high'])} | "
                  f"{iv(a['cell_ci_low'], a['cell_ci_high'])} | {a['sign_k_negative']} | "
                  f"{a['sign_k_positive']} | {a['sign_ties']} | {a['sign_p_two_sided']:.4f} |")
    md += ["", "## Accountability", "",
           "| Model | baseline | CEO | delta | 95% CI stratified | 95% CI cell |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        r = get(m, "accountability")
        md.append(f"| {m} | {r['baseline_mean']:.3f} | {r['ceo_mean']:.3f} | {f3(r['delta'])} | "
                  f"{iv(r['ci_low'], r['ci_high'])} | {iv(r['cell_ci_low'], r['cell_ci_high'])} |")
    a = across["accountability"]
    md.append(f"| **{a['model']} (mean)** | {a['baseline_mean']:.3f} | {a['ceo_mean']:.3f} | "
              f"{f3(a['delta'])} | {iv(a['ci_low'], a['ci_high'])} | {iv(a['cell_ci_low'], a['cell_ci_high'])} |")
    md += ["", f"Sign test: {a['sign_k_negative']} < 0, {a['sign_k_positive']} > 0, "
               f"{a['sign_ties']} ties; two-sided p = {a['sign_p_two_sided']:.4f}.", "",
           "## Original H2 one-sample t-tests (five original models, this data)", "",
           "| Dimension | mean delta | t | p |", "|---|---|---|---|"]
    for r in rows:
        if r["analysis"] == "h2_ttest_original_five":
            md.append(f"| {LABEL[r['dimension']]} | {f3(r['delta'])} | {r['t']:.3f} | {r['p_t']:.4f} |")
    (OUT_DIR / "i3_ceo_model_level.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print((OUT_DIR / "i3_ceo_model_level.md").read_text(encoding="utf-8"))
    print(f"Saved {OUT_DIR / 'i3_ceo_model_level.csv'} and {OUT_DIR / 'i3_ceo_model_level.md'}")
