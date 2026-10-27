"""
Model-level primary inference shared by stats_h1.py and stats_i3_ceo.py.

One value per model (n models). Primary result:
  - bootstrap 95% CI of the mean, resampling models only (10,000, seed 42);
    with n models there are C(2n-1, n) distinct resamples (462 for n = 6)
  - one-sample t-test against 0 (two-sided): mean, SD, t, df, p, and the
    t-based 95% CI: mean +/- t(0.975, df) * SD / sqrt(n)

Also maintains results/stats/sensitivity_summary.csv: one row per test, primary
columns plus each sensitivity CI. Each script updates only its own rows.
"""

import csv
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

N_BOOT = 10_000
SEED   = 42
SUMMARY_FILE = Path("results/stats/sensitivity_summary.csv")

I4_DIMS = ["legal certainty", "accountability", "enforceability", "all three"]
I4_VARIANTS = ["(a) + G6-G10", "(b) Mistral I2 re-collected", "(c) re-scored I1"]
I2_SCOPES = ["all scenarios", "S1 legal certainty", "S2 accountability", "S3 enforceability"]
I2_MEASURES = ["% not met", "CEO-baseline % met"]
I2_VARIANTS = ["(a) claude-sonnet coder", "(b) gpt no quote rule"]
SUMMARY_ORDER = (["H1 diff", "I3 legal certainty", "I3 accountability", "I3 enforceability",
                  "I3 all three", "I3 accountability (key_typo excluded)"]
                 + [f"I4 asymmetry {d}" for d in I4_DIMS]
                 + [f"I4 asymmetry {d} {v}" for v in I4_VARIANTS for d in I4_DIMS]
                 + [f"I2 {m} {s}" for m in I2_MEASURES for s in I2_SCOPES]
                 + [f"I2 {m} {s} {v}" for v in I2_VARIANTS for m in I2_MEASURES for s in I2_SCOPES])
SUMMARY_COLS = ["test", "n_models", "mean", "sd", "t", "df", "p_t",
                "t_ci_low", "t_ci_high",
                "model_ci_low", "model_ci_high", "model_resamples_possible",
                "h1_cluster_ci_low", "h1_cluster_ci_high",
                "i3_stratified_cluster_ci_low", "i3_stratified_cluster_ci_high",
                "i3_cell_cluster_ci_low", "i3_cell_cluster_ci_high"]


def model_level_inference(values) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    rng = np.random.default_rng(SEED)
    boots = v[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    t, p = stats.ttest_1samp(v, 0)
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return dict(n_models=n, mean=float(v.mean()), sd=sd, t=float(t),
                df=n - 1, p_t=float(p),
                t_ci_low=float(v.mean() - half), t_ci_high=float(v.mean() + half),
                model_ci_low=float(lo), model_ci_high=float(hi),
                model_resamples_possible=comb(2 * n - 1, n))


def update_summary(rows: list[dict]) -> None:
    """Replace this script's rows (matched on 'test') and rewrite the file in SUMMARY_ORDER."""
    existing = {}
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, newline="", encoding="utf-8") as f:
            existing = {r["test"]: r for r in csv.DictReader(f)}
    for r in rows:
        existing[r["test"]] = {c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, ""))
                               for c in SUMMARY_COLS}
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for test in SUMMARY_ORDER + sorted(set(existing) - set(SUMMARY_ORDER)):
            if test in existing:
                w.writerow(existing[test])
