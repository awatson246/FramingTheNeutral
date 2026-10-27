"""
Model-level H1 test: within-model I1 similarity vs I1->I2 cross-instrument similarity.

Point estimates are computed exactly as run_hypothesis_tests() in
plot_response_results.py does (baseline condition; within = mean pairwise cosine
of I1 runs 1-3, batch-encoded; cross = cosine of run-1 I1 vs run-1 I2 answers,
encoded singly), then averaged over the three question pairs to one value per model.

Bootstrap (10,000 resamples, seed 42) resamples run indices {1, 2, 3}:
  within* = mean cosine over pairs of *distinct* resampled runs (self-pairs excluded;
            a resample drawing one run three times has no pair and is redrawn)
  cross*  = mean run-matched I1->I2 cosine over the resampled runs
Because H1's cross uses run 1 only, the bootstrap CIs belong to the all-runs version
of the estimator (cross averaged over runs 1-3), reported alongside as diff_allruns.

Across models, PRIMARY (model level, one H1 diff per model, n = 6): bootstrap 95% CI
resampling models only, and a two-sided one-sample t-test against 0 (model_level.py).
Sensitivity: cluster bootstrap (resample models, then runs within each); one-sided
sign test on the number of models with diff > 0.
Mistral Large's I2 answers come from MISTRAL_I2_SOURCE (data_sources.py; default
re-collected, mistral_i2_recollect.json); the other source is the sensitivity row.
Also reproduces the original 15-point paired t-test (five original models, original
Mistral I2 answers, as in the paper).

Writes results/stats/h1_model_level.csv and results/stats/h1_model_level.md.
No API calls; the embedding model loads from the local cache.

Usage:
    python scripts/stats_h1.py
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# The installed torchvision doesn't match torch and breaks `import transformers`;
# text embeddings don't use it, so hide it from this process only.
sys.modules["torchvision"] = None

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

import plot_response_results as prp
from answer_text import get_answer_text
from data_sources import LABELS, SOURCES, load_i2, mistral_i2_source, other_source
from model_level import model_level_inference, update_summary

OUT_DIR        = Path("results/stats")
MODELS        = ["gpt-4o", "claude-sonnet", "deepseek-v3", "mistral-large",
                  "gemini-3.1-flash-lite-preview", "grok-4.7"]
ORIGINAL_FIVE  = MODELS[:5]
DIM_PAIRS      = [("I1_Q1", "I2_S1"), ("I1_Q2", "I2_S2"), ("I1_Q3", "I2_S3")]
RUNS           = [1, 2, 3]
N_BOOT         = 10_000
SEED           = 42


def cell_text(data: dict, model: str, run: int, qid: str) -> str:
    cell = data.get(model, {}).get("baseline", {}).get(str(run), {}).get(qid)
    return get_answer_text(cell) if cell is not None else ""


def similarity_tables(i1: dict, i2: dict, model: str, embed: SentenceTransformer) -> dict:
    """
    Per question pair: W = 3x3 within-I1 cosine matrix (runs batch-encoded, as H1),
    C = run-matched I1->I2 cosines for runs 1-3 (each text encoded singly, as H1).
    """
    tables = []
    for i1_q, i2_q in DIM_PAIRS:
        i1_texts = [cell_text(i1, model, r, i1_q) for r in RUNS]
        i2_texts = [cell_text(i2, model, r, i2_q) for r in RUNS]
        if not all(i1_texts) or not all(i2_texts):
            raise ValueError(f"{model} {i1_q}/{i2_q}: missing baseline run answers")
        batch = embed.encode(i1_texts, show_progress_bar=False)
        W = np.array([[prp.cosine_sim(batch[a], batch[b]) for b in range(3)] for a in range(3)])
        single = lambda t: embed.encode([t], show_progress_bar=False)[0]
        C = np.array([prp.cosine_sim(single(a), single(b)) for a, b in zip(i1_texts, i2_texts)])
        tables.append((W, C))
    return tables


def h1_point(tables) -> tuple[float, float]:
    """Exactly H1's estimator: within over all 3 runs, cross from run 1; mean over questions."""
    within = np.mean([np.mean([W[a, b] for a, b in combinations(range(3), 2)]) for W, _ in tables])
    cross  = np.mean([C[0] for _, C in tables])
    return float(within), float(cross)


def resampled_diff(tables, idx: np.ndarray) -> float:
    pairs = [(a, b) for a, b in combinations(range(len(idx)), 2) if idx[a] != idx[b]]
    within = np.mean([np.mean([W[idx[a], idx[b]] for a, b in pairs]) for W, _ in tables])
    cross  = np.mean([np.mean(C[idx]) for _, C in tables])
    return float(within - cross)


def draw_runs(rng) -> np.ndarray:
    while True:                       # redraw resamples with no distinct pair
        idx = rng.integers(0, 3, size=3)
        if len(set(idx)) > 1:
            return idx


def model_bootstrap(tables) -> tuple[float, float, int]:
    rng = np.random.default_rng(SEED)
    boots = np.array([resampled_diff(tables, draw_runs(rng)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), len(np.unique(boots.round(12)))


def cluster_bootstrap(all_tables: dict) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    keys = list(all_tables)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        picked = rng.integers(0, len(keys), size=len(keys))
        means[i] = np.mean([resampled_diff(all_tables[keys[k]], draw_runs(rng)) for k in picked])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def analyse(scenario: str, all_tables: dict) -> list[dict]:
    rows, diffs = [], []
    for m, tables in all_tables.items():
        within, cross = h1_point(tables)
        diff_all = resampled_diff(tables, np.array([0, 1, 2]))
        lo, hi, distinct = model_bootstrap(tables)
        diffs.append(within - cross)
        rows.append(dict(analysis="per_model", scenario=scenario, model=m,
                         within=within, cross=cross, diff=within - cross,
                         diff_allruns=diff_all, ci_low=lo, ci_high=hi,
                         n_boot=N_BOOT, distinct_boot_values=distinct))
    lo, hi = cluster_bootstrap(all_tables)
    k = int(sum(d > 0 for d in diffs))
    rows.append(dict(analysis="across_models", scenario=scenario, model=f"{len(diffs)} models",
                     within=float(np.mean([r["within"] for r in rows])),
                     cross=float(np.mean([r["cross"] for r in rows])),
                     diff=float(np.mean(diffs)),
                     diff_allruns=float(np.mean([r["diff_allruns"] for r in rows])),
                     **model_level_inference(diffs),          # primary
                     ci_low=lo, ci_high=hi, n_boot=N_BOOT,    # sensitivity: cluster bootstrap
                     sign_k_positive=k, sign_n=len(diffs),
                     sign_p_one_sided=float(stats.binom.sf(k - 1, len(diffs), 0.5))))
    return rows


def ttest_15(all_tables: dict) -> dict:
    """The original H1 paired t-test: one point per (model, question pair), five models."""
    within, cross = [], []
    for m in ORIGINAL_FIVE:
        for W, C in all_tables[m]:
            within.append(np.mean([W[a, b] for a, b in combinations(range(3), 2)]))
            cross.append(C[0])
    t, p = stats.ttest_rel(within, cross)
    return dict(analysis="ttest_15_original_five", scenario="original", model="5 models x 3 pairs",
                within=float(np.mean(within)), cross=float(np.mean(cross)),
                diff=float(np.mean(within) - np.mean(cross)), t=float(t), p_t=float(p),
                n_points=len(within))


def write_outputs(rows: list[dict], primary: str, sensitivity: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["analysis", "scenario", "model", "within", "cross", "diff", "diff_allruns",
            "n_models", "sd", "t", "df", "p_t", "t_ci_low", "t_ci_high",
            "model_ci_low", "model_ci_high",
            "model_resamples_possible",
            "ci_low", "ci_high", "n_boot", "distinct_boot_values", "sign_k_positive", "sign_n",
            "sign_p_one_sided", "n_points"]
    with open(OUT_DIR / "h1_model_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})

    f4 = lambda v: f"{v:.4f}"
    md = ["# H1, model level", "",
          "Generated by `scripts/stats_h1.py`. Baseline condition; all-MiniLM-L6-v2 embeddings.",
          "`within`, `cross`, `diff` use the H1 estimator (cross from run 1). "
          "`diff_allruns` uses cross averaged over runs 1-3; bootstrap CIs (10,000 resamples, "
          "seed 42, runs resampled within model) are for this all-runs estimator.", ""]
    for scenario, title in [(primary, f"Primary ({LABELS[primary]})"),
                            (sensitivity, f"Sensitivity ({LABELS[sensitivity]})")]:
        md += [f"## {title}", "",
               "| Model | within | cross | diff | diff_allruns | 95% CI (diff_allruns) |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            if r["scenario"] != scenario or r["analysis"] not in ("per_model", "across_models"):
                continue
            name = f"**{r['model']} (mean)**" if r["analysis"] == "across_models" else r["model"]
            md.append(f"| {name} | {f4(r['within'])} | {f4(r['cross'])} | {f4(r['diff'])} | "
                      f"{f4(r['diff_allruns'])} | [{f4(r['ci_low'])}, {f4(r['ci_high'])}] |")
        a = next(r for r in rows if r["scenario"] == scenario and r["analysis"] == "across_models")
        md += ["", f"**Primary (model level, n = {a['n_models']}):** mean diff = {f4(a['diff'])}, "
                   f"SD = {f4(a['sd'])}, 95% CI (t) = [{f4(a['t_ci_low'])}, {f4(a['t_ci_high'])}], "
                   f"95% CI (bootstrap over models, {a['model_resamples_possible']} "
                   f"distinct resamples) = [{f4(a['model_ci_low'])}, {f4(a['model_ci_high'])}], "
                   f"t({a['df']}) = {a['t']:.3f}, p = {a['p_t']:.3g} (two-sided).", "",
               f"Sensitivity: cluster bootstrap (models, then runs) CI = "
               f"[{f4(a['ci_low'])}, {f4(a['ci_high'])}] (the CI in the mean row above); "
               f"sign test {a['sign_k_positive']}/{a['sign_n']} models with diff > 0, "
               f"one-sided p = {a['sign_p_one_sided']:.4f}.", ""]
    t = next(r for r in rows if r["analysis"] == "ttest_15_original_five")
    md += ["## Original 15-point paired t-test (five original models, original Mistral I2)", "",
           f"within = {f4(t['within'])}, cross = {f4(t['cross'])}, t = {t['t']:.3f}, "
           f"p = {t['p_t']:.3g}, n = {t['n_points']}", ""]
    (OUT_DIR / "h1_model_level.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    primary, sensitivity = mistral_i2_source(), other_source()
    i1 = prp._try_load("instrument_1")
    i2 = {s: load_i2(s) for s in SOURCES}

    embed = SentenceTransformer(prp.EMBEDDING_MODEL)
    others = {m: similarity_tables(i1, i2[primary], m, embed) for m in MODELS if m != "mistral-large"}
    tables = {s: {m: (others[m] if m != "mistral-large"
                      else similarity_tables(i1, i2[s], m, embed)) for m in MODELS}
              for s in SOURCES}

    rows = analyse(primary, tables[primary]) + analyse(sensitivity, tables[sensitivity])
    rows.append(ttest_15(tables["original"]))
    write_outputs(rows, primary, sensitivity)

    across = {s: next(r for r in rows if r["scenario"] == s and r["analysis"] == "across_models")
              for s in (primary, sensitivity)}
    update_summary([{**a, "test": "H1 diff" + ("" if s == primary else f" ({LABELS[s]})"),
                     "mean": a["diff"], "h1_cluster_ci_low": a["ci_low"],
                     "h1_cluster_ci_high": a["ci_high"]} for s, a in across.items()])

    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved {OUT_DIR / 'h1_model_level.csv'} and {OUT_DIR / 'h1_model_level.md'}")
