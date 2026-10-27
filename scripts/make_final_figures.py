"""
Final paper figures (Figures 1-10) from the final data, written to
results/figures_final/ as HTML and PNG, plus compare.html (old PNG next to new)
and checks.md (figure numbers vs results/stats/). Existing figures elsewhere in
results/ are not touched.

Final data:
  all six models (MODELS in stats_i5), grok-4.7 included
  Mistral I2: MISTRAL_I2_SOURCE=recollect (data_sources.py; required)
  I3: parsed + recovered scores (stats_i3_ceo.load_strata, data/processed/i3_recovered.json)
  I4: primary data of stats_i4_asymmetry.py (P1-P5 + G1-G5, re-collected Mistral I2 cells)
  I5: first judge (extractor), re-collected Mistral I2 sources (load_i5)
Subtitle statistics come from results/stats/ (model level, n = 6).

Sankeys (Figures 3-4): extraction and label-merge calls are pinned to
claude-sonnet-4-5-20250929 and stored in data/processed/sankey_extractions.json
(plot_response_results.py); once that file is complete, no API calls are made.

Usage:
    python scripts/make_final_figures.py --dry-run   # count Sankey API calls, write nothing
    python scripts/make_final_figures.py
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules["torchvision"] = None   # torch/torchvision mismatch; not needed here

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import plot_response_results as prp
import stats_i3_ceo as i3
import stats_i4_asymmetry as i4a
from data_sources import load_i2, load_i5, require_source
from stats_i5 import MODELS

OUT_DIR   = Path("results/figures_final")
STATS_DIR = Path("results/stats")
DIMS      = i3.DIMS
LABELS    = [prp.MODEL_LABELS[m] for m in MODELS]

# number, output stem, old PNG (relative to results/), caption subject
FIGURES = [
    (1,  "fig01_similarity_cross_model",     "instrument_1/similarity_cross_model.png",         "Cross-model similarity"),
    (2,  "fig02_similarity_baseline_vs_ceo", "instrument_1/similarity_baseline_vs_ceo.png",     "Baseline vs CEO similarity"),
    (3,  "fig03_housing_sankey",             "instrument_2/s2_accountability_sankey.png",       "Housing Sankey"),
    (4,  "fig04_audit_sankey",               "instrument_2/s3_enforcement_sankey.png",          "Audit Sankey"),
    (5,  "fig05_i3_scores",                  "instrument_3/condition_side_by_side.png",         "I3 scores"),
    (6,  "fig06_i4_asymmetry",               "instrument_4/asymmetry_heatmap.png",              "I4 asymmetry"),
    (7,  "fig07_self_reported_source_types", "instrument_5/i1_source_type_distribution.png",    "Self-reported source types"),
    (8,  "fig08_self_reported_tiers",        "instrument_5/i1_source_legitimacy_proxy.png",     "Self-reported tiers"),
    (9,  "fig09_third_party_tiers",          "instrument_5/extracted_source_legitimacy_heatmap.png", "Third-party tiers"),
    (10, "fig10_regionality",                "instrument_5/extracted_jurisdiction_radar.png",   "Regionality"),
]


def read_csv(name: str) -> list[dict]:
    with open(STATS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_subtitle(fig, subtitle: str) -> None:
    title = fig.layout.title.text.split("<br>")[0]
    fig.update_layout(title_text=f"{title}<br><sup>{subtitle}</sup>")


def i3_scores_recovered() -> dict:
    """scores[model][cond][scenario][dim] -> list of run scores, parsed + recovered."""
    recovered = json.loads(i3.RECOVERED_FILE.read_text(encoding="utf-8"))["recovered"]
    strata, _ = i3.load_strata(json.loads(i3.I3_FILE.read_text(encoding="utf-8")), recovered)
    return {m: {c: {s: {d: [float(v) for v in strata[m][c][s][:, k] if not np.isnan(v)]
                        for k, d in enumerate(DIMS)}
                    for s in i3.SCENARIOS}
                for c in i3.CONDS}
            for m in MODELS}, strata


def i4_asymmetry(strata: dict) -> dict:
    """elp-shaped {model: {"asymmetry": {dim: value}}} from the primary I4 data."""
    load = i4a.load
    cells = i4a.peer_cells("primary", load(i4a.I4_FILE), load(i4a.I4_GROK_FILE),
                           load(i4a.I4_RESCORE), load(i4a.PAIRS_FILE)["pairs"],
                           load(i4a.GROK_PAIRS)["pairs"])
    pm, _ = i4a.peer_means(cells)
    i3m = i4a.i3_means(strata)
    return {m: {"asymmetry": {d: pm[m][d] - i3m[m][d] for d in DIMS}} for m in MODELS}


def sankeys(i2: dict) -> tuple:
    return (prp.i2_s2_accountability_sankey(i2, MODELS), prp.i2_s3_enforcement_sankey(i2, MODELS))


# ---------------------------------------------------------------------------
# Checks: figure numbers vs results/stats/
# ---------------------------------------------------------------------------

def i5_primary_rows() -> dict:
    """i5_six_models.csv rows for the primary Mistral I2 source (the CSV also has a sensitivity row)."""
    return {r["model"]: r for r in read_csv("i5_six_models.csv") if r["mistral_i2_source"] == "recollect"}


def check(name: str, pairs: list[tuple], tol: float) -> dict:
    """pairs: (cell label, figure value, stats value)."""
    devs = [(lab, abs(a - b), a, b) for lab, a, b in pairs]
    worst = max(devs, key=lambda x: x[1])
    bad = [d for d in devs if d[1] > tol]
    return {"check": name, "n": len(devs), "max_dev": worst[1], "tol": tol, "worst": worst, "bad": bad}


def run_checks(figs: dict, store: dict) -> tuple[list[dict], list[str]]:
    res, notes = [], []

    # every figure shows all six models
    for num, fig in figs.items():
        fj = fig.to_plotly_json()
        text = json.dumps([fj["data"], fj["layout"].get("annotations", [])], ensure_ascii=False)
        missing = [lab for lab in LABELS if lab not in text]
        if missing:
            notes.append(f"Figure {num}: models missing from the figure: {missing}")

    # Fig 2 vs i1_role_similarity.md (3 decimals)
    md = (STATS_DIR / "i1_role_similarity.md").read_text(encoding="utf-8")
    rows = {}
    for line in md.splitlines():          # first table only: model | three dimensions | ...
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 4 and cells[0] in MODELS and cells[0] not in rows:
            rows[cells[0]] = [float(x) for x in cells[1:4]]
    z = figs[2].data[0].z
    res.append(check("Fig 2 vs i1_role_similarity.md",
                     [(f"{m} q{qi+1}", z[qi][mi], rows[m][qi])
                      for qi in range(3) for mi, m in enumerate(MODELS)], 0.0005))

    # Fig 5: per model and condition, mean of the six scenario bars vs the mean over
    # dimensions of the per-dimension baseline/CEO means in i3_ceo_model_level.csv
    i3rows = {(r["model"], r["dimension"]): r for r in read_csv("i3_ceo_model_level.csv")
              if r["analysis"] == "per_model"}
    pairs = []
    for mi, m in enumerate(MODELS):
        for ci, (cond, col) in enumerate([("baseline", "baseline_mean"), ("ceo", "ceo_mean")]):
            bars = figs[5].data[mi * 2 + ci].y
            pairs.append((f"{m} {cond}", float(np.mean(bars)),
                          float(np.mean([float(i3rows[(m, d)][col]) for d in DIMS]))))
    res.append(check("Fig 5 vs i3_ceo_model_level.csv (model x condition means)", pairs, 1e-4))

    # Fig 6 vs i4_asymmetry_model_level.csv (primary)
    i4rows = {(r["model"], r["dimension"]): float(r["asymmetry"]) for r in read_csv("i4_asymmetry_model_level.csv")
              if r["analysis"] == "per_model" and r["variant"] == "primary"}
    z = figs[6].data[0].z
    res.append(check("Fig 6 vs i4_asymmetry_model_level.csv",
                     [(f"{m} {d}", z[mi][di], i4rows[(m, d)])
                      for mi, m in enumerate(MODELS) for di, d in enumerate(DIMS)], 1e-4))

    # Figs 7-10 vs i5_six_models.csv
    i5rows = i5_primary_rows()
    types = list(prp.SOURCE_TYPE_LABELS)
    res.append(check("Fig 7 vs i5_six_models.csv (self_type_*_pct)",
                     [(f"{m} {t}", figs[7].data[ti].y[mi], float(i5rows[m][f"self_type_{t}_pct"]))
                      for ti, t in enumerate(types) for mi, m in enumerate(MODELS)], 1e-3))
    for num, side in [(8, "self"), (9, "third")]:
        z = figs[num].data[0].z
        res.append(check(f"Fig {num} vs i5_six_models.csv ({side}_tier*_pct)",
                         [(f"{m} tier {t}", z[mi][t - 1], float(i5rows[m][f"{side}_tier{t}_pct"]))
                          for mi, m in enumerate(MODELS) for t in range(1, 5)], 1e-3))
    pairs = []
    for mi, m in enumerate(MODELS):
        r = figs[10].data[mi].r
        for ji, j in enumerate(prp.I5_JURISDICTIONS):
            pairs.append((f"{m} {j}", r[ji], float(i5rows[m][f"third_{j}_pct"])))
    res.append(check("Fig 10 vs i5_six_models.csv (third_*_pct)", pairs, 1e-3))

    # Sankeys: one source node per model x condition; extractions all from the pinned model
    for num in (3, 4):
        nodes = list(figs[num].data[0].node.label)
        n_src = sum(1 for n in nodes if "(Baseline)" in n or "(CEO Role)" in n)
        if n_src != 2 * len(MODELS):
            notes.append(f"Figure {num}: {n_src} model x condition source nodes, expected {2 * len(MODELS)}")
    models_used = {e["model"] for e in store["extractions"].values()} | \
                  {e["model"] for e in store["label_maps"].values()}
    notes.append(f"Sankey store: {len(store['extractions'])} extractions, {len(store['label_maps'])} "
                 f"label maps; models used: {sorted(models_used)}")
    return res, notes


def caption_checks(i5_rows: dict) -> list[str]:
    """Figure 10 caption: EU most common among sources with an identifiable region."""
    out = []
    for m in MODELS:
        r = i5_rows[m]
        named = {j: float(r[f"third_{j}_pct"]) for j in ["EU", "US", "UN", "UK", "other"]}
        top = max(named, key=named.get)
        out.append(f"{m}: top identifiable region = {top} ({named[top]:.1f}%; EU {named['EU']:.1f}%, "
                   f"unspecified {float(r['third_unspecified_pct']):.1f}%)")
    return out


def compare_html() -> str:
    rows = []
    for num, stem, old, subject in FIGURES:
        old_src = f"../{old}"
        old_cell = (f'<img src="{old_src}" alt="old Figure {num}">' if (Path("results") / old).exists()
                    else "<p>(no old PNG)</p>")
        rows.append(f"<section><h2>Figure {num}: {subject}</h2><div class='pair'>"
                    f"<figure><figcaption>Old: results/{old}</figcaption>{old_cell}</figure>"
                    f"<figure><figcaption>New: results/figures_final/{stem}.png</figcaption>"
                    f'<img src="{stem}.png" alt="new Figure {num}"></figure></div></section>')
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Figures: old vs new</title>"
            "<style>body{font-family:Georgia,serif;margin:16px;background:#fff;color:#1f2328}"
            ".pair{display:flex;gap:16px;flex-wrap:wrap}figure{flex:1 1 480px;margin:0}"
            "img{max-width:100%;border:1px solid #d0d7de}figcaption{font-family:monospace;font-size:12px;"
            "margin-bottom:4px}section{margin-bottom:40px}</style></head><body>"
            "<h1>Paper figures: old vs final data</h1>" + "".join(rows) + "</body></html>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Count Sankey API calls; write nothing.")
    args = ap.parse_args()
    require_source("recollect", "Final figures")

    i2 = load_i2()
    if args.dry_run:
        prp.SANKEY_DRY_RUN = True
        sankeys(i2)
        ext = [k for k in prp.sankey_pending_calls if k.startswith("s2_") or k.startswith("s3_")]
        merges = [k for k in prp.sankey_pending_calls if k not in ext]
        # label merges depend on extraction output: two per Sankey once extractions exist
        n_merge = 4 if ext else len(merges)
        print(f"Dry run ({prp.SANKEY_LLM_MODEL}): {len(ext)} extraction calls + "
              f"{n_merge} label-merge calls = {len(ext) + n_merge} calls. Nothing written.")
        sys.exit(0)

    i1 = prp._try_load("instrument_1")
    i5 = load_i5()
    embed = SentenceTransformer(prp.EMBEDDING_MODEL)
    scores, strata = i3_scores_recovered()
    elp = i4_asymmetry(strata)

    h1 = next(r for r in read_csv("h1_model_level.csv")
              if r["analysis"] == "across_models" and r["scenario"] == "recollect")
    i4_across = {r["dimension"]: r for r in read_csv("i4_asymmetry_model_level.csv")
                 if r["analysis"] == "across_models" and r["variant"] == "primary"}
    i5_md = (STATS_DIR / "i5_six_models.md").read_text(encoding="utf-8")
    h4 = re.search(r"\| six models, recovered I3 \| 6 \| [-\d.]+ \| [\d.]+ \| ([-\d.]+) \| ([\d.]+) \|", i5_md)

    figs = {}
    figs[1] = prp.i1_cross_model_similarity(i1, MODELS, embed)
    figs[2] = prp.i1_baseline_vs_ceo(i1, MODELS, embed)
    set_subtitle(figs[2], "Lower similarity = greater role-induced framing change<br>"
                 f"H1, model level (n=6): within-I1 minus I1→I2 similarity = {float(h1['diff']):.3f}, "
                 f"t({h1['df']}) = {float(h1['t']):.2f}, p = {float(h1['p_t']):.1e}")
    figs[3], figs[4] = sankeys(i2)
    figs[5] = prp.i3_condition_bars(scores, MODELS)
    figs[6] = prp.i4_asymmetry_heatmap(elp, MODELS)
    set_subtitle(figs[6], "Asymmetry = mean I4 score given to peers minus own mean I3 baseline score · "
                 "red = more lenient with peers, blue = stricter<br>Across models (n=6), one-sample t-test: "
                 + " | ".join(f"{prp.I3_DIMENSIONS[d]} {float(i4_across[d]['mean']):+.2f} "
                              f"(p = {float(i4_across[d]['p_t']):.1e})" for d in DIMS))
    figs[7] = prp.i1_source_type_stacked_bar(i1, MODELS)
    figs[8] = prp.i1_source_legitimacy_proxy_heatmap(i1, MODELS)
    set_subtitle(figs[8], "Tier derived from source type — % of each model's citations; "
                 "blue = small share, red = large share")
    figs[9] = prp.i5_source_legitimacy_heatmap(i5, MODELS)
    set_subtitle(figs[9], "% of citations in each legitimacy tier per model (first judge); "
                 "blue = small share, red = large share")
    figs[10] = prp.i5_jurisdiction_radar(
        i5, MODELS, {"spearman_tier_enf": {"rho": float(h4.group(1)), "p": float(h4.group(2))}})
    set_subtitle(figs[10], "% of citations from each jurisdiction (first judge)<br>"
                 f"H4 Spearman ρ, mean third-party tier vs mean I3 baseline enforceability "
                 f"(n=6, recovered I3): ρ = {float(h4.group(1)):.3f}, p = {float(h4.group(2)):.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for num, stem, _, _ in FIGURES:
        prp.save_fig(figs[num], OUT_DIR / stem)
    (OUT_DIR / "compare.html").write_text(compare_html(), encoding="utf-8")

    store = json.loads(prp.SANKEY_EXTRACTIONS_FILE.read_text(encoding="utf-8"))
    results, notes = run_checks(figs, store)
    i5_rows = i5_primary_rows()
    md = ["# Final figures: checks", "", "Generated by `scripts/make_final_figures.py`.", "",
          "| Check | cells | max abs deviation | tolerance | result |", "|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['check']} | {r['n']} | {r['max_dev']:.2e} | {r['tol']:g} | "
                  f"{'ok' if not r['bad'] else 'MISMATCH: ' + '; '.join(f'{b[0]} fig {b[2]:.4f} vs stats {b[3]:.4f}' for b in r['bad'])} |")
    md += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    md += ["", "## Figure 10 caption check (third-party, first judge)", ""] + \
          [f"- {line}" for line in caption_checks(i5_rows)]
    (OUT_DIR / "checks.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
