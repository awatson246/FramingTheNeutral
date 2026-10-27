"""
I5 source legitimacy with all six models (original five + grok-4.7).

Uses the same code paths as the paper's I5 figures and the H4 test in
plot_response_results.py:
  self-reported (I1 sources):  get_i1_sources; tier proxy via SOURCE_TYPE_TO_TIER
                               (missing type -> implicit_only -> tier 4);
                               jurisdictions as in i1_jurisdiction_breakdown
                               ("other" = non-empty and not EU/US/UN/UK/unspecified)
  third-party (I5 extractor):  get_i5_sources; tier % as in i5_source_legitimacy_heatmap;
                               mean tier as in H4 (missing tier -> 4); jurisdictions as in
                               i5_jurisdiction_radar ("other" = 100% minus the named ones)
  H4 Spearman:                 mean third-party tier vs I3 baseline enforceability
                               variance (np.var) and mean, from extract_i3_scores
                               (unrecovered I3, as in the paper); a recovered-I3
                               version is reported as a sensitivity check.

Mistral's I2 sources follow MISTRAL_I2_SOURCE (data_sources.load_i5; default
re-collected answers, extracted by i5_mistral_recollect.py); the other source is
reported as a sensitivity section, and the five-model reproduction uses the
original answers.

The original five's paper values come from results/hypothesis_tests/results.json
(mean tier, % Tier 3-4) and results/instrument_5/extracted_*.html (tier % and
jurisdiction %).

Writes results/stats/i5_six_models.csv, results/stats/i5_six_models.md and figures
in results/stats/i5_six_models_figs/ (existing figures are not touched).
No API calls; reads data/ only.

Usage:
    python scripts/stats_i5.py
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules["torchvision"] = None   # torch/torchvision mismatch; not needed here

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

import plot_response_results as prp
import stats_i3_ceo as i3
from data_sources import LABELS, load_i5, mistral_i2_source, other_source

OUT_DIR  = Path("results/stats")
FIG_DIR  = OUT_DIR / "i5_six_models_figs"
OLD_H4   = Path("results/hypothesis_tests/results.json")
OLD_TIER = Path("results/instrument_5/extracted_source_legitimacy_heatmap.html")
OLD_JUR  = Path("results/instrument_5/extracted_jurisdiction_radar.html")
MODELS   = ["gpt-4o", "claude-sonnet", "deepseek-v3", "mistral-large",
            "gemini-3.1-flash-lite-preview", "grok-4.7"]
NAMED    = ["EU", "US", "UN", "UK", "unspecified"]


def plotly_traces(html_path: Path) -> list[dict]:
    html = html_path.read_text(encoding="utf-8")
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\])\s*,\s*\{"', html, re.S)
    return json.loads(m.group(1))


def jur(s: dict) -> str:
    return (s.get("jurisdiction") or "").strip()


def self_reported(i1: dict, m: str) -> dict:
    src = prp.get_i1_sources(i1, m)
    n = len(src) or 1
    tiers = [prp.SOURCE_TYPE_TO_TIER.get(s.get("type", "implicit_only"), 4) for s in src]
    out = {"n_self": len(src),
           "self_mean_tier": float(np.mean(tiers)) if tiers else None,
           **{f"self_tier{t}_pct": sum(x == t for x in tiers) / n * 100 for t in [1, 2, 3, 4]},
           **{f"self_type_{k}_pct": sum(s.get("type") == k for s in src) / n * 100
              for k in prp.SOURCE_TYPE_LABELS}}
    for j in NAMED:
        out[f"self_{j}_pct"] = sum(jur(s).upper() == j.upper() for s in src) / n * 100
    out["self_other_pct"] = sum(jur(s).upper() not in {x.upper() for x in NAMED} and jur(s) != ""
                                for s in src) / n * 100
    top = Counter(jur(s).upper() or "(blank)" for s in src).most_common(1)
    out["self_top_jurisdiction"] = top[0][0] if top else None
    return out


def third_party(i5: dict, m: str) -> dict:
    src = prp.get_i5_sources(i5, m)
    n = len(src) or 1
    tiers = [s.get("legitimacy_tier", 4) for s in src]
    out = {"n_third": len(src), "third_mean_tier": float(np.mean(tiers)) if tiers else None,
           **{f"third_tier{t}_pct": sum(s.get("legitimacy_tier") == t for s in src) / n * 100
              for t in [1, 2, 3, 4]},
           "third_tier34_h4_pct": sum(t >= 3 for t in tiers) / n * 100,
           "third_unverifiable_pct": sum(not s.get("verifiable", True) for s in src) / n * 100}
    for j in NAMED:
        out[f"third_{j}_pct"] = sum(jur(s).upper() == j.upper() for s in src) / n * 100
    out["third_other_pct"] = max(0.0, 100 - sum(out[f"third_{j}_pct"] for j in NAMED))
    top = Counter(jur(s).upper() or "(blank)" for s in src).most_common(1)
    out["third_top_jurisdiction"] = top[0][0] if top else None
    return out


def enforceability(i3_scores: dict, m: str) -> tuple[float, float]:
    vals = []
    for s_id in prp.I3_SCENARIOS:
        vals.extend(i3_scores.get(m, {}).get("baseline", {}).get(s_id, {}).get("enforceability", []))
    return float(np.mean(vals)), float(np.var(vals))


def recovered_enforceability() -> dict:
    data = json.loads(i3.I3_FILE.read_text(encoding="utf-8"))
    rec = json.loads(i3.RECOVERED_FILE.read_text(encoding="utf-8"))["recovered"]
    strata, _ = i3.load_strata(data, rec)
    out = {}
    for m in MODELS:
        v = np.vstack([strata[m]["baseline"][s] for s in i3.SCENARIOS])[:, i3.DIMS.index("enforceability")]
        v = v[~np.isnan(v)]
        out[m] = (float(v.mean()), float(v.var()))
    return out


def model_rows(i1: dict, i5: dict, i3_scores: dict, rec_enf: dict, source: str) -> list[dict]:
    rows = []
    for m in MODELS:
        e_mean, e_var = enforceability(i3_scores, m)
        rows.append({"model": m, "label": prp.MODEL_LABELS.get(m, m), "mistral_i2_source": source,
                     **self_reported(i1, m), **third_party(i5, m),
                     "enf_mean": e_mean, "enf_var": e_var,
                     "enf_mean_recovered": rec_enf[m][0], "enf_var_recovered": rec_enf[m][1]})
    return rows


def spearman(rows: list[dict]) -> dict:
    spear = {}
    for tag, key_mean, key_var in [("paper_path", "enf_mean", "enf_var"),
                                   ("recovered_i3", "enf_mean_recovered", "enf_var_recovered")]:
        tier = [r["third_mean_tier"] for r in rows]
        rv, pv = stats.spearmanr(tier, [r[key_var] for r in rows])
        rm, pm = stats.spearmanr(tier, [r[key_mean] for r in rows])
        spear[tag] = dict(var=(float(rv), float(pv)), mean=(float(rm), float(pm)), n=len(rows))
    return spear


def claim_checks(rows: list[dict]) -> dict:
    named_max = lambda r, side: max(["EU", "US", "UN", "UK"], key=lambda j: r[f"{side}_{j}_pct"])
    return {
        "EU most common (third-party, among EU/US/UN/UK)": {r["model"]: named_max(r, "third") == "EU" for r in rows},
        "EU most common (third-party, most frequent raw label)": {r["model"]: r["third_top_jurisdiction"] == "EU" for r in rows},
        "EU most common (self-reported, among EU/US/UN/UK)": {r["model"]: named_max(r, "self") == "EU" for r in rows},
        "EU most common (self-reported, most frequent raw label)": {r["model"]: r["self_top_jurisdiction"] == "EU" for r in rows},
        "Self-reported mostly Tier 1-2 (> 50%)": {r["model"]: r["self_tier1_pct"] + r["self_tier2_pct"] > 50 for r in rows},
        "Third-party mean tier worse than self-reported proxy": {r["model"]: r["third_mean_tier"] > r["self_mean_tier"] for r in rows},
        "Third-party % Tier 3-4 higher than self-reported": {r["model"]: (r["third_tier3_pct"] + r["third_tier4_pct"])
                                                             > (r["self_tier3_pct"] + r["self_tier4_pct"]) for r in rows},
    }


if __name__ == "__main__":
    primary, sensitivity = mistral_i2_source(), other_source()
    i1, i3_raw = prp._try_load("instrument_1"), prp._try_load("instrument_3")
    i5 = load_i5(primary)
    i3_scores = prp.extract_i3_scores(i3_raw, MODELS)
    rec_enf = recovered_enforceability()

    rows = model_rows(i1, i5, i3_scores, rec_enf, primary)
    sens_rows = model_rows(i1, load_i5(sensitivity), i3_scores, rec_enf, sensitivity)

    # ---- paper values for the original five ----
    old_h4 = {r["Model"]: r for r in json.loads(OLD_H4.read_text(encoding="utf-8"))["h4"]["per_model"]}
    tier_trace = plotly_traces(OLD_TIER)[0]
    old_tiers = {lab: z for lab, z in zip(tier_trace["y"], tier_trace["z"])}
    old_jur = {t["name"]: dict(zip(t["theta"], t["r"])) for t in plotly_traces(OLD_JUR)}
    for r in rows:
        lab = r["label"]
        if lab in old_h4:
            r["paper_mean_tier"] = old_h4[lab]["Mean Legitimacy Tier"]
            r["paper_tier34_pct"] = old_h4[lab]["% Tier 3–4 Sources"]
            r["paper_enf_var"] = old_h4[lab]["Enforceability Var."]
        if lab in old_tiers:
            r["paper_tier12_pct"] = old_tiers[lab][0] + old_tiers[lab][1]
        if lab in old_jur:
            for j in ["EU", "US", "other"]:
                r[f"paper_{j}_pct"] = old_jur[lab].get(j)

    # ---- Spearman (H4), six models ----
    spear, sens_spear = spearman(rows), spearman(sens_rows)
    # five-model reproduction of the paper: original Mistral I2 extractions
    orig_rows = rows if primary == "original" else sens_rows
    five = orig_rows[:5]
    rv5, pv5 = stats.spearmanr([r["third_mean_tier"] for r in five], [r["enf_var"] for r in five])

    # ---- claim checks ----
    claims, sens_claims = claim_checks(rows), claim_checks(sens_rows)

    # ---- figures (same functions as the paper's figures) ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    h4_stats = {"spearman_tier_enf": {"rho": spear["paper_path"]["mean"][0], "p": spear["paper_path"]["mean"][1]}}
    for fig, name in [(prp.i1_source_type_stacked_bar(i1, MODELS), "i1_source_type_distribution"),
                      (prp.i1_source_legitimacy_proxy_heatmap(i1, MODELS), "i1_source_legitimacy_proxy"),
                      (prp.i1_jurisdiction_breakdown(i1, MODELS), "i1_jurisdiction_breakdown"),
                      (prp.i5_source_legitimacy_heatmap(i5, MODELS), "extracted_source_legitimacy_heatmap"),
                      (prp.i5_jurisdiction_radar(i5, MODELS, h4_stats), "extracted_jurisdiction_radar")]:
        prp.save_fig(fig, FIG_DIR / name)

    # ---- CSV ----
    cols = list(rows[0].keys()) + [c for c in ["paper_mean_tier", "paper_tier12_pct", "paper_tier34_pct",
                                               "paper_EU_pct", "paper_US_pct", "paper_other_pct",
                                               "paper_enf_var"] if c not in rows[0]]
    with open(OUT_DIR / "i5_six_models.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        # primary rows, then Mistral under the sensitivity source (other models identical)
        for r in rows + [r for r in sens_rows if r["model"] == "mistral-large"]:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    # ---- markdown ----
    f2 = lambda v: "—" if v is None else f"{v:.2f}"
    f1 = lambda v: "—" if v is None else f"{v:.1f}"
    md = ["# I5 source legitimacy, six models", "",
          "Generated by `scripts/stats_i5.py`; same code paths as plot_response_results.py. "
          f"Primary: {LABELS[primary]} (MISTRAL_I2_SOURCE={primary}); sensitivity at the end. "
          "Paper values: results/hypothesis_tests/results.json and results/instrument_5/extracted_*.html "
          "(original Mistral I2).", "",
          "## Third-party (I5 extractor): recomputed vs paper", "",
          "| Model | sources | mean tier (paper) | % Tier 1-2 (paper) | % Tier 3-4 (paper) | "
          "% EU (paper) | % US (paper) | % other (paper) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['n_third']} | {f2(r['third_mean_tier'])} ({f2(r.get('paper_mean_tier'))}) | "
                  f"{f1(r['third_tier1_pct'] + r['third_tier2_pct'])} ({f1(r.get('paper_tier12_pct'))}) | "
                  f"{f1(r['third_tier3_pct'] + r['third_tier4_pct'])} ({f1(r.get('paper_tier34_pct'))}) | "
                  f"{f1(r['third_EU_pct'])} ({f1(r.get('paper_EU_pct'))}) | "
                  f"{f1(r['third_US_pct'])} ({f1(r.get('paper_US_pct'))}) | "
                  f"{f1(r['third_other_pct'])} ({f1(r.get('paper_other_pct'))}) |")
    md += ["", "% Tier 3-4 here = tiers 3 and 4 exactly; the paper's H4 value counts tier >= 3 with a "
           "missing tier as 4 (third_tier34_h4_pct in the CSV).", "",
           "## Self-reported (I1) vs third-party (I5)", "",
           "| Model | self sources | self mean tier (proxy) | self % Tier 1-2 | third mean tier | "
           "self % EU | self % US | self % other |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['n_self']} | {f2(r['self_mean_tier'])} | "
                  f"{f1(r['self_tier1_pct'] + r['self_tier2_pct'])} | {f2(r['third_mean_tier'])} | "
                  f"{f1(r['self_EU_pct'])} | {f1(r['self_US_pct'])} | {f1(r['self_other_pct'])} |")
    md += ["", "## H4 Spearman (mean third-party tier vs I3 baseline enforceability)", "",
           "| Data | n | rho (variance) | p | rho (mean) | p |", "|---|---|---|---|---|---|"]
    for tag, label in [("paper_path", "six models, unrecovered I3 (paper path)"),
                       ("recovered_i3", "six models, recovered I3")]:
        s = spear[tag]
        md.append(f"| {label} | {s['n']} | {s['var'][0]:.3f} | {s['var'][1]:.4f} | "
                  f"{s['mean'][0]:.3f} | {s['mean'][1]:.4f} |")
    md.append(f"| five models, unrecovered I3 (reproduction, original Mistral I2) | 5 | {rv5:.3f} | {pv5:.4f} | | |")
    md += ["", "## Claim checks (per model)", "",
           "| Claim | " + " | ".join(MODELS) + " | holds for all |", "|---|" + "---|" * (len(MODELS) + 1)]
    for c, res in claims.items():
        md.append(f"| {c} | " + " | ".join("yes" if res[m] else "**no**" for m in MODELS)
                  + f" | {'yes' if all(res.values()) else 'no'} |")
    md += ["", "Most frequent raw jurisdiction label per model: third-party "
           + ", ".join(f"{r['model']}={r['third_top_jurisdiction']}" for r in rows)
           + "; self-reported " + ", ".join(f"{r['model']}={r['self_top_jurisdiction']}" for r in rows) + ".", ""]

    pm = next(r for r in rows if r["model"] == "mistral-large")
    sm = next(r for r in sens_rows if r["model"] == "mistral-large")
    md += [f"## Sensitivity: {LABELS[sensitivity]}", "",
           "Only Mistral's I2 extractions differ; other models are identical.", "",
           f"| Mistral third-party | {LABELS[primary]} (primary) | {LABELS[sensitivity]} |", "|---|---|---|"]
    for key, label, fmt in [("n_third", "sources", str), ("third_mean_tier", "mean tier", f2),
                            ("third_tier34_h4_pct", "% Tier 3-4", f1), ("third_EU_pct", "% EU", f1),
                            ("third_US_pct", "% US", f1), ("third_other_pct", "% other", f1),
                            ("third_unspecified_pct", "% unspecified", f1),
                            ("third_top_jurisdiction", "most frequent label", str)]:
        md.append(f"| {label} | {fmt(pm[key])} | {fmt(sm[key])} |")
    md += ["", "| H4 Spearman | rho (variance) | p | rho (mean) | p |", "|---|---|---|---|---|"]
    for name, s in [(f"{LABELS[primary]} (primary)", spear), (LABELS[sensitivity], sens_spear)]:
        s = s["paper_path"]
        md.append(f"| {name} | {s['var'][0]:.3f} | {s['var'][1]:.4f} | {s['mean'][0]:.3f} | {s['mean'][1]:.4f} |")
    md += ["", "| Claim | holds for all, primary | holds for all, sensitivity | Mistral, primary | Mistral, sensitivity |",
           "|---|---|---|---|---|"]
    for c in claims:
        yn = lambda b: "yes" if b else "no"
        md.append(f"| {c} | {yn(all(claims[c].values()))} | {yn(all(sens_claims[c].values()))} | "
                  f"{yn(claims[c]['mistral-large'])} | {yn(sens_claims[c]['mistral-large'])} |")
    md.append("")
    (OUT_DIR / "i5_six_models.md").write_text("\n".join(md), encoding="utf-8")

    print((OUT_DIR / "i5_six_models.md").read_text(encoding="utf-8"))
    print(f"Saved {OUT_DIR / 'i5_six_models.csv'}, {OUT_DIR / 'i5_six_models.md'} and figures in {FIG_DIR}")
