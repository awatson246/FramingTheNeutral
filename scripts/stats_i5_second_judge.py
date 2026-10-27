"""
I5 source tiers: original extractor (claude-sonnet-4-5, data/raw/instrument_5.json)
vs the second judge (gpt-6.1-sol, data/processed/i5_second_judge.json). No API calls.

Unit: one extracted source. Agreement on legitimacy_tier (1-4): percent agreement,
unweighted Cohen's kappa, linearly weighted kappa, 4x4 confusion matrix; also
agreement on verifiable.
Per model and judge: mean tier and % Tier 3-4 over the sources that judge tiered
(for the extractor this equals third_mean_tier / third_tier34_h4_pct in
results/stats/i5_six_models.csv).
Claim check per judge: third-party % Tier 3-4 > self-reported % Tier 3-4, and
third-party mean tier > self-reported mean tier. Self-reported values (I1 source
types mapped through SOURCE_TYPE_TO_TIER) are read from i5_six_models.csv.

Writes results/stats/i5_second_judge.csv and .md.

Usage:
    python scripts/stats_i5_second_judge.py
"""

import csv
import json
from pathlib import Path

import numpy as np

from data_sources import LABELS, load_i5, load_i5_judge, mistral_i2_source, other_source
from i5_second_judge import CONDITIONS, MODELS, QUESTIONS, RUNS

SIX_MODELS_CSV = Path("results/stats/i5_six_models.csv")
OUT_DIR        = Path("results/stats")
TIERS          = [1, 2, 3, 4]
J1, J2         = "claude-sonnet-4-5 (extractor)", "gpt-6.1-sol (second judge)"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_units(i5: dict, judge: dict) -> list[dict]:
    units = []
    for m in MODELS:
        for c in CONDITIONS:
            for r in RUNS:
                for q in QUESTIONS:
                    sources = i5[m][c][r][q]["parsed"]["sources"]
                    cell = judge.get(m, {}).get(c, {}).get(r, {}).get(q, {})
                    j2 = {s["i"]: s for s in (cell.get("parsed") or {}).get("sources", [])}
                    for i, s in enumerate(sources, 1):
                        units.append({"model": m, "t1": s.get("legitimacy_tier"),
                                      "v1": s.get("verifiable"),
                                      "t2": j2.get(i, {}).get("legitimacy_tier"),
                                      "v2": j2.get(i, {}).get("verifiable")})
    return units


def kappa(a: list, b: list, cats: list, weights: str | None = None) -> float:
    k = len(cats)
    idx = {c: n for n, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    if weights == "linear":
        w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    else:
        w = 1 - np.eye(k)
    return float(1 - (w * obs).sum() / (w * exp).sum())


def tier_stats(tiers: list) -> dict:
    return {"n": len(tiers), "mean_tier": float(np.mean(tiers)),
            "tier34_pct": 100 * sum(t >= 3 for t in tiers) / len(tiers),
            **{f"tier{t}_pct": 100 * sum(x == t for x in tiers) / len(tiers) for t in TIERS}}


def analyse(i5: dict, judge: dict, six: dict, source: str) -> tuple[dict, dict, list]:
    units = paired_units(i5, judge)
    both = [u for u in units if u["t1"] in TIERS and u["t2"] in TIERS]
    a, b = [u["t1"] for u in both], [u["t2"] for u in both]
    vboth = [u for u in units if isinstance(u["v1"], bool) and isinstance(u["v2"], bool)]

    agree = {"n_sources": len(units), "n_paired": len(both),
             "n_missing_second": sum(u["t2"] not in TIERS for u in units),
             "pct_agree": 100 * sum(x == y for x, y in zip(a, b)) / len(both),
             "pct_within_one": 100 * sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(both),
             "kappa_unweighted": kappa(a, b, TIERS),
             "kappa_linear": kappa(a, b, TIERS, "linear"),
             "pct_agree_tier34_binary": 100 * sum((x >= 3) == (y >= 3) for x, y in zip(a, b)) / len(both),
             "kappa_tier34_binary": kappa([x >= 3 for x in a], [y >= 3 for y in b], [True, False]),
             "n_verifiable_paired": len(vboth),
             "pct_agree_verifiable": 100 * sum(u["v1"] == u["v2"] for u in vboth) / len(vboth),
             "kappa_verifiable": kappa([u["v1"] for u in vboth], [u["v2"] for u in vboth], [True, False])}
    confusion = {x: {y: sum(1 for p, q in zip(a, b) if p == x and q == y) for y in TIERS} for x in TIERS}

    rows = []
    for m in MODELS:
        ref = six[(m, source)] if (m, source) in six else six[(m, None)]
        self_34 = float(ref["self_tier3_pct"]) + float(ref["self_tier4_pct"])
        self_mean = float(ref["self_mean_tier"])
        for name, key in [(J1, "t1"), (J2, "t2")]:
            s = tier_stats([u[key] for u in units if u["model"] == m and u[key] in TIERS])
            rows.append({"model": m, "judge": name, "mistral_i2_source": source, **s,
                         "self_mean_tier": self_mean, "self_tier34_pct": self_34,
                         "third_34_gt_self": s["tier34_pct"] > self_34,
                         "third_mean_gt_self": s["mean_tier"] > self_mean,
                         "_six_mean": float(ref["third_mean_tier"]),
                         "_six_34": float(ref["third_tier34_h4_pct"])})
    return agree, confusion, rows


if __name__ == "__main__":
    primary, sensitivity = mistral_i2_source(), other_source()
    with open(SIX_MODELS_CSV, newline="", encoding="utf-8") as f:
        six_rows = list(csv.DictReader(f))
    # i5_six_models.csv: primary rows for every model, plus Mistral under the sensitivity source
    six = {(r["model"], r["mistral_i2_source"]): r for r in six_rows}
    six.update({(r["model"], None): r for r in six_rows if r["mistral_i2_source"] == primary})

    agree, confusion, rows = analyse(load_i5(primary), load_i5_judge(primary)["judgments"], six, primary)
    s_agree, s_conf, s_rows = analyse(load_i5(sensitivity), load_i5_judge(sensitivity)["judgments"],
                                      six, sensitivity)
    # extractor values must reproduce i5_six_models.csv
    max_dev = max(max(abs(r["mean_tier"] - r["_six_mean"]), abs(r["tier34_pct"] - r["_six_34"]))
                  for r in rows + s_rows if r["judge"] == J1)

    # ---- outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s_mistral = [r for r in s_rows if r["model"] == "mistral-large"]
    cols = ["model", "judge", "mistral_i2_source", "n", "mean_tier", "tier1_pct", "tier2_pct", "tier3_pct", "tier4_pct",
            "tier34_pct", "self_mean_tier", "self_tier34_pct", "third_34_gt_self", "third_mean_gt_self"]
    with open(OUT_DIR / "i5_second_judge.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows + s_mistral:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]) for c in cols})
        f.write("\n")
        w2 = csv.writer(f)
        w2.writerow(["agreement_measure", "value"])
        for k, v in agree.items():
            w2.writerow([k, f"{v:.6g}" if isinstance(v, float) else v])
        w2.writerow([])
        w2.writerow(["confusion (rows extractor, columns second judge)"] + [f"tier {t}" for t in TIERS])
        for x in TIERS:
            w2.writerow([f"tier {x}"] + [confusion[x][y] for y in TIERS])

    md = ["# I5 source tiers: extractor vs second judge", "",
          f"Generated by `scripts/stats_i5_second_judge.py`. Unit = one extracted source. "
          f"{J1}: tiers in data/raw/instrument_5.json. {J2}: data/processed/i5_second_judge.json "
          f"(same tier definitions; saw name, jurisdiction, quote). Primary: {LABELS[primary]} "
          f"(MISTRAL_I2_SOURCE={primary}; data_sources.load_i5 / load_i5_judge); sensitivity at the end. "
          f"Self-reported: I1 source types via SOURCE_TYPE_TO_TIER (results/stats/i5_six_models.csv).", "",
          "## Agreement", "", "| Measure | Value |", "|---|---|",
          f"| sources / paired / missing second judge | {agree['n_sources']} / {agree['n_paired']} / "
          f"{agree['n_missing_second']} |",
          f"| % agreement (exact tier) | {agree['pct_agree']:.1f} |",
          f"| % within one tier | {agree['pct_within_one']:.1f} |",
          f"| Cohen's kappa, unweighted | {agree['kappa_unweighted']:.3f} |",
          f"| Cohen's kappa, linear weights | {agree['kappa_linear']:.3f} |",
          f"| % agreement, Tier 3-4 vs 1-2 | {agree['pct_agree_tier34_binary']:.1f} |",
          f"| kappa, Tier 3-4 vs 1-2 | {agree['kappa_tier34_binary']:.3f} |",
          f"| % agreement, verifiable (n = {agree['n_verifiable_paired']}) | {agree['pct_agree_verifiable']:.1f} |",
          f"| kappa, verifiable | {agree['kappa_verifiable']:.3f} |", "",
          "## Confusion (rows extractor, columns second judge)", "",
          "| | " + " | ".join(f"Tier {t}" for t in TIERS) + " | total |", "|---|" + "---|" * 5]
    for x in TIERS:
        md.append(f"| Tier {x} | " + " | ".join(str(confusion[x][y]) for y in TIERS)
                  + f" | {sum(confusion[x].values())} |")
    md.append("| total | " + " | ".join(str(sum(confusion[x][y] for x in TIERS)) for y in TIERS)
              + f" | {agree['n_paired']} |")
    md += ["", "## Per model", "",
           "| Model | Judge | sources | mean tier | % Tier 1 | % Tier 2 | % Tier 3 | % Tier 4 | % Tier 3-4 | "
           "self % Tier 3-4 | third % 3-4 > self | self mean tier | third mean > self |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['judge']} | {r['n']} | {r['mean_tier']:.3f} | "
                  + " | ".join(f"{r[f'tier{t}_pct']:.1f}" for t in TIERS)
                  + f" | {r['tier34_pct']:.1f} | {r['self_tier34_pct']:.1f} | "
                  f"{'yes' if r['third_34_gt_self'] else 'no'} | {r['self_mean_tier']:.3f} | "
                  f"{'yes' if r['third_mean_gt_self'] else 'no'} |")
    md += ["", f"## Sensitivity: {LABELS[sensitivity]}", "",
           "Only Mistral's I2 sources differ; other models are identical.", "",
           f"| Agreement (all sources) | {LABELS[primary]} (primary) | {LABELS[sensitivity]} |", "|---|---|---|",
           f"| sources | {agree['n_sources']} | {s_agree['n_sources']} |"]
    for key, label in [("pct_agree", "% agreement"), ("kappa_unweighted", "kappa, unweighted"),
                       ("kappa_linear", "kappa, linear weights")]:
        md.append(f"| {label} | {agree[key]:.3f} | {s_agree[key]:.3f} |")
    md += ["", "| Mistral | Judge | sources | mean tier | % Tier 3-4 | third % 3-4 > self | third mean > self |",
           "|---|---|---|---|---|---|---|"]
    for r in [x for x in rows if x["model"] == "mistral-large"] + s_mistral:
        md.append(f"| {LABELS[r['mistral_i2_source']]} | {r['judge']} | {r['n']} | {r['mean_tier']:.3f} | "
                  f"{r['tier34_pct']:.1f} | {'yes' if r['third_34_gt_self'] else 'no'} | "
                  f"{'yes' if r['third_mean_gt_self'] else 'no'} |")
    md += ["", f"Extractor rows vs results/stats/i5_six_models.csv: max absolute deviation {max_dev:.2e}.", ""]
    (OUT_DIR / "i5_second_judge.md").write_text("\n".join(md), encoding="utf-8")
    print((OUT_DIR / "i5_second_judge.md").read_text(encoding="utf-8"))
