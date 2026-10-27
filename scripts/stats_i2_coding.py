"""
Model-level stats for the I2 criterion codes (data/processed/i2_codes.json,
criteria from i2_criteria_v1.json). No API calls; reads data/processed/ only.

Per answer: % of its criteria met. A criterion counts as met only if coded "met"
and, under the quote rule, its evidence quote was verified in the answer (exact,
or after normalizing quotes/dashes/whitespace/case: quote_check flag "" or
"typography"). Everything else (not_addressed, contradicted, unverified met) is
not met.
  primary   gpt-6.1-sol, quote rule
  (a)       claude-sonnet, quote rule
  (b)       gpt-6.1-sol, no quote rule (every "met" counts)

Per model: mean % met and % not met over the 9 baseline answers (3 scenarios x
3 runs), mean % met in CEO, and CEO minus baseline paired by scenario and run;
for all scenarios and per scenario (S1 legal certainty, S2 accountability,
S3 enforceability).
Across the six models (model_level.py): baseline % not met, one-sample t-test vs 0
with t-based 95% CI and model-only bootstrap CI; CEO minus baseline the same.
Also: the author's hand codes (manual_review.json, 6 answers) vs the primary
coder on the same answers.

Writes results/stats/i2_coding_model_level.csv and .md and the I2 rows of
results/stats/sensitivity_summary.csv.

Usage:
    python scripts/stats_i2_coding.py
"""

import csv
import json
from pathlib import Path

import numpy as np

from i2_criteria import CONDITIONS, MODELS, RUNS
from model_level import model_level_inference, update_summary

CODES_FILE  = Path("data/processed/i2_codes.json")
REVIEW_FILE = Path("data/processed/manual_review.json")
OUT_DIR     = Path("results/stats")

SCENARIOS = ["I2_S1", "I2_S2", "I2_S3"]
ALL       = "all"
SCOPES    = [ALL] + SCENARIOS
LABEL     = {ALL: "all scenarios", "I2_S1": "S1 legal certainty",
             "I2_S2": "S2 accountability", "I2_S3": "S3 enforceability"}
VARIANTS  = {"primary": ("gpt-6.1-sol", True),
             "(a) claude-sonnet coder": ("claude-sonnet", True),
             "(b) gpt no quote rule": ("gpt-6.1-sol", False)}
VERIFIED  = {"", "typography"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct_met(cell: dict, quote_rule: bool) -> float:
    codes = cell["parsed"]["codes"]
    flags = {q["n"]: q["flag"] for q in cell["quote_check"]}
    met = sum(1 for c in codes if c["code"] == "met"
              and (not quote_rule or flags.get(c["n"]) in VERIFIED))
    return 100 * met / len(codes)


def answer_pcts(coder_codes: dict, quote_rule: bool) -> dict:
    """{(model, condition, run, scenario): % met}"""
    return {(m, c, r, s): pct_met(coder_codes[m][c][r][s], quote_rule)
            for m in MODELS for c in CONDITIONS for r in RUNS for s in SCENARIOS}


def per_model(pcts: dict) -> dict:
    """{scope: {model: {baseline_met, baseline_not_met, ceo_met, ceo_minus_baseline}}}"""
    out = {}
    for scope in SCOPES:
        scen = SCENARIOS if scope == ALL else [scope]
        out[scope] = {}
        for m in MODELS:
            base = [pcts[(m, "baseline", r, s)] for s in scen for r in RUNS]
            ceo  = [pcts[(m, "ceo", r, s)] for s in scen for r in RUNS]
            diffs = [pcts[(m, "ceo", r, s)] - pcts[(m, "baseline", r, s)] for s in scen for r in RUNS]
            out[scope][m] = {"baseline_met": float(np.mean(base)),
                             "baseline_not_met": 100 - float(np.mean(base)),
                             "ceo_met": float(np.mean(ceo)),
                             "ceo_minus_baseline": float(np.mean(diffs)),
                             "n_pairs": len(diffs)}
    return out


def human_vs_coder(review: dict, coder_codes: dict) -> list[dict]:
    rows = []
    for a in review["coding"]:
        m, c, r, s = a["answer_id"].split(":")
        rows.append({"answer_id": a["answer_id"], "n_criteria": len(a["codes"]),
                     "author_met": 100 * sum(k["code"] == "met" for k in a["codes"]) / len(a["codes"]),
                     "gpt_met": pct_met(coder_codes[m][c][r][s], True),
                     "gpt_met_no_quote_rule": pct_met(coder_codes[m][c][r][s], False)})
    return rows


if __name__ == "__main__":
    codes = load(CODES_FILE)["codes"]

    per_model_rows, across_rows, pm_by_variant = [], [], {}
    for v, (coder, rule) in VARIANTS.items():
        pm = per_model(answer_pcts(codes[coder], rule))
        pm_by_variant[v] = pm
        for scope in SCOPES:
            for m in MODELS:
                per_model_rows.append({"analysis": "per_model", "variant": v, "coder": coder,
                                       "quote_rule": rule, "scenario": scope, "model": m,
                                       **pm[scope][m]})
            for measure, key in [("baseline % not met", "baseline_not_met"),
                                 ("CEO minus baseline % met", "ceo_minus_baseline")]:
                across_rows.append({"analysis": "across_models", "variant": v, "coder": coder,
                                    "quote_rule": rule, "scenario": scope, "measure": measure,
                                    "model": f"{len(MODELS)} models",
                                    **model_level_inference([pm[scope][m][key] for m in MODELS])})

    human = human_vs_coder(load(REVIEW_FILE), codes["gpt-6.1-sol"])

    # ---- outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["analysis", "variant", "coder", "quote_rule", "scenario", "measure", "model",
            "baseline_met", "baseline_not_met", "ceo_met", "ceo_minus_baseline", "n_pairs",
            "n_models", "mean", "sd", "t", "df", "p_t", "t_ci_low", "t_ci_high",
            "model_ci_low", "model_ci_high", "model_resamples_possible",
            "answer_id", "n_criteria", "author_met", "gpt_met", "gpt_met_no_quote_rule"]
    with open(OUT_DIR / "i2_coding_model_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_model_rows + across_rows + [{"analysis": "author_vs_gpt", **h} for h in human]:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    f1 = lambda x: f"{x:.1f}"
    s1 = lambda x: f"{x:+.1f}"
    iv = lambda lo, hi: f"[{s1(lo)}, {s1(hi)}]"
    md = ["# I2 criterion coding, model level", "",
          "Generated by `scripts/stats_i2_coding.py`. % met per answer; met requires a verified "
          "evidence quote unless noted. Baseline: 9 answers per model; CEO minus baseline paired "
          "by scenario and run. Across models: n = 6, one-sample t-test vs 0 (two-sided), t-based "
          "95% CI, model-only bootstrap 95% CI (10,000, seed 42).", ""]
    for v, pm in pm_by_variant.items():
        coder, rule = VARIANTS[v]
        md += [f"## Per model: {v} ({coder}, quote rule {'on' if rule else 'off'})", "",
               "| Model | " + " | ".join(f"{LABEL[s]}: base % met / % not met / CEO % met / CEO-base"
                                      for s in SCOPES) + " |",
               "|---|" + "---|" * len(SCOPES)]
        for m in MODELS:
            md.append(f"| {m} | " + " | ".join(
                f"{f1(pm[s][m]['baseline_met'])} / {f1(pm[s][m]['baseline_not_met'])} / "
                f"{f1(pm[s][m]['ceo_met'])} / {s1(pm[s][m]['ceo_minus_baseline'])}" for s in SCOPES) + " |")
        md.append("")
    md += ["## Across models", "",
           "| Variant | Scenario | Measure | mean | SD | t | df | p (two-sided) | 95% CI (t) | 95% CI (models) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for a in across_rows:
        md.append(f"| {a['variant']} | {LABEL[a['scenario']]} | {a['measure']} | {s1(a['mean'])} | "
                  f"{a['sd']:.2f} | {a['t']:.3f} | {a['df']} | {a['p_t']:.2e} | "
                  f"{iv(a['t_ci_low'], a['t_ci_high'])} | {iv(a['model_ci_low'], a['model_ci_high'])} |")
    md += ["", "## Author hand codes vs gpt-6.1-sol (same 6 answers, v1 criteria)", "",
           "| Answer | criteria | author % met | gpt % met (quote rule) | gpt % met (no quote rule) |",
           "|---|---|---|---|---|"]
    for h in human:
        md.append(f"| {h['answer_id']} | {h['n_criteria']} | {f1(h['author_met'])} | "
                  f"{f1(h['gpt_met'])} | {f1(h['gpt_met_no_quote_rule'])} |")
    md.append(f"| mean | {np.mean([h['n_criteria'] for h in human]):.1f} | "
              f"{f1(np.mean([h['author_met'] for h in human]))} | "
              f"{f1(np.mean([h['gpt_met'] for h in human]))} | "
              f"{f1(np.mean([h['gpt_met_no_quote_rule'] for h in human]))} |")
    (OUT_DIR / "i2_coding_model_level.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    short = {"baseline % not met": "% not met", "CEO minus baseline % met": "CEO-baseline % met"}
    update_summary([{**a, "test": f"I2 {short[a['measure']]} {LABEL[a['scenario']]}"
                     + ("" if a["variant"] == "primary" else f" {a['variant']}")} for a in across_rows])

    print((OUT_DIR / "i2_coding_model_level.md").read_text(encoding="utf-8"))
