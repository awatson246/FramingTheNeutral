"""
Recompute the I1-based analyses with get_answer_text() and compare them to
earlier published numbers. Writes only to results_recheck/.

Compared versions (read from git, so results/ is never touched):
  april  = f07935c  (H1 computed on raw {raw, parsed} dicts)
  may    = b65a3a8  (H1 computed via the old _extract_text helper)
  new    = this run (everything via get_answer_text)

No LLM calls: the I2 Sankeys / extract_structured are skipped, and the
embedding model is loaded from the local Hugging Face cache (offline).

Usage:
    python scripts/recheck_i1.py
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # embedding model from local cache only

# The installed torchvision (0.27.1) doesn't match torch (2.7.1+cpu) and breaks
# `import transformers`. Text embeddings don't use torchvision, so hide it from
# this process only (transformers then treats it as not installed).
sys.modules["torchvision"] = None

import json
import re
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

import plot_response_results as prp
from answer_text import get_answer_text
from sentence_transformers import SentenceTransformer

OUT_DIR  = Path("results_recheck")
VERSIONS = {"april": "f07935c", "may": "b65a3a8"}

# Words named in the paper's word-frequency claims, and the questions they cover
CLAIMS = {
    "claude-sonnet":                 (["legal", "rule", "accountability"], list(prp.I1_QUESTIONS)),
    "gemini-3.1-flash-lite-preview": (["certainty", "public"],             ["I1_Q1"]),
}


# =============================================================================
# Old-value loaders (git objects, not the working tree)
# =============================================================================

def git_file(rev: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def plotly_trace(html: str) -> dict:
    """Pull the first trace out of a saved plotly HTML file."""
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\])\s*,\s*\{"', html, re.S)
    return json.loads(m.group(1))[0]


def old_values(rev: str) -> dict:
    h1  = json.loads(git_file(rev, "results/hypothesis_tests/results.json"))["h1"]
    sim = plotly_trace(git_file(rev, "results/instrument_1/similarity_cross_model.html"))
    top = {}
    for q_id, q_label in prp.I1_QUESTIONS.items():
        slug = q_label.lower().replace(" ", "_")
        top[q_id] = plotly_trace(git_file(rev, f"results/instrument_1/wordfreq_{slug}.html"))["y"]
    return {"h1": h1, "sim_labels": sim["x"], "sim": sim["z"], "top_words": top}


# =============================================================================
# H1 on plain text (same logic as run_hypothesis_tests, text via get_answer_text)
# =============================================================================

def h1_clean(i1_data: dict, i2_data: dict, models: list[str],
             embed_model: SentenceTransformer) -> dict:
    dim_pairs = [("I1_Q1", "I2_S1"), ("I1_Q2", "I2_S2"), ("I1_Q3", "I2_S3")]
    within_sims, cross_sims = [], []
    for model in models:
        for i1_qid, i2_sid in dim_pairs:
            runs  = i1_data.get(model, {}).get("baseline", {})
            cells = [runs.get(str(r), {}).get(i1_qid) for r in [1, 2, 3]]
            run_texts = [t for t in (get_answer_text(c) for c in cells if c is not None) if t]
            if len(run_texts) < 2:
                continue
            embs = embed_model.encode(run_texts, show_progress_bar=False)
            w_sim = float(np.mean([prp.cosine_sim(embs[a], embs[b])
                                   for a, b in combinations(range(len(embs)), 2)]))

            i1_text = prp.get_run1(i1_data, model, "baseline", i1_qid)
            i2_text = prp.get_run1(i2_data, model, "baseline", i2_sid)
            if not i1_text or not i2_text:
                continue
            e1, e2 = embed_model.encode([i1_text, i2_text], show_progress_bar=False)
            within_sims.append(w_sim)
            cross_sims.append(prp.cosine_sim(e1, e2))

    t_stat, p_val = stats.ttest_rel(within_sims, cross_sims)
    w_mean, c_mean = float(np.mean(within_sims)), float(np.mean(cross_sims))
    return {"within_mean": w_mean, "cross_mean": c_mean, "t": float(t_stat),
            "p": float(p_val), "n": len(within_sims),
            "supported": str(w_mean > c_mean and p_val < 0.05)}


# =============================================================================
# Leak check for the old _extract_text helper
# =============================================================================

def old_extract_text(val) -> str:
    """Verbatim copy of the removed plot_response_results._extract_text."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        parsed = val.get("parsed") or {}
        return parsed.get("response") or val.get("raw") or ""
    return ""


def leak_check(i1_data: dict) -> dict:
    total = parsed_cells = differs = 0
    for conds in i1_data.values():
        for runs in conds.values():
            for qs in runs.values():
                for cell in qs.values():
                    total += 1
                    if isinstance(cell, dict) and (cell.get("parsed") or {}).get("response"):
                        parsed_cells += 1
                    if old_extract_text(cell) != get_answer_text(cell):
                        differs += 1
    return {"cells": total, "parsed": parsed_cells, "unparsed": total - parsed_cells,
            "old_helper_output_differs_from_clean": differs}


def word_counts(data: dict, model: str, cond: str, q_id: str) -> Counter:
    return Counter(prp.tokenize(" ".join(prp.get_responses(data, model, cond, q_id))))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    i1_data = prp._try_load("instrument_1")
    i2_data = prp._try_load("instrument_2")
    models  = list(i1_data.keys())
    labels  = [prp.MODEL_LABELS.get(m, m) for m in models]
    embed_model = SentenceTransformer(prp.EMBEDDING_MODEL)
    old = {name: old_values(rev) for name, rev in VERSIONS.items()}

    # ---- H1 ----
    print("H1 (clean text)...")
    h1_new = h1_clean(i1_data, i2_data, models, embed_model)
    print("H1 (merged plot_response_results code, for comparison)...")
    h1_merged = prp.run_hypothesis_tests(i1_data, i2_data, None, None, None,
                                         models, embed_model)["h1"]

    # ---- Figure 1 ----
    out = OUT_DIR / "instrument_1"
    fig_sim = prp.i1_cross_model_similarity(i1_data, models, embed_model)
    prp.save_fig(fig_sim, out / "similarity_cross_model")
    new_sim = [list(r) for r in fig_sim.data[0].z]
    for name, o in old.items():
        assert o["sim_labels"] == labels, f"{name} similarity labels differ: {o['sim_labels']}"

    sim_pairs = [
        {"pair": f"{labels[i]} / {labels[j]}",
         **{name: o["sim"][i][j] for name, o in old.items()},
         "new": new_sim[i][j]}
        for i, j in combinations(range(len(models)), 2)
    ]

    # ---- Word frequency ----
    wordfreq = {}
    for q_id, q_label in prp.I1_QUESTIONS.items():
        slug = q_label.lower().replace(" ", "_")
        fig = prp.i1_wordfreq_heatmap(i1_data, q_id, q_label, models)
        prp.save_fig(fig, out / f"wordfreq_{slug}")
        wordfreq[q_id] = {
            "heatmap_top_words": {**{n: o["top_words"][q_id] for n, o in old.items()},
                                  "new": list(fig.data[0].y)},
            "top_words_per_model": {
                m: {c: word_counts(i1_data, m, c, q_id).most_common(prp.TOP_N_WORDS)
                    for c in prp.CONDITIONS}
                for m in models
            },
        }

    claims = {}
    for m, (words, q_ids) in CLAIMS.items():
        claims[m] = {}
        for q_id in q_ids:
            claims[m][q_id] = {
                o: {w: sum(word_counts(i1_data, o, c, q_id)[w] for c in prp.CONDITIONS)
                    for w in words + ["_total_tokens"]}
                for o in models
            }
            for o in models:
                claims[m][q_id][o]["_total_tokens"] = sum(
                    sum(word_counts(i1_data, o, c, q_id).values()) for c in prp.CONDITIONS)

    comparison = {
        "h1": {"april": old["april"]["h1"], "may": old["may"]["h1"],
               "merged_code_now": h1_merged, "new": h1_new},
        "similarity_pairs": sim_pairs,
        "similarity_highest": {k: max(sim_pairs, key=lambda p: p[k])["pair"]
                               for k in ["april", "may", "new"]},
        "similarity_lowest":  {k: min(sim_pairs, key=lambda p: p[k])["pair"]
                               for k in ["april", "may", "new"]},
        "wordfreq": wordfreq,
        "claim_counts": claims,
        "extract_text_leak_check": leak_check(i1_data),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        # default=str: run_hypothesis_tests returns numpy bools/floats
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {OUT_DIR / 'comparison.json'}")
