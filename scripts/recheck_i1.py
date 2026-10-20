"""
Recompute the I1-based analyses with get_answer_text() and compare them to
the numbers stored in results/. Writes only to results_recheck/.

No LLM calls: the I2 Sankeys / extract_structured are skipped, and the
embedding model is loaded from the local Hugging Face cache (offline).

Usage:
    python scripts/recheck_i1.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # embedding model from local cache only

import json
import re
from collections import Counter
from pathlib import Path

import plot_response_results as prp
from sentence_transformers import SentenceTransformer

OLD_DIR = Path("results")
OUT_DIR = Path("results_recheck")

# Words named in the paper's word-frequency claims
CLAIM_WORDS = {
    "claude-sonnet":                 ["legal", "rule", "accountability"],
    "gemini-3.1-flash-lite-preview": ["certainty", "public"],
}
# Paper claim scope: Claude across I1; Gemini in legal certainty (I1_Q1)
CLAIM_QUESTIONS = {
    "claude-sonnet":                 list(prp.I1_QUESTIONS),
    "gemini-3.1-flash-lite-preview": ["I1_Q1"],
}


def old_plotly_trace(html_path: Path) -> dict:
    """Pull the first trace's x/y/z out of a saved plotly HTML file."""
    html = html_path.read_text(encoding="utf-8")
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\])\s*,\s*\{', html, re.S)
    return json.loads(m.group(1))[0]


def word_counts(data: dict, model: str, cond: str, q_id: str) -> Counter:
    return Counter(prp.tokenize(" ".join(prp.get_responses(data, model, cond, q_id))))


if __name__ == "__main__":
    i1_data = prp._try_load("instrument_1")
    i2_data = prp._try_load("instrument_2")
    models  = list(i1_data.keys())
    labels  = [prp.MODEL_LABELS.get(m, m) for m in models]
    embed_model = SentenceTransformer(prp.EMBEDDING_MODEL)

    # ---- H1 (I3/ELP/I5 inputs omitted: only H1 is recomputed) ----
    stats = prp.run_hypothesis_tests(i1_data, i2_data, None, None, None, models, embed_model)
    prp.save_hypothesis_results(stats, OUT_DIR / "hypothesis_tests")
    old_h1 = json.loads((OLD_DIR / "hypothesis_tests" / "results.json").read_text(encoding="utf-8"))["h1"]
    new_h1 = stats["h1"]

    # ---- Figure 1: cross-model similarity ----
    out = OUT_DIR / "instrument_1"
    fig_sim = prp.i1_cross_model_similarity(i1_data, models, embed_model)
    prp.save_fig(fig_sim, out / "similarity_cross_model")
    new_mat = [list(r) for r in fig_sim.data[0].z]
    old_mat = old_plotly_trace(OLD_DIR / "instrument_1" / "similarity_cross_model.html")["z"]

    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pairs.append({"pair": f"{labels[i]} / {labels[j]}",
                          "old": old_mat[i][j], "new": new_mat[i][j]})

    # Baseline-vs-CEO figure carries the H1 annotation in its title
    fig_bvc = prp.i1_baseline_vs_ceo(i1_data, models, embed_model)
    h1 = new_h1
    fig_bvc.update_layout(title_text=(fig_bvc.layout.title.text or "") + (
        f"<br><sup style='font-size:11px'>H1 paired t-test (n={h1['n']}): t={h1['t']:.3f}, "
        f"p={h1['p']:.4f} | within-I1={h1['within_mean']:.3f}, I1→I2={h1['cross_mean']:.3f} | "
        f"supported={h1['supported']}</sup>"))
    prp.save_fig(fig_bvc, out / "similarity_baseline_vs_ceo")

    # ---- Word frequency ----
    wordfreq = {}
    for q_id, q_label in prp.I1_QUESTIONS.items():
        slug = q_label.lower().replace(" ", "_")
        fig = prp.i1_wordfreq_heatmap(i1_data, q_id, q_label, models)
        prp.save_fig(fig, out / f"wordfreq_{slug}")
        old_tr = old_plotly_trace(OLD_DIR / "instrument_1" / f"wordfreq_{slug}.html")
        wordfreq[q_id] = {
            "heatmap_top_words_old": old_tr["y"],
            "heatmap_top_words_new": list(fig.data[0].y),
            "heatmap_z_identical":   old_tr["z"] == [list(r) for r in fig.data[0].z],
            "top_words_per_model": {
                m: {c: word_counts(i1_data, m, c, q_id).most_common(prp.TOP_N_WORDS)
                    for c in prp.CONDITIONS}
                for m in models
            },
        }

    claims = {}
    for m, words in CLAIM_WORDS.items():
        claims[m] = {}
        for q_id in CLAIM_QUESTIONS[m]:
            claims[m][q_id] = {}
            for c in prp.CONDITIONS:
                ctr = word_counts(i1_data, m, c, q_id)
                claims[m][q_id][c] = {"total_tokens": sum(ctr.values()),
                                      **{w: ctr[w] for w in words}}
            # Other models' counts for the same words, for comparison
            claims[m][q_id]["others"] = {
                o: {w: sum(word_counts(i1_data, o, c, q_id)[w] for c in prp.CONDITIONS)
                    for w in words}
                for o in models if o != m
            }

    comparison = {
        "h1": {k: {"old": old_h1.get(k), "new": new_h1.get(k)}
               for k in ["within_mean", "cross_mean", "t", "p", "n", "supported"]},
        "similarity_pairs": pairs,
        "similarity_highest": {"old": max(pairs, key=lambda p: p["old"]),
                               "new": max(pairs, key=lambda p: p["new"])},
        "similarity_lowest":  {"old": min(pairs, key=lambda p: p["old"]),
                               "new": min(pairs, key=lambda p: p["new"])},
        "wordfreq": wordfreq,
        "claim_counts": claims,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_DIR / 'comparison.json'}")
