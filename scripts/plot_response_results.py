"""
plot_response_results.py
Generates two sets of visualizations for Instrument 1 responses:
  1. Word frequency heatmaps — per question, across models and conditions
  2. Semantic similarity matrices — cross-model and baseline vs CEO

Outputs:
  results/qualitative/wordfreq_legal_certainty.html
  results/qualitative/wordfreq_accountability.html
  results/qualitative/wordfreq_enforceability.html
  results/qualitative/wordfreq_legal_certainty.png
  results/qualitative/wordfreq_accountability.png
  results/qualitative/wordfreq_enforceability.png
  results/quantitative/similarity_cross_model.html
  results/quantitative/similarity_baseline_vs_ceo.html
  results/quantitative/similarity_cross_model.png
  results/quantitative/similarity_baseline_vs_ceo.png
"""

import json
import re
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# =============================================================================
# Config
# =============================================================================

RESPONSES_FILE = Path("data/raw/responses.json")
QUALITATIVE_OUT = Path("results/qualitative")
QUANTITATIVE_OUT = Path("results/quantitative")

INSTRUMENT = "instrument_1"

QUESTIONS = {
    "I1_Q1": "Legal Certainty",
    "I1_Q2": "Accountability",
    "I1_Q3": "Enforceability",
}

CONDITIONS = ["baseline", "ceo"]
CONDITION_LABELS = {"baseline": "Baseline", "ceo": "CEO Role"}

# Display-friendly model labels
MODEL_LABELS = {
    "gpt-4o":         "GPT-4o",
    "claude-sonnet":  "Claude Sonnet",
    "deepseek-v3":    "DeepSeek-V3",
    "mistral-large":  "Mistral Large",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
}

TOP_N_WORDS = 20  # words per model per condition in frequency heatmap
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# Helpers
# =============================================================================

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# Domain-specific words that are uninformative across all responses
DOMAIN_STOP = {
    "ai", "systems", "system", "governance", "framework", "must",
    "ensure", "also", "including", "well", "may", "one", "used",
    "use", "within", "provide", "based", "without", "new", "key",
    "make", "made", "such", "across", "ways", "involves", "includes",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [
        t for t in tokens
        if t not in STOP_WORDS
        and t not in DOMAIN_STOP
        and len(t) > 3
    ]


def get_responses_for(data: dict, model: str, condition: str,
                      question_id: str) -> list[str]:
    """Return all run responses for a given model/condition/question."""
    responses = []
    runs = data.get(model, {}).get(condition, {})
    for run_num in sorted(runs.keys(), key=int):
        text = runs[run_num].get(INSTRUMENT, {}).get(question_id)
        if text:
            responses.append(text)
    return responses


def avg_embedding(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Mean embedding across a list of texts."""
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.mean(axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# =============================================================================
# 1. Word frequency heatmaps
# =============================================================================

def build_word_freq_heatmap(data: dict, question_id: str, question_label: str,
                             models: list[str]) -> go.Figure:
    """
    Heatmap: rows = top words, columns = model × condition pairs.
    Cell value = normalized frequency (word count / total tokens).
    """
    col_labels = []
    col_freqs = {}  # col_label -> Counter of normalized freqs

    for model in models:
        for condition in CONDITIONS:
            texts = get_responses_for(data, model, condition, question_id)
            if not texts:
                continue
            tokens = tokenize(" ".join(texts))
            total = len(tokens) or 1
            counter = Counter(tokens)
            normalized = {w: c / total for w, c in counter.items()}
            label = f"{MODEL_LABELS.get(model, model)}\n({CONDITION_LABELS[condition]})"
            col_labels.append(label)
            col_freqs[label] = normalized

    # Collect top words globally across all columns
    global_counter = Counter()
    for freq in col_freqs.values():
        global_counter.update({w: v for w, v in freq.items()})
    top_words = [w for w, _ in global_counter.most_common(TOP_N_WORDS)]

    # Build z matrix: rows = words, cols = model×condition
    z = []
    for word in top_words:
        row = [col_freqs.get(col, {}).get(word, 0.0) for col in col_labels]
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=col_labels,
        y=top_words,
        colorscale=[
            [0.0,  "#0d1117"],
            [0.3,  "#1a3a5c"],
            [0.6,  "#2176ae"],
            [1.0,  "#7ecef0"],
        ],
        showscale=True,
        colorbar=dict(
            title="Normalized<br>Frequency",
            tickfont=dict(color="#c9d1d9"),
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Freq: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"Word Frequency — <b>{question_label}</b>",
            font=dict(family="Georgia, serif", size=20, color="#e6edf3"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="'Courier New', monospace", color="#c9d1d9"),
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=10, color="#8b949e"),
            gridcolor="#21262d",
            linecolor="#30363d",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#c9d1d9"),
            gridcolor="#21262d",
            linecolor="#30363d",
        ),
        margin=dict(l=160, r=60, t=80, b=140),
        height=620,
        width=1100,
    )

    return fig


# =============================================================================
# 2. Semantic similarity matrices
# =============================================================================

def build_cross_model_similarity(data: dict, models: list[str],
                                  embed_model: SentenceTransformer) -> go.Figure:
    """
    One similarity matrix per question, averaged across questions.
    Rows/cols = models. Each cell = cosine similarity of mean embeddings
    (averaged across all runs and both conditions).
    """
    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    n = len(models)

    # Accumulate similarities across questions
    sim_sum = np.zeros((n, n))
    q_count = 0

    for question_id in QUESTIONS:
        embeddings = []
        for model in models:
            texts = []
            for condition in CONDITIONS:
                texts.extend(get_responses_for(data, model, condition, question_id))
            if texts:
                embeddings.append(avg_embedding(texts, embed_model))
            else:
                embeddings.append(np.zeros(384))

        for i in range(n):
            for j in range(n):
                sim_sum[i][j] += cosine_similarity(embeddings[i], embeddings[j])
        q_count += 1

    sim_matrix = sim_sum / max(q_count, 1)

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix.tolist(),
        x=model_labels,
        y=model_labels,
        colorscale=[
            [0.0,  "#0d1117"],
            [0.4,  "#1f3d5c"],
            [0.7,  "#1a6b3c"],
            [1.0,  "#3fb68a"],
        ],
        zmin=0.5,
        zmax=1.0,
        showscale=True,
        colorbar=dict(
            title="Cosine<br>Similarity",
            tickfont=dict(color="#c9d1d9"),
        ),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>",
        text=[[f"{v:.3f}" for v in row] for row in sim_matrix.tolist()],
        texttemplate="%{text}",
        textfont=dict(size=12, color="#e6edf3"),
    ))

    fig.update_layout(
        title=dict(
            text="Cross-Model Semantic Similarity<br><sup>Averaged across Legal Certainty, Accountability, Enforceability</sup>",
            font=dict(family="Georgia, serif", size=18, color="#e6edf3"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="'Courier New', monospace", color="#c9d1d9"),
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(size=11, color="#c9d1d9"),
            side="bottom",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#c9d1d9"),
        ),
        margin=dict(l=160, r=60, t=100, b=100),
        height=540,
        width=700,
    )

    return fig


def build_baseline_vs_ceo_similarity(data: dict, models: list[str],
                                      embed_model: SentenceTransformer) -> go.Figure:
    """
    Per-model, per-question: cosine similarity between baseline mean embedding
    and CEO mean embedding. Rows = questions, cols = models.
    Low similarity = large framing shift from role assignment.
    """
    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    question_labels = list(QUESTIONS.values())
    question_ids = list(QUESTIONS.keys())

    z = []
    for q_id in question_ids:
        row = []
        for model in models:
            baseline_texts = get_responses_for(data, model, "baseline", q_id)
            ceo_texts      = get_responses_for(data, model, "ceo", q_id)
            if baseline_texts and ceo_texts:
                b_emb = avg_embedding(baseline_texts, embed_model)
                c_emb = avg_embedding(ceo_texts, embed_model)
                sim = cosine_similarity(b_emb, c_emb)
            else:
                sim = None
            row.append(sim)
        z.append(row)

    # Annotation text
    text = [[f"{v:.3f}" if v is not None else "N/A" for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=model_labels,
        y=question_labels,
        colorscale=[
            [0.0,  "#4a0d0d"],
            [0.3,  "#7a2020"],
            [0.6,  "#c47c2b"],
            [1.0,  "#e8d44d"],
        ],
        zmin=0.5,
        zmax=1.0,
        showscale=True,
        colorbar=dict(
            title="Cosine<br>Similarity",
            tickfont=dict(color="#c9d1d9"),
        ),
        hovertemplate="<b>%{y}</b> — <b>%{x}</b><br>Baseline↔CEO Similarity: %{z:.3f}<extra></extra>",
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#e6edf3"),
    ))

    fig.update_layout(
        title=dict(
            text="Baseline vs CEO Framing Shift<br><sup>Lower similarity = greater role-induced framing change</sup>",
            font=dict(family="Georgia, serif", size=18, color="#e6edf3"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="'Courier New', monospace", color="#c9d1d9"),
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(size=11, color="#c9d1d9"),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#c9d1d9"),
            autorange="reversed",
        ),
        margin=dict(l=180, r=60, t=100, b=100),
        height=400,
        width=820,
    )

    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    QUALITATIVE_OUT.mkdir(parents=True, exist_ok=True)
    QUANTITATIVE_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading responses...")
    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = list(data.keys())
    print(f"Models found: {models}\n")

    # --- Word frequency heatmaps ---
    print("Building word frequency heatmaps...")
    for q_id, q_label in QUESTIONS.items():
        slug = q_label.lower().replace(" ", "_")
        fig = build_word_freq_heatmap(data, q_id, q_label, models)

        html_path = QUALITATIVE_OUT / f"wordfreq_{slug}.html"
        png_path  = QUALITATIVE_OUT / f"wordfreq_{slug}.png"

        fig.write_html(str(html_path))
        fig.write_image(str(png_path), scale=2)
        print(f"  Saved: {html_path}")
        print(f"  Saved: {png_path}")

    # --- Semantic similarity ---
    print("\nLoading embedding model (first run may download weights)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Building cross-model similarity matrix...")
    fig_cross = build_cross_model_similarity(data, models, embed_model)
    fig_cross.write_html(str(QUANTITATIVE_OUT / "similarity_cross_model.html"))
    fig_cross.write_image(str(QUANTITATIVE_OUT / "similarity_cross_model.png"), scale=2)
    print(f"  Saved: {QUANTITATIVE_OUT}/similarity_cross_model.[html|png]")

    print("Building baseline vs CEO framing shift matrix...")
    fig_shift = build_baseline_vs_ceo_similarity(data, models, embed_model)
    fig_shift.write_html(str(QUANTITATIVE_OUT / "similarity_baseline_vs_ceo.html"))
    fig_shift.write_image(str(QUANTITATIVE_OUT / "similarity_baseline_vs_ceo.png"), scale=2)
    print(f"  Saved: {QUANTITATIVE_OUT}/similarity_baseline_vs_ceo.[html|png]")

    print("\nAll plots complete.")