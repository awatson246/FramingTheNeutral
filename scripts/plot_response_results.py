"""
plot_response_results.py

Generates visualizations for Instrument 1 and Instrument 2 responses.
Control which instruments are plotted via ACTIVE_INSTRUMENTS.

Output structure:
  results/
    instrument_1/
      wordfreq_legal_certainty.[html|png]
      wordfreq_accountability.[html|png]
      wordfreq_enforceability.[html|png]
      similarity_cross_model.[html|png]
      similarity_baseline_vs_ceo.[html|png]
    instrument_2/
      s1_wordfreq_cross_model.[html|png]
      s1_wordfreq_baseline_vs_ceo.[html|png]
      s2_responsibility_radar.[html|png]
      s2_accountability_sankey.[html|png]
      s3_enforcement_sankey.[html|png]
"""

import json
import os
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from time import sleep

import anthropic
import nltk
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config
# =============================================================================

RAW_DIR     = Path("data/raw")
RESULTS_DIR = Path("results")

ACTIVE_INSTRUMENTS = ["instrument_1", "instrument_2"]

CONDITIONS       = ["baseline", "ceo"]
CONDITION_LABELS = {"baseline": "Baseline", "ceo": "CEO Role"}

MODEL_LABELS = {
    "gpt-4o":                        "GPT-4o",
    "claude-sonnet":                 "Claude Sonnet",
    "deepseek-v3":                   "DeepSeek-V3",
    "mistral-large":                 "Mistral Large",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
}

I1_QUESTIONS = {
    "I1_Q1": "Legal Certainty",
    "I1_Q2": "Accountability",
    "I1_Q3": "Enforceability",
}

TOP_N_WORDS     = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Responsibility actors for radar chart axes (S2)
RESPONSIBILITY_ACTORS = [
    "city / government", "vendor / developer",
    "regulator", "court / judiciary",
    "official", "applicant / public",
]

# Actor keyword clusters — each axis is matched by any of these terms
ACTOR_KEYWORDS = {
    "city / government":    ["city", "government", "municipal", "authority", "state"],
    "vendor / developer":   ["vendor", "developer", "company", "provider", "contractor"],
    "regulator":            ["regulator", "regulatory", "oversight", "agency", "watchdog"],
    "court / judiciary":    ["court", "judicial", "judge", "tribunal", "legal action"],
    "official":             ["official", "administrator", "bureaucrat", "civil servant"],
    "applicant / public":   ["applicant", "resident", "citizen", "public", "individual"],
}

# =============================================================================
# Shared styling
# =============================================================================

BG       = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
MONO     = "'Courier New', monospace"
SERIF    = "Georgia, serif"

COLORSCALE_BLUE = [
    [0.0, "#0d1117"], [0.3, "#1a3a5c"],
    [0.6, "#2176ae"], [1.0, "#7ecef0"],
]
COLORSCALE_GREEN = [
    [0.0, "#0d1117"], [0.4, "#1f3d5c"],
    [0.7, "#1a6b3c"], [1.0, "#3fb68a"],
]
COLORSCALE_AMBER = [
    [0.0, "#4a0d0d"], [0.3, "#7a2020"],
    [0.6, "#c47c2b"], [1.0, "#e8d44d"],
]

# One color per model, consistent across all plots
MODEL_COLORS = {
    "gpt-4o":                        "#2176ae",
    "claude-sonnet":                 "#3fb68a",
    "deepseek-v3":                   "#e05c5c",
    "mistral-large":                 "#e8a838",
    "gemini-3.1-flash-lite-preview": "#c084fc",
}

# Condition line styles for radar
CONDITION_DASH = {"baseline": "solid", "ceo": "dash"}


def base_layout(title: str, height: int = 560, width: int = 1100) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(family=SERIF, size=18, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        margin=dict(l=160, r=60, t=90, b=140),
        height=height,
        width=width,
    )


def hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    """Convert #rrggbb hex to rgba(r,g,b,alpha) for Sankey link colors."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def save_fig(fig: go.Figure, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path_stem.with_suffix(".html")))
    fig.write_image(str(path_stem.with_suffix(".png")), scale=2)
    print(f"  Saved: {path_stem}.html / .png")


# =============================================================================
# NLP helpers
# =============================================================================

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
STOP_WORDS = set(stopwords.words("english"))

DOMAIN_STOP = {
    "ai", "systems", "system", "governance", "framework", "must",
    "ensure", "also", "including", "well", "may", "one", "used",
    "use", "within", "provide", "based", "without", "new", "key",
    "make", "made", "such", "across", "ways", "involves", "includes",
    "would", "could", "should", "need", "like", "even", "part",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [
        t for t in text.split()
        if t not in STOP_WORDS and t not in DOMAIN_STOP and len(t) > 3
    ]


def get_responses(data: dict, model: str, condition: str,
                  question_id: str) -> list[str]:
    runs = data.get(model, {}).get(condition, {})
    return [
        runs[r][question_id]
        for r in sorted(runs.keys(), key=int)
        if runs[r].get(question_id)
    ]


def get_run1(data: dict, model: str, condition: str, question_id: str) -> str:
    return data.get(model, {}).get(condition, {}).get("1", {}).get(question_id, "")


def avg_embedding(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    embs = model.encode(texts, show_progress_bar=False)
    return embs.mean(axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# =============================================================================
# LLM extraction helper
# Calls Claude to extract structured lists from a response.
# Returns a dict with keys depending on the extraction_type.
# Results are cached in memory to avoid duplicate API calls.
# =============================================================================

_extraction_cache: dict[str, dict] = {}

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def extract_structured(text: str, extraction_type: str) -> dict:
    """
    Use Claude to extract structured information from a response text.

    extraction_type options:
      "s2_accountability" -> returns:
          {
            "responsible_parties": ["city government", "AI vendor", ...],
            "mechanisms": ["independent audit", "judicial review", ...]
          }
      "s3_enforcement" -> returns:
          {
            "challenges": ["jurisdictional fragmentation", "audit opacity", ...],
            "solutions": ["international treaty", "harmonized standards", ...]
          }
    """
    cache_key = f"{extraction_type}::{hash(text)}"
    if cache_key in _extraction_cache:
        return _extraction_cache[cache_key]

    if extraction_type == "s2_accountability":
        prompt = (
            "You are a research assistant extracting structured information from AI governance responses.\n\n"
            "From the following response about AI accountability in a housing denial scenario, extract:\n"
            "1. responsible_parties: a list of 2-5 short labels for the actors identified as responsible "
            "(e.g. 'city government', 'AI vendor', 'regulatory body')\n"
            "2. mechanisms: a list of 2-5 short labels for the accountability mechanisms proposed "
            "(e.g. 'independent audit', 'judicial review', 'ombudsman')\n\n"
            "Respond ONLY with a JSON object with keys 'responsible_parties' and 'mechanisms'. "
            "No preamble, no markdown, no explanation.\n\n"
            f"Response:\n{text[:2000]}"
        )
        fallback = {"responsible_parties": [], "mechanisms": []}

    elif extraction_type == "s3_enforcement":
        prompt = (
            "You are a research assistant extracting structured information from AI governance responses.\n\n"
            "From the following response about AI regulation enforcement challenges, extract:\n"
            "1. challenges: a list of 2-5 short labels for the enforcement challenges identified "
            "(e.g. 'jurisdictional fragmentation', 'audit opacity', 'regulatory arbitrage')\n"
            "2. solutions: a list of 2-5 short labels for the proposed solutions or remedies "
            "(e.g. 'international treaty', 'harmonized standards', 'mutual recognition agreements')\n\n"
            "Respond ONLY with a JSON object with keys 'challenges' and 'solutions'. "
            "No preamble, no markdown, no explanation.\n\n"
            f"Response:\n{text[:2000]}"
        )
        fallback = {"challenges": [], "solutions": []}

    else:
        return {}

    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
        result = json.loads(raw)
        _extraction_cache[cache_key] = result
        sleep(0.3)  # gentle rate limiting
        return result
    except Exception as e:
        print(f"    [extract_structured] failed for {extraction_type}: {e}")
        return fallback


# =============================================================================
# INSTRUMENT 1 plots  (unchanged)
# =============================================================================

def i1_wordfreq_heatmap(data: dict, q_id: str, q_label: str,
                         models: list[str]) -> go.Figure:
    col_labels, col_freqs = [], {}
    for model in models:
        for cond in CONDITIONS:
            texts = get_responses(data, model, cond, q_id)
            if not texts:
                continue
            tokens = tokenize(" ".join(texts))
            total  = len(tokens) or 1
            label  = f"{MODEL_LABELS.get(model, model)}<br>({CONDITION_LABELS[cond]})"
            col_labels.append(label)
            col_freqs[label] = {w: c / total for w, c in Counter(tokens).items()}

    global_ctr = Counter()
    for freq in col_freqs.values():
        global_ctr.update(freq)
    top_words = [w for w, _ in global_ctr.most_common(TOP_N_WORDS)]

    z = [[col_freqs.get(col, {}).get(w, 0.0) for col in col_labels]
         for w in top_words]

    fig = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=top_words,
        colorscale=COLORSCALE_BLUE,
        colorbar=dict(title="Norm. Freq", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Freq: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(f"Word Frequency — <b>{q_label}</b>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI, size=11))
    fig.update_xaxes(tickangle=-35, tickfont=dict(color=TEXT_SEC, size=10))
    return fig


def i1_cross_model_similarity(data: dict, models: list[str],
                               embed_model: SentenceTransformer) -> go.Figure:
    labels  = [MODEL_LABELS.get(m, m) for m in models]
    n       = len(models)
    sim_sum = np.zeros((n, n))
    count   = 0

    for q_id in I1_QUESTIONS:
        embs = []
        for model in models:
            texts = []
            for cond in CONDITIONS:
                texts.extend(get_responses(data, model, cond, q_id))
            embs.append(avg_embedding(texts, embed_model) if texts else np.zeros(384))
        for i in range(n):
            for j in range(n):
                sim_sum[i][j] += cosine_sim(embs[i], embs[j])
        count += 1

    mat = (sim_sum / max(count, 1)).tolist()
    fig = go.Figure(go.Heatmap(
        z=mat, x=labels, y=labels,
        colorscale=COLORSCALE_GREEN, zmin=0.5, zmax=1.0,
        text=[[f"{v:.3f}" for v in row] for row in mat],
        texttemplate="%{text}", textfont=dict(size=12, color=TEXT_PRI),
        colorbar=dict(title="Cosine Sim", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Sim: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        "Cross-Model Semantic Similarity<br>"
        "<sup>Averaged across Legal Certainty, Accountability, Enforceability</sup>",
        height=540, width=700,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI))
    fig.update_xaxes(tickangle=-25, tickfont=dict(color=TEXT_PRI))
    return fig


def i1_baseline_vs_ceo(data: dict, models: list[str],
                        embed_model: SentenceTransformer) -> go.Figure:
    labels   = [MODEL_LABELS.get(m, m) for m in models]
    q_labels = list(I1_QUESTIONS.values())
    q_ids    = list(I1_QUESTIONS.keys())

    z = []
    for q_id in q_ids:
        row = []
        for model in models:
            b   = get_responses(data, model, "baseline", q_id)
            c   = get_responses(data, model, "ceo",      q_id)
            sim = cosine_sim(avg_embedding(b, embed_model),
                             avg_embedding(c, embed_model)) if b and c else None
            row.append(sim)
        z.append(row)

    text = [[f"{v:.3f}" if v is not None else "N/A" for v in row] for row in z]
    fig  = go.Figure(go.Heatmap(
        z=z, x=labels, y=q_labels,
        colorscale=COLORSCALE_AMBER, zmin=0.5, zmax=1.0,
        text=text, texttemplate="%{text}", textfont=dict(size=13, color=TEXT_PRI),
        colorbar=dict(title="Cosine Sim", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b> — <b>%{x}</b><br>Baseline↔CEO: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        "Baseline vs CEO Framing Shift<br>"
        "<sup>Lower similarity = greater role-induced framing change</sup>",
        height=400, width=820,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI))
    fig.update_xaxes(tickangle=-25, tickfont=dict(color=TEXT_PRI))
    return fig


# =============================================================================
# INSTRUMENT 2 — S1: Word frequency  (unchanged)
# =============================================================================

def i2_s1_wordfreq_cross_model(data: dict, models: list[str]) -> go.Figure:
    col_labels, col_freqs = [], {}
    for model in models:
        texts = []
        for cond in CONDITIONS:
            texts.extend(get_responses(data, model, cond, "I2_S1"))
        if not texts:
            continue
        tokens = tokenize(" ".join(texts))
        total  = len(tokens) or 1
        label  = MODEL_LABELS.get(model, model)
        col_labels.append(label)
        col_freqs[label] = {w: c / total for w, c in Counter(tokens).items()}

    global_ctr = Counter()
    for freq in col_freqs.values():
        global_ctr.update(freq)
    top_words = [w for w, _ in global_ctr.most_common(TOP_N_WORDS)]

    z = [[col_freqs.get(col, {}).get(w, 0.0) for col in col_labels]
         for w in top_words]

    fig = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=top_words,
        colorscale=COLORSCALE_BLUE,
        colorbar=dict(title="Norm. Freq", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Freq: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        "S1 Word Frequency — Legal Certainty (Cross-Model)<br>"
        "<sup>Immigration visa scenario — all conditions merged</sup>",
        height=580, width=900,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI, size=11))
    fig.update_xaxes(tickangle=-25, tickfont=dict(color=TEXT_PRI, size=11))
    return fig


def i2_s1_wordfreq_baseline_vs_ceo(data: dict, models: list[str]) -> go.Figure:
    n   = len(models)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[MODEL_LABELS.get(m, m) for m in models],
        shared_yaxes=True,
    )

    global_ctr = Counter()
    for model in models:
        for cond in CONDITIONS:
            texts = get_responses(data, model, cond, "I2_S1")
            global_ctr.update(tokenize(" ".join(texts)))
    top_words  = [w for w, _ in global_ctr.most_common(15)]
    bar_colors = {"baseline": "#2176ae", "ceo": "#e8a838"}

    for col_idx, model in enumerate(models, start=1):
        for cond in CONDITIONS:
            texts  = get_responses(data, model, cond, "I2_S1")
            tokens = tokenize(" ".join(texts))
            total  = len(tokens) or 1
            freqs  = Counter(tokens)
            vals   = [freqs.get(w, 0) / total for w in top_words]
            fig.add_trace(
                go.Bar(
                    name=CONDITION_LABELS[cond],
                    x=vals, y=top_words,
                    orientation="h",
                    marker_color=bar_colors[cond],
                    opacity=0.85,
                    showlegend=(col_idx == 1),
                    hovertemplate=f"<b>%{{y}}</b><br>{CONDITION_LABELS[cond]}: %{{x:.4f}}<extra></extra>",
                ),
                row=1, col=col_idx,
            )

    fig.update_layout(
        title=dict(
            text="S1 Baseline vs CEO Word Frequency — Legal Certainty<br>"
                 "<sup>Immigration visa scenario — per model comparison</sup>",
            font=dict(family=SERIF, size=17, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        barmode="group",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI)),
        margin=dict(l=120, r=40, t=100, b=60),
        height=560, width=1200,
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI, size=10))
    fig.update_xaxes(tickfont=dict(color=TEXT_SEC, size=9))
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_PRI
        ann.font.size  = 11
    return fig


# =============================================================================
# INSTRUMENT 2 — S2: Responsibility radar + Accountability mechanisms Sankey
# =============================================================================

def i2_s2_responsibility_radar(data: dict, models: list[str]) -> go.Figure:
    """
    Radar / spider chart — one trace per model × condition.
    Axes = responsibility actor clusters. Value = normalized mention count.
    Solid line = baseline, dashed = CEO.
    """
    actors = list(ACTOR_KEYWORDS.keys())
    n_axes = len(actors)
    # Close the polygon
    axes_closed = actors + [actors[0]]

    fig = go.Figure()

    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#ffffff")

        for cond in CONDITIONS:
            texts    = get_responses(data, model, cond, "I2_S2")
            combined = " ".join(texts).lower()
            total    = len(combined.split()) or 1

            # Score each actor cluster by summing keyword hits
            scores = []
            for actor in actors:
                keywords = ACTOR_KEYWORDS[actor]
                count = sum(
                    len(re.findall(rf"\b{kw}\w*\b", combined))
                    for kw in keywords
                )
                scores.append(count / total * 1000)  # scale for readability

            scores_closed = scores + [scores[0]]

            fig.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=axes_closed,
                mode="lines+markers",
                name=f"{label} ({CONDITION_LABELS[cond]})",
                line=dict(
                    color=color,
                    width=2,
                    dash=CONDITION_DASH[cond],
                ),
                marker=dict(size=5, color=color),
                opacity=0.85,
                hovertemplate=(
                    f"<b>{label}</b> — {CONDITION_LABELS[cond]}<br>"
                    "%{theta}: %{r:.3f}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=dict(
            text="S2 Responsibility Attribution — Who Is Held Accountable?<br>"
                 "<sup>Housing denial scenario — solid = Baseline, dashed = CEO Role</sup>",
            font=dict(family=SERIF, size=17, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(
                visible=True,
                color=TEXT_SEC,
                gridcolor=BORDER,
                tickfont=dict(size=9, color=TEXT_SEC),
            ),
            angularaxis=dict(
                color=TEXT_PRI,
                gridcolor=BORDER,
                tickfont=dict(size=11, color=TEXT_PRI),
            ),
        ),
        paper_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        legend=dict(
            bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT_PRI, size=10),
            x=1.05, y=1.0,
        ),
        margin=dict(l=80, r=200, t=100, b=80),
        height=620,
        width=1000,
    )
    return fig


def i2_s2_accountability_sankey(data: dict, models: list[str]) -> go.Figure:
    """
    Sankey diagram: model × condition → responsible parties → accountability mechanisms.
    Flows are extracted via LLM from run 1 of each model/condition.
    Node widths reflect flow volume (each link weight = 1).
    """
    print("    Extracting S2 accountability structures via LLM...")

    # ---- Collect all unique labels for node deduplication ----
    all_sources      = []   # model+condition labels
    all_parties      = []   # responsible party labels
    all_mechanisms   = []   # mechanism labels
    flow_records     = []   # (source, party, mechanism) tuples

    for model in models:
        label = MODEL_LABELS.get(model, model)
        for cond in CONDITIONS:
            text = get_run1(data, model, cond, "I2_S2")
            if not text:
                continue
            src_label = f"{label}\n({CONDITION_LABELS[cond]})"
            extracted = extract_structured(text, "s2_accountability")

            parties    = extracted.get("responsible_parties", [])
            mechanisms = extracted.get("mechanisms", [])

            all_sources.append(src_label)
            all_parties.extend(parties)
            all_mechanisms.extend(mechanisms)

            for party in parties:
                for mech in mechanisms:
                    flow_records.append((src_label, party, mech))

    # ---- Build deduplicated node list ----
    # Node order: sources | parties | mechanisms
    source_nodes = list(dict.fromkeys(all_sources))
    party_nodes  = list(dict.fromkeys(all_parties))
    mech_nodes   = list(dict.fromkeys(all_mechanisms))

    nodes        = source_nodes + party_nodes + mech_nodes
    node_idx     = {n: i for i, n in enumerate(nodes)}

    # ---- Build link arrays ----
    link_src, link_tgt, link_val, link_color, link_label = [], [], [], [], []

    # Layer 1: model/condition → responsible party
    flow1: dict[tuple, int] = defaultdict(int)
    flow2: dict[tuple, int] = defaultdict(int)

    for src, party, mech in flow_records:
        flow1[(src, party)] += 1
        flow2[(party, mech)] += 1

    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")
        rgba  = hex_to_rgba(color, alpha=0.6)
        for cond in CONDITIONS:
            src_label = f"{label}\n({CONDITION_LABELS[cond]})"
            for party in party_nodes:
                w = flow1.get((src_label, party), 0)
                if w:
                    link_src.append(node_idx[src_label])
                    link_tgt.append(node_idx[party])
                    link_val.append(w)
                    link_color.append(rgba)
                    link_label.append(f"{src_label} → {party}")

    # Layer 2: responsible party → mechanism (neutral color)
    for party in party_nodes:
        for mech in mech_nodes:
            w = flow2.get((party, mech), 0)
            if w:
                link_src.append(node_idx[party])
                link_tgt.append(node_idx[mech])
                link_val.append(w)
                link_color.append("rgba(48,54,61,0.8)")
                link_label.append(f"{party} → {mech}")

    # ---- Node colors ----
    node_colors = []
    for n in nodes:
        if n in source_nodes:
            # Match to model color
            matched = "#555555"
            for model in models:
                if MODEL_LABELS.get(model, model) in n:
                    matched = MODEL_COLORS.get(model, "#555555")
                    break
            node_colors.append(matched)
        elif n in party_nodes:
            node_colors.append("#2176ae")
        else:
            node_colors.append("#3fb68a")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=18,
            line=dict(color=BORDER, width=0.5),
            label=nodes,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>Flow: %{value}<extra></extra>",
        ),
        link=dict(
            source=link_src,
            target=link_tgt,
            value=link_val,
            color=link_color,
            label=link_label,
            hovertemplate="%{label}<br>Weight: %{value}<extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(
            text="S2 Accountability Mechanisms — Model × Role → Responsible Party → Mechanism<br>"
                 "<sup>Housing denial scenario — LLM-extracted, run 1 per model/condition</sup>",
            font=dict(family=SERIF, size=16, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_PRI, size=11),
        margin=dict(l=40, r=40, t=100, b=40),
        height=700,
        width=1400,
    )
    return fig


# =============================================================================
# INSTRUMENT 2 — S3: Enforcement challenges → solutions Sankey
# =============================================================================

def i2_s3_enforcement_sankey(data: dict, models: list[str]) -> go.Figure:
    """
    Sankey diagram: model × condition → enforcement challenges → proposed solutions.
    Flows extracted via LLM from run 1 of each model/condition.
    """
    print("    Extracting S3 enforcement structures via LLM...")

    all_sources    = []
    all_challenges = []
    all_solutions  = []
    flow_records   = []

    for model in models:
        label = MODEL_LABELS.get(model, model)
        for cond in CONDITIONS:
            text = get_run1(data, model, cond, "I2_S3")
            if not text:
                continue
            src_label = f"{label}\n({CONDITION_LABELS[cond]})"
            extracted  = extract_structured(text, "s3_enforcement")

            challenges = extracted.get("challenges", [])
            solutions  = extracted.get("solutions",  [])

            all_sources.extend([src_label])
            all_challenges.extend(challenges)
            all_solutions.extend(solutions)

            for ch in challenges:
                for sol in solutions:
                    flow_records.append((src_label, ch, sol))

    # ---- Deduplicated node list ----
    source_nodes    = list(dict.fromkeys(all_sources))
    challenge_nodes = list(dict.fromkeys(all_challenges))
    solution_nodes  = list(dict.fromkeys(all_solutions))

    nodes    = source_nodes + challenge_nodes + solution_nodes
    node_idx = {n: i for i, n in enumerate(nodes)}

    # ---- Aggregate flows ----
    flow1: dict[tuple, int] = defaultdict(int)
    flow2: dict[tuple, int] = defaultdict(int)

    for src, ch, sol in flow_records:
        flow1[(src, ch)]  += 1
        flow2[(ch, sol)]  += 1

    link_src, link_tgt, link_val, link_color, link_label = [], [], [], [], []

    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")
        rgba  = hex_to_rgba(color, alpha=0.6)
        for cond in CONDITIONS:
            src_label = f"{label}\n({CONDITION_LABELS[cond]})"
            for ch in challenge_nodes:
                w = flow1.get((src_label, ch), 0)
                if w:
                    link_src.append(node_idx[src_label])
                    link_tgt.append(node_idx[ch])
                    link_val.append(w)
                    link_color.append(rgba)
                    link_label.append(f"{src_label} → {ch}")

    for ch in challenge_nodes:
        for sol in solution_nodes:
            w = flow2.get((ch, sol), 0)
            if w:
                link_src.append(node_idx[ch])
                link_tgt.append(node_idx[sol])
                link_val.append(w)
                link_color.append("rgba(48,54,61,0.8)")
                link_label.append(f"{ch} → {sol}")

    # ---- Node colors ----
    node_colors = []
    for n in nodes:
        if n in source_nodes:
            matched = "#555555"
            for model in models:
                if MODEL_LABELS.get(model, model) in n:
                    matched = MODEL_COLORS.get(model, "#555555")
                    break
            node_colors.append(matched)
        elif n in challenge_nodes:
            node_colors.append("#e05c5c")   # red = problems
        else:
            node_colors.append("#3fb68a")   # green = solutions

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=18,
            line=dict(color=BORDER, width=0.5),
            label=nodes,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>Flow: %{value}<extra></extra>",
        ),
        link=dict(
            source=link_src,
            target=link_tgt,
            value=link_val,
            color=link_color,
            label=link_label,
            hovertemplate="%{label}<br>Weight: %{value}<extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(
            text="S3 Enforcement Challenges → Proposed Solutions<br>"
                 "<sup>Audit jurisdiction scenario — LLM-extracted, run 1 per model/condition</sup>",
            font=dict(family=SERIF, size=16, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_PRI, size=11),
        margin=dict(l=40, r=40, t=100, b=40),
        height=700,
        width=1400,
    )
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    embed_model = None  # lazy-load only if needed

    # -------------------------------------------------------------------------
    # Instrument 1
    # -------------------------------------------------------------------------
    if "instrument_1" in ACTIVE_INSTRUMENTS:
        i1_path = RAW_DIR / "instrument_1.json"
        if not i1_path.exists():
            print(f"[skip] instrument_1.json not found at {i1_path}")
        else:
            print("Loading instrument_1.json...")
            with open(i1_path, "r", encoding="utf-8") as f:
                i1_data = json.load(f)
            models = list(i1_data.keys())
            out    = RESULTS_DIR / "instrument_1"

            print("Building I1 word frequency heatmaps...")
            for q_id, q_label in I1_QUESTIONS.items():
                slug = q_label.lower().replace(" ", "_")
                save_fig(i1_wordfreq_heatmap(i1_data, q_id, q_label, models),
                         out / f"wordfreq_{slug}")

            print("Loading embedding model...")
            if embed_model is None:
                embed_model = SentenceTransformer(EMBEDDING_MODEL)

            print("Building I1 cross-model similarity...")
            save_fig(i1_cross_model_similarity(i1_data, models, embed_model),
                     out / "similarity_cross_model")

            print("Building I1 baseline vs CEO shift...")
            save_fig(i1_baseline_vs_ceo(i1_data, models, embed_model),
                     out / "similarity_baseline_vs_ceo")

    # -------------------------------------------------------------------------
    # Instrument 2
    # -------------------------------------------------------------------------
    if "instrument_2" in ACTIVE_INSTRUMENTS:
        i2_path = RAW_DIR / "instrument_2.json"
        if not i2_path.exists():
            print(f"[skip] instrument_2.json not found at {i2_path}")
        else:
            print("\nLoading instrument_2.json...")
            with open(i2_path, "r", encoding="utf-8") as f:
                i2_data = json.load(f)
            models = list(i2_data.keys())
            out    = RESULTS_DIR / "instrument_2"

            print("Building I2 S1 word frequency — cross-model...")
            save_fig(i2_s1_wordfreq_cross_model(i2_data, models),
                     out / "s1_wordfreq_cross_model")

            print("Building I2 S1 word frequency — baseline vs CEO...")
            save_fig(i2_s1_wordfreq_baseline_vs_ceo(i2_data, models),
                     out / "s1_wordfreq_baseline_vs_ceo")

            print("Building I2 S2 responsibility radar...")
            save_fig(i2_s2_responsibility_radar(i2_data, models),
                     out / "s2_responsibility_radar")

            print("Building I2 S2 accountability mechanisms Sankey...")
            save_fig(i2_s2_accountability_sankey(i2_data, models),
                     out / "s2_accountability_sankey")

            print("Building I2 S3 enforcement challenges Sankey...")
            save_fig(i2_s3_enforcement_sankey(i2_data, models),
                     out / "s3_enforcement_sankey")

    print("\nAll plots complete.")