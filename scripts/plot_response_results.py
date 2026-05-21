"""
plot_response_results.py

Generates visualizations for Instruments 1–5. Controls which instruments
are plotted via ACTIVE_INSTRUMENTS.

Light theme throughout. Statistics (H1–H4) are printed before any plotting.

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
    instrument_3/
      scores_grouped_bar_<dim>.[html|png]
      scores_heatmap_<dim>.[html|png]
      scores_radar_<scenario>.[html|png]
      condition_delta_heatmap.[html|png]
      condition_side_by_side.[html|png]
    instrument_4/
      elp_radar_all_models.[html|png]
      asymmetry_heatmap.[html|png]
      peer_scores_heatmap.[html|png]
      elp_profiles.json
    instrument_5/
      source_legitimacy_heatmap.[html|png]
      source_type_sankey.[html|png]
      citation_overlap_heatmap.[html|png]
      jurisdiction_radar.[html|png]
"""

import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from time import sleep

import anthropic
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.corpus import stopwords
from scipy import stats
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config
# =============================================================================

RAW_DIR           = Path("data/raw")
RESULTS_DIR       = Path("results")
SANKEY_CACHE_FILE = RESULTS_DIR / "sankey_label_cache.json"

ACTIVE_INSTRUMENTS = ["instrument_1", "instrument_2", "instrument_3", "instrument_4", "instrument_5"]

PEER_EVAL_FILE = Path("data/prompts/peer_eval_pairs.json")

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

# Instrument 3 — scenarios and dimensions
I3_SCENARIOS = {
    "I3_S1": "Parole Risk Scores",
    "I3_S2": "AI-Drafted Legislation",
    "I3_S3": "Political Speech Moderation",
    "I3_S4": "Emergency Housing Allocation",
    "I3_S5": "Central Bank Advisory",
    "I3_S6": "Foreign Ministry Risk Assessments",
}

I3_DIMENSIONS = {
    "legal_certainty": "Legal Certainty",
    "accountability":  "Accountability",
    "enforceability":  "Enforceability",
}

# Instrument 4 — peer eval dimensions
I4_DIMENSIONS = {
    "legal_certainty_adequacy":  "Legal Certainty Adequacy",
    "accountability_mechanisms": "Accountability Mechanisms",
    "enforcement_conditions":    "Enforcement Conditions",
}

I4_QUESTION_LABELS = {
    "I1_Q1": "I1: Legal Certainty",
    "I1_Q2": "I1: Accountability",
    "I1_Q3": "I1: Enforceability",
    "I2_S1": "I2: Visa Scenario",
    "I2_S2": "I2: Housing Scenario",
    "I2_S3": "I2: Audit Scenario",
}

# Instrument 5 — source jurisdictions for radar
I5_JURISDICTIONS = ["EU", "US", "UN", "UK", "unspecified", "other"]

# I1 self-reported source type metadata
SOURCE_TYPE_LABELS = {
    "international_treaty": "International Treaty",
    "national_legislation": "National Legislation",
    "court_decision":       "Court Decision",
    "academic_work":        "Academic Work",
    "policy_framework":     "Policy Framework",
    "news_media":           "News / Media",
    "unverifiable":         "Unverifiable",
    "implicit_only":        "Implicit Only",
}

# Proxy legitimacy tier derived from source type (used when legitimacy_tier not present)
SOURCE_TYPE_TO_TIER = {
    "international_treaty": 1,
    "court_decision":       1,
    "national_legislation": 1,
    "academic_work":        2,
    "policy_framework":     2,
    "news_media":           3,
    "unverifiable":         4,
    "implicit_only":        4,
}

SOURCE_TYPE_COLORS = {
    "international_treaty": "#1d4ed8",
    "national_legislation": "#2176ae",
    "court_decision":       "#0891b2",
    "academic_work":        "#16a34a",
    "policy_framework":     "#65a30d",
    "news_media":           "#d97706",
    "unverifiable":         "#dc2626",
    "implicit_only":        "#7c3aed",
}

# Responsibility actors for radar chart axes (S2)
RESPONSIBILITY_ACTORS = [
    "city / government", "vendor / developer",
    "regulator", "court / judiciary",
    "official", "applicant / public",
]

ACTOR_KEYWORDS = {
    "city / government":    ["city", "government", "municipal", "authority", "state"],
    "vendor / developer":   ["vendor", "developer", "company", "provider", "contractor"],
    "regulator":            ["regulator", "regulatory", "oversight", "agency", "watchdog"],
    "court / judiciary":    ["court", "judicial", "judge", "tribunal", "legal action"],
    "official":             ["official", "administrator", "bureaucrat", "civil servant"],
    "applicant / public":   ["applicant", "resident", "citizen", "public", "individual"],
}

# =============================================================================
# Light theme styling
# =============================================================================

BG       = "#ffffff"
SURFACE  = "#f6f8fa"
BORDER   = "#d0d7de"
TEXT_PRI = "#1f2328"
TEXT_SEC = "#57606a"
MONO     = "'Courier New', monospace"
SERIF    = "Georgia, serif"

# Sequential: blue (low) → red (high)  — print-safe, red/blue convention
COLORSCALE_REDBLUE = [
    [0.0, "#053061"], [0.2, "#2166ac"],
    [0.4, "#74add1"], [0.6, "#f4a582"],
    [0.8, "#d6604d"], [1.0, "#67001f"],
]
# Diverging: blue (negative/low) → white (zero) → red (positive/high)
COLORSCALE_DIVERGE = [
    [0.0, "#2166ac"], [0.35, "#74add1"],
    [0.5, "#f7f7f7"], [0.65, "#f4a582"], [1.0, "#d6604d"],
]
# Legitimacy: tier-1 blue (authoritative) → tier-4 red (vague/fabricated)
COLORSCALE_LEGITIMACY = [
    [0.0, "#2166ac"], [0.33, "#74add1"],
    [0.67, "#f4a582"], [1.0, "#d6604d"],
]

# One color per model, consistent across all plots
MODEL_COLORS = {
    "gpt-4o":                        "#2176ae",
    "claude-sonnet":                 "#16a34a",
    "deepseek-v3":                   "#dc2626",
    "mistral-large":                 "#d97706",
    "gemini-3.1-flash-lite-preview": "#7c3aed",
}

CONDITION_DASH = {"baseline": "solid", "ceo": "dash"}

# Second B&W distinguisher: line dash per model (for radar/line charts)
MODEL_DASH = {
    "gpt-4o":                        "solid",
    "claude-sonnet":                 "dash",
    "deepseek-v3":                   "dot",
    "mistral-large":                 "dashdot",
    "gemini-3.1-flash-lite-preview": "longdash",
}

# Second B&W distinguisher: marker symbol per model (for scatter/radar)
MODEL_SYMBOLS = {
    "gpt-4o":                        "circle",
    "claude-sonnet":                 "square",
    "deepseek-v3":                   "diamond",
    "mistral-large":                 "triangle-up",
    "gemini-3.1-flash-lite-preview": "cross",
}

# Second B&W distinguisher: fill pattern per model (for bar charts)
MODEL_PATTERNS = {
    "gpt-4o":                        "",
    "claude-sonnet":                 "/",
    "deepseek-v3":                   "\\",
    "mistral-large":                 "x",
    "gemini-3.1-flash-lite-preview": "+",
}

# Condition fill pattern (for condition-split bar charts)
CONDITION_PATTERNS = {"baseline": "", "ceo": "\\"}

# Source-type fill pattern (for stacked source-type bars)
SOURCE_TYPE_PATTERNS = {
    "international_treaty": "",
    "national_legislation": "/",
    "court_decision":       "\\",
    "academic_work":        "+",
    "policy_framework":     "x",
    "news_media":           "-",
    "unverifiable":         "|",
    "implicit_only":        ".",
}

# Jurisdiction fill pattern
JURIS_PATTERNS = {
    "EU": "", "US": "/", "UN": "\\",
    "UK": "x", "unspecified": "+", "other": "-",
}


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


def hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
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


def _extract_text(val) -> str:
    """
    Extract response text from either legacy string or new {raw, parsed} dict.
    Used to maintain backward compatibility with old I1 plain-string data.
    """
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        parsed = val.get("parsed") or {}
        return parsed.get("response") or val.get("raw") or ""
    return ""


def get_responses(data: dict, model: str, condition: str,
                  question_id: str) -> list[str]:
    runs = data.get(model, {}).get(condition, {})
    out  = []
    for r in sorted(runs.keys(), key=int):
        text = _extract_text(runs[r].get(question_id))
        if text:
            out.append(text)
    return out


def get_run1(data: dict, model: str, condition: str, question_id: str) -> str:
    val = data.get(model, {}).get(condition, {}).get("1", {}).get(question_id)
    return _extract_text(val) if val is not None else ""


def avg_embedding(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    embs = model.encode(texts, show_progress_bar=False)
    return embs.mean(axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# =============================================================================
# LLM extraction helper
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
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
        result = json.loads(raw)
        _extraction_cache[cache_key] = result
        sleep(0.3)
        return result
    except Exception as e:
        print(f"    [extract_structured] failed for {extraction_type}: {e}")
        return fallback


# =============================================================================
# Sankey label deduplication via LLM clustering
# Cache is persisted to SANKEY_CACHE_FILE so clustering only runs once.
# =============================================================================

_sankey_label_cache: dict[str, dict] = {}


def _load_sankey_cache() -> None:
    global _sankey_label_cache
    if SANKEY_CACHE_FILE.exists():
        with open(SANKEY_CACHE_FILE, "r", encoding="utf-8") as f:
            _sankey_label_cache = json.load(f)


def _save_sankey_cache() -> None:
    SANKEY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SANKEY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_sankey_label_cache, f, indent=2, ensure_ascii=False)


def get_sankey_label_mapping(labels: list[str], context: str) -> dict[str, str]:
    """
    Use Claude Sonnet to cluster synonym labels and return a canonical mapping.
    Cache key is context + sorted label set. Returns {original: canonical}.
    """
    if not labels:
        return {}
    cache_key = f"{context}::{':'.join(sorted(labels))}"
    if cache_key in _sankey_label_cache:
        return _sankey_label_cache[cache_key]

    prompt = (
        "You are a research assistant deduplicating labels in a Sankey diagram "
        f"for an AI governance study (context: {context}). "
        "Some labels below are near-synonyms or paraphrases of the same concept.\n\n"
        "Map every label to its best canonical form. "
        "Near-synonyms should share the same canonical label. "
        "If a label is already canonical (no synonym in the list), map it to itself.\n\n"
        f"Labels:\n{json.dumps(labels, indent=2)}\n\n"
        "Respond ONLY with a JSON object mapping each input label to its canonical form. "
        "No preamble, no markdown.\n"
        'Example: {"independent audit": "independent audit", '
        '"third-party audit": "independent audit"}'
    )

    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
        mapping = json.loads(raw)
        # Ensure every input label is present
        for label in labels:
            if label not in mapping:
                mapping[label] = label
        _sankey_label_cache[cache_key] = mapping
        sleep(0.3)
        return mapping
    except Exception as e:
        print(f"    [get_sankey_label_mapping] failed: {e}")
        identity = {label: label for label in labels}
        _sankey_label_cache[cache_key] = identity
        return identity


# =============================================================================
# INSTRUMENT 1 plots
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
        colorscale=COLORSCALE_REDBLUE,
        colorbar=dict(title="Norm. Freq", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Freq: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(f"Word Frequency — <b>{q_label}</b>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI, size=11),
                     gridcolor=BORDER)
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
        colorscale=COLORSCALE_REDBLUE, zmin=0.5, zmax=1.0,
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
        colorscale=COLORSCALE_REDBLUE, zmin=0.5, zmax=1.0,
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
# INSTRUMENT 2 — S1: Word frequency
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
        colorscale=COLORSCALE_REDBLUE,
        colorbar=dict(title="Norm. Freq", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Freq: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        "S1 Word Frequency — Legal Certainty (Cross-Model)<br>"
        "<sup>Immigration visa scenario — all conditions merged</sup>",
        height=580, width=900,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI, size=11),
                     gridcolor=BORDER)
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
    bar_colors = {"baseline": "#2176ae", "ceo": "#d97706"}

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
                    marker=dict(
                        color=bar_colors[cond],
                        pattern=dict(shape=CONDITION_PATTERNS[cond], fillmode="overlay",
                                     fgcolor="rgba(0,0,0,0.35)", size=8),
                    ),
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
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI, size=10),
                     gridcolor=BORDER)
    fig.update_xaxes(tickfont=dict(color=TEXT_SEC, size=9))
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_PRI
        ann.font.size  = 11
    return fig


# =============================================================================
# INSTRUMENT 2 — S2: Responsibility radar + Accountability mechanisms Sankey
# =============================================================================

def i2_s2_responsibility_radar(data: dict, models: list[str]) -> go.Figure:
    actors = list(ACTOR_KEYWORDS.keys())
    axes_closed = actors + [actors[0]]

    fig = go.Figure()

    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")

        for cond in CONDITIONS:
            texts    = get_responses(data, model, cond, "I2_S2")
            combined = " ".join(texts).lower()
            total    = len(combined.split()) or 1

            scores = []
            for actor in actors:
                keywords = ACTOR_KEYWORDS[actor]
                count = sum(
                    len(re.findall(rf"\b{kw}\w*\b", combined))
                    for kw in keywords
                )
                scores.append(count / total * 1000)

            scores_closed = scores + [scores[0]]

            fig.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=axes_closed,
                mode="lines+markers",
                name=f"{label} ({CONDITION_LABELS[cond]})",
                line=dict(color=color, width=2, dash=CONDITION_DASH[cond]),
                marker=dict(size=6, color=color, symbol=MODEL_SYMBOLS.get(model, "circle")),
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
                visible=True, color=TEXT_SEC, gridcolor=BORDER,
                tickfont=dict(size=9, color=TEXT_SEC),
            ),
            angularaxis=dict(
                color=TEXT_PRI, gridcolor=BORDER,
                tickfont=dict(size=11, color=TEXT_PRI),
            ),
        ),
        paper_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI, size=10), x=1.05, y=1.0),
        margin=dict(l=80, r=200, t=100, b=80),
        height=620, width=1000,
    )
    return fig


def i2_s2_accountability_sankey(data: dict, models: list[str]) -> go.Figure:
    """
    Sankey: model × condition → responsible parties → accountability mechanisms.
    Includes LLM synonym-deduplication of node labels before building the diagram.
    """
    print("    Extracting S2 accountability structures via LLM...")

    all_sources    = []
    all_parties    = []
    all_mechanisms = []
    flow_records   = []

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

    # ---- LLM label deduplication ----
    unique_parties = list(dict.fromkeys(all_parties))
    unique_mechs   = list(dict.fromkeys(all_mechanisms))
    if unique_parties:
        print("    Clustering responsible-party synonyms via LLM...")
        party_map = get_sankey_label_mapping(unique_parties, "accountability_responsible_parties")
    else:
        party_map = {}
    if unique_mechs:
        print("    Clustering accountability-mechanism synonyms via LLM...")
        mech_map = get_sankey_label_mapping(unique_mechs, "accountability_mechanisms")
    else:
        mech_map = {}

    flow_records   = [(src, party_map.get(p, p), mech_map.get(m, m))
                      for src, p, m in flow_records]
    all_parties    = [party_map.get(p, p) for p in all_parties]
    all_mechanisms = [mech_map.get(m, m) for m in all_mechanisms]

    # ---- Build deduplicated node list ----
    source_nodes = list(dict.fromkeys(all_sources))
    party_nodes  = list(dict.fromkeys(all_parties))
    mech_nodes   = list(dict.fromkeys(all_mechanisms))

    nodes    = source_nodes + party_nodes + mech_nodes
    node_idx = {n: i for i, n in enumerate(nodes)}

    # ---- Build link arrays ----
    flow1: dict[tuple, int] = defaultdict(int)
    flow2: dict[tuple, int] = defaultdict(int)

    for src, party, mech in flow_records:
        flow1[(src, party)] += 1
        flow2[(party, mech)] += 1

    link_src, link_tgt, link_val, link_color, link_label = [], [], [], [], []

    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")
        rgba  = hex_to_rgba(color)
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

    for party in party_nodes:
        for mech in mech_nodes:
            w = flow2.get((party, mech), 0)
            if w:
                link_src.append(node_idx[party])
                link_tgt.append(node_idx[mech])
                link_val.append(w)
                link_color.append("rgba(160,168,176,0.6)")
                link_label.append(f"{party} → {mech}")

    node_colors = []
    for n in nodes:
        if n in source_nodes:
            matched = "#9ca3af"
            for model in models:
                if MODEL_LABELS.get(model, model) in n:
                    matched = MODEL_COLORS.get(model, "#9ca3af")
                    break
            node_colors.append(matched)
        elif n in party_nodes:
            node_colors.append("#2176ae")
        else:
            node_colors.append("#16a34a")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=18,
            line=dict(color=BORDER, width=0.5),
            label=nodes, color=node_colors,
            hovertemplate="<b>%{label}</b><br>Flow: %{value}<extra></extra>",
        ),
        link=dict(
            source=link_src, target=link_tgt, value=link_val,
            color=link_color, label=link_label,
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
        height=700, width=1400,
    )
    return fig


# =============================================================================
# INSTRUMENT 2 — S3: Enforcement challenges → solutions Sankey
# =============================================================================

def i2_s3_enforcement_sankey(data: dict, models: list[str]) -> go.Figure:
    """
    Sankey: model × condition → enforcement challenges → proposed solutions.
    Includes LLM synonym-deduplication before building the diagram.
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
            extracted = extract_structured(text, "s3_enforcement")

            challenges = extracted.get("challenges", [])
            solutions  = extracted.get("solutions",  [])

            all_sources.extend([src_label])
            all_challenges.extend(challenges)
            all_solutions.extend(solutions)

            for ch in challenges:
                for sol in solutions:
                    flow_records.append((src_label, ch, sol))

    # ---- LLM label deduplication ----
    unique_ch  = list(dict.fromkeys(all_challenges))
    unique_sol = list(dict.fromkeys(all_solutions))
    if unique_ch:
        print("    Clustering enforcement-challenge synonyms via LLM...")
        ch_map = get_sankey_label_mapping(unique_ch, "enforcement_challenges")
    else:
        ch_map = {}
    if unique_sol:
        print("    Clustering enforcement-solution synonyms via LLM...")
        sol_map = get_sankey_label_mapping(unique_sol, "enforcement_solutions")
    else:
        sol_map = {}

    flow_records   = [(src, ch_map.get(c, c), sol_map.get(s, s))
                      for src, c, s in flow_records]
    all_challenges = [ch_map.get(c, c) for c in all_challenges]
    all_solutions  = [sol_map.get(s, s) for s in all_solutions]

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
        rgba  = hex_to_rgba(color)
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
                link_color.append("rgba(160,168,176,0.6)")
                link_label.append(f"{ch} → {sol}")

    node_colors = []
    for n in nodes:
        if n in source_nodes:
            matched = "#9ca3af"
            for model in models:
                if MODEL_LABELS.get(model, model) in n:
                    matched = MODEL_COLORS.get(model, "#9ca3af")
                    break
            node_colors.append(matched)
        elif n in challenge_nodes:
            node_colors.append("#dc2626")
        else:
            node_colors.append("#16a34a")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=18,
            line=dict(color=BORDER, width=0.5),
            label=nodes, color=node_colors,
            hovertemplate="<b>%{label}</b><br>Flow: %{value}<extra></extra>",
        ),
        link=dict(
            source=link_src, target=link_tgt, value=link_val,
            color=link_color, label=link_label,
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
        height=700, width=1400,
    )
    return fig


# =============================================================================
# INSTRUMENT 3 helpers
# =============================================================================

def extract_i3_scores(data: dict, models: list[str]) -> dict:
    """
    Flatten I3 parsed ratings into:
      scores[model][condition][scenario_id][dimension] -> list of scores per run
    """
    scores = {}
    for model in models:
        scores[model] = {}
        for cond in CONDITIONS:
            scores[model][cond] = {}
            runs = data.get(model, {}).get(cond, {})
            for run_num in sorted(runs.keys(), key=int):
                parsed = runs[run_num].get("parsed")
                if not parsed:
                    continue
                for s_id, dims in parsed.items():
                    scores[model][cond].setdefault(s_id, {})
                    for dim, val in dims.items():
                        scores[model][cond][s_id].setdefault(dim, [])
                        if isinstance(val.get("score"), (int, float)):
                            scores[model][cond][s_id][dim].append(val["score"])
    return scores


def mean_std(values: list) -> tuple[float, float]:
    if not values:
        return (None, None)
    arr = np.array(values, dtype=float)
    return (float(arr.mean()), float(arr.std()))


# =============================================================================
# INSTRUMENT 3 plots
# =============================================================================

def i3_grouped_bar(scores: dict, models: list[str], dimension: str,
                   dim_label: str) -> go.Figure:
    scenario_labels = list(I3_SCENARIOS.values())
    scenario_ids    = list(I3_SCENARIOS.keys())
    fig = go.Figure()
    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")
        means, errors = [], []
        for s_id in scenario_ids:
            vals = scores.get(model, {}).get("baseline", {}).get(s_id, {}).get(dimension, [])
            m, s = mean_std(vals)
            means.append(m)
            errors.append(s if s is not None else 0)
        fig.add_trace(go.Bar(
            name=label, x=scenario_labels, y=means,
            error_y=dict(type="data", array=errors, visible=True,
                         color=color, thickness=1.5, width=4),
            marker=dict(
                color=color,
                pattern=dict(shape=MODEL_PATTERNS.get(model, ""), fillmode="overlay",
                             fgcolor="rgba(0,0,0,0.35)", size=8),
            ),
            opacity=0.85,
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>{dim_label}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=f"I3 Scores — <b>{dim_label}</b> (Baseline)<br>"
                        "<sup>Mean across 3 runs — error bars = std dev</sup>",
                   font=dict(family=SERIF, size=17, color=TEXT_PRI), x=0.5, xanchor="center"),
        barmode="group", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=10), gridcolor=BORDER),
        yaxis=dict(range=[0, 10.5], tickfont=dict(color=TEXT_PRI), gridcolor=BORDER,
                   title="Score (1–10)"),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI)),
        margin=dict(l=60, r=40, t=90, b=100), height=520, width=1200,
    )
    return fig


def i3_heatmap(scores: dict, models: list[str], dimension: str,
               dim_label: str) -> go.Figure:
    model_labels    = [MODEL_LABELS.get(m, m) for m in models]
    scenario_labels = list(I3_SCENARIOS.values())
    scenario_ids    = list(I3_SCENARIOS.keys())
    z, text = [], []
    for model in models:
        row, row_text = [], []
        for s_id in scenario_ids:
            vals = scores.get(model, {}).get("baseline", {}).get(s_id, {}).get(dimension, [])
            m, _ = mean_std(vals)
            row.append(m)
            row_text.append(f"{m:.1f}" if m is not None else "N/A")
        z.append(row)
        text.append(row_text)
    fig = go.Figure(go.Heatmap(
        z=z, x=scenario_labels, y=model_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=12, color=TEXT_PRI),
        colorscale=COLORSCALE_REDBLUE, zmin=1, zmax=10,
        colorbar=dict(title="Score", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Score: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"I3 Score Heatmap — <b>{dim_label}</b> (Baseline)<br>"
                        "<sup>Mean across 3 runs</sup>",
                   font=dict(family=SERIF, size=17, color=TEXT_PRI), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickangle=-25, tickfont=dict(color=TEXT_PRI, size=10), gridcolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT_PRI), autorange="reversed", gridcolor=BORDER),
        margin=dict(l=160, r=60, t=90, b=120), height=400, width=1000,
    )
    return fig


def i3_radar(scores: dict, models: list[str], scenario_id: str,
             scenario_label: str) -> go.Figure:
    dims       = list(I3_DIMENSIONS.keys())
    dim_labels = list(I3_DIMENSIONS.values())
    closed     = dim_labels + [dim_labels[0]]
    fig = go.Figure()
    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")
        vals  = []
        for dim in dims:
            run_scores = scores.get(model, {}).get("baseline", {}).get(scenario_id, {}).get(dim, [])
            m, _ = mean_std(run_scores)
            vals.append(m if m is not None else 0)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=closed, mode="lines+markers",
            name=label,
            line=dict(color=color, width=2, dash=MODEL_DASH.get(model, "solid")),
            marker=dict(size=6, color=color, symbol=MODEL_SYMBOLS.get(model, "circle")),
            opacity=0.85,
            hovertemplate=f"<b>{label}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=f"I3 Tripod Scores — <b>{scenario_label}</b><br>"
                        "<sup>Baseline mean across 3 runs</sup>",
                   font=dict(family=SERIF, size=17, color=TEXT_PRI), x=0.5, xanchor="center"),
        polar=dict(bgcolor=SURFACE,
                   radialaxis=dict(visible=True, range=[0, 10], color=TEXT_SEC,
                                   gridcolor=BORDER, tickfont=dict(size=9, color=TEXT_SEC)),
                   angularaxis=dict(color=TEXT_PRI, gridcolor=BORDER,
                                    tickfont=dict(size=12, color=TEXT_PRI))),
        paper_bgcolor=BG, font=dict(family=MONO, color=TEXT_SEC),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI, size=10), x=1.05, y=1.0),
        margin=dict(l=80, r=180, t=100, b=80), height=520, width=800,
    )
    return fig


def i3_delta_heatmap(scores: dict, models: list[str],
                     h2_stats: dict | None = None) -> go.Figure:
    model_labels    = [MODEL_LABELS.get(m, m) for m in models]
    scenario_labels = list(I3_SCENARIOS.values())
    scenario_ids    = list(I3_SCENARIOS.keys())
    dims            = list(I3_DIMENSIONS.keys())
    z, text = [], []
    for model in models:
        row, row_text = [], []
        for s_id in scenario_ids:
            deltas = []
            for dim in dims:
                b_vals = scores.get(model, {}).get("baseline", {}).get(s_id, {}).get(dim, [])
                c_vals = scores.get(model, {}).get("ceo",      {}).get(s_id, {}).get(dim, [])
                b_m, _ = mean_std(b_vals)
                c_m, _ = mean_std(c_vals)
                if b_m is not None and c_m is not None:
                    deltas.append(c_m - b_m)
            delta = float(np.mean(deltas)) if deltas else None
            row.append(delta)
            row_text.append(f"{delta:+.2f}" if delta is not None else "N/A")
        z.append(row)
        text.append(row_text)

    subtitle = "Averaged across all three tripod dimensions — green = CEO rated higher"
    if h2_stats:
        parts = []
        for dim in dims:
            if dim in h2_stats:
                d = h2_stats[dim]
                parts.append(f"{I3_DIMENSIONS[dim]}: p={d['p']:.3f}")
        if parts:
            subtitle += "<br>H2 one-sample t-test: " + " | ".join(parts)

    fig = go.Figure(go.Heatmap(
        z=z, x=scenario_labels, y=model_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=12, color=TEXT_PRI),
        colorscale=COLORSCALE_DIVERGE, zmid=0,
        colorbar=dict(title="CEO − Baseline", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Δ: %{z:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"I3 Condition Shift — CEO Role minus Baseline<br>"
                        f"<sup>{subtitle}</sup>",
                   font=dict(family=SERIF, size=17, color=TEXT_PRI), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickangle=-25, tickfont=dict(color=TEXT_PRI, size=10), gridcolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT_PRI), autorange="reversed", gridcolor=BORDER),
        margin=dict(l=160, r=80, t=110, b=120), height=420, width=1050,
    )
    return fig


def i3_condition_bars(scores: dict, models: list[str]) -> go.Figure:
    scenario_labels = list(I3_SCENARIOS.values())
    scenario_ids    = list(I3_SCENARIOS.keys())
    dims            = list(I3_DIMENSIONS.keys())
    n               = len(models)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[MODEL_LABELS.get(m, m) for m in models],
                        shared_yaxes=True)
    cond_colors = {"baseline": "#2176ae", "ceo": "#d97706"}
    for col_idx, model in enumerate(models, start=1):
        for cond in CONDITIONS:
            means = []
            for s_id in scenario_ids:
                dim_means = []
                for dim in dims:
                    vals = scores.get(model, {}).get(cond, {}).get(s_id, {}).get(dim, [])
                    m, _ = mean_std(vals)
                    if m is not None:
                        dim_means.append(m)
                means.append(float(np.mean(dim_means)) if dim_means else None)
            fig.add_trace(
                go.Bar(name=CONDITION_LABELS[cond], x=scenario_labels, y=means,
                       marker=dict(
                           color=cond_colors[cond],
                           pattern=dict(shape=CONDITION_PATTERNS[cond], fillmode="overlay",
                                        fgcolor="rgba(0,0,0,0.35)", size=8),
                       ),
                       opacity=0.85,
                       showlegend=(col_idx == 1),
                       hovertemplate=f"<b>{CONDITION_LABELS[cond]}</b><br>%{{x}}<br>Avg: %{{y:.2f}}<extra></extra>"),
                row=1, col=col_idx,
            )
    fig.update_layout(
        title=dict(text="I3 Baseline vs CEO — Average Score per Scenario<br>"
                        "<sup>Averaged across all three tripod dimensions and 3 runs</sup>",
                   font=dict(family=SERIF, size=17, color=TEXT_PRI), x=0.5, xanchor="center"),
        barmode="group", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI)),
        margin=dict(l=60, r=40, t=100, b=120), height=480, width=1400,
    )
    fig.update_yaxes(range=[0, 10.5], tickfont=dict(color=TEXT_PRI), gridcolor=BORDER)
    fig.update_xaxes(tickangle=-35, tickfont=dict(color=TEXT_SEC, size=8))
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_PRI
        ann.font.size  = 11
    return fig


# =============================================================================
# INSTRUMENT 4 helpers
# =============================================================================

def extract_i4_scores(i4_data: dict, pairs: list[dict]) -> dict:
    """
    Flatten I4 parsed ratings into:
      peer_scores[pair_id][question_id][dimension] -> score
    """
    peer_scores = {}
    for pair in pairs:
        pair_id = pair["pair_id"]
        peer_scores[pair_id] = {}
        for q_id, q_data in i4_data.get(pair_id, {}).items():
            parsed = q_data.get("parsed")
            if not parsed:
                continue
            peer_scores[pair_id][q_id] = {
                dim: val.get("score")
                for dim, val in parsed.items()
                if isinstance(val.get("score"), (int, float))
            }
    return peer_scores


def build_elp(i3_scores: dict, peer_scores: dict, pairs: list[dict],
              models: list[str]) -> dict:
    """
    Build the Epistemic Legitimacy Profile for each model.
    """
    dims = list(I3_DIMENSIONS.keys())
    elp  = {}

    for model in models:
        label = MODEL_LABELS.get(model, model)

        i3_strict = {}
        for dim in dims:
            all_scores = []
            for s_id in I3_SCENARIOS:
                vals = i3_scores.get(model, {}).get("baseline", {}).get(s_id, {}).get(dim, [])
                all_scores.extend(vals)
            m, _ = mean_std(all_scores)
            i3_strict[dim] = round(10 - m, 3) if m is not None else None

        i4_map = {
            "legal_certainty_adequacy":  "legal_certainty",
            "accountability_mechanisms": "accountability",
            "enforcement_conditions":    "enforceability",
        }
        i4_peer = {dim: [] for dim in dims}
        for pair in pairs:
            if pair["evaluator"] != model:
                continue
            for q_id, dim_scores in peer_scores.get(pair["pair_id"], {}).items():
                for i4_dim, i3_dim in i4_map.items():
                    score = dim_scores.get(i4_dim)
                    if score is not None:
                        i4_peer[i3_dim].append(score)

        i4_means = {}
        for dim in dims:
            m, _ = mean_std(i4_peer[dim])
            i4_means[dim] = round(m, 3) if m is not None else None

        asymmetry = {}
        for dim in dims:
            if i3_strict[dim] is not None and i4_means[dim] is not None:
                asymmetry[dim] = round(i4_means[dim] - (10 - i3_strict[dim]), 3)
            else:
                asymmetry[dim] = None

        elp[model] = {
            "label":         label,
            "i3_strictness": i3_strict,
            "i4_peer_mean":  i4_means,
            "asymmetry":     asymmetry,
            "i1_coding":     None,
        }

    return elp


# =============================================================================
# INSTRUMENT 4 / ELP plots
# =============================================================================

def i4_elp_radar(elp: dict, models: list[str]) -> go.Figure:
    dims       = list(I3_DIMENSIONS.keys())
    dim_labels = list(I3_DIMENSIONS.values())

    axes        = ([f"Strict: {l}" for l in dim_labels] +
                   [f"Peer: {l}" for l in dim_labels])
    axes_closed = axes + [axes[0]]

    fig = go.Figure()

    for model in models:
        profile = elp.get(model, {})
        label   = MODEL_LABELS.get(model, model)
        color   = MODEL_COLORS.get(model, "#888888")

        strict_vals = [profile.get("i3_strictness", {}).get(d) or 0 for d in dims]
        peer_vals   = [profile.get("i4_peer_mean",  {}).get(d) or 0 for d in dims]
        vals_closed = strict_vals + peer_vals + [strict_vals[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=axes_closed,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5, dash=MODEL_DASH.get(model, "solid")),
            marker=dict(size=7, color=color, symbol=MODEL_SYMBOLS.get(model, "circle")),
            opacity=0.9,
            hovertemplate=f"<b>{label}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Epistemic Legitimacy Profile — All Models<br>"
                 "<sup>Left axes: I3 strictness (high = strict standard) · "
                 "Right axes: I4 peer leniency (high = lenient with peers)<br>"
                 "Gap between paired axes = normative self-positioning asymmetry</sup>",
            font=dict(family=SERIF, size=16, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(visible=True, range=[0, 10], color=TEXT_SEC,
                            gridcolor=BORDER, tickfont=dict(size=9, color=TEXT_SEC)),
            angularaxis=dict(color=TEXT_PRI, gridcolor=BORDER,
                             tickfont=dict(size=11, color=TEXT_PRI)),
        ),
        paper_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI, size=11), x=1.08, y=1.0),
        margin=dict(l=80, r=220, t=130, b=80),
        height=660, width=1000,
    )
    return fig


def i4_asymmetry_heatmap(elp: dict, models: list[str],
                          h3_stats: dict | None = None) -> go.Figure:
    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    dim_labels   = list(I3_DIMENSIONS.values())
    dims         = list(I3_DIMENSIONS.keys())

    z, text = [], []
    for model in models:
        row, row_text = [], []
        for dim in dims:
            val = elp.get(model, {}).get("asymmetry", {}).get(dim)
            row.append(val)
            row_text.append(f"{val:+.2f}" if val is not None else "N/A")
        z.append(row)
        text.append(row_text)

    subtitle = "Green = more lenient with peers than own I3 standards · Red = stricter with peers"
    if h3_stats:
        parts = []
        for dim in dims:
            if dim in h3_stats:
                d = h3_stats[dim]
                parts.append(f"{I3_DIMENSIONS[dim]}: d={d['cohens_d']:.2f}")
        if parts:
            subtitle += "<br>H3 Cohen's d vs zero: " + " | ".join(parts)

    fig = go.Figure(go.Heatmap(
        z=z, x=dim_labels, y=model_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=13, color=TEXT_PRI),
        colorscale=COLORSCALE_DIVERGE, zmid=0,
        colorbar=dict(title="Asymmetry Score", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Asymmetry: %{z:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"I4 Normative Self-Positioning Asymmetry<br><sup>{subtitle}</sup>",
            font=dict(family=SERIF, size=17, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=12), gridcolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT_PRI), autorange="reversed", gridcolor=BORDER),
        margin=dict(l=160, r=80, t=130, b=80),
        height=420, width=760,
    )
    return fig


def i4_peer_scores_heatmap(peer_scores: dict, pairs: list[dict]) -> go.Figure:
    dims       = list(I4_DIMENSIONS.keys())
    dim_labels = list(I4_DIMENSIONS.values())

    row_labels, z, text = [], [], []

    for pair in pairs:
        pair_id   = pair["pair_id"]
        evaluator = MODEL_LABELS.get(pair["evaluator"], pair["evaluator"])
        evaluatee = MODEL_LABELS.get(pair["evaluatee"], pair["evaluatee"])

        for q_id in ["I1_Q1", "I1_Q2", "I1_Q3", "I2_S1", "I2_S2", "I2_S3"]:
            q_label = I4_QUESTION_LABELS.get(q_id, q_id)
            dim_scores = peer_scores.get(pair_id, {}).get(q_id, {})

            row_labels.append(f"{evaluator}→{evaluatee}<br>{q_label}")
            row, row_text = [], []
            for dim in dims:
                val = dim_scores.get(dim)
                row.append(val)
                row_text.append(str(int(val)) if val is not None else "—")
            z.append(row)
            text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z, x=dim_labels, y=row_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=11, color=TEXT_PRI),
        colorscale=COLORSCALE_REDBLUE, zmin=1, zmax=10,
        colorbar=dict(title="Score", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="I4 Raw Peer Evaluation Scores<br>"
                 "<sup>Per pair × question × dimension</sup>",
            font=dict(family=SERIF, size=17, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=11), gridcolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT_PRI, size=9), autorange="reversed",
                   gridcolor=BORDER),
        margin=dict(l=260, r=60, t=100, b=60),
        height=max(500, len(row_labels) * 28 + 160),
        width=820,
    )
    return fig


# =============================================================================
# I1 self-reported source helpers and plots
# Works with the new {raw, parsed: {response, sources}} I1 format.
# Gracefully skips models whose data is still in the old plain-string format.
# =============================================================================

def get_i1_sources(data: dict, model: str) -> list[dict]:
    """
    Extract all self-reported source dicts from I1 parsed data for one model.
    Each source dict is augmented with '_question' and '_condition' keys.
    Returns [] if the model's data is still in the legacy string format.
    """
    sources = []
    for cond in CONDITIONS:
        for run_str, qs in data.get(model, {}).get(cond, {}).items():
            if not isinstance(qs, dict):
                continue
            for q_id, val in qs.items():
                if not isinstance(val, dict):
                    continue
                parsed = val.get("parsed") or {}
                for s in parsed.get("sources", []):
                    if isinstance(s, dict):
                        sources.append({**s, "_question": q_id, "_condition": cond})
    return sources


def i1_has_source_data(data: dict, models: list[str]) -> bool:
    """Return True if at least one model has I1 source citations in the new format."""
    return any(bool(get_i1_sources(data, m)) for m in models)


def i1_source_type_stacked_bar(data: dict, models: list[str]) -> go.Figure:
    """
    Grouped + stacked bar: % of each source type per model.
    One bar group per model, stacked by source type.
    """
    all_types = list(SOURCE_TYPE_LABELS.keys())
    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    fig = go.Figure()

    for stype in all_types:
        vals = []
        for model in models:
            sources = get_i1_sources(data, model)
            total   = len(sources) or 1
            count   = sum(1 for s in sources if s.get("type") == stype)
            vals.append(count / total * 100)

        fig.add_trace(go.Bar(
            name=SOURCE_TYPE_LABELS.get(stype, stype),
            x=model_labels,
            y=vals,
            marker=dict(
                color=SOURCE_TYPE_COLORS.get(stype, "#9ca3af"),
                pattern=dict(shape=SOURCE_TYPE_PATTERNS.get(stype, ""), fillmode="overlay",
                             fgcolor="rgba(0,0,0,0.35)", size=8),
            ),
            hovertemplate=(
                f"<b>{SOURCE_TYPE_LABELS.get(stype, stype)}</b><br>"
                "%{x}: %{y:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="I1 Self-Reported Source Types — Distribution per Model<br>"
                 "<sup>% of total citations in each source category</sup>",
            font=dict(family=SERIF, size=17, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        barmode="stack",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=11), gridcolor=BORDER),
        yaxis=dict(range=[0, 100], tickfont=dict(color=TEXT_PRI), gridcolor=BORDER,
                   title="% of Citations"),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI, size=10), x=1.01, y=1.0),
        margin=dict(l=70, r=200, t=100, b=80),
        height=520, width=1100,
    )
    return fig


def i1_source_legitimacy_proxy_heatmap(data: dict, models: list[str]) -> go.Figure:
    """
    Heatmap: rows=models, cols=legitimacy tiers 1-4.
    Legitimacy tier is derived from source type via SOURCE_TYPE_TO_TIER.
    """
    tier_labels  = ["Tier 1\nPrimary Legal", "Tier 2\nSecondary",
                    "Tier 3\nLow Authority", "Tier 4\nVague / Implicit"]
    model_labels = [MODEL_LABELS.get(m, m) for m in models]

    z, text = [], []
    for model in models:
        sources = get_i1_sources(data, model)
        total   = len(sources) or 1
        row, row_text = [], []
        for tier in [1, 2, 3, 4]:
            count = sum(
                1 for s in sources
                if SOURCE_TYPE_TO_TIER.get(s.get("type", "implicit_only"), 4) == tier
            )
            pct = count / total * 100
            row.append(pct)
            row_text.append(f"{pct:.1f}%")
        z.append(row)
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z, x=tier_labels, y=model_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=13, color=TEXT_PRI),
        colorscale=COLORSCALE_LEGITIMACY, zmin=0, zmax=100,
        colorbar=dict(title="% Citations", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="I1 Self-Reported Source Legitimacy (Proxy)<br>"
                 "<sup>Tier derived from source type — green = authoritative, red = vague/implicit</sup>",
            font=dict(family=SERIF, size=17, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=11), gridcolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT_PRI), autorange="reversed", gridcolor=BORDER),
        margin=dict(l=160, r=80, t=110, b=80),
        height=380, width=760,
    )
    return fig


def i1_source_overlap_heatmap(data: dict, models: list[str]) -> go.Figure:
    """
    Jaccard similarity matrix of I1 self-reported source-name sets across models.
    """
    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    source_sets  = {
        model: {
            (s.get("name") or "").lower().strip()
            for s in get_i1_sources(data, model)
            if (s.get("name") or "").strip()
        }
        for model in models
    }

    z, text = [], []
    for m1 in models:
        row, row_text = [], []
        for m2 in models:
            a, b = source_sets.get(m1, set()), source_sets.get(m2, set())
            jac  = len(a & b) / len(a | b) if (a or b) else 0.0
            row.append(jac)
            row_text.append(f"{jac:.3f}")
        z.append(row)
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z, x=model_labels, y=model_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=13, color=TEXT_PRI),
        colorscale=COLORSCALE_REDBLUE, zmin=0, zmax=1,
        colorbar=dict(title="Jaccard Sim.", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Jaccard: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        "I1 Self-Reported Citation Overlap — Cross-Model Jaccard Similarity<br>"
        "<sup>Shared source names across models — higher = shared epistemic tradition</sup>",
        height=520, width=700,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI))
    fig.update_xaxes(tickangle=-25, tickfont=dict(color=TEXT_PRI))
    return fig


def i1_jurisdiction_breakdown(data: dict, models: list[str]) -> go.Figure:
    """
    Grouped bar: % of citations from each jurisdiction per model.
    """
    jurisdictions = I5_JURISDICTIONS[:-1] + ["other"]   # keep same order as radar
    model_labels  = [MODEL_LABELS.get(m, m) for m in models]
    juris_colors  = {
        "EU": "#2176ae", "US": "#dc2626", "UN": "#16a34a",
        "UK": "#d97706", "unspecified": "#9ca3af", "other": "#7c3aed",
    }
    fig = go.Figure()

    for j in jurisdictions:
        vals = []
        for model in models:
            sources = get_i1_sources(data, model)
            total   = len(sources) or 1
            if j == "other":
                known = {"EU", "US", "UN", "UK", "unspecified"}
                count = sum(1 for s in sources
                            if (s.get("jurisdiction") or "").strip().upper() not in known
                            and (s.get("jurisdiction") or "").strip() != "")
            else:
                count = sum(
                    1 for s in sources
                    if (s.get("jurisdiction") or "").strip().upper() == j.upper()
                )
            vals.append(count / total * 100)

        fig.add_trace(go.Bar(
            name=j,
            x=model_labels,
            y=vals,
            marker=dict(
                color=juris_colors.get(j, "#9ca3af"),
                pattern=dict(shape=JURIS_PATTERNS.get(j, ""), fillmode="overlay",
                             fgcolor="rgba(0,0,0,0.35)", size=8),
            ),
            hovertemplate=f"<b>{j}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="I1 Self-Reported Source Jurisdictions — Geographic Basis per Model<br>"
                 "<sup>% of citations from each jurisdiction</sup>",
            font=dict(family=SERIF, size=17, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        barmode="group",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=11), gridcolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT_PRI), gridcolor=BORDER,
                   title="% of Citations"),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI, size=10)),
        margin=dict(l=70, r=40, t=100, b=80),
        height=480, width=1000,
    )
    return fig


# =============================================================================
# Hypothesis test result persistence
# =============================================================================

def save_hypothesis_results(stats_results: dict, out_dir: Path) -> None:
    """
    Save H1–H4 results to:
      <out_dir>/results.json  — machine-readable full output
      <out_dir>/summary.txt  — human-readable formatted summary
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump (filter out non-serialisable objects gracefully)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2, ensure_ascii=False, default=str)

    # Human-readable summary
    lines = [
        "HYPOTHESIS TEST RESULTS",
        "=" * 60,
        "",
    ]

    h1 = stats_results.get("h1", {})
    lines.append("H1 — Surface competence vs scenario divergence")
    lines.append("-" * 40)
    if h1:
        lines.append(f"  Within-model I1 similarity (mean):       {h1.get('within_mean', 'N/A'):.4f}")
        lines.append(f"  I1→I2 cross-instrument similarity (mean): {h1.get('cross_mean', 'N/A'):.4f}")
        lines.append(f"  Paired t-test:  t={h1.get('t', 'N/A'):.3f},  p={h1.get('p', 'N/A'):.4f}  (n={h1.get('n', '?')})")
        lines.append(f"  Supported:      {h1.get('supported', 'N/A')}")
    else:
        lines.append("  [not computed — missing data]")
    lines.append("")

    h2 = stats_results.get("h2", {})
    lines.append("H2 — CEO shift largest for enforceability")
    lines.append("-" * 40)
    dims = list(I3_DIMENSIONS.keys())
    for dim in dims:
        d = h2.get(dim, {})
        if d:
            lines.append(
                f"  {I3_DIMENSIONS[dim]:<22}  mean Δ={d.get('mean_delta', 0):+.3f}  "
                f"t={d.get('t', 0):.3f}  p={d.get('p', 0):.4f}"
            )
    if "friedman" in h2:
        lines.append(
            f"  Friedman χ²={h2['friedman'].get('stat', 0):.3f}  "
            f"p={h2['friedman'].get('p', 0):.4f}"
        )
    if not h2:
        lines.append("  [not computed — missing data]")
    lines.append("")

    h3 = stats_results.get("h3", {})
    lines.append("H3 — Asymmetry smallest for enforceability")
    lines.append("-" * 40)
    for dim in dims:
        d = h3.get(dim, {})
        if d:
            lines.append(
                f"  {I3_DIMENSIONS[dim]:<22}  mean={d.get('mean', 0):+.3f}  "
                f"Cohen's d={d.get('cohens_d', 0):.3f}  "
                f"t={d.get('t', 0):.3f}  p={d.get('p', 0):.4f}"
            )
    if not h3:
        lines.append("  [not computed — missing data]")
    lines.append("")

    h4 = stats_results.get("h4", {})
    lines.append("H4 — Epistemic basis inconsistency")
    lines.append("-" * 40)
    if h4.get("spearman_tier_enf"):
        sp = h4["spearman_tier_enf"]
        lines.append(f"  Spearman ρ (tier vs enforceability mean): ρ={sp['rho']:.3f}  p={sp['p']:.4f}")
    if h4.get("spearman_tier_var"):
        sp = h4["spearman_tier_var"]
        lines.append(f"  Spearman ρ (tier vs enforceability var):  ρ={sp['rho']:.3f}  p={sp['p']:.4f}")
    if h4.get("per_model"):
        lines.append("")
        lines.append("  Per-model source legitimacy summary:")
        for row in h4["per_model"]:
            if row:
                lines.append(
                    f"    {row.get('Model', '?'):<28}  "
                    f"mean tier={row.get('Mean Legitimacy Tier', '?')}  "
                    f"enf={row.get('Mean Enforceability', '?')}"
                )
    if not h4:
        lines.append("  [not computed — missing data]")

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {out_dir / 'results.json'}  +  {out_dir / 'summary.txt'}")


# =============================================================================
# HYPOTHESIS TESTS  (H1–H4)
# Prints results to stdout. Returns a dict for annotating plots.
# =============================================================================

def run_hypothesis_tests(
    i1_data: dict | None,
    i2_data: dict | None,
    i3_scores: dict | None,
    elp: dict | None,
    i5_data: dict | None,
    models: list[str],
    embed_model: SentenceTransformer | None,
) -> dict:
    """
    Run H1–H4 hypothesis tests on available data and print results.
    Returns stats_results dict used to annotate plots.
    """
    hr = "=" * 70
    print(f"\n{hr}")
    print("HYPOTHESIS TEST RESULTS")
    print(hr)

    dims = list(I3_DIMENSIONS.keys())
    results = {"h1": {}, "h2": {}, "h3": {}, "h4": {}}

    # ------------------------------------------------------------------
    # H1: Surface competence vs scenario divergence
    # Paired t-test: within-model I1 run similarity vs I1→I2 cross-instrument similarity
    # H1 predicts I1→I2 similarity significantly lower than I1→I1.
    # ------------------------------------------------------------------
    print("\n--- H1: Surface competence vs scenario divergence ---")
    if i1_data and i2_data and embed_model:
        dim_pairs = [("I1_Q1", "I2_S1"), ("I1_Q2", "I2_S2"), ("I1_Q3", "I2_S3")]
        within_sims, cross_sims = [], []

        for model in models:
            for i1_qid, i2_sid in dim_pairs:
                # Within-I1 pairwise similarity across 3 runs
                run_texts = [
                    _extract_text(i1_data.get(model, {}).get("baseline", {}).get(str(r), {}).get(i1_qid, ""))
                    for r in [1, 2, 3]
                ]
                run_texts = [t for t in run_texts if t]
                if len(run_texts) < 2:
                    continue
                embs = embed_model.encode(run_texts, show_progress_bar=False)
                pair_sims = [cosine_sim(embs[a], embs[b])
                             for a, b in combinations(range(len(embs)), 2)]
                w_sim = float(np.mean(pair_sims))

                # I1→I2 cross-instrument similarity
                i1_text = _extract_text(i1_data.get(model, {}).get("baseline", {}).get("1", {}).get(i1_qid, ""))
                i2_text = _extract_text(i2_data.get(model, {}).get("baseline", {}).get("1", {}).get(i2_sid, ""))
                if not i1_text or not i2_text:
                    continue
                e1 = embed_model.encode([i1_text], show_progress_bar=False)[0]
                e2 = embed_model.encode([i2_text], show_progress_bar=False)[0]
                c_sim = cosine_sim(e1, e2)

                within_sims.append(w_sim)
                cross_sims.append(c_sim)

        if len(within_sims) >= 2 and len(within_sims) == len(cross_sims):
            t_stat, p_val = stats.ttest_rel(within_sims, cross_sims)
            w_mean = float(np.mean(within_sims))
            c_mean = float(np.mean(cross_sims))
            supported = w_mean > c_mean and p_val < 0.05
            print(f"  Within-model I1 mean similarity:       {w_mean:.4f}")
            print(f"  I1→I2 cross-instrument mean similarity: {c_mean:.4f}")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
            print(f"  H1 supported (I1→I2 < I1→I1, p<0.05): {supported}")
            results["h1"] = {
                "within_mean": w_mean, "cross_mean": c_mean,
                "t": float(t_stat), "p": float(p_val), "supported": supported,
                "n": len(within_sims),
            }
        else:
            print("  [skip] Insufficient paired observations for t-test.")
    else:
        print("  [skip] Requires i1_data, i2_data, and embedding model.")

    # ------------------------------------------------------------------
    # H2: CEO shift largest for enforceability
    # One-sample t-test per dimension: mean delta ≠ 0
    # Friedman test comparing deltas across dimensions
    # H2 predicts enforceability delta most negative.
    # ------------------------------------------------------------------
    print("\n--- H2: CEO shift largest for enforceability ---")
    if i3_scores:
        h2_dim_deltas = {dim: [] for dim in dims}

        for model in models:
            for dim in dims:
                model_dim_deltas = []
                for s_id in I3_SCENARIOS:
                    b_vals = i3_scores.get(model, {}).get("baseline", {}).get(s_id, {}).get(dim, [])
                    c_vals = i3_scores.get(model, {}).get("ceo",      {}).get(s_id, {}).get(dim, [])
                    b_m, _ = mean_std(b_vals)
                    c_m, _ = mean_std(c_vals)
                    if b_m is not None and c_m is not None:
                        model_dim_deltas.append(c_m - b_m)
                h2_dim_deltas[dim].append(
                    float(np.mean(model_dim_deltas)) if model_dim_deltas else 0.0
                )

        h2_results = {}
        for dim in dims:
            arr = np.array(h2_dim_deltas[dim])
            if len(arr) >= 2:
                t, p = stats.ttest_1samp(arr, 0)
                h2_results[dim] = {
                    "mean_delta": float(arr.mean()),
                    "t": float(t), "p": float(p),
                }
                print(f"  {I3_DIMENSIONS[dim]}: mean Δ={arr.mean():+.3f}, "
                      f"t={t:.3f}, p={p:.4f}")

        # Friedman test
        arrays = [np.array(h2_dim_deltas[d]) for d in dims]
        if all(len(a) >= 3 for a in arrays) and len({len(a) for a in arrays}) == 1:
            fstat, fp = stats.friedmanchisquare(*arrays)
            h2_results["friedman"] = {"stat": float(fstat), "p": float(fp)}
            print(f"  Friedman test across dimensions: χ²={fstat:.3f}, p={fp:.4f}")

        # H2 check: enforceability most negative
        enf_delta = h2_results.get("enforceability", {}).get("mean_delta")
        other_deltas = [h2_results.get(d, {}).get("mean_delta") for d in dims
                        if d != "enforceability" and h2_results.get(d)]
        if enf_delta is not None and other_deltas:
            supported = all(enf_delta <= o for o in other_deltas)
            print(f"  H2 enforceability most negative delta: {supported}")

        results["h2"] = h2_results
    else:
        print("  [skip] Requires i3_scores.")

    # ------------------------------------------------------------------
    # H3: Asymmetry smallest for enforceability
    # One-sample t-test per dimension: asymmetry ≠ 0
    # Effect size: Cohen's d vs zero
    # H3 predicts enforceability asymmetry smallest in absolute value.
    # ------------------------------------------------------------------
    print("\n--- H3: Asymmetry smallest for enforceability ---")
    if elp:
        h3_results = {}
        for dim in dims:
            asym_vals = [elp.get(m, {}).get("asymmetry", {}).get(dim) for m in models]
            asym_vals = [v for v in asym_vals if v is not None]
            if len(asym_vals) >= 2:
                arr = np.array(asym_vals)
                t, p = stats.ttest_1samp(arr, 0)
                d = float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0
                h3_results[dim] = {
                    "mean": float(arr.mean()), "cohens_d": d,
                    "t": float(t), "p": float(p),
                }
                print(f"  {I3_DIMENSIONS[dim]}: mean={arr.mean():+.3f}, "
                      f"Cohen's d={d:.3f}, t={t:.3f}, p={p:.4f}")

        abs_asym = {d: abs(h3_results[d]["mean"]) for d in h3_results}
        if abs_asym:
            smallest = min(abs_asym, key=abs_asym.get)
            supported = smallest == "enforceability"
            print(f"  Smallest absolute asymmetry: {I3_DIMENSIONS.get(smallest, smallest)}")
            print(f"  H3 enforceability asymmetry smallest: {supported}")

        results["h3"] = h3_results
    else:
        print("  [skip] Requires ELP (i4_data + i3_scores + pairs).")

    # ------------------------------------------------------------------
    # H4: Epistemic basis inconsistency
    # Spearman correlation: mean source legitimacy tier vs I3 enforceability
    # H4 predicts lower-legitimacy sources → less consistent normative positions
    # ------------------------------------------------------------------
    print("\n--- H4: Epistemic basis inconsistency ---")
    if i5_data and i3_scores:
        per_model_rows = []

        for model in models:
            all_sources = []
            for cond in CONDITIONS:
                for run_str, questions in i5_data.get(model, {}).get(cond, {}).items():
                    if not isinstance(questions, dict):
                        continue
                    for q_id, q_data in questions.items():
                        if not isinstance(q_data, dict):
                            continue
                        parsed = q_data.get("parsed") or {}
                        all_sources.extend(
                            s for s in parsed.get("sources", [])
                            if isinstance(s, dict)
                        )

            if not all_sources:
                per_model_rows.append(None)
                continue

            tiers      = [s.get("legitimacy_tier", 4) for s in all_sources]
            mean_tier  = float(np.mean(tiers))
            pct_low    = sum(1 for t in tiers if t >= 3) / len(tiers) * 100
            pct_unver  = sum(1 for s in all_sources
                             if not s.get("verifiable", True)) / len(all_sources) * 100
            n_juris    = len({s.get("jurisdiction", "unspecified") for s in all_sources})

            enf_scores = []
            for s_id in I3_SCENARIOS:
                enf_scores.extend(
                    i3_scores.get(model, {}).get("baseline", {}).get(s_id, {})
                    .get("enforceability", [])
                )
            mean_enf = float(np.mean(enf_scores)) if enf_scores else None
            var_enf  = float(np.var(enf_scores))  if enf_scores else None

            per_model_rows.append({
                "Model":                 MODEL_LABELS.get(model, model),
                "Mean Legitimacy Tier":  round(mean_tier, 3),
                "% Tier 3–4 Sources":   round(pct_low, 1),
                "% Unverifiable":        round(pct_unver, 1),
                "Unique Jurisdictions":  n_juris,
                "Mean Enforceability":   round(mean_enf, 3) if mean_enf else None,
                "Enforceability Var.":   round(var_enf, 3)  if var_enf  else None,
            })

        valid = [r for r in per_model_rows if r and r["Mean Enforceability"] is not None]
        if valid:
            df = pd.DataFrame(valid)
            print(df.to_string(index=False))

        if len(valid) >= 3:
            tier_arr = np.array([r["Mean Legitimacy Tier"] for r in valid])
            enf_arr  = np.array([r["Mean Enforceability"]  for r in valid])
            rho, p   = stats.spearmanr(tier_arr, enf_arr)
            results["h4"]["spearman_tier_enf"] = {"rho": float(rho), "p": float(p)}
            print(f"\n  Spearman ρ (mean tier vs enforceability mean): ρ={rho:.3f}, p={p:.4f}")

            valid_var = [r for r in valid if r["Enforceability Var."] is not None]
            if len(valid_var) >= 3:
                tv = np.array([r["Mean Legitimacy Tier"]  for r in valid_var])
                vv = np.array([r["Enforceability Var."] for r in valid_var])
                rho_v, p_v = stats.spearmanr(tv, vv)
                results["h4"]["spearman_tier_var"] = {"rho": float(rho_v), "p": float(p_v)}
                print(f"  Spearman ρ (mean tier vs enforceability variance): ρ={rho_v:.3f}, p={p_v:.4f}")

            supported = rho > 0 and p < 0.05
            print(f"  H4 supported (lower-quality sources → higher variance): {supported}")

        results["h4"]["per_model"] = per_model_rows
    else:
        print("  [skip] Requires i5_data and i3_scores.")

    print(f"\n{hr}\n")
    return results


# =============================================================================
# INSTRUMENT 5 helpers
# =============================================================================

def get_i5_sources(data: dict, model: str,
                   condition: str | None = None) -> list[dict]:
    """Return a flat list of all source dicts for a model (optionally filtered by condition)."""
    sources = []
    conds = [condition] if condition else list(data.get(model, {}).keys())
    for cond in conds:
        for run_str, questions in data.get(model, {}).get(cond, {}).items():
            if not isinstance(questions, dict):
                continue
            for q_id, q_data in questions.items():
                if not isinstance(q_data, dict):
                    continue
                parsed = q_data.get("parsed") or {}
                sources.extend(
                    s for s in parsed.get("sources", [])
                    if isinstance(s, dict)
                )
    return sources


# =============================================================================
# INSTRUMENT 5 plots
# =============================================================================

def i5_source_legitimacy_heatmap(data: dict, models: list[str]) -> go.Figure:
    """
    Heatmap: rows=models, cols=legitimacy tiers 1-4, values=% of citations.
    Colorscale: green (tier 1) to red (tier 4).
    """
    tier_labels = [
        "Tier 1\nPrimary Legal", "Tier 2\nSecondary",
        "Tier 3\nUnverifiable", "Tier 4\nVague/Fabricated",
    ]
    model_labels = [MODEL_LABELS.get(m, m) for m in models]

    z, text = [], []
    for model in models:
        all_sources = get_i5_sources(data, model)
        total = len(all_sources) or 1
        row, row_text = [], []
        for tier in [1, 2, 3, 4]:
            count = sum(1 for s in all_sources if s.get("legitimacy_tier") == tier)
            pct = count / total * 100
            row.append(pct)
            row_text.append(f"{pct:.1f}%")
        z.append(row)
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z, x=tier_labels, y=model_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=13, color=TEXT_PRI),
        colorscale=COLORSCALE_LEGITIMACY, zmin=0, zmax=100,
        colorbar=dict(title="% Citations", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="I5 Source Legitimacy Profile<br>"
                 "<sup>% of citations in each legitimacy tier per model — green = authoritative</sup>",
            font=dict(family=SERIF, size=17, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=11), gridcolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT_PRI), autorange="reversed", gridcolor=BORDER),
        margin=dict(l=160, r=80, t=110, b=80),
        height=380, width=760,
    )
    return fig


def i5_source_type_sankey(data: dict, models: list[str]) -> go.Figure:
    """
    Sankey: model → source type → jurisdiction.
    Uses LLM deduplication on source types and jurisdictions.
    """
    print("    Extracting I5 source type flows...")

    flow_records = []
    for model in models:
        label = MODEL_LABELS.get(model, model)
        sources = get_i5_sources(data, model)
        for s in sources:
            stype = s.get("type", "unverifiable") or "unverifiable"
            juris = (s.get("jurisdiction") or "unspecified").strip() or "unspecified"
            flow_records.append((label, stype, juris))

    if not flow_records:
        fig = go.Figure()
        fig.update_layout(**base_layout("I5 Source Type Sankey — No data"))
        return fig

    all_stypes = list({r[1] for r in flow_records})
    all_juris  = list({r[2] for r in flow_records})

    if all_stypes:
        print("    Clustering source-type synonyms via LLM...")
        stype_map = get_sankey_label_mapping(all_stypes, "i5_source_types")
    else:
        stype_map = {}
    if all_juris:
        print("    Clustering jurisdiction synonyms via LLM...")
        juris_map = get_sankey_label_mapping(all_juris, "i5_jurisdictions")
    else:
        juris_map = {}

    flow_records = [(src, stype_map.get(st, st), juris_map.get(j, j))
                    for src, st, j in flow_records]

    source_nodes = list(dict.fromkeys([r[0] for r in flow_records]))
    type_nodes   = list(dict.fromkeys([r[1] for r in flow_records]))
    juris_nodes  = list(dict.fromkeys([r[2] for r in flow_records]))

    nodes    = source_nodes + type_nodes + juris_nodes
    node_idx = {n: i for i, n in enumerate(nodes)}

    flow1: dict[tuple, int] = defaultdict(int)
    flow2: dict[tuple, int] = defaultdict(int)
    for src, st, j in flow_records:
        flow1[(src, st)] += 1
        flow2[(st, j)]   += 1

    link_src, link_tgt, link_val, link_color, link_label = [], [], [], [], []

    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")
        rgba  = hex_to_rgba(color)
        for st in type_nodes:
            w = flow1.get((label, st), 0)
            if w:
                link_src.append(node_idx[label])
                link_tgt.append(node_idx[st])
                link_val.append(w)
                link_color.append(rgba)
                link_label.append(f"{label} → {st}")

    for st in type_nodes:
        for j in juris_nodes:
            w = flow2.get((st, j), 0)
            if w:
                link_src.append(node_idx[st])
                link_tgt.append(node_idx[j])
                link_val.append(w)
                link_color.append("rgba(160,168,176,0.6)")
                link_label.append(f"{st} → {j}")

    node_colors = []
    for n in nodes:
        if n in source_nodes:
            matched = "#9ca3af"
            for model in models:
                if MODEL_LABELS.get(model, model) == n:
                    matched = MODEL_COLORS.get(model, "#9ca3af")
                    break
            node_colors.append(matched)
        elif n in type_nodes:
            node_colors.append("#2176ae")
        else:
            node_colors.append("#7c3aed")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=18,
            line=dict(color=BORDER, width=0.5),
            label=nodes, color=node_colors,
            hovertemplate="<b>%{label}</b><br>Flow: %{value}<extra></extra>",
        ),
        link=dict(
            source=link_src, target=link_tgt, value=link_val,
            color=link_color, label=link_label,
            hovertemplate="%{label}<br>Count: %{value}<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=dict(
            text="I5 Epistemic Sources — Model → Source Type → Jurisdiction<br>"
                 "<sup>LLM-extracted citations, all conditions and runs combined</sup>",
            font=dict(family=SERIF, size=16, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_PRI, size=11),
        margin=dict(l=40, r=40, t=100, b=40),
        height=700, width=1400,
    )
    return fig


def i5_citation_overlap_heatmap(data: dict, models: list[str]) -> go.Figure:
    """
    Jaccard similarity matrix of source-name sets across models.
    """
    model_labels = [MODEL_LABELS.get(m, m) for m in models]

    source_sets = {
        model: {
            (s.get("name") or "").lower().strip()
            for s in get_i5_sources(data, model)
            if (s.get("name") or "").strip()
        }
        for model in models
    }

    z, text = [], []
    for m1 in models:
        row, row_text = [], []
        for m2 in models:
            a, b = source_sets.get(m1, set()), source_sets.get(m2, set())
            jac  = len(a & b) / len(a | b) if (a or b) else 0.0
            row.append(jac)
            row_text.append(f"{jac:.3f}")
        z.append(row)
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z, x=model_labels, y=model_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=13, color=TEXT_PRI),
        colorscale=COLORSCALE_REDBLUE, zmin=0, zmax=1,
        colorbar=dict(title="Jaccard Sim.", tickfont=dict(color=TEXT_SEC)),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Jaccard: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        "I5 Citation Overlap — Cross-Model Jaccard Similarity<br>"
        "<sup>Similarity of source-name sets — higher = shared epistemic tradition</sup>",
        height=520, width=700,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_PRI))
    fig.update_xaxes(tickangle=-25, tickfont=dict(color=TEXT_PRI))
    return fig


def i5_jurisdiction_radar(data: dict, models: list[str],
                           h4_stats: dict | None = None) -> go.Figure:
    """
    Radar chart: one trace per model, axes = major jurisdictions.
    Values = % of citations from that jurisdiction.
    """
    juris_axes  = I5_JURISDICTIONS[:]
    axes_closed = juris_axes + [juris_axes[0]]

    subtitle = "% of citations from each jurisdiction — shows geographic bias in epistemic basis"
    if h4_stats and h4_stats.get("spearman_tier_enf"):
        rho = h4_stats["spearman_tier_enf"]["rho"]
        p   = h4_stats["spearman_tier_enf"]["p"]
        subtitle += f"<br>H4 Spearman ρ (legitimacy tier vs enforceability): ρ={rho:.3f}, p={p:.4f}"

    fig = go.Figure()

    for model in models:
        label   = MODEL_LABELS.get(model, model)
        color   = MODEL_COLORS.get(model, "#888888")
        sources = get_i5_sources(data, model)
        total   = len(sources) or 1

        vals = []
        accounted = 0
        for j in juris_axes[:-1]:  # all except "other"
            count = sum(
                1 for s in sources
                if (s.get("jurisdiction") or "").strip().upper() == j.upper()
            )
            pct = count / total * 100
            vals.append(pct)
            accounted += pct
        vals.append(max(0.0, 100.0 - accounted))  # "other"

        vals_closed = vals + [vals[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=axes_closed,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2, dash=MODEL_DASH.get(model, "solid")),
            marker=dict(size=6, color=color, symbol=MODEL_SYMBOLS.get(model, "circle")),
            opacity=0.85,
            hovertemplate=f"<b>{label}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"I5 Jurisdiction Radar — Geographic Bias in Epistemic Basis<br>"
                 f"<sup>{subtitle}</sup>",
            font=dict(family=SERIF, size=16, color=TEXT_PRI),
            x=0.5, xanchor="center",
        ),
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(visible=True, color=TEXT_SEC, gridcolor=BORDER,
                            tickfont=dict(size=9, color=TEXT_SEC)),
            angularaxis=dict(color=TEXT_PRI, gridcolor=BORDER,
                             tickfont=dict(size=12, color=TEXT_PRI)),
        ),
        paper_bgcolor=BG,
        font=dict(family=MONO, color=TEXT_SEC),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRI, size=10), x=1.05, y=1.0),
        margin=dict(l=80, r=200, t=120, b=80),
        height=620, width=1000,
    )
    return fig


# =============================================================================
# Main
# =============================================================================

def _try_load(instrument_id: str) -> dict | None:
    path = RAW_DIR / f"{instrument_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    _load_sankey_cache()

    # -------------------------------------------------------------------------
    # Pre-load all available data
    # -------------------------------------------------------------------------
    i1_data = _try_load("instrument_1")
    i2_data = _try_load("instrument_2")
    i3_data = _try_load("instrument_3")
    i4_data = _try_load("instrument_4")
    i5_data = _try_load("instrument_5")

    # Determine model list from first available dataset
    models = None
    for _d in [i1_data, i2_data, i3_data, i5_data]:
        if _d:
            models = list(_d.keys())
            break
    if models is None:
        print("[error] No instrument data found in data/raw/. Exiting.")
        import sys; sys.exit(1)

    # Load peer eval pairs
    pairs = []
    if PEER_EVAL_FILE.exists():
        with open(PEER_EVAL_FILE, "r", encoding="utf-8") as f:
            pairs = json.load(f)["pairs"]

    # Pre-compute I3 scores and ELP
    i3_scores  = extract_i3_scores(i3_data, models) if i3_data else None
    peer_scores = None
    elp         = None
    if i4_data and i3_scores and pairs:
        print("Extracting I4 peer scores...")
        peer_scores = extract_i4_scores(i4_data, pairs)
        print("Building ELP profiles...")
        elp = build_elp(i3_scores, peer_scores, pairs, models)

    # Load embedding model (needed for I1 plots and H1)
    embed_model = None
    if i1_data:
        print("Loading embedding model...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # -------------------------------------------------------------------------
    # Hypothesis tests — printed to stdout before any plotting
    # -------------------------------------------------------------------------
    stats_results = run_hypothesis_tests(
        i1_data, i2_data, i3_scores, elp, i5_data, models, embed_model
    )

    # Save hypothesis results to results/hypothesis_tests/
    print("Saving hypothesis test results...")
    save_hypothesis_results(stats_results, RESULTS_DIR / "hypothesis_tests")

    # -------------------------------------------------------------------------
    # Instrument 1
    # -------------------------------------------------------------------------
    if "instrument_1" in ACTIVE_INSTRUMENTS and i1_data:
        out = RESULTS_DIR / "instrument_1"
        print("\nBuilding I1 word frequency heatmaps...")
        for q_id, q_label in I1_QUESTIONS.items():
            slug = q_label.lower().replace(" ", "_")
            save_fig(i1_wordfreq_heatmap(i1_data, q_id, q_label, models),
                     out / f"wordfreq_{slug}")

        print("Building I1 cross-model similarity...")
        save_fig(i1_cross_model_similarity(i1_data, models, embed_model),
                 out / "similarity_cross_model")

        print("Building I1 baseline vs CEO shift...")
        fig_bvc = i1_baseline_vs_ceo(i1_data, models, embed_model)
        # Annotate with H1 stats
        if stats_results.get("h1"):
            h1 = stats_results["h1"]
            ann = (f"H1 paired t-test (n={h1['n']}): t={h1['t']:.3f}, p={h1['p']:.4f} | "
                   f"within-I1={h1['within_mean']:.3f}, I1→I2={h1['cross_mean']:.3f} | "
                   f"supported={h1['supported']}")
            current_title = fig_bvc.layout.title.text or ""
            fig_bvc.update_layout(
                title_text=current_title + f"<br><sup style='font-size:11px'>{ann}</sup>"
            )
        save_fig(fig_bvc, out / "similarity_baseline_vs_ceo")
    elif "instrument_1" in ACTIVE_INSTRUMENTS:
        print("[skip] instrument_1.json not found")

    # -------------------------------------------------------------------------
    # Instrument 2
    # -------------------------------------------------------------------------
    if "instrument_2" in ACTIVE_INSTRUMENTS and i2_data:
        out = RESULTS_DIR / "instrument_2"
        print("\nBuilding I2 S1 word frequency — cross-model...")
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
    elif "instrument_2" in ACTIVE_INSTRUMENTS:
        print("[skip] instrument_2.json not found")

    # -------------------------------------------------------------------------
    # Instrument 3
    # -------------------------------------------------------------------------
    if "instrument_3" in ACTIVE_INSTRUMENTS and i3_data and i3_scores:
        out = RESULTS_DIR / "instrument_3"
        print("\nBuilding I3 grouped bar charts...")
        for dim, dim_label in I3_DIMENSIONS.items():
            save_fig(i3_grouped_bar(i3_scores, models, dim, dim_label),
                     out / f"scores_grouped_bar_{dim}")

        print("Building I3 heatmaps...")
        for dim, dim_label in I3_DIMENSIONS.items():
            save_fig(i3_heatmap(i3_scores, models, dim, dim_label),
                     out / f"scores_heatmap_{dim}")

        print("Building I3 radar charts...")
        for s_id, s_label in I3_SCENARIOS.items():
            save_fig(i3_radar(i3_scores, models, s_id, s_label),
                     out / f"scores_radar_{s_id.lower()}")

        print("Building I3 condition delta heatmap...")
        save_fig(i3_delta_heatmap(i3_scores, models, h2_stats=stats_results.get("h2")),
                 out / "condition_delta_heatmap")

        print("Building I3 condition side-by-side bars...")
        save_fig(i3_condition_bars(i3_scores, models),
                 out / "condition_side_by_side")
    elif "instrument_3" in ACTIVE_INSTRUMENTS:
        print("[skip] instrument_3.json not found")

    # -------------------------------------------------------------------------
    # Instrument 4 + ELP
    # -------------------------------------------------------------------------
    if "instrument_4" in ACTIVE_INSTRUMENTS and i4_data and elp:
        out = RESULTS_DIR / "instrument_4"
        out.mkdir(parents=True, exist_ok=True)

        # Save ELP JSON
        elp_out = out / "elp_profiles.json"
        with open(elp_out, "w", encoding="utf-8") as f:
            json.dump(elp, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved ELP profiles: {elp_out}")

        print("Building ELP radar (all models)...")
        save_fig(i4_elp_radar(elp, models),
                 out / "elp_radar_all_models")

        print("Building I4 asymmetry heatmap...")
        save_fig(i4_asymmetry_heatmap(elp, models, h3_stats=stats_results.get("h3")),
                 out / "asymmetry_heatmap")

        print("Building I4 raw peer scores heatmap...")
        save_fig(i4_peer_scores_heatmap(peer_scores, pairs),
                 out / "peer_scores_heatmap")
    elif "instrument_4" in ACTIVE_INSTRUMENTS:
        print("[skip] instrument_4.json not found or ELP could not be computed")

    # -------------------------------------------------------------------------
    # Instrument 5 — source legitimacy and epistemic basis analysis
    #
    # Two data sources, both written to results/instrument_5/:
    #
    #   A. I5 extracted sources  (instrument_5.json — requires prior I5 collection run)
    #      Richer: legitimacy_tier, verifiable flag, quote, all models × conditions × runs
    #
    #   B. I1 self-reported sources  (instrument_1.json new format — available immediately)
    #      Simpler: name / type / jurisdiction self-declared by each model
    #
    # Plots are produced from whichever source(s) are available.
    # -------------------------------------------------------------------------
    if "instrument_5" in ACTIVE_INSTRUMENTS:
        out = RESULTS_DIR / "instrument_5"

        # ---- A. I5 extracted sources ----
        if i5_data:
            print("\n[I5-A] Building extracted source legitimacy heatmap...")
            save_fig(i5_source_legitimacy_heatmap(i5_data, models),
                     out / "extracted_source_legitimacy_heatmap")

            print("[I5-A] Building extracted source type Sankey...")
            save_fig(i5_source_type_sankey(i5_data, models),
                     out / "extracted_source_type_sankey")

            print("[I5-A] Building extracted citation overlap heatmap...")
            save_fig(i5_citation_overlap_heatmap(i5_data, models),
                     out / "extracted_citation_overlap_heatmap")

            print("[I5-A] Building extracted jurisdiction radar...")
            save_fig(i5_jurisdiction_radar(i5_data, models,
                                           h4_stats=stats_results.get("h4")),
                     out / "extracted_jurisdiction_radar")
        else:
            print("\n[I5-A] instrument_5.json not found — skipping extracted-source plots.")
            print("       Run collect_llm_responses.py with instrument_5 active to generate it.")

        # ---- B. I1 self-reported sources ----
        if i1_data and i1_has_source_data(i1_data, models):
            print("\n[I5-B] Building I1 self-reported source type distribution...")
            save_fig(i1_source_type_stacked_bar(i1_data, models),
                     out / "i1_source_type_distribution")

            print("[I5-B] Building I1 self-reported source legitimacy proxy heatmap...")
            save_fig(i1_source_legitimacy_proxy_heatmap(i1_data, models),
                     out / "i1_source_legitimacy_proxy")

            print("[I5-B] Building I1 self-reported citation overlap heatmap...")
            save_fig(i1_source_overlap_heatmap(i1_data, models),
                     out / "i1_citation_overlap")

            print("[I5-B] Building I1 self-reported jurisdiction breakdown...")
            save_fig(i1_jurisdiction_breakdown(i1_data, models),
                     out / "i1_jurisdiction_breakdown")
        else:
            print("\n[I5-B] No I1 source citation data found.")
            print("       Re-run collect_llm_responses.py with updated I1 prompts to populate.")

        if not i5_data and not (i1_data and i1_has_source_data(i1_data, models)):
            print("[skip] No source data available for Instrument 5 analysis.")

    _save_sankey_cache()
    print("\nAll plots complete.")
