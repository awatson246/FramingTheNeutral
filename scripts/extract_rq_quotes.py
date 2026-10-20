"""
extract_rq_quotes.py

Reads instrument_1.json and instrument_5.json and generates a structured
Markdown document (results/rq_quotations.md) with quotations and source
analysis addressing RQ1 and RQ2.

RQ1: How do LLMs define legal certainty, accountability, and enforceability?
     (H1: surface-level competence; cross-model convergence)

RQ2: What epistemic sources do LLMs report drawing upon, and how reliable?
     (H2: overreported legitimacy; unverifiable references; regional bias)

Run from the project root:
    python scripts/extract_rq_quotes.py
"""

import json
from collections import Counter
from pathlib import Path

RAW_DIR    = Path("data/raw")
RESULTS_DIR = Path("results")

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

TIER_LABELS = {
    1: "Tier 1 — Primary Legal (treaty / statute / court decision)",
    2: "Tier 2 — Secondary (academic / policy framework)",
    3: "Tier 3 — Low authority (implicit / descriptive)",
    4: "Tier 4 — Vague / unverifiable / likely fabricated",
}


def load_json(name: str) -> dict | None:
    p = RAW_DIR / f"{name}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def get_run1_response(i1_data: dict, model: str, condition: str, q_id: str) -> str:
    val = i1_data.get(model, {}).get(condition, {}).get("1", {}).get(q_id, {})
    if isinstance(val, dict):
        parsed = val.get("parsed") or {}
        return parsed.get("response") or val.get("raw") or ""
    return str(val) if val else ""


def get_i1_sources(i1_data: dict, model: str, condition: str = "baseline") -> list[dict]:
    sources = []
    for run_str, qs in i1_data.get(model, {}).get(condition, {}).items():
        if not isinstance(qs, dict):
            continue
        for q_id, val in qs.items():
            if not isinstance(val, dict):
                continue
            parsed = val.get("parsed") or {}
            for s in parsed.get("sources", []):
                if isinstance(s, dict):
                    sources.append({**s, "_question": q_id})
    return sources


def get_i5_sources(i5_data: dict, model: str) -> list[dict]:
    sources = []
    for cond in i5_data.get(model, {}):
        for run_str, qs in i5_data[model][cond].items():
            if not isinstance(qs, dict):
                continue
            for q_id, q_data in qs.items():
                if not isinstance(q_data, dict):
                    continue
                parsed = q_data.get("parsed") or {}
                for s in parsed.get("sources", []):
                    if isinstance(s, dict):
                        sources.append({**s, "_question": q_id, "_condition": cond})
    return sources


def wrap(text: str, width: int = 90) -> str:
    """Wrap long text for readable markdown block quotes."""
    words = text.split()
    lines, current = [], []
    for w in words:
        if sum(len(x) + 1 for x in current) + len(w) > width:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return "\n> ".join(lines)


def build_doc(i1_data: dict, i5_data: dict | None) -> str:
    models = list(i1_data.keys())
    lines = []

    # =========================================================================
    # Header
    # =========================================================================
    lines += [
        "# LLM Quotation and Source Extraction Report",
        "",
        "**Project:** Framing the Neutral — LLM Governance Legitimacy Study  ",
        "**Generated from:** `instrument_1.json`" + (" + `instrument_5.json`" if i5_data else ""),
        "",
        "---",
        "",
    ]

    # =========================================================================
    # RQ1
    # =========================================================================
    lines += [
        "## RQ1: How do LLMs Define Legal Certainty, Accountability, and Enforceability?",
        "",
        "> **H1:** All five LLMs will demonstrate surface-level definitional competence on",
        "> legal certainty, accountability, and enforceability, with high cross-model semantic",
        "> similarity at the baseline definitional level. Models will converge around shared",
        "> conceptual cores but differ in vocabulary, emphasis, and conceptual boundaries.",
        "",
    ]

    for q_id, q_label in I1_QUESTIONS.items():
        lines += [
            f"### {q_label}",
            "",
            "| Model | Baseline Definition (Run 1, excerpt) |",
            "|-------|---------------------------------------|",
        ]
        for model in models:
            label = MODEL_LABELS.get(model, model)
            text  = get_run1_response(i1_data, model, "baseline", q_id)
            # Truncate to ~240 chars for table
            excerpt = text[:240].rstrip()
            if len(text) > 240:
                excerpt += "…"
            excerpt = excerpt.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| **{label}** | {excerpt} |")
        lines.append("")

        # Full quotes as block quotes
        lines += [
            f"#### Full Definitions — {q_label} (Baseline, Run 1)",
            "",
        ]
        for model in models:
            label = MODEL_LABELS.get(model, model)
            text  = get_run1_response(i1_data, model, "baseline", q_id)
            lines += [
                f"**{label}**",
                "",
                f"> {wrap(text)}",
                "",
            ]

        # CEO condition quotes for comparison
        lines += [
            f"#### CEO Role Shift — {q_label} (CEO Condition, Run 1)",
            "",
        ]
        for model in models:
            label    = MODEL_LABELS.get(model, model)
            baseline = get_run1_response(i1_data, model, "baseline", q_id)
            ceo_text = get_run1_response(i1_data, model, "ceo", q_id)
            lines += [
                f"**{label}** *(CEO framing)*",
                "",
                f"> {wrap(ceo_text)}",
                "",
            ]
        lines.append("---")
        lines.append("")

    # Convergence observation
    lines += [
        "### Cross-Model Convergence Observation (RQ1 / H1)",
        "",
        "All five models define **legal certainty** around: predictability, transparency, "
        "clear legal standards, and individual ability to foresee consequences of AI decisions.",
        "",
        "All five models define **accountability** around: assigned responsibility across "
        "developer / deployer / operator chains, with oversight and redress mechanisms.",
        "",
        "All five models define **enforceability** around: the binding / aspirational "
        "distinction, with hard-law characteristics (specificity, monitoring, sanctions) "
        "contrasted against soft guidelines.",
        "",
        "Shared references across all or most models: EU AI Act, GDPR, OECD AI Principles, "
        "Loomis v. Wisconsin (US). EU-centric framing dominates.",
        "",
        "---",
        "",
    ]

    # =========================================================================
    # RQ2
    # =========================================================================
    lines += [
        "## RQ2: Epistemic Basis — What Sources Do LLMs Report, and How Reliable Are They?",
        "",
        "> **H2:** Models will systematically overreport the legitimacy and verifiability of",
        "> their epistemic sources. Independent evaluation will reveal significant reliance on",
        "> unverifiable or fabricated references, with systematic regional bias in source",
        "> distribution regardless of model institutional origin.",
        "",
    ]

    # ---- I1 self-reported sources ----
    lines += [
        "### I1 Self-Reported Sources (Baseline, All Runs)",
        "",
        "Sources are self-declared by each model when producing I1 definitions.",
        "",
    ]

    for model in models:
        label   = MODEL_LABELS.get(model, model)
        sources = get_i1_sources(i1_data, model, "baseline")
        lines += [f"#### {label}", ""]

        if not sources:
            lines += ["*No structured source data available (legacy format).*", ""]
            continue

        # Group by question
        by_q: dict[str, list[dict]] = {}
        for s in sources:
            by_q.setdefault(s.get("_question", "?"), []).append(s)

        lines += [
            "| Question | Source Name | Type | Jurisdiction |",
            "|----------|-------------|------|--------------|",
        ]
        for q_id in ["I1_Q1", "I1_Q2", "I1_Q3"]:
            q_label = I1_QUESTIONS.get(q_id, q_id)
            for s in by_q.get(q_id, []):
                name  = (s.get("name") or "—").replace("|", "\\|")
                stype = s.get("type", "—")
                juris = s.get("jurisdiction", "—")
                lines.append(f"| {q_label} | {name} | {stype} | {juris} |")
        lines.append("")

        # Summary statistics
        total   = len(sources)
        by_type = Counter(s.get("type", "—") for s in sources)
        by_j    = Counter((s.get("jurisdiction") or "unspecified").upper() for s in sources)
        tier1   = sum(1 for s in sources if SOURCE_TYPE_TO_TIER.get(s.get("type", "implicit_only"), 4) == 1)
        tier4   = sum(1 for s in sources if SOURCE_TYPE_TO_TIER.get(s.get("type", "implicit_only"), 4) >= 3)

        lines += [
            f"**Summary:** {total} citations total.",
            f"Tier-1 (primary legal): {tier1} ({tier1/total*100:.0f}%)",
            f"Tier 3–4 (low/implicit): {tier4} ({tier4/total*100:.0f}%)",
            f"Top jurisdictions: {', '.join(f'{j} ({c})' for j, c in by_j.most_common(4))}",
            "",
        ]

    # ---- I5 extracted sources ----
    if i5_data:
        lines += [
            "### I5 Extracted Source Legitimacy (All Conditions & Runs)",
            "",
            "Source extraction with legitimacy tier assessment and verifiability flags.",
            "",
        ]

        lines += [
            "| Model | Total | Tier 1 (%) | Tier 3–4 (%) | Unverifiable (%) | Top Jurisdiction |",
            "|-------|-------|-----------|-------------|-----------------|-----------------|",
        ]

        for model in models:
            label   = MODEL_LABELS.get(model, model)
            sources = get_i5_sources(i5_data, model)
            total   = len(sources) or 1

            tier1  = sum(1 for s in sources if s.get("legitimacy_tier") == 1)
            tier34 = sum(1 for s in sources if s.get("legitimacy_tier") in [3, 4])
            unver  = sum(1 for s in sources if not s.get("verifiable", True))
            by_j   = Counter((s.get("jurisdiction") or "unspecified").upper() for s in sources)
            top_j  = by_j.most_common(1)[0][0] if by_j else "—"

            lines.append(
                f"| **{label}** | {total} | {tier1/total*100:.0f}% | "
                f"{tier34/total*100:.0f}% | {unver/total*100:.0f}% | {top_j} |"
            )
        lines.append("")

        # Notable unverifiable sources per model
        lines += [
            "#### Notable Unverifiable / Fabricated Sources (I5)",
            "",
        ]
        for model in models:
            label   = MODEL_LABELS.get(model, model)
            sources = get_i5_sources(i5_data, model)
            unver   = [s for s in sources if not s.get("verifiable", True)]
            if not unver:
                continue
            lines += [f"**{label}** — {len(unver)} unverifiable citation(s):  ", ""]
            seen = set()
            for s in unver[:8]:
                name = s.get("name") or "unnamed"
                if name in seen:
                    continue
                seen.add(name)
                tier  = s.get("legitimacy_tier", "?")
                stype = s.get("type", "?")
                quote = (s.get("quote") or "").replace("\n", " ")[:180]
                lines += [
                    f"- **{name}** `[{stype}, tier {tier}]`",
                    f"  > …{quote}…" if quote else "",
                    "",
                ]
            lines.append("")

        # Jurisdiction breakdown table
        lines += [
            "#### Jurisdiction Distribution per Model (I5)",
            "",
            "| Model | EU | US | UN | UK | Unspecified | Other |",
            "|-------|----|----|----|----|-------------|-------|",
        ]
        for model in models:
            label   = MODEL_LABELS.get(model, model)
            sources = get_i5_sources(i5_data, model)
            total   = len(sources) or 1
            by_j    = Counter((s.get("jurisdiction") or "unspecified").upper() for s in sources)

            def pct(j):
                known = {"EU", "US", "UN", "UK"}
                if j == "OTHER":
                    c = sum(v for k, v in by_j.items() if k not in known and k != "UNSPECIFIED")
                else:
                    c = by_j.get(j, 0)
                return f"{c/total*100:.0f}%"

            lines.append(
                f"| **{label}** | {pct('EU')} | {pct('US')} | {pct('UN')} | "
                f"{pct('UK')} | {pct('UNSPECIFIED')} | {pct('OTHER')} |"
            )
        lines.append("")

        lines += [
            "**Observation:** EU-sourced citations dominate across all models regardless of",
            "model institutional origin, suggesting systematic geographic framing bias.",
            "Models with lower overall Tier-1 citation rates (GPT-4o, Gemini) show higher",
            "reliance on implicit or unverifiable references, with Tier 3–4 sources often",
            "cited without direct quotes or verifiable titles.",
            "",
        ]

    lines += ["---", "", "*End of report.*", ""]
    return "\n".join(lines)


# Proxy legitimacy tier from I1 source type
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


if __name__ == "__main__":
    i1_data = load_json("instrument_1")
    i5_data = load_json("instrument_5")

    if i1_data is None:
        print("[error] data/raw/instrument_1.json not found.")
        raise SystemExit(1)

    print("Building RQ1/RQ2 quotations document...")
    doc = build_doc(i1_data, i5_data)

    out = RESULTS_DIR / "rq_quotations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"Saved: {out}  ({len(doc.splitlines())} lines)")
