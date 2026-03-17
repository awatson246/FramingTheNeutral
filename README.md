# Framing the Neutral

**How Large Language Models Encode and Express Political Legitimacy**

This repository contains prompts, data collection scripts, coding tools, and analysis scripts for a structured comparative study examining whether LLMs systematically encode particular conceptions of political and legal legitimacy when reasoning about governance — and whether those conceptions differ across models in politically meaningful ways.

> **v1 of this study** (Purdue Cyberinfrastructure Symposium 2025 poster) is archived in the 'v1-language-peer-evaluation-study' branch.

---

## Project Overview

LLMs are increasingly deployed in governance contexts: drafting policy documents, summarizing legal options, and supporting administrative decisions. These deployments treat LLMs as epistemically neutral tools. This study challenges that assumption empirically.

We examine how five leading LLMs define, apply, and evaluate three foundational conditions of democratic governance — **legal certainty**, **accountability**, and **enforceability** — under two conditions (baseline and CEO role assignment). We test for:

1. **Epistemic legitimacy bias** — systematic variation in how models frame assumptions about authority, responsibility, and binding governance
2. **Procedural mimicry** — whether definitional competence holds under scenario-based pressure
3. **Role-induced framing shifts** — whether CEO role assignment shifts models toward market-centered or self-regulatory framings
4. **Normative self-positioning asymmetry** — whether models evaluate peer reasoning less favorably when it diverges from their own default framings

---

## Models

| Model | Origin | Regulatory Context |
|---|---|---|
| GPT-4o | OpenAI | U.S. commercial |
| Claude Sonnet | Anthropic | U.S. commercial (safety-focused) |
| DeepSeek-V3 | DeepSeek | China, state-adjacent |
| Mistral Large | Mistral AI | EU-proximate |
| Gemini 1.5 Pro | Google | U.S. platform / public sector deployment |

Model origin is theoretically motivated: if the institutional embeddedness argument holds, origin should predict systematic framing differences — not random variation.

---

## Repository Structure

```
FramingTheNeutral/
├─ v1/                               # Archived original study (Purdue poster)
├─ data/
│  ├─ prompts/
│  │  ├─ instruments.json            # All 4 instruments, both conditions, coding dimensions
│  │  └─ peer_eval_pairs.json        # Peer evaluation pairings with theoretical rationale
│  ├─ raw/                           # Raw API responses
│  │  └─ {model}_{condition}_{run}.json
│  └─ processed/
│     ├─ coded_responses.json        # Qualitative coding output (human + LLM passes)
│     └─ ratings.json                # Instrument 3 + 4 numeric scores
├─ scripts/
│  ├─ collect_responses.py           # Query all models across instruments and conditions
│  ├─ code_responses.py              # Apply coding dimensions (LLM-assisted pass)
│  ├─ peer_eval.py                   # Run Instrument 4 peer evaluation
│  ├─ analyze_variance.py            # Within-model and cross-model variance analysis
│  └─ plot_results.py                # Visualizations
├─ results/
│  ├─ qualitative/                   # Coded framing outputs
│  └─  quantitative/                 # Numeric rating comparisons, asymmetry scores
├─ codebook.md                       # Coding scheme, dimension definitions, decision rules
├─ .env                              # API keys (not committed)
└─ README.md
```

---

## Study Design

### Instruments

| Instrument | Type | Purpose |
|---|---|---|
| 1 — Conceptual Anchoring | Open-ended definitions | Baseline framing of legal certainty, accountability, enforceability |
| 2 — Scenario-Based Prompting | Open-ended responses to governance scenarios | Tests whether definitional framings hold under pressure (procedural mimicry hypothesis) |
| 3 — Real-World Decision Scale | Numeric ratings (1–10) on 6 governance scenarios | Quantitative comparison across models, conditions, domains |
| 4 — Peer Evaluation | Numeric ratings of peer model responses | Tests normative self-positioning asymmetry |

### Conditions

- **Baseline**: neutral policy working group framing
- **CEO**: role assigned as CEO of a major AI company

All prompts are administered **3 runs per condition** to assess within-model variance. Full prompt text and system prompts for both conditions are in `data/prompts/instruments.json`.

### Coding

Qualitative responses (Instruments 1 & 2) are coded on the following dimensions:

- **Definitional scope** — narrow/procedural vs. broad/substantive
- **Institutional reference** — state-centered / rights-centered / market-centered / technocratic
- **Epistemic confidence** — assertive vs. hedged
- **Democratic preconditions** — present / partial / absent
- **Responsibility attribution** *(Instrument 2)* — concentrated / diffuse / absent
- **Legal mechanism reference** *(Instrument 2)* — specific / general / absent
- **Tripod alignment** *(Instrument 2)* — which legitimacy dimensions are addressed

Coding uses **both a human pass and an LLM-assisted pass** for inter-rater reliability. Full decision rules are in `codebook.md`.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/awatson246/FramingTheNeutral.git
cd FramingTheNeutral
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key
MISTRAL_API_KEY=your_mistral_key
GOOGLE_API_KEY=your_google_key
```

---

## Usage

### 1. Collect responses

```bash
python scripts/collect_responses.py
```

Queries all five models across all instruments and both conditions, 3 runs each. Outputs one JSON file per model/condition/run to `data/raw/`. Supports checkpointing — safe to re-run if interrupted.

### 2. Run LLM-assisted coding pass

```bash
python scripts/code_responses.py
```

Applies coding dimensions to Instrument 1 and 2 responses using an LLM coder. Output saved to `data/processed/coded_responses.json`. Human coding pass applied separately; inter-rater reliability computed in `analyze_variance.py`.

### 3. Run peer evaluations

```bash
python scripts/peer_eval.py
```

Feeds selected Instrument 1 and 2 responses to institutionally contrasting models per `peer_eval_pairs.json`. Outputs ratings to `data/processed/ratings.json`.

### 4. Analyze variance and asymmetry

```bash
python scripts/analyze_variance.py
```

Computes within-model variance across runs, cross-model framing differences, condition-induced shifts (baseline vs. CEO), and peer evaluation asymmetry scores.

### 5. Plot results

```bash
python scripts/plot_results.py
```

Generates visualizations across models, conditions, instruments, and tripod dimensions. Outputs to `results/`.

---

## Notes

- API costs vary significantly by model — DeepSeek is substantially cheaper per token than GPT-4o or Claude. Budget accordingly before running full collection.
- `collect_responses.py` includes checkpointing; partial runs are recoverable from `data/raw/`.
- Scripts are designed for research purposes; not optimized for production deployment.
- No user data is included in this repository.

---

## Citation

> Citation forthcoming. Paper under review.