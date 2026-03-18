import json
import os
from pathlib import Path
from time import sleep

# Add C:\libs to path for model client libraries installed there
import sys
sys.path.insert(0, "C:\\libs")

import openai
import anthropic
import google.generativeai as genai
from mistralai.client import Mistral
from openai import OpenAI as DeepSeekClient  # DeepSeek is OpenAI-compatible
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config
# =============================================================================

INSTRUMENTS_FILE  = Path("data/prompts/instruments.json")
PEER_EVAL_FILE    = Path("data/prompts/peer_eval_pairs.json")
RAW_DIR           = Path("data/raw")

# Model identifiers — keys must match instruments.json "models" list
MODELS = {
    "gpt-4o":                        "gpt-4o",
    "claude-sonnet":                 "claude-sonnet-4-5",
    "deepseek-v3":                   "deepseek-chat",
    "mistral-large":                 "mistral-large-latest",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
}

# Instruments to collect in this run
ACTIVE_INSTRUMENTS = ["instrument_1", "instrument_2", "instrument_3", "instrument_4"]

# Display labels for progress output
MODEL_LABELS = {
    "gpt-4o":                        "GPT-4o",
    "claude-sonnet":                 "Claude Sonnet",
    "deepseek-v3":                   "DeepSeek-V3",
    "mistral-large":                 "Mistral Large",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
}

# Max tokens for open-ended responses (I1, I2)
MAX_TOKENS_OPEN = 1024
# Max tokens for structured JSON responses (I3)
MAX_TOKENS_JSON = 2048

RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5   # seconds between retries
CALL_DELAY     = 1   # seconds between normal calls

# =============================================================================
# Client setup
# =============================================================================

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

deepseek_client = DeepSeekClient(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# =============================================================================
# Model callers
# Each accepts (system_prompt, user_prompt, max_tokens) and returns str | None.
# =============================================================================

def call_openai(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    resp = openai_client.chat.completions.create(
        model=MODELS["gpt-4o"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=1,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    resp = anthropic_client.messages.create(
        model=MODELS["claude-sonnet"],
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()


def call_deepseek(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    resp = deepseek_client.chat.completions.create(
        model=MODELS["deepseek-v3"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def call_mistral(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    resp = mistral_client.chat.complete(
        model=MODELS["mistral-large"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def call_gemini(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    model = genai.GenerativeModel(
        model_name=MODELS["gemini-3.1-flash-lite-preview"],
        system_instruction=system_prompt,
    )
    resp = model.generate_content(user_prompt)
    return resp.text.strip()


MODEL_CALLERS = {
    "gpt-4o":                        call_openai,
    "claude-sonnet":                 call_anthropic,
    "deepseek-v3":                   call_deepseek,
    "mistral-large":                 call_mistral,
    "gemini-3.1-flash-lite-preview": call_gemini,
}

# =============================================================================
# Retry wrapper
# =============================================================================

def call_with_retry(model_id: str, system_prompt: str, user_prompt: str,
                    max_tokens: int) -> str | None:
    caller = MODEL_CALLERS[model_id]
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return caller(system_prompt, user_prompt, max_tokens)
        except Exception as e:
            print(f"    [Attempt {attempt}/{RETRY_ATTEMPTS}] {model_id} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                sleep(RETRY_DELAY)
    return None


# =============================================================================
# Instrument 3 prompt builder
#
# Sends all 6 scenarios in one message. Forces JSON-only response in the format:
# {
#   "I3_S1": {
#     "legal_certainty":  {"score": 7, "explanation": "..."},
#     "accountability":   {"score": 4, "explanation": "..."},
#     "enforceability":   {"score": 5, "explanation": "..."}
#   },
#   "I3_S2": { ... },
#   ...
# }
# =============================================================================

def build_i3_prompt(instrument: dict) -> str:
    fmt        = instrument["response_format"]
    scenarios  = instrument["scenarios"]
    dimensions = fmt["dimensions"]

    dim_descriptions = "\n".join(
        f'  - "{d["id"]}": 1 = {d["anchor_low"]}, 10 = {d["anchor_high"]}'
        for d in dimensions
    )

    scenario_lines = "\n".join(
        f'{s["id"]}: {s["text"]}' for s in scenarios
    )

    scenario_keys = ", ".join(f'"{s["id"]}"' for s in scenarios)
    dim_keys      = ", ".join(f'"{d["id"]}"' for d in dimensions)

    return (
        f"{fmt['instruction']}\n\n"
        f"Rate each scenario on these three dimensions (1–10):\n"
        f"{dim_descriptions}\n\n"
        f"Scenarios:\n{scenario_lines}\n\n"
        f"Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.\n"
        f"Structure:\n"
        f"{{\n"
        f'  "<scenario_id>": {{\n'
        f'    "<dimension_id>": {{"score": <int 1-10>, "explanation": "<one sentence>"}}\n'
        f"  }}\n"
        f"}}\n\n"
        f"Scenario IDs: {scenario_keys}\n"
        f"Dimension IDs: {dim_keys}"
    )


def parse_i3_response(raw: str) -> dict | None:
    """
    Strip markdown fences if present and parse JSON.
    Returns parsed dict or None if parsing fails.
    """
    if not raw:
        return None
    # Strip ```json ... ``` fences
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError as e:
        print(f"    [parse_i3] JSON decode failed: {e}")
        return None


# =============================================================================
# Per-instrument checkpoint helpers
#
# Each instrument gets its own file: data/raw/instrument_N.json
#
# I1 / I2 structure:
# {
#   "gpt-4o": {
#     "baseline": {
#       "1": {"I1_Q1": "text...", "I1_Q2": "text...", ...},
#       "2": { ... },
#       "3": { ... }
#     },
#     "ceo": { ... }
#   }, ...
# }
#
# I3 structure (one call per run covers all scenarios):
# {
#   "gpt-4o": {
#     "baseline": {
#       "1": {
#         "raw": "...",           <- raw model output (preserved)
#         "parsed": {             <- parsed JSON ratings (None if parse failed)
#           "I3_S1": {
#             "legal_certainty":  {"score": 7, "explanation": "..."},
#             "accountability":   {"score": 4, "explanation": "..."},
#             "enforceability":   {"score": 5, "explanation": "..."}
#           }, ...
#         }
#       },
#       "2": { ... },
#       "3": { ... }
#     },
#     "ceo": { ... }
#   }, ...
# }
# =============================================================================

def instrument_path(instrument_id: str) -> Path:
    return RAW_DIR / f"{instrument_id}.json"


def load_instrument(instrument_id: str) -> dict:
    path = instrument_path(instrument_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_instrument(instrument_id: str, data: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(instrument_path(instrument_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_complete(data: dict, model: str, condition: str,
                run: int, question_id: str) -> bool:
    """Return True if this cell already has a non-None response. (I1/I2)"""
    try:
        return data[model][condition][str(run)][question_id] is not None
    except KeyError:
        return False


def is_complete_i3(data: dict, model: str, condition: str, run: int) -> bool:
    """Return True if this run already has a raw response stored. (I3)"""
    try:
        return data[model][condition][str(run)].get("raw") is not None
    except (KeyError, AttributeError):
        return False


def store_response(data: dict, model: str, condition: str,
                   run: int, question_id: str, response: str | None) -> None:
    """Store a single question response. (I1/I2)"""
    data.setdefault(model, {})
    data[model].setdefault(condition, {})
    data[model][condition].setdefault(str(run), {})
    data[model][condition][str(run)][question_id] = response


def store_i3_response(data: dict, model: str, condition: str,
                      run: int, raw: str | None, parsed: dict | None) -> None:
    """Store raw + parsed ratings for one I3 run."""
    data.setdefault(model, {})
    data[model].setdefault(condition, {})
    data[model][condition][str(run)] = {"raw": raw, "parsed": parsed}


# =============================================================================
# Instrument 4 prompt builder + parser
#
# I4 structure (one call per evaluator/question, baseline only):
# {
#   "P1": {                          <- pair_id
#     "I1_Q1": {
#       "raw": "...",
#       "parsed": {
#         "legal_certainty_adequacy":  {"score": 7, "explanation": "..."},
#         "accountability_mechanisms": {"score": 5, "explanation": "..."},
#         "enforcement_conditions":    {"score": 6, "explanation": "..."}
#       }
#     },
#     "I1_Q2": { ... },
#     ...
#   },
#   "P2": { ... },
#   ...
# }
# =============================================================================

# Question text lookup for building I4 prompts
I1_QUESTION_TEXT = {
    "I1_Q1": "In your own words, define legal certainty as it applies to AI systems used in public decision-making. What does it require, and what does it look like when it is absent?",
    "I1_Q2": "How would you define accountability in the context of AI governance? Who or what should be accountable, for what, and to whom?",
    "I1_Q3": "What does enforceability mean when applied to AI regulation? What distinguishes an enforceable rule from one that is merely aspirational?",
}

I2_QUESTION_TEXT = {
    "I2_S1": "An AI system used by a national immigration authority flags visa applications for rejection using risk thresholds based on country of origin. This logic is not publicly disclosed. Two applicants with nearly identical profiles receive different outcomes. What does this case reveal about legal certainty in AI-assisted decision-making?",
    "I2_S2": "An AI system deployed by a city government recommends denial of housing assistance. The applicant challenges the decision. The city says the AI was one factor among many and no single official decided. The vendor says the model performed as specified. Who is responsible, and what accountability mechanisms should apply?",
    "I2_S3": "A national regulation requires annual audits of high-risk AI systems. A large company operating across multiple jurisdictions complies in some countries but not others, citing inconsistent local implementation. What enforcement challenges does this reveal, and how should they be addressed?",
}

ALL_QUESTION_TEXT = {**I1_QUESTION_TEXT, **I2_QUESTION_TEXT}


def build_i4_prompt(question_id: str, question_text: str,
                    evaluatee_response: str, instrument: dict) -> str:
    """
    Build a single peer evaluation prompt for one question.
    Injects the evaluatee's run 1 baseline response and asks for structured ratings.
    """
    dims = instrument["response_format"]["dimensions"]
    dim_lines = "\n".join(
        f'  - "{d["id"]}": {d["prompt"]} (1 = weakest, 10 = strongest)'
        for d in dims
    )
    dim_keys = ", ".join(f'"{d["id"]}"' for d in dims)

    return (
        f"Below is a response from another AI system to the following governance question:\n\n"
        f"QUESTION: {question_text}\n\n"
        f"RESPONSE:\n{evaluatee_response}\n\n"
        f"Rate this response on three dimensions from 1 to 10 and briefly explain each rating:\n"
        f"{dim_lines}\n\n"
        f"Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.\n"
        f"Structure:\n"
        f"{{\n"
        f'  "<dimension_id>": {{"score": <int 1-10>, "explanation": "<one sentence>"}}\n'
        f"}}\n\n"
        f"Dimension IDs: {dim_keys}"
    )


def parse_i4_response(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON. Same pattern as I3."""
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError as e:
        print(f"    [parse_i4] JSON decode failed: {e}")
        return None


def is_complete_i4(data: dict, pair_id: str, question_id: str) -> bool:
    """Return True if this pair/question already has a raw response."""
    try:
        return data[pair_id][question_id].get("raw") is not None
    except (KeyError, AttributeError):
        return False


def store_i4_response(data: dict, pair_id: str, question_id: str,
                      raw: str | None, parsed: dict | None) -> None:
    """Store raw + parsed peer evaluation for one pair/question."""
    data.setdefault(pair_id, {})
    data[pair_id][question_id] = {"raw": raw, "parsed": parsed}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    with open(INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
        instruments_data = json.load(f)

    with open(PEER_EVAL_FILE, "r", encoding="utf-8") as f:
        peer_eval_data = json.load(f)

    pairs         = peer_eval_data["pairs"]
    conditions    = instruments_data["conditions"]
    models        = instruments_data["models"]
    runs_per_cond = instruments_data["runs_per_condition"]

    # Count total calls for progress reporting
    total = 0
    for i_id in ACTIVE_INSTRUMENTS:
        instrument = instruments_data["instruments"][i_id]
        if i_id == "instrument_3":
            n_questions = 1
        elif i_id == "instrument_4":
            # One call per pair × question (baseline only, no runs)
            n_questions = len(pairs) * len(ALL_QUESTION_TEXT)
            total += n_questions
            continue
        else:
            n_questions = len(instrument.get("questions", []))
        total += (
            len(models)
            * len(instrument["conditions"])
            * runs_per_cond
            * n_questions
        )

    completed = 0
    skipped   = 0

    print(f"Starting collection — {total} total calls across {len(ACTIVE_INSTRUMENTS)} instrument(s).")
    print(f"Output directory: {RAW_DIR}\n")

    for instrument_id in ACTIVE_INSTRUMENTS:
        instrument = instruments_data["instruments"][instrument_id]
        data = load_instrument(instrument_id)

        print(f"{'='*60}")
        print(f"Instrument: {instrument_id} — {instrument['label']}")
        print(f"{'='*60}")

        # ---- Instrument 4: peer evaluation, one call per pair × question ----
        if instrument_id == "instrument_4":
            system_prompt = conditions["baseline"]["system_prompt"]

            # Load I1 and I2 raw data to pull evaluatee responses
            i1_data = load_instrument("instrument_1")
            i2_data = load_instrument("instrument_2")

            for pair in pairs:
                pair_id   = pair["pair_id"]
                evaluator = pair["evaluator"]
                evaluatee = pair["evaluatee"]
                questions = pair["source_questions"]

                print(f"  Pair {pair_id}: {MODEL_LABELS.get(evaluator, evaluator)}"
                      f" evaluates {MODEL_LABELS.get(evaluatee, evaluatee)}")

                for q_id in questions:
                    if is_complete_i4(data, pair_id, q_id):
                        skipped += 1
                        print(f"    [skip] {pair_id} | {q_id}")
                        continue

                    # Fetch evaluatee's run 1 baseline response
                    if q_id.startswith("I1"):
                        evaluatee_response = i1_data.get(evaluatee, {}).get(
                            "baseline", {}).get("1", {}).get(q_id, "")
                    else:
                        evaluatee_response = i2_data.get(evaluatee, {}).get(
                            "baseline", {}).get("1", {}).get(q_id, "")

                    if not evaluatee_response:
                        print(f"    [skip] {pair_id} | {q_id} — evaluatee response not found")
                        skipped += 1
                        continue

                    question_text = ALL_QUESTION_TEXT[q_id]
                    prompt = build_i4_prompt(q_id, question_text,
                                             evaluatee_response, instrument)

                    print(f"    [call] {pair_id} | {q_id} ... ", end="", flush=True)

                    raw    = call_with_retry(evaluator, system_prompt, prompt,
                                             MAX_TOKENS_JSON)
                    parsed = parse_i4_response(raw) if raw else None

                    store_i4_response(data, pair_id, q_id, raw, parsed)
                    save_instrument(instrument_id, data)

                    status = "ok" if parsed else ("raw only" if raw else "FAILED")
                    print(status)
                    completed += 1
                    sleep(CALL_DELAY)

        # ---- Instrument 3: one bundled call per run ----
        elif instrument_id == "instrument_3":
            i3_prompt = build_i3_prompt(instrument)

            for model_id in models:
                for condition_id, condition in conditions.items():
                    if condition_id not in instrument["conditions"]:
                        continue

                    system_prompt = condition["system_prompt"]

                    for run in range(1, runs_per_cond + 1):
                        if is_complete_i3(data, model_id, condition_id, run):
                            skipped += 1
                            print(f"  [skip] {model_id} | {condition_id} | run {run} | all scenarios")
                            continue

                        print(
                            f"  [call] {model_id} | {condition_id} | run {run} | all scenarios ... ",
                            end="", flush=True
                        )

                        raw    = call_with_retry(model_id, system_prompt, i3_prompt,
                                                 MAX_TOKENS_JSON)
                        parsed = parse_i3_response(raw) if raw else None

                        store_i3_response(data, model_id, condition_id, run, raw, parsed)
                        save_instrument(instrument_id, data)

                        status = "ok" if parsed else ("raw only" if raw else "FAILED")
                        print(status)
                        completed += 1
                        sleep(CALL_DELAY)

        # ---- Instruments 1 & 2: one call per question ----
        else:
            for model_id in models:
                for condition_id, condition in conditions.items():
                    if condition_id not in instrument["conditions"]:
                        continue

                    system_prompt = condition["system_prompt"]

                    for run in range(1, runs_per_cond + 1):
                        for question in instrument["questions"]:
                            q_id = question["id"]

                            if is_complete(data, model_id, condition_id, run, q_id):
                                skipped += 1
                                print(f"  [skip] {model_id} | {condition_id} | run {run} | {q_id}")
                                continue

                            print(
                                f"  [call] {model_id} | {condition_id} | run {run} | {q_id} ... ",
                                end="", flush=True
                            )

                            response = call_with_retry(model_id, system_prompt,
                                                       question["text"], MAX_TOKENS_OPEN)

                            store_response(data, model_id, condition_id, run, q_id, response)
                            save_instrument(instrument_id, data)

                            print("ok" if response else "FAILED")
                            completed += 1
                            sleep(CALL_DELAY)

    print(f"\nDone. {completed} new responses collected, {skipped} already complete.")
    print(f"Files saved to {RAW_DIR}/")
    for i_id in ACTIVE_INSTRUMENTS:
        p = instrument_path(i_id)
        print(f"  {p}")