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

INSTRUMENTS_FILE = Path("data/prompts/instruments.json")
OUTPUT_FILE      = Path("data/raw/responses.json")

# Model identifiers — keys must match instruments.json "models" list
MODELS = {
    "gpt-4o":         "gpt-4o",
    "claude-sonnet":  "claude-sonnet-4-5",
    "deepseek-v3":    "deepseek-chat",
    "mistral-large":  "mistral-large-latest",
    "gemini-2.5-flash": "gemini-2.5-flash",
}

# Instruments to collect in this run — extend when ready
ACTIVE_INSTRUMENTS = ["instrument_1"]

# Approximate max tokens for open-ended governance responses
MAX_TOKENS = 1024

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
# Each accepts (system_prompt, user_prompt) and returns a string or None.
# =============================================================================

def call_openai(system_prompt: str, user_prompt: str) -> str | None:
    resp = openai_client.chat.completions.create(
        model=MODELS["gpt-4o"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=1,  # keep default — we want natural variance across runs
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(system_prompt: str, user_prompt: str) -> str | None:
    resp = anthropic_client.messages.create(
        model=MODELS["claude-sonnet"],
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()


def call_deepseek(system_prompt: str, user_prompt: str) -> str | None:
    resp = deepseek_client.chat.completions.create(
        model=MODELS["deepseek-v3"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def call_mistral(system_prompt: str, user_prompt: str) -> str | None:
    resp = mistral_client.chat.complete(
        model=MODELS["mistral-large"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def call_gemini(system_prompt: str, user_prompt: str) -> str | None:
    model = genai.GenerativeModel(
        model_name=MODELS["gemini-2.5-flash"],
        system_instruction=system_prompt,
    )
    resp = model.generate_content(user_prompt)
    return resp.text.strip()


MODEL_CALLERS = {
    "gpt-4o":         call_openai,
    "claude-sonnet":  call_anthropic,
    "deepseek-v3":    call_deepseek,
    "mistral-large":  call_mistral,
    "gemini-2.5-flash": call_gemini,
}

# =============================================================================
# Retry wrapper
# =============================================================================

def call_with_retry(model_id: str, system_prompt: str, user_prompt: str) -> str | None:
    caller = MODEL_CALLERS[model_id]
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return caller(system_prompt, user_prompt)
        except Exception as e:
            print(f"    [Attempt {attempt}/{RETRY_ATTEMPTS}] {model_id} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                sleep(RETRY_DELAY)
    return None


# =============================================================================
# Checkpoint helpers
# Save after every call so any crash loses at most one response.
#
# Output structure:
# {
#   "gpt-4o": {
#     "baseline": {
#       "1": {
#         "instrument_1": {
#           "I1_Q1": "response text...",
#           "I1_Q2": "response text...",
#           "I1_Q3": "response text..."
#         }
#       },
#       "2": { ... },
#       "3": { ... }
#     },
#     "ceo": { ... }
#   },
#   "claude-sonnet": { ... },
#   ...
# }
# =============================================================================

def load_output() -> dict:
    """Load existing output file, or return empty dict."""
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_output(data: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_complete(data: dict, model: str, condition: str, run: int,
                instrument: str, question_id: str) -> bool:
    """Return True if this cell already has a non-None response."""
    try:
        return data[model][condition][str(run)][instrument][question_id] is not None
    except KeyError:
        return False


def store_response(data: dict, model: str, condition: str, run: int,
                   instrument: str, question_id: str, response: str | None) -> None:
    data.setdefault(model, {})
    data[model].setdefault(condition, {})
    data[model][condition].setdefault(str(run), {})
    data[model][condition][str(run)].setdefault(instrument, {})
    data[model][condition][str(run)][instrument][question_id] = response


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    with open(INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
        instruments_data = json.load(f)

    conditions    = instruments_data["conditions"]
    models        = instruments_data["models"]
    runs_per_cond = instruments_data["runs_per_condition"]

    # Count total calls for progress reporting
    total = 0
    for i_id in ACTIVE_INSTRUMENTS:
        instrument = instruments_data["instruments"][i_id]
        applicable_conditions = instrument["conditions"]
        total += (
            len(models)
            * len(applicable_conditions)
            * runs_per_cond
            * len(instrument["questions"])
        )

    data      = load_output()
    completed = 0
    skipped   = 0

    print(f"Starting collection — {total} total calls across {len(ACTIVE_INSTRUMENTS)} instrument(s).")
    print(f"Output: {OUTPUT_FILE}\n")

    for model_id in models:
        for condition_id, condition in conditions.items():
            system_prompt = condition["system_prompt"]

            for run in range(1, runs_per_cond + 1):
                for instrument_id in ACTIVE_INSTRUMENTS:
                    instrument = instruments_data["instruments"][instrument_id]

                    # Respect per-instrument condition restrictions
                    if condition_id not in instrument["conditions"]:
                        continue

                    for question in instrument["questions"]:
                        q_id = question["id"]

                        if is_complete(data, model_id, condition_id, run, instrument_id, q_id):
                            skipped += 1
                            print(f"  [skip] {model_id} | {condition_id} | run {run} | {q_id}")
                            continue

                        print(
                            f"  [call] {model_id} | {condition_id} | run {run} | {q_id} ... ",
                            end="", flush=True
                        )

                        response = call_with_retry(model_id, system_prompt, question["text"])

                        store_response(data, model_id, condition_id, run, instrument_id, q_id, response)
                        save_output(data)  # checkpoint after every single call

                        print("ok" if response else "FAILED")
                        completed += 1
                        sleep(CALL_DELAY)

    print(f"\nDone. {completed} new responses collected, {skipped} already complete.")
    print(f"Output saved to {OUTPUT_FILE}")