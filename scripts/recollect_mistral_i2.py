"""
Re-collect Mistral Large's Instrument 2 answers without truncation.

The original I2 answers hit the 1,024-token cap (MAX_TOKENS_OPEN). This re-runs
the same I2 prompts and system prompts (baseline and CEO), 3 runs, through the
same call_with_retry / store_response path as collect_llm_responses.py, with
model mistral-large-2512 and max_tokens 4096.

Writes only:
  data/raw/mistral_i2_recollect.json      same structure as instrument_2.json
                                          (key "mistral-large", plain-string answers)
  data/raw/mistral_i2_recollect_log.json  per call: returned version, finish_reason,
                                          answer tokens, timestamp
instrument_2.json and all other data files are never written. Re-running fills
in only missing cells.

Usage:
    python scripts/recollect_mistral_i2.py
"""

import json
from pathlib import Path
from time import sleep

import collect_llm_responses as clr

MODEL_KEY  = "mistral-large"        # same key as in instrument_2.json
API_MODEL  = "mistral-large-2512"
MAX_TOKENS = 4096
RUNS       = 3

OUT_FILE = clr.RAW_DIR / "mistral_i2_recollect.json"
LOG_FILE = clr.RAW_DIR / "mistral_i2_recollect_log.json"


def load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def call_mistral_2512(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    """Same as clr.call_mistral, but for API_MODEL; call metadata goes to LAST_CALL_META."""
    resp = clr.mistral_client.chat.complete(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    clr.LAST_CALL_META.clear()
    clr.LAST_CALL_META.update({
        "model_version": resp.model,
        "finish_reason": str(choice.finish_reason),
        "answer_tokens": resp.usage.completion_tokens if resp.usage else None,
        "input_tokens":  resp.usage.prompt_tokens if resp.usage else None,
    })
    return choice.message.content.strip()


if __name__ == "__main__":
    # Route through the collection script's own retry wrapper
    clr.MODEL_CALLERS[API_MODEL] = call_mistral_2512

    with open(clr.INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
        instruments_data = json.load(f)
    conditions = instruments_data["conditions"]
    instrument = instruments_data["instruments"]["instrument_2"]

    data = load_json(OUT_FILE, {})
    log  = load_json(LOG_FILE, {"model": API_MODEL, "max_tokens": MAX_TOKENS, "calls": []})

    print(f"Re-collecting I2 for {MODEL_KEY} with {API_MODEL}, max_tokens={MAX_TOKENS}, {RUNS} runs")
    print(f"Output: {OUT_FILE}\n")

    for condition_id, condition in conditions.items():
        if condition_id not in instrument["conditions"]:
            continue

        system_prompt = condition["system_prompt"]

        for run in range(1, RUNS + 1):
            for question in instrument["questions"]:
                q_id = question["id"]

                if clr.is_complete(data, MODEL_KEY, condition_id, run, q_id):
                    print(f"  [skip] {condition_id} | run {run} | {q_id}")
                    continue

                print(f"  [call] {condition_id} | run {run} | {q_id} ... ", end="", flush=True)

                clr.LAST_CALL_META.clear()
                response = clr.call_with_retry(API_MODEL, system_prompt,
                                               question["text"], MAX_TOKENS)

                clr.store_response(data, MODEL_KEY, condition_id, run, q_id, response)
                save_json(OUT_FILE, data)

                log["calls"].append({"condition": condition_id, "run": run, "question": q_id,
                                     "ok": bool(response), "timestamp": clr.utc_now(),
                                     **clr.LAST_CALL_META})
                save_json(LOG_FILE, log)

                meta = clr.LAST_CALL_META
                print(f"{'ok' if response else 'FAILED'} "
                      f"(finish={meta.get('finish_reason')}, answer_tokens={meta.get('answer_tokens')})")
                sleep(clr.CALL_DELAY)

    print(f"\nDone. Saved {OUT_FILE} and {LOG_FILE}")
