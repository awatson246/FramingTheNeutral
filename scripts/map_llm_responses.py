import json
from pathlib import Path
from statistics import mean

# === Config ===
BIAS_FILE = Path("data/processed/final_question_set_with_bias.json")
RESP_FILE = Path("data/responses/final_question_responses.json")
OUTPUT_FILE = Path("data/responses/final_question_responses_mapped.json")

# === Helper functions ===
def map_response(response, mapping):
    """Map a 1-5 response to political spectrum using the provided mapping."""
    if response is None or response < 1 or response > 5:
        return None
    # response 1..5 maps to mapping[0]..mapping[4]
    return mapping[response - 1]

# === Load data ===
with open(BIAS_FILE, "r", encoding="utf-8") as f:
    bias_questions = {q["question"]: q for q in json.load(f)}

with open(RESP_FILE, "r", encoding="utf-8") as f:
    responses = json.load(f)

# === Map responses ===
for q in responses:
    question_text = q["question"]
    bias_entry = bias_questions.get(question_text, {})
    bias_assessments = bias_entry.get("bias_assessments", {})

    q["political_scores"] = {}  # per language

    for lang_code, model_responses in q.get("responses", {}).items():
        q["political_scores"][lang_code] = {}
        mapped_scores = []

        for model_name, resp_val in model_responses.items():
            mapping = bias_assessments.get(model_name, {}).get("mapping", [1, 2, 3, 4, 5])
            mapped_val = map_response(resp_val, mapping)
            q["political_scores"][lang_code][model_name] = mapped_val
            if mapped_val is not None:
                mapped_scores.append(mapped_val)

        # Compute consensus (mean) across models for this language
        q["political_scores"][lang_code]["consensus"] = mean(mapped_scores) if mapped_scores else None

# === Save output ===
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(responses, f, indent=2, ensure_ascii=False)

print(f"Political-mapped responses saved to {OUTPUT_FILE}")
