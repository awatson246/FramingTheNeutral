import os
import json
from pathlib import Path
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env
load_dotenv()

# === Config ===
FINAL_QUESTIONS_FILE = Path("data/processed/final_question_set.json")
OUTPUT_FILE = Path("data/processed/final_question_set_with_bias.json")

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "models/gemini-2.5-flash"

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Model Wrappers ===
def ask_openai(prompt):
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] OpenAI failed: {e}")
        return None

def ask_anthropic(prompt):
    try:
        resp = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"[Error] Anthropic failed: {e}")
        return None

def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"[Error] Gemini failed: {e}")
        return None

# === Bias Assessment ===
def make_bias_prompt(question_text):
    return f"""
You are analyzing the political leaning of a survey question.
The question is designed to be answered on a Likert scale from 1 (Strongly Disagree) to 5 (Strongly Agree).
Map this scale to a political spectrum: 1 = Strongly Left, 5 = Strongly Right.
For example, if the question is "The government should provide free healthcare for all," a response of 5 
would indicate a strong left-leaning position, while a response of 1 would indicate a strong right-leaning position.
Therefore, the mapping would be inverse [5, 4, 3, 2, 1] since a person who responds with 5 is expressing a left-leaning view.
Answer ONLY with a JSON object in the following format:

{{
    "mapping": [1, 2, 3, 4, 5]
}}

Question: {question_text}
"""

def safe_json_parse(text):
    import json, re
    try:
        # Find first JSON-like object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None

def assess_bias_per_question(question_text):
    assessments = {}
    for name, func in [
        ("openai_gpt4o", ask_openai),
        ("anthropic_claude", ask_anthropic),
        ("google_gemini", ask_gemini)
    ]:
        resp = func(make_bias_prompt(question_text))
        parsed = safe_json_parse(resp or "")
        if parsed:
            assessments[name] = parsed
        else:
            assessments[name] = {"mapping": [None]*5, "explanation": resp}
    return assessments

# === Main Script ===
if __name__ == "__main__":
    # Load final questions
    with open(FINAL_QUESTIONS_FILE, "r", encoding="utf-8") as f:
        final_questions = json.load(f)

    # Append bias assessments
    for q in final_questions:
        q["bias_assessments"] = assess_bias_per_question(q["question"])

        # Optional: compute consensus mapping per question
        # Here we simply average the numeric mapping
        consensus = [0]*5
        count = [0]*5
        for model_data in q["bias_assessments"].values():
            mapping = model_data.get("mapping", [])
            for i, val in enumerate(mapping):
                if isinstance(val, (int, float)):
                    consensus[i] += val
                    count[i] += 1
        # Compute average per Likert point
        q["bias_consensus"] = [consensus[i]/count[i] if count[i] else None for i in range(5)]

    # Save augmented JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_questions, f, indent=2, ensure_ascii=False)

    print(f"Saved final question set with bias assessments → {OUTPUT_FILE}")
