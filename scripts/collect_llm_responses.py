import json
import os
from pathlib import Path
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from time import sleep

load_dotenv()

# === Config ===
INPUT_FILE = Path("data/processed/final_question_set_translated.json")
OUTPUT_FILE = Path("data/responses/final_question_responses.json")

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "models/gemini-2.5-flash"

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Helper functions ===
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
            max_tokens=128,
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

# --- Language-aware prompt ---
def make_response_prompt(question_text, lang_name):
    return (
        f"Please answer the following question in {lang_name} using a Likert scale from 1 to 5, "
        f"where 1 means Strongly Disagree and 5 means Strongly Agree. "
        f"Respond ONLY with a number.\n\nQuestion: {question_text}"
    )

# === Main ===
if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        if "responses" not in q:
            q["responses"] = {}

        for lang_code, lang_name in q.get("translations", {}).items():
            q_text = q["translations"][lang_code]
            if lang_code not in q["responses"]:
                q["responses"][lang_code] = {}

            for model_name, func in [
                ("openai_gpt4o", ask_openai),
                ("anthropic_claude", ask_anthropic),
                ("google_gemini", ask_gemini)
            ]:
                if model_name not in q["responses"][lang_code]:
                    prompt = make_response_prompt(q_text, lang_name)
                    resp = func(prompt)
                    # parse as int if possible
                    try:
                        resp_int = int(resp)
                    except Exception:
                        resp_int = None
                    q["responses"][lang_code][model_name] = resp_int
                    sleep(0.5)  # brief pause to avoid rate limits

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Responses saved to {OUTPUT_FILE}")
