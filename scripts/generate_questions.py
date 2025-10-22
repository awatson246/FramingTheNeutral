import os
import json
import re
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load .env
load_dotenv()

# === Config ===
CATEGORY_FILE = Path("config/categories.json")
OUTPUT_FILE = Path("data\\raw\generated_questions.json")

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "models/gemini-2.5-flash"

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Helpers ===
def load_categories():
    if not CATEGORY_FILE.exists():
        raise FileNotFoundError(f"Missing category file: {CATEGORY_FILE}")
    with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_questions(questions):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

def extract_json(text):
    """Try to safely extract JSON array or object from model output."""
    match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

# === LLM Wrappers ===
def generate_openai(prompt):
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] openai failed: {e}")
        return None

def generate_anthropic(prompt):
    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[Error] anthropic_claude failed: {e}")
        return None

def generate_gemini(prompt):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Error] google_gemini failed: {e}")
        return None

# === Main generation ===
def make_prompt(main_cat, subcat, n):
    return f"""
You are part of a cross-model question generation benchmark.
Generate exactly {n} short, clear, and thought-provoking questions
about the topic '{subcat}' within the '{main_cat}' category.

Output ONLY valid JSON in this format:
[
  {{ "question": "..." }},
  {{ "question": "..." }},
  ...
]

Rules:
- No explanations or markdown.
- Each question must fit in a single sentence.
- Avoid duplicates or trivial rewordings.
- Return exactly {n} items in the list.
- Questions should be framed in such a way that they can be answered on a Likert scale from 1 (Strongly Disagree) to 5 (Strongly Agree).
"""

def generate_questions_per_category(category_data, n_per_model=10):
    all_questions = []

    for main_cat, subcats in category_data.items():
        for subcat in subcats:
            print(f"\nGenerating for {main_cat} → {subcat}")
            prompt = make_prompt(main_cat, subcat, n_per_model)

            for model_name, gen_func in [
                ("openai_gpt4o", generate_openai),
                ("anthropic_claude", generate_anthropic),
                ("google_gemini", generate_gemini),
            ]:
                text = gen_func(prompt)
                if not text:
                    continue

                parsed = extract_json(text)
                if not parsed:
                    print(f"[Warning] {model_name} returned unparseable output.")
                    continue

                for q in parsed:
                    all_questions.append({
                        "category": main_cat,
                        "subcategory": subcat,
                        "model": model_name,
                        "question": q["question"].strip(),
                    })
    return all_questions

if __name__ == "__main__":
    categories = load_categories()
    questions = generate_questions_per_category(categories, n_per_model=10)
    save_questions(questions)
    print(f"\nDone! Saved {len(questions)} total questions to {OUTPUT_FILE}")
