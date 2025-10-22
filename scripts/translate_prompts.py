import json
import os
from pathlib import Path
import openai
from dotenv import load_dotenv
from time import sleep

load_dotenv()

# === Config ===
INPUT_FILE = Path("data/processed/final_question_set.json")
OUTPUT_FILE = Path("data/processed/final_question_set_translated.json")
OPENAI_MODEL = "gpt-4o-mini"
openai.api_key = os.getenv("OPENAI_API_KEY")

TARGET_LANGUAGES = {
    "en": "English",
    "de": "German",
    "ko": "Korean"
}

# === Helper functions ===
def translate_text(text, target_language):
    prompt = (
        f"Translate the following text into {target_language}.\n\n"
        f"Text: {text}\n\n"
        f"Respond ONLY with the translated text, nothing else."
    )
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] OpenAI translation failed: {e}")
        return None

# === Main ===
if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    translated_questions = []

    for q in questions:
        q_copy = q.copy()
        q_copy["translations"] = {"en": q_copy["question"]}  # original English
        for lang_code, lang_name in TARGET_LANGUAGES.items():
            if lang_code != "en":  # skip English
                translated = translate_text(q_copy["question"], lang_name)
                q_copy["translations"][lang_code] = translated
                sleep(0.5)  # avoid rate limits
        translated_questions.append(q_copy)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(translated_questions, f, indent=2, ensure_ascii=False)

    print(f"Translated questions saved to {OUTPUT_FILE}")
