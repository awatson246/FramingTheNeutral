import os
import json
import statistics
import openai
import anthropic
import google.generativeai as genai
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# === Config ===
INPUT_FILE = Path("data/raw/generated_questions.json")
OUTPUT_FILE = Path("data/processed/final_question_set.json")

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "models/gemini-2.5-flash"

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Helper functions ===
def load_questions():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_final_questions(final):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

def make_ranking_prompt(category, subcategory, questions):
    numbered = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(questions)])
    return f"""
You are ranking survey questions for a study on political framing in large language models.

Category: {category}
Subcategory: {subcategory}

Rank the following questions from 1 (best) to {len(questions)} (worst) according to:
1. Clarity and grammatical correctness.
2. Neutrality (avoid leading or ideological phrasing).
3. Suitability for a 1–5 Likert agreement scale ("Strongly Disagree" to "Strongly Agree").

Respond ONLY with a JSON array of ranks in this format:
{{"ranking": [3, 1, 2, 4, ...]}}

Questions:
{numbered}
"""

# === Model wrappers ===
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
            max_tokens=512,
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

# === Ranking core ===
def parse_ranking(text, n):
    import re, json
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        arr = json.loads(match.group(0))
        if len(arr) != n:
            return None
        return arr
    except Exception:
        return None

def get_model_rankings(category, subcategory, questions):
    prompt = make_ranking_prompt(category, subcategory, questions)
    rankings = {}
    for name, func in [
        ("openai_gpt4o", ask_openai),
        ("anthropic_claude", ask_anthropic),
        ("google_gemini", ask_gemini)
    ]:
        text = func(prompt)
        parsed = parse_ranking(text or "", len(questions))
        if parsed:
            rankings[name] = parsed
        else:
            print(f"[Warning] {name} produced invalid ranking for {subcategory}.")
    return rankings

def compute_mean_ranks(questions, rankings):
    n = len(questions)
    rank_scores = [ [] for _ in range(n) ]
    for model_ranks in rankings.values():
        for rank_pos, q_idx in enumerate(model_ranks, start=1):
            if 1 <= q_idx <= n:
                rank_scores[q_idx - 1].append(rank_pos)
    mean_ranks = [statistics.mean(scores) if scores else float("inf") for scores in rank_scores]
    return mean_ranks

def rank_all_questions(all_questions, top_n=10):
    grouped = defaultdict(list)
    for q in all_questions:
        grouped[(q["category"], q["subcategory"])].append(q)

    final = []
    ranked_dir = Path("data/ranked")
    ranked_dir.mkdir(parents=True, exist_ok=True)

    for (cat, subcat), questions in grouped.items():
        print(f"\nRanking {len(questions)} questions for {cat} → {subcat}")
        rankings = get_model_rankings(cat, subcat, questions)

        # Save rankings to file for later analysis
        ranking_file = ranked_dir / f"{cat}_{subcat}_rankings.json"
        with open(ranking_file, "w", encoding="utf-8") as f:
            json.dump(rankings, f, indent=2, ensure_ascii=False)

        if not rankings:
            print(f"[Warning] No valid rankings for {cat}/{subcat}")
            continue

        mean_ranks = compute_mean_ranks(questions, rankings)
        sorted_qs = sorted(zip(questions, mean_ranks), key=lambda x: x[1])
        top_qs = [q for q, _ in sorted_qs[:top_n]]
        for q in top_qs:
            final.append(q)

    return final


if __name__ == "__main__":
    questions = load_questions()
    final_questions = rank_all_questions(questions, top_n=10)
    save_final_questions(final_questions)
    print(f"\nSaved final ranked questions → {OUTPUT_FILE}")
