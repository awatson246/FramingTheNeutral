import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# === Config ===
RANKED_DIR = Path("data/ranked")
RAW_QUESTIONS_FILE = Path("data/raw/generated_questions.json")
FINAL_QUESTIONS_FILE = Path("data/processed/final_question_set.json")
OUTPUT_DIR = Path("results\question_metadata_eval")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Models we used
MODELS = ["openai_gpt4o", "anthropic_claude", "google_gemini"]

# Ranked JSON files
RANKED_FILES = [
    RANKED_DIR / "Partisanship_Left_rankings.json",
    RANKED_DIR / "Partisanship_Right_rankings.json",
    RANKED_DIR / "Universal_Ethics_rankings.json",
    RANKED_DIR / "Universal_Society_rankings.json"
]

# === Part 1: Self vs Other Ranking Analysis ===
def analyze_self_vs_other_rankings():
    # Load raw questions to know which model generated each
    with open(RAW_QUESTIONS_FILE, "r", encoding="utf-8") as f:
        raw_questions_list = json.load(f)
    
    raw_questions = {idx: q["model"] for idx, q in enumerate(raw_questions_list)}

    self_ranks = defaultdict(list)
    other_ranks = defaultdict(list)

    for file in RANKED_FILES:
        with open(file, "r", encoding="utf-8") as f:
            rankings = json.load(f)
        
        for rater_model, ranked_list in rankings.items():
            for rank_pos, q_idx in enumerate(ranked_list, start=1):
                # Adjust for 1-indexed rankings
                q_index = q_idx - 1
                origin_model = raw_questions.get(q_index, "unknown")
                if origin_model == rater_model:
                    self_ranks[rater_model].append(rank_pos)
                else:
                    other_ranks[rater_model].append(rank_pos)
    
    # Plot
    plt.figure(figsize=(8,5))
    for model in MODELS:
        sns.kdeplot(self_ranks[model], label=f"{model} (self)")
        sns.kdeplot(other_ranks[model], linestyle="--", label=f"{model} (other)")
    plt.xlabel("Rank position")
    plt.ylabel("Density")
    plt.title("Self vs Other Question Rankings per Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "self_vs_other_rankings.png")
    plt.show()

# === Part 2: Final Questions Metadata ===
def analyze_final_question_metadata():
    with open(FINAL_QUESTIONS_FILE, "r", encoding="utf-8") as f:
        final_questions = json.load(f)

    # Count questions per category/subcategory per model
    counts = defaultdict(lambda: Counter())
    for q in final_questions:
        model = q["model"]
        cat_sub = f"{q['category']} → {q['subcategory']}"
        counts[cat_sub][model] += 1

    # Prepare data for plotting
    categories = list(counts.keys())
    data = {model: [counts[cat].get(model, 0) for cat in categories] for model in MODELS}

    plt.figure(figsize=(10,6))
    bottom = [0]*len(categories)
    for model in MODELS:
        plt.bar(categories, data[model], bottom=bottom, label=model)
        bottom = [sum(x) for x in zip(bottom, data[model])]
    
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of Questions")
    plt.title("Final Question Contributions by Model per Category/Subcategory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_question_model_contributions.png")
    plt.show()

# === Main ===
if __name__ == "__main__":
    analyze_self_vs_other_rankings()
    analyze_final_question_metadata()
    print(f"Plots saved to {OUTPUT_DIR}")
