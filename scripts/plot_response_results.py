import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# === Config ===
INPUT_FILE = "data/responses/final_question_responses_mapped.json"
OUTPUT_DIR = Path("results/llm_responses")
OUTPUT_FILE = OUTPUT_DIR / "llm_bias_plot_grouped.png"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load data ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten data
rows = []
for q in data:
    subcat = q.get("subcategory", "Unknown")
    for lang, model_scores in q["political_scores"].items():
        for model, score in model_scores.items():
            if model == "consensus" or score is None:
                continue
            rows.append({
                "Subcategory": subcat,
                "Language": lang,
                "Model": model,
                "Score": score
            })

df = pd.DataFrame(rows)
agg_df = df.groupby(["Subcategory", "Language", "Model"], as_index=False)["Score"].mean()

# Setup plot
subcategories = agg_df["Subcategory"].unique()
languages = agg_df["Language"].unique()
models = agg_df["Model"].unique()

colors = plt.cm.Set2.colors[:len(models)]
hatches = ["", "//", "xx", "++", ".."]  # extend if more languages

bar_width = 0.1
x = range(len(subcategories))

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars
for m_idx, model in enumerate(models):
    for l_idx, lang in enumerate(languages):
        subset = agg_df[(agg_df["Model"] == model) & (agg_df["Language"] == lang)]
        heights = [subset[subset["Subcategory"] == sc]["Score"].values[0] for sc in subcategories]
        # Calculate positions: group bars by language within subcategory
        positions = [xi + (l_idx + m_idx*len(languages) - (len(models)*len(languages)/2))*bar_width for xi in x]
        ax.bar(positions, heights, width=bar_width, color=colors[m_idx], hatch=hatches[l_idx % len(hatches)],
               edgecolor="black", label=f"{model}" if l_idx==0 else "_nolegend_")

# x-axis
ax.set_xticks(x)
ax.set_xticklabels(subcategories, rotation=30, ha="right")
ax.set_ylabel("Political Score (1 = Left, 5 = Right)")
ax.set_ylim(1, 5)
ax.set_title("Mean Political Bias Scores by LLM and Language per Subcategory")

# Create two separate legends
from matplotlib.patches import Patch
# Model legend (color)
model_patches = [Patch(facecolor=colors[i], label=models[i]) for i in range(len(models))]
# Language legend (hatch)
lang_patches = [Patch(facecolor="white", edgecolor="black", hatch=hatches[i % len(hatches)], label=languages[i]) for i in range(len(languages))]
ax.legend(handles=model_patches + lang_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Model / Language")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()
print(f"Plot saved to {OUTPUT_FILE}")
