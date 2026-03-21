"""
nlp_corpus_enhanced_femi.py - Module 3 Script

Purpose

  Perform exploratory analysis of a small, controlled text corpus.
  Demonstrate how structure emerges from token distributions,
  category comparisons, co-occurrence patterns, and bigrams.

New Analytical Questions

- Which categories have the highest total number of tokens?
- Which tokens are unique to one category and not shared with others?
- What are the most common bigrams within each category?
- How dense is the vocabulary in each category (unique tokens vs. total tokens)?
- Which context words appear most often around a selected target token?

Run from root project folder with:

  uv run python -m nlp.nlp_corpus_enhanced_femi
"""

# ============================================================
# Section 1. Setup and Imports
# ============================================================

from collections import Counter, defaultdict
import logging
from pathlib import Path

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
import polars as pl

print("Imports complete.")

# ============================================================
# Configure Logging
# ============================================================

LOG: logging.Logger = get_logger("CI", level="DEBUG")

ROOT_PATH: Path = Path.cwd()
NOTEBOOKS_PATH: Path = ROOT_PATH / "notebooks"
SCRIPTS_PATH: Path = ROOT_PATH / "scripts"

log_header(LOG, "MODULE 3: CORPUS EXPLORATION")
LOG.info("START script.....")

log_path(LOG, "ROOT_PATH", ROOT_PATH)
log_path(LOG, "NOTEBOOKS_PATH", NOTEBOOKS_PATH)
log_path(LOG, "SCRIPTS_PATH", SCRIPTS_PATH)

LOG.info("Logger configured.")

# ============================================================
# Section 2. Define Corpus (Labeled Text Documents)
# ============================================================

corpus: list[dict[str, str]] = [
    {"category": "dog", "text": "A dog barks loudly."},
    {"category": "dog", "text": "The puppy runs in the yard."},
    {"category": "dog", "text": "A canine wears a leash."},
    {"category": "dog", "text": "The kennel holds the dog."},
    {"category": "dog", "text": "The dog ran across the yard."},
    {"category": "dog", "text": "The puppy ran across the yard."},
    {"category": "cat", "text": "A cat sleeps quietly."},
    {"category": "cat", "text": "The kitten plays with yarn."},
    {"category": "cat", "text": "A feline purrs softly."},
    {"category": "cat", "text": "The cat has whiskers."},
    {"category": "cat", "text": "The cat slept near the window."},
    {"category": "cat", "text": "The kitten slept near the window."},
    {"category": "car", "text": "A car drives on the road."},
    {"category": "car", "text": "The sedan parks in the garage."},
    {"category": "car", "text": "A vehicle has four wheels."},
    {"category": "car", "text": "The car moves down the highway."},
    {"category": "car", "text": "The car stopped near the garage."},
    {"category": "car", "text": "The sedan stopped near the garage."},
    {"category": "truck", "text": "A truck carries cargo."},
    {"category": "truck", "text": "The pickup pulls a trailer."},
    {"category": "truck", "text": "The engine powers the truck."},
    {"category": "truck", "text": "The truck hauls heavy loads."},
]

print(f"Corpus contains {len(corpus)} documents.")

# ============================================================
# Section 3. Tokenize and Clean Text
# ============================================================


def tokenize(text: str) -> list[str]:
    tokens = text.lower().split()
    return [
        t.strip(".,:;!?()[]\"'") for t in tokens if len(t.strip(".,:;!?()[]\"'")) > 2
    ]


records_list: list[dict[str, str]] = []

for doc in corpus:
    tokens = tokenize(doc["text"])
    for token in tokens:
        records_list.append({"category": doc["category"], "token": token})

token_df: pl.DataFrame = pl.DataFrame(records_list)

print("Tokenization complete.")
print(token_df.head(10))

# ============================================================
# Section 4. Compute Global Token Frequencies
# ============================================================

global_freq_df: pl.DataFrame = (
    token_df.group_by("token").len().sort("len", descending=True)
)

print("Top global tokens:")
print(global_freq_df.head(10))

# ============================================================
# Section 5. Compute Token Frequencies by Category
# ============================================================

category_freq_df: pl.DataFrame = (
    token_df.group_by(["category", "token"])
    .len()
    .sort(["category", "len"], descending=[False, True])
)

print("Top tokens by category:")
print(category_freq_df.head(12))

# ============================================================
# Section 6. Identify Top Tokens per Category
# ============================================================

top_per_category_dict: dict[str, list[str]] = {}

for category in token_df["category"].unique().to_list():
    subset_df = category_freq_df.filter(pl.col("category") == category).head(5)
    top_tokens_list = subset_df["token"].to_list()
    top_per_category_dict[category] = top_tokens_list
    print(f"{category.upper()} top tokens: {top_tokens_list}")

# ============================================================
# Section 7. Analyze Co-occurrence (Context Windows)
# ============================================================

WINDOW_SIZE: int = 2
co_occurrence_dict: dict[str, list[str]] = defaultdict(list)

for doc in corpus:
    tokens = tokenize(doc["text"])
    for i, token in enumerate(tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        context = tokens[start:end]
        for ctx in context:
            if ctx != token:
                co_occurrence_dict[token].append(ctx)

for target in ["dog", "cat", "car", "truck"]:
    print(f"\nContext for '{target}':")
    print(co_occurrence_dict[target][:10])

# ============================================================
# Section 8. Create Bigrams and Compute Frequencies
# ============================================================

bigrams_records: list[dict[str, str]] = []

for doc in corpus:
    tokens = tokenize(doc["text"])
    for i in range(len(tokens) - 1):
        bigrams_records.append(
            {
                "category": doc["category"],
                "bigram": f"{tokens[i]} {tokens[i + 1]}",
            }
        )

bigram_df: pl.DataFrame = pl.DataFrame(bigrams_records)

bigram_freq_df: pl.DataFrame = (
    bigram_df.group_by("bigram").len().sort("len", descending=True)
)

print("Top bigrams:")
print(bigram_freq_df.head(10))

# ============================================================
# Section 9. NEW QUESTION:
# Which categories have the highest total number of tokens?
# ============================================================

category_token_totals_df: pl.DataFrame = (
    token_df.group_by("category").len().sort("len", descending=True)
)

print("\nTotal token count by category:")
print(category_token_totals_df)

# ============================================================
# Section 10. NEW QUESTION:
# Which tokens are unique to one category and not shared with others?
# ============================================================

token_category_count_df: pl.DataFrame = token_df.group_by("token").agg(
    pl.col("category").n_unique().alias("category_count")
)

unique_tokens_df: pl.DataFrame = (
    token_df.join(token_category_count_df, on="token", how="left")
    .filter(pl.col("category_count") == 1)
    .group_by(["category", "token"])
    .len()
    .sort(["category", "token"])
)

print("\nTokens unique to one category:")
print(unique_tokens_df)

# ============================================================
# Section 11. NEW QUESTION:
# What are the most common bigrams within each category?
# ============================================================

category_bigram_freq_df: pl.DataFrame = (
    bigram_df.group_by(["category", "bigram"])
    .len()
    .sort(["category", "len"], descending=[False, True])
)

print("\nTop bigrams by category:")
for category in bigram_df["category"].unique().to_list():
    print(f"\n{category.upper()} top bigrams:")
    print(category_bigram_freq_df.filter(pl.col("category") == category).head(5))

# ============================================================
# Section 12. NEW QUESTION:
# How dense is the vocabulary in each category?
# ============================================================

vocab_density_rows: list[dict[str, float | int | str]] = []

for category in token_df["category"].unique().to_list():
    subset_tokens = token_df.filter(pl.col("category") == category)["token"].to_list()
    total_tokens = len(subset_tokens)
    unique_tokens = len(set(subset_tokens))
    lexical_density = unique_tokens / total_tokens if total_tokens > 0 else 0.0

    vocab_density_rows.append(
        {
            "category": category,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "lexical_density": round(lexical_density, 4),
        }
    )

vocab_density_df: pl.DataFrame = pl.DataFrame(vocab_density_rows)

print("\nVocabulary density by category:")
print(vocab_density_df)

# ============================================================
# Section 13. NEW QUESTION:
# Which context words appear most often around a selected target token?
# ============================================================

target_word: str = "dog"
target_context_counts = Counter(co_occurrence_dict[target_word])

target_context_df: pl.DataFrame = pl.DataFrame(
    {
        "context_word": list(target_context_counts.keys()),
        "count": list(target_context_counts.values()),
    }
).sort("count", descending=True)

print(f"\nMost common context words around '{target_word}':")
print(target_context_df.head(10))

# ============================================================
# Section 14. Visual 1 - Top Tokens in Dog Category
# ============================================================

print("IMPORTANT: Close each chart window to continue execution.")

dog_df = category_freq_df.filter(pl.col("category") == "dog").head(5)

plt.figure(figsize=(8, 4))
plt.bar(dog_df["token"], dog_df["len"])
ax = plt.gca()
ax.tick_params(axis="x", labelrotation=45)
plt.title("Top Tokens in Dog Category")
plt.xlabel("Token")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# Section 15. Visual 2 - Total Tokens by Category
# ============================================================

plt.figure(figsize=(8, 4))
plt.bar(category_token_totals_df["category"], category_token_totals_df["len"])
plt.title("Total Tokens by Category")
plt.xlabel("Category")
plt.ylabel("Total Tokens")
plt.tight_layout()
plt.show()

# ============================================================
# Section 16. Visual 3 - Lexical Density by Category
# ============================================================

plt.figure(figsize=(8, 4))
plt.bar(vocab_density_df["category"], vocab_density_df["lexical_density"])
plt.title("Lexical Density by Category")
plt.xlabel("Category")
plt.ylabel("Lexical Density")
plt.tight_layout()
plt.show()

# ============================================================
# Section 17. Visual 4 - Most Common Bigrams Overall
# ============================================================

top_bigram_plot_df = bigram_freq_df.head(8)

plt.figure(figsize=(10, 4))
plt.bar(top_bigram_plot_df["bigram"], top_bigram_plot_df["len"])
ax = plt.gca()
ax.tick_params(axis="x", labelrotation=45)
plt.title("Most Common Bigrams Overall")
plt.xlabel("Bigram")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# Section 18. Visual 5 - Context Words Around Target Token
# ============================================================

top_context_plot_df = target_context_df.head(8)

plt.figure(figsize=(10, 4))
plt.bar(top_context_plot_df["context_word"], top_context_plot_df["count"])
ax = plt.gca()
ax.tick_params(axis="x", labelrotation=45)
plt.title(f"Most Common Context Words Around '{target_word}'")
plt.xlabel("Context Word")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================================
# Section 19. Interpret Results and Identify Patterns
# ============================================================

print("\nCASE GENERAL OBSERVATIONS:")
print("- Some categories contain more total tokens than others.")
print("- Unique tokens help distinguish one category from another.")
print("- Bigrams reveal local phrase structure inside categories.")
print("- Lexical density shows how varied the vocabulary is within each category.")
print("- Context words help explain how a target token is used in nearby text.")

print("\nYOURNAME SPECIFIC OBSERVATIONS:")
print("TODO: Add your own observations here.")

# ============================================================
# END
# ============================================================

LOG.info("========================")
LOG.info("Pipeline executed successfully!")
LOG.info("========================")
LOG.info("END main()")
