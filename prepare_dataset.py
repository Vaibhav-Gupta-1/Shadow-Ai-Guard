"""
Shadow AI Guard - Dataset Preparation Script

Downloads ai4privacy/pii-masking-200k from HuggingFace and converts it
into the pii_dataset.csv format expected by train_model.py:
    tokens  -> stringified list of words, e.g. "['My', 'name', 'is', 'John']"
    labels  -> stringified list of BIO tags, e.g. "['O', 'O', 'O', 'B-FIRSTNAME']"

Run this BEFORE train_model.py.
Before running: pip install datasets pandas
"""

import pandas as pd
from datasets import load_dataset

# -----------------------------------------------------------------
# 1. Load the dataset from HuggingFace
# -----------------------------------------------------------------
print("Downloading ai4privacy/pii-masking-200k ...")
ds = load_dataset("ai4privacy/pii-masking-200k")

# Use only the English subset to keep things simple for a college project.
# The dataset includes a 'language' or 'locale' field in some versions —
# check available columns first.
print("\nAvailable columns:", ds["train"].column_names)
print("\nSample row:")
print(ds["train"][0])

# -----------------------------------------------------------------
# 2. Convert each row into (tokens, BIO labels)
# -----------------------------------------------------------------
# The dataset provides the source text plus a list of PII spans
# (start/end character offsets + label). We need to turn that into
# word-level tokens + one BIO tag per word, which is what
# train_model.py's tokenizer alignment step expects.


def text_to_tokens_and_labels(text, spans):
    """
    Very simple whitespace tokenizer + BIO tagging.
    Good enough for a college mini-project; not as robust as a
    proper tokenizer, but keeps this script dependency-free.
    """
    tokens = []
    labels = []

    # Build a character position -> word index map by splitting on whitespace
    # while tracking each word's start/end offset in the original text.
    word_spans = []
    idx = 0
    for word in text.split():
        start = text.find(word, idx)
        end = start + len(word)
        word_spans.append((start, end))
        idx = end
        tokens.append(word)

    labels = ["O"] * len(tokens)

    for span in spans:
        span_start, span_end, span_label = span["start"], span["end"], span["label"]
        first = True
        for i, (w_start, w_end) in enumerate(word_spans):
            if w_start >= span_start and w_end <= span_end:
                labels[i] = ("B-" if first else "I-") + span_label
                first = False

    return tokens, labels


print("\nConverting rows to token/BIO format (this may take a few minutes)...")

all_tokens = []
all_labels = []

for row in ds["train"]:
    text = row.get("source_text") or row.get("unmasked_text")
    spans = row.get("privacy_mask") or row.get("span_labels")

    if not text or not spans:
        continue  # skip rows that don't match expected fields

    tokens, labels = text_to_tokens_and_labels(text, spans)
    if len(tokens) == 0:
        continue

    all_tokens.append(tokens)
    all_labels.append(labels)

print(f"Converted {len(all_tokens)} rows.")

# -----------------------------------------------------------------
# 3. Save as pii_dataset.csv, in the exact format train_model.py expects
# -----------------------------------------------------------------
df = pd.DataFrame({
    "tokens": [str(t) for t in all_tokens],   # stringified list, e.g. "['My','name']"
    "labels": [str(l) for l in all_labels],
})

OUTPUT_FILE = "pii_dataset.csv"
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved {len(df)} rows to {OUTPUT_FILE}")
print("Place this file in the same folder as train_model.py (or in backend/), then run train_model.py.")