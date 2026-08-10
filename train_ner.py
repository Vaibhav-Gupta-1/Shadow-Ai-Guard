"""
Shadow AI Guard - PII/NER Training Pipeline
Trains a DistilBERT-based Named Entity Recognition model on the
ai4privacy/pii-masking-200k dataset to detect PII in text before
it gets sent to an external LLM (ChatGPT, Claude, Gemini, etc.)

Run this in Google Colab (recommended, free GPU) or locally with a GPU.
Before running: pip install datasets transformers seqeval accelerate
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# -----------------------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------------------
print("Loading dataset...")
ds = load_dataset("ai4privacy/pii-masking-200k")
print(ds)
print("\nSample row:")
print(ds["train"][0])

# IMPORTANT: Check the actual field names for your dataset version.
# The dataset has changed field names across versions. Run this once
# to confirm before trusting the code below:
print("\nAvailable fields:", ds["train"].features)

# -----------------------------------------------------------------
# 2. Tokenizer + label alignment
# -----------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def align_labels(example):
    """
    Converts character-level PII spans into token-level BIO labels
    (B-EMAIL, I-EMAIL, O, etc.) that the model can learn from.

    NOTE: Adjust 'text' and 'spans' keys below to match whatever field
    names printed out in step 1 (e.g. it might be 'source_text' and
    'privacy_mask', or 'unmasked_text' and 'span_labels' depending on
    dataset version).
    """
    text = example["source_text"]
    spans = example["privacy_mask"]  # list of {start, end, label}

    tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True)
    labels = ["O"] * len(tokenized["input_ids"])

    for i, (start, end) in enumerate(tokenized["offset_mapping"]):
        if start == end:
            continue
        for span in spans:
            if start >= span["start"] and end <= span["end"]:
                prefix = "B-" if start == span["start"] else "I-"
                labels[i] = prefix + span["label"]

    tokenized["labels"] = labels
    return tokenized


print("\nTokenizing and aligning labels (this may take a few minutes)...")
tokenized_ds = ds.map(align_labels, batched=False)

# -----------------------------------------------------------------
# 3. Build label list -> id mappings
# -----------------------------------------------------------------
label_list = sorted(set(
    label for ex in tokenized_ds["train"]["labels"] for label in ex
))
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}
print(f"\nFound {len(label_list)} label types: {label_list}")


def convert_labels(example):
    example["labels"] = [label2id[l] for l in example["labels"]]
    return example


tokenized_ds = tokenized_ds.map(convert_labels)

# -----------------------------------------------------------------
# 4. Model + training setup
# -----------------------------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./shadow-ai-guard-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -----------------------------------------------------------------
# 5. Train
# -----------------------------------------------------------------
print("\nStarting training...")
trainer.train()

# -----------------------------------------------------------------
# 6. Save model
# -----------------------------------------------------------------
trainer.save_model("./shadow-ai-guard-ner")
tokenizer.save_pretrained("./shadow-ai-guard-ner")
print("\nModel saved to ./shadow-ai-guard-ner")

# -----------------------------------------------------------------
# 7. Quick test
# -----------------------------------------------------------------
from transformers import pipeline

ner = pipeline(
    "ner",
    model="./shadow-ai-guard-ner",
    tokenizer=tokenizer,
    aggregation_strategy="simple",
)

test_sentence = "Hi, my name is Priya Sharma and my SSN is 123-45-6789"
print(f"\nTest input: {test_sentence}")
print("Detected entities:")
print(ner(test_sentence))
