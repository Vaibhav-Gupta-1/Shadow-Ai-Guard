import pandas as pd
import ast
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)

# 1. Determine absolute path to avoid FileNotFoundError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check potential locations for the CSV file
possible_paths = [
    os.path.join(BASE_DIR, "pii_dataset.csv"),
    os.path.join(BASE_DIR, "backend", "pii_dataset.csv"),
    os.path.join(os.getcwd(), "pii_dataset.csv"),
    os.path.join(os.getcwd(), "backend", "pii_dataset.csv")
]

DATASET_FILE = None
for path in possible_paths:
    if os.path.exists(path):
        DATASET_FILE = path
        break

if not DATASET_FILE:
    raise FileNotFoundError(
        "❌ Could not locate 'pii_dataset.csv'! Please ensure you downloaded the Kaggle CSV and placed it in your project folder or backend folder."
    )

print(f"📥 Loading Kaggle PII Dataset from: {DATASET_FILE}")

# 2. Read and Parse Kaggle CSV (Using open context to prevent Windows file locks)
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    df = pd.read_csv(f)

# Parse stringified Python lists stored inside CSV columns
df['tokens'] = df['tokens'].apply(ast.literal_eval)
df['labels'] = df['labels'].apply(ast.literal_eval)

# Build Label Map dynamically based on tags in Kaggle dataset
unique_labels = sorted(list(set([label for labels in df['labels'] for label in labels])))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}

print(f"✅ Found {len(unique_labels)} unique PII entity tags: {unique_labels}")

# Convert string BIO labels into integer IDs
df['ner_tags'] = df['labels'].apply(lambda label_list: [label2id[l] for l in label_list])
dataset = Dataset.from_pandas(df[['tokens', 'ner_tags']])

# 3. Tokenizer and Subword Alignment
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # PyTorch ignores -100 during loss computation
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("🔄 Tokenizing dataset and aligning subword tags...")
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# 4. Initialize Pretrained Transformer Model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# 5. Training Setup
OUTPUT_DIR = "./shadowai_pii_model"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    learning_rate=3e-5,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

# 6. Execute Fine-Tuning
print("🚀 Starting DistilBERT Fine-Tuning on Kaggle PII Data...")
import os

# Resume automatically if a checkpoint already exists from a previous run
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(OUTPUT_DIR, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
        print(f"🔁 Resuming from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)

# 7. Save Fine-Tuned Weights and Tokenizer locally for app.py
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model successfully trained and saved to '{OUTPUT_DIR}'!")