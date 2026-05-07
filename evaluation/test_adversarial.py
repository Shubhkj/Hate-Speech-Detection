import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Ensure project root is on sys.path for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.clean_text import clean_text
from adversarial.perturbations import (
    leetspeak,
    char_repeat,
    char_delete,
    char_insert,
    random_spacing,
    apply_random_attack
)

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "GroNLP/hateBERT"
MODEL_PATH = "checkpoints/hatebert_jigsaw_robust.pt"

TEST_PATH = "data/jigsaw-toxic-comment-classification-challenge/test.csv"
TEST_LABELS_PATH = "data/jigsaw-toxic-comment-classification-challenge/test_labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
BATCH_SIZE = 32
THRESHOLD = 0.92


# -----------------------------
# Load Model
# -----------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -----------------------------
# Load Data
# -----------------------------
print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)
labels_df = pd.read_csv(TEST_LABELS_PATH)

df = test_df.merge(labels_df, on="id")

label_cols = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

df = df[(df[label_cols] != -1).all(axis=1)]
df["binary_label"] = df[label_cols].max(axis=1)
df["clean_text"] = df["comment_text"].apply(clean_text)

texts = df["clean_text"].tolist()
labels = df["binary_label"].tolist()


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate(texts, labels, description="Clean"):
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]

            encoding = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(DEVICE)
            attention_mask = encoding["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    preds = (all_probs >= THRESHOLD).astype(int)

    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    print(f"\n=== {description} Evaluation ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(classification_report(labels, preds))


# -----------------------------
# Run Evaluations
# -----------------------------

# Clean
evaluate(texts, labels, "Clean")

# Leetspeak
leet_texts = [leetspeak(t) for t in texts]
evaluate(leet_texts, labels, "Leetspeak")

# Character Noise (combined simple)
noise_texts = [char_insert(char_delete(t)) for t in texts]
evaluate(noise_texts, labels, "Char Noise")

# Random Combined Attack
random_texts = [apply_random_attack(t) for t in texts]
evaluate(random_texts, labels, "Random Combined Attack")