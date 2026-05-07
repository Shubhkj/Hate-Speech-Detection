# evaluation/test_base.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    classification_report
)

from preprocessing.clean_text import clean_text


# -------------------------------
# Config
# -------------------------------

MODEL_NAME = "GroNLP/hateBERT"
MODEL_PATH = "checkpoints/hatebert_jigsaw.pt"

TEST_PATH = "data/jigsaw-toxic-comment-classification-challenge/test.csv"
TEST_LABELS_PATH = "data/jigsaw-toxic-comment-classification-challenge/test_labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
BATCH_SIZE = 32


# -------------------------------
# Load Model
# -------------------------------

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -------------------------------
# Load Test Data
# -------------------------------

print("Loading Jigsaw test dataset...")

test_df = pd.read_csv(TEST_PATH)
labels_df = pd.read_csv(TEST_LABELS_PATH)

df = test_df.merge(labels_df, on="id")

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

df = df[(df[label_cols] != -1).all(axis=1)]

df["binary_label"] = df[label_cols].max(axis=1)
df["clean_text"] = df["comment_text"].apply(clean_text)

texts = df["clean_text"].tolist()
true_labels = df["binary_label"].tolist()


# -------------------------------
# Inference
# -------------------------------

print("Running inference...")

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

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())

all_probs = np.array(all_probs)
true_labels = np.array(true_labels)


# -------------------------------
# Default Evaluation (IMPORTANT)
# -------------------------------

print("\n=== Default Evaluation (Threshold = 0.5) ===")

default_preds = (all_probs >= 0.5).astype(int)

accuracy = accuracy_score(true_labels, default_preds)
precision = precision_score(true_labels, default_preds)
recall = recall_score(true_labels, default_preds)
f1 = f1_score(true_labels, default_preds)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report (Default):")
print(classification_report(true_labels, default_preds))


# -------------------------------
# Threshold Sweep
# -------------------------------

thresholds = np.arange(0.80, 0.96, 0.01)

best_f1 = 0
best_threshold = 0.5

print("\nThreshold Analysis:")

for t in thresholds:
    preds = (all_probs >= t).astype(int)

    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)

    print(f"Threshold={t:.2f} | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t


print("\nBest Threshold:", best_threshold)
print("Best F1:", best_f1)


# -------------------------------
# Final Metrics at Best Threshold
# -------------------------------

final_preds = (all_probs >= best_threshold).astype(int)

print("\n=== Evaluation at Best Threshold ===")
print(classification_report(true_labels, final_preds))


# -------------------------------
# Plot Precision-Recall Curve
# -------------------------------

precision, recall, _ = precision_recall_curve(true_labels, all_probs)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()


# -------------------------------
# Plot F1 vs Threshold
# -------------------------------

f1_scores = []

for t in thresholds:
    preds = (all_probs >= t).astype(int)
    f1_scores.append(f1_score(true_labels, preds))

plt.figure()
plt.plot(thresholds, f1_scores)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 vs Threshold")
plt.grid()
plt.show()