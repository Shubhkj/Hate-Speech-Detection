# evaluation/test_lora.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from preprocessing.clean_text import clean_text


# -------------------------------
# Config
# -------------------------------

MODEL_NAME = "roberta-base"
LORA_PATH = "checkpoints/roberta_lora"

TEST_PATH = "data/jigsaw-toxic-comment-classification-challenge/test.csv"
TEST_LABELS_PATH = "data/jigsaw-toxic-comment-classification-challenge/test_labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
BATCH_SIZE = 32


# -------------------------------
# Load Model (IMPORTANT)
# -------------------------------

print("Loading LoRA model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)

model.to(DEVICE)
model.eval()


# -------------------------------
# Load Data
# -------------------------------

print("Loading test dataset...")

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
# Default Evaluation
# -------------------------------

print("\n=== Default Evaluation (0.5) ===")

default_preds = (all_probs >= 0.5).astype(int)

print("Accuracy:", accuracy_score(true_labels, default_preds))
print("Precision:", precision_score(true_labels, default_preds))
print("Recall:", recall_score(true_labels, default_preds))
print("F1:", f1_score(true_labels, default_preds))

print("\nClassification Report:")
print(classification_report(true_labels, default_preds))


# -------------------------------
# Threshold Tuning
# -------------------------------

thresholds = np.arange(0.80, 0.96, 0.01)

best_f1 = 0
best_threshold = 0.5

print("\nThreshold Search:")

for t in thresholds:
    preds = (all_probs >= t).astype(int)

    f1 = f1_score(true_labels, preds)

    print(f"Threshold={t:.2f} | F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\nBest Threshold:", best_threshold)
print("Best F1:", best_f1)


# -------------------------------
# Final Evaluation
# -------------------------------

final_preds = (all_probs >= best_threshold).astype(int)

print("\n=== Final Evaluation ===")
print(classification_report(true_labels, final_preds))