# evaluation/test_multitask_on_jigsaw.py

import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_model import MultiTaskHateModel
from preprocessing.clean_text import clean_text


# -------------------------------
# Configuration
# -------------------------------

MODEL_PATH = "checkpoints/joint_multitask_model.pt"
MODEL_NAME = "roberta-base"

TEST_PATH = "data/jigsaw-toxic-comment-classification-challenge/test.csv"
TEST_LABELS_PATH = "data/jigsaw-toxic-comment-classification-challenge/test_labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
BATCH_SIZE = 32


# -------------------------------
# Load Model
# -------------------------------

print("Loading multi-task model...")

# You must match num_hate_types used during training
NUM_HATE_TYPES =  len(pd.read_csv(
    "data/HateXplain-master/Data/processed/hatexplain_cleaned.csv"
)["hate_type"].unique())

model = MultiTaskHateModel(
    model_name=MODEL_NAME,
    num_hate_types=NUM_HATE_TYPES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -------------------------------
# Load & Prepare Jigsaw Test Data
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

# Remove hidden competition samples (-1)
df = df[(df[label_cols] != -1).all(axis=1)]

df["binary_label"] = df[label_cols].max(axis=1)
df["clean_text"] = df["comment_text"].apply(clean_text)

texts = df["clean_text"].tolist()
true_labels = df["binary_label"].tolist()


# -------------------------------
# Run Inference (Hate Head Only)
# -------------------------------

print("Running inference...")

all_preds = []

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

        hate_logits = outputs["hate_logits"]
        preds = torch.argmax(hate_logits, dim=1)

        all_preds.extend(preds.cpu().numpy())


# -------------------------------
# Metrics
# -------------------------------

accuracy = accuracy_score(true_labels, all_preds)
f1 = f1_score(true_labels, all_preds)

print("\nMulti-Task Model on Jigsaw Test:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(true_labels, all_preds))
