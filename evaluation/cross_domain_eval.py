# evaluation/cross_domain_eval.py

import os
import sys
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.clean_text import clean_text


# -------------------------------
# Configuration
# -------------------------------

MODEL_NAME = "roberta-base"

# Change this path to evaluate different models
MODEL_PATH = "checkpoints/roberta_jigsaw_fgsm.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAB_PATH = "data/gab_reddit_hate_speech_dataset-main/gab_dataset.csv"
REDDIT_PATH = "data/gab_reddit_hate_speech_dataset-main/reddit_dataset.csv"

MAX_LENGTH = 128


# -------------------------------
# Load Model
# -------------------------------

print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -------------------------------
# Inference Function
# -------------------------------

def run_inference(csv_path, dataset_name):

    print(f"\nRunning cross-domain evaluation on {dataset_name}")

    df = pd.read_csv(csv_path)

    # detect text column
    text_col = "text" if "text" in df.columns else df.columns[0]

    # detect label column
    label_col = "label" if "label" in df.columns else df.columns[-1]

    texts = df[text_col].astype(str).apply(clean_text).tolist()
    labels = df[label_col].astype(int).tolist()

    preds = []

    for text in texts[:500]:  # limit evaluation for speed

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(
                input_ids=encoding["input_ids"].to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE)
            )

        pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)

    labels = labels[:500]

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels, preds))


# -------------------------------
# Run Evaluation
# -------------------------------

run_inference(GAB_PATH, "Gab")
run_inference(REDDIT_PATH, "Reddit")