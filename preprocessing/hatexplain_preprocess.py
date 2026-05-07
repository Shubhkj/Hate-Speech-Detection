# preprocessing/hatexplain_preprocess.py

import json
import os
import sys
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.clean_text import clean_text


# -------------------------------------------------
# Configuration
# -------------------------------------------------

INPUT_PATH = "data/HateXplain-master/Data/dataset.json"
OUTPUT_DIR = "data/HateXplain-master/Data/processed"
OUTPUT_FILE = "hatexplain_cleaned.csv"


# -------------------------------------------------
# Target → Hate Type Mapping (normalized)
# -------------------------------------------------

TARGET_TO_TYPE = {
    "women": "gender",
    "men": "gender",
    "female": "gender",
    "male": "gender",

    "black people": "race",
    "white people": "race",
    "asians": "race",

    "muslims": "religion",
    "jews": "religion",
    "christians": "religion",

    "immigrants": "ethnicity",

    "gay people": "sexual_orientation",
    "lesbians": "sexual_orientation",
    "lgbtq": "sexual_orientation",

    "politicians": "politics",
    "nazis": "politics"
}


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def majority_vote(values):
    """Return the most common value."""
    return Counter(values).most_common(1)[0][0]


def extract_text(post):
    """Reconstruct sentence from token list."""
    return " ".join(post["post_tokens"])


def get_hate_label(annotators):
    """
    HateXplain labels may be either:
    - int: 0 (normal), 1 (offensive), 2 (hateful)
    - str: "normal", "offensive", "hateful"

    If ANY annotator marks offensive or hateful → hate = 1
    """
    for ann in annotators:
        label = ann.get("label")

        # Case 1: numeric labels
        if isinstance(label, int) and label in [1, 2]:
            return 1

        # Case 2: string labels
        if isinstance(label, str) and label.lower() in ["offensive", "hateful"]:
            return 1

    return 0


def get_hate_type(annotators):
    targets = []

    for ann in annotators:
        for t in ann.get("target", []):
            targets.append(t.lower())

    if not targets:
        return "none"

    most_common_target = majority_vote(targets)

    for key, hate_type in TARGET_TO_TYPE.items():
        if key in most_common_target:
            return hate_type

    return "other"


def get_target_group(annotators):
    """
    HateXplain targets are community-based.
    """
    for ann in annotators:
        if ann.get("target"):
            return "group"
    return "none"


# -------------------------------------------------
# Main Preprocessing Function
# -------------------------------------------------

def preprocess_hatexplain(input_path):
    print("Loading HateXplain dataset...")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for post_id, post in data.items():
        text = clean_text(extract_text(post))
        annotators = post["annotators"]

        records.append({
            "text": text,
            "hate": get_hate_label(annotators),
            "hate_type": get_hate_type(annotators),
            "target": get_target_group(annotators)
        })

    return pd.DataFrame(records)


# -------------------------------------------------
# Script Execution
# -------------------------------------------------

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = preprocess_hatexplain(INPUT_PATH)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)

    print(f"\nSaved processed HateXplain dataset to:\n{output_path}")

    print("\nHate label distribution:")
    print(df["hate"].value_counts())

    print("\nHate type distribution:")
    print(df["hate_type"].value_counts())

    print("\nSample rows:")
    print(df.sample(5))
