# preprocessing/jigsaw_preprocess.py

import os
import pandas as pd
from clean_text import clean_text


# -------------------------------
# Configuration
# -------------------------------

INPUT_PATH = "data/jigsaw-toxic-comment-classification-challenge/train.csv"
OUTPUT_DIR = "data/jigsaw-toxic-comment-classification-challenge/processed"
OUTPUT_FILE = "jigsaw_cleaned.csv"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# -------------------------------
# Main preprocessing function
# -------------------------------

def preprocess_jigsaw(input_path: str) -> pd.DataFrame:
    """
    Load Jigsaw dataset, clean text, and convert multi-labels to binary hate label.
    """
    print("Loading Jigsaw dataset...")
    df = pd.read_csv(input_path)

    # Safety check
    assert "comment_text" in df.columns, "comment_text column missing"

    print("Cleaning text...")
    df["clean_text"] = df["comment_text"].apply(clean_text)

    print("Converting multi-labels to binary hate label...")
    df["binary_label"] = df[LABEL_COLUMNS].max(axis=1)

    return df[["clean_text", "binary_label"]]


# -------------------------------
# Script execution
# -------------------------------

if __name__ == "__main__":

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processed_df = preprocess_jigsaw(INPUT_PATH)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    processed_df.to_csv(output_path, index=False)

    # -------------------------------
    # Review-friendly statistics
    # -------------------------------
    print("\nPreprocessing completed.")
    print(f"Saved cleaned dataset to: {output_path}")
    print("\nDataset statistics:")
    print(processed_df["binary_label"].value_counts())
    print("\nSample cleaned text:")
    print(processed_df.sample(3))
