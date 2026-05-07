# datasets/base_dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class BaseHateDataset(Dataset):
    """
    PyTorch Dataset for binary hate speech detection.
    Used for Jigsaw, Gab, and Reddit datasets.
    """

    def __init__(
        self,
        csv_path: str,
        model_name: str = "roberta-base",
        max_length: int = 128
    ):
        """
        Args:
            csv_path: Path to cleaned CSV file
            model_name: HuggingFace transformer model name
            max_length: Maximum token length
        """
        self.data = pd.read_csv(csv_path)
        self.texts = self.data["clean_text"].tolist()
        self.labels = self.data["binary_label"].tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns one sample as a dictionary of tensors
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "raw_text": text  # ✅ added this line
        }
    

# -------------------------------
# Debug / quick test
# -------------------------------
if __name__ == "__main__":
    dataset = BaseHateDataset(
        csv_path="data/jigsaw-toxic-comment-classification-challenge/processed/jigsaw_cleaned.csv"
    )

    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Label:", sample["labels"])
