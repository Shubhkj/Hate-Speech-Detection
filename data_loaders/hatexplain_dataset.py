# data_loaders/hatexplain_dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HateXplainDataset(Dataset):
    """
    PyTorch Dataset for multi-task hate speech detection using HateXplain.
    Outputs:
      - hate label (binary)
      - hate type (categorical)
      - target group (binary)
    """

    def __init__(
        self,
        csv_path: str,
        model_name: str = "roberta-base",
        max_length: int = 128
    ):
        self.data = pd.read_csv(csv_path)

        self.texts = self.data["text"].astype(str).tolist()
        self.hate_labels = self.data["hate"].astype(int).tolist()
        self.hate_types = self.data["hate_type"].astype(str).tolist()
        self.targets = self.data["target"].astype(str).tolist()

        # Encode hate types as integers
        self.hate_type_map = {
            label: idx
            for idx, label in enumerate(sorted(set(self.hate_types)))
        }
        self.hate_type_ids = [
            self.hate_type_map[label] for label in self.hate_types
        ]

        # Encode target: group → 1, none → 0
        self.target_ids = [
            1 if t == "group" else 0 for t in self.targets
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

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
            "hate_label": torch.tensor(self.hate_labels[idx], dtype=torch.long),
            "hate_type": torch.tensor(self.hate_type_ids[idx], dtype=torch.long),
            "target": torch.tensor(self.target_ids[idx], dtype=torch.long)
        }


# -------------------------------
# Debug / sanity check
# -------------------------------
if __name__ == "__main__":
    dataset = HateXplainDataset(
        csv_path="data/HateXplain-master/Data/processed/hatexplain_cleaned.csv"
    )

    sample = dataset[0]

    print("Keys:", sample.keys())
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Hate label:", sample["hate_label"])
    print("Hate type ID:", sample["hate_type"])
    print("Target:", sample["target"])
