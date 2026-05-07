# training/train_multitask.py

import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.hatexplain_dataset import HateXplainDataset
from models.multitask_model import MultiTaskHateModel


# -------------------------------
# Configuration
# -------------------------------

MODEL_NAME = "roberta-base"
DATA_PATH = "data/HateXplain-master/Data/processed/hatexplain_cleaned.csv"

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Load Dataset
# -------------------------------

dataset = HateXplainDataset(DATA_PATH)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# Get number of hate types dynamically
num_hate_types = len(dataset.hate_type_map)


# -------------------------------
# Load Model
# -------------------------------

model = MultiTaskHateModel(
    model_name=MODEL_NAME,
    num_hate_types=num_hate_types
)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# Training Loop
# -------------------------------

def train_epoch():
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        hate_label = batch["hate_label"].to(DEVICE)
        hate_type = batch["hate_type"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hate_label=hate_label,
            hate_type=hate_type,
            target=target
        )

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# -------------------------------
# Evaluation Loop
# -------------------------------

def eval_epoch():
    model.eval()

    all_hate_preds = []
    all_hate_labels = []

    all_type_preds = []
    all_type_labels = []

    all_target_preds = []
    all_target_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            hate_label = batch["hate_label"].to(DEVICE)
            hate_type = batch["hate_type"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            hate_preds = torch.argmax(outputs["hate_logits"], dim=1)
            type_preds = torch.argmax(outputs["hate_type_logits"], dim=1)
            target_preds = torch.argmax(outputs["target_logits"], dim=1)

            all_hate_preds.extend(hate_preds.cpu().numpy())
            all_hate_labels.extend(hate_label.cpu().numpy())

            all_type_preds.extend(type_preds.cpu().numpy())
            all_type_labels.extend(hate_type.cpu().numpy())

            all_target_preds.extend(target_preds.cpu().numpy())
            all_target_labels.extend(target.cpu().numpy())

    hate_acc = accuracy_score(all_hate_labels, all_hate_preds)
    hate_f1 = f1_score(all_hate_labels, all_hate_preds)

    type_acc = accuracy_score(all_type_labels, all_type_preds)
    target_acc = accuracy_score(all_target_labels, all_target_preds)

    return hate_acc, hate_f1, type_acc, target_acc


# -------------------------------
# Run Training
# -------------------------------

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_epoch()
    hate_acc, hate_f1, type_acc, target_acc = eval_epoch()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Hate Accuracy: {hate_acc:.4f}")
    print(f"Hate F1: {hate_f1:.4f}")
    print(f"Hate Type Accuracy: {type_acc:.4f}")
    print(f"Target Accuracy: {target_acc:.4f}")


# -------------------------------
# Save Model
# -------------------------------

torch.save(model.state_dict(), "checkpoints/multitask_model.pt")

print("\nMulti-task model saved.")
