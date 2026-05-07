# training/train_base.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from data_loaders.base_dataset import BaseHateDataset


# -------------------------------
# Configuration
# -------------------------------

MODEL_NAME = "GroNLP/hateBERT"
DATA_PATH = "data/jigsaw-toxic-comment-classification-challenge/processed/jigsaw_cleaned.csv"

BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Load Dataset
# -------------------------------

print("Loading dataset...")
dataset = BaseHateDataset(csv_path=DATA_PATH, model_name=MODEL_NAME)
print(f"Using full dataset: {len(dataset)} samples")

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# -------------------------------
# Load Model
# -------------------------------

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)

# Class weights (handle imbalance)
class_weights = torch.tensor([1.0, 3.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# Training Loop
# -------------------------------

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_binary = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, accuracy, f1_binary


# -------------------------------
# Run Training
# -------------------------------

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc, val_f1 = eval_epoch(model, val_loader)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val   Loss: {val_loss:.4f}")
    print(f"Val   Accuracy: {val_acc:.4f}")
    print(f"Val   F1 (Hate): {val_f1:.4f}")


# -------------------------------
# Save Model
# -------------------------------

output_path = "checkpoints/hatebert_jigsaw.pt"
torch.save(model.state_dict(), output_path)
print(f"\nModel saved to {output_path}")