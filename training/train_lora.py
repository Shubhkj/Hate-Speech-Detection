# training/train_lora.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from data_loaders.base_dataset import BaseHateDataset


# -------------------------------
# Config
# -------------------------------

MODEL_NAME = "roberta-base"
DATA_PATH = "data/jigsaw-toxic-comment-classification-challenge/processed/jigsaw_cleaned.csv"

BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Dataset
# -------------------------------

dataset = BaseHateDataset(csv_path=DATA_PATH, model_name=MODEL_NAME)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# -------------------------------
# Model + LoRA
# -------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

model.to(DEVICE)


# -------------------------------
# Loss & Optimizer
# -------------------------------

class_weights = torch.tensor([1.0, 3.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# Training
# -------------------------------

def train_epoch():
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def eval_epoch():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return acc, f1


# -------------------------------
# Run Training
# -------------------------------

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}")

    train_loss = train_epoch()
    val_acc, val_f1 = eval_epoch()

    print(f"Loss: {train_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f}")


# -------------------------------
# Save
# -------------------------------

model.save_pretrained("checkpoints/roberta_lora")