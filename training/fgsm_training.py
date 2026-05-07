import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm

from data_loaders.base_dataset import BaseHateDataset


# -------------------------------
# Configuration
# -------------------------------

MODEL_NAME = "roberta-base"
ROBUST_PATH = "checkpoints/roberta_jigsaw_robust.pt"  # 1-epoch robust model
DATA_PATH = "data/jigsaw-toxic-comment-classification-challenge/processed/jigsaw_cleaned.csv"

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-5
EPSILON = 0.1   # FGSM strength

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Load Dataset
# -------------------------------

print("Loading dataset...")
dataset = BaseHateDataset(csv_path=DATA_PATH, model_name=MODEL_NAME)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# -------------------------------
# Load Robust Model
# -------------------------------

print("Loading robust model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.load_state_dict(torch.load(ROBUST_PATH, map_location=DEVICE))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# FGSM Training
# -------------------------------

def fgsm_train_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="FGSM Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Get embeddings
        embeddings = model.roberta.embeddings.word_embeddings(input_ids).detach()
        embeddings.requires_grad = True

        # Forward pass (clean)
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )

        loss = criterion(outputs.logits, labels)
        loss.backward(retain_graph=True)

        # Compute FGSM perturbation
        grad = embeddings.grad
        perturbation = EPSILON * torch.sign(grad)

        adv_embeddings = embeddings + perturbation

        # Forward pass (adversarial)
        adv_outputs = model(
            inputs_embeds=adv_embeddings,
            attention_mask=attention_mask
        )

        adv_loss = criterion(adv_outputs.logits, labels)

        # Total loss
        total_batch_loss = loss + adv_loss

        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()

    return total_loss / len(loader)


# -------------------------------
# Run FGSM Training
# -------------------------------

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    avg_loss = fgsm_train_epoch(model, train_loader)
    print(f"Epoch Loss: {avg_loss:.4f}")


# -------------------------------
# Save FGSM Model
# -------------------------------

FGSM_PATH = "checkpoints/roberta_jigsaw_fgsm.pt"
torch.save(model.state_dict(), FGSM_PATH)

print(f"\nFGSM model saved to {FGSM_PATH}")