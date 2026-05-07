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
from adversarial.perturbations import apply_random_attack


# -------------------------------
# Configuration
# -------------------------------

MODEL_NAME = "GroNLP/hateBERT"
BASELINE_PATH = "checkpoints/hatebert_jigsaw.pt"
DATA_PATH = "data/jigsaw-toxic-comment-classification-challenge/processed/jigsaw_cleaned.csv"

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-5  # lower LR for fine-tuning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print device information
print(f"\n{'='*50}")
print(f"DEVICE INFORMATION")
print(f"{'='*50}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Using Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"{'='*50}\n")

# -------------------------------
# Load Dataset
# -------------------------------

print("Loading dataset...")
dataset = BaseHateDataset(csv_path=DATA_PATH, model_name=MODEL_NAME)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# -------------------------------
# Load Baseline Model
# -------------------------------

print("Loading baseline model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
model.to(DEVICE)
print(f"Model moved to: {next(model.parameters()).device}")

# Initialize tokenizer once (not in batch loop)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# Adversarial Fine-Tuning
# -------------------------------

def adversarial_train_epoch(model, loader):
    model.train()
    total_loss = 0
    first_batch = True

    for batch in tqdm(loader, desc="Adversarial Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        # Debug: Print device info for first batch
        if first_batch:
            print(f"\nFirst batch tensor devices:")
            print(f"  input_ids device: {input_ids.device}")
            print(f"  attention_mask device: {attention_mask.device}")
            print(f"  labels device: {labels.device}")
            first_batch = False

        # 50% perturbation
        batch_size = input_ids.size(0)
        half = batch_size // 2

        texts = batch["raw_text"]  # must exist in dataset

        perturbed_texts = []
        for i in range(half):
            perturbed_texts.append(apply_random_attack(texts[i]))

        # Re-tokenize perturbed texts
        perturbed_encoding = tokenizer(
            perturbed_texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        perturbed_input_ids = perturbed_encoding["input_ids"].to(DEVICE)
        perturbed_attention_mask = perturbed_encoding["attention_mask"].to(DEVICE)

        # Combine clean + perturbed
        combined_input_ids = torch.cat([input_ids[half:], perturbed_input_ids], dim=0)
        combined_attention_mask = torch.cat([attention_mask[half:], perturbed_attention_mask], dim=0)
        combined_labels = torch.cat([labels[half:], labels[:half]], dim=0)

        outputs = model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )

        loss = criterion(outputs.logits, combined_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------
# Run Training
# -------------------------------

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    avg_loss = adversarial_train_epoch(model, train_loader)
    print(f"Epoch Loss: {avg_loss:.4f}")


# -------------------------------
# Save Robust Model
# -------------------------------

ROBUST_PATH = "checkpoints/hatebert_jigsaw_robust.pt"
torch.save(model.state_dict(), ROBUST_PATH)

print(f"\nRobust model saved to {ROBUST_PATH}")