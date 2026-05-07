import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.hatexplain_dataset import HateXplainDataset
from models.multitask_model import MultiTaskHateModel


# -------------------------------
# Config
# -------------------------------

BASELINE_PATH = "checkpoints/roberta_jigsaw.pt"
DATA_PATH = "data/HateXplain-master/Data/processed/hatexplain_cleaned.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 1e-5  # small LR


# -------------------------------
# Load Dataset
# -------------------------------

dataset = HateXplainDataset(DATA_PATH)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

num_hate_types = len(dataset.hate_type_map)


# -------------------------------
# Load Model
# -------------------------------

model = MultiTaskHateModel(
    model_name="roberta-base",
    num_hate_types=num_hate_types
)

# Load Jigsaw-trained encoder
model.load_encoder_from_baseline(BASELINE_PATH)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# Training
# -------------------------------

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            hate_label=batch["hate_label"].to(DEVICE),
            hate_type=batch["hate_type"].to(DEVICE),
            target=batch["target"].to(DEVICE)
        )

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Train Loss: {total_loss / len(train_loader):.4f}")


# Save model
torch.save(model.state_dict(), "checkpoints/multitask_finetuned.pt")

print("\nFine-tuned multi-task model saved.")
