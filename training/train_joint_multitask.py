import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer
from models.multitask_model import MultiTaskHateModel
from data_loaders.hatexplain_dataset import HateXplainDataset
from data_loaders.base_dataset import BaseHateDataset


# -------------------------------
# Config
# -------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

JIGSAW_PATH = "data/jigsaw-toxic-comment-classification-challenge/processed/jigsaw_cleaned.csv"
HATEXPLAIN_PATH = "data/HateXplain-master/Data/processed/hatexplain_cleaned.csv"

BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5

ALPHA = 0.3   # HX hate loss
BETA  = 0.1   # HX type loss
GAMMA = 0.1   # HX target loss


# -------------------------------
# Load Datasets
# -------------------------------

print("Loading datasets...")

jigsaw_dataset = BaseHateDataset(JIGSAW_PATH)
hx_dataset = HateXplainDataset(HATEXPLAIN_PATH)

jigsaw_loader = DataLoader(jigsaw_dataset, batch_size=BATCH_SIZE, shuffle=True)
hx_loader = DataLoader(hx_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_hate_types = len(hx_dataset.hate_type_map)


# -------------------------------
# Load Model
# -------------------------------

model = MultiTaskHateModel(
    model_name="roberta-base",
    num_hate_types=num_hate_types
)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# -------------------------------
# Training Loop
# -------------------------------

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()

    hx_iter = iter(hx_loader)
    total_loss = 0

    for jigsaw_batch in tqdm(jigsaw_loader):

        optimizer.zero_grad()

        # -----------------------
        # JIGSAW PRIMARY TASK
        # -----------------------
        j_input_ids = jigsaw_batch["input_ids"].to(DEVICE)
        j_attention = jigsaw_batch["attention_mask"].to(DEVICE)
        j_labels = jigsaw_batch["labels"].to(DEVICE)

        j_outputs = model(
            input_ids=j_input_ids,
            attention_mask=j_attention,
            hate_label=j_labels,
        )

        loss = j_outputs["loss"]

        # -----------------------
        # HATEXPLAIN AUXILIARY
        # -----------------------
        try:
            hx_batch = next(hx_iter)
        except StopIteration:
            hx_iter = iter(hx_loader)
            hx_batch = next(hx_iter)

        hx_outputs = model(
            input_ids=hx_batch["input_ids"].to(DEVICE),
            attention_mask=hx_batch["attention_mask"].to(DEVICE),
            hate_label=hx_batch["hate_label"].to(DEVICE),
            hate_type=hx_batch["hate_type"].to(DEVICE),
            target=hx_batch["target"].to(DEVICE)
        )

        # Weighted auxiliary loss
        hx_loss = (
            ALPHA * model.loss_hate(
                hx_outputs["hate_logits"],
                hx_batch["hate_label"].to(DEVICE)
            )
            + BETA * model.loss_hate_type(
                hx_outputs["hate_type_logits"],
                hx_batch["hate_type"].to(DEVICE)
            )
            + GAMMA * model.loss_target(
                hx_outputs["target_logits"],
                hx_batch["target"].to(DEVICE)
            )
        )

        total_batch_loss = loss + hx_loss

        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()

    print(f"Epoch Loss: {total_loss / len(jigsaw_loader):.4f}")


# -------------------------------
# Save Model
# -------------------------------

torch.save(model.state_dict(), "checkpoints/joint_multitask_model.pt")

print("\nJoint multi-task model saved.")
