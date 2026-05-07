#!/usr/bin/env python
# Quick test script
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

print("\nLoading transformers...")
from transformers import AutoModelForSequenceClassification
print("Loading RoBERTa model...")
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✓ Model loaded on {device}")

print("\nLoading dataset...")
import sys
sys.path.insert(0, 'D:\\Major Code')
from data_loaders.base_dataset import BaseHateDataset
dataset = BaseHateDataset(csv_path="Data/jigsaw-toxic-comment-classification-challenge/processed/jigsaw_cleaned.csv")
print(f"✓ Dataset loaded: {len(dataset)} samples")
print("All systems ready!")
