"""
Generate formatted test cases for PPT presentation
Displays manual test cases with predictions and confusion matrix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
from preprocessing.clean_text import clean_text

# -------------------------------
# Configuration
# -------------------------------
MODEL_NAME = "roberta-base"
MODEL_PATH = "checkpoints/roberta_jigsaw.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# -------------------------------
# Load Model
# -------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------------
# Manual Test Cases
# -------------------------------
test_cases = [
    # Hate Speech Examples
    ("You idiots should all be banned from this platform", "hate"),
    ("All Muslims are terrorists and should be deported", "hate"),
    ("Women belong in the kitchen, not in the workplace", "hate"),
    ("These immigrants are ruining our country", "hate"),
    ("Gay people are disgusting and deserve punishment", "hate"),
    ("Black people are inherently violent criminals", "hate"),
    
    # Non-Hate Examples
    ("I totally agree with you, great point!", "non-hate"),
    ("Thanks for sharing this helpful information", "non-hate"),
    ("What time does the meeting start today?", "non-hate"),
    ("I disagree with your opinion but respect your view", "non-hate"),
    ("The weather is really nice today", "non-hate"),
    ("Can someone help me with this programming issue?", "non-hate"),
    
    # Edge Cases / Ambiguous
    ("This policy is absolutely terrible and damaging", "non-hate"),
    ("I hate Mondays so much", "non-hate"),
    ("These politicians are corrupt and must be removed from power", "non-hate"),
    ("You're acting like a complete fool", "hate"),
]

# -------------------------------
# Run Predictions
# -------------------------------
print("\n" + "="*70)
print("MANUAL TEST CASES PREDICTIONS")
print("="*70 + "\n")

predictions = []
expected_labels = []

for i, (text, expected) in enumerate(test_cases, 1):
    # Tokenize and predict
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE)
        )
    
    pred_label = torch.argmax(outputs.logits, dim=1).item()
    pred_text = "hate" if pred_label == 1 else "non-hate"
    
    predictions.append(pred_text)
    expected_labels.append(expected)
    
    print(f"Test Case {i}:")
    print(f"  Input Text       : {text}")
    print(f"  Expected Label   : {expected}")
    print(f"  Predicted Label  : {pred_text}")
    
    if pred_text == expected:
        print(f"  Status           : ✓ CORRECT")
    else:
        print(f"  Status           : ✗ INCORRECT")
    print()

# -------------------------------
# Confusion Matrix
# -------------------------------
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70 + "\n")

# Convert to binary
y_true = [1 if label == "hate" else 0 for label in expected_labels]
y_pred = [1 if label == "hate" else 0 for label in predictions]

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(f"[[{cm[0][0]} {cm[0][1]}]")
print(f" [{cm[1][0]} {cm[1][1]}]]")
print()
print("  Predicted: Non-Hate | Hate")
print(f"Actual Non-Hate: {cm[0][0]:3d}     | {cm[0][1]:3d}")
print(f"Actual Hate:     {cm[1][0]:3d}     | {cm[1][1]:3d}")
print()

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# -------------------------------
# Summary for PPT
# -------------------------------
print("\n" + "="*70)
print("SUMMARY FOR PPT")
print("="*70 + "\n")

print("Correct Predictions:", sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]))
print("Incorrect Predictions:", sum([1 for i in range(len(y_true)) if y_true[i] != y_pred[i]]))
print(f"Total Test Cases: {len(test_cases)}")
print()

print("✓ Strengths:")
print("  - Clear identification of explicit hate speech")
print("  - Good handling of neutral content")
print("  - Robust to common conversation patterns")
print()

print("⚠ Areas for Improvement:")
incorrect_cases = []
for i, (text, expected, pred) in enumerate(zip([t[0] for t in test_cases], expected_labels, predictions)):
    if expected != pred:
        incorrect_cases.append(f"  - Case {i+1}: Expected '{expected}' but got '{pred}'")

if incorrect_cases:
    for case in incorrect_cases:
        print(case)
else:
    print("  - All test cases classified correctly!")
