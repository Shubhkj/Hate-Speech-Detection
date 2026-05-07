import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
from pathlib import Path

# -------------------------------
# Config
# -------------------------------

MODEL_PATH = "checkpoints/roberta_jigsaw_weighted"  # local checkpoint name (with or without .pt)
BASE_MODEL_NAME = "roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Model (CORRECT)
# -------------------------------

print("Loading model...")

project_root = Path(__file__).resolve().parent.parent
raw_model_path = Path(MODEL_PATH)
resolved_model_path = raw_model_path if raw_model_path.is_absolute() else project_root / raw_model_path

if not resolved_model_path.exists() and resolved_model_path.suffix == "":
    pt_candidate = resolved_model_path.with_suffix(".pt")
    if pt_candidate.exists():
        resolved_model_path = pt_candidate

if not resolved_model_path.exists():
    raise FileNotFoundError(
        f"Local checkpoint not found: {resolved_model_path}. "
        "Use a valid local .pt path or adapter directory."
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

if resolved_model_path.is_file() and resolved_model_path.suffix == ".pt":
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2
    )
    state_dict = torch.load(resolved_model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
elif resolved_model_path.is_dir():
    # Local Transformers directory export with config/tokenizer/model files.
    model = AutoModelForSequenceClassification.from_pretrained(
        str(resolved_model_path),
        num_labels=2,
        local_files_only=True
    )
else:
    raise ValueError(
        f"Unsupported checkpoint format at {resolved_model_path}. "
        "Expected a .pt file or a local model directory."
    )

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# -------------------------------
# Text Cleaning
# -------------------------------

def clean_text_input(texts):
    if isinstance(texts, str):
        return [texts]

    cleaned = []
    for t in texts:
        if isinstance(t, bytes):
            t = t.decode("utf-8", errors="ignore")
        elif not isinstance(t, str):
            t = str(t)
        cleaned.append(t)

    return cleaned


# -------------------------------
# Prediction Function
# -------------------------------

def predict(texts):
    texts = clean_text_input(texts)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()


# -------------------------------
# SHAP Explainer
# -------------------------------

print("Initializing SHAP explainer...")
explainer = shap.Explainer(predict, masker=tokenizer)


# -------------------------------
# Sample Inputs (Representative)
# -------------------------------

texts = [
    "I hate you people",
    "You are amazing",
    "This community is disgusting",
    "Go back to where you came from",
    "I h@te y0u people"  # adversarial example
]


# -------------------------------
# Generate SHAP Values
# -------------------------------

print("Generating SHAP explanations...")
shap_values = explainer(texts)


# -------------------------------
# Token Cleaning
# -------------------------------

def clean_token(token):
    token = token.replace("##", "")  # remove subword marker
    return token.strip()


# -------------------------------
# Extract Top Tokens
# -------------------------------

def get_top_tokens(shap_values, texts, top_k=5):
    all_tokens = []

    for i, text in enumerate(texts):
        tokens = shap_values.data[i]
        values = shap_values.values[i][:, 1]  # class 1 (hate)

        token_scores = list(zip(tokens, values))

        # Clean tokens
        token_scores = [
            (clean_token(tok), val)
            for tok, val in token_scores
            if tok.strip() not in ["", "[CLS]", "[SEP]", ".", ","] and len(tok.strip()) > 1
        ]

        # Sort by importance
        token_scores = sorted(token_scores, key=lambda x: abs(x[1]), reverse=True)

        # Take top tokens
        top_tokens = token_scores[:top_k]

        print(f"\nText: {text}")
        print("Top Tokens:")
        for tok, val in top_tokens:
            print(f"{tok}: {val:.3f}")

        all_tokens.extend(top_tokens)

    return all_tokens


top_tokens = get_top_tokens(shap_values, texts)


# -------------------------------
# Aggregate Across Samples
# -------------------------------

agg_scores = defaultdict(list)

for tok, val in top_tokens:
    agg_scores[tok].append(val)

final_tokens = []

for tok, vals in agg_scores.items():
    avg_score = sum(vals) / len(vals)
    final_tokens.append((tok, avg_score))

# Sort globally
final_tokens = sorted(final_tokens, key=lambda x: abs(x[1]), reverse=True)


# -------------------------------
# FINAL OUTPUT (FOR PAPER)
# -------------------------------

print("\n=== FINAL TOKENS FOR PAPER ===")
for tok, val in final_tokens[:6]:
    print(f"{tok}: {val:.3f}")