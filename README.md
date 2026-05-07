# Hate Speech Detection

A research-oriented Python project for hate speech detection using transformer models, multitask learning, adversarial robustness, and explainability methods.

## What this project includes

- Base binary hate speech classification pipelines
- Multitask learning for hate label, hate type, and target prediction
- Adversarial components (including perturbation utilities and FGSM training scripts)
- Evaluation modules for robustness, fairness, and cross-domain testing
- Explainability utilities (Integrated Gradients and SHAP)
- Training and experiment scripts for RoBERTa/HateBERT variants

## Repository layout

- `adversarial/`: perturbation and robustness-related logic
- `data_loaders/`: dataset wrappers and tokenization loaders
- `evaluation/`: metric and test scripts
- `explainability/`: SHAP and Integrated Gradients scripts
- `models/`: encoder, context module, and multitask model definitions
- `preprocessing/`: cleaning and dataset preprocessing scripts
- `training/`: model training and fine-tuning scripts
- `utils/`: helpers, logging, and seed utilities
- `notebooks/`: experimentation notebooks

## Prerequisites

- Python 3.10+
- Git
- (Recommended) NVIDIA GPU with CUDA for faster training

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install PyTorch separately depending on your platform/CUDA version:

```bash
# Example only - choose the right command from pytorch.org
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick environment checks

```bash
python test_cuda.py
python test_setup.py
```

## Datasets

This project uses multiple publicly available hate speech datasets. Large datasets and checkpoints are ignored in version control.

### Jigsaw Toxic Comment Classification Challenge

- **Source**: Kaggle (https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- **Description**: Multi-label toxic comment classification with 6 toxicity types
- **Usage**: Binary hate speech detection; used in `training/train_base.py`
- **Local path**: `data/jigsaw-toxic-comment-classification-challenge/`
- **Expected file**: `processed/jigsaw_cleaned.csv`

### HateXplain Dataset

- **Source**: HateXplain GitHub (https://github.com/hate-speech-cnerg/HateXplain)
- **Description**: Annotated hate speech dataset with target groups and rationale spans
- **Usage**: Multitask learning (hate label, hate type, target prediction)
- **Local path**: `data/HateXplain-master/`
- **Expected file**: `Data/processed/hatexplain_cleaned.csv`
- **Key features**: Explainability rationales for model interpretation

### GAB & Reddit Hate Speech Dataset

- **Source**: GitHub (https://github.com/emilyemorehouse/gab_reddit_hate_speech_dataset)
- **Description**: Hate speech samples from social media platforms (GAB and Reddit)
- **Usage**: Cross-domain evaluation and robustness testing
- **Local path**: `data/gab_reddit_hate_speech_dataset-main/`
- **Files**: `gab_dataset.csv`, `reddit_dataset.csv`

### A-Benchmark Dataset

- **Source**: GitHub (https://github.com/varshakishore/An-Empirical-Benchmark-for-Learning-to-Intervene-in-Online-Hate-Speech)
- **Description**: Benchmark dataset for hate speech intervention and mitigation
- **Usage**: Evaluation and fairness testing
- **Local path**: `data/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech-master/`

### Setting up datasets

1. Download each dataset from its source (links above)
2. Extract into the `data/` directory, preserving folder names
3. Run preprocessing scripts to generate cleaned CSVs:
   - `preprocessing/jigsaw_preprocess.py`
   - `preprocessing/hatexplain_preprocess.py`
   - `preprocessing/gab_reddit_preprocess.py`

Before running training/evaluation, ensure processed CSV files exist at paths referenced in each script (e.g., `training/train_base.py` and `training/train_multitask.py`).

## Training examples

Base classifier:

```bash
python training/train_base.py
```

Multitask model:

```bash
python training/train_multitask.py
```

Additional training variants are available in:

- `training/train_joint_multitask.py`
- `training/train_lora.py`
- `training/fine_tune_multitask.py`
- `training/adversarial_training.py`
- `training/fgsm_training.py`

## Evaluation examples

```bash
python evaluation/test_base.py
python evaluation/test_multitask_on_jigsaw.py
python evaluation/test_adversarial.py
python evaluation/test_lora.py
```

## Explainability

Use:

- `explainability/integrated_gradients.py`
- `explainability/shap_explain.py`

SHAP tip: normalize incoming text batches to `list[str]` before tokenization in prediction wrappers to avoid input type errors when SHAP passes masked/object-form inputs.

## Troubleshooting

- If CUDA is unavailable, verify GPU drivers and install the correct CUDA-enabled PyTorch build.
- If model checkpoints fail to load, confirm file format (`.pt` state dict vs Hugging Face directory format).
- If imports fail, run commands from the repository root.

## Current status

This codebase is structured as a research/workbench repository with multiple experiment scripts rather than a single production CLI entrypoint.

The root `main.py` is currently empty; use script-level entrypoints in `training/`, `evaluation/`, and `explainability/`.

## License

Add a license file if you plan to distribute or reuse this project publicly.
