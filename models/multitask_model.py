# models/multitask_model.py

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskHateModel(nn.Module):
    """
    Multi-task transformer model for:
      1. Hate detection (binary)
      2. Hate type classification (multi-class)
      3. Target group prediction (binary)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_hate_types: int = 5
    ):
        super().__init__()

        # Shared encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Task-specific heads
        self.hate_classifier = nn.Linear(hidden_size, 2)
        self.hate_type_classifier = nn.Linear(hidden_size, num_hate_types)
        self.target_classifier = nn.Linear(hidden_size, 2)

        # Loss functions
        self.loss_hate = nn.CrossEntropyLoss()
        self.loss_hate_type = nn.CrossEntropyLoss()
        self.loss_target = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        hate_label=None,
        hate_type=None,
        target=None
    ):
        # Encoder forward pass
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Task heads
        hate_logits = self.hate_classifier(pooled_output)
        hate_type_logits = self.hate_type_classifier(pooled_output)
        target_logits = self.target_classifier(pooled_output)

        loss = None

        if hate_label is not None:
            loss = self.loss_hate(hate_logits, hate_label)

            if hate_type is not None:
                loss += self.loss_hate_type(hate_type_logits, hate_type)

            if target is not None:
                loss += self.loss_target(target_logits, target)

        return {
            "loss": loss,
            "hate_logits": hate_logits,
            "hate_type_logits": hate_type_logits,
            "target_logits": target_logits
        }
    
    def load_encoder_from_baseline(self, baseline_path):
        baseline_state = torch.load(baseline_path, map_location="cpu")

        encoder_state = {
            k.replace("roberta.", ""): v
            for k, v in baseline_state.items()
            if k.startswith("roberta.")
        }

        self.encoder.load_state_dict(encoder_state, strict=False)



# -------------------------------
# Debug / sanity check
# -------------------------------
if __name__ == "__main__":
    model = MultiTaskHateModel(num_hate_types=5)

    dummy_input_ids = torch.randint(0, 1000, (2, 128))
    dummy_attention = torch.ones((2, 128))

    outputs = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention,
        hate_label=torch.tensor([1, 0]),
        hate_type=torch.tensor([2, 1]),
        target=torch.tensor([1, 0])
    )

    print("Loss:", outputs["loss"])
    print("Hate logits shape:", outputs["hate_logits"].shape)
    print("Hate type logits shape:", outputs["hate_type_logits"].shape)
    print("Target logits shape:", outputs["target_logits"].shape)
