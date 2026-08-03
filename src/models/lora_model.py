"""
DistilBERT + LoRA adapter setup, via HuggingFace transformers + peft.

Owner: Track B

TODO:
- Confirm target_modules match DistilBERT's actual attention layer names
  (check model.named_modules() -- typically 'q_lin', 'v_lin' for DistilBERT).
- Sanity check: right after applying LoRA, delta_W should be ~0 since B
  is zero-initialized (per the slides' "Why Initializing B to Zero Matters").
- Add save/load helpers for LoRA adapter weights only (this is the whole
  point -- these are what get sent over the federated network, not the
  full 66M-param model).
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "distilbert-base-uncased"


def build_lora_model(num_labels: int = 2, r: int = 8, lora_alpha: int = 16,
                      lora_dropout: float = 0.1):
    """
    Loads frozen DistilBERT + attaches trainable LoRA adapters (A, B) to the
    attention projection layers.

    Args:
        num_labels: number of classification classes.
        r: LoRA rank -- the key axis being swept in the benchmark
           (Privacy Budget -> LoRA Rank -> Target F1-Score).
        lora_alpha: LoRA scaling factor.
        lora_dropout: dropout applied inside the LoRA adapters.

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


if __name__ == "__main__":
    model, tok = build_lora_model()
