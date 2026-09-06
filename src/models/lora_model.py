import logging

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def setup_lora_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> tuple[PeftModel, AutoTokenizer]:
    """Loads a pre-trained base model and injects trainable LoRA adapters.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of target classification classes.
        r: Rank bottleneck dimension for low-rank projection matrices.
        lora_alpha: Scaling factor for LoRA updates.
        lora_dropout: Dropout probability for LoRA layers.

    Returns:
        Tuple containing wrapped PeftModel and associated AutoTokenizer.
    """
    logger.info("Loading base model: %s...", model_name)

    # 1. Load the tokenizer and the base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # 2. Configure the LoRA adapters
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,       # Sequence Classification (e.g., positive/negative review)
        inference_mode=False,             # Set to False because we are actively training
        r=r,                              # The rank of the update matrices (your sweep parameter!)
        lora_alpha=lora_alpha,            # Scaling factor for the LoRA updates
        lora_dropout=lora_dropout,        # Dropout probability to prevent overfitting
        target_modules=["q_lin", "v_lin"], # For DistilBERT, we target the attention projection layers
    )

    # 3. Wrap the base model with PEFT
    lora_model = get_peft_model(base_model, peft_config)

    # Print out the parameter savings
    lora_model.print_trainable_parameters()

    return lora_model, tokenizer