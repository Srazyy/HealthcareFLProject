import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import Tuple

def setup_lora_model(model_name="distilbert-base-uncased", num_labels=2, r=8, lora_alpha=16, lora_dropout=0.1) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Loads a pre-trained DistilBERT model and injects LoRA adapters.
    The 'r' parameter is your bottleneck rank (part of your FL sweep).
    """
    print(f"Loading base model: {model_name}...")
    
    # 1. Load the tokenizer and the base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )

    # 2. Configure the LoRA adapters
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,       # Sequence Classification (e.g., positive/negative review)
        inference_mode=False,             # Set to False because we are actively training
        r=r,                              # The rank of the update matrices (your sweep parameter!)
        lora_alpha=lora_alpha,            # Scaling factor for the LoRA updates
        lora_dropout=lora_dropout,        # Dropout probability to prevent overfitting
        target_modules=["q_lin", "v_lin"] # For DistilBERT, we target the attention projection layers
    )

    # 3. Wrap the base model with PEFT
    lora_model = get_peft_model(base_model, peft_config)

    # Print out the parameter savings
    lora_model.print_trainable_parameters()

    return lora_model, tokenizer