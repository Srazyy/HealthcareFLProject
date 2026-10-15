# Trained Federated LoRA Model Checkpoints

This directory contains the complete collection of **48 trained LoRA adapter models** generated from the comprehensive Federated Learning (Flower) benchmark sweep across Differential Privacy budgets ($\varepsilon$), LoRA ranks ($r$), and Dirichlet data heterogeneity ($\alpha$).

---

## 1. Directory & Naming Convention

Each subdirectory follows the naming pattern:
```
saved_models/model_r{rank}_eps{epsilon}_alpha{dirichlet_alpha}/
```

For example:
- `saved_models/model_r8_eps3.0_alpha100.0/` — LoRA rank 8, Differential Privacy $\varepsilon=3.0$, IID-like $\alpha=100.0$
- `saved_models/model_r16_epsnone_alpha0.1/` — LoRA rank 16, Non-DP baseline ($\varepsilon=\infty$), extreme non-IID $\alpha=0.1$
- `saved_models/model_r4_eps1.0_alpha1.0/` — LoRA rank 4, Strict DP $\varepsilon=1.0$, moderate non-IID $\alpha=1.0$

### Contents of Each Model Folder
Each checkpoint folder is a standard Hugging Face PEFT adapter package containing:
- `adapter_model.safetensors` (~2.4 MB – 3.4 MB): The trained low-rank adapter weights ($\Delta W = BA$) and classification head weights.
- `adapter_config.json`: PEFT LoRA configuration (rank, alpha, target modules, base model identifier).
- `tokenizer.json` & `tokenizer_config.json`: Fast tokenizer configuration matching the training vocabulary.
- `README.md`: Hugging Face model card with hyperparameters.

---

## 2. Quickstart: How to Load and Run Inference

Teammates working on the frontend dashboard, FastAPI/Flask backend, or evaluation reports can load and run any adapter in 3 lines of code:

### Option A: Using Standard Hugging Face `peft` & `transformers`

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

model_dir = "saved_models/model_r8_eps3.0_alpha100.0"

# 1. Load base DistilBERT backbone
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 2. Attach the trained LoRA adapter weights
model = PeftModel.from_pretrained(base_model, model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model.eval()

# 3. Predict on clinical text or patient review
text = "The doctor was extremely thorough and the medication alleviated the severe symptoms quickly."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()

print(f"Predicted class: {predicted_class} (Probabilities: {probs.tolist()[0]})")
```

### Option B: Using Repository Helper (`src.models.lora_model`)

```python
from src.models.lora_model import load_lora_checkpoint, get_lora_model

# Load model directly with repository utility
model, tokenizer = load_lora_checkpoint("saved_models/model_r8_eps3.0_alpha100.0")
```

---

## 3. Parameter Grid Coverage (48 Configurations)

All combinations of the benchmark grid are available:
- **LoRA Ranks ($r$):** `2`, `4`, `8`, `16`
- **DP Budgets ($\varepsilon$):** `1.0` (high privacy), `3.0` (moderate privacy), `8.0` (relaxed privacy), `none` (no privacy noise / baseline)
- **Data Heterogeneity ($\alpha$):** `0.1` (extreme skew), `1.0` (moderate skew), `100.0` (uniform IID)

To look up exact test metrics (Accuracy, F1, Precision, Recall, MCC, AUC-ROC) for each model, refer to:
- `results/sweep_results.csv`
- `results/sweep_summary.json`
- `dashboard/data.js`
