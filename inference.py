#!/usr/bin/env python3
"""
Interactive Clinical Inference Script for Saved LoRA Adapters.

Allows querying any of the trained federated models on clinical text.

Usage:
    python inference.py --model saved_models/model_r8_eps8.0_alpha1.0 --claim "Patient scheduled for laparoscopic cholecystectomy."
    python inference.py --interactive
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

LABEL_MAP = {
    0: "medical_conservative (Conservative Management / Medication / Diagnostics)",
    1: "surgical_procedural (Surgical Intervention / Operative Procedure)",
}


def load_inference_model(adapter_dir: str, base_model_name: str = "distilbert-base-uncased"):
    path = Path(adapter_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model adapter path not found: {adapter_dir}")

    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if (path / "tokenizer_config.json").exists() else base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    
    print(f"Attaching LoRA adapter from: {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
    return pred_idx, confidence, probs.tolist()


def main():
    parser = argparse.ArgumentParser(description="Clinical Claim Inference using Saved LoRA Adapters")
    parser.add_argument("--model", type=str, default=None, help="Path to saved LoRA adapter folder")
    parser.add_argument("--claim", type=str, default=None, help="Clinical claim or note text to classify")
    parser.add_argument("--interactive", action="store_true", help="Start interactive CLI loop")
    args = parser.parse_args()

    # Find default model if none specified
    if not args.model:
        models_dir = Path("saved_models")
        if models_dir.exists():
            available = list(models_dir.glob("model_*"))
            if available:
                args.model = str(available[0])
        if not args.model:
            print("[Error] No model directory specified and none found in saved_models/.")
            print("Usage: python inference.py --model saved_models/model_r8_eps8.0_alpha1.0")
            sys.exit(1)

    model, tokenizer = load_inference_model(args.model)

    if args.interactive:
        print("\n" + "=" * 70)
        print("  INTERACTIVE CLINICAL INFERENCE CONSOLE")
        print(f"  Model: {args.model}")
        print("  Type a clinical sentence and press Enter (or 'quit' to exit):")
        print("=" * 70)
        while True:
            try:
                line = input("\n[Clinical Text] > ").strip()
                if line.lower() in ("quit", "exit", "q"):
                    break
                if not line:
                    continue
                pred, conf, probs = predict(model, tokenizer, line)
                print(f"  -> Prediction:  {LABEL_MAP[pred]}")
                print(f"  -> Confidence:  {conf:.2%}")
                print(f"  -> Probabilities: [Class 0 (Medical): {probs[0]:.3f}, Class 1 (Surgical): {probs[1]:.3f}]")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting inference console.")
        return

    # Single claim prediction
    claim_text = args.claim or "Laparoscopic lysis of adhesions and open laparoscopic pyeloplasty."
    pred, conf, probs = predict(model, tokenizer, claim_text)
    
    print("\n" + "=" * 70)
    print("  CLINICAL CLAIM PREDICTION RESULT")
    print("=" * 70)
    print(f"  Input Text:      {claim_text}")
    print(f"  Model Adapter:   {args.model}")
    print(f"  Prediction:      {LABEL_MAP[pred]}")
    print(f"  Confidence:      {conf:.2%}")
    print(f"  Probabilities:   Medical={probs[0]:.2%}, Surgical={probs[1]:.2%}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
