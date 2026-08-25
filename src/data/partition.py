"""
Dirichlet-based non-IID data partitioning across simulated hospitals.

Owner: Track A

TODO:
- Load the actual dataset (patient reviews / clinical notes, or a stand-in
  public dataset for now).
- Implement partition_data() below.
- Add a quick script/plot showing class distribution per client for a given
  alpha, to visually confirm the skew (e.g. alpha=0.1 -> highly skewed,
  alpha=100 -> near-IID).
"""

import numpy as np
from typing import Tuple, List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
import torch


def partition_data(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42):
    """
    Partition sample indices across `num_clients` using a Dirichlet
    distribution over class proportions, controlled by `alpha`.

    Lower alpha -> more skewed / non-IID (matches the "Hospital A: 90%
    negative, Hospital B: 90% positive" scenario from the project brief).
    Higher alpha -> closer to IID.

    Args:
        labels: 1D array of class labels for the full dataset.
        num_clients: number of simulated hospitals.
        alpha: Dirichlet concentration parameter.
        seed: RNG seed for reproducibility.

    Returns:
        List[np.ndarray]: one array of sample indices per client.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # convert proportions to split points
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)

        for client_id, idx in enumerate(splits):
            client_indices[client_id].extend(idx.tolist())

    return [np.array(idx) for idx in client_indices]


import logging
logger = logging.getLogger(__name__)

def download_and_tokenize(
    dataset_name: str = "health_fact",
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
) -> Tuple[Dataset, AutoTokenizer]:
    """
    Downloads a medical NLP dataset from HuggingFace, collapses labels to
    binary (verified=0 / unverified=1), tokenizes with the DistilBERT tokenizer,
    and sets the format to PyTorch tensors.
    
    Falls back gracefully to a synthetic medical review dataset if HF Hub
    is unavailable or the dataset script is deprecated.
    
    Returns:
        (tokenized_dataset, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.filter(lambda x: x["label"] != -1)
        dataset = dataset.map(lambda x: {"label": 0 if x["label"] == 0 else 1})
        
        def tokenize_fn(example):
            return tokenizer(example["claim"], truncation=True, padding="max_length", max_length=max_length)
        
        cols_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset, tokenizer
    except Exception as e:
        logger.warning(f"Could not load HuggingFace dataset '{dataset_name}': {e}. Using synthetic dataset.")
        synthetic_claims = [
            "Patient responded well to treatment and reported no severe side effects.",
            "Unverified claim regarding experimental treatment without clinical trial data.",
            "Doctor verified successful recovery following standard procedure.",
            "Unconfirmed report of adverse reaction without diagnostic proof.",
        ] * 125
        synthetic_labels = [0, 1, 0, 1] * 125
        raw_ds = Dataset.from_dict({"claim": synthetic_claims, "labels": synthetic_labels})
        
        def tokenize_fn(example):
            return tokenizer(example["claim"], truncation=True, padding="max_length", max_length=max_length)
            
        dataset = raw_ds.map(tokenize_fn, batched=True, remove_columns=["claim"])
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset, tokenizer


def create_client_dataloaders(
    dataset: Dataset,
    client_indices: List[np.ndarray],
    batch_size: int = 16,
    val_split: float = 0.1,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    For each client's index partition:
    1. Subset the tokenized dataset
    2. Split into train / validation (90/10)
    3. Wrap in PyTorch DataLoaders
    
    Returns:
        List of (train_loader, val_loader) tuples, one per client.
    """
    dataloaders = []
    generator = torch.Generator().manual_seed(seed)
    
    for indices in client_indices:
        client_subset = Subset(dataset, indices.tolist())
        
        val_size = int(len(client_subset) * val_split)
        train_size = len(client_subset) - val_size
        
        train_ds, val_ds = torch.utils.data.random_split(
            client_subset, [train_size, val_size], generator=generator
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        dataloaders.append((train_loader, val_loader))
        
    return dataloaders


if __name__ == "__main__":
    # quick smoke test with synthetic labels
    fake_labels = np.random.default_rng(0).integers(0, 2, size=1000)
    parts = partition_data(fake_labels, num_clients=3, alpha=0.3)
    for i, idx in enumerate(parts):
        pos_rate = fake_labels[idx].mean() if len(idx) else float("nan")
        print(f"client {i}: n={len(idx)}, positive_rate={pos_rate:.2f}")
    
    # Full pipeline smoke test
    print("\n--- Full Pipeline Test ---")
    dataset, tokenizer = download_and_tokenize()
    labels = np.array(dataset["labels"])
    print(f"Dataset size: {len(dataset)}, Label distribution: {np.bincount(labels)}")
    client_indices = partition_data(labels, num_clients=3, alpha=0.5)
    dataloaders = create_client_dataloaders(dataset, client_indices, batch_size=8)
    for i, (tl, vl) in enumerate(dataloaders):
        print(f"Client {i}: train_batches={len(tl)}, val_batches={len(vl)}")
