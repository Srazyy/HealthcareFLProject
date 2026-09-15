"""
Dirichlet-based non-IID data partitioning across simulated hospitals.

Provides:
  - download_and_tokenize(): fetches health_fact from HF Hub (binary labels),
    with synthetic fallback.
  - partition_data(): splits sample indices across N clients using Dirichlet(α).
  - create_client_dataloaders(): builds per-client train/val DataLoaders.
"""

import numpy as np
import os
from typing import Any, cast
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import DataLoader, Subset
import torch


def partition_data(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples: int = 10,
) -> list[np.ndarray]:
    """Partition sample indices across `num_clients` using a Dirichlet distribution.

    Lower alpha -> more skewed / non-IID (matches the "Hospital A: 90%
    negative, Hospital B: 90% positive" scenario from the project brief).
    Higher alpha -> closer to IID.

    Args:
        labels: 1D array of class labels for the full dataset.
        num_clients: Number of simulated hospitals.
        alpha: Dirichlet concentration parameter.
        seed: RNG seed for reproducibility.
        min_samples: Minimum required samples per client partition.

    Returns:
        List of sample index arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    min_required = min(min_samples, len(labels) // num_clients)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for _ in range(100):
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

        if all(len(idx) >= min_required for idx in client_indices):
            break

    return [np.array(idx, dtype=np.int64) for idx in client_indices]


import logging
import random as _random
logger = logging.getLogger(__name__)


def _generate_medical_claims(n: int = 1500, seed: int = 42) -> tuple[list[str], list[int]]:
    """Generate *n* diverse synthetic medical claims with binary labels.

    Label 0 = verified / trustworthy claim.
    Label 1 = unverified / dubious claim.

    Uses combinatorial template expansion so every sentence is unique,
    covering realistic clinical vocabulary, varying lengths, and both
    classes in roughly equal proportion.
    """
    rng = _random.Random(seed)

    # ── building blocks ──────────────────────────────────────────────
    conditions = [
        "type 2 diabetes", "hypertension", "chronic kidney disease",
        "asthma", "COPD", "heart failure", "atrial fibrillation",
        "major depressive disorder", "generalized anxiety disorder",
        "rheumatoid arthritis", "osteoarthritis", "migraine",
        "epilepsy", "Parkinson's disease", "Alzheimer's disease",
        "breast cancer", "lung cancer", "colorectal cancer",
        "prostate cancer", "leukemia", "lymphoma", "melanoma",
        "hepatitis B", "hepatitis C", "HIV infection",
        "tuberculosis", "pneumonia", "urinary tract infection",
        "sepsis", "anemia", "deep vein thrombosis",
        "pulmonary embolism", "stroke", "myocardial infarction",
        "celiac disease", "Crohn's disease", "ulcerative colitis",
        "psoriasis", "eczema", "lupus", "multiple sclerosis",
        "chronic pain syndrome", "fibromyalgia", "gout",
        "hypothyroidism", "hyperthyroidism", "obesity",
        "sleep apnea", "irritable bowel syndrome", "pancreatitis",
    ]

    treatments = [
        "metformin", "lisinopril", "amlodipine", "atorvastatin",
        "omeprazole", "levothyroxine", "albuterol", "insulin glargine",
        "sertraline", "fluoxetine", "amoxicillin", "azithromycin",
        "ibuprofen", "acetaminophen", "prednisone", "warfarin",
        "apixaban", "rivaroxaban", "doxycycline", "ciprofloxacin",
        "gabapentin", "pregabalin", "montelukast", "losartan",
        "hydrochlorothiazide", "carvedilol", "tamsulosin",
        "escitalopram", "bupropion", "trazodone", "clonazepam",
        "methylprednisolone", "rituximab", "adalimumab",
        "pembrolizumab", "nivolumab", "trastuzumab",
        "physical therapy", "cognitive behavioral therapy",
        "dietary intervention", "surgical resection",
        "radiation therapy", "chemotherapy regimen",
        "stem cell transplant", "immunoglobulin therapy",
        "plasma exchange", "phototherapy", "acupuncture",
        "mindfulness-based stress reduction",
    ]

    outcomes_good = [
        "showed significant clinical improvement",
        "achieved full remission after 12 weeks",
        "reported reduced symptom severity",
        "demonstrated improved biomarker levels",
        "maintained stable vitals throughout treatment",
        "experienced no adverse events during follow-up",
        "showed statistically significant improvement (p<0.01)",
        "achieved target therapeutic outcomes",
        "reported improved quality of life scores",
        "met primary endpoint criteria in the trial",
        "had fewer emergency department visits post-treatment",
        "showed reduced inflammatory markers",
        "demonstrated sustained response over 6 months",
        "reported decreased pain intensity on VAS scale",
        "achieved glycemic control within target range",
    ]

    outcomes_bad = [
        "showed no measurable improvement",
        "experienced significant adverse effects",
        "had treatment discontinued due to toxicity",
        "demonstrated worsening laboratory values",
        "required hospitalization during treatment",
        "reported increased symptom burden",
        "failed to meet study primary endpoint",
        "developed treatment-resistant symptoms",
        "showed poor medication adherence",
        "experienced disease progression despite therapy",
    ]

    sources_verified = [
        "according to a peer-reviewed randomized controlled trial",
        "as documented in the patient's electronic health record",
        "based on findings from a multi-center clinical study",
        "per the attending physician's clinical assessment",
        "as confirmed by laboratory and imaging results",
        "according to published meta-analysis data",
        "as reported in the New England Journal of Medicine",
        "per FDA-approved prescribing information",
        "based on Cochrane systematic review evidence",
        "as verified by the hospital's quality assurance review",
        "according to WHO treatment guidelines",
        "per clinical pharmacology reference data",
    ]

    sources_unverified = [
        "according to an unverified social media post",
        "based on anecdotal patient testimony without controls",
        "from a non-peer-reviewed preprint with limited sample size",
        "according to claims on an alternative medicine website",
        "based on a single case report without replication",
        "from an anonymous online health forum",
        "per manufacturer marketing materials only",
        "based on preliminary in-vitro results not validated in humans",
        "according to retracted or disputed research findings",
        "from a self-published blog without clinical credentials",
        "based on observational data with significant confounders",
        "per traditional remedy claims without clinical evidence",
    ]

    qualifiers_verified = [
        "The treatment protocol followed evidence-based guidelines.",
        "All outcomes were independently verified by two clinicians.",
        "Results were consistent across multiple patient subgroups.",
        "The study used double-blind placebo-controlled methodology.",
        "Findings have been replicated in subsequent studies.",
        "Safety data was reviewed by an independent monitoring board.",
        "The patient cohort met standard inclusion criteria.",
        "Statistical analysis was pre-registered before data collection.",
        "",  # some sentences have no qualifier for variety
        "",
    ]

    qualifiers_unverified = [
        "No control group was used in the observation.",
        "The claim has not been reviewed by medical professionals.",
        "Sample size was insufficient for statistical significance.",
        "Potential conflicts of interest were not disclosed.",
        "The methodology has been criticized by domain experts.",
        "Results could not be reproduced in independent testing.",
        "The claim contradicts established medical consensus.",
        "No adverse event monitoring was conducted.",
        "",
        "",
    ]

    claims: list[str] = []
    labels: list[int] = []

    half = n // 2

    # ── verified claims (label=0) ────────────────────────────────────
    for _ in range(half):
        cond = rng.choice(conditions)
        treat = rng.choice(treatments)
        outcome = rng.choice(outcomes_good)
        source = rng.choice(sources_verified)
        qualifier = rng.choice(qualifiers_verified)

        templates = [
            f"Patient diagnosed with {cond} and treated with {treat} {outcome}, {source}. {qualifier}",
            f"A clinical study on {treat} for {cond} found that patients {outcome}, {source}. {qualifier}",
            f"In a cohort of patients with {cond}, administration of {treat} {outcome}. {qualifier}",
            f"Treatment with {treat} in {cond} patients {outcome}, {source}.",
            f"Evidence from clinical practice confirms that {treat} for {cond} {outcome}. {qualifier}",
            f"Hospital records indicate that {cond} patients receiving {treat} {outcome}.",
        ]
        claim = rng.choice(templates).replace("  ", " ").strip()
        claims.append(claim)
        labels.append(0)

    # ── unverified claims (label=1) ──────────────────────────────────
    for _ in range(n - half):
        cond = rng.choice(conditions)
        treat = rng.choice(treatments)
        outcome_pool = outcomes_good + outcomes_bad
        outcome = rng.choice(outcome_pool)
        source = rng.choice(sources_unverified)
        qualifier = rng.choice(qualifiers_unverified)

        templates = [
            f"It is claimed that {treat} can cure {cond} completely, {source}. {qualifier}",
            f"Reports suggest {treat} {outcome} for {cond}, {source}. {qualifier}",
            f"An unconfirmed source states that {treat} is a breakthrough treatment for {cond}. {qualifier}",
            f"Patients on social media report that {treat} {outcome} in {cond} cases, {source}.",
            f"Alleged benefits of {treat} for {cond} include rapid recovery, {source}. {qualifier}",
            f"Without clinical validation, {treat} is promoted as effective against {cond}. {qualifier}",
        ]
        claim = rng.choice(templates).replace("  ", " ").strip()
        claims.append(claim)
        labels.append(1)

    # shuffle so labels aren't sorted
    combined = list(zip(claims, labels))
    rng.shuffle(combined)
    shuffled_claims = [c for c, _ in combined]
    shuffled_labels = [y for _, y in combined]
    return shuffled_claims, shuffled_labels


def download_and_tokenize(
    dataset_name: str = "health_fact",
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
) -> tuple[Dataset, PreTrainedTokenizerBase]:
    """
    Downloads a medical NLP dataset from HuggingFace, collapses labels to
    binary (verified=0 / unverified=1), tokenizes with the DistilBERT tokenizer,
    and sets the format to PyTorch tensors.

    Loading priority:
      1. ``health_fact`` from HuggingFace Hub (original project dataset).
      2. ``GonzaloA/fake_news`` — a proven-working parquet-based binary
         classification dataset re-framed as medical claim verification.
      3. Rich synthetic medical claims corpus (1 500 diverse sentences)
         as an offline-safe fallback.

    Returns:
        (tokenized_dataset, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None, f"Failed to load tokenizer for {model_name}"

    # ── Attempt 1: original health_fact dataset ──────────────────────
    try:
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.filter(lambda x: x["label"] != -1)
        dataset = dataset.map(lambda x: {"label": 0 if x["label"] == 0 else 1})

        def tokenize_fn(example: dict) -> dict:
            return tokenizer(example["claim"], truncation=True, padding="max_length", max_length=max_length)

        cols_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        logger.info(f"Loaded '{dataset_name}' from HuggingFace Hub ({len(dataset)} samples).")
        return cast(Dataset, dataset), cast(PreTrainedTokenizerBase, tokenizer)
    except Exception as e:
        logger.warning(f"Could not load HuggingFace dataset '{dataset_name}': {e}.")

    # ── Attempt 2: local parquet file (no download needed) ───────────
    local_parquet = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fake_news_3k.parquet")
    local_parquet = os.path.normpath(local_parquet)
    if os.path.isfile(local_parquet):
        try:
            logger.info(f"Loading local dataset from {local_parquet}...")
            dataset = load_dataset("parquet", data_files=local_parquet, split="train")

            def tokenize_fn(example: dict) -> dict:
                return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)

            cols_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
            dataset = dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
            dataset = dataset.rename_column("label", "labels")
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            logger.info(f"Loaded local parquet dataset ({len(dataset)} samples).")
            return cast(Dataset, dataset), cast(PreTrainedTokenizerBase, tokenizer)
        except Exception as e:
            logger.warning(f"Failed to load local parquet: {e}.")

    # ── Attempt 3: GonzaloA/fake_news from HF Hub (downloads ~20MB) ──
    try:
        logger.info("Trying fallback dataset 'GonzaloA/fake_news' from HuggingFace...")
        hf_dataset = load_dataset("GonzaloA/fake_news", split="train")
        # Use the 'text' column; labels are already 0/1 (real/fake → verified/unverified)
        filtered_ds = cast(Dataset, hf_dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 20))
        dataset = filtered_ds.select(range(min(3000, len(filtered_ds))))  # cap at 3k for speed

        def tokenize_fn(example: dict) -> dict:
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)

        cols_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        logger.info(f"Loaded 'GonzaloA/fake_news' fallback ({len(dataset)} samples).")
        return cast(Dataset, dataset), cast(PreTrainedTokenizerBase, tokenizer)
    except Exception as e:
        logger.warning(f"Fallback 'GonzaloA/fake_news' also failed: {e}.")

    # ── Attempt 3: rich synthetic medical claims ─────────────────────
    logger.info("Generating diverse synthetic medical claims corpus (1500 samples)...")
    synthetic_claims, synthetic_labels = _generate_medical_claims(n=1500, seed=42)
    raw_ds = Dataset.from_dict({"claim": synthetic_claims, "labels": synthetic_labels})

    def tokenize_fn(example: dict) -> dict:
        return tokenizer(example["claim"], truncation=True, padding="max_length", max_length=max_length)

    dataset = cast(Dataset, raw_ds.map(tokenize_fn, batched=True, remove_columns=["claim"]))
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    logger.info(f"Using synthetic medical claims dataset ({len(dataset)} unique samples).")
    return dataset, cast(PreTrainedTokenizerBase, tokenizer)


def create_client_dataloaders(
    dataset: Dataset,
    client_indices: list[np.ndarray],
    batch_size: int = 16,
    val_split: float = 0.1,
    seed: int = 42,
) -> list[tuple[DataLoader, DataLoader]]:
    """Builds per-client train and validation DataLoaders.

    Args:
        dataset: The tokenized dataset to slice.
        client_indices: List of index arrays, one per client.
        batch_size: DataLoader batch size.
        val_split: Fraction of each client's data reserved for validation.
        seed: Random seed for train/val split reproducibility.

    Returns:
        List of (train_loader, val_loader) tuples, one per client.
    """
    dataloaders = []
    generator = torch.Generator().manual_seed(seed)

    for indices in client_indices:
        client_subset = Subset(cast(Any, dataset), indices.tolist())

        val_size = int(len(client_subset) * val_split)
        train_size = len(client_subset) - val_size
        if val_size == 0 and len(client_subset) > 1:
            val_size = 1
            train_size = len(client_subset) - 1

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
