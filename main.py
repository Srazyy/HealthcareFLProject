"""
Master engine for the Federated Learning healthcare pipeline.

Ties together:
  - Track A: data partitioning (Dirichlet non-IID) + differential privacy (Opacus)
  - Track B: LoRA-adapted DistilBERT + Flower federation (FedAvg)

Usage:
    python main.py                                           # defaults from config.yaml
    python main.py --config configs/config.yaml              # explicit config
    python main.py --rank 4 --epsilon 8.0 --alpha 0.1        # single-run override
    python main.py --rounds 2 --epsilon null                 # no DP baseline
"""

import argparse
import logging
import yaml
import torch
import numpy as np
import flwr as fl

from src.data.partition import (
    download_and_tokenize,
    partition_data,
    create_client_dataloaders,
)
from src.models.lora_model import setup_lora_model
from src.privacy.dp_engine import validate_and_fix_model, make_private, get_epsilon_spent
from src.federated.client import HealthcareClient
from src.federated.server import run_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and return the YAML experiment configuration."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


def detect_device() -> torch.device:
    """Detect the best available accelerator (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device for acceleration")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available — falling back to CPU")
    return device


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client_fn(
    config: dict,
    client_dataloaders: list,
    device: torch.device,
    current_rank: int,
    current_epsilon: float | None,
):
    """
    Returns a Flower `client_fn(cid) -> NumPyClient` closure.

    Each simulated hospital gets:
      1. A fresh LoRA-adapted DistilBERT model (on the target device)
      2. An AdamW optimizer filtered to trainable (LoRA) parameters only
      3. (Optional) Opacus DP-SGD wrapping if epsilon is not None

    PRIVACY INVARIANT: raw text and labels are captured inside the closure
    via pre-partitioned DataLoaders. They never leave the client boundary.
    Only noised LoRA adapter deltas are returned to the server.
    """

    def client_fn(cid: str) -> fl.client.NumPyClient:
        client_id = int(cid)
        logger.info(
            "Spawning client %d  (r=%d, ε=%s)",
            client_id,
            current_rank,
            current_epsilon if current_epsilon is not None else "∞ (no DP)",
        )

        # 1. Fresh model per client — each hospital has its own copy
        model, _ = setup_lora_model(
            num_labels=config["data"]["num_labels"],
            r=current_rank,
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
        )
        model = model.to(device)

        # 2. Grab this client's pre-partitioned data (stays local)
        trainloader, valloader = client_dataloaders[client_id]

        # 3. Optimizer on trainable (LoRA) params ONLY — avoids wasting
        #    memory on momentum/variance buffers for the 66M frozen params
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
        )

        # 4. Conditionally attach Opacus differential privacy
        privacy_engine = None
        if current_epsilon is not None:
            model = validate_and_fix_model(model)

            # Total epochs across ALL federation rounds, not just one round.
            # This ensures noise calibration (σ) is correct for the full
            # training horizon and the reported ε is truthful.
            total_epochs = (
                config["training"]["local_epochs"]
                * config["federated"]["num_rounds"]
            )

            model, optimizer, trainloader, privacy_engine = make_private(
                model=model,
                optimizer=optimizer,
                data_loader=trainloader,
                target_epsilon=current_epsilon,
                target_delta=config["privacy"]["target_delta"],
                epochs=total_epochs,
                max_grad_norm=config["privacy"]["max_grad_norm"],
            )
            logger.info(
                "Client %d: DP attached (target ε=%.2f, δ=%.2e, σ calibrated for %d epochs)",
                client_id,
                current_epsilon,
                config["privacy"]["target_delta"],
                total_epochs,
            )

        return HealthcareClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            optimizer=optimizer,
            privacy_engine=privacy_engine,
            lr=config["training"]["learning_rate"],
        ).to_client()

    return client_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning + LoRA + DP Healthcare Pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML experiment config",
    )
    parser.add_argument(
        "--rank", type=int, default=None,
        help="LoRA rank (r) override for a single run. If omitted, uses the first rank in config.",
    )
    parser.add_argument(
        "--epsilon",
        type=str,
        default=None,
        help="Privacy budget ε override. Use 'null' or 'none' for no DP. If omitted, uses the first epsilon in config.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dirichlet α override for non-IID partitioning. If omitted, uses the first alpha in config.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of federation rounds override.",
    )
    args = parser.parse_args()

    # ── Load configuration ──────────────────────────────────────────────
    config = load_config(args.config)
    device = detect_device()

    # Resolve single-run overrides (fall back to first element in sweep lists)
    current_rank = args.rank or config["lora"]["ranks"][0]
    current_alpha = args.alpha or config["data"]["dirichlet_alphas"][0]
    num_rounds = args.rounds or config["federated"]["num_rounds"]

    # Parse epsilon: "null"/"none" → None (no DP), else float
    if args.epsilon is not None:
        current_epsilon = (
            None
            if args.epsilon.lower() in ("null", "none", "inf")
            else float(args.epsilon)
        )
    else:
        raw_eps = config["privacy"]["epsilons"][0]
        current_epsilon = None if raw_eps is None else float(raw_eps)

    # Persist resolved rounds back into config for client_fn
    config["federated"]["num_rounds"] = num_rounds

    logger.info("=" * 60)
    logger.info("  FL + LoRA + DP  —  Single Run Configuration")
    logger.info("  LoRA rank (r)     : %d", current_rank)
    logger.info("  Dirichlet α       : %.2f", current_alpha)
    logger.info(
        "  Privacy budget (ε): %s",
        f"{current_epsilon:.2f}" if current_epsilon is not None else "∞ (no DP — baseline)",
    )
    logger.info("  Federation rounds : %d", num_rounds)
    logger.info("  Device            : %s", device)
    logger.info("=" * 60)

    # ── Step 1: Download & tokenize the dataset ─────────────────────────
    logger.info("Step 1/4: Downloading and tokenizing dataset...")
    dataset, tokenizer = download_and_tokenize()
    labels = np.array(dataset["labels"])
    logger.info(
        "Dataset ready: %d samples, label distribution: %s",
        len(dataset),
        np.bincount(labels),
    )

    # ── Step 2: Dirichlet non-IID partitioning ──────────────────────────
    num_clients = config["federated"]["num_clients"]
    logger.info(
        "Step 2/4: Partitioning data across %d clients (α=%.2f)...",
        num_clients,
        current_alpha,
    )
    client_indices = partition_data(
        labels=labels,
        num_clients=num_clients,
        alpha=current_alpha,
        seed=config["training"]["seed"],
    )
    for i, idx in enumerate(client_indices):
        label_dist = np.bincount(labels[idx], minlength=config["data"]["num_labels"])
        logger.info(
            "  Hospital %d: %d samples, class distribution: %s (positive rate: %.1f%%)",
            i,
            len(idx),
            label_dist,
            100.0 * label_dist[1] / len(idx) if len(idx) > 0 else 0.0,
        )

    # ── Step 3: Create DataLoaders ──────────────────────────────────────
    logger.info("Step 3/4: Creating per-client DataLoaders...")
    client_dataloaders = create_client_dataloaders(
        dataset=dataset,
        client_indices=client_indices,
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
    )

    # ── Step 4: Build client factory & run simulation ───────────────────
    logger.info("Step 4/4: Launching Flower simulation...")
    client_fn = build_client_fn(
        config=config,
        client_dataloaders=client_dataloaders,
        device=device,
        current_rank=current_rank,
        current_epsilon=current_epsilon,
    )

    history = run_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        num_rounds=num_rounds,
    )

    # ── Results ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Simulation Complete")
    logger.info("=" * 60)
    if hasattr(history, "metrics_distributed") and history.metrics_distributed:
        for metric_name, rounds in history.metrics_distributed.items():
            for round_num, value in rounds:
                logger.info("  Round %d — %s: %.4f", round_num, metric_name, value)
    else:
        logger.info("  No distributed metrics recorded (check evaluate() return).")

    logger.info("Run config: r=%d, α=%.2f, ε=%s", current_rank, current_alpha, current_epsilon)
    logger.info("Done.")


if __name__ == "__main__":
    main()
