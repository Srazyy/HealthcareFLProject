"""
Sweep runner for the FL + LoRA + DP Healthcare benchmark.

Iterates over the full (r, ε, α) grid from configs/config.yaml, runs each
configuration through the Flower simulation pipeline, and aggregates
per-round metrics into results/sweep_results.csv.

Usage:
    python sweep.py                          # full grid, 5 rounds
    python sweep.py --rounds 2               # quick validation (2 rounds)
    python sweep.py --config alt_config.yaml  # alternate config
"""

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch

from main import build_client_fn, detect_device, load_config
from src.data.partition import (
    create_client_dataloaders,
    download_and_tokenize,
    partition_data,
)
from src.federated.server import run_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "mcc", "auc_roc")
CSV_COLUMNS = ["rank", "epsilon", "alpha", "round", *METRIC_KEYS, "loss"]


def extract_history_rows(
    history,
    rank: int,
    epsilon: float | None,
    alpha: float,
) -> list[dict]:
    """Extract per-round metric rows from a Flower History object."""
    rounds_set: set[int] = set()
    metrics_by_round: dict[int, dict] = {}

    # Distributed metrics (from evaluate())
    if hasattr(history, "metrics_distributed") and history.metrics_distributed:
        for metric_name, round_values in history.metrics_distributed.items():
            for round_num, value in round_values:
                rounds_set.add(round_num)
                if round_num not in metrics_by_round:
                    metrics_by_round[round_num] = {}
                metrics_by_round[round_num][metric_name] = value

    # Distributed loss
    if hasattr(history, "losses_distributed") and history.losses_distributed:
        for round_num, loss_val in history.losses_distributed:
            rounds_set.add(round_num)
            if round_num not in metrics_by_round:
                metrics_by_round[round_num] = {}
            metrics_by_round[round_num]["loss"] = loss_val

    rows = []
    for r in sorted(rounds_set):
        m = metrics_by_round.get(r, {})
        row = {
            "rank": rank,
            "epsilon": epsilon if epsilon is not None else "none",
            "alpha": alpha,
            "round": r,
            "loss": m.get("loss", ""),
        }
        for key in METRIC_KEYS:
            row[key] = m.get(key, "")
        rows.append(row)

    return rows


def run_sweep(config_path: str, rounds_override: int | None = None) -> Path:
    """Execute the full (r, ε, α) sweep and write results to CSV."""
    config = load_config(config_path)
    device = detect_device()

    ranks = config["lora"]["ranks"]
    epsilons = config["privacy"]["epsilons"]
    alphas = config["data"]["dirichlet_alphas"]
    num_rounds = rounds_override or config["federated"]["num_rounds"]
    num_clients = config["federated"]["num_clients"]

    # Override rounds in config so client_fn sees the correct value
    config["federated"]["num_rounds"] = num_rounds

    total_configs = len(ranks) * len(epsilons) * len(alphas)
    logger.info("=" * 60)
    logger.info("  SWEEP CONFIGURATION")
    logger.info("  Ranks:    %s", ranks)
    logger.info("  Epsilons: %s", epsilons)
    logger.info("  Alphas:   %s", alphas)
    logger.info("  Rounds:   %d", num_rounds)
    logger.info("  Total:    %d configurations", total_configs)
    logger.info("  Device:   %s", device)
    logger.info("=" * 60)

    # Step 1: Download & tokenize ONCE
    logger.info("Step 1: Downloading and tokenizing dataset (one-time)...")
    dataset, tokenizer = download_and_tokenize()
    labels = np.array(dataset["labels"])
    logger.info("Dataset ready: %d samples", len(dataset))

    # Prepare output directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "sweep_results.csv"
    summary_path = results_dir / "sweep_summary.json"

    all_rows: list[dict] = []
    final_round_table: dict[str, dict[str, float]] = {}
    completed = 0
    sweep_start = time.time()

    # Step 2: Iterate — partition once per alpha, then sweep (r, ε) within
    for alpha in alphas:
        logger.info("-" * 60)
        logger.info("Partitioning data for α=%.2f...", alpha)
        client_indices = partition_data(
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=config["training"]["seed"],
        )
        for i, idx in enumerate(client_indices):
            label_dist = np.bincount(labels[idx], minlength=config["data"]["num_labels"])
            logger.info(
                "  Hospital %d: %d samples, distribution: %s",
                i, len(idx), label_dist,
            )

        client_dataloaders = create_client_dataloaders(
            dataset=dataset,
            client_indices=client_indices,
            batch_size=config["training"]["batch_size"],
            seed=config["training"]["seed"],
        )

        for rank, raw_eps in product(ranks, epsilons):
            current_epsilon = None if raw_eps is None else float(raw_eps)
            completed += 1
            eps_label = f"{current_epsilon:.1f}" if current_epsilon is not None else "none"

            logger.info(
                "[%d/%d] r=%d, ε=%s, α=%.2f — starting %d-round simulation...",
                completed, total_configs, rank, eps_label, alpha, num_rounds,
            )
            run_start = time.time()

            client_fn = build_client_fn(
                config=config,
                client_dataloaders=client_dataloaders,
                device=device,
                current_rank=rank,
                current_epsilon=current_epsilon,
            )

            history = run_simulation(
                client_fn=client_fn,
                num_clients=num_clients,
                num_rounds=num_rounds,
            )

            rows = extract_history_rows(history, rank, current_epsilon, alpha)
            all_rows.extend(rows)

            elapsed = time.time() - run_start
            # Capture final-round F1 for the reference table
            if rows:
                final_f1 = rows[-1].get("f1", "N/A")
                key = f"ε={eps_label}, α={alpha:.1f}"
                if key not in final_round_table:
                    final_round_table[key] = {}
                final_round_table[key][f"r={rank}"] = final_f1
                logger.info(
                    "[%d/%d] Done in %.1fs — final-round F1: %s",
                    completed, total_configs, elapsed, final_f1,
                )
            else:
                logger.warning(
                    "[%d/%d] No metrics recorded for r=%d, ε=%s, α=%.2f",
                    completed, total_configs, rank, eps_label, alpha,
                )

    sweep_elapsed = time.time() - sweep_start

    # Step 3: Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Results written to %s (%d rows)", csv_path, len(all_rows))

    # Step 4: Write summary JSON
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "total_configs": total_configs,
        "rounds_per_config": num_rounds,
        "total_rows": len(all_rows),
        "total_runtime_seconds": round(sweep_elapsed, 1),
        "device": str(device),
        "grid": {
            "ranks": ranks,
            "epsilons": epsilons,
            "alphas": alphas,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", summary_path)

    # Step 5: Print reference table
    logger.info("=" * 60)
    logger.info("  REFERENCE TABLE — Final-Round F1 Scores")
    logger.info("=" * 60)
    rank_headers = [f"r={r}" for r in ranks]
    header = f"{'Config':<25} | " + " | ".join(f"{h:>8}" for h in rank_headers)
    logger.info(header)
    logger.info("-" * len(header))
    for key, rank_vals in final_round_table.items():
        vals = []
        for rh in rank_headers:
            v = rank_vals.get(rh, "N/A")
            vals.append(f"{v:>8.4f}" if isinstance(v, (int, float)) else f"{v!s:>8}")
        logger.info(f"{key:<25} | " + " | ".join(vals))
    logger.info("=" * 60)
    logger.info("Sweep completed in %.1f seconds", sweep_elapsed)

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Sweep runner for FL + LoRA + DP Healthcare benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML experiment config",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override number of federation rounds (e.g., 2 for quick validation)",
    )
    args = parser.parse_args()
    run_sweep(config_path=args.config, rounds_override=args.rounds)


if __name__ == "__main__":
    main()
