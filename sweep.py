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
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from itertools import product

import numpy as np
import torch

from main import build_client_fn, detect_device, load_config
from src.data.partition import (
    create_client_dataloaders,
    download_and_tokenize,
    partition_data,
)
from src.federated.server import run_simulation
from src.models.lora_model import setup_lora_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    force=True,
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "mcc", "auc_roc")
CSV_COLUMNS = ["rank", "epsilon", "alpha", "round", *METRIC_KEYS, "loss"]


def cleanup_ray_temp():
    """Removes dangling Ray temporary and session files to protect disk space."""
    try:
        ray_dir = Path("/tmp/ray")
        if ray_dir.exists():
            for p in ray_dir.iterdir():
                if p.is_dir() and p.name.startswith("session_"):
                    shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def save_lora_adapter(
    parameters_ndarrays: list[np.ndarray],
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    save_path: Path,
):
    """Reconstructs global LoRA model from aggregated weights and saves to disk."""
    save_path.mkdir(parents=True, exist_ok=True)
    model, tokenizer = setup_lora_model(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_labels=2,
    )
    trainable_names = [name for name, val in model.named_parameters() if val.requires_grad]
    params_dict = zip(trainable_names, parameters_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Saved trained global LoRA adapter to %s", save_path)


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


def run_sweep(
    config_path: str,
    rounds_override: int | None = None,
    tag: str | None = None,
    output_dir: str | Path | None = None,
    save_adapters: bool = True,
) -> Path:
    """Execute the full (r, ε, α) sweep and write results to CSV with automatic archiving."""
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["caffeinate", "-dims", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Activated macOS caffeinate assertion (preventing sleep while sweep runs)")
        except Exception as e:
            logger.warning("Could not activate caffeinate: %s", e)

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
    logger.info("  Ranks:         %s", ranks)
    logger.info("  Epsilons:      %s", epsilons)
    logger.info("  Alphas:        %s", alphas)
    logger.info("  Rounds:        %d", num_rounds)
    logger.info("  Save Adapters: %s", save_adapters)
    logger.info("  Total:         %d configurations", total_configs)
    logger.info("  Device:        %s", device)
    if tag:
        logger.info("  Run Tag:       %s", tag)
    logger.info("=" * 60)

    # Step 1: Download & tokenize ONCE
    logger.info("Step 1: Downloading and tokenizing dataset (one-time)...")
    dataset, tokenizer = download_and_tokenize()
    labels = np.array(dataset["labels"])
    logger.info("Dataset ready: %d samples", len(dataset))

    # Protect simulation workers from network stalls and HF Hub timeouts
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Prepare output directory
    results_dir = Path(output_dir) if output_dir else Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)
    csv_path = results_dir / "sweep_results.csv"
    summary_path = results_dir / "sweep_summary.json"

    # Historical runs archive
    history_dir = results_dir / "history"
    history_dir.mkdir(exist_ok=True)
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{tag}" if tag else ""
    archive_csv_path = history_dir / f"sweep_{timestamp_slug}{tag_suffix}.csv"
    archive_summary_path = history_dir / f"summary_{timestamp_slug}{tag_suffix}.json"

    all_rows: list[dict] = []
    if csv_path.exists():
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                all_rows = [row for row in reader if row.get("rank")]
            if all_rows:
                logger.info("Loaded %d existing rows from %s for seamless resumption", len(all_rows), csv_path)
        except Exception:
            all_rows = []

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
            model_dir = Path("saved_models") / f"model_r{rank}_eps{eps_label}_alpha{alpha:.1f}"

            # Auto-resume check: skip if adapter is already trained and saved
            if save_adapters and (model_dir / "adapter_config.json").exists():
                msg = f"[{completed}/{total_configs}] r={rank}, ε={eps_label}, α={alpha:.2f} — adapter exists at {model_dir}. Skipping (resuming from checkpoint)..."
                print(msg, flush=True)
                logger.info(msg)
                matching = [
                    row for row in all_rows
                    if str(row.get("rank")) == str(rank)
                    and str(row.get("epsilon")) == eps_label
                    and abs(float(row.get("alpha", -1)) - alpha) < 1e-4
                ]
                if matching:
                    final_f1 = matching[-1].get("f1", "N/A")
                    key = f"ε={eps_label}, α={alpha:.1f}"
                    if key not in final_round_table:
                        final_round_table[key] = {}
                    final_round_table[key][f"r={rank}"] = final_f1
                continue

            msg = f"[{completed}/{total_configs}] r={rank}, ε={eps_label}, α={alpha:.2f} — starting {num_rounds}-round simulation..."
            print(msg, flush=True)
            logger.info(msg)
            run_start = time.time()

            client_fn = build_client_fn(
                config=config,
                client_dataloaders=client_dataloaders,
                device=device,
                current_rank=rank,
                current_epsilon=current_epsilon,
            )

            if save_adapters:
                history, final_params = run_simulation(
                    client_fn=client_fn,
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    return_parameters=True,
                )
                if final_params is not None:
                    model_dir = Path("saved_models") / f"model_r{rank}_eps{eps_label}_alpha{alpha:.1f}"
                    try:
                        save_lora_adapter(
                            parameters_ndarrays=final_params,
                            rank=rank,
                            lora_alpha=config["lora"]["lora_alpha"],
                            lora_dropout=config["lora"]["lora_dropout"],
                            save_path=model_dir,
                        )
                    except Exception as e:
                        logger.warning("Could not save adapter to %s: %s", model_dir, e)
            else:
                history = run_simulation(
                    client_fn=client_fn,
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    return_parameters=False,
                )

            rows = extract_history_rows(history, rank, current_epsilon, alpha)
            all_rows.extend(rows)

            # Incremental save: write to CSV immediately so progress is never lost if interrupted
            if rows:
                try:
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                        writer.writeheader()
                        writer.writerows(all_rows)
                except Exception as e:
                    logger.warning("Could not incrementally write to %s: %s", csv_path, e)

            elapsed = time.time() - run_start
            cleanup_ray_temp()
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

    # Step 3: Write primary CSV and archive copy
    for target_path in (csv_path, archive_csv_path):
        with open(target_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
    logger.info("Results written to %s and archived to %s (%d rows)", csv_path, archive_csv_path, len(all_rows))

    # Step 4: Write summary JSON and archive copy
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
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
    for target_path in (summary_path, archive_summary_path):
        with open(target_path, "w") as f:
            json.dump(summary, f, indent=2)
    logger.info("Summary written to %s and %s", summary_path, archive_summary_path)

    # Step 5: Automatically sync with web dashboard data.js if available
    dashboard_data_js = Path("dashboard/data.js")
    if dashboard_data_js.parent.exists():
        try:
            with open(dashboard_data_js, "w") as f:
                f.write("window.INITIAL_SWEEP_DATA = " + json.dumps(all_rows, indent=2) + ";\n")
            logger.info("Synced sweep results to %s for the comparison dashboard", dashboard_data_js)
        except Exception as e:
            logger.warning("Could not sync to dashboard/data.js: %s", e)

    # Step 6: Print reference table
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
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag/name for this run (included in archive filename)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (defaults to results/)",
    )
    parser.add_argument(
        "--no-save-adapters",
        action="store_true",
        help="Skip saving trained LoRA adapters to saved_models/",
    )
    args = parser.parse_args()
    run_sweep(
        config_path=args.config,
        rounds_override=args.rounds,
        tag=args.tag,
        output_dir=args.output_dir,
        save_adapters=not args.no_save_adapters,
    )


if __name__ == "__main__":
    main()
