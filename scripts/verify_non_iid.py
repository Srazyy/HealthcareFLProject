"""
Verification & Proof Script for Dirichlet Non-IID Hospital Data Splits.

Proves mathematically and visually how Dirichlet alpha values (0.1, 1.0, 100.0)
partition clinical data non-IID across simulated hospitals.
Outputs:
  - Markdown distribution proof table
  - results/plots/non_iid_partitions.png
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.partition import partition_data


def main():
    parquet_path = REPO_ROOT / "data" / "medical_claims_3k.parquet"
    out_dir = REPO_ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist in data/.")
        return

    df = pd.read_parquet(parquet_path)
    labels = np.array(df["label"])
    num_clients = 3
    alphas = [0.1, 1.0, 100.0]
    seed = 42

    print("=" * 70)
    print("  MATHEMATICAL PROOF OF DIRICHLET NON-IID PARTITIONING")
    print(f"  Total Dataset: {len(df)} medical claims (50% verified, 50% unverified)")
    print("=" * 70)

    plot_records = []

    for alpha in alphas:
        print(f"\n--- Dirichlet Alpha = {alpha} ({'Extreme Skew' if alpha == 0.1 else 'Moderate Heterogeneity' if alpha == 1.0 else 'Uniform IID'}) ---")
        client_indices = partition_data(labels, num_clients=num_clients, alpha=alpha, seed=seed)

        for client_id, idx in enumerate(client_indices):
            n = len(idx)
            n_neg = int(np.sum(labels[idx] == 0))
            n_pos = int(np.sum(labels[idx] == 1))
            pct_neg = 100.0 * n_neg / n if n > 0 else 0.0
            pct_pos = 100.0 * n_pos / n if n > 0 else 0.0

            print(
                f"  Hospital {client_id}: {n:4d} samples | "
                f"Verified (0): {n_neg:4d} ({pct_neg:5.1f}%) | "
                f"Unverified (1): {n_pos:4d} ({pct_pos:5.1f}%)"
            )

            plot_records.append({
                "alpha": f"α = {alpha}",
                "hospital": f"Hospital {client_id}",
                "class": "Verified (0)",
                "count": n_neg,
                "percentage": pct_neg,
            })
            plot_records.append({
                "alpha": f"α = {alpha}",
                "hospital": f"Hospital {client_id}",
                "class": "Unverified (1)",
                "count": n_pos,
                "percentage": pct_pos,
            })

    # Generate visual plot
    plot_df = pd.DataFrame(plot_records)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    palette = {"Verified (0)": "#2b5c8f", "Unverified (1)": "#d95f02"}

    for ax, alpha in zip(axes, alphas):
        sub_df = plot_df[plot_df["alpha"] == f"α = {alpha}"]
        sns.barplot(
            data=sub_df,
            x="hospital",
            y="percentage",
            hue="class",
            palette=palette,
            ax=ax,
            edgecolor="black",
            linewidth=0.8,
        )
        subtitle = "Extreme Skew" if alpha == 0.1 else "Moderate Non-IID" if alpha == 1.0 else "Uniform IID"
        ax.set_title(f"α = {alpha} ({subtitle})", fontweight="bold")
        ax.set_xlabel("Hospital Node")
        ax.set_ylabel("Class Percentage (%)")
        ax.set_ylim(0, 105)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.6, label="50% Balanced Line")

        # Label percentages above bars
        for p in ax.patches:
            height = p.get_height()
            if height > 4:
                ax.annotate(
                    f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2.0, height / 2.0),
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=9,
                )

    plt.suptitle("Proof of Non-IID Dirichlet Partitioning across Distributed Hospitals", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = out_dir / "non_iid_partitions.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Proof Plot Exported]: {out_file}")


if __name__ == "__main__":
    main()
