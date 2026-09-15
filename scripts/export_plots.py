import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def main():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11

    csv_path = Path("results/sweep_results.csv")
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Cast numeric columns
    for col in ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc", "loss"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("−", "-"), errors="coerce")

    df["epsilon_label"] = df["epsilon"].apply(
        lambda x: "∞ (no DP)" if str(x).lower() in ("none", "null", "nan") else f"ε={x}"
    )

    max_round = df["round"].max()
    final_df = df[df["round"] == max_round].copy()

    # 1. Plot Heatmaps
    metrics_to_plot = [
        ("f1", "F1-Score", "YlGnBu"),
        ("mcc", "Matthews Corr Coef (MCC)", "magma"),
        ("auc_roc", "AUC-ROC", "cividis")
    ]

    for alpha_val in sorted(final_df["alpha"].unique()):
        alpha_subset = final_df[final_df["alpha"] == alpha_val]
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 4.5))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, (metric_col, metric_title, cmap_name) in zip(axes, metrics_to_plot):
            pivot_data = alpha_subset.pivot_table(
                index="epsilon_label",
                columns="rank",
                values=metric_col,
                aggfunc="mean"
            )
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".3f",
                cmap=cmap_name,
                cbar=True,
                ax=ax,
                linewidths=1.0,
                vmin=0.0,
                vmax=1.0
            )
            ax.set_title(f"{metric_title} (α={alpha_val})", fontweight="bold")
            ax.set_xlabel("LoRA Rank (r)")
            ax.set_ylabel("Privacy Budget (ε)")

        plt.tight_layout()
        plot_path = out_dir / f"heatmap_alpha_{alpha_val}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")

    # 2. Convergence & Loss Trajectory
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.lineplot(
        data=df,
        x="round",
        y="loss",
        hue="epsilon_label",
        style="rank",
        markers=True,
        dashes=False,
        markersize=9,
        ax=axes[0]
    )
    axes[0].set_title("Training Loss Convergence across Rounds", fontweight="bold")
    axes[0].set_xlabel("Federation Round")
    axes[0].set_ylabel("Aggregated Validation Loss")

    sns.lineplot(
        data=df,
        x="round",
        y="auc_roc",
        hue="epsilon_label",
        style="rank",
        markers=True,
        dashes=False,
        markersize=9,
        ax=axes[1]
    )
    axes[1].set_title("AUC-ROC Improvement across Rounds", fontweight="bold")
    axes[1].set_xlabel("Federation Round")
    axes[1].set_ylabel("Aggregated AUC-ROC")
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plot_path = out_dir / "convergence_trajectories.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

    print("All plots generated successfully in results/plots/")

if __name__ == "__main__":
    main()
