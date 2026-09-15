"""
Master Runner for Complete Healthcare FL Benchmark Analysis.

Executes the entire end-to-end evaluation pipeline:
  1. Generates 3,000 balanced clinical claim records (data/medical_claims_3k.parquet & .csv)
  2. Computes Dirichlet non-IID split distributions and outputs proof chart (results/plots/non_iid_partitions.png)
  3. Executes the full FL + LoRA + DP grid sweep (sweep.py -> results/sweep_results.csv & sweep_summary.json)
  4. Generates publication-ready figures (results/plots/heatmap_*.png, convergence_trajectories.png, tradeoff_curves.png)
  5. Refreshes the Jupyter analysis notebook with latest executed cells

Usage:
  python scripts/run_complete_analysis.py
  python scripts/run_complete_analysis.py --rounds 2
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON_EXEC = sys.executable


def run_step(step_name: str, cmd: list[str]):
    print("\n" + "=" * 75)
    print(f"  [STEP] {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 75)
    start_t = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with return code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"  --> Completed in {time.time() - start_t:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run complete Healthcare FL benchmark analysis pipeline.")
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds per configuration (default: from config.yaml)")
    args = parser.parse_args()

    total_start = time.time()
    print("*" * 75)
    print("  HEALTHCARE FL BENCHMARK — COMPLETE END-TO-END ANALYSIS SUITE")
    print(f"  Repository: {REPO_ROOT}")
    print("*" * 75)

    # 1. Dataset Generation
    run_step(
        "1. Generate 3,000 Clinical Claim Records",
        [PYTHON_EXEC, "scripts/generate_dataset.py"],
    )

    # 2. Non-IID Dirichlet Proof & Visualization
    run_step(
        "2. Compute Non-IID Proof & Plot Distributions",
        [PYTHON_EXEC, "scripts/verify_non_iid.py"],
    )

    # 3. Federated Learning Benchmark Sweep
    sweep_cmd = [PYTHON_EXEC, "sweep.py"]
    if args.rounds:
        sweep_cmd.extend(["--rounds", str(args.rounds)])
    run_step(
        "3. Execute Federated Sweep Grid",
        sweep_cmd,
    )

    # 4. Export Figures (Heatmaps, Trajectories, Tradeoffs)
    run_step(
        "4. Export High-Resolution Plots",
        [PYTHON_EXEC, "scripts/export_plots.py"],
    )

    # 5. Notebook Execution / Refresh
    try:
        run_step(
            "5. Execute Analysis Notebook",
            [PYTHON_EXEC, "-m", "jupyter", "execute", "--kernel_name", "fl-healthcare", "--inplace", "notebooks/analysis.ipynb"],
        )
    except Exception as e:
        print(f"[Warning] Jupyter execute encountered: {e}. Notebook can be viewed directly.")

    print("\n" + "*" * 75)
    print(f"  COMPLETE ANALYSIS FINISHED SUCCESSFULLY IN {time.time() - total_start:.1f}s!")
    print("*" * 75)
    print("\nWHERE TO VIEW YOUR RESULTS & GRAPHS:")
    print("  1. Visual Charts & Figures (PNG):")
    print("     - results/plots/non_iid_partitions.png     (Proof of Dirichlet Non-IID splits)")
    print("     - results/plots/heatmap_alpha_*.png       (F1, MCC, AUC-ROC heatmaps per alpha)")
    print("     - results/plots/tradeoff_curves.png       (Privacy-utility and rank tradeoff barplots)")
    print("     - results/plots/convergence_trajectories.png (Loss and AUC-ROC curves over rounds)")
    print("  2. Tabular Data & Metrics:")
    print("     - results/sweep_results.csv               (Full raw metrics across all rounds/configs)")
    print("     - results/sweep_summary.json              (High-level run summary)")
    print("     - data/medical_claims_3k.csv              (Human-readable clinical claims dataset)")
    print("  3. Interactive Notebook:")
    print("     - notebooks/analysis.ipynb                (Jupyter notebook with rendered tables & plots)\n")


if __name__ == "__main__":
    main()
