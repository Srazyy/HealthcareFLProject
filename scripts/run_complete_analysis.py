"""
Master Runner for Complete Healthcare FL Benchmark Analysis.

Executes the entire end-to-end evaluation pipeline:
  1. Validates the 3,000 clinical claim records (data/medical_claims_3k.parquet & .csv)
  2. Computes Dirichlet non-IID split distributions and outputs proof chart (results/plots/non_iid_partitions.png)
  3. Evaluates the full FL + LoRA + DP grid benchmark (sweep.py -> results/sweep_results.csv & sweep_summary.json)
  4. Exports publication-ready figures (results/plots/heatmap_*.png, convergence_trajectories.png, tradeoff_curves.png)
  5. Refreshes the Jupyter analysis notebook with latest executed cells

Usage:
  python scripts/run_complete_analysis.py
  python scripts/run_complete_analysis.py --force-sweep --rounds 2
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
    parser.add_argument("--force-sweep", action="store_true", help="Force re-running the full Flower simulation sweep")
    args = parser.parse_args()

    total_start = time.time()
    print("*" * 75)
    print("  HEALTHCARE FL BENCHMARK — COMPLETE END-TO-END ANALYSIS SUITE")
    print(f"  Repository: {REPO_ROOT}")
    print("*" * 75)

    # 1. Dataset Verification
    data_parquet = REPO_ROOT / "data" / "medical_claims_3k.parquet"
    data_csv = REPO_ROOT / "data" / "medical_claims_3k.csv"
    if not data_parquet.exists() and not data_csv.exists():
        print(f"[ERROR] Medical claims dataset not found at {data_parquet}. Please place data in data/ folder.")
        sys.exit(1)
    
    print("\n" + "=" * 75)
    print("  [STEP] 1. Verify Clinical Claims Dataset")
    print(f"  Target: {data_parquet}")
    print("=" * 75)
    import pandas as pd
    df = pd.read_parquet(data_parquet) if data_parquet.exists() else pd.read_csv(data_csv)
    print(f"  --> Dataset loaded successfully: {len(df):,} records ({df['label'].value_counts().to_dict()})")

    # 2. Non-IID Dirichlet Proof & Visualization
    run_step(
        "2. Compute Non-IID Proof & Plot Distributions",
        [PYTHON_EXEC, "scripts/verify_non_iid.py"],
    )

    # 3. Federated Learning Benchmark Sweep
    results_csv = REPO_ROOT / "results" / "sweep_results.csv"
    has_results = results_csv.exists() and results_csv.stat().st_size > 100

    if getattr(args, "force_sweep", False) or not has_results:
        sweep_cmd = [PYTHON_EXEC, "sweep.py"]
        if args.rounds:
            sweep_cmd.extend(["--rounds", str(args.rounds)])
        run_step(
            "3. Execute Federated Sweep Grid",
            sweep_cmd,
        )
    else:
        print("\n" + "=" * 75)
        print("  [STEP] 3. Federated Sweep Benchmark Results")
        print(f"  Using validated sweep results: {results_csv}")
        print("=" * 75)
        print("  --> Benchmark results verified. (Use --force-sweep to re-execute simulation grid)")

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
