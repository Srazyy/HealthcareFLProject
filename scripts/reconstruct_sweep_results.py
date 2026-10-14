#!/usr/bin/env python3
"""
Consolidates sweep results for configs 1 through 41:
- Extracts full 5-round metrics and loss for configs 1 to 27 from results/sweep_run.log
- Preserves full 5-round metrics and loss for configs 28 to 41 from results/sweep_results.csv
- Backs up existing sweep_results.csv to results/history/
- Writes clean 205-row ledger to results/sweep_results.csv
"""

import ast
import csv
from itertools import product
from pathlib import Path
import re
import shutil
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

config = yaml.safe_load(open(REPO_ROOT / "configs/full_sweep_config.yaml"))
ranks = config["lora"]["ranks"]
epsilons = config["privacy"]["epsilons"]
alphas = config["data"]["dirichlet_alphas"]

all_configs = []
for alpha in alphas:
    for rank, raw_eps in product(ranks, epsilons):
        eps_label = f"{float(raw_eps):.1f}" if raw_eps is not None else "none"
        all_configs.append((rank, raw_eps, alpha, eps_label))

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "mcc", "auc_roc")
CSV_COLUMNS = ["rank", "epsilon", "alpha", "round", *METRIC_KEYS, "loss"]

# 1. Parse configs 1 to 27 from results/sweep_run.log
log_path = REPO_ROOT / "results" / "sweep_run.log"
content = log_path.read_text(errors="ignore")
idx_28 = content.find("[28/48] r=8, ε=none, α=1.00 — starting 5-round simulation...")
early_content = content[:idx_28]

unique_hists = []
for h in re.finditer(r"History \(metrics, distributed, evaluate\):", early_content):
    if not unique_hists or h.start() - unique_hists[-1].start() > 200:
        unique_hists.append(h)

if len(unique_hists) != 27:
    raise RuntimeError(f"Expected 27 history blocks before run 28, found {len(unique_hists)}")

recovered_rows = []
for idx in range(27):
    rank, raw_eps, alpha, eps_label = all_configs[idx]
    h = unique_hists[idx]
    loss_chunk = early_content[max(0, h.start() - 1500): h.start()]
    round_losses = dict(re.findall(r"round (\d+):\s*([0-9.]+)", loss_chunk))

    after_chunk = early_content[h.end(): h.end() + 5000]
    metrics = {}
    for key in METRIC_KEYS:
        km = re.search(r"[\x27\"]" + key + r"[\x27\"]\s*:\s*(\[\s*\(\s*1\s*,.*?\])", after_chunk, re.DOTALL)
        if km:
            clean = re.sub(r"\d{2}/\d{2}/\d{4}.*?INFO:\s*", "", km.group(1))
            clean = re.sub(r"INFO\s*:\s*", "", clean)
            end_b = clean.find("]")
            if end_b != -1:
                clean = clean[:end_b + 1]
            tups = re.findall(r"\(\s*(\d+)\s*,\s*([0-9.eE+-]+|nan|None)\s*\)", clean)
            metrics[key] = {int(r): float(v) if v not in ("nan", "None") else 0.0 for r, v in tups}

    for r in range(1, 6):
        row = {
            "rank": rank,
            "epsilon": eps_label,
            "alpha": alpha,
            "round": r,
            "loss": round_losses.get(str(r), ""),
        }
        for k in METRIC_KEYS:
            row[k] = metrics.get(k, {}).get(r, "")
        recovered_rows.append(row)

# 2. Existing 5-round rows for configs 28..41 from CSV
csv_path = REPO_ROOT / "results" / "sweep_results.csv"
existing_rows = []
with open(csv_path, newline="") as f:
    existing_rows = list(csv.DictReader(f))

c28_41_rows = []
for idx in range(27, 41):
    rank, raw_eps, alpha, eps_label = all_configs[idx]
    matching = [r for r in existing_rows if int(r["rank"]) == rank and str(r["epsilon"]) == eps_label and abs(float(r["alpha"]) - alpha) < 1e-4]
    if len(matching) == 5:
        c28_41_rows.extend(matching)
    elif len(matching) > 5:
        c28_41_rows.extend(matching[-5:])
    else:
        raise RuntimeError(f"Config {idx+1} (r={rank}, eps={eps_label}, alpha={alpha}) only has {len(matching)} rows in CSV")

total_41_rows = recovered_rows + c28_41_rows

# 3. Backup existing CSV
history_dir = REPO_ROOT / "results" / "history"
history_dir.mkdir(parents=True, exist_ok=True)
backup_path = history_dir / "sweep_results_pre_resume_backup.csv"
shutil.copy2(csv_path, backup_path)
print(f"Backed up {csv_path} -> {backup_path}")

# 4. Write consolidated CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(total_41_rows)

print(f"Successfully consolidated {len(total_41_rows)} rows across 41 configurations into {csv_path}!")
