# Federated Learning in Healthcare

**Mapping where privacy, compression, and data imbalance collide in clinical NLP**

Shresth Kumar Gupta (23BCE1578) · Anantveer Singh Beniwal (23BCE1748) · Abhay Singh Khinchi (23BCE1075)

---

## 1. Overview

Hospitals want to train shared NLP models on patient reviews and clinical notes
without ever centralizing raw patient data. This project builds a **Federated
Learning (FL)** pipeline over a **DistilBERT** backbone, made bandwidth-efficient
with **LoRA** adapters and made privacy-preserving with **Differential Privacy
(DP-SGD)**.

**The problem we're studying:** existing FL+LoRA+DP frameworks are validated on
clean, class-balanced datasets. Real hospitals are not balanced — one hospital's
notes may skew 90% negative, another's 90% positive. This project empirically
maps how much the three-way combination of **DP noise + LoRA compression +
worst-case data heterogeneity** degrades model accuracy, and produces a
reference table:

```
Privacy Budget (ε)  →  LoRA Rank (r)  →  Target F1-Score
```

## 2. Architecture

| Layer | Tool | Role |
|---|---|---|
| Base model | `transformers` (DistilBERT, 66M params) | Feature extraction from clinical/review text |
| Adaptation | `peft` (LoRA) | Freezes base weights, trains low-rank matrices `A`, `B` so `ΔW = BA`, cutting trainable params >99% |
| Privacy | `opacus` (DP-SGD) | Per-sample gradient clipping + calibrated Gaussian noise before every update |
| Orchestration | `flwr` (Flower) | Simulates 3 hospital nodes, handles local training + weight aggregation |
| Data skew | Dirichlet distribution (`α`) | Mathematically partitions data non-IID across simulated hospitals |
| Engine | `PyTorch` | Tensor ops / backprop |

### Pipeline flow
```
Raw text (per-hospital, non-IID via Dirichlet α)
        │
        ▼
DistilBERT tokenizer + frozen backbone
        │
        ▼
LoRA adapters (A, B)  ── ΔW = BA, B initialized to 0
        │
        ▼
Opacus DP-SGD: clip per-sample grads (C) → add N(0, σ²C²) noise
        │
        ▼
Flower client sends compressed, noised LoRA update → server aggregates
        │
        ▼
Evaluation: F1-score per (ε, r, α) configuration
```

## 3. Repo structure

```
fl-healthcare-repo/
├── README.md                  ← you are here
├── main.py                    ← pipeline entry point (single-run simulation)
├── sweep.py                   ← grid runner for full (r, ε, α) benchmark sweep
├── configs/
│   └── config.yaml            ← experiment sweep settings (ε, r, α, rounds)
├── requirements/
│   ├── base.txt               ← OS-agnostic packages (flwr, peft, opacus, seaborn, etc.)
│   ├── mac.txt                ← macOS-specific (Apple Silicon / MPS)
│   ├── windows.txt            ← Windows-specific (CUDA or CPU)
│   └── linux.txt              ← Linux-specific (CUDA or CPU)
├── setup/
│   ├── setup_mac.sh
│   ├── setup_windows.ps1
│   └── setup_linux.sh
├── src/
│   ├── data/
│   │   └── partition.py       ← Dirichlet-based non-IID partitioning
│   ├── models/
│   │   └── lora_model.py      ← DistilBERT + LoRA via peft
│   ├── privacy/
│   │   └── dp_engine.py       ← Opacus wrapper (clipping + noise)
│   └── federated/
│       ├── client.py          ← Flower client (local DP-SGD train loop + 6 metrics)
│       └── server.py          ← Flower server (FedAvg aggregation strategy)
├── tests/                     ← zero-dependency unittest test suite
│   ├── test_partition.py
│   ├── test_lora_model.py
│   ├── test_dp_engine.py
│   └── test_server.py
├── notebooks/
│   └── analysis.ipynb         ← benchmark analysis: heatmaps, tradeoff curves, tables
├── data/                      ← raw/processed datasets (gitignored)
└── results/                   ← sweep results CSV, summaries, metrics (gitignored)
```

## 4. Setup

Python **3.10 or 3.11** recommended (Opacus/PEFT compatibility). Use a virtual
environment either way.

### macOS

```bash
bash setup/setup_mac.sh
```
Installs PyTorch with **MPS** (Apple Silicon GPU) support where available, falls
back to CPU on Intel Macs. See `requirements/mac.txt`.

### Windows

```powershell
.\setup\setup_windows.ps1
```
Installs CUDA-enabled PyTorch if an NVIDIA GPU + CUDA toolkit is detected,
otherwise CPU build. See `requirements/windows.txt`. Run this from **PowerShell**,
not cmd.exe. If script execution is blocked, run once as admin:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Linux

```bash
bash setup/setup_linux.sh
```
See `requirements/linux.txt`.

### Why OS-specific files at all?
`torch` needs a different install index depending on OS/GPU (CUDA build on
Windows/Linux with NVIDIA GPUs, MPS build on Apple Silicon, CPU-only otherwise).
Everything else (`flwr`, `transformers`, `peft`, `opacus`, etc.) is identical
across platforms and lives in `requirements/base.txt`.

## 5. Running Experiments

### Single Simulation Run

Execute a single federated learning simulation using `main.py`:

```bash
# Default parameters from configs/config.yaml
python main.py

# Custom single-run configuration with CLI overrides
python main.py --rank 4 --epsilon 8.0 --alpha 0.5 --rounds 5

# Baseline run without differential privacy
python main.py --rank 8 --epsilon null --alpha 1.0 --rounds 5
```

### Full Benchmark Sweep

Run the full grid sweep over all rank ($r$), privacy budget ($\varepsilon$), and heterogeneity ($\alpha$) combinations:

```bash
# Full sweep across all combinations in configs/config.yaml
python sweep.py

# Fast validation sweep (2 rounds per configuration)
python sweep.py --rounds 2
```

Outputs are saved to `results/sweep_results.csv` and `results/sweep_summary.json`.

### Evaluation & Analysis

Open and run `notebooks/analysis.ipynb` to visualize benchmark outputs:
- **Reference Table:** Performance breakdown across privacy budgets and LoRA ranks.
- **Metric Heatmaps:** F1-score, MCC, and AUC-ROC across $(r, \varepsilon)$ configurations.
- **Tradeoff Curves:** Accuracy vs. privacy budget $(\varepsilon)$ and rank $(r)$.
- **Convergence Trajectories:** Federated round-by-round training loss.

### Running Unit Tests

Run the zero-dependency test suite:

```bash
python -m unittest discover tests/ -v
```

## 6. Architecture & System Tracks

The system is structured into two core engineering tracks:
- **Track A (Privacy & Data):** Non-IID Dirichlet data partitioning with sample-count recovery safeguards (`src/data/partition.py`), alongside per-sample gradient clipping and calibrated Gaussian DP-SGD noise engines (`src/privacy/dp_engine.py`).
- **Track B (Model & Federation):** Parameter-efficient DistilBERT fine-tuning via low-rank adapters (`src/models/lora_model.py`), coordinated over a multi-client Flower federation simulation with comprehensive metric tracking (Accuracy, Precision, Recall, F1, MCC, AUC-ROC) (`src/federated/`).

## 7. Status

✅ **Pipeline Functional & Tested** — Core architecture implemented with zero-dependency unit tests passing, end-to-end simulation validated, grid sweep infrastructure operational, and analysis visualization notebook configured.
