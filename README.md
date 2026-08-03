# Federated Learning in Healthcare

**Mapping where privacy, compression, and data imbalance collide in clinical NLP**

Anantveer Singh Beniwal (23BCE1748) · Shresth Kumar Gupta (23BCE1578) · Abhay Singh Khinchi (23BCE1075)

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
├── configs/
│   └── config.yaml            ← experiment sweep settings (ε, r, α, rounds)
├── requirements/
│   ├── base.txt               ← OS-agnostic packages
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
│       ├── client.py          ← Flower client (local train loop)
│       └── server.py          ← Flower server (aggregation strategy)
├── notebooks/                 ← exploratory analysis, result plots
├── data/                      ← raw/processed datasets (gitignored)
└── results/                   ← metrics, logs, benchmark tables (gitignored except summaries)
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

## 5. Running a simulation

```bash
python -m src.federated.server --config configs/config.yaml
```
(Skeleton — flesh out once data pipeline + model are ready. See TASKS below.)

## 6. Team task split

See [`TASKS.md`](./TASKS.md) for the two-track division of work.

## 7. Status

🚧 Early stage — architecture and repo scaffolded, implementation in progress.
