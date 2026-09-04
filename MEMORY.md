# Project Memory

## Overview

**Federated Learning in Healthcare** — empirically mapping how DP noise + LoRA compression + data heterogeneity interact when training a shared clinical NLP model across hospitals that never share raw data.

**Goal output:** A reference table: `Privacy Budget (ε) → LoRA Rank (r) → Dirichlet α → F1-Score`.

**Team:** Anantveer Singh Beniwal, Shresth Kumar Gupta, Abhay Singh Khinchi.

---

## Tech Stack

| Layer | Library | Version constraint |
|---|---|---|
| Base model | `transformers` (DistilBERT-base-uncased, 66M params) | — |
| Adaptation | `peft` (LoRA, `TaskType.SEQ_CLS`) | — |
| Privacy | `opacus` (DP-SGD via `PrivacyEngine`) | — |
| Federation | `flwr[simulation]` (Flower) | — |
| Data skew | Dirichlet via `numpy` | — |
| Engine | `PyTorch` | MPS-first, CPU fallback |
| Dataset | `health_fact` from HF Hub (binary: verified/unverified) | Falls back to synthetic 500-sample set |
| Config | `pyyaml` — single `configs/config.yaml` | — |
| Python | 3.10 or 3.11 | Opacus/PEFT compat |

---

## Module Map

```
main.py                          — Master orchestrator (CLI + 4-step pipeline)
src/data/partition.py            — download_and_tokenize(), partition_data(), create_client_dataloaders()
src/models/lora_model.py         — setup_lora_model() → PeftModel wrapping DistilBERT
src/privacy/dp_engine.py         — validate_and_fix_model(), make_private(), get_epsilon_spent()
src/federated/client.py          — HealthcareClient(NumPyClient) — fit/evaluate/get_parameters/set_parameters
src/federated/server.py          — run_simulation(), weighted_average() metric aggregator
configs/config.yaml              — Sweep grid: ranks, epsilons, alphas, training params
```

---

## Settled Design Decisions

| Date | Decision |
|---|---|
| 2026-09 | LoRA targets `q_lin` + `v_lin` (DistilBERT attention projections). |
| 2026-09 | Binary classification: `health_fact` labels collapsed to verified (0) / unverified (1). |
| 2026-09 | 3 simulated hospital clients, `FedAvg` aggregation, 100% client participation per round. |
| 2026-09 | Opacus `make_private_with_epsilon()` used (auto-calibrates σ from target ε). |
| 2026-09 | `main.py` runs a **single (r, ε, α)** config per invocation; full grid sweep is N serial runs. |
| 2026-09 | Synthetic fallback dataset (4 templates × 125 = 500 samples) when HF Hub is unreachable. |
| 2026-09 | Client evaluation returns aggregate 6-metric suite: accuracy, precision, recall, F1, MCC, AUC-ROC. |
| 2026-09 | Device parameter passed directly from main orchestrator into HealthcareClient with fallback. |
| 2026-09 | Standardized unit tests using Python's built-in `unittest` in `tests/` (17 tests covering partition, LoRA, DP engine, and server aggregation). |
| 2026-09 | Server metric aggregation strictly casts Scalar metrics to float with presence/type guards and filters out NaNs (e.g. from single-class validation skew) to prevent global metric poisoning. |
| 2026-09 | Dirichlet data partitioner includes retry loop ensuring all clients receive at least `min_samples`, preventing empty partition runtime errors on small datasets or extreme alphas. |
| 2026-09 | Workspace pyrefly configured via `pyrefly.toml` with `python-interpreter-path` pointing to `.venv`. |

---

## Sweep Grid (from config.yaml)

- **LoRA ranks (r):** `[2, 4, 8, 16]`
- **Privacy budgets (ε):** `[1, 3, 8, null]` (null = no DP baseline)
- **Dirichlet α:** `[0.1, 1.0, 100.0]` (0.1 = extreme skew, 100 = ~IID)
- **Rounds:** 5, **Local epochs:** 1, **Batch size:** 16, **LR:** 5e-5, **Seed:** 42

---

## Known Issues / Tech Debt

- No full sweep runner / results aggregation script.
- `notebooks/` and `results/` are empty (only `.gitkeep`).
