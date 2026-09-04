# Progress

## Current Milestone: Pipeline Hardening & Metrics

The core end-to-end pipeline (`main.py` → partition → model → DP → Flower simulation) is assembled and runnable. Focus now shifts to cleaning up tech debt, adding F1-score evaluation (the project's primary metric), and building the sweep infrastructure to generate the reference table.

---

### Checklist

- [x] Scaffold repo structure and setup scripts (mac/win/linux)
- [x] Implement Dirichlet non-IID partitioning (`partition_data`)
- [x] Implement `download_and_tokenize` with `health_fact` + synthetic fallback
- [x] Implement `create_client_dataloaders` (train/val split per client)
- [x] Implement LoRA model setup (`setup_lora_model` → DistilBERT + PEFT)
- [x] Implement Opacus DP wrapper (`validate_and_fix_model`, `make_private`, `get_epsilon_spent`)
- [x] Implement Flower client (`HealthcareClient` — fit/evaluate)
- [x] Implement Flower server (`run_simulation` with FedAvg + weighted avg metrics)
- [x] Build master orchestrator (`main.py` — config loading, CLI overrides, 4-step pipeline)
- [x] Add 6-metric suite (F1, precision, recall, MCC, AUC-ROC, accuracy) to client evaluate() and server aggregation
- [x] Unify device detection (pass device into HealthcareClient)
- [x] Clean stale TODOs from partition.py, dp_engine.py, config.yaml
- [x] Add unit tests across all modules (partition, LoRA, DP engine, server aggregation — 17 tests passing)
- [x] Harden server.py and partition.py (Dirichlet min-sample guards, NaN-safe metric aggregation, pyrefly LSP config)
- [ ] Current: Build sweep runner script to iterate over full (r, ε, α) grid and aggregate results
- [ ] Create analysis notebook in notebooks/ — plots for the reference table

---

### Blockers

- None currently. Pipeline runs end-to-end on synthetic data. HF `health_fact` dataset availability is intermittent (fallback works).
