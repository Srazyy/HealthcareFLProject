# Task Split — First Sprint

Two parallel tracks so you're not blocked on each other. Both meet in the middle
at `src/federated/client.py`, which needs *both* the model and the data pipeline.

## Track A — Data + Privacy
**Owns:** `src/data/partition.py`, `src/privacy/dp_engine.py`

1. Source/simulate a hospital review or clinical-notes text dataset (start with
   something public like a healthcare review dataset or IMDB-style sentiment
   data as a stand-in if real clinical text isn't available yet).
2. Implement Dirichlet-based non-IID partitioning (`α` parameter) to split data
   across 3 simulated hospitals with controllable skew.
3. Wrap Opacus `PrivacyEngine` around a standard PyTorch optimizer: implement
   per-sample gradient clipping (`C`) + Gaussian noise injection (`σ`).
4. Write a small standalone script that trains a plain classifier (no FL yet)
   under different `(ε, C)` settings, just to sanity-check the DP engine works
   and produces a sensible privacy/accuracy tradeoff curve.

**Deliverable for sprint 1:** a function `partition_data(dataset, num_clients, alpha)`
and a function `make_private(model, optimizer, dataloader, epsilon, ...)` that
Track B can import directly.

## Track B — Model + Federation
**Owns:** `src/models/lora_model.py`, `src/federated/client.py`, `src/federated/server.py`

1. Load DistilBERT from `transformers`, attach LoRA adapters via `peft`
   (target the self-attention query/value projections first — standard LoRA
   practice).
2. Confirm `B` initializes to zero and `ΔW = BA` is genuinely near-zero at
   init (sanity check from the slides).
3. Set up a minimal Flower `NumPyClient` (or `Client`) that can `fit()` /
   `evaluate()` on a local shard of data — stub the data with random tensors
   until Track A's partitioning is ready.
4. Set up a Flower server with `flwr.simulation` spawning 3 clients on one
   machine, using `FedAvg` (or similar) as the starting aggregation strategy.

**Deliverable for sprint 1:** a working 3-node Flower simulation that trains a
LoRA-adapted DistilBERT for a few rounds on dummy/local data, end to end,
even before DP is wired in.

## Merge point (sprint 2)
Once both tracks have their piece working standalone:
- Plug Track A's `make_private()` into Track B's local training loop inside
  `client.py` (this is where Opacus intercepts the PyTorch optimizer).
- Plug Track A's `partition_data()` into the simulation's data loading so each
  Flower client gets its correctly-skewed shard.
- Run the first real sweep across `(ε, r, α)` and start filling in
  `results/`.

## Suggested first standup questions
- What text dataset are we actually using — real clinical/review data, or a
  stand-in for now?
- Rank `r` values to sweep (slides imply this is a key axis — start with
  r ∈ {2, 4, 8, 16}).
- ε values to sweep (e.g. {1, 3, 8, ∞/no-DP as baseline}).
