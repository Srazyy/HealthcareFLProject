"""
Flower server: spawns the 3-node simulation and aggregates LoRA updates.

Owner: Track B

Run with:
    python -m src.federated.server --config configs/config.yaml

TODO:
- Read sweep params (epsilon, r, alpha, num_rounds) from configs/config.yaml.
- Implement/choose an aggregation strategy (start with FedAvg).
- Log per-round F1 to results/ for building the benchmark table.
"""

import argparse

import flwr as fl

from src.federated.client import make_client_fn


def main(num_clients: int = 3, num_rounds: int = 5, r: int = 8):
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(r=r),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--r", type=int, default=8)
    args = parser.parse_args()

    # TODO: actually load args.config and override defaults below
    main(num_clients=args.num_clients, num_rounds=args.num_rounds, r=args.r)
