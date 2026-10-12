from collections.abc import Callable
import logging
import math

import flwr as fl
from flwr.common import Context, Metrics
from flwr.server.history import History

logger = logging.getLogger(__name__)


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregates evaluation metrics from all hospitals.

    Args:
        metrics: List of (num_examples, metrics_dict) from client evaluations.

    Returns:
        Aggregated metrics dictionary with weighted averages.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Metrics = {}
    for key in ("accuracy", "precision", "recall", "f1", "mcc", "auc_roc"):
        valid_items = [
            (num_examples, float(m[key]))
            for num_examples, m in metrics
            if key in m and isinstance(m[key], (int, float)) and not math.isnan(float(m[key]))
        ]
        key_examples = sum(num_examples for num_examples, _ in valid_items)
        aggregated[key] = (
            sum(num_examples * val for num_examples, val in valid_items) / key_examples
            if key_examples > 0
            else 0.0
        )
    return aggregated


import numpy as np
from flwr.common import parameters_to_ndarrays


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that captures the final aggregated parameter tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters: fl.common.Parameters | None = None

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.final_parameters = parameters
        return parameters, metrics


def run_simulation(
    client_fn: Callable[[Context], fl.client.Client | fl.client.NumPyClient] | Callable[[str], fl.client.Client | fl.client.NumPyClient],
    num_clients: int = 3,
    num_rounds: int = 5,
    return_parameters: bool = False,
) -> History | tuple[History, list[np.ndarray] | None]:
    """Launches the central server and coordinates the simulation.

    Args:
        client_fn: Factory function mapping Flower Context to Client.
        num_clients: Total simulated hospital clients.
        num_rounds: Number of federated training rounds.
        return_parameters: If True, returns a tuple of (History, list[np.ndarray] | None).

    Returns:
        Simulation history, or (history, final_parameters) if return_parameters=True.
    """
    logger.info("Starting Federated Learning simulation with %d hospitals...", num_clients)

    strategy = SaveModelFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2.5},
        ray_init_args={
            "num_cpus": 8,
            "object_store_memory": 1024 * 1024 * 1024,
            "include_dashboard": False,
        },
    )

    if return_parameters:
        final_ndarrays = (
            parameters_to_ndarrays(strategy.final_parameters)
            if strategy.final_parameters is not None
            else None
        )
        return history, final_ndarrays

    return history
