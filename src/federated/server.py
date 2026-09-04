from collections.abc import Callable
import logging
import math

import flwr as fl
from flwr.common import Metrics
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


def run_simulation(
    client_fn: Callable[[str], fl.client.Client | fl.client.NumPyClient],
    num_clients: int = 3,
    num_rounds: int = 5,
) -> History:
    """Launches the central server and coordinates the simulation.

    Args:
        client_fn: Factory function mapping client ID string to Client.
        num_clients: Total simulated hospital clients.
        num_rounds: Number of federated training rounds.

    Returns:
        Simulation history containing metrics and loss.
    """
    logger.info("Starting Federated Learning simulation with %d hospitals...", num_clients)

    strategy = fl.server.strategy.FedAvg(
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
        client_resources={"num_cpus": 2},
    )

    return history
