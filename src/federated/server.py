import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregates the evaluation metrics from all hospitals.
    If Hospital A has 1000 patients and Hospital B has 100, 
    Hospital A's accuracy carries more weight in the final calculation.
    """
    # Multiply each hospital's accuracy by its number of data samples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Return the weighted average
    return {"accuracy": sum(accuracies) / sum(examples)}


def run_simulation(client_fn, num_clients=3, num_rounds=5):
    """
    Launches the central server and coordinates the simulation.
    """
    print(f"Starting Federated Learning simulation with {num_clients} hospitals...")

    # 1. Define the strategy (FedAvg is the industry standard)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,                  # Train on 100% of available clients each round
        fraction_evaluate=1.0,             # Evaluate on 100% of clients each round
        min_fit_clients=num_clients,       # Wait for all hospitals to be ready before training
        min_evaluate_clients=num_clients,  # Wait for all hospitals to be ready before evaluating
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average, # Use our custom math above
    )

    # 2. Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,               # A factory function that spawns your HealthcareClient
        num_clients=num_clients,           # Total simulated hospitals
        config=fl.server.ServerConfig(num_rounds=num_rounds), # How many times they sync
        strategy=strategy,                 # The FedAvg strategy defined above
        client_resources={"num_cpus": 2}   # Allocate CPU cores per simulated hospital
    )
    
    return history