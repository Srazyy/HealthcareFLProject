"""
Dirichlet-based non-IID data partitioning across simulated hospitals.

Owner: Track A

TODO:
- Load the actual dataset (patient reviews / clinical notes, or a stand-in
  public dataset for now).
- Implement partition_data() below.
- Add a quick script/plot showing class distribution per client for a given
  alpha, to visually confirm the skew (e.g. alpha=0.1 -> highly skewed,
  alpha=100 -> near-IID).
"""

import numpy as np


def partition_data(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42):
    """
    Partition sample indices across `num_clients` using a Dirichlet
    distribution over class proportions, controlled by `alpha`.

    Lower alpha -> more skewed / non-IID (matches the "Hospital A: 90%
    negative, Hospital B: 90% positive" scenario from the project brief).
    Higher alpha -> closer to IID.

    Args:
        labels: 1D array of class labels for the full dataset.
        num_clients: number of simulated hospitals.
        alpha: Dirichlet concentration parameter.
        seed: RNG seed for reproducibility.

    Returns:
        List[np.ndarray]: one array of sample indices per client.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # convert proportions to split points
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)

        for client_id, idx in enumerate(splits):
            client_indices[client_id].extend(idx.tolist())

    return [np.array(idx) for idx in client_indices]


if __name__ == "__main__":
    # quick smoke test with synthetic labels
    fake_labels = np.random.default_rng(0).integers(0, 2, size=1000)
    parts = partition_data(fake_labels, num_clients=3, alpha=0.3)
    for i, idx in enumerate(parts):
        pos_rate = fake_labels[idx].mean() if len(idx) else float("nan")
        print(f"client {i}: n={len(idx)}, positive_rate={pos_rate:.2f}")
