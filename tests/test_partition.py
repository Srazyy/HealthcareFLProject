import unittest
import numpy as np
from src.data.partition import partition_data


class TestPartition(unittest.TestCase):
    def test_partition_covers_all_indices(self):
        labels = np.array([0, 1] * 100)
        num_clients = 3
        client_indices = partition_data(labels, num_clients=num_clients, alpha=0.5, seed=42)
        
        all_assigned = np.concatenate(client_indices)
        self.assertEqual(len(all_assigned), len(labels))
        self.assertEqual(set(all_assigned), set(range(len(labels))))

    def test_partition_respects_num_clients(self):
        labels = np.array([0, 1] * 50)
        num_clients = 4
        client_indices = partition_data(labels, num_clients=num_clients, alpha=1.0, seed=42)
        self.assertEqual(len(client_indices), num_clients)

    def test_low_alpha_produces_skew(self):
        labels = np.array([0] * 500 + [1] * 500)
        client_indices = partition_data(labels, num_clients=3, alpha=0.01, seed=42)
        
        pos_rates = []
        for idx in client_indices:
            if len(idx) > 0:
                pos_rates.append(labels[idx].mean())
                
        self.assertTrue(any(rate > 0.8 or rate < 0.2 for rate in pos_rates))

    def test_high_alpha_near_iid(self):
        labels = np.array([0] * 500 + [1] * 500)
        client_indices = partition_data(labels, num_clients=3, alpha=1000.0, seed=42)
        
        pos_rates = [labels[idx].mean() for idx in client_indices if len(idx) > 0]
        for rate in pos_rates:
            self.assertTrue(0.35 <= rate <= 0.65)

    def test_seed_reproducibility(self):
        labels = np.array([0, 1] * 100)
        run1 = partition_data(labels, num_clients=3, alpha=0.5, seed=123)
        run2 = partition_data(labels, num_clients=3, alpha=0.5, seed=123)
        
        for c1, c2 in zip(run1, run2):
            np.testing.assert_array_equal(c1, c2)


if __name__ == "__main__":
    unittest.main()
