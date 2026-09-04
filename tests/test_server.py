import unittest
from flwr.common import Metrics
from src.federated.server import weighted_average


class TestServerWeightedAverage(unittest.TestCase):
    def test_standard_weighted_average(self):
        # Client 0 has 100 samples, Client 1 has 300 samples (total 400)
        # Expected accuracy = (100 * 0.8 + 300 * 0.6) / 400 = (80 + 180) / 400 = 0.65
        metrics: list[tuple[int, Metrics]] = [
            (
                100,
                {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.9,
                    "f1": 0.8,
                    "mcc": 0.5,
                    "auc_roc": 0.85,
                },
            ),
            (
                300,
                {
                    "accuracy": 0.6,
                    "precision": 0.5,
                    "recall": 0.7,
                    "f1": 0.6,
                    "mcc": 0.3,
                    "auc_roc": 0.65,
                },
            ),
        ]
        aggregated = weighted_average(metrics)
        self.assertAlmostEqual(float(aggregated["accuracy"]), 0.65)
        self.assertAlmostEqual(float(aggregated["precision"]), 0.55)
        self.assertAlmostEqual(float(aggregated["recall"]), 0.75)
        self.assertAlmostEqual(float(aggregated["f1"]), 0.65)
        self.assertAlmostEqual(float(aggregated["mcc"]), 0.35)
        self.assertAlmostEqual(float(aggregated["auc_roc"]), 0.70)

    def test_empty_metrics_list(self):
        self.assertEqual(weighted_average([]), {})

    def test_zero_total_examples(self):
        metrics: list[tuple[int, Metrics]] = [
            (0, {"accuracy": 0.9, "f1": 0.8}),
            (0, {"accuracy": 0.8, "f1": 0.7}),
        ]
        self.assertEqual(weighted_average(metrics), {})

    def test_missing_and_none_metric_keys(self):
        # Client 0 is missing 'auc_roc', Client 1 has 'auc_roc': 0.0
        metrics: list[tuple[int, Metrics]] = [
            (100, {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "mcc": 1.0}),
            (100, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0}),
        ]
        aggregated = weighted_average(metrics)
        self.assertAlmostEqual(float(aggregated["accuracy"]), 0.5)
        self.assertAlmostEqual(float(aggregated["auc_roc"]), 0.0)

    def test_nan_metrics_filtered(self):
        # Client 0 has valid auc_roc (0.8), Client 1 has float('nan') due to single-class validation skew
        metrics: list[tuple[int, Metrics]] = [
            (100, {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8, "mcc": 0.5, "auc_roc": 0.8}),
            (100, {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6, "mcc": 0.3, "auc_roc": float("nan")}),
        ]
        aggregated = weighted_average(metrics)
        self.assertAlmostEqual(float(aggregated["accuracy"]), 0.7)
        # Should average over the valid client only, not resulting in NaN
        self.assertAlmostEqual(float(aggregated["auc_roc"]), 0.8)


if __name__ == "__main__":
    unittest.main()
