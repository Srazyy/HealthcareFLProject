import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from src.privacy.dp_engine import validate_and_fix_model, make_private, get_epsilon_spent


class TestDPEngine(unittest.TestCase):
    def test_validate_and_fix_idempotent(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        fixed_once = validate_and_fix_model(model)
        fixed_twice = validate_and_fix_model(fixed_once)
        self.assertIsNotNone(fixed_twice)

    def test_make_private_returns_privacy_engine(self):
        model = nn.Linear(8, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        x = torch.randn(32, 8)
        y = torch.randint(0, 2, (32,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=8)
        
        priv_model, priv_opt, priv_loader, privacy_engine = make_private(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=5.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0,
        )
        
        self.assertIsInstance(privacy_engine, PrivacyEngine)
        self.assertIsNotNone(priv_model)
        self.assertIsNotNone(priv_opt)
        self.assertIsNotNone(priv_loader)

    def test_make_private_wraps_optimizer(self):
        model = nn.Linear(8, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        
        x = torch.randn(16, 8)
        y = torch.randint(0, 2, (16,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=8)
        
        _, priv_opt, _, privacy_engine = make_private(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=10.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0,
        )
        
        from opacus.optimizers import DPOptimizer
        # Optimizer should now be Opacus-wrapped
        self.assertIsInstance(priv_opt, DPOptimizer)


if __name__ == "__main__":
    unittest.main()
