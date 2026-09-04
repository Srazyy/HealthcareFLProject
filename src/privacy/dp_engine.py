"""
Differential privacy wrapper around a PyTorch optimizer, via Opacus.

Implements the two-step DP-SGD from the project brief:
  1. Clipping:  g_i_bar = g_i * min(1, C / ||g_i||_2)
  2. Noise:     g_tilde = g_bar + N(0, sigma^2 * C^2 * I)

Provides:
  - validate_and_fix_model(): ensures Opacus compatibility for PEFT models.
  - make_private(): attaches PrivacyEngine with auto-calibrated σ from target ε.
  - get_epsilon_spent(): queries cumulative privacy budget consumed.
"""

import torch
from torch import nn
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import logging

logger = logging.getLogger(__name__)


def validate_and_fix_model(model: nn.Module) -> nn.Module:
    """
    Validates a model for Opacus compatibility and fixes any incompatible
    layers. This is critical for PEFT/LoRA models, where custom Linear
    subclasses may not be recognized by Opacus's GradSampleModule.

    The fix replaces incompatible layers (e.g., LayerNorm without
    elementwise_affine, or custom Linear variants) with Opacus-compatible
    equivalents. LoRA adapter weights are preserved through this process
    because PEFT wraps the base model's layers — ModuleValidator.fix()
    operates on the underlying modules without disturbing the PEFT wrapper.

    Args:
        model: A PyTorch model (potentially PEFT-wrapped).

    Returns:
        The validated/fixed model, ready for PrivacyEngine attachment.
    """
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(
            "Opacus found %d incompatible module(s). Applying automatic fixes...",
            len(errors),
        )
        model = ModuleValidator.fix(model)
        # Re-validate after fix
        errors_after = ModuleValidator.validate(model, strict=False)
        if errors_after:
            raise RuntimeError(
                f"Model still has {len(errors_after)} incompatible layers after "
                f"ModuleValidator.fix(). Manual intervention required: {errors_after}"
            )
        logger.info("All modules are now Opacus-compatible.")
    else:
        logger.info("Model is already Opacus-compatible. No fixes needed.")
    return model


def make_private(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    target_epsilon: float,
    target_delta: float = 1e-5,
    epochs: int = 3,
    max_grad_norm: float = 1.0,
) -> Tuple[nn.Module, Optimizer, DataLoader, PrivacyEngine]:
    """
    Wraps model/optimizer/data_loader with Opacus so training automatically
    performs per-sample gradient clipping (max_grad_norm = C) and Gaussian
    noise injection calibrated to hit `target_epsilon` after `epochs` worth
    of passes over `data_loader`.

    Returns:
        (private_model, private_optimizer, private_data_loader, privacy_engine)
        Use privacy_engine.get_epsilon(target_delta) after training to confirm
        the actual epsilon spent.
    """
    privacy_engine = PrivacyEngine()

    private_model, private_optimizer, private_data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )

    return private_model, private_optimizer, private_data_loader, privacy_engine


def get_epsilon_spent(privacy_engine: PrivacyEngine, delta: float) -> float:
    """
    Queries the Opacus privacy accountant for the cumulative epsilon
    consumed so far during training.

    This should be called after training completes (or after each round)
    to verify the actual privacy budget spent stays within the target.

    Args:
        privacy_engine: The PrivacyEngine instance attached to training.
        delta: The delta parameter of (ε, δ)-differential privacy.

    Returns:
        The cumulative epsilon spent.
    """
    epsilon = privacy_engine.get_epsilon(delta)
    logger.info("Privacy budget spent: ε = %.4f (δ = %.2e)", epsilon, delta)
    return epsilon


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("dp_engine.py: Functions available:")
    print("  - validate_and_fix_model(model) -> model")
    print("  - make_private(model, optimizer, data_loader, ...) -> (model, opt, dl, engine)")
    print("  - get_epsilon_spent(privacy_engine, delta) -> epsilon")
    print("Import these into the training loop once the model + dataloader are ready.")
