"""
Differential privacy wrapper around a PyTorch optimizer, via Opacus.

Owner: Track A

Implements the two-step DP-SGD from the project brief:
  1. Clipping:  g_i_bar = g_i * min(1, C / ||g_i||_2)
  2. Noise:     g_tilde = g_bar + N(0, sigma^2 * C^2 * I)

TODO:
- Wire this into a real training loop (with a real model/dataloader).
- Sweep epsilon and log the resulting accuracy/F1 to build the
  privacy-budget vs accuracy curve referenced in the slides.
"""

from opacus import PrivacyEngine


def make_private(model, optimizer, data_loader, target_epsilon: float,
                  target_delta: float = 1e-5, epochs: int = 3,
                  max_grad_norm: float = 1.0):
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


if __name__ == "__main__":
    print("dp_engine.py: import make_private() into the training loop once "
          "the model + dataloader are ready.")
