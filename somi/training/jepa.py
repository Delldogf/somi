"""
SOMI 4.0 JEPA Loss — Joint Embedding Predictive Architecture
===============================================================

JEPA connects SOMI to the rest of the model:
  X-Encoder encodes input -> phi_target
  Y-Encoder encodes output -> phi_hat (prediction of what phi should be)
  SOMI settles phi from X-Encoder's output
  JEPA Loss = ||phi_settled - phi_hat||^2 (how well did SOMI predict?)

The key insight: SOMI's STRESS IS the JEPA loss.
  - High stress in a Part = bad prediction = the Part is struggling
  - Low stress = good prediction = the Part is doing its job

This means we don't need a separate loss function for SOMI — the physics
(stress) IS the loss. The JEPA loss just adds a training signal for the
macro parameters (X-Encoder, Y-Encoder) via backprop.

Source: SOMI_3_0/theory/02_THE_CIRCUIT_BRAIN.md
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class JEPALoss(nn.Module):
    """
    JEPA loss for training the macro parameters.

    L_JEPA = ||sg(phi_settled) - phi_hat||^2

    Where:
    - phi_settled: The SOMI brain's output (stop gradient — don't backprop into SOMI)
    - phi_hat: Y-Encoder's prediction of what phi should be
    - sg(): stop gradient operator

    The stop gradient on phi_settled means:
    - SOMI learns via its own physics (stress, STDP) — local learning
    - Y-Encoder learns via backprop through this loss — macro learning
    - They converge toward each other without mode collapse

    Variance-Covariance regularization (VICReg-style) prevents collapse:
    - Variance: each feature should have variance > 0 (don't collapse to a constant)
    - Covariance: features should be decorrelated (don't all do the same thing)
    """

    def __init__(
        self,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.04,
    ):
        super().__init__()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight

    def forward(
        self,
        phi_settled: torch.Tensor,
        phi_hat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute JEPA loss with VICReg regularization.

        Args:
            phi_settled: [batch, seq, hidden] SOMI's settled output (stop-graded)
            phi_hat: [batch, seq, hidden] Y-Encoder's prediction

        Returns:
            loss: scalar total loss
            diagnostics: Dict with loss components
        """
        # Main JEPA loss: MSE between settled and predicted
        # Stop gradient on phi_settled — SOMI doesn't need backprop
        prediction_loss = (phi_settled.detach() - phi_hat).pow(2).mean()

        # Variance regularization: keep features alive
        # For each feature, compute variance across batch*seq
        flat = phi_hat.reshape(-1, phi_hat.shape[-1])  # [N, hidden]
        var = flat.var(dim=0)  # [hidden]
        variance_loss = torch.relu(1.0 - var.sqrt()).mean()

        # Covariance regularization: keep features decorrelated
        flat_centered = flat - flat.mean(dim=0)
        N = flat_centered.shape[0]
        cov = (flat_centered.T @ flat_centered) / max(N - 1, 1)
        # Only penalize off-diagonal (correlations, not variance)
        cov.fill_diagonal_(0)
        covariance_loss = (cov ** 2).sum() / flat.shape[-1]

        # Total loss
        total = (
            prediction_loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
        )

        diagnostics = {
            'jepa_prediction_loss': prediction_loss.item(),
            'jepa_variance_loss': variance_loss.item(),
            'jepa_covariance_loss': covariance_loss.item(),
            'jepa_total_loss': total.item(),
            'jepa_phi_hat_mean': phi_hat.detach().abs().mean().item(),
            'jepa_feature_variance_mean': var.mean().item(),
        }

        return total, diagnostics


class YEncoder(nn.Module):
    """
    Y-Encoder: Predicts what phi SHOULD be from the output context.

    In the full pipeline:
    - X-Encoder: input -> h (what the brain receives)
    - SOMI brain: h -> phi_settled (what the brain computes)
    - Y-Encoder: output_context -> phi_hat (what the brain should compute)
    - JEPA Loss: ||phi_settled - phi_hat||^2

    The Y-Encoder learns to provide good targets for SOMI.
    It's trained by backprop through the JEPA loss.

    Args:
        input_dim: Dimension of output context
        hidden_dim: SOMI's hidden dimension
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Encode output context to prediction target.

        Args:
            context: [batch, seq, input_dim]

        Returns:
            phi_hat: [batch, seq, hidden_dim]
        """
        return self.encoder(context)
