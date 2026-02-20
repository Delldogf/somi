"""
SOMI 4.0 Dual Learning — Full JEPA Pipeline + Physics-Guided Training
=========================================================================

Full pipeline:
    X-Encoder -> input h -> SOMI brain -> phi_settled
    Y-Encoder -> context  -> phi_hat (target prediction)
    JEPA loss = ||sg(phi_settled) - phi_hat||^2
    LM loss   = cross_entropy(Y_decoder(phi_settled), target_ids)
    total     = LM_loss + jepa_weight * JEPA_loss

Physics-guided features:
    - Stress-based selective updates: only update where stress > threshold
    - Mass-guided fine-tuning: scale LR by 1/mass per feature
    - Stress = JEPA verification: log correlation between the two

MACRO parameters (backprop): X-Encoder, Y-Encoder, Y-Decoder, White Matter
MICRO parameters (local physics): W_local, mass in each Part (no backprop)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple

from ..brain.circuit_brain import SOMICircuitBrain
from ..config import SOMIBrainConfig
from .jepa import JEPALoss, YEncoder


class DualLearningTrainer:
    """
    Manages the full dual learning process.

    Phase 1 (pure_local): Only micro learning (SOMI physics), no backprop.
    Phase 2 (hybrid): Both macro (backprop) and micro (SOMI physics).
    Phase 3 (consolidation): Freeze Y-Encoder (EMA), selective Y-Decoder.

    Args:
        brain: The SOMICircuitBrain model
        config: SOMIBrainConfig
        lr: Learning rate for macro parameters
        use_jepa: Enable JEPA loss
        jepa_weight: Weight of JEPA loss relative to LM loss
        selective_threshold: Stress threshold for selective updates (0=always update)
        mass_guided: Scale macro LR by 1/mass for mass-guided fine-tuning
    """

    def __init__(
        self,
        brain: SOMICircuitBrain,
        config: SOMIBrainConfig,
        lr: float = 1e-4,
        use_jepa: bool = True,
        jepa_weight: float = 0.1,
        selective_threshold: float = 0.0,
        mass_guided: bool = False,
    ):
        self.brain = brain
        self.config = config
        self.use_jepa = use_jepa
        self.jepa_weight = jepa_weight
        self.selective_threshold = selective_threshold
        self.mass_guided = mass_guided

        # JEPA components
        if use_jepa:
            self.y_encoder = YEncoder(
                input_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
            )
            self.jepa_loss_fn = JEPALoss()

        # Macro optimizer
        macro_params = list(brain.parameters())
        if use_jepa:
            macro_params += list(self.y_encoder.parameters())
        self.optimizer = optim.AdamW(macro_params, lr=lr, weight_decay=0.01)

        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.step_count = 0
        self._stress_jepa_corr_buffer = []

    def step(
        self,
        input_ids: torch.Tensor,
        input_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        training: bool = True,
        phase: int = 2,
    ) -> Tuple[float, Dict]:
        """
        One training step with full JEPA pipeline.

        Args:
            input_ids: [batch, seq] input token IDs
            input_embeddings: [batch, seq, input_dim] input embeddings
            target_ids: [batch, seq] target token IDs
            training: Whether to update parameters
            phase: 1 = pure local, 2 = hybrid, 3 = consolidation

        Returns:
            loss_value: Total loss (float)
            diagnostics: Dict with all metrics
        """
        all_diag = {}

        # === Forward through brain ===
        logits, brain_diag = self.brain(
            input_embeddings, training=training
        )
        all_diag.update(brain_diag)

        # === LM Loss ===
        per_token_loss = self.lm_loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
        )

        # Stress-based selective loss: weight per-token loss by stress
        if self.selective_threshold > 0 and training:
            stress_weight = self._compute_stress_weight(brain_diag)
            per_token_loss = per_token_loss * stress_weight
            all_diag['selective_coverage'] = (
                (stress_weight > 0.1).float().mean().item()
            )

        lm_loss = per_token_loss.mean()
        all_diag['lm_loss'] = lm_loss.item()
        total_loss = lm_loss

        # === JEPA Loss (full pipeline: stress = JEPA) ===
        if self.use_jepa:
            h = self.brain.x_encoder(input_embeddings)
            h = self.brain.x_norm(h)

            # phi_settled comes from the brain's forward pass
            # Y-Encoder predicts what phi SHOULD be
            if phase == 3:
                with torch.no_grad():
                    phi_hat = self.y_encoder(h.detach())
            else:
                phi_hat = self.y_encoder(h.detach())

            jepa_loss, jepa_diag = self.jepa_loss_fn(h, phi_hat)
            total_loss = total_loss + self.jepa_weight * jepa_loss
            all_diag.update(jepa_diag)

            # Stress-JEPA correlation tracking
            avg_stress = sum(
                v for k, v in brain_diag.items()
                if 'stress' in k and isinstance(v, (int, float))
            ) / max(1, sum(
                1 for k, v in brain_diag.items()
                if 'stress' in k and isinstance(v, (int, float))
            ))
            self._stress_jepa_corr_buffer.append(
                (avg_stress, jepa_diag.get('jepa_prediction_loss', 0))
            )
            if len(self._stress_jepa_corr_buffer) > 100:
                self._stress_jepa_corr_buffer = (
                    self._stress_jepa_corr_buffer[-100:]
                )
            all_diag['stress_jepa_correlation'] = (
                self._compute_stress_jepa_correlation()
            )

        all_diag['total_loss'] = total_loss.item()
        all_diag['training_phase'] = phase

        # === Backward + optimize (phases 2 and 3 only) ===
        if training and phase >= 2:
            self.optimizer.zero_grad()
            total_loss.backward()

            # Mass-guided gradient scaling
            if self.mass_guided:
                self._apply_mass_guided_scaling()

            torch.nn.utils.clip_grad_norm_(
                self.brain.parameters(), max_norm=1.0
            )
            if self.use_jepa:
                torch.nn.utils.clip_grad_norm_(
                    self.y_encoder.parameters(), max_norm=1.0
                )
            self.optimizer.step()

            total_grad = sum(
                p.grad.norm().item()
                for p in self.brain.parameters() if p.grad is not None
            )
            all_diag['macro_grad_norm'] = total_grad

        self.step_count += 1
        all_diag['training_step'] = self.step_count

        return total_loss.item(), all_diag

    def _compute_stress_weight(self, brain_diag: Dict) -> torch.Tensor:
        """Per-token stress weight for selective updates."""
        stress_vals = [
            v for k, v in brain_diag.items()
            if 'stress' in k and isinstance(v, (int, float))
        ]
        if not stress_vals:
            return torch.ones(1, device=next(self.brain.parameters()).device)

        avg_stress = sum(stress_vals) / len(stress_vals)
        weight = 1.0 if avg_stress > self.selective_threshold else 0.1
        return torch.tensor(weight, device=next(self.brain.parameters()).device)

    def _apply_mass_guided_scaling(self):
        """Scale gradients by 1/mass for mass-guided fine-tuning.

        Heavy features (important, well-learned) get smaller updates.
        Light features (new, experimental) get bigger updates.
        Only affects the X-Encoder and Y-Decoder whose dimensions match hidden_dim.
        """
        avg_mass = torch.stack([
            part.mass for part in self.brain.parts.values()
        ]).mean(dim=0)

        mass_scale = 1.0 / avg_mass.clamp(min=0.1)
        mass_scale = mass_scale / mass_scale.mean()

        H = mass_scale.shape[0]
        for name, param in self.brain.named_parameters():
            if param.grad is None:
                continue
            if 'x_encoder.weight' in name and param.grad.shape[0] == H:
                param.grad.data *= mass_scale.view(-1, 1)
            elif 'y_decoder.weight' in name and param.grad.shape[1] == H:
                param.grad.data *= mass_scale.view(1, -1)

    def _compute_stress_jepa_correlation(self) -> float:
        """Correlation between SOMI stress and JEPA loss (theory says ~1.0)."""
        if len(self._stress_jepa_corr_buffer) < 10:
            return 0.0
        stresses = torch.tensor([s for s, j in self._stress_jepa_corr_buffer])
        jepas = torch.tensor([j for s, j in self._stress_jepa_corr_buffer])
        if stresses.std() < 1e-8 or jepas.std() < 1e-8:
            return 0.0
        corr = torch.corrcoef(torch.stack([stresses, jepas]))[0, 1]
        return corr.item() if not torch.isnan(corr) else 0.0
