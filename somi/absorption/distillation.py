"""
SOMI 4.0 Output-Only Distillation
====================================

When weight-level transplant isn't possible (different architecture),
use output-level distillation: match SOMI's output to the teacher's output.

This is the most general absorption method — works with ANY teacher model
regardless of architecture. But it's slower (requires running the teacher
on data).

KL divergence: D_KL(teacher || student) = sum(p_teacher * log(p_teacher / p_student))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class OutputDistiller:
    """
    Distill knowledge from a teacher model's outputs.

    The teacher provides soft labels (probability distributions over outputs).
    The student (SOMI) learns to match these distributions.

    Temperature scaling: Higher temperature makes the distributions softer,
    revealing more of the teacher's knowledge about non-obvious choices.

    Args:
        temperature: Softmax temperature (higher = softer distributions)
        alpha: Weight of distillation loss vs task loss
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss (KL divergence on soft labels).

        Args:
            student_logits: [batch, seq, vocab] student's raw outputs
            teacher_logits: [batch, seq, vocab] teacher's raw outputs

        Returns:
            loss: Scalar distillation loss
            diagnostics: Loss components
        """
        T = self.temperature

        # Soft probabilities (high temperature = softer)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        # KL divergence
        kl = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        loss = kl * (T * T)  # Scale by T^2 to match gradient magnitudes

        diagnostics = {
            'distillation_kl': kl.item(),
            'distillation_loss': loss.item(),
            'distillation_temperature': T,
        }

        return loss, diagnostics
