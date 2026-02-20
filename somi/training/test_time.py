"""
SOMI 4.0 Test-Time Learning
==============================

SOMI can learn AT INFERENCE TIME — no backprop needed.

When the model encounters surprising input (high stress), the Parts
automatically update their W_local to adapt. This is like how your brain
adjusts when you encounter something unexpected — you don't need to
"retrain" yourself, you just adapt in real-time.

How it works:
1. Forward pass through the brain (normal inference)
2. Measure surprise: how much stress did each Part experience?
3. If surprise > threshold: W_local already updated during forward()
   (because SOMIPart.forward() always runs local physics)
4. No extra work needed — it's built into the architecture!

This is a FUNDAMENTAL difference from transformers:
- Transformer: frozen at inference. Can't adapt to new patterns.
- SOMI: continuously adapts via physics-based local learning.

From Titans paper (Behrouz 2024): "surprise-based memory updates"
enable real-time adaptation to distribution shifts.

Data-dependent forgetting:
When context changes dramatically (high stress change), the model
forgets old patterns and learns fresh. When context is stable,
it retains and reinforces.

Source: SOMI_3_0/theory/02_THE_CIRCUIT_BRAIN.md
Paper: Titans (Behrouz 2024)
"""

import torch
from typing import Dict, Optional, Tuple

from ..brain.circuit_brain import SOMICircuitBrain
from ..config import SOMIBrainConfig


class TestTimeLearner:
    """
    Manages test-time learning for the Circuit Brain.

    During inference, this module:
    1. Monitors surprise (stress) across all Parts
    2. Allows Parts to update W_local when surprised
    3. Tracks adaptation metrics
    4. Optionally limits the rate of test-time updates (to prevent drift)

    Usage:
        learner = TestTimeLearner(brain, config)
        logits = learner.infer(input_embeddings)
        # W_local may have been updated if the input was surprising!

    Args:
        brain: The SOMICircuitBrain
        config: SOMIBrainConfig
    """

    def __init__(
        self,
        brain: SOMICircuitBrain,
        config: SOMIBrainConfig,
    ):
        self.brain = brain
        self.config = config
        self.adaptation_count = 0
        self.total_inferences = 0

    def infer(
        self,
        input_embeddings: torch.Tensor,
        allow_adaptation: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run inference with optional test-time learning.

        The key: we pass training=True to the brain if adaptation is allowed.
        This causes SOMIPart.forward() to run its local physics update
        (stress computation, W_local update, etc.).

        But we DON'T run backprop — there's no loss.backward() call.
        The learning is entirely LOCAL to each Part.

        Args:
            input_embeddings: [batch, seq, input_dim]
            allow_adaptation: Whether to allow W_local updates

        Returns:
            logits: [batch, seq, output_dim]
            diagnostics: Dict with inference and adaptation metrics
        """
        self.total_inferences += 1

        # Run forward pass
        # training=True enables local physics updates in Parts
        # training=False freezes everything (standard inference)
        should_adapt = (
            allow_adaptation
            and self.config.test_time_learning
        )

        with torch.no_grad():
            logits, diagnostics = self.brain(
                input_embeddings,
                training=should_adapt,
            )

        diagnostics['test_time_adaptation_enabled'] = should_adapt
        diagnostics['test_time_total_inferences'] = self.total_inferences

        if should_adapt:
            self.adaptation_count += 1
            diagnostics['test_time_adaptation_count'] = self.adaptation_count
            diagnostics['test_time_adaptation_rate'] = (
                self.adaptation_count / self.total_inferences
            )

        return logits, diagnostics

    def reset_adaptation_stats(self):
        """Reset adaptation counters (e.g., between evaluation runs)."""
        self.adaptation_count = 0
        self.total_inferences = 0
