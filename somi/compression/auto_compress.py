"""
SOMI Auto-Compress: Physics-Triggered Model Compression
==========================================================

Monitors stress and spectral redundancy during training. When the
model is over-provisioned (low stress for sustained periods, many
redundant eigenmodes), it automatically compresses — the inverse
of AutoGrowth.

Together with AutoGrowthFull, this completes the grow/compress loop:
  - High stress for long time → grow (add neurons or Parts)
  - Low stress for long time  → compress (prune and quantize)

The trigger is physics-driven:
  1. Stress EMA stays BELOW threshold for `patience` steps
  2. Spectral redundancy: spectral_rank / total_modes < ratio_threshold
  3. After triggering, adaptive_compress runs on each Part
  4. Topological quality is checked; if quality drops, roll back

After compression, recalibrate_config() updates physics for new size.

Usage:
    compressor = AutoCompress(brain)
    for step, batch in enumerate(dataloader):
        logits, loss, diagnostics = brain(batch, training=True)
        compressor.step(diagnostics)  # may trigger compression
"""

import torch
from typing import Optional

from .adaptive import adaptive_compress
from .spectral_rank import spectral_rank_selection
from .topological_quality import check_topological_quality


class AutoCompress:
    """
    Monitors model stress and spectral redundancy; compresses when safe.

    Args:
        brain: SOMICircuitBrain instance
        low_stress_threshold: stress below which compression is considered
        spectral_ratio_threshold: if spectral_rank/total_modes < this, redundant
        patience: consecutive low-stress steps before compressing
        compression_ratio: target compression per event (1.5 = 33% smaller)
        quality_threshold: minimum topological quality to accept compression
        cooldown: steps to wait after compress before considering again
        min_hidden: never compress below this hidden_dim
    """

    def __init__(
        self,
        brain: 'SOMICircuitBrain',
        low_stress_threshold: float = 0.1,
        spectral_ratio_threshold: float = 0.5,
        patience: int = 200,
        compression_ratio: float = 1.5,
        quality_threshold: float = 0.85,
        cooldown: int = 300,
        min_hidden: int = 16,
    ):
        self.brain = brain
        self.low_stress_threshold = low_stress_threshold
        self.spectral_ratio_threshold = spectral_ratio_threshold
        self.patience = patience
        self.compression_ratio = compression_ratio
        self.quality_threshold = quality_threshold
        self.cooldown = cooldown
        self.min_hidden = min_hidden

        self.low_stress_count = 0
        self.steps_since_compress = 0
        self.compress_events = []
        self.stress_ema = 0.5
        self.spectral_ratio_ema = 1.0

    def step(self, diagnostics: dict) -> bool:
        """
        Check stress/spectrum and potentially trigger compression.

        Args:
            diagnostics: the diagnostics dict from brain.forward()

        Returns:
            True if compression was triggered this step
        """
        self.steps_since_compress += 1

        stress_values = [
            v for k, v in diagnostics.items()
            if 'stress' in k and isinstance(v, (int, float))
        ]
        if not stress_values:
            return False

        avg_stress = sum(stress_values) / len(stress_values)
        alpha = 0.95
        self.stress_ema = alpha * self.stress_ema + (1 - alpha) * avg_stress

        spectral_modes = [
            v for k, v in diagnostics.items()
            if 'eigen_used_modes' in k and isinstance(v, (int, float))
        ]
        spectral_total = [
            v for k, v in diagnostics.items()
            if 'eigen_total_modes' in k and isinstance(v, (int, float))
        ]
        if spectral_modes and spectral_total:
            ratio = sum(spectral_modes) / max(sum(spectral_total), 1)
            self.spectral_ratio_ema = alpha * self.spectral_ratio_ema + (1 - alpha) * ratio

        stress_low = self.stress_ema < self.low_stress_threshold
        spectral_redundant = self.spectral_ratio_ema < self.spectral_ratio_threshold

        if stress_low or spectral_redundant:
            self.low_stress_count += 1
        else:
            self.low_stress_count = max(0, self.low_stress_count - 1)

        if (
            self.low_stress_count >= self.patience
            and self.steps_since_compress >= self.cooldown
            and self.brain.config.hidden_dim > self.min_hidden
        ):
            return self._trigger_compress()

        return False

    def _trigger_compress(self) -> bool:
        """Run adaptive compression on each Part, roll back if quality drops."""
        print(f"\n[AutoCompress] Low stress ({self.stress_ema:.3f}) for "
              f"{self.low_stress_count} steps. Spectral ratio: {self.spectral_ratio_ema:.3f}")
        print(f"[AutoCompress] Compressing Parts (ratio={self.compression_ratio:.1f})")

        all_results = {}
        any_applied = False

        for pid, part in self.brain.parts.items():
            result = adaptive_compress(
                part,
                target_compression=self.compression_ratio,
                quality_threshold=self.quality_threshold,
            )
            all_results[f'part_{pid}'] = result

            if result.get('compression_applied', False):
                any_applied = True
                print(f"  Part {pid}: compressed "
                      f"(quality={result.get('weight_correlation', 0):.3f}, "
                      f"pruned={result.get('pruned_fraction', 0):.1%})")
            else:
                quality = result.get('compression_rejected_quality', 'N/A')
                print(f"  Part {pid}: rejected (quality={quality})")

        if any_applied:
            self.brain.recalibrate_config()
            print(f"[AutoCompress] Recalibrated physics for current state.")

        self.compress_events.append({
            'step': self.steps_since_compress,
            'stress_at_trigger': self.stress_ema,
            'spectral_ratio': self.spectral_ratio_ema,
            'any_applied': any_applied,
            'results': all_results,
        })

        self.low_stress_count = 0
        self.steps_since_compress = 0

        return any_applied

    def get_status(self) -> dict:
        return {
            'hidden_dim': self.brain.config.hidden_dim,
            'stress_ema': self.stress_ema,
            'spectral_ratio_ema': self.spectral_ratio_ema,
            'low_stress_count': self.low_stress_count,
            'compress_events': len(self.compress_events),
            'steps_since_compress': self.steps_since_compress,
        }
