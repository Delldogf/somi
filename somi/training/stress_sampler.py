"""
SOMI Stress-Based Data Sampler
=================================

Samples batches that produce HIGH stress more frequently, so the model
focuses on what it doesn't know (hard examples) rather than rehearsing
what it already knows (easy examples).

Like a student who focuses on the topics they find hardest.

Usage:
    sampler = StressDataSampler(dataset)
    for batch_indices in sampler:
        batch = dataset[batch_indices]
        logits, diag = brain(batch, training=True)
        sampler.update_stress(batch_indices, diag)
"""

import torch
from typing import List, Optional


class StressDataSampler:
    """
    Samples data indices weighted by their last-observed stress.

    High-stress examples (the model struggles with them) are sampled
    more often. Low-stress examples (the model knows them) are sampled
    less. Stress estimates are updated after each forward pass.

    Args:
        dataset_size: Number of examples in the dataset
        batch_size: Number of examples per batch
        temperature: Stress sampling temperature. Higher = more uniform.
                     Lower = more aggressive focus on hard examples.
        min_weight: Minimum sampling weight (ensures all data seen sometimes)
        initial_stress: Starting stress for unseen examples
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int = 32,
        temperature: float = 1.0,
        min_weight: float = 0.01,
        initial_stress: float = 1.0,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.min_weight = min_weight

        self.stress_scores = torch.full((dataset_size,), initial_stress)
        self.visit_counts = torch.zeros(dataset_size)
        self.total_samples = 0

    def sample(self) -> List[int]:
        """
        Sample a batch of indices, weighted by stress.

        Returns:
            indices: List of dataset indices
        """
        weights = (self.stress_scores / self.temperature).clamp(min=self.min_weight)
        weights = weights / weights.sum()

        indices = torch.multinomial(
            weights, self.batch_size, replacement=False
        ).tolist()

        self.visit_counts[indices] += 1
        self.total_samples += self.batch_size

        return indices

    def update_stress(self, indices: List[int], diagnostics: dict):
        """
        Update stress scores for the examples just processed.

        Args:
            indices: Which examples were in this batch
            diagnostics: Brain diagnostics from forward pass
        """
        stress_vals = [
            v for k, v in diagnostics.items()
            if 'stress' in k and isinstance(v, (int, float))
        ]
        if not stress_vals:
            return

        avg_stress = sum(stress_vals) / len(stress_vals)
        for idx in indices:
            old = self.stress_scores[idx].item()
            self.stress_scores[idx] = 0.9 * old + 0.1 * avg_stress

    def get_stats(self) -> dict:
        return {
            'mean_stress': self.stress_scores.mean().item(),
            'max_stress': self.stress_scores.max().item(),
            'min_stress': self.stress_scores.min().item(),
            'coverage': (self.visit_counts > 0).float().mean().item(),
            'total_samples': self.total_samples,
        }

    def __iter__(self):
        """Iterate: yields batches of indices."""
        while True:
            yield self.sample()
