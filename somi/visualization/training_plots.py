"""
SOMI 4.0 Training Plots — 6 training visualizations
======================================================

1. Loss curve (LM + JEPA)
2. Accuracy over time
3. Stress-JEPA scatter (correlation between stress and JEPA loss)
4. Accuracy vs stress (are high-stress Parts limiting performance?)
5. Learning rate schedule
6. Pathology timeline (when did issues appear and get fixed?)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_training_curves(
    history: List[Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot loss and accuracy over training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = [h.get('training_step', i) for i, h in enumerate(history)]
    losses = [h.get('total_loss', 0) for h in history]
    lm_losses = [h.get('lm_loss', 0) for h in history]
    jepa_losses = [h.get('jepa_total_loss', 0) for h in history]

    ax1.plot(steps, losses, 'b-', label='Total Loss', linewidth=2)
    ax1.plot(steps, lm_losses, 'g--', label='LM Loss', linewidth=1)
    ax1.plot(steps, jepa_losses, 'r--', label='JEPA Loss', linewidth=1)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Health score over time
    scores = [h.get('health_score', 100) for h in history]
    ax2.plot(steps, scores, 'g-', linewidth=2)
    ax2.fill_between(steps, scores, alpha=0.2, color='green')
    ax2.set_title('Brain Health Score')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Health (0-100)')
    ax2.set_ylim(0, 105)
    ax2.axhline(70, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax2.axhline(40, color='red', linestyle='--', alpha=0.5, label='Critical')
    ax2.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_pathology_timeline(
    history: List[Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Timeline showing when pathologies appeared and were fixed."""
    fig, ax = plt.subplots(figsize=(12, 5))

    steps = []
    n_pathologies = []
    n_fixes = []
    for h in history:
        steps.append(h.get('step', 0))
        n_pathologies.append(h.get('n_pathologies', 0))
        n_fixes.append(h.get('n_fixes', 0))

    ax.bar(steps, n_pathologies, color='red', alpha=0.6, label='Pathologies')
    ax.bar(steps, n_fixes, color='green', alpha=0.6, label='Fixes Applied')
    ax.set_title('Pathology Timeline')
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
