"""
SOMI 4.0 Neuroscience Plots — 6 brain-inspired visualizations
================================================================

1. E/I Balance (excitation vs inhibition)
2. Criticality (spectral radius near 1.0)
3. Modularity (community structure)
4. Neuromodulators (NE, DA, ACh, 5-HT levels)
5. Structure-Function coupling (W vs activity)
6. Eigenmode-Task alignment
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional


def plot_ei_balance(
    neuro_diagnostics: Dict[str, float],
    config: 'SOMIBrainConfig',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot E/I balance across all Parts."""
    fig, ax = plt.subplots(figsize=(8, 5))

    parts = []
    balances = []
    for k, v in sorted(neuro_diagnostics.items()):
        if 'ei_balance' in k:
            pid = k.split('_')[1].replace('part', '')
            parts.append(f'Part {pid}')
            balances.append(v)

    if not parts:
        ax.text(0.5, 0.5, 'No E/I data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    colors = ['green' if 0.6 < b < 0.9 else 'red' for b in balances]
    ax.bar(parts, balances, color=colors, edgecolor='white')
    ax.axhline(0.8, color='gray', linestyle='--', label='Target (80% E)')
    ax.axhspan(0.6, 0.9, alpha=0.1, color='green', label='Healthy range')
    ax.set_title('E/I Balance (Target: 80% Excitatory)')
    ax.set_ylabel('E / (E + I)')
    ax.set_ylim(0, 1)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_criticality(
    neuro_diagnostics: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot criticality index (distance from edge of chaos)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    parts = []
    criticality = []
    spectral_radii = []
    for k, v in sorted(neuro_diagnostics.items()):
        if 'criticality_index' in k and v >= 0:
            pid = k.split('_')[1].replace('part', '')
            parts.append(f'Part {pid}')
            criticality.append(v)
        if 'spectral_radius' in k and v >= 0:
            spectral_radii.append(v)

    if not parts:
        ax.text(0.5, 0.5, 'No criticality data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    colors = ['green' if c < 0.1 else 'orange' if c < 0.3 else 'red' for c in criticality]
    ax.bar(parts, criticality, color=colors, edgecolor='white')
    ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='Near critical')
    ax.set_title('Criticality (Lower = Closer to Edge of Chaos)')
    ax.set_ylabel('|spectral_radius - 1.0|')
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_neuromodulators(
    brain: 'SOMICircuitBrain',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Radar chart of neuromodulator levels per Part."""
    fig, axes = plt.subplots(1, min(4, len(brain.parts)), figsize=(16, 4))
    if not hasattr(axes, '__len__'):
        axes = [axes]

    categories = ['NE\n(Arousal)', 'DA\n(Reward)', 'ACh\n(Attention)', '5-HT\n(Mood)']

    for idx, (pid, part) in enumerate(brain.parts.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        if brain.config.neuromodulators_enabled:
            values = [
                part.ne_level.item(),
                part.da_level.item(),
                part.ach_level.item(),
                part.serotonin_level.item(),
            ]
        else:
            values = [0.5, 0.5, 0.5, 0.5]

        x = np.arange(len(categories))
        ax.bar(x, values, color=['red', 'blue', 'green', 'purple'], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f'Part {pid}')

    fig.suptitle('Neuromodulator Levels', fontsize=14, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
