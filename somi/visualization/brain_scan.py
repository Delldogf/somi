"""
SOMI 4.0 Brain Scan Dashboard — 9-Panel Overview
===================================================

A single figure showing the brain's complete state at a glance:
1. Stress heatmap (W-weighted stress per connection)
2. Kinetic stress heatmap
3. Arousal gauge (per Part)
4. Mass distribution (histogram)
5. Entropy timeline
6. Settling convergence (velocity over steps)
7. Phi distribution (histogram of activations)
8. W heatmap (connectivity matrix)
9. Frequency spectrum (eigenvalue distribution)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from typing import Dict, Optional


def plot_brain_scan(
    brain: 'SOMICircuitBrain',
    diagnostics: Optional[Dict] = None,
    title: str = 'SOMI 4.0 Brain Scan',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate the 9-panel brain scan dashboard.

    Args:
        brain: SOMICircuitBrain instance
        diagnostics: Optional diagnostics dict (from dashboard.report)
        title: Figure title
        save_path: If provided, save to this path

    Returns:
        fig: matplotlib Figure
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35)

    # Collect data from first Part for visualization
    part = list(brain.parts.values())[0]
    W = part.W_local.detach().cpu().numpy()
    mass = part.mass.detach().cpu().numpy()

    # 1. W Heatmap (connectivity)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(W, cmap='viridis', aspect='auto')
    ax1.set_title('W Connectivity')
    ax1.set_xlabel('Target')
    ax1.set_ylabel('Source')
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    # 2. Stress Heatmap (if available — use W * (1 - correlation) as proxy)
    ax2 = fig.add_subplot(gs[0, 1])
    stress_proxy = np.abs(W - W.T)  # Asymmetry as stress proxy
    im2 = ax2.imshow(stress_proxy, cmap='hot', aspect='auto')
    ax2.set_title('Stress (Asymmetry)')
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    # 3. Mass Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(mass, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax3.set_title(f'Mass Distribution (mean={mass.mean():.3f})')
    ax3.set_xlabel('Mass')
    ax3.set_ylabel('Count')
    ax3.axvline(mass.mean(), color='red', linestyle='--', label='Mean')
    ax3.legend()

    # 4. Arousal per Part
    ax4 = fig.add_subplot(gs[1, 0])
    part_ids = []
    arousals = []
    for pid, p in brain.parts.items():
        part_ids.append(f'Part {pid}')
        arousals.append(p.arousal.item())
    colors = ['green' if 0.1 < a < 0.9 else 'red' for a in arousals]
    ax4.barh(part_ids, arousals, color=colors, edgecolor='white')
    ax4.set_title('Arousal per Part')
    ax4.set_xlim(0, 1)
    ax4.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    # 5. Eigenspectrum
    ax5 = fig.add_subplot(gs[1, 1])
    eigs = part.eigenvalues.detach().cpu().numpy()
    pos_eigs = eigs[eigs > 1e-6]
    if len(pos_eigs) > 0:
        ax5.plot(range(len(pos_eigs)), pos_eigs, 'bo-', markersize=3)
        ax5.set_title(f'Eigenspectrum (gap={pos_eigs[1]-pos_eigs[0]:.4f}' if len(pos_eigs) > 1 else 'Eigenspectrum')
    ax5.set_xlabel('Mode Index')
    ax5.set_ylabel('Eigenvalue')

    # 6. Mass vs Connectivity
    ax6 = fig.add_subplot(gs[1, 2])
    kappa = np.abs(W).sum(axis=1)
    ax6.scatter(kappa, mass, alpha=0.5, s=10, c='purple')
    ax6.set_title('Mass-Conductivity')
    ax6.set_xlabel('Conductivity (row sum)')
    ax6.set_ylabel('Mass')

    # 7. W Row Sum Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    row_sums = W.sum(axis=1)
    ax7.hist(row_sums, bins=30, color='orange', edgecolor='white', alpha=0.8)
    ax7.set_title('W Row Sums')
    ax7.set_xlabel('Row Sum')

    # 8. System Weights
    ax8 = fig.add_subplot(gs[2, 1])
    weights = torch.softmax(brain.system_weights, dim=0).detach().cpu().numpy()
    sys_names = [f'Sys {i}' for i in range(len(weights))]
    ax8.bar(sys_names, weights, color='teal', edgecolor='white')
    ax8.set_title('System Weights')
    ax8.set_ylim(0, 1)

    # 9. Health Score (if diagnostics provided)
    ax9 = fig.add_subplot(gs[2, 2])
    if diagnostics and 'health_score' in diagnostics:
        score = diagnostics['health_score']
        color = 'green' if score > 70 else 'orange' if score > 40 else 'red'
        ax9.text(0.5, 0.5, f'{score:.0f}', fontsize=48, fontweight='bold',
                 ha='center', va='center', color=color,
                 transform=ax9.transAxes)
        ax9.text(0.5, 0.2, 'Health Score', fontsize=14,
                 ha='center', va='center', transform=ax9.transAxes)
        n_path = diagnostics.get('n_pathologies', 0)
        ax9.text(0.5, 0.1, f'({n_path} pathologies)', fontsize=10,
                 ha='center', va='center', transform=ax9.transAxes, alpha=0.7)
    else:
        ax9.text(0.5, 0.5, 'No diagnostics', ha='center', va='center',
                 transform=ax9.transAxes)
    ax9.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
