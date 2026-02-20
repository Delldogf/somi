"""
SOMI 4.0 Physics Plots — 9 physics visualizations
=====================================================

1. Stress profile (per-connection stress magnitude)
2. Settling curve (velocity norm over settling steps)
3. Domain bars (per-domain stress breakdown)
4. Mass distribution (histogram + Herfindahl)
5. Eigenspectrum (eigenvalues of L_rw)
6. Hamiltonian trajectory (H over time)
7. W heatmap (full connectivity matrix)
8. Norm flow (W norm over training steps)
9. Force balance (relative magnitude of each force)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional


def plot_hamiltonian_trajectory(
    H_history: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Hamiltonian over settling steps."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(H_history, 'b-', linewidth=2)
    ax.set_title('Hamiltonian Trajectory (Should Decrease)')
    ax.set_xlabel('Settling Step')
    ax.set_ylabel('H = T + V')
    # Highlight violations
    for i in range(1, len(H_history)):
        if H_history[i] > H_history[i-1] + 1e-6:
            ax.axvline(i, color='red', alpha=0.3, linewidth=1)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_force_balance(
    force_diagnostics: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of relative force magnitudes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    force_names = []
    magnitudes = []
    for k, v in sorted(force_diagnostics.items()):
        if k.startswith('force_') and k.endswith('_magnitude') and 'total' not in k:
            name = k.replace('force_', '').replace('_magnitude', '')
            force_names.append(name)
            magnitudes.append(v)
    colors = plt.cm.Set2(np.linspace(0, 1, len(force_names)))
    ax.barh(force_names, magnitudes, color=colors)
    ax.set_title('Force Balance')
    ax.set_xlabel('Mean Magnitude')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_eigenspectrum(
    eigenvalues: torch.Tensor,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot eigenvalue distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    eigs = eigenvalues.detach().cpu().numpy()
    pos = eigs[eigs > 1e-6]

    ax1.plot(range(len(pos)), pos, 'bo-', markersize=3)
    ax1.set_title('Eigenvalues (Linear)')
    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Eigenvalue')

    if len(pos) > 0:
        ax2.semilogy(range(len(pos)), pos, 'ro-', markersize=3)
        ax2.set_title('Eigenvalues (Log)')
        ax2.set_xlabel('Mode Index')
        ax2.set_ylabel('Eigenvalue (log)')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_w_heatmap(
    W: torch.Tensor,
    title: str = 'W Connectivity',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot W as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(W.detach().cpu().numpy(), cmap='viridis', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Target Feature')
    ax.set_ylabel('Source Feature')
    fig.colorbar(im, ax=ax)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_stress_heatmap(
    S: torch.Tensor,
    title: str = 'Stress Tensor',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot stress tensor as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S.detach().cpu().numpy(), cmap='RdBu_r', aspect='auto')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Stress')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
