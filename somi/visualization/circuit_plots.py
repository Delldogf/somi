"""
SOMI 4.0 Circuit Plots — 7 circuit flow visualizations
=========================================================

1. Circuit flow diagram (Parts and connections)
2. Bottleneck detection (Part arousal comparison)
3. Throughput per system
4. Coherence matrix (system-system correlation)
5. Tract heatmap (white matter utilization)
6. Shared Part map (which Parts are shared)
7. Generalization pressure heatmap
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional


def plot_circuit_flow(
    brain: 'SOMICircuitBrain',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize the circuit architecture with shared Parts highlighted."""
    fig, ax = plt.subplots(figsize=(12, 8))

    config = brain.config
    n_parts = config.n_parts
    shared = set(config.shared_part_ids or [])

    # Position Parts in a circle
    angles = np.linspace(0, 2 * np.pi, n_parts, endpoint=False)
    positions = [(2 * np.cos(a), 2 * np.sin(a)) for a in angles]

    # Draw Parts
    for i, (x, y) in enumerate(positions):
        color = 'gold' if i in shared else 'lightblue'
        size = 800 if i in shared else 500
        ax.scatter(x, y, s=size, c=color, edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(f'Part {i}', (x, y), ha='center', va='center',
                    fontweight='bold', fontsize=10, zorder=6)

    # Draw System routes
    colors = plt.cm.Set1(np.linspace(0, 1, len(config.system_routes)))
    for sys_id, route in enumerate(config.system_routes):
        color = colors[sys_id]
        for j in range(len(route) - 1):
            src = route[j]
            tgt = route[j + 1]
            sx, sy = positions[src]
            tx, ty = positions[tgt]
            offset = 0.05 * sys_id  # Offset for visual clarity
            ax.annotate('', xy=(tx + offset, ty + offset),
                        xytext=(sx + offset, sy + offset),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.plot([], [], color=color, linewidth=2, label=f'System {sys_id}: {route}')

    ax.set_title('SOMI Circuit Architecture')
    ax.legend(loc='upper right')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_shared_part_pressure(
    circuit_diagnostics: Dict[str, float],
    config: 'SOMIBrainConfig',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap showing generalization pressure on shared Parts."""
    fig, ax = plt.subplots(figsize=(8, 5))

    shared = config.shared_part_ids or []
    if not shared:
        ax.text(0.5, 0.5, 'No shared Parts', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    pressures = []
    labels = []
    for sid in shared:
        p = circuit_diagnostics.get(f'circuit_shared{sid}_pressure', 0)
        n = circuit_diagnostics.get(f'circuit_shared{sid}_n_systems', 0)
        pressures.append(p)
        labels.append(f'Part {sid}\n({n} systems)')

    colors = ['green' if p < 0.5 else 'orange' if p < 1.0 else 'red' for p in pressures]
    ax.bar(labels, pressures, color=colors, edgecolor='white')
    ax.set_title('Generalization Pressure on Shared Parts')
    ax.set_ylabel('Pressure (n_systems * arousal)')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_bottleneck(
    brain: 'SOMICircuitBrain',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Show which Part is the bottleneck (highest arousal)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    parts = []
    arousals = []
    for pid, part in brain.parts.items():
        parts.append(f'Part {pid}')
        arousals.append(part.arousal.item())

    colors = ['red' if a == max(arousals) else 'steelblue' for a in arousals]
    ax.barh(parts, arousals, color=colors, edgecolor='white')
    ax.set_title('Bottleneck Detection (Red = Highest Arousal)')
    ax.set_xlabel('Arousal')
    ax.set_xlim(0, 1)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
