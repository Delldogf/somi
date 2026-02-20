"""
SOMI 4.0 Gauge Plots — 4 Level 4 visualizations
===================================================

1. Wilson loop heatmap (curvature around paths)
2. Curvature distribution (per-tract curvature)
3. Chern-Simons timeline (topological invariant over training)
4. Instanton markers (sudden topology changes)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_wilson_loops(
    gauge_diagnostics: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Wilson loop traces for each system."""
    fig, ax = plt.subplots(figsize=(8, 5))

    systems = []
    traces = []
    for k, v in sorted(gauge_diagnostics.items()):
        if 'wilson_trace' in k:
            sys_name = k.split('_')[1]
            systems.append(sys_name)
            traces.append(v)

    if not systems:
        ax.text(0.5, 0.5, 'No Wilson loop data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    ax.bar(systems, traces, color='teal', edgecolor='white')
    ax.set_title('Wilson Loop Traces (higher = flatter connection)')
    ax.set_ylabel('Trace(W_loop)')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_chern_simons_timeline(
    cs_history: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Chern-Simons invariant over training."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cs_history, 'b-', linewidth=2)
    ax.set_title('Chern-Simons Invariant Over Training')
    ax.set_xlabel('Step')
    ax.set_ylabel('CS Value')
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
