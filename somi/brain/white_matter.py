"""
SOMI 4.0 White Matter — Gauge Connections Between Parts
=========================================================

White Matter connects Parts (brain regions) to each other, like the
highway system connecting cities. In the brain, white matter is literally
white-colored nerve fiber bundles (axons wrapped in myelin) that carry
signals between different brain regions.

In SOMI physics (Level 4 — Gauge Theory):
  White Matter = gauge connections (like parallel transport in GR)
  Each connection transforms signals as they travel between Parts.

Implementation: Low-rank linear projections between Parts.
  signal_out = W_down @ signal_in @ W_up  (bottleneck = white_matter_rank)

Why low-rank:
  - Real white matter has limited bandwidth (finite number of axon bundles)
  - Forces the connection to compress information (only essential signals pass)
  - Fewer parameters = easier to learn

Wilson loops (Level 4 diagnostic):
  Transport a signal around a closed loop of Parts. If it comes back
  unchanged, the connection is "flat" (no curvature). If it changes,
  there's "curvature" — the connections are doing nontrivial transformations.

Source: SOMI_3_0/theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md (Level 4)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class WhiteMatterTract(nn.Module):
    """
    A single white matter tract connecting two Parts.

    This is a low-rank linear transformation:
        signal_out = up_proj(down_proj(signal_in))

    The bottleneck (rank) forces compression — only the most important
    information passes through. Like a highway that can only carry a
    limited number of cars at once.

    In gauge theory terms, this is the "connection" or "parallel transport"
    operator. It defines how vectors (signals) are mapped from one Part's
    coordinate system to another's.

    Args:
        hidden_dim: Dimension of signals in each Part
        rank: Bottleneck dimension (compression ratio)
        source_id: ID of the source Part
        target_id: ID of the target Part
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        source_id: int,
        target_id: int,
    ):
        super().__init__()
        self.source_id = source_id
        self.target_id = target_id
        self.rank = rank

        # Low-rank projection: hidden -> rank -> hidden
        self.down_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_dim, bias=False)

        # Initialize with small weights (signals start weak, strengthen with use)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transport a signal from source Part to target Part.

        Args:
            x: [batch, seq, hidden] signal from source Part

        Returns:
            y: [batch, seq, hidden] signal transformed for target Part
        """
        return self.up_proj(self.down_proj(x))

    def get_transport_matrix(self) -> torch.Tensor:
        """
        Get the full transport matrix (for Wilson loop computation).

        Returns:
            T: [hidden, hidden] the full low-rank transport matrix
        """
        return self.up_proj.weight @ self.down_proj.weight


class WhiteMatter(nn.Module):
    """
    The complete white matter system connecting all Parts.

    Manages all the tracts (connections) between Parts and provides
    methods for:
    1. Sending signals between Parts
    2. Computing Wilson loops (gauge curvature diagnostic)
    3. Computing connection curvature (how much signals are distorted)

    Structure: One tract per directed edge in the system routes.
    If System 0 uses Parts [0, 1, 2], there are tracts: 0->1, 1->2, 2->0.

    Args:
        config: SOMIBrainConfig
    """

    def __init__(self, config: 'SOMIBrainConfig'):
        super().__init__()
        self.config = config

        # Create tracts for each connection implied by system routes
        self.tracts = nn.ModuleDict()
        edges = set()

        for route in config.system_routes:
            for i in range(len(route) - 1):
                src = route[i]
                tgt = route[i + 1]
                edge_key = f'{src}_to_{tgt}'
                if edge_key not in edges:
                    edges.add(edge_key)
                    self.tracts[edge_key] = WhiteMatterTract(
                        hidden_dim=config.hidden_dim,
                        rank=config.white_matter_rank,
                        source_id=src,
                        target_id=tgt,
                    )

    def transport(
        self,
        signal: torch.Tensor,
        source_id: int,
        target_id: int,
    ) -> torch.Tensor:
        """
        Transport a signal from one Part to another through white matter.

        Args:
            signal: [batch, seq, hidden] from source Part
            source_id: Source Part ID
            target_id: Target Part ID

        Returns:
            transported: [batch, seq, hidden] signal for target Part
        """
        edge_key = f'{source_id}_to_{target_id}'
        if edge_key in self.tracts:
            return self.tracts[edge_key](signal)
        else:
            # No direct tract — return signal unchanged (identity transport)
            return signal

    def compute_wilson_loop(
        self,
        part_ids: List[int],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Wilson loop around a closed path of Parts.

        A Wilson loop measures the "curvature" of the gauge connections.
        Transport a signal around a closed loop: A -> B -> C -> A.
        If the result equals the original, curvature = 0 (flat connection).
        If it differs, there's curvature (nontrivial transformation).

        In physics: Wilson loops detect gauge field strength (like measuring
        magnetic flux by sending a charge around a loop).

        In SOMI: High curvature means the Parts have very different
        "coordinate systems" — they process information differently.
        Some curvature is good (specialization). Too much is bad (incompatibility).

        Args:
            part_ids: List of Part IDs forming a loop (e.g., [0, 1, 2, 0])

        Returns:
            W_loop: [hidden, hidden] — the composition of all transport matrices
            diagnostics: Dict with curvature metrics
        """
        if len(part_ids) < 2:
            H = self.config.hidden_dim
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
            return torch.eye(H, device=device), {'wilson_trace': float(H)}

        # Compose transport matrices along the loop
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        H = self.config.hidden_dim
        W_loop = torch.eye(H, device=device)

        for i in range(len(part_ids) - 1):
            src = part_ids[i]
            tgt = part_ids[i + 1]
            edge_key = f'{src}_to_{tgt}'
            if edge_key in self.tracts:
                T = self.tracts[edge_key].get_transport_matrix()
                W_loop = T @ W_loop
            # else: identity (no tract = no transformation)

        # Curvature diagnostic
        trace = W_loop.trace().item()
        # Perfect flat connection: trace = hidden_dim (identity matrix)
        # Deviation from identity measures curvature
        deviation = (W_loop - torch.eye(H, device=device)).norm().item()

        diagnostics = {
            'wilson_trace': trace,
            'wilson_deviation': deviation,
            'wilson_curvature': deviation / H,  # Normalized
            'wilson_loop_length': len(part_ids) - 1,
        }

        return W_loop, diagnostics

    def get_all_curvatures(self) -> Dict[str, float]:
        """
        Compute curvature for all tracts (simplified: just tract norms).

        Returns:
            diagnostics: Dict mapping tract names to their norms
        """
        diagnostics = {}
        for name, tract in self.tracts.items():
            T = tract.get_transport_matrix()
            diagnostics[f'tract_{name}_norm'] = T.norm().item()
            diagnostics[f'tract_{name}_rank_ratio'] = (
                tract.rank / self.config.hidden_dim
            )

        return diagnostics
