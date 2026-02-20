"""
SOMI 4.0 System — A Brain Circuit Routing Through Shared Parts
================================================================

A System is a circuit — a path through multiple Parts.
Example: Language System = [Auditory Part -> Association Part -> Motor Part]

THE KEY INSIGHT: Parts are SHARED across Systems.
  - Association Part might appear in BOTH the Language System AND the Vision System
  - This forces it to learn representations that work for BOTH tasks
  - This is "generalization pressure" — the core of SOMI's architecture

NOT Mixture of Experts (MoE):
  - MoE: Each expert is separate, router picks one
  - SOMI: Parts participate in MULTIPLE circuits SIMULTANEOUSLY
  - SOMI is more like the real brain: PFC is in every cognitive circuit

Source: SOMI_3_0/theory/02_THE_CIRCUIT_BRAIN.md
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .part import SOMIPart
from .white_matter import WhiteMatter


class SOMISystem(nn.Module):
    """
    A single brain circuit (System) routing through a sequence of Parts.

    A System doesn't OWN its Parts — it borrows them. Multiple Systems
    can borrow the same Part. This sharing is what creates generalization.

    Think of it like a factory assembly line:
    - Each station (Part) does its processing
    - The conveyor belt (White Matter) moves work between stations
    - Multiple product lines (Systems) might share the same stations
    - Shared stations must work for all products = generalization

    Processing flow for one System:
        input -> Part[0] -> WhiteMatter(0->1) -> Part[1] -> ... -> Part[N] -> output

    Args:
        system_id: Unique ID for this System
        route: List of Part IDs this System routes through
        parts: Dict mapping Part ID to SOMIPart (shared, not owned)
        white_matter: WhiteMatter connections
        config: SOMIBrainConfig
    """

    def __init__(
        self,
        system_id: int,
        route: List[int],
        parts: nn.ModuleDict,
        white_matter: WhiteMatter,
        config: 'SOMIBrainConfig',
    ):
        super().__init__()
        self.system_id = system_id
        self.route = route
        self.parts = parts  # Shared reference, not owned
        self.white_matter = white_matter
        self.config = config

    def forward(
        self,
        h: torch.Tensor,
        phi_target: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Route signal through the System's Parts.

        Each Part:
        1. Receives signal (from input or white matter)
        2. Settles its phi (SOMI physics)
        3. Outputs settled representation
        4. White matter transports it to the next Part

        Args:
            h: [batch, seq, hidden] input signal
            phi_target: Optional target for JEPA loss
            training: Training mode

        Returns:
            output: [batch, seq, hidden] final output after all Parts
            diagnostics: Dict with per-Part and per-tract metrics
        """
        signal = h
        all_diagnostics = {}
        part_outputs = []

        for step_idx, part_id in enumerate(self.route):
            part = self.parts[str(part_id)]

            # Settle this Part
            output, phi_dot, part_diag = part(
                signal, phi_target=phi_target, training=training
            )
            part_outputs.append(output)

            # Add system-specific diagnostics prefix
            for k, v in part_diag.items():
                all_diagnostics[f'sys{self.system_id}_{k}'] = v

            # Transport through white matter to next Part
            if step_idx < len(self.route) - 1:
                next_part_id = self.route[step_idx + 1]
                signal = self.white_matter.transport(
                    output, source_id=part_id, target_id=next_part_id
                )
            else:
                signal = output

        # System-level diagnostics
        all_diagnostics[f'system_{self.system_id}_n_parts'] = len(self.route)
        all_diagnostics[f'system_{self.system_id}_output_magnitude'] = (
            signal.detach().abs().mean().item()
        )

        return signal, all_diagnostics
