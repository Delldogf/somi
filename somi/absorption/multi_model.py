"""
SOMI 4.0 Multi-Model Absorption
==================================

Absorb knowledge from MULTIPLE specialist models at once.
Each specialist's delta goes to a different Part (brain region).

Example:
  - Code specialist delta -> Part 0 (becomes the "coding region")
  - Math specialist delta -> Part 1 (becomes the "math region")
  - Writing specialist delta -> Part 2 (becomes the "writing region")
  - General LLM delta -> Part 3 (becomes the "general knowledge region")

The brain then routes between these specialized Parts through its Systems,
combining expertise as needed.
"""

import torch
from typing import Dict, List, Optional, Tuple

from .transplant import compute_delta, transplant_knowledge


def absorb_multiple(
    brain: 'SOMICircuitBrain',
    specialists: List[Dict],
    base_weights: torch.Tensor,
) -> Dict[str, float]:
    """
    Absorb knowledge from multiple specialist models.

    Args:
        brain: The SOMICircuitBrain to absorb into
        specialists: List of dicts, each with:
            - 'weights': torch.Tensor — specialist weight matrix
            - 'part_id': int — which Part to absorb into
            - 'strength': float — absorption strength (0.0 to 1.0)
            - 'name': str — specialist name (for logging)
        base_weights: The base model weights (shared baseline)

    Returns:
        diagnostics: Dict with per-specialist absorption metrics
    """
    all_diagnostics = {}

    for i, spec in enumerate(specialists):
        # Compute what this specialist learned
        delta, delta_diag = compute_delta(spec['weights'], base_weights)

        # Transplant into the target Part
        part_id = spec.get('part_id', i % len(brain.parts))
        part = brain.parts[str(part_id)]
        strength = spec.get('strength', 1.0)
        name = spec.get('name', f'specialist_{i}')

        transplant_diag = transplant_knowledge(
            part=part,
            delta=delta,
            strength=strength,
        )

        # Log with specialist name
        for k, v in {**delta_diag, **transplant_diag}.items():
            all_diagnostics[f'{name}_{k}'] = v

        all_diagnostics[f'{name}_target_part'] = part_id

    all_diagnostics['total_specialists_absorbed'] = len(specialists)

    return all_diagnostics
