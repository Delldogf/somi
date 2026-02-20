"""
SOMI 4.0 Circuit Brain — The Full Assembly
=============================================

The Circuit Brain is the complete SOMI model:
  X-Encoder -> [System 0, System 1, ...] -> Y-Decoder

Where each System routes through SHARED Parts connected by White Matter.

Architecture (for Circuit-S with 4 Parts, 2 Systems):

    Input
      |
    X-Encoder (linear, backprop-trained)
      |
    +--> System 0: Part[0] -> WM -> Part[1] -> WM -> Part[2]
    |
    +--> System 1: Part[0] -> WM -> Part[3]
      |
    Aggregate (weighted sum of System outputs)
      |
    Y-Decoder (linear, backprop-trained)
      |
    Output

Notice: Part[0] is SHARED — it appears in both Systems.
This is the "PFC" Part. It must generalize across all tasks.

Dual Learning:
  - MACRO parameters (X-Encoder, Y-Decoder, White Matter): backprop
  - MICRO parameters (W_local in each Part): local SOMI physics (no backprop)
  - Straight-through estimator bridges the two

Source: SOMI_3_0/theory/02_THE_CIRCUIT_BRAIN.md, 01_FROM_OSCILLATORS_TO_INTELLIGENCE.md
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from ..config import SOMIBrainConfig
from .part import SOMIPart
from .white_matter import WhiteMatter
from .system import SOMISystem


class SOMICircuitBrain(nn.Module):
    """
    The complete SOMI 4.0 Circuit Brain.

    This is the main model class that puts everything together:
    1. Creates all Parts (brain regions)
    2. Creates White Matter (connections between Parts)
    3. Creates Systems (circuits routing through shared Parts)
    4. Creates X-Encoder and Y-Decoder (macro interface)
    5. Manages dual learning (macro backprop + micro local physics)

    Usage:
        config = SOMIBrainConfig.circuit_s()
        brain = SOMICircuitBrain(config, input_dim=768, output_dim=32000)
        output, diagnostics = brain(input_embeddings, training=True)

    Args:
        config: SOMIBrainConfig
        input_dim: Dimension of input (e.g., transformer embedding dim)
        output_dim: Dimension of output (e.g., vocabulary size)
    """

    def __init__(
        self,
        config: SOMIBrainConfig,
        input_dim: int = 768,
        output_dim: int = 32000,
    ):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # ===== X-Encoder (macro, backprop-trained) =====
        # Projects from input space to SOMI's hidden dimension
        # LayerNorm stabilizes the signal before it enters the physics.
        # Without it, the X-Encoder output magnitude varies wildly between
        # training steps, sending the SOMI physics into instability.
        self.x_encoder = nn.Linear(input_dim, H)
        self.x_norm = nn.LayerNorm(H)

        # ===== Parts (micro, locally-trained) =====
        # Create all Parts. They're stored in a ModuleDict so PyTorch
        # tracks their buffers/parameters correctly.
        self.parts = nn.ModuleDict()
        for part_id in range(config.n_parts):
            self.parts[str(part_id)] = SOMIPart(part_id, config)

        # ===== White Matter (macro, backprop-trained) =====
        self.white_matter = WhiteMatter(config)

        # ===== Systems (circuits through shared Parts) =====
        self.systems = nn.ModuleList()
        for sys_id, route in enumerate(config.system_routes):
            self.systems.append(SOMISystem(
                system_id=sys_id,
                route=route,
                parts=self.parts,
                white_matter=self.white_matter,
                config=config,
            ))

        # ===== System Aggregation Weights =====
        # Learnable weights for combining System outputs
        self.system_weights = nn.Parameter(
            torch.ones(config.n_systems) / config.n_systems
        )

        # ===== Y-Decoder (macro, backprop-trained) =====
        # Projects from SOMI's hidden dimension to output space
        # LayerNorm before decoding stabilizes the logits.
        self.y_norm = nn.LayerNorm(H)
        self.y_decoder = nn.Linear(H, output_dim)

        # ===== Step counter =====
        self.register_buffer('global_step', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        phi_target: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass through the Circuit Brain.

        Steps:
        1. X-Encoder: project input to SOMI's hidden space
        2. Run each System in parallel (each routes through its Parts)
        3. Aggregate System outputs (weighted sum)
        4. Y-Decoder: project to output space
        5. Collect diagnostics from all components

        Dual Learning:
        - The X-Encoder and Y-Decoder are normal PyTorch modules.
          Gradients flow through them via backprop (macro learning).
        - The Parts update their W_local via SOMI stress/STDP (micro learning).
          No gradients flow into W_local.
        - The straight-through estimator: during backprop, gradients pass
          THROUGH the settled phi as if the settling didn't happen.
          This lets the macro parameters (X, Y, WM) learn from the loss,
          while the micro parameters (W_local) learn from physics.

        Args:
            x: [batch, seq, input_dim] input embeddings
            phi_target: [batch, seq, hidden] JEPA target (optional)
            training: Training mode flag

        Returns:
            logits: [batch, seq, output_dim] output logits
            diagnostics: Dict with all metrics from all components
        """
        # 1. X-Encoder (with LayerNorm for stability)
        h = self.x_norm(self.x_encoder(x))  # [batch, seq, hidden], normalized

        # 2. Run all Systems
        system_outputs = []
        all_diagnostics = {}

        for system in self.systems:
            sys_output, sys_diag = system(
                h, phi_target=phi_target, training=training
            )
            system_outputs.append(sys_output)
            all_diagnostics.update(sys_diag)

        # 3. Aggregate System outputs
        # Softmax weights so they sum to 1
        weights = torch.softmax(self.system_weights, dim=0)
        aggregated = torch.zeros_like(system_outputs[0])
        for i, output in enumerate(system_outputs):
            aggregated = aggregated + weights[i] * output

        # NaN guard: if aggregated is corrupted, fall back to h
        if torch.isnan(aggregated).any():
            aggregated = h.clone()
            all_diagnostics['brain_nan_recovery'] = True

        # Dual learning — gradients flow THROUGH the settled physics to update
        # macro parameters (X-Encoder, Y-Decoder, White Matter), while micro
        # parameters (W_local in Parts) update via local physics rules.
        # With only 5 settling steps, gradient flow through the physics is stable.
        # The W_local buffers don't receive gradients (they're not nn.Parameters),
        # so backprop only affects the macro parameters.

        # 4. Y-Decoder (with LayerNorm for stability)
        logits = self.y_decoder(self.y_norm(aggregated))  # [batch, seq, output_dim]

        # 5. Brain-level diagnostics
        all_diagnostics['brain_global_step'] = self.global_step.item()
        all_diagnostics['brain_system_weights'] = weights.detach().tolist()
        all_diagnostics['brain_aggregated_magnitude'] = (
            aggregated.detach().abs().mean().item()
        )

        # Shared Part diagnostics
        if self.config.shared_part_ids:
            for shared_id in self.config.shared_part_ids:
                part = self.parts[str(shared_id)]
                all_diagnostics[f'shared_part_{shared_id}_mass_mean'] = (
                    part.mass.mean().item()
                )
                all_diagnostics[f'shared_part_{shared_id}_arousal'] = (
                    part.arousal.item()
                )

        # Wilson loop diagnostics (gauge curvature)
        for route in self.config.system_routes:
            if len(route) >= 3:
                loop = route + [route[0]]  # Close the loop
                _, wilson_diag = self.white_matter.compute_wilson_loop(loop)
                for k, v in wilson_diag.items():
                    all_diagnostics[f'wm_{k}'] = v

        self.global_step += 1

        return logits, all_diagnostics

    def grow_brain(self, new_hidden_dim: int):
        """
        Neurogenesis at the whole-brain level.

        Grows ALL Parts to the new hidden dimension, then rebuilds
        White Matter tracts and Encoder/Decoder to match.

        This implements the "Gestation" phase from theory/23_SOMI_LIFE_CYCLE.md
        at the Circuit Brain level. Each growth step:
        1. Grow all Parts (preserves learned W_local in top-left)
        2. Rebuild White Matter tracts (preserves old weights in top-left)
        3. Rebuild X-Encoder/Y-Decoder (preserves old weights in top-left)

        Args:
            new_hidden_dim: Target hidden dimension (must be > current)
        """
        old_H = self.config.hidden_dim
        if new_hidden_dim <= old_H:
            return

        # --- Grow all Parts ---
        for part in self.parts.values():
            part.grow(new_hidden_dim)

        # --- Rebuild White Matter tracts (preserve old weights) ---
        for name, tract in self.white_matter.tracts.items():
            old_down_w = tract.down_proj.weight.data  # [rank, old_H]
            old_up_w = tract.up_proj.weight.data      # [old_H, rank]
            rank = tract.rank

            new_down = nn.Linear(new_hidden_dim, rank, bias=False)
            new_up = nn.Linear(rank, new_hidden_dim, bias=False)
            nn.init.xavier_uniform_(new_down.weight, gain=0.1)
            nn.init.xavier_uniform_(new_up.weight, gain=0.1)

            with torch.no_grad():
                new_down.weight.data[:, :old_H] = old_down_w
                new_up.weight.data[:old_H, :] = old_up_w

            device = old_down_w.device
            tract.down_proj = new_down.to(device)
            tract.up_proj = new_up.to(device)

        # --- Rebuild X-Encoder (preserve old weights) ---
        old_enc_w = self.x_encoder.weight.data  # [old_H, input_dim]
        old_enc_b = self.x_encoder.bias.data    # [old_H]
        input_dim = self.x_encoder.in_features

        new_encoder = nn.Linear(input_dim, new_hidden_dim)
        nn.init.xavier_uniform_(new_encoder.weight)
        nn.init.zeros_(new_encoder.bias)
        with torch.no_grad():
            new_encoder.weight.data[:old_H, :] = old_enc_w
            new_encoder.bias.data[:old_H] = old_enc_b

        device = old_enc_w.device
        self.x_encoder = new_encoder.to(device)
        self.x_norm = nn.LayerNorm(new_hidden_dim).to(device)

        # --- Rebuild Y-Decoder (preserve old weights) ---
        old_dec_w = self.y_decoder.weight.data  # [output_dim, old_H]
        old_dec_b = self.y_decoder.bias.data    # [output_dim]
        output_dim = self.y_decoder.out_features

        new_decoder = nn.Linear(new_hidden_dim, output_dim)
        nn.init.xavier_uniform_(new_decoder.weight)
        nn.init.zeros_(new_decoder.bias)
        with torch.no_grad():
            new_decoder.weight.data[:, :old_H] = old_dec_w
            new_decoder.bias.data[:] = old_dec_b

        self.y_decoder = new_decoder.to(device)
        self.y_norm = nn.LayerNorm(new_hidden_dim).to(device)

        # --- Update config ---
        self.config.hidden_dim = new_hidden_dim

    def add_part(self) -> int:
        """
        Add a new Part to the brain (structural neurogenesis).

        When to call: stress is high on shared Parts but low on
        specialized ones -- the brain needs a new region.

        Steps:
        1. Create a new SOMIPart with the current hidden_dim
        2. Add white matter tracts from/to the new Part
        3. Update config (n_parts, shared_part_ids)
        4. Return the new Part's ID

        The new Part starts empty (random W_local) and learns from
        stress like every other Part.
        """
        new_id = self.config.n_parts
        device = next(self.parameters()).device
        H = self.config.hidden_dim

        new_part = SOMIPart(new_id, self.config).to(device)
        self.parts[str(new_id)] = new_part

        self.config.n_parts = new_id + 1

        shared_id = self.config.shared_part_ids[0] if self.config.shared_part_ids else 0
        edge_from = f'{shared_id}_to_{new_id}'
        edge_to = f'{new_id}_to_{shared_id}'

        from .white_matter import WhiteMatterTract
        self.white_matter.tracts[edge_from] = WhiteMatterTract(
            H, self.config.white_matter_rank, shared_id, new_id
        ).to(device)
        self.white_matter.tracts[edge_to] = WhiteMatterTract(
            H, self.config.white_matter_rank, new_id, shared_id
        ).to(device)

        return new_id

    def add_system(self, route: list) -> int:
        """
        Add a new System (circuit pathway) to the brain.

        When to call: a new task or domain can't be routed through
        existing Systems -- the brain needs a new circuit.

        Args:
            route: List of Part IDs for the new System to route through.
                   Should include at least one shared Part for
                   generalization pressure.

        Returns:
            The new System's ID.
        """
        new_sys_id = len(self.systems)

        new_system = SOMISystem(
            system_id=new_sys_id,
            route=route,
            parts=self.parts,
            white_matter=self.white_matter,
            config=self.config,
        )
        self.systems.append(new_system)

        old_weights = self.system_weights.data
        new_weights = torch.ones(len(self.systems), device=old_weights.device)
        new_weights[:len(old_weights)] = old_weights
        new_weights[-1] = old_weights.mean()
        self.system_weights = nn.Parameter(new_weights)

        self.config.n_systems = len(self.systems)
        if self.config.system_routes is not None:
            self.config.system_routes.append(route)

        self.config.shared_part_ids = self.config._auto_detect_shared_parts()

        return new_sys_id

    def recalibrate_config(self):
        """
        Re-derive action-based parameters for the current size.

        Call after grow_brain(), add_part(), or add_system() so that
        physics parameters (alpha_1, dt, target_zeta, etc.) match the
        new architecture dimensions.
        """
        from ..physics.action_derived import derive_all_from_action
        d = derive_all_from_action(self.config.hidden_dim, self.config.n_parts)

        self.config.alpha_1 = d['alpha_1']
        self.config.alpha_0 = d['alpha_0']
        self.config.dt = d['dt']
        self.config.target_zeta = d['target_zeta']
        self.config.timescale_ratio = d['timescale_ratio']
        self.config.lambda_W = d['lambda_W']
        self.config.noise_ratio = d['noise_ratio']
        self.config.kappa_0 = d['kappa_0']
        self.config.kappa_1 = d['kappa_1']
        self.config.kappa_stdp = d['kappa_stdp']
        self.config.surprise_gate = d['surprise_gate']
        self.config.sparsity = d['sparsity']
        self.config.lambda_E = d['lambda_E']
        self.config.lambda_C = d['lambda_C']

    def get_all_parts_diagnostics(self) -> Dict:
        """Get a summary of all Parts' health."""
        diag = {}
        for pid, part in self.parts.items():
            diag[f'part_{pid}_mass_mean'] = part.mass.mean().item()
            diag[f'part_{pid}_mass_std'] = part.mass.std().item()
            diag[f'part_{pid}_arousal'] = part.arousal.item()
            diag[f'part_{pid}_W_mean'] = part.W_local.abs().mean().item()
            diag[f'part_{pid}_W_sparsity'] = (
                (part.W_local.abs() < 1e-6).float().mean().item()
            )
        return diag

    def test_time_update(
        self,
        x: torch.Tensor,
        phi_target: torch.Tensor,
    ) -> Dict:
        """
        Test-time learning: update W_local when surprised.

        This is what makes SOMI fundamentally different from transformers:
        it can learn AT INFERENCE TIME without backprop.

        If the model is "surprised" (high stress on a Part), it updates
        that Part's W_local to adapt to the new input.

        Args:
            x: [batch, seq, input_dim] input
            phi_target: [batch, seq, hidden] expected target

        Returns:
            diagnostics: Which Parts updated and how much
        """
        if not self.config.test_time_learning:
            return {}

        h = self.x_encoder(x)
        diagnostics = {}

        for pid, part in self.parts.items():
            # Quick forward pass to measure surprise
            phi = h.clone()
            output, phi_dot, settle_diag = part(
                phi, phi_target=phi_target, training=True  # training=True enables W update
            )
            diagnostics[f'test_time_part_{pid}_updated'] = True

        return diagnostics
