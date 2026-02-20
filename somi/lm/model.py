"""
Pure SOMI-SSM Language Model
==============================

A language model where SOMI IS the architecture — no transformer inside.
Each layer is a SOMI Part with SSM-based settling, local learning, and
the ability to grow (neurogenesis).

Architecture:
    Input Tokens -> Embedding + Position -> [SOMI-SSM Layer 0] -> ... -> [SOMI-SSM Layer N] -> LayerNorm -> LM Head -> Logits

Each SOMI-SSM Layer:
    - Has its own W_local (connection geometry, evolves via local stress/STDP)
    - Settles representations using the SSM solver (closed-form, fast)
    - Applies residual connection + LayerNorm for stability
    - Can grow via neurogenesis (grow() method)

Training:
    - LM head + embeddings trained via backprop (macro learning)
    - W_local in each layer evolves via SOMI stress physics (micro learning, no backprop)
    - Dual learning: gradients flow through the settled phi via straight-through

Usage:
    model = SOMILanguageModel(vocab_size=32000, hidden_dim=256, n_layers=8)
    logits = model(input_ids)  # standard next-token prediction
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple

from somi.config import SOMIBrainConfig
from somi.physics.forces import compute_laplacian
from somi.physics.settling import settle, compute_eigendecomposition
from somi.physics.geometry import (
    compute_stress_tensor, geometry_step, initialize_W,
    structural_plasticity, mass_conductivity_constraint,
)


class SOMISSMLayer(nn.Module):
    """
    A single SOMI-SSM language model layer.

    Replaces a transformer decoder layer. Instead of attention + MLP,
    representations settle through a physics-based field equation on
    the feature geometry W.

    Forward pass:
    1. Input arrives as [batch, seq, hidden]
    2. SSM settling: solve the damped harmonic oscillator on features
    3. Residual connection: output = input + settled - input = settled
    4. LayerNorm for stability

    Local learning (no backprop):
    - After settling, compute stress tensor S from prediction error
    - Update W via geometry equation: W_dot = -eta * S
    - Periodic structural plasticity (prune/grow connections)
    """

    def __init__(self, layer_id: int, config: SOMIBrainConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        H = config.hidden_dim

        W_init, mask = initialize_W(H, config.sparsity, torch.device('cpu'))
        self.register_buffer('W_local', W_init)
        self.register_buffer('mask', mask)

        h_idx = (W_init ** 2).sum(dim=1).clamp(min=1e-8)
        h_mean = h_idx.mean()
        mass = (config.M * h_mean / h_idx).clamp(0.1, 10.0)
        self.register_buffer('mass', mass)

        self.register_buffer('eigenvalues', torch.zeros(H))
        self.register_buffer('eigenvectors', torch.eye(H))
        self.register_buffer('error_variance', torch.ones(H))
        self.register_buffer('arousal', torch.tensor(0.5))
        self.register_buffer('error_running_avg', torch.tensor(1.0))

        self.norm = nn.LayerNorm(H)

        from somi.physics.forces import BasalGangliaGate
        self.gate = BasalGangliaGate(H, config.gate_bottleneck)

        self.register_buffer('global_step', torch.tensor(0))
        self._prev_stress = None

        self._update_eigen()

    def _update_eigen(self):
        L_rw = compute_laplacian(self.W_local)
        eigenvalues, eigenvectors, _ = compute_eigendecomposition(L_rw)
        n = min(eigenvalues.shape[0], self.eigenvalues.shape[0])
        self.eigenvalues.zero_()
        self.eigenvalues[:n] = eigenvalues[:n]
        if eigenvectors.shape[1] <= self.eigenvectors.shape[1]:
            self.eigenvectors.zero_()
            self.eigenvectors[:, :eigenvectors.shape[1]] = eigenvectors
        else:
            self.eigenvectors.copy_(eigenvectors[:, :self.eigenvectors.shape[1]])

    def forward(
        self, h: torch.Tensor, training: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        device = h.device

        # Clone + detach physics tensors: local learning modifies them
        # in-place later, so we need independent copies for the graph.
        W_snap = self.W_local.detach().clone()
        eig_vals = self.eigenvalues.detach().clone()
        eig_vecs = self.eigenvectors.detach().clone()
        mass_snap = self.mass.detach().clone()

        L_rw = compute_laplacian(W_snap)
        precision = (1.0 / self.error_variance.clamp(min=0.1)).to(device)

        omega_med = math.sqrt(max(1e-8,
            self.config.alpha_1 * max(0.01, eig_vals[1].item() if eig_vals.shape[0] > 1 else 1.0)
            + self.config.alpha_0 + 1.0))
        beta = 2 * self.config.target_zeta * omega_med
        n_settle = max(3, min(10, int(math.pi / (omega_med * self.config.dt))))

        phi = h.clone()
        phi_target = h

        phi_settled, phi_dot, settle_info = settle(
            phi=phi, phi_target=phi_target,
            W=W_snap, L_rw=L_rw, precision=precision,
            beta=beta, n_steps=n_settle, config=self.config,
            gate=self.gate, training=training,
            M_vector=mass_snap,
            eigenvalues=eig_vals,
            eigenvectors=eig_vecs,
            method='ssm',
        )

        output = self.norm(phi_settled)

        diagnostics = {
            f'layer_{self.layer_id}_settle_steps': n_settle,
            f'layer_{self.layer_id}_velocity': phi_dot.detach().abs().mean().item(),
        }

        if training:
            with torch.no_grad():
                if not torch.isnan(phi_settled).any():
                    error = phi_settled - phi_target
                    error_mag = error.abs().mean().item()

                    self.error_running_avg = (
                        self.config.arousal_ema * self.error_running_avg
                        + (1 - self.config.arousal_ema) * error_mag
                    )
                    arousal_input = error_mag / (self.error_running_avg.item() + 1e-8) - 1.0
                    self.arousal = torch.sigmoid(torch.tensor(arousal_input, device=device))

                    alpha = self.config.precision_ema
                    err_var = error.detach().pow(2).mean(dim=tuple(range(error.dim()-1)))
                    self.error_variance = alpha * self.error_variance + (1 - alpha) * err_var.to(self.error_variance.device)

                    S, _ = compute_stress_tensor(phi_settled, phi_target, self.config, phi_dot)
                    eta = (0.1 / self.config.timescale_ratio) * (0.5 + self.arousal.item())

                    self.W_local, _ = geometry_step(
                        W=self.W_local, S=S, eta=eta, config=self.config, mask=self.mask,
                    )
                    self.W_local.clamp_(-5.0, 5.0)

                    self.global_step += 1
                    if self.global_step % self.config.plasticity_interval == 0:
                        self.W_local, self.mask = structural_plasticity(
                            self.W_local, self.mask, S,
                            target_sparsity=self.config.sparsity,
                        )
                    if self.global_step % self.config.eigen_update_interval == 0:
                        cpu_W = self.W_local.cpu()
                        self.eigenvalues = self.eigenvalues.cpu()
                        self.eigenvectors = self.eigenvectors.cpu()
                        self.W_local = cpu_W
                        self._update_eigen()
                        self.W_local = self.W_local.to(device)
                        self.eigenvalues = self.eigenvalues.to(device)
                        self.eigenvectors = self.eigenvectors.to(device)

                    diagnostics[f'layer_{self.layer_id}_stress'] = S.abs().mean().item()
                    diagnostics[f'layer_{self.layer_id}_arousal'] = self.arousal.item()

        if torch.isnan(output).any():
            output = h.clone()

        return output, diagnostics

    def grow(self, new_hidden_dim: int):
        """Neurogenesis: expand this layer's capacity."""
        old_H = self.W_local.shape[0]
        if new_hidden_dim <= old_H:
            return

        device = self.W_local.device
        W_new, mask_new = initialize_W(new_hidden_dim, self.config.sparsity, torch.device('cpu'))
        with torch.no_grad():
            W_new[:old_H, :old_H] = self.W_local.cpu()
            mask_new[:old_H, :old_H] = self.mask.cpu()
        self.register_buffer('W_local', W_new.to(device))
        self.register_buffer('mask', mask_new.to(device))

        h_idx = (self.W_local ** 2).sum(dim=1).clamp(min=1e-8)
        h_mean = h_idx.mean()
        mass_new = (self.config.M * h_mean / h_idx).clamp(0.1, 10.0)
        with torch.no_grad():
            mass_new[:old_H] = self.mass[:old_H]
        self.register_buffer('mass', mass_new)

        self.register_buffer('error_variance', torch.ones(new_hidden_dim, device=device))
        self.register_buffer('eigenvalues', torch.zeros(new_hidden_dim, device=device))
        self.register_buffer('eigenvectors', torch.eye(new_hidden_dim, device=device))

        from somi.physics.forces import BasalGangliaGate
        self.gate = BasalGangliaGate(new_hidden_dim, self.config.gate_bottleneck).to(device)

        self.norm = nn.LayerNorm(new_hidden_dim).to(device)

        if device.type != 'cpu':
            self.W_local = self.W_local.cpu()
            self.eigenvalues = self.eigenvalues.cpu()
            self.eigenvectors = self.eigenvectors.cpu()
            self._update_eigen()
            self.register_buffer('W_local', self.W_local.to(device))
            self.register_buffer('eigenvalues', self.eigenvalues.to(device))
            self.register_buffer('eigenvectors', self.eigenvectors.to(device))
        else:
            self._update_eigen()


class SOMILanguageModel(nn.Module):
    """
    Pure SOMI-SSM Language Model.

    No transformer inside. Representations flow through stacked SOMI-SSM
    layers where physics-based settling replaces attention + MLP.

    Args:
        vocab_size: vocabulary size
        hidden_dim: feature dimension per layer
        n_layers: number of SOMI-SSM layers
        max_seq_len: maximum sequence length for position embeddings
        config: optional SOMIBrainConfig (None = auto-derive)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 256,
        n_layers: int = 8,
        max_seq_len: int = 2048,
        config: Optional[SOMIBrainConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = SOMIBrainConfig.auto(hidden_dim=hidden_dim, n_parts=1)
        self.config = config
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.embed_pos = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList([
            SOMISSMLayer(layer_id=i, config=config)
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        self.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq] token IDs
            labels: [batch, seq] target IDs for loss (optional)
            training: enables local learning in SOMI layers

        Returns:
            logits: [batch, seq, vocab]
            loss: scalar cross-entropy loss (if labels provided)
            diagnostics: Dict with per-layer metrics
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        h = self.embed_tokens(input_ids) + self.embed_pos(positions)

        all_diagnostics = {}
        for layer in self.layers:
            h, layer_diag = layer(h, training=training)
            all_diagnostics.update(layer_diag)

        h = self.final_norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, all_diagnostics

    def grow(self, new_hidden_dim: int):
        """
        Neurogenesis: grow the entire model to a larger hidden dimension.

        Expands all layers, embeddings, and the LM head while preserving
        learned weights in the top-left corner.
        """
        old_H = self.hidden_dim
        if new_hidden_dim <= old_H:
            return

        device = next(self.parameters()).device

        for layer in self.layers:
            layer.grow(new_hidden_dim)

        old_embed_w = self.embed_tokens.weight.data
        vocab_size = old_embed_w.shape[0]
        new_embed = nn.Embedding(vocab_size, new_hidden_dim)
        nn.init.normal_(new_embed.weight, std=0.02)
        with torch.no_grad():
            new_embed.weight.data[:, :old_H] = old_embed_w
        self.embed_tokens = new_embed.to(device)

        old_pos_w = self.embed_pos.weight.data
        max_seq = old_pos_w.shape[0]
        new_pos = nn.Embedding(max_seq, new_hidden_dim)
        nn.init.normal_(new_pos.weight, std=0.02)
        with torch.no_grad():
            new_pos.weight.data[:, :old_H] = old_pos_w
        self.embed_pos = new_pos.to(device)

        self.final_norm = nn.LayerNorm(new_hidden_dim).to(device)

        new_head = nn.Linear(new_hidden_dim, vocab_size, bias=False)
        nn.init.normal_(new_head.weight, std=0.02)
        with torch.no_grad():
            new_head.weight.data[:, :old_H] = old_embed_w
        self.lm_head = new_head.to(device)

        self.embed_tokens.weight = self.lm_head.weight

        self.hidden_dim = new_hidden_dim
        self.config.hidden_dim = new_hidden_dim

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.8,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            logits, _, _ = self(input_ids, training=False)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
