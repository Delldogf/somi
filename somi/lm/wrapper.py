"""
SOMI Wrapper: Add SOMI Physics to Any HuggingFace Language Model
=================================================================

Wraps any HuggingFace causal LM with SOMI settling on every decoder layer.
At initialization (gate=0), the wrapped model behaves identically to the
original. After fine-tuning, SOMI geometry evolves and settling contributes.

Usage:
    model, tokenizer = create_somi_lm("meta-llama/Llama-3.2-1B")
    # model is now SOMI-enhanced but starts as pure transformer
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from somi.config import SOMIBrainConfig
from somi.physics.forces import compute_laplacian
from somi.physics.settling import settle, compute_eigendecomposition


class SOMISettlingLayer(nn.Module):
    """
    SOMI settling dynamics for a single transformer layer's features.

    Runs the SOMI field equation on the hidden dimension, letting features
    "resonate" through the geometry W before being passed to the next layer.

    At gate=0 (initialization): output = input (pure transformer).
    After training: output = input + gate * settling_contribution.
    """

    def __init__(self, hidden_dim: int, config: SOMIBrainConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config

        self.gate_alpha = nn.Parameter(torch.zeros(1))

        from somi.physics.geometry import initialize_W
        W_init, mask = initialize_W(hidden_dim, config.sparsity, torch.device('cpu'))
        self.register_buffer('W', W_init)
        self.register_buffer('mask', mask)

        self.register_buffer('eigenvalues', torch.zeros(hidden_dim))
        self.register_buffer('eigenvectors', torch.eye(hidden_dim))
        self.register_buffer('arousal', torch.tensor(0.5))
        self.register_buffer('error_variance', torch.ones(hidden_dim))
        self.register_buffer('step_count', torch.tensor(0))

        self._update_eigen()

    def _update_eigen(self):
        L_rw = compute_laplacian(self.W)
        eigenvalues, eigenvectors, _ = compute_eigendecomposition(L_rw)
        n = min(eigenvalues.shape[0], self.eigenvalues.shape[0])
        self.eigenvalues.zero_()
        self.eigenvalues[:n] = eigenvalues[:n]
        if eigenvectors.shape[1] <= self.eigenvectors.shape[1]:
            self.eigenvectors.zero_()
            self.eigenvectors[:, :eigenvectors.shape[1]] = eigenvectors
        else:
            self.eigenvectors.copy_(eigenvectors[:, :self.eigenvectors.shape[1]])

    def forward(self, hidden_states: torch.Tensor, update_geometry: bool = False):
        gate = torch.tanh(self.gate_alpha)
        if gate.abs().item() < 1e-6 and not update_geometry:
            return hidden_states

        device = hidden_states.device
        W_dev = self.W.to(device)
        L_rw = compute_laplacian(W_dev)
        precision = (1.0 / self.error_variance.clamp(min=0.1)).to(device)

        omega_med = math.sqrt(max(1e-8, self.config.alpha_1 * 1.0 + self.config.alpha_0 + 1.0))
        beta = 2 * self.config.target_zeta * omega_med
        n_settle = max(3, min(10, int(math.pi / (omega_med * self.config.dt))))

        phi = hidden_states.clone()
        phi_target = hidden_states

        phi_settled, phi_dot, info = settle(
            phi=phi, phi_target=phi_target,
            W=W_dev, L_rw=L_rw, precision=precision,
            beta=beta, n_steps=n_settle, config=self.config,
            training=self.training,
            eigenvalues=self.eigenvalues.to(device),
            eigenvectors=self.eigenvectors.to(device),
            method='ssm',
        )

        output = hidden_states + gate * (phi_settled - hidden_states)

        if update_geometry and self.training:
            with torch.no_grad():
                error = phi_settled - phi_target
                self._update_precision(error)
                self._update_geometry(phi_settled, phi_dot, phi_target, L_rw, device)
                self.step_count += 1
                if self.step_count % self.config.eigen_update_interval == 0:
                    self.W = self.W.to('cpu')
                    self._update_eigen()
                    self.W = self.W.to(device)

        return output

    def _update_precision(self, error: torch.Tensor):
        error_mag = error.detach().pow(2).mean(dim=tuple(range(error.dim() - 1)))
        alpha = self.config.precision_ema
        self.error_variance = alpha * self.error_variance + (1 - alpha) * error_mag.to(self.error_variance.device)

    def _update_geometry(self, phi, phi_dot, phi_target, L_rw, device):
        from somi.physics.geometry import compute_stress_tensor, geometry_step

        S, _ = compute_stress_tensor(phi, phi_target, self.config, phi_dot)

        eta = (0.1 / self.config.timescale_ratio) * (0.5 + self.arousal.item())

        self.W = self.W.to(device)
        self.W, _ = geometry_step(
            W=self.W, S=S, eta=eta, config=self.config, mask=self.mask.to(device),
        )
        self.W.clamp_(-5.0, 5.0)


class SOMIDecoderLayerWrapper(nn.Module):
    """Wraps a transformer decoder layer with SOMI settling."""

    def __init__(self, original_layer: nn.Module, hidden_dim: int, config: SOMIBrainConfig):
        super().__init__()
        self.original_layer = original_layer
        self.somi = SOMISettlingLayer(hidden_dim, config)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        outputs = self.original_layer(hidden_states, *args, **kwargs)

        if isinstance(outputs, torch.Tensor):
            settled = self.somi(outputs, update_geometry=self.training)
            return settled
        elif isinstance(outputs, tuple):
            settled = self.somi(outputs[0], update_geometry=self.training)
            return (settled,) + outputs[1:]
        else:
            settled = self.somi(outputs[0], update_geometry=self.training)
            try:
                outputs_list = list(outputs)
                outputs_list[0] = settled
                return type(outputs)(*outputs_list)
            except (TypeError, ValueError):
                return settled


def _get_decoder_layers(model):
    """Find decoder layers in any HuggingFace causal LM."""
    candidates = [
        ('model.model.layers', lambda m: m.model.layers),
        ('model.transformer.h', lambda m: m.transformer.h),
        ('model.gpt_neox.layers', lambda m: m.gpt_neox.layers),
        ('model.transformer.layers', lambda m: m.transformer.layers),
    ]

    for path, accessor in candidates:
        try:
            layers = accessor(model)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers, path
        except (AttributeError, TypeError):
            continue

    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 10:
            return module, f"model.{name} (auto-detected)"

    raise ValueError(
        f"Cannot find decoder layers in {type(model).__name__}. "
        f"Add the layer path to _get_decoder_layers()."
    )


def create_somi_lm(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    n_settle: int = 5,
    load_in_4bit: bool = True,
    device: str = "auto",
    config: Optional[SOMIBrainConfig] = None,
):
    """
    Create a SOMI-enhanced language model from any HuggingFace causal LM.

    Wraps each transformer decoder layer with SOMI settling dynamics.
    At initialization, the model behaves identically to the original.

    Args:
        model_name: HuggingFace model name or local path
        n_settle: settling steps per layer (5 is a good default)
        load_in_4bit: use 4-bit quantization to save VRAM
        device: device placement ("auto" for automatic)
        config: optional SOMIBrainConfig (None = auto-derive from model)

    Returns:
        model: the SOMI-wrapped model
        tokenizer: the tokenizer (unchanged)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nSOMI Language Model Wrapper")
    print(f"  Base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": device, "trust_remote_code": True}

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    if config is None:
        config = SOMIBrainConfig.auto(hidden_dim=hidden_dim, n_parts=1)
        config.n_settle = n_settle

    layers, layer_path = _get_decoder_layers(model)
    print(f"  Hidden: {hidden_dim}, Layers: {n_layers}")
    print(f"  Layer path: {layer_path}")
    print(f"  Wrapping {n_layers} layers with SOMI (SSM settling)...")

    for i in range(n_layers):
        original = layers[i]
        if isinstance(original, SOMIDecoderLayerWrapper):
            continue

        try:
            layer_device = next(original.parameters()).device
        except StopIteration:
            layer_device = torch.device('cpu')

        wrapped = SOMIDecoderLayerWrapper(original, hidden_dim, config)
        wrapped.somi = wrapped.somi.to(layer_device)
        layers[i] = wrapped

    somi_params = sum(p.numel() for n, p in model.named_parameters() if 'somi' in n)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  SOMI params: {somi_params:,} ({somi_params/max(total_params,1)*100:.1f}% overhead)")
    print(f"  Gate alpha: 0.0 (starts as pure transformer)\n")

    return model, tokenizer
