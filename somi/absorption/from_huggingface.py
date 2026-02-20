"""
SOMI Absorption from HuggingFace Models
==========================================

Loads any HuggingFace transformer model and absorbs its knowledge
into a SOMI brain. Works with any model size because it uses SVD
alignment and spectral transfer to bridge dimension mismatches.

Extraction strategy:
  - Each transformer layer maps to one SOMI Part
  - Attention projection matrices (Q, K, V, O) → W_local connectivity
  - LayerNorm weights → mass (feature importance)
  - MLP weight matrices → additional connectivity signal
  - If dimensions differ: SVD projects to SOMI's hidden_dim

Usage:
    brain = absorb_from_huggingface(
        model_name="Qwen/Qwen2.5-0.5B",
        somi_hidden=128,
    )
    # brain is now a SOMI CircuitBrain initialized with LLM knowledge

Or step by step:
    weights = extract_transformer_weights("Qwen/Qwen2.5-0.5B")
    brain = SOMICircuitBrain(config, input_dim=..., output_dim=...)
    absorb_weights_into_brain(brain, weights)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math

from ..config import SOMIBrainConfig
from ..brain.circuit_brain import SOMICircuitBrain
from .alignment import svd_align
from .transplant import spectral_mode_transfer


def extract_transformer_weights(
    model_name: str,
    device: str = 'cpu',
    max_layers: Optional[int] = None,
) -> Dict:
    """
    Load a HuggingFace model and extract weight matrices.

    This downloads the model (if not cached), extracts the relevant
    weight matrices from each layer, and returns them in a structured
    dict. The model is then deleted to free memory.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "microsoft/phi-2", etc.)
        device: Device to load onto ('cpu' recommended to save GPU memory)
        max_layers: Only extract this many layers (None = all)

    Returns:
        weights: Dict with:
            'model_name': str
            'hidden_dim': int (transformer's hidden dimension)
            'n_layers': int (number of layers extracted)
            'vocab_size': int
            'layers': List of dicts, each with:
                'attn_qkv': tensor or None
                'attn_out': tensor or None
                'mlp_up': tensor or None
                'mlp_down': tensor or None
                'ln_weight': tensor or None
                'connectivity': [H, H] tensor (derived W)
                'mass': [H] tensor (derived from LayerNorm)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        raise ImportError(
            "Need `transformers` package. Run: pip install transformers"
        )

    print(f"[Absorb] Loading {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32,
        device_map=device, trust_remote_code=True,
    )
    model.eval()

    hidden_dim = config.hidden_size
    n_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)

    print(f"[Absorb] Model: H={hidden_dim}, layers={n_layers}, vocab={vocab_size}")

    layers_data = []
    for i in range(n_layers):
        layer_data = _extract_layer(model, i, hidden_dim)
        layers_data.append(layer_data)
        if (i + 1) % 4 == 0:
            print(f"  Extracted layer {i+1}/{n_layers}")

    # Extract embedding matrix (token -> hidden vector lookup table)
    embed_weight = None
    lm_head_weight = None
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if 'embed' in name_lower and 'weight' in name_lower and param.dim() == 2:
            embed_weight = param.data.clone().cpu()
        if 'lm_head' in name_lower and 'weight' in name_lower and param.dim() == 2:
            lm_head_weight = param.data.clone().cpu()

    # Many models tie embed and lm_head (same matrix used for both)
    if lm_head_weight is None and embed_weight is not None:
        lm_head_weight = embed_weight

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Absorb] Extraction complete. {n_layers} layers extracted.")
    if embed_weight is not None:
        print(f"  Embedding: {embed_weight.shape} (vocab x hidden)")
    if lm_head_weight is not None:
        print(f"  LM head: {lm_head_weight.shape}")

    return {
        'model_name': model_name,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'vocab_size': vocab_size,
        'layers': layers_data,
        'embed_weight': embed_weight,
        'lm_head_weight': lm_head_weight,
    }


def _extract_layer(model, layer_idx: int, hidden_dim: int) -> Dict:
    """Extract weight matrices from one transformer layer."""
    layer_data = {
        'attn_qkv': None, 'attn_out': None,
        'mlp_up': None, 'mlp_down': None,
        'ln_weight': None,
        'connectivity': None, 'mass': None,
    }

    # Walk through the model to find layer components
    # This is architecture-agnostic: we search by name patterns
    for name, param in model.named_parameters():
        # Match layer index
        layer_patterns = [
            f'.{layer_idx}.', f'layers.{layer_idx}.',
            f'h.{layer_idx}.', f'blocks.{layer_idx}.',
        ]
        if not any(p in name for p in layer_patterns):
            continue

        pdata = param.data.clone().cpu()
        name_lower = name.lower()

        # Attention weights (skip biases — only want 2D weight matrices)
        if any(k in name_lower for k in ['q_proj', 'k_proj', 'v_proj', 'qkv']):
            if pdata.dim() < 2:
                continue
            if layer_data['attn_qkv'] is None:
                layer_data['attn_qkv'] = pdata
            elif pdata.dim() == layer_data['attn_qkv'].dim():
                layer_data['attn_qkv'] = torch.cat(
                    [layer_data['attn_qkv'], pdata], dim=0
                )
        elif any(k in name_lower for k in ['o_proj', 'attn.out', 'c_proj']):
            layer_data['attn_out'] = pdata
        # MLP weights
        elif any(k in name_lower for k in ['up_proj', 'gate_proj', 'fc1', 'c_fc']):
            if layer_data['mlp_up'] is None:
                layer_data['mlp_up'] = pdata
        elif any(k in name_lower for k in ['down_proj', 'fc2', 'c_proj']) and 'attn' not in name_lower:
            layer_data['mlp_down'] = pdata
        # LayerNorm
        elif 'norm' in name_lower and 'weight' in name_lower and pdata.dim() == 1:
            layer_data['ln_weight'] = pdata

    # Derive connectivity matrix from attention projection
    layer_data['connectivity'] = _derive_connectivity(
        layer_data['attn_qkv'], layer_data['attn_out'],
        layer_data['mlp_up'], layer_data['mlp_down'],
        hidden_dim,
    )

    # Derive mass from LayerNorm
    if layer_data['ln_weight'] is not None:
        ln = layer_data['ln_weight']
        importance = ln.abs()
        layer_data['mass'] = (importance / importance.mean().clamp(min=1e-8)).clamp(0.1, 10.0)

    return layer_data


def _derive_connectivity(
    attn_qkv, attn_out, mlp_up, mlp_down, hidden_dim
) -> Optional[torch.Tensor]:
    """Derive a [H, H] connectivity matrix from transformer weight matrices.

    The key idea: in a transformer, the attention pattern Q*K^T tells you
    "which features attend to which." We approximate this from the
    projection matrices. If we have W_Q and W_K (both [H, head_dim]),
    then W_Q @ W_K^T gives an [H, H] connectivity estimate.
    """
    W = None

    if attn_qkv is not None and attn_out is not None:
        # Use attn_out @ attn_qkv^T as connectivity proxy
        # Both relate to hidden_dim, their product gives feature-to-feature
        try:
            # Ensure compatible dims
            a = attn_out
            b = attn_qkv
            if a.dim() == 2 and b.dim() == 2:
                # a: [H, head_dim_total], b: [qkv_dim, H]
                if a.shape[0] == hidden_dim and b.shape[-1] == hidden_dim:
                    W = (a @ a.T)  # [H, H] self-correlation
                elif a.shape[-1] == hidden_dim:
                    W = (a.T @ a)
        except Exception:
            pass

    if W is None and mlp_down is not None and mlp_up is not None:
        # MLP: down_proj @ up_proj gives [H, H] connectivity
        try:
            d = mlp_down
            u = mlp_up
            if d.dim() == 2 and u.dim() == 2:
                if d.shape[0] == hidden_dim and u.shape[-1] == hidden_dim:
                    W = d @ u
                elif d.shape[-1] == hidden_dim and u.shape[0] == hidden_dim:
                    W = d.T @ u.T
        except Exception:
            pass

    if W is None:
        return None

    # Normalize to valid connectivity
    W = W.abs()
    W.fill_diagonal_(0)
    row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    W = W / row_sums

    return W


def absorb_weights_into_brain(
    brain: SOMICircuitBrain,
    weights: Dict,
    strength: float = 0.7,
    method: str = 'direct',
) -> Dict:
    """
    Absorb extracted transformer weights into a SOMI brain.

    Maps transformer layers to SOMI Parts:
      - If n_layers <= n_parts: one layer per Part
      - If n_layers > n_parts: group layers and average

    Args:
        brain: Target SOMICircuitBrain
        weights: Output of extract_transformer_weights()
        strength: Absorption strength (0 = ignore, 1 = full replacement)
        method: 'direct' (SVD align + transplant) or 'spectral' (eigenmode transfer)

    Returns:
        diagnostics: Absorption metrics
    """
    from ..physics.forces import compute_laplacian
    from ..physics.settling import compute_eigendecomposition

    diagnostics = {}
    n_parts = len(brain.parts)
    n_layers = weights['n_layers']
    src_dim = weights['hidden_dim']
    tgt_dim = brain.config.hidden_dim

    print(f"[Absorb] Mapping {n_layers} layers -> {n_parts} Parts "
          f"(src_dim={src_dim}, tgt_dim={tgt_dim}, method={method})")

    # Group layers into Parts
    layers_per_part = max(1, n_layers // n_parts)

    for pid_int, part in enumerate(brain.parts.values()):
        start = pid_int * layers_per_part
        end = min(start + layers_per_part, n_layers)
        if start >= n_layers:
            break

        # Average connectivity across grouped layers
        connectivity_matrices = []
        mass_vectors = []
        for li in range(start, end):
            layer = weights['layers'][li]
            if layer['connectivity'] is not None:
                connectivity_matrices.append(layer['connectivity'])
            if layer['mass'] is not None:
                mass_vectors.append(layer['mass'])

        if not connectivity_matrices:
            diagnostics[f'part_{pid_int}_skipped'] = 'no_connectivity'
            continue

        # Average connectivity
        avg_W = torch.stack(connectivity_matrices).mean(dim=0)

        # Align dimensions if needed
        if avg_W.shape[0] != tgt_dim:
            avg_W, align_diag = svd_align(avg_W, tgt_dim)
            diagnostics[f'part_{pid_int}_aligned'] = True
            diagnostics[f'part_{pid_int}_energy_preserved'] = (
                align_diag['alignment_energy_preserved']
            )

        if method == 'spectral':
            # Transfer via eigenmodes
            L = compute_laplacian(avg_W)
            evals, evecs, _ = compute_eigendecomposition(L)
            transfer_diag = spectral_mode_transfer(
                part, evals, evecs, strength=strength
            )
            diagnostics.update({
                f'part_{pid_int}_{k}': v
                for k, v in transfer_diag.items()
            })
        else:
            # Direct blend into W_local
            with torch.no_grad():
                avg_W = avg_W.to(part.W_local.device)
                old_W = part.W_local.clone()
                part.W_local.copy_(
                    (1 - strength) * part.W_local + strength * avg_W
                )
                part.W_local.fill_diagonal_(0)
                change = (part.W_local - old_W).abs().mean().item()
                diagnostics[f'part_{pid_int}_W_change'] = change

        # Absorb mass
        if mass_vectors:
            avg_mass = torch.stack(mass_vectors).mean(dim=0)
            if avg_mass.shape[0] != tgt_dim:
                # Interpolate mass to target dim
                avg_mass = torch.nn.functional.interpolate(
                    avg_mass.unsqueeze(0).unsqueeze(0),
                    size=tgt_dim, mode='linear', align_corners=False,
                ).squeeze()
            with torch.no_grad():
                avg_mass = avg_mass.to(part.mass.device)
                part.mass.copy_(
                    (1 - strength) * part.mass + strength * avg_mass.clamp(0.1, 10.0)
                )
            diagnostics[f'part_{pid_int}_mass_absorbed'] = True

        diagnostics[f'part_{pid_int}_layers'] = list(range(start, end))

    # Update eigen decomposition in all parts (connectivity changed)
    for part in brain.parts.values():
        part._update_eigen()

    # === ABSORB VOCABULARY (embedding + LM head) ===
    # This is the part that transfers actual word knowledge, not just wiring.
    # The embedding maps token IDs -> vectors. The LM head maps vectors -> token probs.
    # We SVD-project from source_dim to target_dim to bridge the size gap.
    vocab_absorbed = _absorb_vocabulary(
        brain, weights, src_dim, tgt_dim, strength
    )
    diagnostics.update(vocab_absorbed)

    # Recalibrate physics for absorbed state
    brain.recalibrate_config()

    diagnostics['method'] = method
    diagnostics['strength'] = strength
    diagnostics['source_model'] = weights['model_name']
    diagnostics['source_hidden_dim'] = src_dim
    diagnostics['target_hidden_dim'] = tgt_dim
    diagnostics['n_layers_absorbed'] = min(n_layers, n_parts * layers_per_part)

    print(f"[Absorb] Done. Absorbed {diagnostics['n_layers_absorbed']} layers.")
    return diagnostics


def _absorb_vocabulary(
    brain: SOMICircuitBrain,
    weights: Dict,
    src_dim: int,
    tgt_dim: int,
    strength: float,
) -> Dict:
    """
    Absorb vocabulary knowledge into X-Encoder and Y-Decoder.

    The transformer's embedding matrix is [vocab_size, source_hidden_dim].
    SOMI's Y-Decoder is [output_dim, target_hidden_dim].

    We use SVD to find the best projection from source_dim to target_dim,
    then install the projected embedding into SOMI's encoder/decoder.

    This transfers the actual word knowledge: which words are similar,
    what patterns connect words, etc. — not just the wiring between neurons.
    """
    diagnostics = {}

    embed = weights.get('embed_weight')
    lm_head = weights.get('lm_head_weight')

    if embed is None:
        diagnostics['vocab_absorbed'] = False
        diagnostics['vocab_reason'] = 'no_embedding_found'
        return diagnostics

    # embed shape: [vocab_size, source_hidden_dim]
    vocab_size_src = embed.shape[0]
    output_dim = brain.y_decoder.out_features

    print(f"[Absorb] Vocabulary: source vocab={vocab_size_src}, "
          f"SOMI output_dim={output_dim}, src_H={src_dim}, tgt_H={tgt_dim}")

    # Step 1: Find the best linear projection from src_dim -> tgt_dim
    # using SVD of the embedding matrix.
    # embed = U @ S @ V^T, where V^T rows are the principal directions.
    # We keep the top tgt_dim directions -> projection matrix P = V[:tgt_dim]^T
    U, S, Vh = torch.linalg.svd(embed.float(), full_matrices=False)
    K = min(tgt_dim, S.shape[0], Vh.shape[0])

    # Projection: [src_dim] -> [tgt_dim] via top-K right singular vectors
    # P: [K, src_dim] — projects src_dim down to K dimensions
    P = Vh[:K, :]  # [K, src_dim]

    energy_total = S.pow(2).sum().item()
    energy_kept = S[:K].pow(2).sum().item()
    diagnostics['vocab_svd_energy_preserved'] = energy_kept / max(energy_total, 1e-8)
    diagnostics['vocab_svd_K'] = K
    print(f"  SVD projection: {src_dim} -> {K} dims, "
          f"energy preserved: {diagnostics['vocab_svd_energy_preserved']:.1%}")

    # Step 2: Project embedding into SOMI's hidden space
    # projected_embed: [vocab_size, K]
    projected_embed = embed.float() @ P.T  # [vocab, K]

    # Step 3: Install into Y-Decoder
    # Y-Decoder.weight: [output_dim, hidden_dim]
    # We need to handle vocab size mismatch: use min of both
    with torch.no_grad():
        decoder_w = brain.y_decoder.weight.data
        n_vocab = min(output_dim, vocab_size_src)
        n_hidden = min(K, decoder_w.shape[1])

        # Normalize projected embeddings to match SOMI's scale
        proj_slice = projected_embed[:n_vocab, :n_hidden]
        scale = decoder_w[:n_vocab, :n_hidden].std() / proj_slice.std().clamp(min=1e-8)
        proj_scaled = proj_slice * scale

        decoder_w[:n_vocab, :n_hidden] = (
            (1 - strength) * decoder_w[:n_vocab, :n_hidden]
            + strength * proj_scaled.to(decoder_w.device)
        )
        brain.y_decoder.weight.data = decoder_w
        diagnostics['y_decoder_absorbed'] = True
        diagnostics['y_decoder_vocab_covered'] = n_vocab
        print(f"  Y-Decoder: absorbed {n_vocab} token embeddings")

    # Step 4: Install into X-Encoder
    # X-Encoder.weight: [hidden_dim, input_dim]
    # The embedding's projection gives us a good init for how to
    # map input features to SOMI's hidden space.
    # We use the projection matrix P itself as X-Encoder init
    with torch.no_grad():
        encoder_w = brain.x_encoder.weight.data
        n_out = min(K, encoder_w.shape[0])
        n_in = min(encoder_w.shape[1], K)

        # P transposed: [src_dim, K] -> we need [tgt_dim, input_dim]
        # Use the singular values to weight the projection
        P_weighted = (S[:K].unsqueeze(1) * P[:K, :])  # [K, src_dim]

        # Take the submatrix that fits
        proj_enc = P_weighted[:n_out, :n_in]
        enc_scale = encoder_w[:n_out, :n_in].std() / proj_enc.std().clamp(min=1e-8)
        proj_enc_scaled = proj_enc * enc_scale

        encoder_w[:n_out, :n_in] = (
            (1 - strength) * encoder_w[:n_out, :n_in]
            + strength * proj_enc_scaled.to(encoder_w.device)
        )
        brain.x_encoder.weight.data = encoder_w
        diagnostics['x_encoder_absorbed'] = True
        print(f"  X-Encoder: absorbed projection matrix")

    diagnostics['vocab_absorbed'] = True
    return diagnostics


def absorb_from_huggingface(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    somi_hidden: int = 128,
    somi_parts: int = 4,
    output_dim: int = 32000,
    strength: float = 0.7,
    method: str = 'direct',
    max_layers: Optional[int] = None,
    device: str = 'cpu',
) -> Tuple[SOMICircuitBrain, Dict]:
    """
    One-line absorption: load a HuggingFace model, create SOMI, absorb.

    Args:
        model_name: HuggingFace model name
        somi_hidden: SOMI hidden dimension
        somi_parts: Number of SOMI Parts
        output_dim: Output vocabulary size (use model's vocab_size if known)
        strength: Absorption strength
        method: 'direct' or 'spectral'
        max_layers: Max layers to extract (None = all)
        device: Device

    Returns:
        brain: SOMI brain with absorbed knowledge
        diagnostics: Full absorption report
    """
    # Extract
    weights = extract_transformer_weights(model_name, device, max_layers)

    # Use model's actual vocab size
    if output_dim is None:
        output_dim = weights['vocab_size']

    # Create SOMI brain
    config = SOMIBrainConfig.auto(somi_hidden, somi_parts)
    brain = SOMICircuitBrain(
        config,
        input_dim=somi_hidden,
        output_dim=output_dim,
    )

    # Absorb
    diag = absorb_weights_into_brain(brain, weights, strength, method)

    # Integrity check
    from .integrity import check_integrity
    integrity = check_integrity(brain, verbose=True)
    diag['post_absorb_healthy'] = integrity['overall_healthy']

    return brain, diag
