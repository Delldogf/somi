"""
SOMI 4.0 Absorption — Knowledge Transfer from Pretrained Models
=================================================================

Instead of training from scratch, SOMI can absorb knowledge from existing
models (like LLaMA, Qwen, etc.) through surgical weight transplantation.

Core equation: delta_W = W_specialist - W_base
  The "connectivity delta" captures what a specialist model learned
  relative to its base. We transplant this delta into SOMI's W_local.

7 absorption modules:
  1. pretrained_init — Extract W, mass, potential from pretrained transformer
  2. transplant — Core delta transplant: W_specialist - W_base
  3. alignment — Cross-size alignment (different architectures)
  4. distillation — Output-only knowledge distillation
  5. multi_model — Absorb from multiple specialists
  6. fingerprint — Verify what knowledge was absorbed
  7. integrity — Check transplant didn't break physics

Theory vs implemented: see docs/ABSORPTION_THEORY_VS_IMPLEMENTED.md
"""

from .pretrained_init import init_from_pretrained
from .transplant import (
    compute_delta, transplant_knowledge,
    stress_guided_transplant, spectral_mode_transfer,
)
from .fingerprint import (
    compute_fingerprint, compare_fingerprints, knowledge_diff,
)
from .integrity import check_integrity
from .alignment import svd_align, cka_similarity
from .multi_model import absorb_multiple
from .distillation import OutputDistiller
from .universal_absorber import UniversalAbsorber
from .from_huggingface import (
    absorb_from_huggingface,
    extract_transformer_weights,
    absorb_weights_into_brain,
)
from .lossless import (
    lossless_absorb_all,
    lossless_spectral_extract,
    procrustes_align,
    KalmanSpectralFuser,
    install_spectrum_into_part,
    fullrank_vocab_transfer,
)
from .spectral_analysis import (
    marchenko_pastur_threshold,
    analyze_full_model,
)
