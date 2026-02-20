"""
SOMI 4.0 Compression — Physics-Guided Model Compression
==========================================================

SOMI's physics provides natural guidance for compression:
- Heavy features (high mass) are important — keep high precision
- Light features (low mass) are less important — can use lower precision
- Low-stress connections are unneeded — can be pruned
- Low-energy eigenmodes contribute little — can be truncated

6 modules:
1. mass_precision — Mass-guided mixed precision (heavy=fp32, light=fp8)
2. stress_pruning — Remove low-stress connections
3. topological_quality — Measure quality of compressed model
4. spectral_rank — Eigenmode-based rank selection
5. adaptive — Unified adaptive compression pipeline
6. auto_compress — Physics-triggered automatic compression monitor
"""

from .mass_precision import mass_guided_quantization
from .stress_pruning import stress_guided_pruning
from .spectral_rank import spectral_rank_selection
from .auto_compress import AutoCompress
