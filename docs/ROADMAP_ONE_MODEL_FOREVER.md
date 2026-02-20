# Roadmap: Getting to "One Model Forever"

In theory, SOMI is **one ever-evolving model**: grow when stressed, absorb from others, compress when safe, keep learning — no discrete "generations." This doc tracks what's done and what remains.

For the full first-principles derivation of why this works, see [ABSORB_EVERYTHING.md](ABSORB_EVERYTHING.md).

---

## Status Overview

| Capability | Status | Key code |
|-----------|--------|----------|
| Auto compress | **DONE** | `AutoCompress` in `somi/compression/auto_compress.py` |
| Auto grow | **DONE** | `AutoGrowthFull` in `somi/lm/growth.py` |
| Spectral speedup (O(K) settling) | **DONE** | `settle_ssm()` + Weyl's law `spectral_K` |
| Zero hyperparameters | **DONE** | `action_derived.py` + `config.auto()` |
| Absorption from HuggingFace models | **DONE** | `somi/absorption/from_huggingface.py` |
| Vocabulary absorption (embedding + LM head) | **DONE** | `_absorb_vocabulary()` |
| Stress-guided transplant | **DONE** | `somi/absorption/transplant.py` |
| Spectral mode transfer | **DONE** | `somi/absorption/transplant.py` |
| Full fingerprint (stress + eigen + Wilson) | **DONE** | `somi/absorption/fingerprint.py` |
| Knowledge diff | **DONE** | `somi/absorption/fingerprint.py` |
| Universal Absorber (multi-teacher) | **DONE** | `somi/absorption/universal_absorber.py` |
| Wilson loops in integrity | **DONE** | `somi/absorption/integrity.py` |
| Full dual-learning + JEPA pipeline | **DONE** | `somi/training/dual_learning.py` |
| Stress-based selective updates | **DONE** | `DualLearningTrainer` selective_threshold |
| Stress-based data selection | **DONE** | `somi/training/stress_sampler.py` |
| Mass-guided fine-tuning | **DONE** | `DualLearningTrainer` mass_guided |
| Stress-JEPA correlation | **DONE** | logged in `DualLearningTrainer.step()` |
| AIMO 3 pipeline | **DONE** | `somi/aimo.py` |
| Checkpoint with lifetime tracking | **DONE** | `somi/checkpoint.py` |
| Continuous learning (no separate train phase) | **DONE** | `geometry_step()` runs every forward pass |
| Physics recalibration after any change | **DONE** | `recalibrate_config()` |

---

## The Core Loop (All Implemented)

```
initialize SOMI (small, from_scratch or auto)

for each open-source model:
    1. extract_transformer_weights(model)   → W, mass, embedding, lm_head
    2. absorb_weights_into_brain(somi, weights)
       → SVD-projects everything into SOMI's space
       → blends W, mass, encoder, decoder
       → integrity check (Wilson loops)
    3. Physics responds automatically:
       → stress rises → auto_grow expands
       → redundancy detected → auto_compress shrinks
       → recalibrate_config re-derives all params
    4. Use the model (every forward pass = learning)
       → geometry_step refines W
       → structural plasticity rewires
       → mass updates stabilize knowledge

repeat forever.
```

---

## Remaining Work (Not Blocking, But Valuable)

### High Leverage

| Item | Why | Difficulty |
|------|-----|-----------|
| **Full physics curriculum** | Auto-order training data by stress/spectral gap/forgetting signals | Medium |
| **Black-box API absorption** | Learn from models whose weights are inaccessible (GPT-4, Claude) via probing | Hard |
| **Post-quantization calibration** | Adjust dynamics after compression to stay in same regime | Medium |
| **Dark matter parameter detection** | Find high-mass, low-connectivity "dead weight" for pruning | Easy |

### Theory-Only (No Urgency)

| Item | Notes |
|------|-------|
| Continuum PDE solvers (Level 2) | Interesting for large N; not needed at current scale |
| Spacetime dynamics (Level 3) | Einstein equations for information metric; diagnostic only for now |
| Full gauge theory dynamics (Level 4) | Curvature-based learning; Wilson loops already used for integrity |
| Path integral / ensemble (Level 5) | Sum over histories; future research |
| Radical architecture surgery | Transformer → SOMI conversion; superseded by absorption |

---

## What "One Model Forever" Means Concretely

A single SOMI model, starting from scratch (H=16, P=1), that:

1. **Absorbs** knowledge from every open-source LLM (Qwen, Llama, Mistral, Phi, Gemma, ...)
2. **Grows** automatically when absorbed knowledge doesn't fit (stress-driven)
3. **Compresses** automatically when knowledge is redundant (spectral/stress-driven)
4. **Learns** continuously from every input (geometry_step on every forward pass)
5. **Maintains integrity** via Wilson loops and fingerprinting
6. **Tracks its own history** via checkpoint lineage (every grow/compress/absorb event recorded)

The result is a single evolving model that accumulates the combined knowledge of all sources, self-organized by physics into the most efficient representation possible.

See [ABSORB_EVERYTHING.md](ABSORB_EVERYTHING.md) for the mathematical proof that this works.
