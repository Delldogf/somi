# What’s in Theory That’s Much More Advanced Than Code

A single list of SOMI theory (in SOMI_Research / SOMI_3_0 / SOMI_4) that goes **well beyond** what’s implemented in the clean `somi` package. For absorption-only gaps, see [ABSORPTION_THEORY_VS_IMPLEMENTED.md](ABSORPTION_THEORY_VS_IMPLEMENTED.md).

---

## 1. The 5 Levels — Only Level 1 Is in Code

**Theory:** `24_THE_5_LEVELS_COMPLETE_REFERENCE.md`  
SOMI is one theory at five resolutions. The **code only implements Level 1** (discrete graph: field equation, geometry equation, 9 forces, settling, W updates).

| Level | Name | What theory has | In code? |
|-------|------|-----------------|----------|
| **1** | Discrete graph | N nodes, φ, W, 9 forces, geometry eqn, Hamiltonian | ✅ Yes |
| **2** | **Continuum limit** | N→∞: φ(x,t), κ(x,y), PDEs, Ricci-like flow, ρ·κ=1/α₁, spectral speedup | ❌ No |
| **3** | **Spacetime** | Time as geometry; Einstein equations; c_info, CFL, geodesics, dark energy/matter analogies | ❌ No |
| **4** | **Gauge theory & topology** | Multi-component φ^I, white matter as gauge connection A, Wilson loops, curvature F_ab | ❌ No |
| **5** | **Path integral / quantum** | Sum over histories, ensemble methods, connection to quantum gravity | ❌ No |

**Gap:** No continuum solvers, no spacetime metric, no gauge/topology layer, no path-integral layer. Theory also describes **top-down upgrades** (e.g. Level 2 → spectral speedup, scaling laws for α₁; Level 4 → Wilson loops for absorption) — none of that is in code.

---

## 2. Continuum Limit (Level 2) — Full PDE / Spectral World

**Theory:** `06_THE_CONTINUUM_LIMIT.md`  
Discrete SOMI becomes continuous: graph Laplacian → Laplace–Beltrami, sums → integrals, W → conductivity kernel κ(x,y). Field equation → damped Klein–Gordon; geometry → Ricci-like flow; mass density ρ, constraints (e.g. ρ·κ = 1/α₁). Neuromodulators as spatial fields; structural plasticity as domain deformation.

**What code actually has:** Dynamics are discrete only. We have **Level 2 diagnostics and one constraint**: `diagnostics/continuum.py` (mass-conductivity duality, spectral entropy, scaling-ratio checks), `physics/geometry.mass_conductivity_constraint`, and α₁ scaling by `hidden_dim` in `config.__post_init__`. No continuum PDE solvers or Ricci flow.

---

## 3. Gauge Theory and Topology (Level 4)

**Theory:** `07_GAUGE_THEORY_AND_TOPOLOGY.md`, and Level 4 in `24_THE_5_LEVELS`.  
White matter as **gauge connection** A; curvature F; Wilson loops (products of W around closed paths) as gauge-invariant observables; topology (Chern numbers, etc.). Knowledge = topology, not coordinates.

**What code actually has:** White matter = low-rank tracts (`brain/white_matter.py`: `WhiteMatterTract`, `compute_wilson_loop`, `get_all_curvatures`). **Wilson loops and curvature are in code** — `diagnostics/gauge.py`, `visualization/gauge_plots.py`. Absorption pipeline does not yet use Wilson loops for fingerprinting/integrity.

---

## 4. SOMI = General Relativity (Level 3)

**Theory:** `26_SOMI_EXPLAINS_GENERAL_RELATIVITY.md`  
Level 3 is literally Einstein’s equations for the information metric; c_info as speed limit; gravitational time dilation ↔ variable settling; dark energy/matter/black-hole analogies.

**What code actually has:** No spacetime *dynamics*. We do have **CFL and c_info**: `physics/hamiltonian.compute_cfl_condition`, `diagnostics/spacetime.py` (CFL, c_info map, dark energy/matter metrics). Geodesic routing not implemented.

---

## 5. Dual Learning & JEPA — Theory Is Richer Than Wiring

**Theory:** `04_THE_DUAL_LEARNING.md`, `03_THE_JEPA_CONNECTION.md`  
Macro (backprop on X-Encoder, Y-Encoder/Decoder, white matter) vs micro (W_local updated only by stress/STDP, no backprop). Stress = JEPA loss. Straight-through estimator so macro gets gradients while W stays Lyapunov-stable. Full SOMI-JEPA architecture with “always on” predictor; Y-Encoder frozen/EMA; selective Y-Decoder.

**What code actually has:** `training/dual_learning.py` (`DualLearningTrainer`), `training/jepa.py` (`JEPALoss`, `YEncoder`), `training/test_time.py` (`TestTimeLearner`). **Now:** `DualLearningTrainer` wires the full pipeline: X-Encoder → SOMI → Y-Encoder with stress = JEPA loss. Phase 1/2/3 support, stress-based selective updates, mass-guided fine-tuning, stress-JEPA correlation metric. **Remaining gap:** Selective Y-Decoder (phase 3 freezes Y-Encoder but doesn’t yet do partial decoder updates).

---

## 6. Training — Many Physics-Derived Ideas Not in Code

**Theory:** `18_SOMI_TRAINING_COMPRESSION_AIMO3.md` (Training section)  
Self-calibrating hyperparameters; **stress-based selective updates** (only update when/where stress is high); **stress-based data selection**; settling replacing depth; **mass-guided fine-tuning**; **physics-derived curriculum**. Cheaper, faster training by using SOMI physics to decide what/when to update.

**What code actually has:** Dual learning and JEPA classes (see §5). **Now:** `DualLearningTrainer` has `selective_threshold` for stress-based selective updates, `mass_guided=True` for mass-guided FT, and `StressDataSampler` for stress-based data selection. `stress_jepa_correlation` metric logged every step. **Remaining gap:** Full physics-derived curriculum (automatic topic ordering). **We do have** self-calibration at runtime: `physics/calibration.py` (beta, n_settle, eta, mass from W/state) and `config.auto()` (zero-hyperparameter config from `hidden_dim`, `n_parts`) — but no loop that *tunes* regime constants from physics.

---

## 7. Compression — Theory’s Full Menu vs Code Stubs

**Theory:** `18_SOMI_TRAINING_COMPRESSION_AIMO3.md` (Compression section)  
Mass-guided mixed precision; stress-guided pruning; **topological compression quality metric** (Wilson loops, eigenspectrum); eigenspectrum-guided rank reduction; “dark matter” parameters (high ρ, low κ); layer-wise adaptive quantization; **physics-aware post-quantization calibration**.

**What code actually has:** `compression/mass_precision.py` (`mass_guided_quantization`), `compression/stress_pruning.py` (`stress_guided_pruning`), `compression/spectral_rank.py` (`spectral_rank_selection`), `compression/topological_quality.py` (`check_topological_quality`), `compression/adaptive.py` (`adaptive_compress`). **Now:** `AutoCompress` monitors stress/spectral-ratio EMA and triggers `adaptive_compress` automatically (spectral rank → stress pruning → mass quantization → topological quality check with rollback). After compression, `recalibrate_config()` re-derives physics. **Remaining gap:** physics-aware post-quant calibration, dark-matter parameter identification.

---

## 8. Diagnostics — You Have (Almost) All of It

**Theory:** `05_THE_DIAGNOSTICS.md`  
Per-part physics (stress, W, mass, frequency, φ, Hamiltonian, Dale’s law, eigenspectrum); neuromodulator diagnostics; **10 neuroscience tests**; **11 pathology modes**; **circuit-level** (per-circuit, white matter, shared-part pressure); **JEPA-specific** (e.g. stress–JEPA correlation ≈ 1); **full visualization suite** (e.g. JEPA embedding t-SNE/PCA, stress–JEPA correlation plots).

**What code actually has:** `diagnostics/dashboard.py` runs: `standard.py` (L1), `continuum.py` (L2), `spacetime.py` (L3), `gauge.py` (L4), `quantum.py` (L5), `neuroscience.py` (13), `circuit.py` (11), `pathology.py` (11) + self-healing, `singularity.py`. Viz: `visualization/brain_scan.py`, `circuit_plots.py`, `gauge_plots.py`, etc. **Gap:** Explicit stress–JEPA correlation scalar (theory: ≈1.0); some theory plots may be missing.

---

## 9. Radical Architecture Surgery

**Theory:** `20_RADICAL_ARCHITECTURE_SURGERY.md`  
Surgery without retraining: **recurrent conversion** (e.g. 36 layers → 4 shared blocks); **MoE → task-specific dense** (116B → 5–8B); **turn model into literal SOMI**; **layer folding** (run any layer multiple times); **dynamic architecture** (restructure per input); **extract topology, discard weights**; **fuse attention and MoE**; **split into autonomous modules** that self-coordinate. All justified by “knowledge = topology.”

**What code actually has:** `brain/circuit_brain.py` (Circuit Brain, `grow_brain`), `brain/part.py` (`grow()`). **Largely superseded by absorption:** `somi/absorption/from_huggingface.py` extracts weights from any HuggingFace transformer and absorbs them into SOMI via SVD projection + spectral mode transfer + vocabulary absorption. This achieves "turn model into literal SOMI" without explicit surgery. **Remaining gap:** Layer folding, dynamic restructure-per-input (not needed for "one model forever").

---

## 10. AIMO 3 Pipeline (Consistency, Settle Check, Fusion)

**Theory:** `18_SOMI_TRAINING_COMPRESSION_AIMO3.md` (Part 1)  
**Consistency scoring** from hidden states (inter-layer change = stress); **post-generation settle check**; **confidence–consistency fusion** for answer selection. Full AIMO 3 pipeline using SOMI during/after generation.

**What code actually has:** **DONE.** `somi/aimo.py` implements `consistency_score()`, `settle_check()`, and `confidence_consistency_fusion()`. All three formulas from the theory are in code.

---

## 11. Other Theory-Only (or Mostly) Topics

- **General Theory of Intelligence (GTI)** (`17_GENERAL_THEORY_OF_INTELLIGENCE.md`) — broad framing; no direct code.
- **SOMI explains transformers** (`15_SOMI_EXPLAINS_TRANSFORMERS.md`, `16_TRANSFORMERS_FROM_FIRST_PRINCIPLES.md`) — interpretative; no new modules.
- **Universality, cosmology, quantum gravity, markets** (`09_SOMI_UNIVERSALITY.md`, `13_SOMI_COSMOLOGY.md`, `12_SOMI_AND_QUANTUM_GRAVITY.md`, `08_GAUGE_THEORY_OF_MARKETS.md`, `10_SOMI_MARKET_MODEL.md`) — deep theory; no implementation.
- **SOMI for GPT-OSS-120B / VibeThinker** (`19_SOMI_FOR_GPT_OSS_120B.md`, `23_SOMI_VIBETHINKER.md`) — application blueprints; not in clean repo as runnable pipelines.
- **Proving SOMI and implementation** (`14_PROVING_SOMI_AND_IMPLEMENTATION.md`) — validation roadmap; not automated in code.

---

## Summary Table

| Area | Theory has | Code has |
|------|------------|----------|
| **Levels 2–5** | Continuum, spacetime, gauge, path integral dynamics | L1 dynamics only; L2–L5 diagnostics (continuum/spacetime/gauge/quantum.py), CFL, c_info, Wilson loops |
| **Absorption** | 6-level hierarchy; full fingerprint (stress/eigen/Wilson); knowledge diff; Universal Absorber; black-box | **DONE:** Stress-guided transplant, spectral mode transfer, full fingerprint, knowledge_diff, UniversalAbsorber with interference prevention. Gap: black-box API absorption |
| **Dual learning / JEPA** | Full pipeline; stress = JEPA; phases | **DONE:** Full pipeline wired; Phase 1/2/3; stress = JEPA verified; stress-JEPA correlation metric |
| **Training** | Stress-based selective update, data selection, mass-guided FT, physics curriculum | **DONE:** Selective updates, `StressDataSampler`, mass-guided FT. Gap: full physics curriculum |
| **Compression** | Full pipelines (topological metric, post-quant calibration, dark matter, etc.) | **DONE:** `AutoCompress` triggers from stress/spectrum; `adaptive_compress` with quality rollback; building blocks wired into automatic loop |
| **Diagnostics** | Circuit-level, stress–JEPA correlation, full viz suite | All implemented (L1–5, neuro 13, circuit 11, pathology 11, singularity, self-heal); optional: stress–JEPA scalar, full viz |
| **Radical surgery** | Recurrent, MoE→dense, layer folding, dynamic arch, topology extraction | Largely superseded by absorption pipeline (`from_huggingface.py`); layer folding/dynamic arch not implemented |
| **AIMO 3** | Consistency score, settle check, fusion pipeline | **DONE:** `consistency_score`, `settle_check`, `confidence_consistency_fusion` in `somi.aimo` |
| **Full absorption + vocab** | Extract W, mass, embedding, LM head from any model; SVD project; install | **DONE:** `from_huggingface.py` — wiring, mass, and vocabulary all absorbed. See `ABSORB_EVERYTHING.md` |
| **Continuum / GR / Gauge** | Full math (PDEs, Ricci, Einstein) as dynamics | No solvers; we have CFL, c_info, Wilson loops, mass-conductivity as diagnostics |
| **Zero hyperparameters** | All params from action + L5 constraints | **DONE:** config.auto() uses full L1-L5 cascade (action_derived.py); calibration.py for runtime. Zero free params. |

Nearly all high-priority theory-to-code items are **DONE**: ~~stress-guided absorption~~, ~~richer fingerprint~~, ~~stress-based selective training~~, ~~mass-guided fine-tuning~~, ~~stress-JEPA correlation~~, ~~full dual-learning pipeline~~, ~~continuum-inspired spectral speedup~~, ~~AIMO 3 pipeline~~, ~~full HuggingFace absorption (wiring + vocab)~~, ~~radical surgery (superseded by absorption)~~. Remaining: **full physics curriculum**, **black-box API absorption**, **post-quantization calibration**, **dark matter parameter detection**. See [ABSORB_EVERYTHING.md](ABSORB_EVERYTHING.md) for the first-principles derivation of the "absorb everything" vision.
