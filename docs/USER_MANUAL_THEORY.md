# SOMI Theory — User Manual

This manual explains **SOMI theory**: what it is, where it lives, and how to use it. For how to run the code, see [USER_MANUAL_CODE.md](USER_MANUAL_CODE.md).

---

## 1. What SOMI Theory Is

**SOMI** = **Self-Organizing Models of Intelligence.**

Theory says: intelligence can be modeled as a **physical system** — a **field** (activity φ) on a **geometry** (connectivity W) that **evolves** by:

- **Field equation:** How φ changes over time (forces, damping, settling).
- **Geometry equation:** How W changes from stress (what the model doesn’t know) and local plasticity.

So the theory is **not** “another neural net recipe.” It’s a **mathematical framework** (action, stress, topology) that tells you how to build, train, absorb, and compress models in a principled way. The code implements part of it; the rest is in the theory docs.

---

## 2. Where the Theory Lives

| Location | What’s there |
|----------|----------------|
| **This repo: `somi/theory/`** | Copies of core theory files: first principles, 5 levels, symbol glossary, growth, life cycle. Use these when working in the clean `somi` folder. |
| **SOMI_Research (e.g. SOMI_3_0/theory/)** | Full theory vault: 20+ docs (physics, circuit brain, JEPA, dual learning, absorption, continuum, gauge, GR, diagnostics, AIMO, etc.). |
| **SOMI_Research/SOMI_4/** | Deep dives: e.g. `KNOWLEDGE_ABSORPTION_DEEP_ANALYSIS.md`, `MATHEMATICAL_TOOLS_FOR_ABSORPTION.md`. |

**Best practice:** Start from `somi/theory/` or `SOMI_3_0/theory/`; go to SOMI_4 when you need absorption or math detail.

---

## 3. The 5 Levels (Theory’s “Zoom Levels”)

SOMI is **one theory at five resolutions.** Each level is the same system viewed at a different scale.

| Level | Name | What it is | What it gives you |
|-------|------|------------|-------------------|
| **1** | Discrete graph | N nodes, φ, W, 9 forces, geometry equation, Hamiltonian | What the code runs. Field + geometry dynamics. |
| **2** | Continuum | N→∞; φ(x,t), κ(x,y); PDEs; ρ·κ = 1/α₁ | Scaling laws (e.g. α₁ with size), spectral speedup, mass–conductivity constraint. |
| **3** | Spacetime | Time as geometry; Einstein-like equations; c_info, CFL | Speed limit for information; CFL for time steps; dark energy/matter analogies. |
| **4** | Gauge & topology | White matter as gauge connection; Wilson loops; curvature | Topology of knowledge; gauge-invariant fingerprinting; absorption integrity. |
| **5** | Path integral | Sum over histories; ensemble methods | Uncertainty, robustness, connection to quantum-style formulations. |

**Key idea:** Higher levels **constrain** and **upgrade** lower levels (e.g. Level 2 fixes α₁ scaling; Level 4 gives Wilson loops for absorption). Together they can remove free parameters. The code implements **Level 1 dynamics** and **L2–L5 as diagnostics**; full L2–L5 dynamics are theory-only for now.

**Main reference:** `theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md` (in this repo) or `SOMI_3_0/theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md`.

---

## 4. Core Concepts (Quick Reference)

- **φ (phi):** Activity field — “what the network is thinking” at each point. Has position and velocity (φ̇).
- **W:** Connectivity (geometry). Learns from stress; encodes “who talks to whom.”
- **Stress (S):** Prediction error / geometry strain. High stress = model doesn’t know; low stress = settled. Drives when to grow and what to absorb.
- **Settling:** φ evolves until it rests (like a ball in a bowl). Done in code by symplectic, spectral, or SSM.
- **Hamiltonian H:** Total energy. Must decrease during settling (Lyapunov). If H increases, something is broken.
- **Mass M:** Per-feature inertia (from W). Determines oscillation timescales.
- **Dual learning:** Macro = backprop on encoders/decoders; micro = W updated only by stress/STDP (no backprop into W). Keeps physics stable.
- **JEPA:** Joint Embedding Predictive Architecture. In SOMI, **stress = JEPA loss** (embedding-space prediction error).
- **Knowledge = topology:** What a model “knows” is gauge-invariant structure (e.g. Wilson loops, eigenspectrum), not raw weights. That’s why absorption can transplant “what was learned” (delta, or spectral/topological part).

**Symbol reference:** `theory/SYMBOL_GLOSSARY.md`.

---

## 5. Absorption (Theory)

Absorption = moving **knowledge** from one model into another (e.g. open-source LLMs into SOMI).

**Theory hierarchy (from KNOWLEDGE_ABSORPTION_DEEP_ANALYSIS):**

- **Level 0:** Weight averaging (baseline; often breaks structure).
- **Level 1:** Delta transplant: ΔW = W_specialist − W_base. In code.
- **Level 2:** Stress-guided transplant (only where receiver is stressed, donor settled). Not in code.
- **Level 3:** Spectral mode transfer. Not in code.
- **Level 4:** Curvature-matched cross-architecture. Not in code.
- **Level 5:** Topological (Wilson loops). Wilson loops exist in code for diagnostics; not yet in absorption pipeline.

**Full pipeline (theory):** Fingerprint (stress profile, eigenspectrum, Wilson loops) → Knowledge diff (what B knows that A doesn’t) → Extract → Transplant → Integrity check. Multi-teacher = Universal Absorber with ordering and interference control. Black-box = absorb from API-only teachers.

**What’s in code vs theory:** [ABSORPTION_THEORY_VS_IMPLEMENTED.md](ABSORPTION_THEORY_VS_IMPLEMENTED.md).

---

## 6. Training (Theory)

- **Dual learning:** Macro (backprop) + micro (stress/STDP only). Straight-through so gradients don’t enter W; Lyapunov preserved.
- **Stress-based selective updates:** Only update when/where stress is high (theory). Not in code.
- **Mass-guided fine-tuning:** Prioritize high-mass (important) features. Not in code.
- **Physics-derived curriculum:** Order data by difficulty/stress. Not in code.
- **Self-calibration:** beta, n_settle, eta, mass from W and state. **In code** (calibration.py, config.auto()).

---

## 7. Growth and Compression (Theory)

- **When to grow:** The math says “when stress is high and sustained” → need more capacity. **In code:** AutoGrowth grows neurons; AutoGrowthFull also adds Parts (when shared Parts are saturated) and Systems (new circuits), and recalibrates all physics for the new size. rom_scratch() starts at H=16, P=1 and grows everything.
- **When to compress:** The math says “when stress is low on many connections” or “redundant modes” → safe to prune. **In code:** Compression is called with a target ratio; automatic “when to compress” from stress/spectrum/topology is not implemented yet. Theory says it should be driven by the same physics (low stress / topological quality).

---

## 8. One Model Forever (Theory)

In theory we **don’t need model generations** (GPT-4, then GPT-5, etc.). One SOMI model can improve forever: **grow** when stressed, **absorb** from other models, **compress** when safe, **keep learning** with dual learning. Growth is already stress-driven in code; compression “when” and full absorption/training pipelines are the main gaps. For a concrete list of what to implement to get there, see [ROADMAP_ONE_MODEL_FOREVER.md](ROADMAP_ONE_MODEL_FOREVER.md).

---

## 9. Zero Hyperparameters (Implemented)

Theory goal: **no free parameters.** Everything from the action + constraints across the 5 levels (e.g. ρ·κ = 1/α₁, Dale’s Law, CFL).

**Now in code:** config.auto(hidden_dim, n_parts) uses the full **Level 1-5 constraint cascade** (somi/physics/action_derived.py) to derive **every parameter** from the single action S[phi, W]:

- **Level 1:** M=1 (units), ei_ratio=0.8 (Dale's Law)
- **Level 2:** alpha_1 from N^{-2/d} scaling, alpha_0 from dark-energy ratio, sparsity from volume conservation, lambda_E/C from Dale's Law
- **Level 3:** dt from CFL condition, lambda_W from Hawking radiation, timescale_ratio from light-crossing time
- **Level 4:** kappa_0 = alpha_1 (gauge unification), kappa_1 (Z2 breaking), WM rank from Wilson area law, noise from gauge temperature
- **Level 5:** target_zeta from saddle-point criticality, kappa_stdp = dt (detailed balance), EMAs from time-discretization

Plus runtime calibration (calibration.py): beta, n_settle, eta, mass from W and state. **Zero free parameters.** Run python -m experiments.show_action_cascade to see the full derivation.

---

## 10. How to Use This Manual

- **Learning SOMI from scratch:** Read `theory/SOMI_3_0_FROM_FIRST_PRINCIPLES.md`, then `theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md`, then `theory/SYMBOL_GLOSSARY.md`.
- **Absorption:** Read SOMI_3_0/theory/22_KNOWLEDGE_ABSORPTION.md; for math depth, SOMI_4/KNOWLEDGE_ABSORPTION_DEEP_ANALYSIS.md.
- **What’s implemented vs not:** [THEORY_AHEAD_OF_CODE.md](THEORY_AHEAD_OF_CODE.md).
- **Dual learning / JEPA:** SOMI_3_0/theory/04_THE_DUAL_LEARNING.md, 03_THE_JEPA_CONNECTION.md.
- **Diagnostics:** SOMI_3_0/theory/05_THE_DIAGNOSTICS.md.
- **Life cycle and growth:** `theory/23_SOMI_LIFE_CYCLE.md`, `theory/24_GROWTH_SOLVES_LOCAL_LEARNING.md` (in this repo).

---

## 11. Theory vs Code at a Glance

| You want to… | Use theory for… | Use code for… |
|--------------|------------------|----------------|
| Understand what SOMI is | First principles, 5 levels, symbols | START_HERE_WHAT_IS_THIS.md, demo |
| Implement dynamics | Level 1 equations, 9 forces, geometry eqn | physics/, brain/ |
| Absorb from other models | Full hierarchy, fingerprint, Universal Absorber | absorption/ (transplant, multi-model, alignment, probe fingerprint) |
| Train with macro + micro | Dual learning, JEPA = stress | training/ (DualLearningTrainer, JEPALoss) |
| Know when to grow/compress | Stress-driven grow; stress/spectrum-driven compress | lm/growth.py (auto-grow); compression/ (no auto-compress yet) |
| Check health | Diagnostics list, pathologies | diagnostics/dashboard.py |
| See what’s missing in code | — | THEORY_AHEAD_OF_CODE.md, ABSORPTION_THEORY_VS_IMPLEMENTED.md |

---

*For running and configuring the code, see [USER_MANUAL_CODE.md](USER_MANUAL_CODE.md).*
