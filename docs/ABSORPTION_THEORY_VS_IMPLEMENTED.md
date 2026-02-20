# Absorption: Theory vs Implemented

You had a **much more advanced** absorption story in the theory. This doc spells out what exists on paper vs what’s in the clean `somi` package so we don’t lose track.

---

## Theory (SOMI_Research)

### Hierarchy of methods (KNOWLEDGE_ABSORPTION_DEEP_ANALYSIS.md)

| Level | Name | Idea |
|-------|------|------|
| **0** | Weight averaging | Baseline; often breaks structure. |
| **1** | **Delta transplant** | $\Delta W = W_{\text{specialist}} - W_{\text{base}}$; graft the “what was learned” into SOMI. |
| **2** | **Stress-guided transplant** | Transplant only where *target* is stressed (ignorant) and *source* is settled (knowledgeable). |
| **3** | **Spectral mode transfer** | Transfer knowledge via eigenspectrum / Laplacian modes, not raw weights. |
| **4** | **Curvature-matched cross-architecture** | Match information curvature between different architectures before transferring. |
| **5** | **Topological absorption (Wilson loops)** | Gauge-invariant loops; transfer structure that doesn’t depend on coordinate system. |

### Full pipeline (22_KNOWLEDGE_ABSORPTION.md)

1. **Knowledge fingerprinting** — Stress profile, eigenspectrum, Wilson loops, settling rates (what does each model *know*?).
2. **Knowledge diff** — What does model B know that A doesn’t? (Domain-level comparison.)
3. **Topological extraction** — Pull out the knowledge (delta, or spectral/topological component).
4. **Surgical transplant** — Graft into SOMI with physics-preserving constraints.
5. **Integrity check** — Hamiltonian, stress profile, Wilson loop stability, mass–conductivity.

Plus: **open-weight** absorption, **black-box (API)** absorption, and the **Universal Absorber** (multi-model with ordering, interference prevention, schedule).

---

## What’s in the clean `somi` package

### Implemented and in the repo

| Piece | Module | What it does |
|-------|--------|--------------|
| **Delta transplant** (Level 1) | `somi/absorption/transplant.py` | `compute_delta(W_spec, W_base)`, `transplant_knowledge(part, delta, strength)`. |
| **Multi-model absorb** | `somi/absorption/multi_model.py` | `absorb_multiple(brain, specialists, base_weights)` — one delta per Part (e.g. code→Part0, math→Part1). |
| **Cross-size alignment** | `somi/absorption/alignment.py` | `svd_align(W_source, target_dim)`, `cka_similarity(W_a, W_b)` for different-dimension transplant. |
| **Fingerprint (probe-based)** | `somi/absorption/fingerprint.py` | `compute_fingerprint(brain, probe_inputs, probe_labels)`, `compare_fingerprints(before, after)` — probe confidence/entropy, not (yet) stress/eigenspectrum/Wilson. |
| **Integrity** | `somi/absorption/integrity.py` | `check_integrity(brain)` — physics and health after transplant. |
| **Output-only distillation** | `somi/absorption/distillation.py` | `OutputDistiller` — KL on soft labels when weight transplant isn’t possible. |
| **Pretrained init** | `somi/absorption/pretrained_init.py` | `init_from_pretrained` — initialize SOMI from a pretrained transformer’s weights. |

So in the clean package you **do** have: Level 1 delta transplant, multi-specialist absorption, alignment, probe-based fingerprinting, integrity, output distillation, and pretrained init.

### Not implemented in code (theory only)

- **Level 2:** Stress-guided transplant (mask/weight delta by stress maps).
- **Level 3:** Spectral mode transfer (eigenvectors/eigenvalues of Laplacian).
- **Level 4:** Curvature-matched cross-architecture (information Ricci, etc.).
- **Level 5:** Wilson loops and topological absorption.
- **Fingerprint (full):** Stress profile, eigenspectrum, Wilson loops per domain (theory 22_KNOWLEDGE_ABSORPTION §4); we only have probe confidence/entropy.
- **Knowledge diff:** “What does B know that A doesn’t?” as a formal, domain-level diff.
- **Black-box (API) absorption:** Using only API outputs; topological distillation, active probing via stress (theory §9–10).
- **Universal Absorber schedule:** Ordering of teachers, interference prevention, full absorption schedule (theory §10).

---

## Summary

- **Advanced absorption in theory:** 6-level hierarchy (0–5), full 5-step pipeline, fingerprint with stress/eigenspectrum/Wilson, knowledge diff, multi-model scheduling, black-box absorption.
- **Advanced absorption in the clean package:** Level 1 + multi-model + alignment + probe fingerprint + integrity + output distillation + pretrained init. Levels 2–5 and the full fingerprint/diff/scheduling/black-box are **not** implemented yet; they’re the “much more advanced” part that exists on paper and in SOMI_Research docs.

If you want to prioritize bringing the “advanced” into code, the next logical steps are: **stress-guided transplant (Level 2)** and **richer fingerprinting** (stress profile + eigenspectrum), then spectral transfer (Level 3) and scheduling for the Universal Absorber.
