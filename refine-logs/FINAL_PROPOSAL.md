# Research Proposal: Calibrated Perturbation-Guided Stage-2 Placement for Mixed-Signal AIMC

---

## Problem Anchor
- **Bottom-line problem**: Mixed-signal AIMC suffers from hardware non-idealities (IR-drop, ADC quantization, device variation) whose impact varies with layer-to-tile assignment. No existing mapper uses a calibrated, noise-mode-specific measurement of layer vulnerability to guide placement. Current heuristics predict power/thermal/timing, not robustness under analog noise.
- **Must-solve bottleneck**: A per-layer sensitivity signal that predicts which layers degrade most when placed on tiles with specific risk profiles.
- **Non-goals**: NOT hardware-aware training, NOT NAS, NOT coarse analog/digital partitioning (HILAL stage 1). Deployment-time mapping only.
- **Success condition**: Demonstrable improvement in accuracy under analog noise from calibrated perturbation-guided mapping vs. heuristic baselines, with ablation showing the measurement protocol drives the gain.

---

## Method Thesis
We directly measure per-layer vulnerability to specific hardware noise modes via calibrated perturbation-induced KL divergence, producing a noise-mode vulnerability vector U_i = [U_i^IR, U_i^ADC, U_i^VAR]. We match this to tile risk vectors H_r = [h_r^IR, h_r^ADC, h_r^VAR] expressed in the same physical severity units, via a magnitude-preserving dot-product cost C_{i,r} = -U_i · H_r, and solve layer-to-tile assignment via Hungarian algorithm.

**Why this is the smallest adequate intervention**: Zero new trainable components. One calibrated measurement protocol + Hungarian optimization. The entire contribution is the measurement — if it doesn't predict vulnerability, the paper is a clean negative result.

**Why now**: HILAL (DATE 2026, April 20) introduced Hessian-informed stage-1 allocation (analog/digital partitioning). This paper is explicitly "stage 2": spatial placement inside the analog fabric using noise-mode-calibrated vulnerability profiles.

---

## System Overview

```
[Single deployed model (one checkpoint, standard training)]
        ↓
[For each layer i, for each noise mode k ∈ {IR, ADC, VAR}]
  Inject calibrated perturbation δ_k^i to layer i's analog computations
  Measure KL divergence between clean and perturbed output distributions
        ↓
[Vulnerability vector: U_i = [U_i^IR, U_i^ADC, U_i^VAR] in KL units]
        ↓
[Tile risk vector: H_r = [h_r^IR, h_r^ADC, h_r^VAR] in physical units]
  (mV for IR-drop, LSB for ADC, σ_G for variation — from simulator)
        ↓
[Cost matrix: C_{i,r} = - U_i · H_r]
  (magnitude-preserving: both sides in absolute units)
        ↓
[Hungarian assignment: π* = argmin_π Σ_i C_{i,π(i)}]
        ↓
[Mapped network → Accuracy under AIMC noise evaluation]
```

---

## U_i Computation: Perturbation-Induced KL Divergence

**For each layer i, for each noise mode k ∈ {IR, ADC, VAR}:**

1. **Input batch**: N=500 random CIFAR-10 training images (clean, no labels needed)
2. **Reference prediction**: p(y|x) from single forward pass of deployed checkpoint
3. **Calibrated perturbation** δ_k^i applied to layer i's analog computations:
   - **IR-drop**: Conductance perturbation ΔG ~ N(0, σ_IR²) where σ_IR is calibrated from hardware data (typically 1-5% of nominal conductance). Resulting voltage shift: h_r^IR in mV.
   - **ADC quantization**: Quantize MAC output at layer i to b-bit, dequantize. Error measured in LSB units. h_r^ADC in LSB.
   - **Device variation**: Lognormal conductance variation σ_VAR applied to weights at layer i. h_r^VAR in σ_G units.
4. **KL divergence**: U_i^k = (1/N) Σ_n KL(p(y|x_n) || p_k^i(y|x_n, δ_n))
   - Use symmetric JS divergence: (KL(p||q) + KL(q||p)) / 2 to avoid asymmetry
   - Average over N input samples and multiple perturbation draws (T=20-50)
5. **Output**: U_i = [U_i^IR, U_i^ADC, U_i^VAR] in KL units

**Robustness analysis** (reported in appendix, not part of deployment):
- Repeat U_i computation with 3 different random seeds for the checkpoint
- Report coefficient of variation: CV_i = std(U_i) / mean(U_i) per mode
- High CV (>0.2) indicates seed-sensitive layer → flag for the reader, not excluded from mapper

---

## H_r Computation: Physical-Severity Tile Risk Map

**From AIMC simulator characterization (pre-computed, frozen for all experiments):**

- **h_r^IR**: IR-drop magnitude (mV) at tile r under typical activation patterns. Measures how much voltage sag tile r experiences under the expected workload.
- **h_r^ADC**: ADC quantization + mismatch error (LSB) for tile r at default precision. Includes offset error, gain error, and quantization step size.
- **h_r^VAR**: Device variation severity (σ_G, lognormal conductance spread) for tile r. Derived from measured or simulated conductance distribution statistics.

**No normalization across tiles or across modes** — each component is in its natural physical unit. This ensures the dot product U_i · H_r is a calibrated expected-harm measure.

---

## Assignment Optimization

**Cost matrix**: C_{i,r} = - U_i · H_r = -(U_i^IR · h_r^IR + U_i^ADC · h_r^ADC + U_i^VAR · h_r^VAR)

- The cost is the expected KL divergence increase when placing layer i on tile r, summed over all noise modes
- This is a calibrated expected-harm surrogate when the tile's physical risk profile and the layer's perturbation response are on commensurate physical scales
- The layer with the largest U_i component for a given noise mode is most vulnerable to that mode; the tile with the largest h_r for a given mode is most risky for that mode

**Optimizer**: Hungarian algorithm (linear assignment, optimal in O(L³))

- One layer per tile (capacity C=1) for typical ResNet/L compared to tile count
- For larger networks: capacity C > 1 (multiple layers per tile), still solvable via Hungarian with dummy tiles

**Baseline comparators** (all use Hungarian to control for optimizer):
1. **Random**: random permutation baseline
2. **IR-drop-only scalar**: U_i = U_i^IR only, H_r = h_r^IR only (scalar coupling)
3. **Best single-mode**: whichever noise mode has highest predictor correlation in Claim 1
4. **Hessian scalar**: replace U_i with Hessian trace per layer (HILAL-style, adapted to spatial placement)
5. **Activation L2**: replace U_i with mean activation magnitude per layer

---

## Why This Is the Smallest Adequate Mechanism

- **ZERO new trainable components**: No new layers, no new losses, no new training procedures
- **One new measurement**: Perturbation-induced KL divergence per layer per noise mode
- **One standard algorithm**: Hungarian assignment (well-studied, optimal, no hyperparameters)
- The entire contribution is the calibrated measurement protocol — not a new model, not a new optimizer
- If the measurement doesn't predict vulnerability: clean negative result (publish as diagnostic)
- The paper's scientific claim is falsifiable and bounded

---

## Failure Modes

| Failure | Detection | Mitigation |
|---|---|---|
| U_i doesn't predict degradation | Spearman ρ ≈ 0 for all modes | Publish as "which perturbation protocol predicts layer vulnerability?" |
| One mode dominates | ρ_vector ≈ ρ_single | Simplify to best single-mode scalar U_i |
| All mappers equal | No statistically significant difference | Publish as "mapping-invariant regime" diagnostic |
| Seed-CV > 0.2 | Flagged layers have high variance across seeds | Report as robustness caveat; don't exclude from mapper |
| Simulator ≠ silicon | Rankings change with noise model | Add noise model sensitivity analysis |

---

## Claim-Driven Validation

### Claim 1 (MANDATORY — Predictor Validation)
**Calibrated perturbation-induced KL divergence predicts per-layer vulnerability under mapping-sensitive analog noise.**

- **Experiment**: For each layer i, inject each noise mode independently at that layer. Measure per-layer accuracy drop on CIFAR-10 test set (n=10000). Rank by realized drop. Compare with ranking by:
  - Full U_i vector (KL divergence per mode)
  - U_i^IR only, U_i^ADC only, U_i^VAR only
  - Hessian trace, gradient norm, activation L2, random
- **Metrics**: Spearman rank correlation ρ (primary); top-k hit rate (secondary); permutation test p < 0.05
- **Gate**: ρ_U ≥ 0.5 AND ρ_U ≥ max(ρ_baselines) → proceed to Claim 2. If false: publish Claim 1 as standalone diagnostic and stop.

### Claim 2 (Main Paper Result)
**Calibrated perturbation-guided mapping achieves better accuracy-robustness trade-off than heuristic mapping.**

- **Experiment**: 4 mappers (all Hungarian, same optimizer):
  (1) random, (2) IR-drop-only, (3) best single-mode from Claim 1, (4) U_i-vector-guided
  Evaluate on CIFAR-10 ResNet-20/56 + CIFAR-100 ResNet-32. 3 noise severity levels × 3 seeds.
- **Additional ablation**: U_i-vector vs. U_i-scalar (is the multi-mode vector worth it?)
- **Metrics**: Accuracy@noise_level (primary); ECE under noise (secondary); statistical significance (paired t-test, p < 0.05)

---

## Compute Estimate

| Experiment | GPU-Hours |
|-----------|-----------|
| U_i computation (CIFAR-10, 3 arch, 3 seeds) | ~15h |
| Predictor validation (Claim 1) | ~20h |
| Mapper comparison (Claim 2, CIFAR-10) | ~20h |
| CIFAR-100 extension | ~15h |
| Ablations + robustness checks | ~15h |
| **Total** | **~85h** |

**Timeline**: 3-4 weeks with GPU access.

---

## Explicit Non-Contributions
- NOT hardware-aware training or fine-tuning
- NOT neural architecture search
- NOT ADC design or precision adaptation (even if Claim 1 suggests it)
- NOT coarse analog/digital partitioning (HILAL's stage 1)
- NOT a theoretical bound
- NOT silicon validation (simulator only)
