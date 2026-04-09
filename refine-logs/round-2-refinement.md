# Round 2 Refinement

## Problem Anchor
- **Bottom-line problem**: Mixed-signal AIMC suffers from hardware non-idealities (IR-drop, ADC quantization/mismatch, device variation) that vary with layer-to-tile assignment. No principled method predicts which layers are most vulnerable to mapping-dependent noise. No existing mapper uses model-side uncertainty for placement decisions.
- **Must-solve bottleneck**: A per-layer sensitivity signal that predicts which layers degrade most under specific noise modes (IR-drop, ADC, variation) when placed on tiles with those specific risk profiles.
- **Non-goals**: NOT hardware-aware training, NOT NAS, NOT coarse analog/digital partitioning (HILAL stage 1). NOT a new training method. Deployment-time mapping only.
- **Success condition**: Demonstrable improvement in accuracy under analog noise from calibrated perturbation-guided mapping vs. heuristic baselines, with ablation showing the calibrated measurement protocol (not the optimizer) drives the gain.

## Anchor Check
- **Original bottleneck**: Per-layer sensitivity signal for mapping-dependent noise.
- **Revised method still addresses it**: YES. U_i^k is now defined as perturbation-induced KL divergence, cleanly separating vulnerability from ensemble disagreement. Dot-product coupling with magnitude awareness. The core question is the same — does calibrated perturbation response predict layer vulnerability?
- **Reviewer suggestions rejected as drift**: None. All fixes sharpen the same mechanism.

## Simplicity Check
- **Dominant contribution after revision**: The calibrated perturbation measurement protocol — noise-mode-specific, perturbation-induced KL divergence as layer vulnerability predictor for stage-2 AIMC placement.
- **Components removed or merged**: Cosine similarity replaced with magnitude-aware dot product. U_i is now KL divergence (cleaner signal). Ensemble is validator of U_i stability, not the primary signal source.
- **Why the remaining mechanism is still the smallest adequate route**: ZERO new trainable components. One clean measurement definition (KL divergence). One optimizer (Hungarian). No added complexity.

## Changes Made

### 1. U_i definition — CRITICAL fix
- **Reviewer said**: "Logit variance across ensemble under perturbation" conflates clean ensemble disagreement with perturbation-induced vulnerability. Cosine throws away magnitude.
- **Action**: Replace U_i^k with perturbation-induced KL divergence. Define:
  - Let p(y|x) = (1/M) Σ_m p_m(y|x) (average predictive distribution of ensemble on clean input)
  - Let p_k^i(y|x, ε) = predictive distribution after applying calibrated perturbation ε of noise mode k to layer i
  - U_i^k = E_{x~D, ε~P_k}[KL(p(y|x) || p_k^i(y|x, ε))] (expected KL divergence between clean and perturbed predictions, averaged over data distribution and perturbation draws)
  - Û_i^k = U_i^k (no normalization across layers — magnitude carries information about absolute vulnerability)
- **Reasoning**: This cleanly isolates perturbation-induced vulnerability from baseline ensemble disagreement. KL divergence is non-negative, zero when the perturbation has no effect, and increases with prediction distribution shift. It directly measures "how much does this noise mode at this layer change the output predictions?"
- **Impact on core method**: Cleaner signal definition. Magnitude preserved in dot product.

### 2. Coupling function — CRITICAL fix
- **Reviewer said**: Cosine similarity loses tile risk magnitude. Mildly bad tile and catastrophic tile look identical.
- **Action**: Replace cosine with magnitude-aware dot product:
  - Û_i = [U_i^IR, U_i^ADC, U_i^VAR] (L2-normalized across modes only: each component divided by ||U_i||_2, so noise modes are weighted equally in direction)
  - Ĥ_r = [ĥ_r^IR, ĥ_r^ADC, ĥ_r^VAR] (each component normalized to [0,1] across tiles, so tile risk magnitude is preserved)
  - S_{i,r} = Û_i · Ĥ_r ∈ [-1, 1] × magnitude(Ĥ_r) — actually: since ĥ components are [0,1], use S_{i,r} = Û_i · Ĥ_r (this preserves relative risk magnitude across tiles)
  - Cost matrix: C_{i,r} = -S_{i,r} (we minimize risk)
- **Reasoning**: Û_i's L2 normalization across modes means all noise modes contribute equally to the directional signal — the placement decision is about which noise mode dominates vulnerability for each layer. Ĥ_r's [0,1] normalization per component preserves the absolute risk of each tile. The dot product combines directional preference (which modes layer i is sensitive to) with absolute risk magnitude (how risky each tile is per mode).
- **Impact on core method**: More informative coupling. Magnitude preserved.

### 3. Deployment clarification — IMPORTANT fix
- **Reviewer said**: Should clarify whether placement is computed per-target model or ensemble signal transferred to one checkpoint.
- **Action**: The mapped model is a SINGLE checkpoint (the one being deployed). The U_i vector is computed on that single checkpoint via repeated perturbation sampling. The M-model ensemble serves as a ROBUSTNESS CHECK on U_i — if U_i is unstable across ensemble members (high variance), the signal is unreliable for that layer.
  - Primary: single checkpoint + perturbation sampling (M_reps = 20-50 random perturbation draws per layer per noise mode)
  - Validation: check U_i stability across ensemble members; if std/mean > threshold, flag layer as unreliable for mapping
- **Reasoning**: The deployed system runs one model, not an ensemble. The ensemble validates that U_i is not an artifact of a single model's randomness.
- **Impact on core method**: Cleaner deployment story. No change to the mapping mechanism.

### 4. Venue novelty framing — IMPORTANT fix
- **Reviewer said**: Reviewers can dismiss it as "vectorized proxy + standard assignment" without proving the calibrated measurement protocol is the real mechanism.
- **Action**: Make the calibrated perturbation measurement protocol the EXPLICIT novelty. Reframe:
  - "We propose a calibrated perturbation measurement protocol for AIMC placement. Instead of using generic sensitivity scores (gradient/Hessian/activation), we directly measure how each layer's output distribution shifts when exposed to hardware-realistic noise perturbations. We show this predicts layer vulnerability better than generic scores."
  - Ablations frame: calibrated perturbation vs. uncalibrated perturbation (random noise); vector U_i vs. scalar U_i (best single mode)
- **Reasoning**: This focuses the contribution on the MEASUREMENT PROTOCOL, not just the result. The protocol is the mechanism — it is what makes the signal informative.
- **Impact on core method**: Cleaner contribution framing. No change to mechanism.

---

## Final Proposal (Round 2)

# Research Proposal: Calibrated Perturbation-Guided Stage-2 Placement for Mixed-Signal AIMC

## Problem Anchor
Mixed-signal AIMC suffers from hardware non-idealities (IR-drop, ADC quantization, device variation) whose impact varies with layer-to-tile assignment. No existing mapper uses a calibrated, noise-mode-specific measurement of layer vulnerability to guide placement. Current heuristics (power, IR-drop magnitude, activation range) predict power/thermal/timing, not robustness under analog noise. HILAL (DATE 2026) does stage-1 coarse analog/digital partitioning. We do stage-2 spatial placement inside the analog fabric.

## Method Thesis
**One-sentence thesis**: We directly measure per-layer vulnerability to specific hardware noise modes (IR-drop, ADC quantization, device variation) via calibrated perturbation-induced KL divergence, and use the resulting noise-mode vulnerability vector to guide layer-to-tile assignment in heterogeneous AIMC via magnitude-aware dot-product coupling.

## System Overview
```
[Single deployed model (one checkpoint)]
         ↓
[For each layer i, for each noise mode k ∈ {IR, ADC, VAR}]
  Apply calibrated perturbation δ_k^i to layer i
  Measure: KL(p(y|x) || p_k^i(y|x, δ))
  Average over: input batch, perturbation draws
         ↓
[Vulnerability vector: U_i = [U_i^IR, U_i^ADC, U_i^VAR]]
  (KL divergence, not variance)
         ↓
[Tile risk vector: H_r = [ĥ_r^IR, ĥ_r^ADC, ĥ_r^VAR]]
  (simulator-measured, per-component normalized)
         ↓
[Coupling: C_{i,r} = - Û_i · Ĥ_r]
  (L2-normalized U_i · component-normalized H_r)
         ↓
[Hungarian assignment: π* = argmin_π Σ_i C_{i,π(i)}]
         ↓
[Mapped network → Accuracy under AIMC noise evaluation]
```

## U_i Computation: Calibrated Perturbation-Induced KL Divergence

For each layer i and noise mode k:

1. **Input batch**: N=500 random CIFAR-10 training images (clean, no labels needed for the perturbation step)
2. **Reference prediction**: p(y|x) = (1/M) Σ_m p(y|x; θ_m) where M=1 (single checkpoint) or M=3 (small ensemble for stability check)
3. **Perturbation**: Apply calibrated noise δ_k^i to layer i's analog computations:
   - **IR-drop**: Conductance perturbation ΔG ~ N(0, σ_IR²) where σ_IR calibrated from hardware data (typically 1-5% of nominal conductance)
   - **ADC quantization**: Quantize the MAC output at layer i to b-bit, dequantize, measure distribution shift
   - **Device variation**: Lognormal conductance variation σ_VAR applied to weights at layer i
4. **Perturbed prediction**: p_k^i(y|x, δ) from perturbed forward pass
5. **KL divergence**: U_i^k = (1/N) Σ_n KL(p(y|x_n) || p_k^i(y|x_n, δ_n))
   - Average over N input samples and multiple perturbation draws
   - Use symmetric KL or JS divergence to avoid asymmetry issues
6. **Ensemble stability check**: If using single checkpoint, repeat with 3 seeds and check std(U_i^k) — if coefficient of variation > 0.2, flag layer as unreliable

## H_r Computation

- From AIMC simulator: measure each tile's risk components
- h_r^IR: IR-drop magnitude (V) at tile r under typical activation patterns
- h_r^ADC: ADC quantization error (LSB) + mismatch (offset/gain error) for tile r at default precision
- h_r^VAR: Device variation severity (σ_G) for tile r
- Normalize each component: ĥ_r^k = (h_r^k - min_k) / (max_k - min_k) across tiles → [0, 1]
- Result: Ĥ_r = [ĥ_r^IR, ĥ_r^ADC, ĥ_r^VAR]

## Coupling and Assignment

- **Direction (Û_i)**: L2-normalize U_i across noise modes: Û_i = U_i / ||U_i||_2. This ensures equal directional weight for each noise mode.
- **Magnitude (Ĥ_r)**: Component-normalized [0,1] tile risk, preserving absolute risk differences across tiles.
- **Score**: S_{i,r} = Û_i · Ĥ_r. Range: [-1, 1] but since Ĥ_r components are non-negative and Û_i ≥ 0, range is [0, 1].
- **Cost matrix**: C_{i,r} = -S_{i,r} (we minimize total risk)
- **Optimizer**: Hungarian algorithm with capacity C (default C=1, one layer per tile; C>1 for larger networks)

## Why This Is the Smallest Adequate Mechanism
- ZERO new trainable components
- No new hardware model — uses existing AIMC simulator for H_r
- The entire contribution is one calibrated measurement protocol (U_i) plus one assignment algorithm (Hungarian)
- If the measurement doesn't predict vulnerability: clean negative result

## Failure Modes

| Failure | Detection | Mitigation |
|---|---|---|
| U_i doesn't predict degradation | Spearman ρ ≈ 0 | Publish diagnostic: "which calibrated perturbation protocol predicts layer vulnerability?" |
| Scalar beats vector | ρ_vector ≈ ρ_scalar | Simplify to best single mode (whichever has highest ρ) |
| All mappers equal | No sig. diff. | Publish as "mapping-invariant regime for CIFAR-10/AIMC" |
| U_i unstable across seeds | CV > 0.2 | Use ensemble average; if still unstable, flag layer and exclude from ranking |
| Sim vs. silicon gap | Ranking changes with noise model | Add noise model sensitivity analysis |

---

## Claim-Driven Validation

### Claim 1 (MANDATORY — Predictor Test)
**Calibrated KL-divergence perturbation response ranks per-layer vulnerability better than generic sensitivity scores.**

- **Experiment**: For each layer i: inject controlled IR-drop, ADC, variation noise independently at that layer. Measure per-layer accuracy drop on CIFAR-10 (n=10000). Rank by realized drop. Compare with ranking by U_i^k per mode, full U_i vector, Hessian trace, gradient norm, activation L2, IR-drop-only scalar.
- **Metrics**: Spearman ρ (primary); top-k hit rate (secondary); permutation test p < 0.05
- **Gate**: ρ_U_vector > 0.5 AND ρ_U_vector ≥ ρ_baselines → proceed to Claim 2. Else: publish Claim 1 as diagnostic.

### Claim 2 (Main Result)
**Calibrated perturbation-guided mapping achieves better accuracy-robustness trade-off than heuristic mapping under AIMC noise.**

- **Experiment**: 4 mappers (all Hungarian to control for optimizer): (1) random, (2) IR-drop-only scalar, (3) Hessian/gradient scalar (best baseline from Claim 1), (4) U_i-vector-guided. CIFAR-10 ResNet-20/56 + CIFAR-100 ResNet-32. 3 noise levels × 3 seeds.
- **Metrics**: Accuracy@noise_level, ECE under noise, Pareto frontier
- **Expected**: Statistically significant improvement for U_i-vector at high-noise conditions

### Claim 3 (Conditional Extension — only if Claim 1 validated with vector advantage)
**The vector structure (noise-mode specificity) is what drives the gain — not just the calibration.**

- **Experiment**: Compare full vector U_i vs. best single-mode scalar U_i^k. If vector ≈ scalar: simplify to scalar in final paper.

---

## Compute Estimate
- ~70-90 GPU-hours total
  - U_i computation (Claim 1): ~25h (perturbation sweeps × layers × noise modes × seeds)
  - Mapper comparison (Claim 2): ~30h
  - CIFAR-100 extension: ~20h
  - Ablations + noise sensitivity: ~15h
- ~2-3 weeks with GPU access
