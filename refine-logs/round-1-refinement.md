# Round 1 Refinement

## Problem Anchor
- **Bottom-line problem**: Mixed-signal AIMC suffers from hardware non-idealities (IR-drop, ADC quantization/mismatch, device variation) that vary with layer-to-tile assignment. No principled method predicts which layers are most vulnerable to mapping-dependent noise. No existing mapper uses model-side uncertainty for placement decisions.
- **Must-solve bottleneck**: A per-layer sensitivity signal that predicts which layers degrade most when placed on noisy tiles. Current heuristics (power, IR-drop, activation range) predict power/thermal/timing, not robustness under analog noise.
- **Non-goals**: NOT hardware-aware training, NOT NAS, NOT ADC adaptation alone, NOT coarse analog/digital partitioning (HILAL stage 1). Deployment-time mapping only.
- **Success condition**: A demonstrable improvement in accuracy under analog noise from uncertainty-guided mapping vs. heuristic baselines, with a clear ablation showing the uncertainty signal drives the gain.

## Anchor Check
- **Original bottleneck**: Per-layer sensitivity signal for mapping-dependent noise.
- **Revised method still addresses it**: YES. We sharpen U_i to a calibrated, per-noise-mode vulnerability vector matched against per-tile risk components. This is a more precise version of the same bottleneck.
- **Reviewer suggestions rejected as drift**: Deleting the ADC extension is NOT drift — it is simplification that sharpens the contribution. Adding the ensemble estimator is NOT drift — it makes the same uncertainty signal more credible without changing the problem.

## Simplicity Check
- **Dominant contribution after revision**: Calibrated noise-mode-aware uncertainty vector U_i = [U_i^IR, U_i^ADC, U_i^VAR] as layer vulnerability predictor for stage-2 spatial placement in heterogeneous AIMC.
- **Components removed or merged**: ADC extension deleted. MC dropout replaced by small deep ensemble (3-5 seeds) for credibility. Greedy/Hungarian alternatives → fixed to Hungarian only.
- **Reviewer suggestions rejected as unnecessary complexity**: The noise-type-aware vector upgrade is NOT complexity addition — it is a more precise specification of the same scalar coupling. It makes U_i more informative without adding a new module.
- **Why the remaining mechanism is still the smallest adequate route**: Zero new trainable components. The entire contribution is the signal design (U_i vector) + Hungarian optimization. The method is maximally simple — all complexity is in the measurement question, not the mechanism.

## Changes Made

### 1. U_i operationalization — CRITICAL fix
- **Reviewer said**: U_i = Var(z_i^t) is not comparable across layers (tracks width/scale); MC dropout on frozen model trained without dropout is underspecified; need normalized downstream uncertainty.
- **Action**: Reformulate U_i as a calibrated noise-mode-aware vulnerability vector:
  - For each layer i, run a small ensemble (M=5) of trained models (different random seeds)
  - For each noise mode k ∈ {IR, ADC, VAR}: inject calibrated noise at layer i's analog computations and measure output logit variance
  - U_i^k = Var_{m}(f_i(x; θ_m + perturbation_k))
  - Normalize by feature dimension: Û_i^k = U_i^k / d_i (layer output dimensionality)
  - This gives a vector U_i = [Û_i^IR, Û_i^ADC, Û_i^VAR]
- **Reasoning**: This directly measures "how unstable is layer i's output when perturbed by IR-drop-like / ADC-like / variation-like noise?" — precisely what we need for mapping. Normalization makes it comparable across layers. Ensemble (not MC dropout) is more credible since models were trained normally.
- **Impact on core method**: Changes U_i from a scalar to a vector. Changes coupling from S_{i,r} = U_i · H_r to S_{i,r} = Û_i · Ĥ_r (dot product of normalized vectors). Preserves the core hypothesis.

### 2. Contribution focus — IMPORTANT fix
- **Reviewer said**: Risks being "swap one heuristic for another"; Bayesian framing weak; ADC extension adds sprawl.
- **Action**: Delete the ADC precision extension entirely from the main paper. Rename from "Bayesian uncertainty" to "predictive instability" — honest framing that doesn't overclaim Bayesian credentials. Keep only the spatial placement contribution.
- **Reasoning**: One dominant contribution (uncertainty-guided stage-2 placement) is stronger than placement + ADC. "Predictive instability" accurately describes what we measure without implying full Bayesian posterior semantics.
- **Impact on core method**: Sharper, cleaner paper. No loss of core functionality.

### 3. Venue readiness upgrade — IMPORTANT fix
- **Reviewer said**: U_i · H_r scalar product looks shallow; upgrade to noise-type-aware vulnerability vector.
- **Action**: Already addressed in Change 1. U_i is now a [U_i^IR, U_i^ADC, U_i^VAR] vector matched against tile risk vector [h_r^IR, h_r^ADC, h_r^VAR]. The dot product captures which noise mode each layer is vulnerable to. This is a meaningful mechanistic upgrade.
- **Reasoning**: Makes the novelty "uncertainty resolves which noise mode each layer is vulnerable to" — not just "uncertainty is another scalar." Stronger story for top venues.
- **Impact on core method**: Changes H_r from a scalar to a vector. The formulation is now a dot product of vulnerability and risk vectors.

### 4. Optimizer consolidation — MINOR fix
- **Reviewer said**: Greedy/Hungarian/annealing choice is undecided.
- **Action**: Fix to Hungarian algorithm with explicit capacity constraints (one layer per tile or multiple layers per tile with capacity C). Report greedy/annealing results in appendix.
- **Reasoning**: Hungarian is optimal for linear assignment. Fixing it removes a variable from the design space and makes ablations cleaner.
- **Impact on core method**: No change to the mapping result — just one less degree of freedom.

## Revised Proposal

# Research Proposal: Calibrated Uncertainty-Guided Stage-2 Placement for Mixed-Signal AIMC

## Problem Anchor
- **Bottom-line problem**: Mixed-signal AIMC suffers from hardware non-idealities (IR-drop, ADC quantization/mismatch, device variation) that vary with layer-to-tile assignment. No principled method predicts which layers are most vulnerable to mapping-dependent noise. No existing mapper uses model-side uncertainty for placement decisions.
- **Must-solve bottleneck**: A per-layer sensitivity signal that predicts which layers degrade most when placed on noisy tiles. Current heuristics (power, IR-drop, activation range) predict power/thermal/timing, not robustness under analog noise.
- **Non-goals**: NOT hardware-aware training, NOT NAS, NOT coarse analog/digital partitioning (HILAL stage 1). NOT a new training method. Deployment-time mapping only.
- **Success condition**: A demonstrable improvement in accuracy under analog noise from uncertainty-guided mapping vs. heuristic baselines, with ablation showing the signal (not the optimizer) drives the gain.

## Technical Gap
**Current failure**: All existing AIMC mappers use deterministic heuristics (IR-drop magnitude, activation range, variation) to assign layers to tiles. These scores predict power/thermal/timing, not per-layer robustness under analog noise.

**The mechanism missing**: A per-layer signal that quantifies how much each noise mode (IR-drop, ADC error, device variation) specifically destabilizes that layer's output. Not generic sensitivity — noise-mode-specific vulnerability.

**Why naive fixes fail**: MC dropout on a frozen model trained without dropout gives degenerate variance estimates. Generic gradient/Hessian measures training loss sensitivity, not analog noise sensitivity. Activation magnitude measures forward-pass scale, not noise amplification.

## Method Thesis
**One-sentence thesis**: We measure per-layer predictive instability under calibrated, noise-mode-specific perturbations (using a small ensemble of normally-trained models) and use the resulting vulnerability vector U_i = [Û_i^IR, Û_i^ADC, Û_i^VAR] to guide layer-to-tile assignment in heterogeneous AIMC, matching layer vulnerability profiles to tile risk profiles via dot-product coupling.

**Why this is the smallest adequate intervention**: Zero new trainable components. The entire contribution is a calibrated measurement protocol (ensemble + perturbation) plus Hungarian assignment. If the measurement is wrong, the paper is a clean negative result — no mechanism to hide behind.

**Why this route is timely**: HILAL (DATE 2026) introduced Hessian-informed stage-1 allocation (analog vs digital). This creates a natural "stage 2" framing: we do spatial placement inside the analog fabric using noise-mode-calibrated vulnerability profiles instead of Hessian sensitivity.

## Contribution Focus
- **Dominant contribution**: First noise-mode-calibrated uncertainty vector as layer vulnerability predictor for stage-2 spatial placement in heterogeneous AIMC fabrics. The key insight is that different noise modes (IR-drop, ADC error, variation) affect layers differently, and a vector vulnerability profile matched to tile risk profile is more informative than a scalar score.
- **Explicit non-contributions**: NOT a new training method, NOT hardware-aware training, NOT ADC design, NOT theoretical bound, NOT HILAL's stage-1 problem.

## Proposed Method

### Complexity Budget
- **Frozen / reused backbone**: Pre-trained CIFAR-10/100 ResNet with 5 seeds (standard SGD, different random init). No hardware-aware training. No dropout at inference.
- **New trainable components**: ZERO.
- **Tempting additions intentionally excluded**: Hardware-aware training, Bayesian layers, per-tile weight tuning, runtime rerouting, ADC precision adaptation.

### System Overview
```
[Pre-trained ensemble: M=5 models, different random seeds]
           ↓
[For each layer i, for each noise mode k ∈ {IR, ADC, VAR}]
  Inject calibrated perturbation at layer i
  Measure output logit variance across ensemble
           ↓
[Per-layer vulnerability vector: U_i = [Û_i^IR, Û_i^ADC, Û_i^VAR]]
  (normalized by feature dimension)
           ↓
[Tile risk vector: H_r = [h_r^IR, h_r^ADC, h_r^VAR] from simulator]
           ↓
[Coupling score: S_{i,r} = U_i · H_r / ||U_i|| ||H_r||]
  (cosine similarity — invariant to scale)
           ↓
[Hungarian assignment: π* = argmin_π Σ_i S_{i,π(i)}]
           ↓
[Mapped network → Accuracy under AIMC noise]
```

### U_i Vector Computation (Core Mechanism)

**Step 1: Train ensemble**
- Train M=5 ResNet models on CIFAR-10 with different random seeds (standard SGD, no dropout)
- These models represent the posterior distribution p(w|D) in a cheap, approximation-free way
- Ensemble is more credible than MC dropout on a frozen model since these models were trained normally

**Step 2: Calibrated perturbation per noise mode**
For each layer i and each noise mode k:

- **IR-drop perturbation**: Inject conductance perturbation ΔG ~ N(0, σ_IR²) to the analog weights at layer i. Measure output logit change across ensemble members.
  - σ_IR calibrated from hardware data: typical IR-drop induced conductance shift (~1-5% of nominal)
- **ADC perturbation**: Quantize the analog multiply-accumulate output at layer i to b-bit (e.g., b=4 for low precision), then dequantize. Measure logit variance across ensemble.
  - ADC bit-width matches the hardware specification
- **Variation perturbation**: Inject device-to-device conductance variation (lognormal distribution, σ_VAR from hardware spec) to layer i's weights. Measure logit variance.

**Step 3: Vulnerability vector**
- For noise mode k and layer i: U_i^k = (1/M) Σ_m (f_i(x; θ_m + δ_k) - μ_i)² (ensemble variance of logit output under perturbation δ_k)
- Normalize: Û_i^k = U_i^k / d_i (d_i = layer output dimension) to make scores comparable across layers
- Final: U_i = [Û_i^IR, Û_i^ADC, Û_i^VAR]

**Why cosine similarity instead of raw dot product**:
- U_i and H_r may have different magnitudes across noise modes
- Cosine similarity is scale-invariant: S_{i,r} = (U_i · H_r) / (||U_i|| ||H_r||) ∈ [-1, 1]
- This ensures the coupling measures angle (which modes matter most) not magnitude

### H_r Vector Computation
- From AIMC simulator: measure each tile's IR-drop magnitude, ADC quantization error, and device variation severity
- h_r^IR: simulated IR-drop error magnitude for tile r
- h_r^ADC: ADC quantization + mismatch error for tile r at default precision
- h_r^VAR: device variation (conductance distribution spread) for tile r
- Normalize each component across tiles to [0, 1] for cosine similarity

### Assignment Optimization
- **Primary**: Hungarian algorithm on cost matrix C_{i,r} = -S_{i,r} (minimize total similarity-weighted risk)
- **Capacity constraints**: Each tile can hold up to C layers (configurable, default C=1)
- **Baseline comparators** (all with Hungarian optimizer to control for search):
  - Random: random permutation baseline
  - IR-drop-only: Û_i^IR only, h_r^IR only (scalar coupling)
  - Variation-only: Û_i^VAR only, h_r^VAR only
  - Hessian baseline: replace U_i with Hessian trace per layer (HILAL-style sensitivity, adapted to spatial placement)
  - Activation L2: replace U_i with mean activation magnitude per layer

### Why This Is the Smallest Adequate Mechanism
- No new training, no new modules, no new hardware
- The entire contribution is a calibrated measurement protocol
- If the signal doesn't work: clean negative result, no hidden confounds

### Failure Modes and Diagnostics

| Failure Mode | How to Detect | Fallback / Mitigation |
|---|---|---|
| U_i does not predict degradation | Spearman ρ ≈ 0 | Publish as "which proxy predicts layer vulnerability?" — still valuable |
| U_i worse than Hessian | ρ_U < ρ_Hessian | Pivot to Hessian-informed mapping (aligns with HILAL literature) |
| All mappers equal (null result) | No statistically significant difference | Publish as diagnostic: "mapping-invariant regime for CIFAR-10/AIMC" |
| Noise-mode vector not better than scalar | ρ_vector ≈ ρ_scalar | Simplify to scalar U_i (drop the vector) |
| Ensemble variance dominated by model diversity, not noise sensitivity | U_i mostly constant across layers | Try larger ensemble or layer-specific perturbation magnitude calibration |

---

## Claim-Driven Validation Sketch

### Claim 1 (MANDATORY — the predictor test): U_i vector predicts per-layer vulnerability under mapping-sensitive analog noise

- **Minimal experiment**: For each layer i, inject controlled IR-drop, ADC, and variation noise independently at that layer. Measure per-layer accuracy drop on CIFAR-10 test set (n=10000). Rank layers by realized drop. Compare with ranking by Û_i^k for each noise mode separately, and by the full U_i vector (cosine similarity ranking).
- **Baselines / ablations**: (a) Hessian trace per layer, (b) gradient norm per layer, (c) activation L2 norm, (d) random ranking. Report Spearman ρ and top-5 / top-10 hit rate for each.
- **Metric**: Spearman rank correlation ρ (primary); top-k vulnerability identification accuracy (secondary); statistical significance (permutation test, p < 0.05)
- **Expected evidence**: If ρ > 0.5 for U_i and ρ_U > ρ_baselines → Claim 1 validated. If U_i vector is not better than scalar: simplify to scalar and re-test.
- **Decision gate**: If ρ_U < ρ_baselines → the paper pivots to studying WHICH proxy best predicts layer vulnerability (a diagnostic paper).

### Claim 2 (main paper result): Uncertainty-guided mapping achieves better robustness-accuracy trade-off than heuristic mapping

- **Minimal experiment**: Compare 4 mappers (all with Hungarian to control for optimizer): (1) random, (2) IR-drop-only, (3) Hessian-only (HILAL-style adapted to spatial), (4) U_i-guided (full vector). Evaluate on CIFAR-10 ResNet-20/56 under AIMC noise at 3 levels (low/medium/high severity). Report accuracy curves and Pareto frontier.
- **Additional ablations**: (a) U_i vector vs. U_i scalar (to test vector value), (b) same U_i with different optimizers (to confirm signal vs. optimizer), (c) CIFAR-100 + ResNet-32 (to test generalization)
- **Metric**: Accuracy@noise_level (primary), ECE under noise (secondary), energy-accuracy if applicable
- **Expected evidence**: Statistically significant improvement in accuracy under high-noise conditions for U_i-guided vs. all baselines

---

## Experiment Handoff Inputs

- **Must-prove claims**: Claim 1 (predictor validation) is mandatory gate. Claim 2 is the main paper result.
- **Must-run ablations**:
  1. Control for optimizer: same Hungarian algorithm across all mappers
  2. Compare U_i vector vs. U_i scalar (is the vector structure worth it?)
  3. Compare U_i vs. Hessian vs. gradient vs. activation (which proxy wins?)
  4. Test across noise levels (low/medium/high severity of each noise mode)
  5. Test across architectures (ResNet-20, ResNet-56, VGG-11)
  6. Test across datasets (CIFAR-10, CIFAR-100)
- **Critical datasets / metrics**: CIFAR-10/100; Accuracy@noise_level, Spearman ρ for predictor validation, ECE
- **Highest-risk assumptions**: (1) Ensemble variance ≈ noise sensitivity; (2) Simulator noise is realistic enough; (3) Hungarian is the right optimizer

## Compute & Timeline Estimate
- **Estimated GPU-hours**: ~60-80h total
  - Ensemble training (M=5, CIFAR-10, 3 architectures): ~20h
  - Predictor validation (Claim 1): ~20h
  - Mapper comparison (Claim 2): ~25h
  - CIFAR-100 extension + ablations: ~20h
- **Data / annotation cost**: Zero
- **Timeline**: 2-3 weeks with GPU access
