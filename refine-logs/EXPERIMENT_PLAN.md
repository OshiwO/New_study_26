# Experiment Plan: Calibrated Perturbation-Guided Stage-2 Placement for Mixed-Signal AIMC

## From FINAL_PROPOSAL

**Method thesis**: We directly measure per-layer vulnerability to specific hardware noise modes (IR-drop, ADC quantization, device variation) via calibrated perturbation-induced KL divergence, producing vulnerability vector U_i = [U_i^IR, U_i^ADC, U_i^VAR]. We match this to tile risk vectors H_r in physical units (mV, LSB, σ_G) via magnitude-preserving cost C_{i,r} = -U_i · H_r, and solve layer-to-tile assignment via Hungarian algorithm.

**Must-prove claims**:
1. Calibrated KL-divergence perturbation response predicts per-layer vulnerability under mapping-sensitive analog noise (Claim 1 — MANDATORY GATE)
2. Perturbation-guided mapping achieves better accuracy-robustness trade-off than heuristic mapping (Claim 2)

**Must-run ablations**:
- U_i-vector vs. U_i-scalar (which noise mode matters?)
- Calibrated vs. uncalibrated perturbation (does calibration matter?)
- All mappers with same Hungarian optimizer (control for optimizer)
- Across noise levels, architectures, and datasets

---

## Experiment Block 1: U_i Computation + Stability Analysis

**What**: Compute U_i vectors for ResNet-20, ResNet-56, VGG-11 on CIFAR-10. Validate stability across seeds.

**Run order**: FIRST (gate for everything else)

### Step 1.1: Ensemble Training
- Train ResNet-20, ResNet-56, VGG-11 on CIFAR-10 with M=5 seeds each
- Standard SGD: lr=0.1, epochs=200, batch=128, no dropout, no augmentation beyond standard
- Save checkpoints at epoch 200 for all seeds
- **GPU estimate**: ~15h (CIFAR-10 trains fast, ~2h per seed per arch)

### Step 1.2: U_i Computation
- For each checkpoint × each layer × each noise mode:
  - Input: N=500 random CIFAR-10 training images
  - Reference forward pass: collect pre-softmax logits
  - Perturbation: calibrated δ_k^i (see calibration below)
  - Perturbed forward pass: collect logits
  - Compute JS divergence: U_i^k = JS(p || p_perturbed)
  - Repeat with T=20 perturbation draws, average
- **Perturbation calibration**:
  - IR-drop: σ_IR = 0.01 × G_nominal (1% nominal conductance, hardware-calibrated)
  - ADC: b=4-bit quantization at default ADC precision
  - Variation: σ_VAR = 0.1 × G_nominal (10% variation, hardware-calibrated)
- **GPU estimate**: ~10h (batch all layers/noise modes)
- **Total Block 1**: ~25h

### Step 1.3: Seed Stability Analysis
- Compute U_i with 3 different checkpoint seeds
- Report CV_i = std(U_i) / mean(U_i) per layer per mode
- Flag layers with CV > 0.2 (seed-sensitive)
- Report in appendix; do NOT exclude from mapper

**Decision gate**: If U_i variance is high (CV > 0.5 for most layers), re-calibrate perturbation magnitude before proceeding.

---

## Experiment Block 2: Claim 1 — Predictor Validation (MANDATORY GATE)

**What**: Does U_i predict actual layer vulnerability under hardware noise?

**Run order**: SECOND (must pass to justify Claim 2)

### Step 2.1: Per-Layer Controlled Perturbation Test
- For each layer i in ResNet-20:
  - Inject IR-drop noise ONLY at layer i (σ_IR, same calibration as U_i computation)
  - Measure accuracy drop: Δacc_i^IR = acc_clean - acc_perturbed
  - Repeat for ADC noise (b=4-bit at layer i only) and variation noise
  - Use CIFAR-10 test set (n=10000)
- Repeat with 3 noise severity levels per mode (low: 0.5× calibrated, medium: 1×, high: 2×)
- **GPU estimate**: ~15h (10000 images × 27 layers × 3 modes × 3 levels × 3 seeds)

### Step 2.2: Correlation Analysis
- For each noise mode k: compute Spearman ρ between U_i^k and realized Δacc_i^k
- For full vector: rank by ||U_i||_2 vs. rank by ||Δacc_i||_2
- Compare against baselines:
  - Hessian trace per layer (approximated via finite differences of loss)
  - Gradient norm per layer (first-order backprop)
  - Activation L2 norm (mean activation magnitude per layer)
  - Random ranking
- Report: ρ values, top-5 hit rate, top-10 hit rate, permutation test p-value

### Decision Gate
- If ρ_U ≥ 0.5 AND ρ_U ≥ max(ρ_baselines): PASS → proceed to Block 3
- If ρ_U < 0.5 OR ρ_U < all baselines: FAIL → publish Block 2 as standalone diagnostic paper ("Which perturbation protocol predicts layer vulnerability?"); STOP main paper path
- If ρ_scalar > ρ_vector: simplify to best single-mode scalar in Block 3

**GPU estimate**: ~15h
**Total Block 2**: ~30h

---

## Experiment Block 3: Claim 2 — Mapper Comparison

**What**: Does perturbation-guided mapping beat heuristic mapping under AIMC noise?

**Run order**: THIRD (only if Block 2 passes)

### Step 3.1: AIMC Simulator Setup
- Configure simulator with realistic tile heterogeneity:
  - Tile grid: e.g., 4×4 or 8×8 tiles with spatially-varying IR-drop, ADC quality, device variation
  - Default: uniform baseline (all tiles same risk)
  - Heterogeneous: vary h_r^IR by ±30%, h_r^ADC by ±2 LSB, h_r^VAR by ±0.05 σ_G across tiles
  - High-severity: extreme heterogeneity for stress test
- Validate that simulator produces visible accuracy degradation under noise

### Step 3.2: Baseline Mappers (all Hungarian)
1. **Random**: random permutation π_rand
2. **IR-drop-only**: U_i = U_i^IR scalar, H_r = h_r^IR scalar, C_{i,r} = -U_i^IR × h_r^IR
3. **Best single-mode** (from Block 2 gate): whichever k had highest ρ
4. **Hessian scalar** (HILAL-style): U_i = Hessian trace per layer, H_r = h_r^IR + h_r^VAR (best combined heuristic)
5. **Activation L2**: U_i = activation L2 norm
6. **U_i-vector (Ours)**: full U_i = [U_i^IR, U_i^ADC, U_i^VAR], H_r = [h_r^IR, h_r^ADC, h_r^VAR], C_{i,r} = -U_i · H_r

### Step 3.3: Evaluation
- For each mapper × each network × each noise level × each seed:
  - Assign layers to tiles via Hungarian
  - Evaluate accuracy under AIMC noise
  - Evaluate ECE under noise
  - Evaluate accuracy under CLEAN (to check for accuracy loss under nominal conditions)
- **Datasets**: CIFAR-10 (primary), CIFAR-100 + ResNet-32 (secondary)
- **Noise levels**: low, medium, high severity
- **GPU estimate**: ~30h (4 mappers × 3 archs × 3 noise levels × 3 seeds × CIFAR-10; CIFAR-100 adds ~15h)

### Metrics
- **Primary**: Accuracy@noise_level (mean ± std across seeds, paired t-test for significance)
- **Secondary**: ECE@noise_level, clean accuracy (should not degrade)
- **Pareto frontier**: plot accuracy vs. "total assigned risk" Σ_i U_i·H_{π(i)} for all mappers

**Total Block 3**: ~30h primary + 15h CIFAR-100 = ~45h

---

## Experiment Block 4: Ablations

**Run order**: FOURTH (after main results)

### Ablation 1: U_i-vector vs. U_i-scalar
- Does the vector structure (all 3 noise modes) add value over the best single mode?
- Compare U_i-vector vs. best single-mode scalar mapper on high-severity noise
- If ρ_vector ≈ ρ_scalar: collapse to scalar in final paper

### Ablation 2: Calibrated vs. Uncalibrated Perturbation
- Replace calibrated σ_IR with random noise of the same magnitude
- Compare U_i rankings — does calibration matter?
- Expected: calibrated should outperform random (validates the measurement protocol)

### Ablation 3: Same Signal, Different Optimizer
- Compare Hungarian vs. greedy on same U_i-vector signal
- If greedy ≈ Hungarian: the contribution is the signal, not the optimizer
- Expected result: Hungarian ≥ greedy (optimizer should not dominate)

### Ablation 4: Noise Model Sensitivity
- Re-run Block 3 with different noise models (Gaussian vs. lognormal variation, different IR-drop profiles)
- Check if mapper rankings change — if yes, results are simulator-dependent
- Report as robustness caveat

**GPU estimate**: ~15h total

---

## Budget Summary

| Block | Description | GPU-Hours |
|-------|-------------|-----------|
| 1 | Ensemble training + U_i computation + stability | ~25h |
| 2 | Claim 1: Predictor validation (MANDATORY GATE) | ~30h |
| 3 | Claim 2: Mapper comparison | ~45h |
| 4 | Ablations | ~15h |
| **Total** | | **~115h** |

**With PILOT budget (2h max per idea)**: Run Block 2 predictor test ONLY for ResNet-20, single noise level, 1 seed. ~3h. If positive, run full Block 2 then Block 3.

**Recommended staged execution**:
1. **Pilot** (~3h): Block 2 ResNet-20, medium noise, 1 seed → confirms predictor validity
2. **Stage 1** (~25h): Full Block 2 + Block 1 stability → Claim 1 validated
3. **Stage 2** (~45h): Full Block 3 mapper comparison → Claim 2 results
4. **Stage 3** (~15h): Ablations + CIFAR-100

---

## Decision Gates

| Gate | Condition | Action if FAIL |
|------|-----------|----------------|
| Block 1 (U_i stability) | CV > 0.5 for most layers | Re-calibrate perturbation magnitude |
| Block 2 (Predictor) | ρ_U ≥ 0.5 AND ρ_U ≥ max baselines | Publish Block 2 as diagnostic, STOP main paper |
| Block 3 (Mapper) | Statistically significant improvement over ALL baselines | If null result: publish as "mapping-invariant regime" |

---

## Key Baselines to Implement

1. **HILAL-style Hessian mapper**: For spatial placement (not just analog/digital partitioning), compute Hessian trace per layer, use same Hungarian assignment with H_r
2. **IR-drop-only scalar**: U_i = U_i^IR, H_r = h_r^IR — existing practice
3. **Random**: null baseline to establish scale of effect
