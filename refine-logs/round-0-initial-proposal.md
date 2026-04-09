# Research Proposal: Uncertainty-Guided Spatial Layer Placement for Mixed-Signal AIMC

## Problem Anchor
- **Bottom-line problem**: Mixed-signal AIMC suffers from hardware non-idealities (IR-drop, ADC quantization/mismatch, device variation) that vary with how layers are assigned to hardware tiles. No principled method exists to predict which layers are most vulnerable to this mapping-dependent noise, and no existing mapper uses Bayesian model uncertainty to guide placement decisions.
- **Must-solve bottleneck**: The mapping function π: {1..L} → {1..T} (layers to tiles) is currently solved with heuristic scores (power, IR-drop, activation range) that do NOT predict per-layer vulnerability under analog noise. The critical missing piece is a signal that predicts which specific layers will degrade most when placed on noisy tiles.
- **Non-goals**: This is NOT about hardware-aware training (HAT), NOT about architectural search (NAS), NOT about ADC precision adaptation alone, NOT about coarse analog/digital partitioning (HILAL's stage 1). NOT about training-time robustness — deployment-time mapping only.
- **Constraints**: CIFAR-10/100 and TinyImageNet datasets; simulated AIMC with realistic noise models (no silicon); ~50-100 GPU-hours budget; target NeurIPS/ICML/DAC.
- **Success condition**: A demonstrable, statistically significant improvement in accuracy under analog noise from uncertainty-guided mapping vs. heuristic baselines, with a clear ablation showing the uncertainty signal (not the search procedure) drives the gain.

---

## Technical Gap

**Current failure**: All existing AIMC mappers use deterministic heuristics (IR-drop, variation magnitude, activation range) to assign layers to tiles. These scores do NOT predict per-layer vulnerability under mapping-dependent analog noise — they predict power/thermal/timing, not robustness.

**Why the gap persists**: The community has not asked "which layers, if mis-mapped, cause the most accuracy drop under analog noise?" Instead, it asks "which layers consume the most power or have the largest IR-drop?" These are different optimization targets.

**The mechanism missing**: A per-layer sensitivity signal that quantifies how much a layer's output degrades when perturbed by mapping-sensitive analog noise sources. Bayesian posterior variance U_i is a candidate, but it measures parameter uncertainty, not noise sensitivity. The gap is connecting model-side fragility to hardware-side risk.

**Why naive fixes fail**:
- "Just use MC dropout variance" assumes epistemic uncertainty = analog noise sensitivity. Not proven.
- "Just use gradient/Hessian" (HILAL route) measures training loss sensitivity, not analog noise sensitivity.
- "Just use activation magnitude" measures forward-pass scale, not noise amplification.
- The correct signal must predict how a layer's OUTPUT (not parameters or activations alone) changes under mapping-dependent hardware perturbations.

---

## Method Thesis

**One-sentence thesis**: We test whether Bayesian posterior predictive variance (MC dropout), estimated under deployment-relevant analog noise conditions, predicts per-layer vulnerability better than existing heuristics, and whether minimizing Σ_i U_i · H_{π(i)} produces better robustness than IR-drop-only or variation-only placement.

**Why this is the smallest adequate intervention**: We do NOT propose a new training method, new architecture, or new hardware. We propose one new scoring function U_i (posterior variance under analog noise) plus an existing assignment optimizer. The entire contribution is the signal — if U_i is better than heuristics, we win; if not, we have a clean negative result.

**Why this route is timely**: HILAL (DATE 2026) has just introduced Hessian-informed allocation for analog/digital partitioning. This creates a natural "stage 1 vs stage 2" framing: HILAL decides which layers go analog vs digital; we decide where inside the analog fabric those layers should go, guided by Bayesian uncertainty. The window for this positioning is NOW.

---

## Contribution Focus

- **Dominant contribution**: First evaluation of Bayesian posterior uncertainty as a layer vulnerability predictor for spatial placement inside heterogeneous AIMC fabrics. Not claiming it is "the right" signal — empirically testing whether it is a *useful* signal vs. Hessian/gradient/activation baselines.
- **Supporting contribution (optional, only if evidence supports)**: A joint placement + ADC precision extension if the uncertainty signal also predicts which layers benefit from higher ADC precision.
- **Explicit non-contributions**: NOT a new training method, NOT a new hardware architecture, NOT a theoretical bound, NOT an ADC design paper, NOT a noise model characterization paper.

---

## Proposed Method

### Complexity Budget
- **Frozen / reused backbone**: Pre-trained CIFAR-10/100 ResNet (standard SGD training, no hardware-aware training). The network weights are frozen at deployment time — only the mapper uses them.
- **New trainable components**: ZERO. The U_i computation is inference-time scoring on frozen weights. No training required.
- **Tempting additions intentionally excluded**: Hardware-aware training, Bayesian layers (DropConnect/Bayes by Backprop), per-tile weight tuning, runtime rerouting.

### System Overview
```
[Frozen DNN weights]
       ↓
[MC dropout forward passes: T times]
       ↓
[Per-layer uncertainty: U_i = Var_{w~p(w|D)}[f_i(x;w)]]
       ↓
[Hardware risk map: H_r per tile]  (from AIMC simulator characterization)
       ↓
[Coupled score: S_{i,r} = U_i · H_r]
       ↓
[Assignment optimizer: greedy / Hungarian / RL]
       ↓
[Mapped network → Accuracy under AIMC noise evaluation]
```

### Core Mechanism

**U_i computation**:
- Standard MC dropout: run T=30 forward passes with dropout active on the frozen trained network
- Per layer i: collect the pre-activation outputs {z_i^t}, compute variance across passes
- U_i = (1/T) Σ_t ||z_i^t - μ_i||²  (Euclidean variance, per layer)
- Alternative: predictive entropy H_i = -Σ_k p̂_i^k log p̂_i^k (softmax entropy, per layer)
- Use Pearson/Spearman correlation against realized per-layer accuracy drop as the validation metric

**H_r computation**:
- From AIMC simulator: characterize each tile r independently
- h_r^IR: simulated IR-drop error magnitude for tile r
- h_r^ADC: ADC quantization + mismatch error for tile r at default precision
- h_r^VAR: device variation (conductance distribution spread) for tile r
- H_r = α·h_r^IR + β·h_r^ADC + γ·h_r^VAR (weighted sum, weights from regression on chip data or simulator)

**Assignment optimization**:
- Score-based greedy: assign highest U_i layer to lowest H_r tile iteratively
- OR: Hungarian assignment on cost matrix C_{i,r} = U_i · H_r
- OR: Simulated annealing with restart on J(π) = Σ_i U_i·H_{π(i)} + λ·Cost(π)
- Baseline comparators: random, IR-drop-only (U_i replaced by layer IR-drop magnitude), variation-only, uniform

**Critical validation step (the predictor test)**:
- For each layer i: measure actual accuracy degradation when layer i is perturbed by controlled IR-drop, ADC noise, or variation
- Compare ranking of realized degradation against ranking by U_i, Hessian trace, gradient norm, activation L2, random
- Report Spearman ρ and top-k hit rate for each proxy
- THIS is the experiment that decides whether the paper lives or dies

### Optional Supporting Component (conditional)
- **Joint ADC precision allocation** (Idea 3 extension): Only if U_i correlation experiment shows U_i also predicts marginal benefit from extra ADC bits
- If pursued: extend H_r to H_r^p where p ∈ {4, 6, 8} bits; add energy(π) to objective; optimize with two-level search
- This is intentionally NOT the primary method — it is an optional extension if the core signal is validated

### Modern Primitive Usage
- **Which LLM/VLM/Diffusion/RL-era primitive is used**: NONE. This is intentionally old-school: MC dropout (widely available since 2016), standard DNNs, greedy/Hungarian assignment. The contribution is not a modern primitive — it is a careful empirical comparison of signals.
- **Why no frontier primitive is needed**: The bottleneck is not representation learning or generation. It is a measurement question: which proxy best predicts layer vulnerability? This does not require diffusion models or LLMs.
- **Why we don't use Bayesian neural networks**: Bayesian layers would give "better" uncertainty estimates but at massive training cost. MC dropout gives a free uncertainty estimate on top of a pre-trained network. If MC dropout is uninformative, Bayesian layers won't save the paper — they would just be an expensive failure.

### Integration into Base Pipeline
1. Train ResNet on CIFAR-10 normally (frozen, no hardware-aware training)
2. Characterize H_r from simulator (pre-computed, frozen)
3. Compute U_i from MC dropout on test set (or validation set)
4. Run mapper to get π*
5. Evaluate under AIMC noise: accuracy, ECE, calibration

### Training Plan
- No training of the mapper. No training of the network with hardware awareness.
- This is a pure inference-time method evaluation.

### Failure Modes and Diagnostics

| Failure Mode | How to Detect | Fallback / Mitigation |
|---|---|---|
| U_i does not predict degradation | Spearman ρ near 0 or negative | Abandon mapping method, publish as negative result (which proxies work?) |
| U_i worse than Hessian/gradient | ρ_U < ρ_Hessian | Re-label as "Hessian-informed mapping" (aligns with HILAL) |
| All mappers equal (null result) | No statistically significant difference | Publish as diagnostic: "mapping-invariant regime" paper |
| Simulator results don't match silicon | Ranking changes with noise model | Add sensitivity analysis to simulator assumptions |
| Search procedure, not signal, drives gain | Ablation: same signal + random search vs. greedy | Control for optimizer by comparing same optimizer with different signals |

---

## Claim-Driven Validation Sketch

### Claim 1: U_i (MC dropout variance) predicts per-layer vulnerability under mapping-sensitive analog noise

- **Minimal experiment**: Controlled perturbation experiment. For each layer i: inject controlled IR-drop (targeted conductance perturbation), ADC quantization noise, or variation noise ONLY at that layer's computations. Measure per-layer accuracy drop on CIFAR-10 test set (n=10000). Rank layers by realized drop. Compare with ranking by U_i from MC dropout.
- **Baselines / ablations**: (a) Hessian trace per layer, (b) gradient norm, (c) activation L2 norm, (d) random ranking. Report Spearman ρ and top-5 / top-10 hit rate for each.
- **Metric**: Spearman rank correlation ρ (primary); top-k vulnerability identification accuracy (secondary); statistical significance (p < 0.05 via permutation test)
- **Expected evidence**: If ρ > 0.5 for U_i and ρ_U > ρ_baselines → Claim 1 validated. If ρ_U < ρ_baseline → Claim 1 falsified, pivot to better proxy.
- **This is the MUST-RUN experiment. If this fails, the paper has no center.**

### Claim 2: Uncertainty-guided mapping achieves better robustness-accuracy trade-off than heuristic mapping

- **Minimal experiment**: Compare 4 mappers (random, IR-drop-only, variation-only, U_i·H_r) on CIFAR-10 ResNet-20/56 under AIMC noise simulation. Report accuracy vs. noise level curves and Pareto frontier.
- **Baselines**: Same 4 mappers with same Hungarian optimization (control for optimizer), plus HILAL-style Hessian mapper adapted to spatial placement
- **Metric**: Accuracy under 3 noise levels × 3 seeds; energy-accuracy Pareto if applicable
- **Expected evidence**: Statistically significant improvement in accuracy under high-noise conditions for U_i·H_r vs. baselines

### Claim 3 (conditional): U_i also predicts marginal benefit from higher ADC precision

- **Run only if**: Claim 1 validated AND U_i correlates with "gain from extra ADC bits" (checked by post-hoc analysis)
- **Minimal experiment**: For each layer i, simulate with 4-bit and 8-bit ADC and measure accuracy delta. Check correlation between U_i and Δaccuracy(8bit−4bit).
- **Metric**: Pearson correlation between U_i and per-layer ADC precision benefit
- **Expected evidence**: If ρ > 0.4 → justify pursuing Idea 3 (joint ADC+placement)

---

## Experiment Handoff Inputs

- **Must-prove claims**: Claim 1 (U_i predictor validity) is mandatory. Claim 2 is the main paper result. Claim 3 is optional.
- **Must-run ablations**: (1) Compare mappers with same optimizer (control for search); (2) Compare U_i vs. Hessian vs. gradient vs. activation baselines; (3) Test across noise levels (low/medium/high IR-drop, low/medium/high ADC error); (4) Test across network architectures (ResNet-20, ResNet-56, VGG-11)
- **Critical datasets / metrics**: CIFAR-10 (primary), CIFAR-100 (secondary), TinyImageNet (if budget allows); Accuracy@noise_level, ECE, Spearman ρ for predictor validation
- **Highest-risk assumptions**: (1) MC dropout variance ≈ analog noise sensitivity; (2) Simulator noise is realistic enough; (3) Layer-level assignment is the right granularity

---

## Compute & Timeline Estimate

- **Estimated GPU-hours**: ~60-80h total
  - Predictor validation (Claim 1): ~15h (MC dropout × 3 architectures × 3 seeds)
  - Mapper comparison (Claim 2): ~30h (4 mappers × 3 noise levels × 3 seeds)
  - CIFAR-100 extension: ~20h
  - Exploratory / ADC analysis: ~15h
- **Data / annotation cost**: Zero (public CIFAR-10/100)
- **Timeline**: 2-3 weeks with GPU access
