# Research Idea Report

**Direction**: "Uncertainty-Guided Robustness-Aware Mapping for Mixed-Signal Analog In-Memory Computing"
**Generated**: 2026-04-09
**Pipeline**: research-lit → idea-creator (Codex GPT-5.4) → novelty-check (Codex) → research-review
**Ideas evaluated**: 10 generated → 6 survived filtering → 0 piloted (no GPU available) → 3 recommended

---

## Executive Summary

The literature landscape confirms a clear structural gap: all prior AIMC mapping strategies use deterministic heuristics (IR-drop, power, latency), and no existing work couples Bayesian posterior uncertainty to layer-to-hardware assignment. The user's paper outline with its noise taxonomy (mapping-related vs mapping-weakly-related noise) and `R(i,r) = U_i · H_r` formulation is novel. Three prioritized ideas emerged: (1) Posterior-Weighted Layer Mapping as the core paper contribution; (2) Joint ADC Precision + Uncertainty-Aware Placement as the top-venue upgrade path; (3) Is Bayesian Uncertainty Actually the Right Signal? as the critical scientific validation that de-risks the entire agenda.

Pilot experiments could not be run — no GPU infrastructure available. All ideas flagged as **"needs manual pilot"**.

---

## Literature Landscape Summary

### Core Gap Confirmed
No existing work uses Bayesian posterior statistics (predictive variance/entropy) to guide layer-to-hardware assignment in mixed-signal AIMC. All prior mapping work uses deterministic heuristics.

### Key Related Work
| Paper | Venue | Contribution | Gap |
|-------|-------|-------------|-----|
| Zhong et al. | Nature Electronics 2023 | Hardware-aware training for PCM-AIMC | Doesn't guide mapping |
| Wen et al. | IEEE TCAD 2022 | Variation-aware tile allocation | No Bayesian uncertainty |
| Guo et al. | Nature Comms 2023 | Per-layer ADC precision adaptation | No uncertainty coupling |
| Jiang et al. | DAC 2022 | Uncertainty-aware NAS | Architecture, not hardware mapping |
| Sardar & Gkesiou | IEEE TBioCAS | BNNs in spiking hardware | Different substrate |
| HILAL (IBM) | DATE 2026 | Hessian-informed layer allocation (analog-digital) | Adjacent — use as Related Work |

### Structural Gaps
1. No uncertainty-guided mapping in AIMC literature
2. No noise taxonomy separating mapping-sensitive vs. mapping-invariant noise
3. No joint optimization of Bayesian uncertainty + hardware reliability profiles
4. The `R(i,r) = U_i · H_r` coupled risk formulation is structurally novel

---

## Ranked Ideas

### Idea 1: Posterior-Weighted Layer Mapping — **RECOMMENDED (Core Paper)**
- **Hypothesis**: Layers with higher Bayesian posterior predictive variance are more error-amplifying under mapping-sensitive analog noise (IR-drop, ADC mismatch, device variation). Assigning them to lower-risk AIMC regions via `π* = argmin_π Σ_i U_i · H_{π(i)}` should improve robustness more than IR-drop-only, variation-only, or activation-size-based heuristics.
- **Minimum experiment**: Train CIFAR-10 ResNet with MC dropout (10 forward passes), compute per-layer `U_i = Var_{w~p(w|D)}[f_i(x;w)]`, simulate AIMC with heterogeneous tile risk maps, compare 4 mapping strategies: random, IR-drop-only, variation-only, and `U_i·H_r`-minimization. Measure accuracy under injected noise and calibration curves.
- **Expected outcome**: If positive: +1-3% accuracy under high noise, Pareto improvement on accuracy–robustness–cost frontier. If null: eliminates the core premise and redirects the paper.
- **Novelty**: HIGH (MEDIUM-HIGH confidence) — no prior work couples Bayesian posterior uncertainty to AIMC layer placement. Exact coupling of `U_i` from Bayesian posterior with tile-level `H_r` risk profiles appears new. Closest: HILAL (DATE 2026) uses Hessian not Bayesian uncertainty.
- **Feasibility**: LOW compute — ~20 GPU-hours for full experiment with 3 seeds. PILOT: ~2h on 1 GPU (smaller noise sweep, 1 seed).
- **Risk**: LOW — formulation is well-anchored in existing outline.
- **Contribution type**: new method
- **Pilot result**: SKIPPED — no GPU available. Flagged as **needs manual pilot**.
- **Reviewer's likely objection**: "Why is Bayesian uncertainty the right quantity rather than any generic sensitivity metric?" — addressed by running Idea 2 in parallel.
- **Why do this**: This is the cleanest instantiation of the paper's core novelty. It directly validates the noise taxonomy and the `R(i,r) = U_i·H_r` hypothesis.

---

### Idea 2: Is Bayesian Uncertainty Actually the Right Signal? — **RECOMMENDED (De-risking)**
- **Hypothesis**: If Bayesian uncertainty (MC dropout variance) is genuinely the right signal for mapping, it should outperform cheap deterministic proxies (gradient norm, activation statistics, Hessian trace) in ranking layer vulnerability. If it doesn't, that negative result redirects the entire agenda.
- **Minimum experiment**: On CIFAR-10 ResNet/20, compute 5 per-layer scores: (a) MC dropout variance, (b) ensemble variance (3-5 seeds), (c) gradient norm, (d) activation L2 norm, (e) Hessian trace estimate. Plug each into `R(i,r)=U_i·H_r` and rank mapping priorities. Measure Spearman correlation between ranking and actual accuracy degradation under AIMC noise. Also test whether Bayesian calibration (ECE) correlates with mapping benefit.
- **Expected outcome**: If Bayesian wins: strengthens the paper's scientific story. If a cheap proxy (gradient norm) wins: negative result that changes what's recommended for practical deployment, still publishable as "what actually predicts hardware vulnerability?"
- **Novelty**: MEDIUM — the comparative scientific question is novel, but the individual signals (MC dropout, Hessian, gradients) are all known. HILAL (DATE 2026) is adjacent work using Hessian for analog-digital allocation — must cite and differentiate. Risk: "just a benchmark study."
- **Feasibility**: LOW compute — ~10-15 GPU-hours. Can run in parallel with Idea 1.
- **Risk**: MEDIUM — may be perceived as benchmarking study without a new algorithm. Framing is critical.
- **Contribution type**: empirical finding / diagnostic
- **Pilot result**: SKIPPED — no GPU available. Flagged as **needs manual pilot**.
- **Reviewer's likely objection**: "HILAL already does Hessian-based layer allocation — why should we care about another comparison?" — Response: HILAL optimizes for performance, not robustness under mapping-dependent analog noise.
- **Why do this**: De-risks the entire research agenda. If the hypothesis is wrong, better to know before building the full paper.

---

### Idea 3: Joint ADC Precision Allocation + Uncertainty-Aware Placement — **RECOMMENDED (Top-Venue Upgrade)**
- **Hypothesis**: Posterior-uncertain layers benefit disproportionately from higher ADC precision. A joint optimizer that simultaneously assigns (a) layers to tiles AND (b) ADC precision levels per tile under a fixed energy budget produces a stronger Pareto frontier than either placement-only or ADC-scaling-only baselines.
- **Minimum experiment**: Extend the AIMC simulator to support per-tile ADC precision (e.g., 4-bit, 6-bit, 8-bit options). Define an energy cost per precision level. Replace the `H_r` risk map with a joint `(region, precision)` risk model. Optimize with a two-level search: (1) greedy placement + discrete precision allocation, (2) gradient-free optimization (random restart hill climbing). Compare against Guo et al. (ADC-only) and Idea 1 (placement-only).
- **Expected outcome**: If joint optimization wins: +2-5% accuracy at iso-energy vs. either单独 method. Strong DAC/ISCA paper.
- **Novelty**: MEDIUM-HIGH — the coupling of Bayesian uncertainty to ADC precision is new. Guo et al. (Nature Comms 2023) does ADC adaptation but not uncertainty-guided. LRMP (Frontiers 2024) does mixed-precision mapping but for area/performance, not robustness.
- **Feasibility**: MEDIUM compute — ~30 GPU-hours for full sweep with energy Pareto.
- **Risk**: MEDIUM — requires more engineering (simulator extension) but produces a stronger systems contribution.
- **Contribution type**: new method
- **Pilot result**: SKIPPED — no GPU available. Flagged as **needs manual pilot**.
- **Reviewer's likely objection**: "This is just combining mixed-precision allocation with mapping — where is the conceptual novelty?" — Defense: the novelty is uncertainty-guided joint co-design targeting robustness under mapping-dependent analog non-idealities, not accuracy under nominal conditions.
- **Why do this**: Strongest path to DAC/ISCA. Builds on Idea 1, extends to a full cross-layer co-design problem.

---

### Idea 4: Mapping-Sensitive vs. Mapping-Invariant Noise Decomposition — **BACKUP (Diagnostic)**
- **Hypothesis**: Only a subset of AIMC non-idealities (IR-drop, ADC mismatch, region-dependent variation) materially changes accuracy under remapping. The rest (global read noise, thermal noise) is invariant to mapping decisions and should not appear in `H_r`.
- **Minimum experiment**: For each noise source independently, sweep multiple mappings (10-20 random assignments) and measure accuracy spread. Compute mapping sensitivity index `S_n` per noise source. Validate the taxonomy empirically.
- **Expected outcome**: Confirm that only 3-4 of 6-7 noise sources are mapping-sensitive. Identify which ones matter most for the `H_r` objective.
- **Novelty**: MEDIUM — taxonomy is novel, but the finding is more validation than method.
- **Feasibility**: LOW — ~5-10 GPU-hours.
- **Risk**: LOW — diagnostic, but as standalone paper is weak.
- **Contribution type**: diagnostic
- **Pilot result**: SKIPPED — no GPU available. Flagged as **needs manual pilot**.
- **Reviewer's likely objection**: "This is a taxonomy, not a research contribution." — Must be framed as ablation support for the main paper.
- **Why as backup**: Useful as ablation/protocol within the main paper, not as a standalone contribution.

---

### Idea 5: Channel/Block-Level Uncertainty Mapping — **BACKUP (Granularity Extension)**
- **Hypothesis**: Uncertainty is spatially concentrated within layers (per residual block, per channel). Finer-grained mapping units recover gains left on the table by whole-layer placement.
- **Minimum experiment**: Compute uncertainty per ResNet block (not per layer), per group of channels, and map at that granularity onto heterogeneous-risk subarrays. Compare accuracy–energy Pareto vs. layer-level mapping.
- **Expected outcome**: Modest improvement if layer-level gains are already near saturation.
- **Novelty**: MEDIUM — granularity tweak is not a new principle.
- **Feasibility**: MEDIUM — requires subarray-level simulator support, ~30 GPU-hours.
- **Risk**: MEDIUM — gains may not justify complexity.
- **Contribution type**: new method
- **Pilot result**: SKIPPED — no GPU available. Flagged as **needs manual pilot**.
- **Why as backup**: Only pursue if Idea 1 shows whole-layer gains are near saturation.

---

### Idea 6: Robustness-Aware Mapping Under Distribution Shift — **ELIMINATED**
- **Reason eliminated**: Orthogonal to the core hardware mapping problem. "We also tested CIFAR-10-C" reads as opportunistic benchmarking. Only有价值 if it reveals a surprising interaction between epistemic uncertainty and analog non-idealities under OOD inputs — but this is a stretch without a specific mechanistic hypothesis.
- **Revised verdict**: Deprioritize until after Ideas 1-3 are validated.

---

### Idea 7: Instance-Adaptive Routing — **ELIMINATED**
- **Reason eliminated**: Changes the deployment model too much — introduces runtime policy, control overhead, latency jitter, and heterogeneous execution semantics. Reviewers will say the problem solved is different from the one motivated. Too disruptive to the static mapping framing of the paper.
- **Revised verdict**: Not appropriate for this paper's scope. Could be a future extension if the static mapping paper is successful.

---

### Idea 8: Theoretical Bound for Uncertainty-Risk Mapping — **ELIMINATED**
- **Reason eliminated**: Very high effort, low probability of persuasive payoff. The bound will likely require strong Lipschitz/linearization assumptions that gap significantly from real mixed-signal AIMC behavior. Weak theory is often worse than no theory in hardware venues.
- **Revised verdict**: Only revive if, after empirical work, a simple bound falls out naturally from the analysis.

---

### Idea 9: Uncertainty-Guided Robustness Saturation Curve — **ELIMINATED**
- **Reason eliminated**: "Of course mapping matters less when mapping-invariant noise dominates" — risks being a sanity check. Publishable as one figure inside a larger paper, not as a standalone contribution.
- **Revised verdict**: Include as an ablation/plot in the main paper, not as an independent idea.

---

### Idea 10: Negative Result: Robustness from Curvature, Not Bayesian Uncertainty — **ELIMINATED**
- **Reason eliminated**: If curvature wins, the paper undermines its own premise. If Bayesian wins, it's just an auxiliary comparison. Too risky as a standalone paper framing.
- **Revised verdict**: Run this as part of Idea 2 (the "Is Bayesian Uncertainty Right?" study), not as an independent paper.

---

## Recommended Execution Sequence

### Immediate (next 4-6 weeks)
1. **Idea 3 + Idea 4 (diagnostic) first** — validate the noise taxonomy: confirm which noise sources actually vary with mapping. This defines `H_r` properly before any mapping optimization.
2. **Idea 1 (core) third** — deploy `U_i · H_r` mapping in the regime where it should matter (validated by Idea 3 findings).
3. **Idea 2 (de-risking) in parallel with Idea 1** — test whether Bayesian uncertainty is actually the best signal. If a cheap proxy wins, pivot to using that instead.

### If promising (weeks 6-12)
4. **Idea 5 (ADC joint optimization)** — only if Ideas 1+2 confirm the core hypothesis. This is the high-upside extension to a DAC/ISCA paper.

### If Ideas 1 or 2 fail
- If Bayesian uncertainty loses to curvature: pivot to curvature-based mapping (reformulate `U_i` as gradient/Hessian sensitivity)
- If only 1-2 noise sources matter for `H_r`: simplify `H_r` to just those components
- If mapping gains are null: treat as negative result — the paper becomes "why mapping-aware deployment is harder than it looks"

---

## Pilot Experiment Summary (SKIPPED — No GPU Available)

| Idea | Est. Pilot Time | Est. GPU-Hours | Signal Metric | Status |
|------|----------------|----------------|---------------|--------|
| Idea 1 | ~2h (single seed, reduced noise sweep) | 2h | +1-3% under high noise | NEEDS MANUAL PILOT |
| Idea 2 | ~1.5h (single network, 5 proxy methods) | 1.5h | Spearman ρ vs. accuracy drop | NEEDS MANUAL PILOT |
| Idea 3 | ~3h (greedy + restart search) | 3h | Energy-accuracy Pareto improvement | NEEDS MANUAL PILOT |
| Idea 4 | ~1h (noise sweep, 10 mappings) | 1h | Mapping sensitivity index per noise | NEEDS MANUAL PILOT |

**Total recommended pilot budget**: ~8 GPU-hours across 4 ideas (parallelized if GPUs available).

---

## What Was Eliminated and Why

| Idea | Elimination Reason |
|------|-------------------|
| Idea 6 (Distribution Shift) | Orthogonal to AIMC mapping — opportunistic extra benchmark |
| Idea 7 (Instance-Adaptive Routing) | Changes deployment model; too disruptive to core framing |
| Idea 8 (Theoretical Bound) | High effort, weak payoff, strong assumptions gap from reality |
| Idea 9 (Saturation Curve) | Sanity check, not standalone contribution |
| Idea 10 (Curvature vs. Bayesian) | Undermines premise if negative; auxiliary comparison if positive |

---

## Next Steps

- [ ] **Run pilots manually**: Ideas 1-4 are all pilotable in 1-3h on available GPU
- [ ] If pilot positive for Idea 1 → proceed to full experiment with multi-seed validation
- [ ] If pilot for Idea 2 shows cheap proxy wins → reformulate `U_i` accordingly
- [ ] If Ideas 1+2 both positive → pursue Idea 5 as DAC/ISCA upgrade path
- [ ] Invoke `/research-refine-pipeline` with pilot results to get formal method refinement and experiment plan
- [ ] Or invoke `/run-experiment` to deploy the validated experiment plan to GPU

---

## Key References for This Agenda

- Zhong et al., Nature Electronics 2023 — hardware-aware training for PCM-AIMC
- Wen et al., IEEE TCAD 2022 — variation-aware tile allocation
- Guo et al., Nature Comms 2023 — ADC precision scaling
- HILAL (IBM Research), DATE 2026 — Hessian-informed layer allocation [NEW — cite as related work]
- LRMP, Frontiers in AI 2024 — mixed-precision mapping for IMC
- Lin et al., Nature Machine Intelligence 2023 — memristor Bayesian DNN
- Lin et al., Nature Computational Science 2024/2025 — deep Bayesian active learning with IMC hardware
