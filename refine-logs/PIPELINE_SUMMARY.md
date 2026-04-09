# Pipeline Summary

**Problem**: Uncertainty-Guided Robustness-Aware Mapping for Mixed-Signal Analog In-Memory Computing
**Final Method Thesis**: We directly measure per-layer vulnerability to IR-drop/ADC/variation noise via calibrated perturbation-induced KL divergence, produce vulnerability vector U_i, match to physical-severity tile risk H_r via magnitude-preserving cost C_{i,r} = -U_i · H_r, and assign via Hungarian algorithm.
**Final Verdict**: REVISE — 7.9/10, near READY. Final cost-function fix applied.
**Date**: 2026-04-09

---

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

---

## Contribution Snapshot
- **Dominant contribution**: First calibrated perturbation measurement protocol (noise-mode-specific KL divergence) as layer vulnerability predictor for stage-2 spatial placement in heterogeneous AIMC fabrics
- **Optional supporting contribution**: Noise-mode vector vs. scalar (only if vector adds value over best single mode)
- **Explicitly rejected complexity**: Hardware-aware training, Bayesian layers, ADC precision adaptation, HILAL stage-1 partitioning, theoretical bounds

---

## Must-Prove Claims
1. **Claim 1 (MANDATORY GATE)**: Calibrated perturbation-induced KL divergence predicts per-layer vulnerability under mapping-sensitive analog noise (Spearman ρ ≥ 0.5 AND ρ ≥ all baselines)
2. **Claim 2 (main result)**: Perturbation-guided mapping achieves better accuracy-robustness trade-off than heuristic mapping under AIMC noise

---

## First Runs to Launch
1. **P1 (Pilot, ~3h)**: Block 2 predictor test on ResNet-20, medium noise, 1 seed — MANDATORY GATE
2. **S1.1-S1.3**: Train ResNet-20/56 and VGG-11 ensembles (M=5 seeds, ~12h)
3. **S1.4-S1.6**: Compute U_i vectors for all architectures (~13h)
4. **S1.8-S1.9**: Full Block 2 predictor validation + baselines (~15h)
5. **S2.1-S2.5**: Mapper comparison + CIFAR-100 (~65h)

---

## Main Risks

| Risk | Mitigation |
|------|------------|
| U_i doesn't predict vulnerability (Spearman ρ < 0.5) | Publish as predictor diagnostic; stop main paper path |
| U_i worse than Hessian/gradient baseline | Pivot to best proxy; frame paper as "comparing placement signals" |
| All mappers equal (null result) | Publish as "mapping-invariant regime" |
| GPU budget exceeded | Run pilot first; only proceed if pilot is positive |
| Simulator ≠ silicon reality | Add noise model sensitivity analysis (Ablation 4) |

---

## Evolution of the Method

| Round | Method | Score |
|-------|--------|-------|
| Initial | MC dropout variance, R(i,r)=U_i·H_r, ADC extension | 6.7 |
| After R1 | Ensemble (M=5), noise-mode vulnerability vector, ADC deleted, Hungarian | 7.4 |
| After R2 | Perturbation-induced KL divergence, magnitude-aware dot product | 7.4 |
| After R3 | Physical-unit H_r (mV/LSB/σ_G), raw calibrated cost C_{i,r}=-U_i·H_r | 7.9 |

---

## Next Action
- **Immediate**: Run pilot P1 (~3h) to validate predictor signal before committing full GPU budget
- **If pilot positive**: Proceed to full experiment plan (Stage 1 → Stage 2 → Stage 3)
- **If pilot negative**: Publish Block 2 predictor diagnostic; pivot to comparing which proxy (Hessian/gradient/KL) best predicts layer vulnerability
- **Then**: `/run-experiment` to deploy the validated experiment plan
- **After results**: `/auto-review-loop` to iterate until submission-ready
