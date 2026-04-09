# Review Summary

**Problem**: Uncertainty-Guided Robustness-Aware Mapping for Mixed-Signal Analog In-Memory Computing
**Date**: 2026-04-09
**Rounds**: 3 / MAX_ROUNDS
**Final Score**: 7.9 / 10
**Final Verdict**: REVISE (at 3 rounds; proposal is substantially improved from 6.7 → 7.9)

## Problem Anchor
Mixed-signal AIMC suffers from hardware non-idealities (IR-drop, ADC quantization, device variation) whose impact varies with layer-to-tile assignment. No principled method predicts which layers are most vulnerable to mapping-dependent noise. No existing mapper uses model-side uncertainty for placement decisions. HILAL (DATE 2026) handles stage-1 (analog/digital partitioning); we address stage-2 (spatial placement inside the analog fabric).

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|------------------------|------------------------------------------|---------|----------------|
| 1 | Method specificity: MC dropout on frozen model is underspecified; U_i not comparable across layers; optimizer choice undecided; ADC extension adds sprawl | Replaced MC dropout with small ensemble (M=5 seeds); reformulated U_i as noise-mode-calibrated vulnerability vector; deleted ADC extension; fixed optimizer to Hungarian; honest "predictive instability" framing | Partial | Cost function still loses magnitude |
| 2 | Cosine similarity throws away tile risk magnitude; logit variance conflates disagreement with perturbation sensitivity | Replaced with perturbation-induced KL divergence (U_i); magnitude-aware dot product (U_i · H_r); single checkpoint with seed ensemble as robustness check; explicit novelty framing as measurement protocol | Partial | Final round: cost function still loses magnitude — Û_i L2-norm drops absolute fragility |
| 3 | Cost function drops absolute layer fragility; two layers with same mode mix but different total vulnerability look identical; [0,1] H_r normalization loses cross-mode unit alignment | Final fix: C_{i,r} = - U_i^T H_r with physical-unit H_r (mV, LSB, σ_G) and raw KL-unit U_i — magnitude-preserving calibrated cost | YES (final fix applied) | Near READY at 7.9/10 |

## Overall Evolution
- **Round 1**: From vague "Bayesian uncertainty" to concrete noise-mode vulnerability vector + ensemble measurement. ADC extension deleted.
- **Round 2**: From ensemble variance to perturbation-induced KL divergence. From cosine to magnitude-aware dot product. Deployment clarified (single checkpoint, ensemble as robustness check).
- **Round 3**: From normalized scores to raw physical-unit cost. From [0,1]-normalized H_r to physical-severity H_r in mV/LSB/σ_G units. The cost is now a calibrated expected-harm surrogate.

## Final Status
- **Anchor status**: PRESERVED — every revision sharpened the same bottleneck, never drifted
- **Focus status**: TIGHT — one dominant contribution (calibrated perturbation measurement protocol), zero trainable components, no contribution sprawl
- **Modernity status**: APPROPRIATELY CONSERVATIVE — no LLM/VLM/diffusion/RL needed; this is a measurement protocol question
- **Strongest parts**: Clean falsifiability (predictor test gate); honest framing (no Bayesian overclaiming); zero new trainable components; HILAL positioning as "stage 2"
- **Remaining weaknesses**: Not quite at 9/10 yet (reviewer flagged remaining 7.9); needs one more round to close the cost-function fix, but that fix is already applied in the FINAL_PROPOSAL.md

## Key Method Upgrades Across Rounds
1. **MC dropout → small ensemble**: More credible uncertainty without Bayesian training overhead
2. **Variance → KL divergence**: Cleanly separates perturbation-induced vulnerability from baseline disagreement
3. **Cosine → raw dot product**: Preserves absolute fragility magnitude in the cost
4. **[0,1] H_r → physical-severity H_r**: Calibrated expected-harm surrogate, not just similarity heuristic
5. **ADC extension deleted**: Sharper single contribution
6. **"Bayesian" → "predictive instability"**: Honest framing, avoids overclaiming posterior semantics
