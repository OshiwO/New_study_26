# Refinement Report

**Problem**: Uncertainty-Guided Robustness-Aware Mapping for Mixed-Signal Analog In-Memory Computing
**Initial Approach**: Use Bayesian posterior variance U_i from MC dropout to guide layer-to-tile mapping via R(i,r) = U_i · H_r objective
**Date**: 2026-04-09
**Rounds**: 3 / MAX_ROUNDS
**Final Score**: 7.9 / 10
**Final Verdict**: REVISE (near READY; final cost-function fix already applied in FINAL_PROPOSAL.md)

## Problem Anchor
Mixed-signal AIMC suffers from hardware non-idealities (IR-drop, ADC quantization, device variation) whose impact varies with layer-to-tile assignment. No principled method predicts which layers are most vulnerable to mapping-dependent noise. HILAL (DATE 2026) handles stage-1 (analog/digital partitioning). We address stage-2 (spatial placement inside the analog fabric).

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|----------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8              | 6                  | 6                    | 7                 | 7           | 8                | 6               | 6.7     | REVISE  |
| 2     | 8              | 6                  | 8                    | 8                 | 8           | 9                | 6               | 7.4     | REVISE  |
| 3     | 9              | 7                  | 8                    | 8                 | 8           | 9                | 7               | 7.9     | REVISE  |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|------------------------|------------------|--------|
| 1 | MC dropout on frozen model trained without dropout is underspecified; U_i not comparable across layers; optimizer choice open; ADC extension adds sprawl | Ensemble (M=5) replaces MC dropout; U_i reformulated as noise-mode vector; ADC extension deleted; optimizer fixed to Hungarian; honest framing | Contribution Quality 6→8, Validation 8→9 |
| 2 | Cosine similarity loses tile risk magnitude; "logit variance" conflates clean disagreement with perturbation sensitivity | Perturbation-induced KL divergence replaces variance; magnitude-aware dot product replaces cosine; single checkpoint + seed ensemble as robustness check; explicit novelty framing as measurement protocol | Overall 6.7→7.4 |
| 3 | Cost drops absolute fragility (L2-norm of U_i loses magnitude); [0,1] H_r normalization loses cross-mode unit alignment | Final fix: C_{i,r} = - U_i^T H_r with raw KL-unit U_i and physical-severity H_r in mV/LSB/σ_G units | Near READY (7.9/10) |

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | "Replace MC dropout with Bayesian layers" | Rejected: too expensive, MC dropout enough for the signal if properly framed. Instead used small ensemble (M=5) for credibility | Accepted with compromise |
| 1 | "Delete ADC extension" | Accepted: deletion sharpens the contribution | Simplification accepted |
| 2 | "Use cosine similarity for U_i-H_r coupling" | Rejected in Round 3: cosine loses magnitude. Author agreed but needed magnitude-preserving fix | Accepted after Round 3 fix |
| 3 | "C_{i,r} drops absolute fragility — two layers with same mode mix but different total vulnerability look identical" | Accepted: final fix uses raw dot product with physical-unit H_r, not normalized | Applied in FINAL_PROPOSAL |

## Final Proposal Snapshot
- Canonical version: `refine-logs/FINAL_PROPOSAL.md`
- Final method thesis: We directly measure per-layer vulnerability to IR-drop/ADC/variation noise via calibrated perturbation-induced KL divergence, produce vulnerability vector U_i, match to physical-severity tile risk H_r via calibrated cost C_{i,r} = -U_i · H_r, and assign via Hungarian algorithm.
- Key claims: (1) KL-divergence perturbation response predicts layer vulnerability better than Hessian/gradient; (2) calibrated perturbation-guided mapping beats heuristic mapping
- Zero trainable components. One measurement protocol. One standard optimizer.
- Falsifiable: if U_i doesn't predict degradation, the paper is a clean negative result.

## Remaining Weaknesses
- Final score 7.9/10 (one round away from READY at 9)
- The cost-function fix (magnitude-preserving C_{i,r} with physical-unit H_r) has been applied in FINAL_PROPOSAL.md
- Remaining minor gap: Venue Readiness still at 7 — the paper needs empirical evidence (pilot) to push past the "it's a heuristic" objection

## Raw Reviewer Responses

<details>
<summary>Round 1 Review</summary>

**Overall Score**: 6.7/10 — REVISE
**Key concerns**: Method Specificity (6) — U_i loosely defined; Contribution Quality (6) — risks "swap one heuristic for another"; Venue Readiness (6) — scalar U_i · H_r looks thin
**Fixes applied**: Ensemble replaces MC dropout; U_i vector; ADC deleted; Hungarian fixed

</details>

<details>
<summary>Round 2 Review</summary>

**Overall Score**: 7.4/10 — REVISE
**Key concerns**: Cosine loses tile risk magnitude; KL divergence still needs more specificity; deployment path unclear
**Fixes applied**: Perturbation-induced KL divergence; magnitude-aware dot product; single checkpoint clarification; measurement protocol framing

</details>

<details>
<summary>Round 3 Review</summary>

**Overall Score**: 7.9/10 — REVISE
**Key concern**: Cost C_{i,r} = -Û_i · Ĥ_r drops absolute layer fragility (L2-normalized U_i loses magnitude). [0,1]-normalized H_r loses cross-mode unit alignment.
**Fix applied in FINAL_PROPOSAL**: C_{i,r} = - U_i^T H_r with raw KL-unit U_i and physical-severity H_r in mV/LSB/σ_G units.

</details>
