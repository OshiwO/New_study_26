# Round 2 Review

**Reviewer**: GPT-5.4 xhigh (continued from Round 1)
**Round**: 2 / MAX_ROUNDS
**Overall Score**: 7.4 / 10
**Verdict**: REVISE

## Scores

| Dimension | Round 1 | Round 2 |
|-----------|---------|---------|
| Problem Fidelity | 8 | 8 |
| Method Specificity | 6 | 6 |
| Contribution Quality | 6 | 8 |
| Frontier Leverage | 7 | 8 |
| Feasibility | 7 | 7 |
| Validation Focus | 8 | 9 |
| Venue Readiness | 6 | 6 |
| **Overall** | **6.7** | **7.4** |

## Progress
- Contribution Quality: 6 → 8 (ADC extension deleted, Bayesian framing replaced with honest "predictive instability")
- Validation Focus: 8 → 9 (cleaner claim structure)
- Method Specificity: 6 → 6 (still needs one more tightening pass)
- Venue Readiness: 6 → 6 (needs explicit novelty framing)

## Sub-7 Dimensions

### Method Specificity (6) — CRITICAL
**Weakness**: Pure cosine similarity throws away absolute tile risk magnitude — a mildly bad tile and a catastrophic tile with the same noise-type mix look identical. Also, "logit variance across ensemble under perturbation" still conflates clean ensemble disagreement with perturbation-induced vulnerability.

**Fix**: Define U_i^k as perturbation-induced output change, not raw ensemble variance. For example:
`U_i^k = E_{x,eps}[KL(p(y|x) || p(y|x, δ_k^i(eps)))]`
Then use magnitude-aware placement cost `S_{i,r} = Û_i^T H_r` (weighted dot product, not cosine). Keep Hungarian fixed.

**Priority**: CRITICAL

### Venue Readiness (6) — IMPORTANT
**Weakness**: Reviewers can still dismiss it as "vectorized proxy + standard assignment" without proving the calibrated measurement protocol itself is the real mechanism.

**Fix**: Make the calibrated measurement protocol the explicit novelty. Keep ablations minimal: calibrated vs uncalibrated perturbations, and vector U_i vs scalarized U_i. Do not add more system pieces.

**Priority**: IMPORTANT

## Simplification Opportunities
1. Replace cosine coupling with direct nonnegative risk cost S_{i,r} = Û_i^T H_r — simpler and magnitude-aware
2. Treat the ensemble as an estimator stabilizer, not the conceptual centerpiece
3. If single deployed checkpoint transfer is weak: compute U_i on target checkpoint with repeated perturbation samples; ensemble only as robustness check

## Modernization Opportunities
NONE

## Drift Warning
NONE. Mild practical caveat: the proposal should explicitly state whether placement is computed for each target model individually or whether an ensemble-derived signal is transferred to one deployed checkpoint.

## Verdict
REVISE — Still needs one more tightening pass to fix the objective definition.
