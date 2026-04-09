# Round 3 Review

**Reviewer**: GPT-5.4 xhigh (continued)
**Round**: 3 / MAX_ROUNDS
**Overall Score**: 7.9 / 10
**Verdict**: REVISE

## Scores

| Dimension | Round 1 | Round 2 | Round 3 |
|-----------|---------|---------|---------|
| Problem Fidelity | 8 | 8 | 9 |
| Method Specificity | 6 | 6 | 7 |
| Contribution Quality | 6 | 8 | 8 |
| Frontier Leverage | 7 | 8 | 8 |
| Feasibility | 7 | 7 | 8 |
| Validation Focus | 8 | 9 | 9 |
| Venue Readiness | 6 | 6 | 7 |
| **Overall** | **6.7** | **7.4** | **7.9** |

## Remaining Issue: Cost Function Loses Magnitude

**Problem**: `C_{i,r} = - Û_i · Ĥ_r` with Û_i L2-normalized and Ĥ_r [0,1]-normalized per component drops absolute layer fragility magnitude. Two layers with the same noise-mode mix but very different total vulnerability look identical to the mapper. Similarly, [0,1]-normalized H_r loses cross-mode unit alignment.

**Fix**: Use magnitude-preserving calibrated cost:
- C_{i,r} = - U_i^T H_r (raw KL-divergence vulnerability × physical-severity tile risk)
- U_i in KL units (directly from perturbation measurement)
- H_r expressed in same physical severity units: ĥ_r^IR = IR-drop severity in mV, ĥ_r^ADC = ADC error in LSB, ĥ_r^VAR = variation severity in σ_G units
- No normalization of U_i across modes or across layers — magnitude carries information about absolute fragility
- Hungarian still optimal; result: layers with higher absolute vulnerability are more strongly penalized for being on risky tiles

## Method-Level Fixes Applied
1. C_{i,r} = - U_i^T H_r with physical-unit H_r (not [0,1]-normalized)
2. Seed-CV check removed from deployment path; kept as appendix robustness analysis only
3. H_r on commensurate physical scale with U_i perturbations

## Drift: NONE
## Simplification: Removing the L2 normalization layer — simpler and more informative
## Verdict: After this final fix, the proposal should be READY.
