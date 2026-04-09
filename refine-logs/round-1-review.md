# Round 1 Review

**Reviewer**: GPT-5.4 xhigh
**Round**: 1 / MAX_ROUNDS
**Overall Score**: 6.7 / 10
**Verdict**: REVISE

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8 |
| Method Specificity | 6 |
| Contribution Quality | 6 |
| Frontier Leverage | 7 |
| Feasibility | 7 |
| Validation Focus | 8 |
| Venue Readiness | 6 |
| **Overall** | **6.7** |

## Sub-7 Dimensions

### Method Specificity (6) — CRITICAL
**Weakness**: U_i = Var(z_i^t) is not operationalized tightly. Raw pre-activation variance is not comparable across layers (mostly tracks width/scale). MC dropout on a frozen model trained WITHOUT dropout is underspecified and may give degenerate variance estimates. The optimizer choice (greedy/Hungarian/annealing) is undecided.

**Fix**: Define a normalized downstream uncertainty score under layer-local calibrated analog perturbation — e.g., dataset-averaged logit variance or JS divergence induced by perturbing only layer i, normalized by feature dimension. Fix assignment to a capacity-constrained Hungarian formulation. Relegate other solvers to appendix.

**Priority**: CRITICAL

### Contribution Quality (6) — IMPORTANT
**Weakness**: Current framing risks reading as "replace Hessian/activation heuristic with MC-dropout heuristic." The optional ADC extension adds visible contribution sprawl. The "Bayesian posterior" framing is weak if the estimator is just inference-time dropout on a frozen network.

**Fix**: Make the paper about one thing only: uncertainty as the stage-2 placement signal. Delete the ADC extension from the main paper. Either use a more credible uncertainty estimator (small deep ensemble) or honestly reframe the signal as "stochastic predictive instability under calibrated noise" rather than "Bayesian posterior."

**Priority**: IMPORTANT

### Venue Readiness (6) — IMPORTANT
**Weakness**: The separable scalar score U_i · H_r may look too shallow to support a strong method claim at top venues.

**Fix**: Upgrade the mechanism without adding system bloat: use a noise-type-aware vulnerability vector U_i = [U_i^IR, U_i^ADC, U_i^VAR] matched against a tile risk vector H_r = [h_r^IR, h_r^ADC, h_r^VAR]. Then the dot product U_i · H_r captures which noise mode each layer is vulnerable to. This makes novelty "uncertainty resolves which noise mode each layer is vulnerable to," not just "uncertainty is another scalar."

**Priority**: IMPORTANT

## Simplification Opportunities
1. Delete the ADC precision extension entirely unless the core placement signal wins clearly
2. Pick one optimizer for the main method — prefer Hungarian with explicit capacity constraints
3. Do not learn an extra regression for H_r weights — use direct simulator-measured risk components

## Modernization Opportunities
- No LLM/VLM/diffusion/RL route is appropriate here
- If Bayesian language remains central: a lightweight Laplace or small ensemble is more credible than inference-only MC dropout on a frozen network trained without dropout
- For a more credible uncertainty estimator without training Bayesian layers: use a small deep ensemble (3-5 seeds) of normally-trained models — practical since CIFAR-10 trains fast

## Drift Warning
NONE — the proposal still solves the anchored problem. The ADC-precision optional branch is the only drift risk if retained.

## Verdict
REVISE — The direction is viable. Make the uncertainty signal more precise, more defensible, and less like a generic heuristic replacement.
