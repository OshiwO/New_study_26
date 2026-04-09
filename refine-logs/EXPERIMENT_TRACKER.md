# Experiment Tracker

**Project**: Calibrated Perturbation-Guided Stage-2 Placement for Mixed-Signal AIMC
**Date**: 2026-04-09
**Total estimated GPU-hours**: ~115h (or ~3h pilot + ~85h full)

---

## Status: NOT STARTED

---

## Pilot Run (Decision Gate Only)

| Run ID | Description | GPU | Est. Time | Status | Result |
|--------|-------------|-----|-----------|--------|--------|
| P1 | Block 2 predictor test: ResNet-20, medium noise, 1 seed | 1 | ~3h | NOT RUN | — |

**Pilot decision**: If Spearman ρ_U ≥ 0.5 and ρ_U ≥ Hessian/gradient baselines → proceed to full run. If not → publish predictor diagnostic, pivot or stop.

---

## Stage 1: U_i Computation + Claim 1 Validation

| Run ID | Description | GPU | Est. Time | Status | Result |
|--------|-------------|-----|-----------|--------|--------|
| S1.1 | Train ResNet-20 (M=5 seeds) | 1 | ~2h | NOT RUN | — |
| S1.2 | Train ResNet-56 (M=5 seeds) | 1 | ~5h | NOT RUN | — |
| S1.3 | Train VGG-11 (M=5 seeds) | 1 | ~5h | NOT RUN | — |
| S1.4 | Compute U_i vectors: ResNet-20 | 1 | ~3h | NOT RUN | — |
| S1.5 | Compute U_i vectors: ResNet-56 | 1 | ~5h | NOT RUN | — |
| S1.6 | Compute U_i vectors: VGG-11 | 1 | ~5h | NOT RUN | — |
| S1.7 | Seed stability analysis | 1 | ~2h | NOT RUN | — |
| S1.8 | Block 2: Per-layer perturbation test (ResNet-20) | 1 | ~10h | NOT RUN | — |
| S1.9 | Block 2: Correlation analysis + baselines | 1 | ~5h | NOT RUN | — |

**Stage 1 decision**: If predictor gate PASS → proceed to Stage 2. If FAIL → stop and publish diagnostic.

---

## Stage 2: Claim 2 — Mapper Comparison

| Run ID | Description | GPU | Est. Time | Status | Result |
|--------|-------------|-----|-----------|--------|--------|
| S2.1 | Simulator heterogeneous tile setup | 1 | ~5h | NOT RUN | — |
| S2.2 | Mapper comparison: 6 mappers × ResNet-20 × 3 noise levels × 3 seeds | 1 | ~15h | NOT RUN | — |
| S2.3 | Mapper comparison: 6 mappers × ResNet-56 × 3 noise levels × 3 seeds | 1 | ~15h | NOT RUN | — |
| S2.4 | Mapper comparison: 6 mappers × VGG-11 × 3 noise levels × 3 seeds | 1 | ~15h | NOT RUN | — |
| S2.5 | CIFAR-100 extension: ResNet-32 × 6 mappers × 3 noise levels × 1 seed | 1 | ~15h | NOT RUN | — |

---

## Stage 3: Ablations

| Run ID | Description | GPU | Est. Time | Status | Result |
|--------|-------------|-----|-----------|--------|--------|
| S3.1 | U_i-vector vs. U_i-scalar | 1 | ~5h | NOT RUN | — |
| S3.2 | Calibrated vs. uncalibrated perturbation | 1 | ~5h | NOT RUN | — |
| S3.3 | Hungarian vs. greedy optimizer | 1 | ~3h | NOT RUN | — |
| S3.4 | Noise model sensitivity | 1 | ~5h | NOT RUN | — |

---

## Summary

| Stage | Runs | Est. Time |
|-------|------|-----------|
| Pilot | 1 | ~3h |
| Stage 1 | 9 | ~42h |
| Stage 2 | 5 | ~65h |
| Stage 3 | 4 | ~18h |
| **Total** | **19** | **~128h** |

**Note**: Stages can be parallelized across GPUs. With 4 GPUs: ~35h total wall time.
