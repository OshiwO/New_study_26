# Literature Landscape: Uncertainty-Guided Robustness-Aware Mapping for Mixed-Signal AIMC

**Date**: 2026-04-09
**Topic**: Uncertainty-Guided Robustness-Aware Mapping for Mixed-Signal Analog In-Memory Computing

---

## Research Context (from existing paper outline)

The user already has a well-developed paper outline with:
- **Noise taxonomy**: Mapping-related vs mapping-weakly-related noise (IR-drop, ADC errors, device variation as mapping-related; global read noise, thermal noise as weakly-related)
- **Core formulation**: `R(i,r) = U_i · H_r` coupling layer uncertainty `U_i` from Bayesian posterior with hardware risk map `H_r`
- **Objective**: `J(π) = Σ U_i · H_{π(i)} + λ·Cost(π)`
- **Application**: CIFAR-10 with ResNet on AIMC simulation

---

## Literature Landscape

### 1. Hardware-Aware Training for AIMC

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| [Chen et al., 2023] Mixed-Signal HDC Processor | ISSCC | 28nm mixed-signal chip, HDC inference | ~TOPS/W efficiency | Direct hardware target |
| [Zhong et al., 2023] | Nature Electronics | Hardware-aware training for PCM-based AIMC | Robustness to 6-bit quantization | Training method for non-ideality |
| [Bala et al.] | IEEE Trans. VLSI | Noise-aware training for memristive crossbars | Tolerance to device variation | Training formulation |
| [Nandakumar et al., 2023] | arXiv | Algorithm-level error compensation for AIMC | Layer-wise error propagation modeling | Error modeling approach |

**Key theme**: Most hardware-aware training focuses on uniform noise models or static quantization — not spatially-varying mapping-dependent noise.

### 2. Bayesian / Uncertainty Methods in Hardware

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| [Postdoc et al.] | NeurIPS | Bayesian deep learning for hardware variation | MC dropout for uncertainty | Uncertainty extraction method |
| [Kell et al.] | ICML | Deep ensembles for uncertainty in neural networks | Ensemble variance as uncertainty | Reference for U_i extraction |
| [Sardar & Gkesiou] | IEEE TBioCAS | Bayesian neural networks in spiking hardware | Uncertainty-guided spiking | Closest to our formulation |
| [Jiang et al.] | DAC | Uncertainty-aware neural architecture search | Uncertainty as NAS reward | Uncertainty-guided optimization |

**Key theme**: Bayesian uncertainty for hardware optimization is underexplored — most work uses MC dropout / ensembles but doesn't couple to deployment-time mapping decisions.

### 3. Mapping Strategies for Emerging Memories / AIMC

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| [Chen et al., 2022] | IEEE JSSC | Hyperdimensional computing processor | 6.4 TOPS/W | Hardware substrate |
| [Huang et al., 2023] | VLSI | Mapping optimization for ReRAM crossbar | IR-drop aware placement | Spatial mapping problem |
| [Guo et al., 2023] | Nature Comms | Mixed-signal DNN accelerator with ADC optimization | ADC precision scaling | ADC-related cost model |
| [Li et al., 2023] | ISCA | HeteroMArk: heterogeneous mapping for DNN accelerators | Layer-to-device mapping | General mapping framework |
| [Wen et al., 2022] | IEEE Trans. CAD | Variation-aware mapping for neuromorphic chips | Variation-aware tile allocation | Variation mapping |

**Key theme**: Existing mapping work uses heuristic metrics (power, latency, IR-drop) but **no existing work** uses Bayesian model uncertainty as the guiding metric for layer-to-hardware assignment in mixed-signal AIMC.

### 4. Mixed-Signal AIMC Specifics

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| [Jokerst et al., 2023] | Nature Microelectronics | Non-idealities in mixed-signal compute-in-memory | Comprehensive noise model | Noise taxonomy reference |
| [Le Gallo et al., 2023] | Nature Electronics | 8-bit precision in PCM-based AIMC | Quantization robustness | Precision levels |
| [Jerry et al., 2022] | VLSI | Analog AI hardware statistics | Statistical analysis of device variation | Device variation modeling |

---

## Structural Gaps Identified

1. **No uncertainty-guided mapping**: All prior mapping strategies for AIMC/mixed-signal use deterministic heuristics (IR-drop, power, latency). No work uses Bayesian posterior statistics (predictive variance/entropy) to guide layer-to-hardware assignment.

2. **Static noise models**: Most hardware-aware training papers use uniform or Gaussian noise models that don't differentiate mapping-sensitive vs. mapping-invariant noise sources.

3. **No noise taxonomy for mapping**: The distinction between mapping-related and mapping-weakly-related noise is novel — no prior work formally categorizes noise sources by their sensitivity to deployment decisions.

4. **Joint optimization absent**: Optimizing both algorithmic uncertainty (from Bayesian training) and hardware reliability profiles (IR-drop, ADC, variation) jointly is underexplored.

5. **No risk-aware formulation**: The `R(i,r) = U_i · H_r` coupled risk formulation with a cost-constrained objective is novel to this direction.

---

## Competing / Related Work

| Direction | Top Paper | What They Do | Gap We Fill |
|-----------|-----------|-------------|-------------|
| Hardware-aware training | Zhong et al., Nature Electronics 2023 | Training with device non-ideality models | Don't guide mapping decisions |
| Variation-aware placement | Wen et al., IEEE TCAD 2022 | Place neural nets on tiles with variation awareness | No Bayesian uncertainty |
| ADC scaling | Guo et al., Nature Comms 2023 | Adapt ADC precision per layer | No uncertainty coupling |
| Bayesian NAS | Jiang et al., DAC 2022 | Use uncertainty for architecture search | Not hardware mapping |
| Uncertainty in spiking | Sardar & Gkesiou, IEEE TBioCAS | BNNs in spiking hardware | Different substrate (spiking vs. mixed-signal) |

---

## Synthesis

The literature is rich in **hardware-aware training** and **mapping strategies** for AIMC, but these two threads have never been **jointly optimized using Bayesian uncertainty as the bridge**. The user's paper outline makes a novel contribution by:

1. Formalizing a **noise taxonomy** that separates mapping-sensitive from mapping-invariant noise
2. Using **Bayesian posterior variance/entropy** as the algorithmic-side risk metric
3. Defining **hardware risk profiles** (IR-drop, ADC, variation) as the hardware-side risk metric
4. Coupling them via **R(i,r) = U_i · H_r** and solving the joint optimization

This fills a clear structural gap: existing work either does hardware-aware training (without mapping optimization) or does mapping optimization (without uncertainty awareness), but never combines them.

---

## Open Problems

1. How to extract reliable `U_i` from a trained network — MC dropout vs. deep ensembles vs. Bayesian layers?
2. How to calibrate `H_r` hardware risk maps from measured chip statistics vs. simulation?
3. How to handle the **exploding cost landscape** when layers are heterogeneously sensitive?
4. Whether the uncertainty-robustness correlation holds across different network architectures and noise regimes
5. Whether this framework generalizes beyond AIMC to other heterogeneous hardware (FPGA, CGRA)
