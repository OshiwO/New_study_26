**Uncertainty-Guided Robustness-Aware Mapping for Mixed-Signal Analog In-Memory Computing**


Analog in-memory computing (AIMC) offers significant gains in energy efficiency and throughput for deep neural network inference, but suffers from substantial hardware-induced non-idealities such as device mismatch, noise, and quantization errors. While prior work has focused on hardware-aware training or heuristic mapping strategies, these approaches do not explicitly leverage model-side uncertainty to guide deployment decisions.

In this work, we propose an uncertainty-guided robustness-aware mapping framework for mixed-signal AIMC systems. Our key idea is to reinterpret Bayesian posterior statistics—such as predictive variance and entropy—as deployment-time risk indicators, and jointly optimize them with hardware reliability profiles. We formulate the mapping process as a risk-aware optimization problem that assigns network layers or blocks to heterogeneous hardware resources (e.g., analog/digital units, ADC precision levels, or spatial regions) based on both algorithmic uncertainty and hardware-induced error costs.

Compared to conventional accuracy-aware or heuristic mapping strategies, the proposed approach achieves improved robustness under hardware non-idealities while maintaining competitive accuracy and resource efficiency. Extensive experiments on benchmark datasets (e.g., CIFAR-10 with ResNet architectures) using AIMC simulation frameworks demonstrate consistent gains in accuracy–robustness–cost trade-offs.

This work establishes a new connection between Bayesian uncertainty and hardware deployment, providing a principled framework for robustness-aware system-level optimization in emerging mixed-signal AI hardware.



#### 优化了什么

We focus on deployment-time mapping under hardware non-idealities that are directly affected by allocation decisions, including tile-dependent variation, IR-drop, and ADC-related errors. Other hardware noises are retained as background perturbations during evaluation but are not treated as optimization variables.


### Noise

## 3. Noise Taxonomy and Problem Formulation

### 3.1 Noise Taxonomy for Mapping-Aware Deployment

Analog in-memory computing (AIMC) systems are affected by various hardware non-idealities, including device variation, IR-drop, and quantization errors. However, not all noise sources equally impact deployment decisions. To enable principled mapping optimization, we categorize hardware non-idealities into two groups based on their sensitivity to mapping decisions.

#### Definition 1 (Mapping-Related Noise)

A noise source $n$ is defined as *mapping-related* if its impact on the network performance varies significantly across different mapping strategies. Formally, for two mapping policies $\pi_1$ and $\pi_2$, we define:

$$
\Delta_n(\pi_1, \pi_2) =
\left|
\mathbb{E}\big[\mathcal{L}_{hw} \mid n, \pi_1\big]
-
\mathbb{E}\big[\mathcal{L}_{hw} \mid n, \pi_2\big]
\right|
$$

If $\Delta_n(\pi_1, \pi_2)$ is large across candidate mappings, then $n \in \mathcal{N}_{map}$.

Typical mapping-related noise sources include:

- IR-drop and wire parasitics (spatially dependent)  
- ADC quantization and mismatch (resource-dependent)  
- Region-dependent device variation  
- Analog vs. digital execution path differences  

These noise sources are explicitly influenced by deployment decisions and are therefore incorporated into the optimization objective.

---

#### Definition 2 (Mapping-Weakly-Related Noise)

A noise source $n$ is defined as *mapping-weakly-related* if its impact on performance is relatively invariant across mapping strategies:

$$
\Delta_n(\pi_1, \pi_2) \approx 0
$$

Such noise sources affect overall accuracy but do not significantly alter the relative ranking of different mappings.

Examples include:

- Global read noise  
- Uniform programming noise  
- Background thermal noise  
- Long-term drift and retention effects  

These noise sources are not included in the optimization objective but are considered during final deployment evaluation.

---

#### Mapping Sensitivity Index (Optional)

To quantify the mapping sensitivity of each noise source, we define:

$$
S_n =
\frac{
\mathrm{Var}_{\pi}\left(
\mathbb{E}[\mathcal{L}_{hw} \mid \pi, n]
\right)
}{
\mathbb{E}_{\pi}\left(
\mathbb{E}[\mathcal{L}_{hw} \mid \pi, n]
\right) + \epsilon
}
$$

A noise source is classified as mapping-related if:

$$
S_n > \tau
$$

---

### 3.2 Problem Formulation

Given a neural network with $L$ layers, we define a mapping policy:

$$
\pi: i \rightarrow r
$$

which assigns layer $i$ to hardware resource $r \in \mathcal{R}$.

---

#### Algorithm-Side Uncertainty

We extract a layer-wise uncertainty map from a Bayesian model:

$$
U_i = \mathrm{Var}_{w \sim p(w|D)} \big[ f_i(x; w) \big]
$$

where $U_i$ quantifies the sensitivity or confidence of layer $i$.

---

#### Hardware-Side Risk Modeling

We define a hardware risk map over resources:

$$
H_r =
\alpha_{IR} \, h_r^{IR}
+
\alpha_{ADC} \, h_r^{ADC}
+
\alpha_{VAR} \, h_r^{VAR}
$$

where:

- $h_r^{IR}$: IR-drop induced error  
- $h_r^{ADC}$: ADC-related error (quantization, mismatch)  
- $h_r^{VAR}$: device/programming variation  

Only mapping-related noise sources are included in $H_r$.

---

#### Algorithm–Hardware Coupled Risk

We define the deployment risk of assigning layer $i$ to resource $r$ as:

$$
R(i, r) = U_i \cdot H_r
$$

---

#### Mapping Objective

The overall mapping objective is:

$$
J(\pi) =
\sum_{i=1}^{L}
U_i \cdot H_{\pi(i)}
+
\lambda \cdot \mathrm{Cost}(\pi)
$$

where:

- $\mathrm{Cost}(\pi)$: resource cost (energy, latency, ADC usage)  
- $\lambda$: trade-off coefficient  

---

### 3.3 Bayesian Optimization for Mapping

We parameterize the mapping policy as $\pi_\theta$, where $\theta$ controls allocation behavior.

The mapping rule is:

$$
\pi_\theta(i) =
\arg\min_{r \in \mathcal{R}}
\left[
U_i \cdot H_r(\theta_h)
+
\lambda \cdot C_r
\right]
$$

We then optimize:

$$
\theta^\star =
\arg\min_{\theta}
F(\theta)
$$

where the evaluation objective is:

$$
F(\theta) =
- A_{hw}(\pi_\theta)
+
\eta \cdot E(\pi_\theta)
$$

---

### 3.4 Deployment and Evaluation

After obtaining the optimal mapping:

$$
\pi^\star = \pi_{\theta^\star}
$$

we evaluate the system under both noise categories:

$$
\mathcal{N} =
\mathcal{N}_{map}
\cup
\mathcal{N}_{weak}
$$

This ensures robustness under realistic hardware conditions.