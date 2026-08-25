# TIGeR: Honest Limitations & Future Work

No system is perfect, and peer reviewers expect a critical, honest appraisal of a system's boundaries. This document outlines the true limitations of TIGeR, which you can use to write a strong, scientifically mature "Limitations & Future Work" section in your paper.

---

### 1. Schema Rigidity and the Open-World Problem
**The Limitation:** TIGeR relies on a predefined `schema.yaml` to dictate what attributes exist and which are required (e.g., `color`, `material`). As demonstrated by our initial ABO run, if the pipeline encounters a product category that breaks these assumptions (e.g., trying to enforce a `color` requirement on a motherboard), the system will safely escalate, but it will fail to automate the repair.
**Why it matters:** TIGeR cannot dynamically discover or infer *new* attributes that are not in its schema.
**Future Work:** Future iterations could integrate an LLM to dynamically generate or adapt the schema by reading the raw textual descriptions in a zero-shot manner, bridging the gap between structured databases and unstructured text.

### 2. Generative Fallback Detail Loss
**The Limitation:** SDXL-Turbo is highly effective at generating macro-level replacements (e.g., generating a picture of a "red polo shirt"), rescuing 15 products from being complete dead-ends in our evaluation. However, diffusion models notoriously struggle with fine-grained pattern adherence. For example, if a caption says "Red and White Striped Checkered Shirt", the model may drop the pattern and generate a solid red shirt. 
**Why it matters:** The generated image might fulfill the schema's basic constraints to pass the VLM Judge, but it loses the high-fidelity commercial nuance of the original product.
**Future Work:** This validates our design decision to use text-to-vision (T2V) k-NN image swapping from the existing database *first*, using Generative Fallback strictly as a last resort. Future systems could employ control-nets or mask-based inpainting to preserve patterns.

### 3. Covariate Shift and Classifier Underconfidence
**The Limitation:** The Arbiter is a Logistic Regression model trained on a specific distribution of simulated noise. When evaluated on out-of-domain data (ABO), the model exhibited severe underconfidence, with over 75% of predictions falling below the default $\gamma=0.60$ threshold. 
**Why it matters:** If deployed blindly to a new domain without recalibration, the pipeline loses its automation capabilities, escalating almost every flagged item back to humans. 
**Future Work:** While we mitigated this via a lightweight 25th-percentile recalibration step, future architectures could replace the rigid Logistic Regression model with a more robust uncertainty-quantification mechanism, such as conformal prediction or Bayesian neural networks, to naturally adapt to shifting confidence distributions.

### 4. Dependency on Calibration Data Quality
**The Limitation:** Both the Sieve (for multimodal similarity thresholds) and the Arbiter (for routing probabilities) require a calibration dataset. TIGeR assumes that this calibration data (even when corrupted synthetically) is representative of the actual catalogue.
**Why it matters:** If a real-world catalog is *already* so massively corrupted that the "clean" baseline is noisy, the Sieve thresholds will become extremely wide, leading to false negatives (failing to flag actual anomalies). 
**Future Work:** Exploring unsupervised or self-supervised anomaly detection methods that do not require clean calibration splits would make the pipeline more resilient to heavily degraded starting states.
