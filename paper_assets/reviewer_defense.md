# TIGeR: Reviewer Defense & Rebuttal Guide

This document anticipates the most common and aggressive critiques from peer reviewers (e.g., at KDD, CVPR, ECCV) and provides scientifically sound, defensible rebuttals based on your experimental data.

---

### Attack 1: "Why not just use a massive, end-to-end Multimodal LLM (like GPT-4V or Gemini 1.5 Pro) to fix the whole catalog? Why build a complex 5-stage pipeline?"

**The Rebuttal (Cost, Latency, and Scalability):**
While large VLMs have high reasoning capabilities, using them as a zero-shot, end-to-end solver for catalog maintenance is economically and computationally intractable. 
1. **Cost:** Processing a 100,000-item catalog with an API-based VLM would cost thousands of dollars per audit. TIGeR achieves this with **$0 API cost** by running locally on a single consumer-grade GPU (T4).
2. **Throughput:** API rate limits and massive inference times bottleneck VLMs. TIGeR achieves a throughput of **6,000 products per hour** (an 80x speedup over human curation). 
3. **Targeted Compute:** TIGeR uses cheap, fast models (CLIP) for the Sieve to instantly ignore the 70% of clean data, saving the heavy compute (SDXL-Turbo, SigLIP) *only* for the specific products that actually need it. TIGeR is an intelligent triage system, not a brute-force model.

---

### Attack 2: "Generative AI (SDXL) is prone to hallucinations. How can you safely use it to repair a factual database? Doesn't this corrupt the catalog further?"

**The Rebuttal (The Cascading Safety Net):**
We agree that generative models hallucinate, which is exactly why TIGeR **does not blindly trust them.** 
The generative fallback is strictly protected by a "cascading safety net":
1. **The Gamma Gate:** The Arbiter evaluates the confidence of the required repair. If the product is highly complex or ambiguous (the bottom 25% of confidence scores), it is instantly escalated to a human.
2. **The Independent Verifier:** Even if SDXL generates an image, it is not immediately committed to the database. An independent verifier (SigLIP) audits the generated image against the original text constraints. If SDXL hallucinated (e.g., generated a blue shirt when the text demanded red), SigLIP vetoes the repair and escalates it. 
Our ablation study proves this: the Gamma Gate and VLM Judge successfully and safely escalated 269 ambiguous items rather than forcing bad repairs.

---

### Attack 3: "A 52.6% accuracy for text repairs (V2T) seems low. Is this system actually effective?"

**The Rebuttal (Triangulation vs. Random Guessing):**
In highly corrupted, real-world datasets, perfect accuracy is an unrealistic baseline. The goal of an applied AI system is to maximize automation while minimizing destructive actions.
1. **Baseline Comparison:** Randomly routing repairs achieves only **3.2% accuracy**. TIGeR's structured, evidence-based Arbiter boosts this to **52.6%** — a massive +49.4% absolute improvement.
2. **The Goal is Safe Escalation:** The remaining 47.4% of cases were not "failures" that corrupted the database; they were caught by the schema validator or the independent verifier and **safely escalated to a human**. TIGeR is a triage system: it successfully automated half of the workload and safely quarantined the rest, saving hundreds of hours of manual labor.

---

### Attack 4: "You only tested on synthetic noise injected into the catalog. How do we know this works on actual, organic real-world noise?"

**The Rebuttal (Representative Noise Modeling):**
While the noise was synthetically injected, the *distribution* of that noise was explicitly modeled on organic e-commerce failure modes.
We injected 30% corruption, heavily weighted toward the most common human data-entry errors: complete image swaps (10%) and color attribute flips (6%). Subtler edge cases like missing images were weighted lower (1%), reflecting the fact that modern SQL databases usually enforce non-null constraints on image fields. By testing on a dataset with realistic, high-entropy corruption, we ensure the Sieve faces a statistically representative "haystack."

---

### Attack 5: "Your system requires a predefined schema (schema.yaml). Doesn't this limit its usefulness for open-world data?"

**The Rebuttal (Lightweight Adaptation over Structural Retraining):**
TIGeR is designed for enterprise databases, which are inherently schema-bound (e.g., SQL tables or strict JSON). Open-world, schema-less approaches are rarely used in production e-commerce. 
Furthermore, our cross-domain experiment on the Amazon Berkeley Objects (ABO) dataset proves that generalizing TIGeR to a completely new vertical (e.g., from fashion to electronics) does *not* require retraining the pipeline architecture. It simply requires a lightweight, 3-line configuration change (e.g., scoping the `color` requirement using `required_for_categories`). This proves the architecture is highly modular and adaptable.

---

### Attack 6: "Why didn't you use [New SOTA Model] instead of CLIP and SigLIP? The vision-language landscape moves too fast for these to be relevant."

**The Rebuttal (Model-Agnostic Architecture):**
The primary contribution of TIGeR is not establishing the absolute performance ceiling of specific foundational models, but rather introducing a **novel system architecture** (Detect $\rightarrow$ Diagnose $\rightarrow$ Route $\rightarrow$ Repair $\rightarrow$ Verify). 
Our implementation explicitly abstracts the embedding and verification logic (`encoders.py`, `vlm_judge.py`) so that models are perfectly swappable. We evaluated both Gemini (API-based) and SigLIP (local), ultimately selecting SigLIP to prove the pipeline operates effectively and securely entirely on open-source, local-first models. Upgrading the underlying embedding models in the future will seamlessly plug into the TIGeR framework and only raise its baseline performance.

---

### Attack 7: "Your generative fallback (SDXL-Turbo) generates images purely from text. Why not use ControlNet or IP-Adapter to preserve the structure of the original image?"

**The Rebuttal (Fallback Semantics):**
The generative fallback is strictly invoked by the Arbiter when an image is either **missing entirely** or **fatally corrupted** (e.g., the text describes a shoe, but the image is a polo shirt). 
If the image is completely missing or belongs to a contradictory category, there is no valid structural information to preserve via a ControlNet. When valid structural information *does* exist (e.g., the category is correct but the color is wrong), TIGeR actively avoids generation. Instead, it prioritizes text-patching (V2T) or k-NN retrieval from the existing catalog (T2V) to guarantee fidelity. Generation is explicitly relegated to a last resort precisely to avoid hallucinating structures unnecessarily.

---

### Attack 8: "You used a simplistic Logistic Regression model for the Arbiter. Why not a more sophisticated Neural Network?"

**The Rebuttal (Interpretability and Probability Calibration):**
The Arbiter’s job is not feature extraction (the CLIP models handle that). Its job is low-dimensional routing based on exactly four continuous evidence metrics (Eq. 18, Eq. 19, swap margins, pixel signals). 
For a 4-dimensional input space, a deep neural network is highly prone to overfitting and, more importantly, suffers from **uncalibrated overconfidence** on out-of-distribution data. Logistic Regression was specifically chosen because it provides well-calibrated, monotonic probability distributions (`predict_proba`), which are mathematically required for our $\gamma$-gate (Eq. 22) to function reliably. Furthermore, the linear weights provide exact interpretability as to *why* a specific repair path was chosen, a critical requirement for enterprise data systems.

---

### Attack 9: "E-commerce datasets suffer from massive class imbalance (e.g., 80% shirts, 1% hats). How does TIGeR prevent minority classes from being swallowed or misrouted?"

**The Rebuttal (Constrained Search and Dynamic Thresholding):**
TIGeR mitigates class imbalance structurally at two different stages:
1. **Dynamic Sieve Thresholds:** The anomaly detection thresholds are not hardcoded; they are statistically derived from the baseline similarity distribution of the ingested data during the calibration phase. This ensures that minority classes with naturally lower cross-modal alignment do not trigger false positives.
2. **Schema-Constrained Retrieval:** During T2V (Text-to-Vision) repairs, the Candidate Pool is strictly constrained to $k$-Nearest Neighbors of the *same product category*. A corrupted hat can only ever retrieve candidate images from the "hat" subset of the database, completely immunizing the repair stage against majority-class dominance.
