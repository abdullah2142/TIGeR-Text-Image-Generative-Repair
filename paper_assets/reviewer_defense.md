# TIGeR: Reviewer Defense & Rebuttal Guide

This document anticipates the most common and aggressive critiques from peer reviewers (e.g., at KDD, CVPR, ECCV) and provides scientifically sound, defensible rebuttals based on your experimental data.

---

### Attack 1: "Why not just use a massive, end-to-end Multimodal LLM (like GPT-4V or Gemini 1.5 Pro) to fix the whole catalog? Why build a complex 5-stage pipeline?"

**The Rebuttal (Reliability first, cost second):**
The durable argument here is reliability, not cost — cost arguments age badly as API prices fall and rate limits loosen, but the reliability gap does not (E4).
1. **VLM judges are not a trustworthy sole arbiter of correctness.** MLLM-as-a-Judge (Chen et al., ICML 2024) documents that VLM judges diverge from human judgment on absolute scoring and exhibit position bias, egocentric bias, length bias, and hallucination — *including in GPT-4V*. Using one VLM as the entire detect-diagnose-repair-verify pipeline means every one of those failure modes is unguarded; TIGeR's five-stage structure means no single model's blind spot is the last word on a repair.
2. **Cost and throughput are real, but secondary.** Processing a 100,000-item catalog with an API-based VLM end-to-end would cost thousands of dollars per audit; TIGeR achieves this with **$0 API cost** on a single consumer-grade GPU (T4) at **6,000 products/hour** (an 80x speedup over human curation) — worth stating, but not the reason to prefer this architecture over a strong future VLM.
3. **Targeted compute:** TIGeR uses cheap, fast models (CLIP) for the Sieve to instantly ignore the 70% of clean data, saving heavier compute (SDXL-Turbo, SigLIP) *only* for products that actually need it — an intelligent triage system, not a brute-force model.

---

### Attack 2: "Generative AI (SDXL) is prone to hallucinations. How can you safely use it to repair a factual database? Doesn't this corrupt the catalog further?"

**The Rebuttal (The Cascading Safety Net):**
We agree that generative models hallucinate, which is exactly why TIGeR **does not blindly trust them.** 
The generative fallback is strictly protected by a "cascading safety net":
1. **The Gamma Gate:** The Arbiter evaluates the confidence of the required repair. If confidence falls below the configured threshold γ, it is instantly escalated to a human. *(The specific share of rows this affects depends on γ and the domain's confidence distribution — see `code_fixes/FIXES.md` A3 — and is not a fixed "bottom 25%".)*
2. **The Independent Verifier:** Even if SDXL generates an image, it is not immediately committed to the database. An independent verifier (SigLIP) audits the generated image against the original text constraints. If SDXL hallucinated (e.g., generated a blue shirt when the text demanded red), SigLIP vetoes the repair and escalates it. 
Our ablation study quantifies this: on ABO the full system escalates **1,018** rows rather than forcing repairs on them, against 798 with the γ-gate disabled — the gate alone accounts for 220 additional escalations, and disabling it costs 9 points of colour restoration accuracy (0.519 → 0.432).

*Two corrections a reviewer may catch first.* The ablation row historically labelled "VLM Judge" ablates the **SigLIP encoder cross-check**, not a VLM — no VLM was involved in any reported number (`FIXES.md` E13). And the independent verifier's image-side check was, until recently, structurally incapable of rejecting anything: it asked whether the proposed image beat the old one on caption similarity, which is the criterion the candidate was *selected* to maximise, and it approved **304 of 304** swaps. It now requires the second encoder to also prefer the chosen image over the runner-up, and rejects **131 of 261** (`FIXES.md` D17). The safety net is real, but it was only made real by measuring it.

---

### Attack 3: "A 51.9% accuracy for text repairs (V2T) seems low. Is this system actually effective?"

**The Rebuttal (Triangulation vs. Random Guessing):**
In highly corrupted, real-world datasets, perfect accuracy is an unrealistic baseline. The goal of an applied AI system is to maximize automation while minimizing destructive actions.
1. **Baseline comparison.** Random routing achieves **0.000** colour restoration accuracy (0/3 committed); TIGeR's evidence-based Arbiter reaches **0.519** (41/79). Measured 2026-09-16 on ABO, seed 7 (`paper_assets/results/abo/repair_ablations_summary.csv`). The qualitative claim the earlier draft made — that structured routing dominates random routing — holds, and more starkly than the superseded 3.2% / 52.6% figures suggested: the random baseline does not merely score worse, it barely commits at all, because random routing rarely produces a repair that survives schema validation and verification.
2. **Escalation and repair error are not the same population.** 51.9% is restoration accuracy measured only among the rows TIGeR actually **committed** to the database (79 colour repairs in the 2026-09-16 ABO run). The **1,018** rows the Gamma Gate or Independent Verifier caught and escalated to a human are a *separate, already-excluded* population — they are not part of this 79-row denominator at all. The remaining 48.1% are rows TIGeR **did commit, and got wrong**: real errors written into the corrected dataset, not cases that were safely escalated. Escalation is a genuine safety feature and correctly caught routing- and schema-level uncertainty; at the time of the run these figures describe, it did not yet catch *value*-level uncertainty in a repair it had already decided to commit — the source of these committed errors (see `code_fixes/FIXES.md` finding B6). That gap has since been closed and **measured**: a repair whose two internal estimators (pixel colour vs. CLIP probe) disagree now escalates instead of committing silently. On the 2026-09-16 run, 36 rows disagreed and **none** were written. The effect is visible in the accuracy split by estimator state — rows where both estimators agreed are correct **56.8%** of the time, against **46.2%** where the pixel estimator declined and the probe decided alone. Escalating disagreements is what keeps the committed population in the higher band.

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

**The Rebuttal (Model-Agnostic Architecture — and yes, we know CLIP is the weak link):**
The primary contribution of TIGeR is not establishing the absolute performance ceiling of specific foundational models, but rather introducing a **novel system architecture** (Detect $\rightarrow$ Diagnose $\rightarrow$ Route $\rightarrow$ Repair $\rightarrow$ Verify).
This question has a sharper form worth answering directly: ARO (Yuksekgonul et al., ICLR 2023) measures CLIP at just **62%** on attribute-binding tasks — barely above the 50% chance floor — against BLIP at 88% and XVLM at 87%. TIGeR's per-field contrastive probes are exactly an attribute-binding task, so this is not a hypothetical concern; it plausibly sets the ceiling on the probe path today (see `code_fixes/FIXES.md` B7, and B0's estimator-attribution instrumentation, which decomposes exactly how much of the pipeline's error is attributable to this encoder choice versus the pixel-colour path). We do not treat this as disqualifying: `encoders.py` abstracts the embedding logic so CLIP is a swappable component, not an architectural commitment, and `compare_encoders` already supports substituting BLIP/XVLM or applying a post-hoc linear correction (Koishigarina et al., ICLR 2026) with no retraining. Separately, for the Independent Verifier stage specifically, we evaluated both Gemini (API-based) and SigLIP (local), selecting SigLIP for speed, no rate limits, and comparable empirical safety performance (see `project_chronicle.md`). Upgrading the probe encoder in the future plugs into the same framework and would only raise TIGeR's baseline performance, not require re-architecting it.

---

### Attack 7: "Your generative fallback (SDXL-Turbo) generates images purely from text. Why not use ControlNet or IP-Adapter to preserve the structure of the original image?"

**The Rebuttal (Fallback Semantics):**
The generative fallback is strictly invoked by the Arbiter when an image is either **missing entirely** or **fatally corrupted** (e.g., the text describes a shoe, but the image is a polo shirt). 
If the image is completely missing or belongs to a contradictory category, there is no valid structural information to preserve via a ControlNet. When valid structural information *does* exist (e.g., the category is correct but the color is wrong), TIGeR actively avoids generation. Instead, it prioritizes text-patching (V2T) or k-NN retrieval from the existing catalog (T2V) to guarantee fidelity. Generation is explicitly relegated to a last resort precisely to avoid hallucinating structures unnecessarily.

---

### Attack 8: "You used a simplistic Logistic Regression model for the Arbiter. Why not a more sophisticated Neural Network?"

**The Rebuttal (Interpretability and Probability Calibration):**
The Arbiter's job is not feature extraction (the CLIP models handle that). Its job is low-dimensional routing based on a compact, hand-engineered evidence vector — 14 features in total (`tiger/arbiter.py::FEATURES`: Eq. 18/19 signals, swap margins, pixel-agreement, per-field probe z-scores, and two text-only flags), not raw high-dimensional embeddings.
For an input space this low-dimensional and this structured, a deep neural network is highly prone to overfitting and, more importantly, suffers from **uncalibrated overconfidence** on out-of-distribution data. Logistic Regression was specifically chosen because it provides well-calibrated, monotonic probability distributions (`predict_proba`), which are mathematically required for our $\gamma$-gate (Eq. 22) to function reliably. Furthermore, the linear weights provide exact interpretability as to *why* a specific repair path was chosen, a critical requirement for enterprise data systems.

**Calibration is measured, not assumed.** The obvious objection to the paragraph above is that the router is trained with `class_weight="balanced"`, and reweighting the training objective is a standard way to *lose* calibration. On the held-out calibration seed of the ABO run (n = 1,318 rows; trained on seven seeds, evaluated on an eighth), expected calibration error is **0.027**, and at the $\gamma = 0.60$ decision boundary itself the router is under-confident by **0.040** — it slightly over-escalates rather than over-commits, which is the safe direction for a gate whose purpose is abstention. Every run now writes this number into the model file (`calibration_holdout` in `tiger_arbiter_model.json`) alongside the reliability curve, so the claim is re-checked rather than restated; the ABO figures are committed at `paper_assets/results/abo/arbiter_calibration.json`.

Removing the class weighting raises aggregate holdout accuracy (0.636 → 0.724) and leaves ECE essentially unchanged (0.026 → 0.025), so it is not a calibration trade at all — but it collapses E3 recall from **0.691 to 0.064**, predicting the mixed-error class 11 times where it occurs 94 times. The balancing is buying minority-class recall, not calibration, and the aggregate accuracy it costs is the price of not discarding the class the two-pass repair path exists to serve.

**And aggregate calibration is not sufficient — state this before a reviewer does.** The same run that reports ECE 0.027 overall has a **CLEAN head that is flat**: rows the router calls CLEAN are actually clean 68–83% of the time whether it states 0.55 or 0.92. A decision keyed to one class needs that class measured, and the aggregate hides it. That per-class failure is precisely why the dismiss path was measured at 39% precision and subsequently **disabled** (`FIXES.md` D19), and the calibration report now emits per-class figures so it cannot recur silently.

---

### Attack 9: "E-commerce datasets suffer from massive class imbalance (e.g., 80% shirts, 1% hats). How does TIGeR prevent minority classes from being swallowed or misrouted?"

**The Rebuttal (Constrained Search and Dynamic Thresholding):**
TIGeR mitigates class imbalance structurally at two different stages:
1. **Dynamic Sieve Thresholds:** The anomaly detection thresholds are not hardcoded; they are statistically derived from the baseline similarity distribution of the ingested data during the calibration phase. This ensures that minority classes with naturally lower cross-modal alignment do not trigger false positives.
2. **Schema-Constrained Retrieval:** During T2V (Text-to-Vision) repairs, the Candidate Pool is strictly constrained to $k$-Nearest Neighbors of the *same product category*. A corrupted hat can only ever retrieve candidate images from the "hat" subset of the database, completely immunizing the repair stage against majority-class dominance.
