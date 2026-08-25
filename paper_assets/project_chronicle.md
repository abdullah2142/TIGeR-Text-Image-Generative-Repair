# TIGeR Project Chronicle: A Journey from Concept to Pipeline

This document serves as a comprehensive history of the TIGeR (Text-Image Generative Repair) project. It chronicles the decisions, hiccups, architectural pivots, and final outcomes that shaped the pipeline into a publication-ready system. Use this as a reference guide when writing the narrative arc of your research manuscript.

---

## 1. The Core Problem
E-commerce catalogues suffer from a silent disease: image-text misalignment. A vendor uploads a photo of a blue shirt but mistakenly tags it as "red." A database migration accidentally swaps the images of two different shoes. Traditional databases cannot catch these errors because the text itself is perfectly valid JSON/SQL; the anomaly only exists *between* the modalities. 

The goal of TIGeR was to build an autonomous, agentic pipeline that not only **detects** these cross-modal anomalies but automatically figures out **how to repair them** (either by patching the text or swapping the image) and rigorously **verifies** the fix before committing it.

---

## 2. Key Decisions & Model Selections

### The Sieve: Sticking with CLIP (Over SigLIP)
* **The Decision:** We kept OpenAI's CLIP (Base/32) as the primary multimodal encoder for the Phase 1 Sieve, despite testing Google's newer SigLIP model.
* **The Reasoning:** We conducted a head-to-head probe accuracy test on our synthetic data. While SigLIP is technically a newer architecture, it did not provide a statistically significant upgrade for our specific use case. Both models struggled identically with identifying "material" (e.g., leather vs. cotton), proving that the limitation was in the *visual data* (synthetic silhouettes lack texture), not the encoder. We stuck with CLIP because it is the industry-standard baseline for multimodal research, making our paper's results more comparable to existing literature.

### The Independent Verifier: SigLIP (Over Gemini)
* **The Decision:** For Phase 5 (Verification), we ultimately chose to use SigLIP as the final safety checkpoint instead of a heavyweight Vision-Language Model (VLM) like Gemini.
* **The Reasoning:** We built and tested both. Gemini was incredibly smart but introduced severe engineering bottlenecks: it was slow, required an API key, and strictly enforced a 500 Requests-Per-Day (RPD) rate limit. SigLIP runs locally, is completely free, processes thousands of images in seconds, and surprisingly achieved a marginally higher accuracy (+13.3%) than Gemini (+13.1%) on the Fashion dataset because it was less aggressively paranoid about escalating ambiguous cases.

### Generative Fallback: SDXL-Turbo
* **The Decision:** We integrated Stable Diffusion XL Turbo (SDXL-Turbo) for Phase 4's Generative Fallback.
* **The Reasoning:** When the pipeline needs to replace an image but the catalogue has no valid replacements (a "dead end"), we synthesize one. We chose SDXL-Turbo because it generates high-fidelity images in 1-4 steps, making it feasible to run on Kaggle's free T4 GPUs without causing timeouts. It provided a perfect balance of speed and photorealism.

---

## 3. Hiccups, Hurdles, and Resolutions

### Hiccup 1: The "Wrong Direction" Blindspot
* **The Problem:** Early in development, we noticed that CLIP would confidently approve terrible repairs. For example, if a blue hat was labeled "red hat," the system might "fix" it by swapping the image for a picture of a red shoe. CLIP approved this because the text ("red") now matched the new image (red shoe) better than before, completely ignoring that the category (hat vs shoe) was now wrong.
* **The Resolution:** This proved that encoder-only models (like CLIP) are blind to semantic "wrong direction" repairs. To fix this, we introduced the Phase 5 Independent Verifier (SigLIP) combined with strict Schema constraints (Eq. 27). The verifier specifically checks if the new repair still obeys the categorical rules, successfully blocking 28 bad repairs in the final run.

### Hiccup 2: VLM API Rate Limits
* **The Problem:** When running the full 450-item ablation study with the Gemini VLM Judge, we instantly hit the free tier's 500 RPD quota and crashed with `429 Quota Exceeded` errors.
* **The Resolution:** Instead of downgrading the model or paying for a higher tier, we engineered an elegant software solution: **Prompt Caching**. Because the ablation study runs the same dataset through 5 different configurations, it asks Gemini the exact same questions multiple times. We built an in-memory LRU cache that hashes the image+text prompt. This eliminated ~75% of duplicate network calls, bypassing the rate limit entirely and allowing the study to finish in one go.

### Hiccup 3: The Double-Image Confusion Bug
* **The Problem:** Initially, when asking Gemini to verify an image swap (T2V), we sent it *both* the old corrupted image and the new proposed image. Gemini got completely confused, returning verbose paragraphs of reasoning ("In the first image I see X, but in the second I see Y...") instead of a simple YES/NO.
* **The Resolution:** We rewrote the T2V logic in `vlm_judge.py` to only send the *proposed new image* against the text description. This simplified the VLM's task to a binary cross-modal alignment check, immediately fixing the parsing errors.

### Hiccup 4: Hardware Limits & The Kaggle Migration
* **The Problem:** The local development environment lacked the GPU VRAM necessary to run CLIP embedding loops on 1,500 real-world high-res fashion images, leading to runs taking over an hour.
* **The Resolution:** We migrated the entire execution pipeline to Kaggle, wrapping the CLI commands into a sequential `tiger.ipynb` notebook. This allowed us to leverage free T4x2 GPUs, dropping the execution time from hours to minutes and making the real-world scale evaluation possible.

---

## 4. Final Outcomes & Publication Readiness
The project culminated in a highly successful ablation study on the real-world Kaggle Fashion Product Images dataset. The final metrics validate the entire architecture:

1. **The Arbiter is Necessary:** Without the logistic router (relying on random guessing), the system destroyed data, achieving a dismal 3.2% accuracy. With the Arbiter, accuracy jumped by +49.4%.
2. **The Verifier is Necessary:** The SigLIP verifier caught the "wrong direction" semantic swaps, boosting overall repair accuracy by +13.3%.
3. **Generative Fallback works:** It successfully rescued 15 products that were complete dead-ends, generating studio-quality placeholders that passed the final safety checks.

The TIGeR pipeline stands as a complete, agentic solution to cross-modal anomaly detection and repair, combining deterministic schema logic, multi-modal embeddings, logistic routing, and generative AI into a single, cohesive architecture.
