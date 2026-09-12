# TIGeR: Core Research Concepts & Methodologies

This document outlines the core intellectual contributions and mathematical paradigms used in the TIGeR project, designed to be adapted directly into a research paper methodology section.

## 1. The Multimodal Error Taxonomy (E1-E4)
Traditional dataset curation treats noise as binary: data is either "clean" and kept, or "dirty" and discarded. This results in massive data loss.
TIGeR shifts the paradigm to **Dataset Repair** by establishing a fine-grained taxonomy of multimodal misalignment:
- **E1 (Text Fault)**: The image is correct, but the text contains typos, contradictions, or missing attributes. (Requires Vision-to-Text repair — read the image, fix the text).
- **E2 (Image Fault)**: The text accurately describes a product, but the wrong photo was associated with it. (Requires Text-to-Vision repair — read the text, fix the image).
  *(Corrected 2026-09-12: this document previously had E1/E2 swapped relative to `tiger/arbiter.py`'s authoritative convention — E1=text wrong/V2T, E2=image wrong/T2V, per its own docstring and `CLASSES`.)*
- **E3 (Dual Fault)**: Both modalities are corrupted or mismatched beyond salvage.
- **E4 (Ambiguous)**: The misalignment is too vague to resolve safely.

## 2. Detection: per-field contrastive probes, with LOO for field attribution

**Corrected 2026-09-12 (E9).** This section previously credited LOO Masking as
the detection contribution and reported "Adding LOO Masking: +X F1" as the
headline ablation number. That was a mis-credit: `tiger/eval/ablation.py`'s
`no_loo` config is actually per-field contrastive probes vs. everything
except them — Eq. 18 leave-one-out lives in `tiger/analyzer.py` and only runs
on rows *already* flagged, so it contributes nothing to detection at all. The
mechanism behind the project's strongest verified result — `mutate_text`
recall **0.267 → 0.853** — is the contrastive probes, not LOO. This correction
runs in the project's favour: the probe result is stronger and more novel
than the LOO framing gave it credit for.

**Per-field contrastive probes (detection):** for each attribute field
(colour, material, pattern), the sieve tests whether the image matches the
*declared* value better than every other value in that field's domain. This
catches subtle text mutations (e.g. "blue" → "red") that a single global
CLIP score misses — without probes, `mutate_text` recall is ~0.27; with them,
~0.85.

**Leave-One-Out (LOO) masking (field attribution for routing, not detection):**
once a row is already flagged, the system masks one attribute at a time from
the canonical text and re-scores with CLIP. If removing a token (e.g. "red")
causes similarity to jump, that field is the suspect — this is what tells the
Arbiter *which* field to patch (V2T), not what flagged the row in the first
place.
- **Z-Score Calculation**: the jump is measured as a z-score against the
  clean-calibration distribution, turning "this field looks suspicious" into
  a statistically grounded, pinpointed signal for routing.

## 3. The Strict-Precision Decision Fusion (Arbiter)
Automated repair systems risk corrupting clean data if they guess blindly (hallucination). TIGeR introduces a provably safe **Decision Fusion Arbiter**.
- **Architecture**: A Multinomial Logistic Regression Router that ingests 14 dimensions of multimodal evidence (LOO Z-scores, swap margins, pixel-level color checks, and k-NN consistency).
- **The Gamma (γ) Gate**: A configured confidence threshold (Eq. 22), read statically from `configs/tiger.yaml` (currently 0.40; overridable per run with `--gamma`, see `code_fixes/FIXES.md` C1/C2). If the Arbiter's predicted probability for the winning error class fails to beat γ, the row is marked E4 (ambiguous) and routed to human review rather than automated. *(Note: γ is not dynamically calibrated and is unrelated to the Sieve's separate fusion `precision_floor` (0.85) — an earlier draft of this document conflated the two; they are different mechanisms on different components.)*

## 4. Generative Fallback for Missing Modalities
Traditional curation pipelines fail when attempting to repair an image (E1) if a suitable replacement does not exist within the catalogue.
- **Mechanism**: TIGeR incorporates a closed-loop **Generative Fallback**. When the Candidate Pool fails to find a valid image swap, the system routes the canonical text to a diffusion model (Stable Diffusion XL Turbo) to dynamically synthesize the missing modality.
- **Impact**: This guarantees that the dataset can be algorithmically plugged and repaired even when 100% of the candidate visual data is corrupted or missing.

## 5. Independent Semantic Verification
Encoder-only models (like CLIP and SigLIP) suffer from "bag-of-words" blindness, often failing to recognize spatial relationships or deep semantic intent (e.g., swapping a left-facing shoe for a right-facing shoe).
- **Mechanism**: TIGeR employs an independent vision encoder (SigLIP) or Vision-Language Model (Gemini) as a final semantic safety checkpoint to audit repairs before they are committed to the dataset, catching "wrong-direction" repairs that pass primary encoder-only thresholds.
## 6. Compound AI System Architecture
TIGeR is not a single monolithic model; it is designed as a **Compound AI System** where four specialized models interact to create a closed-loop repair pipeline:
1. **The Evidence Gatherer (CLIP)**: A Vision-Language Encoder used to rapidly calculate baseline similarities and perform the LOO masking.
2. **The Router (Logistic Regression)**: The mathematical Arbiter. It ingests the evidence vectors from the encoder and routes the product to the correct repair strategy using the strict γ-gate.
3. **The Synthesizer (SDXL-Turbo)**: The Generative Fallback model that wakes up to draw mathematically perfect replacement images when the catalogue lacks a valid candidate.
4. **The Auditor (SigLIP / Gemini)**: The final semantic safety checkpoint. It audits the repaired image and text pair to ensure they logically align before committing the data to the database, catching edge-case hallucinations that primary encoder-only models miss.

## 7. Category-Specific Dynamic Thresholding (The Sieve)
A major flaw in naive dataset filtering is using a single, global similarity threshold across all data. 
- **The Problem**: Different product categories naturally have different baseline similarities. A "clean" photo of a simple white shirt will naturally have a higher CLIP score than a "clean" photo of a highly complex, multi-colored handbag. A global threshold would aggressively false-flag all the handbags.
- **The Solution**: TIGeR utilizes a pre-calibration Sieve. It scans a subset of the dataset to learn the natural Gaussian distribution of similarity scores for *each individual category*. It then establishes dynamic, category-specific thresholds, ensuring error detection is equally sensitive across both simple and complex domains.
