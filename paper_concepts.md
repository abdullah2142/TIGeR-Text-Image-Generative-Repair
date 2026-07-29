# TIGeR: Core Research Concepts & Methodologies

This document outlines the core intellectual contributions and mathematical paradigms used in the TIGeR project, designed to be adapted directly into a research paper methodology section.

## 1. The Multimodal Error Taxonomy (E1-E4)
Traditional dataset curation treats noise as binary: data is either "clean" and kept, or "dirty" and discarded. This results in massive data loss.
TIGeR shifts the paradigm to **Dataset Repair** by establishing a fine-grained taxonomy of multimodal misalignment:
- **E1 (Image Fault)**: The text accurately describes a product, but the wrong photo was associated with it. (Requires Text-to-Vision repair).
- **E2 (Text Fault)**: The image is correct, but the text contains typos, contradictions, or missing attributes. (Requires Vision-to-Text repair).
- **E3 (Dual Fault)**: Both modalities are corrupted or mismatched beyond salvage.
- **E4 (Ambiguous)**: The misalignment is too vague to resolve safely.

## 2. Deep Evidence Gathering via LOO Masking
Global CLIP similarity scores (cosine similarity between an image and a full caption) are fragile and lack explainability. TIGeR introduces **Leave-One-Out (LOO) Masking** for dataset curation.
- **Mechanism**: The system systematically masks specific attributes (e.g., color, material) from the canonical text and recalculates the CLIP score.
- **Z-Score Calculation**: If the removal of a specific token (e.g., "red") causes a statistically massive jump in the similarity score, it mathematically proves that the token is the culprit. This transforms a vague "bad match" into actionable, pinpointed evidence.

## 3. The Strict-Precision Decision Fusion (Arbiter)
Automated repair systems risk corrupting clean data if they guess blindly (hallucination). TIGeR introduces a provably safe **Decision Fusion Arbiter**.
- **Architecture**: An XGBoost classifier that ingests 14 dimensions of multimodal evidence (LOO Z-scores, swap margins, pixel-level color checks, and k-NN consistency).
- **The Gamma (γ) Gate**: A dynamically calibrated confidence threshold. The system sweeps a holdout validation set to find the minimum confidence required to achieve an 85% precision floor. If the Arbiter's predicted probability for a repair (e.g., `P(E2) = 0.60`) fails to beat the strict γ-gate (e.g., `0.85`), the system refuses to automate the repair and flags it for human review.

## 4. Generative Fallback for Missing Modalities
Traditional curation pipelines fail when attempting to repair an image (E1) if a suitable replacement does not exist within the catalogue.
- **Mechanism**: TIGeR incorporates a closed-loop **Generative Fallback**. When the Candidate Pool fails to find a valid image swap, the system routes the canonical text to a diffusion model (Stable Diffusion v1.5) to dynamically synthesize the missing modality.
- **Impact**: This guarantees that the dataset can be algorithmically plugged and repaired even when 100% of the candidate visual data is corrupted or missing.

## 5. Independent Semantic Verification (VLM Judge)
Encoder-only models (like CLIP and SigLIP) suffer from "bag-of-words" blindness, often failing to recognize spatial relationships or deep semantic intent (e.g., swapping a left-facing shoe for a right-facing shoe).
- **Mechanism**: TIGeR employs an independent Vision-Language Model (Gemini) as a final semantic safety checkpoint to audit repairs before they are committed to the dataset, catching "wrong-direction" repairs that pass encoder-only thresholds.
## 6. Compound AI System Architecture
TIGeR is not a single monolithic model; it is designed as a **Compound AI System** where four specialized models interact to create a closed-loop repair pipeline:
1. **The Evidence Gatherer (CLIP / SigLIP)**: A Vision-Language Encoder used to rapidly calculate baseline similarities and perform the LOO masking.
2. **The Router (XGBoost)**: The mathematical Arbiter. It ingests the evidence vectors from the encoder and routes the product to the correct repair strategy using the strict γ-gate.
3. **The Synthesizer (Stable Diffusion v1.5)**: The Generative Fallback model that wakes up to draw mathematically perfect replacement images when the catalogue lacks a valid candidate.
4. **The Auditor (Gemini VLM Judge)**: The final semantic safety checkpoint. It audits the repaired image and text pair to ensure they logically align before committing the data to the database, catching edge-case hallucinations that encoder-only models miss.

## 7. Category-Specific Dynamic Thresholding (The Sieve)
A major flaw in naive dataset filtering is using a single, global similarity threshold across all data. 
- **The Problem**: Different product categories naturally have different baseline similarities. A "clean" photo of a simple white shirt will naturally have a higher CLIP score than a "clean" photo of a highly complex, multi-colored handbag. A global threshold would aggressively false-flag all the handbags.
- **The Solution**: TIGeR utilizes a pre-calibration Sieve. It scans a subset of the dataset to learn the natural Gaussian distribution of similarity scores for *each individual category*. It then establishes dynamic, category-specific thresholds, ensuring error detection is equally sensitive across both simple and complex domains.
