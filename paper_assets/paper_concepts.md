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
- **The Gamma (γ) Gate**: A configured confidence threshold (Eq. 22), read statically from `configs/tiger.yaml` (default 0.40; overridable per run with `--gamma`, see `code_fixes/FIXES.md` C1/C2). If the Arbiter's predicted probability for the winning error class fails to beat γ, the row is marked E4 (ambiguous) and routed to human review rather than automated. *(Note: γ is not dynamically calibrated and is unrelated to the Sieve's separate fusion `precision_floor` (0.85) — an earlier draft of this document conflated the two; they are different mechanisms on different components.)*
  - **Which γ produced the reported numbers (corrected 2026-09-13).** The config default and the reported runs are not the same number, and quoting only the former misleads. **Every ABO result in `paper_assets/results/` was produced at γ = 0.60**, passed per-run and recorded in that run's `run_manifest.json`. Cite 0.60 with the results; cite 0.40 only as the repository default.
  - **The gate reads a probability, so the probability is measured.** The router is trained with `class_weight="balanced"`, which is a standard way to lose calibration, so the claim is checked rather than asserted: expected calibration error **0.023** on a held-out calibration seed (n = 1,653), with the router **under**-confident by 0.033 at γ itself — it over-escalates slightly, the safe direction for a gate whose purpose is abstention. Every training run now writes this into the model artifact as `calibration_holdout` (`code_fixes/FIXES.md` D1).

## 4. Repair-Value Estimation, and Abstention on Value Uncertainty

Sections 2 and 3 cover *detecting* a fault and *routing* it. Neither answers the
question the repair actually turns on: once the Arbiter says "the text is wrong"
(E1 → V2T), **what value should be written instead?** Routing can be perfect and
the written value still wrong. This is where restoration accuracy is won or
lost, and it is the part of the system with two independent estimators and an
explicit refusal to act when they conflict.

### 4.1 Two independent estimators

For a suspect attribute field, TIGeR derives the replacement value from two
mechanisms that share no machinery:

- **The pixel estimator** (`tiger/colors.py`) — deterministic arithmetic on the
  photograph, no learned parameters. It localises the product by *connectivity*
  rather than by position: the studio ground is flooded inward from the image
  border with a tolerance derived from the border's own variance, and the
  histogram is taken over what the flood cannot reach. Three outcomes are
  recorded per estimate — `foreground` (a ground was found and the product
  survived it), `center_box` (no uniform ground, as in a lifestyle shot, so a
  fixed central crop is used and the estimate is marked untrusted), and
  `flooded` (the flood reached everything, meaning product and ground are the
  same colour — a white product on white, where that colour *is* the answer).
  Applies to colour only.
- **The CLIP probe** (`tiger/sieve.py`) — the same contrastive instrument used
  for detection in §2, read for its argmax rather than its margin: score the
  image against one prompt-ensembled caption per value in the field's domain
  Ω_j and take the winner. Applies to colour, material and pattern.

The two are independent by construction, which is what makes their agreement
informative and their disagreement actionable.

### 4.2 Abstention on value uncertainty

The γ-gate (§3) abstains on *routing* uncertainty and the verifier (§6) abstains
on *outcome* uncertainty. Neither asks whether the system knows what value to
write. A confident-but-wrong estimate that nonetheless raises image-text
similarity is committed silently and passes every downstream check.

TIGeR therefore treats estimator conflict as a first-class abstention signal:

- **Agreement → repair.** Both estimators name the same value; write it.
- **Disagreement → escalate.** The row goes to human review rather than
  committing whichever estimator a threshold happened to favour.
- **Unlocalised → refuse the pixel path.** When the pixel estimator reports
  `center_box`, it measured a region it could not confirm contains the product;
  a high pixel share of an unknown region is not evidence, regardless of its
  magnitude.

This converts a class of silent wrong-writes into visible escalations, trading
coverage for precision on the acted-on set — the standard selective-prediction
trade, and the honest way to report it is a risk–coverage curve rather than a
point estimate.

### 4.3 Two-pass re-entry

An E3 row is corrupt in both modalities, so a single repair cannot finish it. An
accepted repair is therefore held as *pending* rather than immediately marked
repaired, and the row re-enters diagnosis on the next pass with its updated
image and text: an image swapped in pass 1 lets pass 2 re-examine the text
against the new image. A row is promoted to repaired only when a fresh sieve
pass finds it no longer flagged. On the reported ABO run 96 rows reached a
second pass and 8 received repairs in both directions.

### 4.4 What this costs, honestly

On the reported ABO run the two estimators agree on **29%** of candidate rows,
and of the repairs that agreement commits, **49%** write the correct value. The
system's abstention machinery is therefore doing real work — the disagreeing
71% are escalated rather than guessed — but agreement is not a guarantee of
correctness, and this should be stated rather than implied. The dominant
residual failure is vocabulary, not vision: `multicolour` is the most common
pixel verdict (28% of estimates) and is almost never the declared value,
because a twelve-value colour domain cannot describe a patterned rug or a wood
grain. That is a schema limitation, and it bounds what any estimator behind it
can achieve.

## 5. Generative Fallback for Missing Modalities
Traditional curation pipelines fail when attempting to repair an image (E2) if a suitable replacement does not exist within the catalogue.
- **Mechanism**: TIGeR incorporates a closed-loop **Generative Fallback**. When the Candidate Pool fails to find a valid image swap, the system routes the canonical text to a diffusion model (Stable Diffusion XL Turbo) to dynamically synthesize the missing modality.
- **Scope, corrected 2026-09-13.** This section previously claimed the fallback *"guarantees that the dataset can be algorithmically plugged and repaired even when 100% of the candidate visual data is corrupted or missing."* That is withdrawn: it describes a property that has never been exercised. **The fallback has never fired on either corpus.** Turning it off changes nothing — "Full System" and "No Generative Fallback" are identical in every column of both committed ablations. The reason is structural and correct: generation is reachable only when the Candidate Pool returns nothing, i.e. when no other usable image exists in the same category, and on a catalogue with thousands of rows per category that never happens. Retrieving a real photograph *should* beat synthesising one.
- **What can honestly be claimed**: the fallback is a **last-resort path for the sparse-category case**, present and tested, but not a demonstrated contribution of the reported runs. It is the mechanism that makes the pipeline total — every routed row has *some* available action — not a mechanism that carried any measured result.
- **Measured separately (D10).** Because the path never runs end-to-end, the generator was exercised directly: the same product rendered at `pattern` = solid / striped / dotted with everything else fixed (`paper_figures/generation_pattern_panel_panel.png`). Colour, material and the category noun all reach the image; **the pattern does not appear in any render**. This supports the pattern limitation in `honest_limitations.md` §2 — with the caveats recorded there: n = 3, a single category whose pattern may be semantically implausible, and SDXL-Turbo at 4 steps with `guidance_scale=0.0`, a configuration known to weaken prompt adherence.

## 6. Independent Semantic Verification
Encoder-only models (like CLIP and SigLIP) suffer from "bag-of-words" blindness, often failing to recognize spatial relationships or deep semantic intent (e.g., swapping a left-facing shoe for a right-facing shoe).
- **Mechanism**: TIGeR employs an independent vision encoder (SigLIP) or Vision-Language Model (Gemini) as a final semantic safety checkpoint to audit repairs before they are committed to the dataset, catching "wrong-direction" repairs that pass primary encoder-only thresholds.
## 7. Compound AI System Architecture
TIGeR is not a single monolithic model; it is designed as a **Compound AI System** where four specialized models interact to create a closed-loop repair pipeline:
1. **The Evidence Gatherer (CLIP)**: A Vision-Language Encoder used to rapidly calculate baseline similarities and perform the LOO masking.
2. **The Router (Logistic Regression)**: The mathematical Arbiter. It ingests the evidence vectors from the encoder and routes the product to the correct repair strategy using the strict γ-gate.
3. **The Synthesizer (SDXL-Turbo)**: The Generative Fallback model, invoked to draw a replacement image when the catalogue lacks any valid candidate. *(Corrected 2026-09-13: this line previously read "wakes up to draw mathematically perfect replacement images". Nothing about a diffusion sample is mathematically perfect, and on the reported runs it never woke up at all — see §5.)*
4. **The Auditor (SigLIP / Gemini)**: The final semantic safety checkpoint. It audits the repaired image and text pair to ensure they logically align before committing the data to the database, catching edge-case hallucinations that primary encoder-only models miss.

## 8. Category-Specific Dynamic Thresholding (The Sieve)
A major flaw in naive dataset filtering is using a single, global similarity threshold across all data. 
- **The Problem**: Different product categories naturally have different baseline similarities. A "clean" photo of a simple white shirt will naturally have a higher CLIP score than a "clean" photo of a highly complex, multi-colored handbag. A global threshold would aggressively false-flag all the handbags.
- **The Solution**: TIGeR utilizes a pre-calibration Sieve. It scans a subset of the dataset to learn the natural Gaussian distribution of similarity scores for *each individual category*. It then establishes dynamic, category-specific thresholds, ensuring error detection is equally sensitive across both simple and complex domains.
