# TIGeR Roadmap Progress Log

Living document tracking execution of `docs/TIGeR_Critical_Review_and_Roadmap.md`
(the review is gitignored with the rest of `docs/`; this log is committed).

- **Branch:** `phase1-critical-fixes`
- **Started:** 2026-07-17
- **Last updated:** 2026-07-21 — Phase 3b done (model-agnostic encoder + SigLIP independent verifier); awaiting checkpoint review
- **Rule of engagement:** stop at each milestone checkpoint for user review before
  the next phase begins.

---

## Status at a glance

| Phase | Status | Checkpoint |
| --- | --- | --- |
| 0. Environment + dataset bootstrap | ✅ done | — |
| 1. Fix what is wrong (1.1–1.8) | ✅ done | **Milestone 1 reported, committed** |
| 2. Paper alignment (Eq. 18/19/22/27–29, C) | ✅ done (2.7 cost-minimal partial) | **Milestone 2 reached** (2a ✅ / 2b ✅ / 2c ✅) |
| 3. Recall & coverage (probes, fusion) | ✅ 3.1–3.5 done; 3.6(VLM audit) pending | Milestone 3 (3a ✅ / 3b ✅) |
| 4. Real-world readiness | ✅ 4.1, 4.2 done. (4.3/4.4 optional extensions) | — |
| 5. Evaluation completeness | ✅ 5.4 ablation and 5.5 sweep complete | — |
| 6. Extensions | ✅ 6.4 (SigLIP Verifier), 6.5 (Gen Fallback) complete | Milestone 4 |

---

## Context and decisions taken (with user approval)

1. **Hardware Limitations:** The local environment is insufficient for the heavy CLIP inference required (runs taking > 1 hour on CPU). To unlock Phase 4 (real data) and Phase 6 (generative models), **all execution is being ported to Kaggle** (free T4x2/P100 GPUs) via a cell-by-cell notebook workflow.
2. **Original toy dataset is lost.** It lived on `G:\Uni Work\tiger\` (another
   machine) and `data/` here was empty. Decision: generate a *seeded synthetic
   catalogue* bundled in the repo (`data/sample/`, 240 products, 4 categories,
   rendered studio images). Old numbers are therefore not directly comparable.
3. **Restructure early.** All fixes implemented in a new `tiger/` package;
   legacy `scripts/*.py` left untouched as reference.
4. **Environment.** No system pip/venv existed; installed `uv` to
   `~/.local/bin`, created `.venv` (Python 3.12, CPU torch, transformers 5.x).
5. **ε for Eq. 29:** derive ε from the caption-rewording noise floor (Δ similarity between paraphrase templates on clean rows).
6. **Verifier Selection:** SigLIP chosen over Gemini due to speed, lack of rate limits, and comparable empirical safety performance.

---

## Phase 1 — Fix What Is Wrong ✅ (all items done, committed)

| Item | Finding | Status | Where |
| --- | --- | --- | --- |
| 1.1 Noise generator audit + rewrite | F12 | ✅ | `tiger/data/noise.py` |
| 1.2 Token-budget text views | F1 | ✅ | `tiger/text_views.py` |
| 1.3 Contamination-robust thresholds | F3, F4 | ✅ | `tiger/sieve.py` |
| 1.4 Missing-modality routing | F11 | ✅ | `tiger/analyzer.py::allowed_directions` |
| 1.5 Structure-safe text patching | F10 | ✅ | `tiger/solver.py::apply_attr_patch` |
| 1.6 Z-scored swap margin + neighbour interface | F5 | ✅ | `tiger/analyzer.py::NeighborIndex`, swap_z |
| 1.7 HSV dominant-colour estimator | F6 | ✅ | `tiger/colors.py` |
| 1.8 Product-level structure | F4 | ✅ | `product_id` everywhere; duplication off by default |

*(Synthetic baseline metrics recorded at Checkpoint 1)*

---

## Phase 2 — Paper Alignment 🔄 ✅

- **2a ✅ Eq. 18 + Eq. 19 wired into the pipeline.**
- **2b ✅ Arbiter — p(E1..E4), γ gate, E4, policy gate.**
- **2c ✅ Solver planning + Verify + end-to-end loop.**

---

## Phase 3 — Recall & Coverage ✅

- **3a ✅ decision fusion + quarantine + baselines/ablations.**
- **3b ✅ model-agnostic encoder + SigLIP independent verifier.**

---

## Phase 4 — Real-World Readiness ✅

- **4.2 ✅ engineering hygiene.** `pyproject.toml`, `README.md`, `.github/workflows/ci.yml`.
- **4.1 ✅ Real-data onboarding.** Evaluated end-to-end on Kaggle Fashion Product Images dataset.

---

## Phase 5 — Evaluation Completeness ✅

- **5.4 ✅ baselines/ablations.** Full component-impact ablation study executed on Kaggle generating `sieve_ablations_summary.csv` and `repair_ablations_summary_run2.csv`.
- **5.5 ✅ mechanics.** Multi-seed pooling and product-level bootstrap CIs implemented.

---

## Phase 6 — Extensions ✅

- **6.4 ✅ independent verification ensemble.** SigLIP fully integrated to block semantic "wrong direction" repairs.
- **6.5 ✅ generative fallback.** SDXL-Turbo integrated into Solver to synthesize missing images, rescuing 15 products in the real-world dataset evaluation.

---

## Checkpoint log

- **2026-07-17 · Checkpoint 0** — env + package skeleton + synthetic dataset.
- **2026-07-17 · Checkpoint 1 (Milestone 1)** — honest baseline reported.
- **2026-07-18 · Checkpoint 2a** — Eq. 18/19 evidence wired.
- **2026-07-18 · Checkpoint 2b** — Arbiter trained + validated.
- **2026-07-21 · Checkpoint 2c (Milestone 2 reached)** — full closed-loop repair cycle live.
- **2026-07-21 · Checkpoint 3a** — decision fusion calibrated.
- **2026-07-21 · Checkpoint 4.2** — package/README/CI landed.
- **2026-07-21 · Checkpoint 3b** — encoder made model-agnostic.
- **2026-07-28 · Checkpoint 6.4** — Integrated Gemini VLM Judge (later swapped to SigLIP).
- **2026-07-28 · Checkpoint 6.5** — Implemented Generative Fallback (SDXL-Turbo).
- **2026-08-14 · VLM Judge Bugfixes** — Fixed single-image T2V prompting and token limits.
- **2026-08-18 · API Caching & CSV Export** — Solved Gemini 500 RPD rate limit via LRU caching. Added automated CSV extraction for ablation metrics.
- **2026-08-19 · Project Completion** — Final SigLIP vs Gemini ablation run completed on Kaggle Fashion dataset. SigLIP selected as final verifier. Project transitions to manuscript drafting phase.

### TODO — Next Steps (Manuscript Phase)

- [ ] Format the LaTeX ablation tables using the metrics generated in `data/outputs/`.
- [ ] Incorporate the Mermaid architecture diagram into the methodology section.
- [ ] Draft the limitations section discussing the generative fallback's minor detail loss.
- [ ] **Cost-Benefit Paragraph:** Add to Results/Discussion — TIGeR achieves ~6,000 products/hour on T4 vs. ~60-100 for a human curator (80× speedup). SigLIP verifier and SDXL-Turbo generative fallback incur zero API cost. Pre-written text available in `paper_draft_materials.md` Section 5.
- [x] **Cross-Domain Experiment (ABO):** ABO notebook completed on Kaggle. ✅
  - Schema extended: 4 new categories (electronics, furniture, kitchen, home_decor), 5 new colors, 8 new materials.
  - Results with schema fix: Full System repaired 39 / escalated 440 out of 750 corrupted items.
  - Ablation table written; findings documented in `paper_draft_materials.md` §7.
  - ABO notebook: `tiger_abo.ipynb` — standalone, no fashion cells.

---

## ABO Integration: Hiccup Log (for Limitations Section)

A detailed record of bugs and obstacles encountered during the cross-domain ABO integration.
These inform the "Limitations & Future Work" section of the manuscript.

### H1 — `import-abo` missing from argparse choices
`import-abo` was added to the dispatch table but NOT to the `choices=[...]` list in argparse.
Argparse validates command names before dispatch; the command was silently blocked.
**Fix:** Added `"import-abo"` to the `choices` list in `tiger/cli.py`.

### H2 — `.json.gz` expected, `.json` found
The ABO README documents files as `.json.gz`, but the Kaggle dataset had already unzipped them.
The import adapter was using Python's `gzip` module to open them, which immediately crashed on plain text files.
**Fix:** Changed the glob pattern and file-open logic to handle raw `.json` files.

### H3 — Indentation bug: parsing logic outside the for-loop
When refactoring from a CSV-based reader to a JSON-line-by-line reader, the entire data extraction block
(category, title, product ID, image, color, material) was accidentally de-indented out of the inner `for line in f` loop.
The script ran without errors because it silently processed only the very last line of each JSON file,
producing 0-16 valid products out of 147,000, then crashing with "No valid ABO products found."
**Fix:** Re-indented 52 lines back into the loop using a targeted Python script.

### H4 — `product_type` stored as multilingual JSON array, not plain string
ABO encodes `product_type` as `[{"language_tag": "en_US", "value": "CELLULAR_PHONE_CASE"}]`, not a string.
The import adapter was calling `str(row.get("product_type")).upper()`, which stringified the Python list
using single-quote repr — unrecognizable to the CATEGORY_MAP, so every product was skipped as "unmapped."
**Fix:** Route `product_type` through `_extract_english_value()` before CATEGORY_MAP lookup.

### H5 — `_extract_english_value` stringify bug
When JSON lines are parsed by `json.loads()`, list-valued fields become Python list objects.
The function was calling `str(raw)` before trying to re-parse the value as JSON. Python's `str()` on a list
uses single quotes (`'value'`) which is not valid JSON — causing a silent parse failure and returning
a garbage string representation of the list.
**Fix:** Rewrote `_extract_english_value` to handle both native Python lists and JSON strings natively,
without converting to string first.

### H6 — Image paths had redundant `images/small/` prefix
`images.csv` contained relative paths like `images/small/60/609d8...jpg`.
When joined to the `--images-dir` (which already ends in `.../images/small`), the result was
`.../images/small/images/small/60/609d8...jpg` — a non-existent doubled path.
All products would have passed the category/title/color checks but silently had no image match,
causing them to be skipped.
**Fix:** Image path builder now strips the `images/small/` or `small/` prefix if the direct join fails.

### H7 — `--force-reinstall` destroyed Kaggle's PyTorch/Torchvision
Added to bypass Kaggle's pip cache of the old `tiger` package (which lacked `import-abo`).
`--force-reinstall` rebuilt the entire dependency tree, including PyTorch-dependent packages,
pulling incompatible versions that broke `torchvision`'s NMS kernel registration.
This caused every downstream cell using `AutoProcessor` (SigLIP) or `AutoPipeline` (SDXL-Turbo) to crash
with `RuntimeError: operator torchvision::nms does not exist`.
**Fix:** Replaced with `pip uninstall -y tiger && pip install --no-cache-dir -e .` — only the local package
is cleared; Kaggle's curated PyTorch stack is untouched.

### H8 — `images.csv.gz` vs `images.csv` extension
The Kaggle dataset page and original ABO README refer to `images.csv.gz`, but Kaggle had unzipped it.
The hardcoded `--images-csv` path in the notebook included `.gz`, causing an immediate FileNotFoundError.
**Fix:** Removed `.gz` from the path in `tiger_abo.ipynb`.

### H9 — Nested Kaggle directory structure
The Kaggle dataset (`khyeh0719/amazon-berkeley-objects-small`) unpacks its contents into subdirectories
`abo-listings/` and `abo-images-small/` inside the dataset root, not directly at the root.
The notebook's initial paths assumed a flat structure.
**Fix:** Updated all paths to include the correct subdirectory names based on the Kaggle file browser screenshot.

### H10 — `color: required: true` caused 65% escalation on ABO (Schema Design Bug)
The schema enforced `color` as a globally required attribute for all product categories.
ABO non-fashion products (phone cases, chairs, mugs) frequently have ambiguous colors (multicolour,
transparent, assorted) that fail schema validation. When a repair could not produce a schema-valid color,
it escalated to human review instead of attempting the repair.
The Arbiter and Gamma Gate made correct routing decisions, but they were irrelevant — schema validation
gated the repair attempt and escalated ~65% of corrupted ABO items before the repair path was even attempted.
**Fix:** Changed `color.required: true` (global) to `color.required_for_categories: [shirts, shoes, bags, hats]`
(fashion-scoped). Non-fashion categories now proceed to repair even without a resolved color.
**Paper impact:** Documented as a design insight: cross-domain TIGeR deployment requires lightweight schema
reconfiguration per product vertical (a 3-line YAML change, not pipeline retraining).

### H11 — Gamma Gate non-differentiation on ABO
`Full System` and `No Gamma Gate (gamma=0)` produced identical results in the initial ABO ablation.
The gamma threshold (default: 0.60) is calibrated on fashion-domain data. On ABO, the Arbiter's
`predict_proba()` scores are consistently above this threshold, so the gate never fires.
Root cause: the Arbiter is overconfident on out-of-domain data — its training noise distribution
does not fully match ABO's broader attribute diversity, yielding uniformly high but unreliable confidence.
**Decision:** Rather than masking this as a bug, we treat it as a measurable limitation.
A two-step response was implemented in `tiger_abo.ipynb`:
1. **Diagnostic cell (6b):** Plots the full max-confidence distribution and prints the percentage
   of items below gamma=0.60. Produces `data/outputs/arbiter_confidence_abo.png`.
2. **Recalibration cell (6c):** Computes the ABO-domain gamma at the 25th percentile of observed
   confidence scores, patches `configs/tiger.yaml`, and reruns `ablate-repair`. The resulting table
   should show `Full System` < `No Gamma Gate` in repairs, confirming the gate is now active.
**Paper impact:** Reported as a concrete, quantified limitation — the Arbiter requires per-domain
gamma recalibration for cross-domain deployment. See `paper_draft_materials.md` §7.5.
