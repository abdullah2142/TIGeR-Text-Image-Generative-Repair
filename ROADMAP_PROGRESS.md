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
- [ ] **Cross-Domain Experiment (ABO):** Run ABO notebook on Kaggle to demonstrate domain-agnosticism.
  - Duplicate `tiger.ipynb` → remove: synthgen, import-fashion, repair-demo, detection ablation, VLM ablation
  - Keep: setup, `import-abo`, `calibrate`, `train-arbiter`, `ablate-repair --independent`
  - Expected output: `data/outputs/repair_ablations_summary_run_abo.csv`
  - Compare side-by-side with `repair_ablations_summary_run2.csv` (Fashion baseline)
  - Schema extended: 4 new categories (electronics, furniture, kitchen, home_decor), 5 new colors, 8 new materials
