# TIGeR — Roadmap from here

Forward plan. The defect detail lives in `FIXES.md`; this is the ordering and
the critical path. Written 2026-09-10, after the measurement-harness pass.

**State (updated 2026-09-13):** 40 items done, 8 open, 0 parked, 2 blocked,
1 withdrawn, 1 discarded (52 total). Phase 4 (Section E, paper corrections) is
fully closed. D4 has hard proof it fires on real ABO data (98 rows reached
pass 2+, 6 got both a T2V and V2T repair). `C7` discarded — was never
applicable to this repo. The only genuinely open items left are: `B2`/`B3`/`B5`
(repair-accuracy, Phase 3 — prioritize over `B7`: real B0 data now shows the
pixel path, not the encoder, is the weaker estimator when they disagree —
11.1% vs 33.7% correct), `B4` (newly unblocked — real B0 data now exists to
calibrate against), `C5` (needs local data — partially done, thresholds
committed), `D1` (needs a trained model — one now exists), `D10` (regen needs
GPU, but Kaggle is available), `B1` (blocked, needs fashion imagery). 178
tests passing.

---

## The critical path, in one line

`Phase 1 (dataset) → Phase 2 (corrected run) → everything else`

Nothing in Phases 3–5 is worth starting before Phase 2 produces numbers. Nine
open items are literally or effectively blocked on it.

---

## Phase 0 — Measurement harness ✅ DONE

Section A closed apart from A2's model pin. The instrument that produces every
repair-side number was broken four separate ways; it now works, and the 73
restored tests plus 44 new ones pin it.

Consequence not yet acted on: **every number currently in `paper_assets/` was
produced by the broken harness.** They are not defensible until Phase 2 replaces
them.

---

## Phase 1 — Dataset migration ✅ DONE (`7541322`)

Verified end to end against the local ABO release: 11,839 products across 15
categories, balanced splits, 0 schema-invalid rows. 162 tests passing.

Two things the restored suite caught during this phase, both H10-class:
putting ABO types into `required_for_categories` on *raw* coverage (real
post-normalisation coverage is 45.5–92.8%, so requiring colour would flag every
clean-but-incomplete row as dirty), and dropping `shirts`, which would have
silently killed `attribute_drop` detection in the synthetic run.

Also removed a silent selection bias: the importer discarded any product whose
colour string did not resolve — 22% of the corpus — which meant `attribute_drop`
noise was the only way a row could ever lack a colour.

Decision taken: drop the Kaggle Myntra fashion set, use **official ABO**
(Collins et al., CVPR 2022) for both verticals.

| | Task |
|---|---|
| 1.1 | `CATEGORY_MAP` → two furnishing verticals. **A:** CHAIR, SOFA, TABLE, OTTOMAN, STOOL_SEATING, RUG, LAMP, LIGHT_FIXTURE, WALL_ART (~7.5k). **B:** FINERING, FINENECKLACEBRACELETANKLET, FINEEARRING, HANDBAG, SUITCASE, HAT (~5.9k). Explicit allowlist, not substring matching. |
| 1.2 | `schema.yaml`: category list, `required_for_categories`, and drop the two fashion-only size constraints (`shoes_have_numeric_sizes`, `apparel_letter_sizes`) — furniture has no size enum. |
| 1.3 | **Colour normalisation layer.** 273–1893 distinct free-text colour strings per type vs a 12-value Ω. Build against the observed distribution; `surface_forms` (D5) is the hook. Start on WALL_ART (21 distinct values) to validate. |
| 1.4 | Per-type cap. CELLULAR_PHONE_CASE is 64,853 items — 44% of the catalogue. Exclude, or subsample deliberately as an imbalance test for `reviewer_defense.md` Attack 9. |
| 1.5 | **D8** (category singularisation) moves onto the critical path: new categories need real nouns or every probe caption is malformed. |
| 1.6 | Delete `configs/config.yaml` — dead legacy MVP config, read by nothing, containing pre-fix values (F4's `copies_per_row: 15`, IQR thresholds, the flat noise model). It reads as live calibration and is a trap. |

**Why this vertical:** furniture and homeware give 56–88% material coverage on
real photos where wood, metal, glass and fabric look different. Fashion gave
`material_flip` recall of **0.200**, the weakest number in the paper. This makes
that signal measurable for the first time rather than a documented failure.

**Side effect:** with no fashion vertical, **B1 (skin counted as product colour)
stops being a defect that affects your evaluation.** Consider closing it as
out-of-scope rather than carrying it.

---

## Phase 2 — The corrected baseline run ⟵ everything waits here

Both notebooks are built and pushed.

| | Task |
|---|---|
| 2.1 | `tiger_corrected_run.ipynb` — synthetic catalogue → **detection** numbers. Retitled and narrowed: its Fashion half is gone, since both real-data verticals now come from ABO. The verified 0.267→0.853 probe result lives here. |
| 2.2 | `tiger_abo_corrected_run.ipynb` — the two ABO verticals → **repair** numbers. Phase 1 landed, so this is ready to run. |
| 2.3 | Record the manifest. γ, both seeds, the allowlist and both model IDs are written to `run_manifest.json` so C1 cannot recur. |

**Unblocks:** B4, E3, E8, and makes E1/E2/E4–E12 worth writing.

**Expect movement in an unpredictable direction.** Four independent defects fed
the old table and none biased it consistently.

**Dry-run update (2026-09-11):** first Kaggle attempt at both notebooks failed
before producing any usable numbers — `synthgen` crashed on the first
non-fashion category (`FIXES.md` F1), and ABO import's non-English-title
fallback crashed `calibrate` on a CLIP token-limit violation (`FIXES.md` F2).
Both fixed and tested (165 tests passing); `tiger_corrected_run.ipynb` also
picked up a missing `ablate-repair` cell it needed to produce repair numbers
at all. Both notebooks need a clean re-run from a fresh clone.

---

## Phase 3 — Repair accuracy (Section B)

The actual goal. Sequence from the B0 estimator-attribution report, which
Phase 2 produces — it says whether the pixel path or the encoder path is the
bottleneck, so do not guess.

| | Task | Status |
|---|---|---|
| 3.1 | **B2** — product localisation. The fixed central 70% crop is exactly wrong for furniture, where a rug, a lamp and a sofa occupy different regions. | UNBLOCKED — `data/raw/abo/` is local |
| 3.2 | **B5** — aspect ratio destroyed before cropping. Only 36.2% of ABO images are square (21–2871 px). Compounds B2. | UNBLOCKED |
| 3.3 | **B4** — calibrate `pixel_color_confidence` against actual correctness instead of gating a raw pixel share at 0.55. | after B0 run |
| 3.4 | **B3** — the white-discount rule at 85%. | TODO |
| 3.5 | **B7** — CLIP is measured at 62% on attribute binding vs BLIP 88%. `compare_encoders` already supports the swap. | TODO |

B2, B5 and B3 are PIL/NumPy on images already on disk — cheap locally, no GPU.

---

## Phase 4 — Paper corrections (Section E) ✅ DONE (2026-09-12)

All of Section E closed same day, once Phase 2 completed. See `FIXES.md`
E1–E12 for what each fix actually did; summary:

| | Task | Outcome |
|---|---|---|
| 4.1 | **E2** | Rewritten — no longer calls committed-wrong-value rows "safely escalated". |
| 4.2 | **E3 / E8** | §7.5's "cascading safety net" formally **withdrawn** — the corrected ABO run shows Full System and No Gamma Gate are genuinely different (268 vs. 498 repaired), so there was never an identity to prove. `_evaluate_run` now reports all 5 outcome statuses, not just 2. |
| 4.3 | **E9** | Ablation labels renamed ("Probes Only"/"No Probes"); `paper_concepts.md` §2 rewritten to credit contrastive probes, not LOO, for detection. Bonus catch: §1 had E1/E2 defined backwards vs. `arbiter.py` — fixed too. |
| 4.4 | **E1, E6, E7, E10, E11, E12** | Feature count fixed (14, not 4); README/`tiger_project_doc.md` dead paths corrected (`scripts/`, `kaggle_workflow.ipynb`, broken link); `data/thresholds/` actually committed (`data/sample/` still can't be — notebooks don't export it); verifier naming already consistent elsewhere, README's `--vlm-judge` example changed to `--independent`; line counts regenerated; planted `forced_gen_000` row disclosed in `honest_limitations.md`; swap_image granularity separated. |
| 4.5 | **E4, E5** | Attack 1 now leads with the MLLM-as-a-Judge reliability argument; Attack 6 answers the ARO objection directly (CLIP 62% vs. BLIP 88%) instead of dodging it. |
| 4.6 | Citation: ABO as `collins2022abo` (added to `related_work.bib`). **Licence discrepancy RESOLVED (2026-09-11):** checked both first-party sources directly — the dataset's own landing page (`amazon-berkeley-objects.s3.amazonaws.com/index.html`, published by Amazon) and the licence file bundled in the archive itself (`LICENSE-CC-BY-4.0.txt`) both state **CC BY 4.0** (commercial use permitted). The AWS Open Data registry listing (`registry.opendata.aws/amazon-berkeley-objects/`) is a third-party catalogue entry, not maintained by Amazon, and its "CC BY-NC 4.0" tag is stale/incorrect. **Cite CC BY 4.0.** Worth one sentence in the reproducibility section noting the registry's stale tag, pre-empting a reviewer who checks that page and gets confused. |

---

## Phase 5 — Parked design decisions ⚑

Not bugs. Each needs a decision before it needs a patch.

| | |
|---|---|
| **B6** | ✅ **BUILT 2026-09-11.** Nothing abstained on *value* uncertainty — the γ-gate abstains on routing, Eq. 27–29 on schema and similarity, but nothing checked whether the pixel estimator and the CLIP probe agreed. Now: a real disagreement between the two escalates instead of silently committing whichever the old confidence threshold favoured. See `FIXES.md` B6. Not yet sized against real data — that still needs the Phase 2 run. |
| **D4** | ✅ **BUILT 2026-09-12.** The two-pass loop never ran, so E3 had no behaviour distinct from E2. Sized against the real ABO run first: E3 = 239/1703 routed rows (~14%), not rare. Fix: an accepted repair now stays "pending" and re-enters the next pass instead of being marked "repaired" immediately — `repair.py`'s own docstring already described this as the design; the bug was one line excluding it. See `FIXES.md` D4. |

Both Phase 5 decisions are now closed and both re-run and verified on real
ABO data (2026-09-13) — see `FIXES.md` D4's hard-proof instrumentation note.

---

## Housekeeping (any time)

~~`A2` pin a verified Gemini model ID~~ · ~~`C1` γ consistency~~ ·
~~`C2` `--gamma` flag~~ · ~~`C4` hardcoded generated-image path~~ · `C5` commit the
evaluation artifacts · ~~`C6` requirements/pyproject divergence~~ ·
~~`C7` the 73 MB AWS installer in the repo root~~ (discarded — not applicable) ·
`D1` `class_weight="balanced"` vs the
calibration claim · ~~`D2` document the verifier's asymmetric failure handling~~ ·
~~`D9`, `D11`–`D13`~~ done · `D10` code fixed, regen needs GPU (Kaggle available).

---

## If you only do three things

1. **E2** — today, no run required. It is the claim most likely to be challenged.
2. **Phase 1 + Phase 2** — produce numbers that can be defended.
3. **B2 + B5** — cheap, local, no GPU, and they attack the component the whole
   pipeline is named after.
