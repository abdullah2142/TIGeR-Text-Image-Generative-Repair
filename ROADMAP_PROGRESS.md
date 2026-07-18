# TIGeR Roadmap Progress Log

Living document tracking execution of `docs/TIGeR_Critical_Review_and_Roadmap.md`
(the review is gitignored with the rest of `docs/`; this log is committed).

- **Branch:** `phase1-critical-fixes`
- **Started:** 2026-07-17
- **Last updated:** 2026-07-17 — Phase 1 complete, awaiting go-ahead for Phase 2
- **Rule of engagement:** stop at each milestone checkpoint for user review before
  the next phase begins.

---

## Status at a glance

| Phase | Status | Checkpoint |
| --- | --- | --- |
| 0. Environment + dataset bootstrap | ✅ done | — |
| 1. Fix what is wrong (1.1–1.8) | ✅ done | **Milestone 1 reported, committed** |
| 2. Paper alignment (Eq. 18/19/22/27–29, C) | ⏸ NOT STARTED — awaiting user go-ahead | Milestone 2 |
| 3. Recall & coverage (probes, fusion) | 🔶 infrastructure landed early (see notes) | Milestone 3 |
| 4. Real-world readiness | 🔶 4.2 partially (tests); rest not started | — |
| 5. Evaluation completeness | 🔶 5.5 groundwork (seeds, product-level CIs) | — |
| 6. Extensions | ⛔ not started (VLM API key pending from user) | Milestone 4 |

---

## Context and decisions taken (with user approval)

1. **Original toy dataset is lost.** It lived on `G:\Uni Work\tiger\` (another
   machine) and `data/` here was empty. Decision: generate a *seeded synthetic
   catalogue* bundled in the repo (`data/sample/`, 240 products, 4 categories,
   rendered studio images). Old numbers are therefore not directly comparable —
   but the review established they weren't trustworthy anyway (F4/F12/F14).
2. **Restructure early.** All fixes implemented in a new `tiger/` package;
   legacy `scripts/*.py` left untouched as reference.
3. **VLM access.** User will provide an API key in `.env` later; VLM-dependent
   items (3.6 audits, 6.4 VLM judge) are gated until then.
4. **Environment.** No system pip/venv existed; installed `uv` to
   `~/.local/bin`, created `.venv` (Python 3.12, CPU torch, transformers 5.x).
5. **ε for Eq. 29 (proposed, not yet implemented):** derive ε from the
   caption-rewording noise floor (Δ similarity between paraphrase templates on
   clean rows) rather than an arbitrary constant. Flagged at Checkpoint 1; no
   objection raised so far. Will be revisited at Phase 2 start.

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

**F12 audit verdict (1.1):** confirmed. The legacy generator could label a row
`swap_image` while the swap silently failed (donor == self index guard only
skipped the copy, not the label; donors could be duplicates of the same product
carrying the identical image). The new injector forbids same-product donors,
verifies every injected row actually differs from its original, and hard-fails
on no-op noise. Audit CSV written per run (`data/processed/noise_audit_*.csv`).

**Also landed early (prerequisites pulled forward):**
- Schema/constraint set C (roadmap 2.4) — needed by 1.5. `configs/schema.yaml`
  + `tiger/schema.py`. Includes required-attribute rules (catches
  `attribute_drop` with 100% precision).
- Per-field contrastive probes with prompt ensembling (roadmap 3.1/3.2) —
  the colour probe is Phase 1 scope (F15) and the generalisation to
  material/pattern was nearly free. z-scored margins per category.
- Image-independent text checks (3.5): out-of-domain + title↔attr
  contradiction.
- Per-error-type recall, Wilson CIs, product-level bootstrap, multi-seed sweep
  (5.5 groundwork) — needed to report Milestone 1 honestly.
- Eq. 18/19 evidence code exists in `tiger/analyzer.py` but is **not wired**
  into any pipeline run (Phase 2 work; kept dormant per stop-gate).

### Milestone 1 — honest baseline (reported at Checkpoint 1)

Protocol: 240-product synthetic catalogue; product-level calibration/report
split (120/120); thresholds calibrated on the clean calibration split only and
locked; 5 noise seeds (7–11); ~30% row noise across 9 error subtypes incl. E3
mixed and subtle variants; product-level bootstrap CIs.

**Pooled over 5 seeds (600 rows / 120 products):**

| Metric | Value | 95% CI (product-level bootstrap) |
| --- | --- | --- |
| Precision | 0.793 ± 0.025 | [0.706, 0.880] |
| Recall | 0.924 ± 0.034 | [0.886, 0.958] |
| F1 | 0.853 ± 0.026 | [0.801, 0.904] |

**Recall by error type (vs the old broken baseline):**

| Error type | Recall now | Old |
| --- | --- | --- |
| swap_image | 0.975 | 0.274 |
| swap_image_same_category (subtle) | 0.950 | — |
| mutate_text (all) | 0.853 | **0.000** |
| — color_flip | 0.971 | 0 |
| — near_color_flip (subtle) | 0.800 | — |
| — title_contradiction | 1.000 | — |
| — attribute_drop | 1.000 | 0 |
| — material_flip | **0.200** | 0 |
| mixed (E3) | 1.000 | — |
| missing_image | 1.000 | — |

**Per-signal precision (pooled):** low_sim 0.911 · probe_color 0.842 ·
probe_pattern 0.857 · probe_material 0.727 · text_out_of_domain 1.000 ·
title_contradiction 1.000 · missing_image 1.000.

**Honest caveats (to carry into the paper):**
- Precision is no longer 1.000. The old 1.000 was an artifact of thresholds
  calibrated on contaminated data (F3): the fence tolerated the errors.
- material_flip recall 0.200 is a genuine CLIP capability limit on synthetic
  silhouettes — expected to improve on real photos, must be re-measured in 4.1.
- The review's core diagnosis is confirmed empirically: the global cosine
  alone catches swaps but not single-field edits; the per-field contrastive
  probe is what raised mutate_text recall from 0.

### Commits on `phase1-critical-fixes`

| Commit | Content |
| --- | --- |
| `bb932d1` | Package core: schema/Ω_j validator, token-budget captions, HSV colour, cached encoder |
| `0784681` | Synthetic catalogue generator + bundled `data/sample/` + self-verifying noise |
| `d946ef0` | Sieve (clean-calibrated locked thresholds, probes), analyzer primitives, solver, eval, CLI |
| `2c34dd8` | 38 unit tests pinning every Phase 1 fix as regression tests |

### How to reproduce

```bash
.venv/bin/python -m tiger.cli synthgen    # regenerate data/sample (seeded)
.venv/bin/python -m tiger.cli calibrate   # lock thresholds on clean calibration split
.venv/bin/python -m tiger.cli sweep       # 5-seed noise -> detect -> evaluate
.venv/bin/python -m pytest tests/ -q      # 38 tests
```

Outputs: `data/outputs/detection_metrics_sweep.json`,
`data/thresholds/tiger_locked_thresholds.json`, per-seed sieve parquets.

---

## Phase 2 — Paper Alignment ⏸ NOT STARTED (stop-gate)

Planned scope when unblocked (Checkpoint 2 = Milestone 2):

- 2.1 Wire Eq. 18 leave-one-out attribution (code staged in analyzer, unwired)
- 2.2 Wire Eq. 19 kNN neighbour evidence (code staged, unwired)
- 2.3 Calibrated p(E1..E4) + γ gate + explicit E4 state (Arbiter rewrite)
- 2.4 ✅ schema/constraints already landed (see Phase 1 notes)
- 2.5 Per-repair acceptance Eq. 27–29 with ε, rollback, re-route, 2-pass cap
- 2.6 T→V policy gate
- 2.7 Cost-minimal V→T repair
- Open design point: ε from caption-rewording noise floor (see decision #5)

## Phase 3 — Recall & Coverage 🔶 partially landed early

Done: 3.1 probes, 3.2 ensembling+category conditioning, 3.5 text-only checks
(all Phase-1-adjacent, see above). Remaining: 3.3 encoder upgrade path
(SigLIP), 3.4 formal decision-fusion calibration to a precision floor,
3.6 VLM precision audits (**blocked on API key**).

## Phase 4 — Real-World Readiness 🔶

Done: package structure, unit tests (4.2 partial). Remaining: ABO real-data
onboarding (4.1), robustness hardening (4.3), ANN index + scaling bench (4.4),
CI workflow, README.

## Phase 5 — Evaluation Completeness 🔶

Done: 5.5 mechanics (≥5 seeds, product-level bootstrap, calibration/report
split). Remaining: 5.1 repair-quality metrics with held-out originals (F14
protocol), 5.2 efficiency, 5.3 downstream NDCG, 5.4 baselines/ablations
(ablation interface already exists: `apply_thresholds(enabled_signals=...)`).

## Phase 6 — Extensions ⛔

Not started. 6.4 independent verification ensemble is Must for any deployment
claim; needs second encoder + VLM judge (key pending).

---

## Checkpoint log

- **2026-07-17 · Checkpoint 0** — env + package skeleton + synthetic dataset
  approved (user chose: synthetic data, restructure early, VLM key later).
- **2026-07-17 · Checkpoint 1 (Milestone 1)** — honest baseline reported
  (numbers above). User: commit to new branch `phase1-critical-fixes`; stop
  after Phase 1; do NOT start Phase 2 yet. Done.
- *(next)* **Checkpoint 2 (Milestone 2)** — pending user go-ahead.
