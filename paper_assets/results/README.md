# Committed run artifacts (C5)

Every repair- and detection-side number in `paper_assets/` traces to a Kaggle
run whose outputs are otherwise gitignored (`data/outputs/`). These are those
outputs — the summaries, not the caches. Embedding caches, per-seed `.npz`
arrays and the per-seed sieve/evidence dumps are deliberately excluded: they
are large, they regenerate from the notebook, and they are not what a reader
checking a number needs.

Both runs used the same configuration; `run_manifest.json` in each directory is
the authoritative record of it (γ, both seeds, the T2V category allowlist, the
precision floor, and both model IDs). It exists so `C1` — γ having four
different values across the repo — cannot recur silently.

## `synthetic/` — detection numbers

Produced by `tiger_corrected_run.ipynb` (Kaggle, 2026-09-12), the seeded
synthetic catalogue. Seeds 7–11 for the noise sweep, 1007–1014 for calibration.

| file | what it is |
|---|---|
| `detection_metrics_sweep.json` | per-seed confusion matrices, precision/recall/F1 — the detection table |
| `sweep_summary.csv` | the same sweep, aggregated |
| `ablations.json` | detection ablations (Probes Only / No Probes / …), see `E9` |
| `sieve_ablations_summary.csv` | sieve-stage ablations |
| `repair_ablations.json`, `repair_ablations_summary.csv` | the repair side on the synthetic catalogue |
| `v2t_estimator_diagnostics.csv` | `B0` estimator attribution: pixel vs. probe, per repaired row |
| `thresholds/` | the locked thresholds, arbiter model and calibrations that run produced |

## `abo/` — repair numbers

Produced by `tiger_abo_corrected_run.ipynb` on the two ABO verticals. The
committed copy is the **2026-09-13 run**, the first with the rebuilt colour
estimator (`B2`/`B3`/`B5`/`B8`). It supersedes the 2026-09-12 run, whose numbers
were produced by the estimator that measured a fixed central crop of a
square-squashed image.

What moved, Full System:

| | 2026-09-12 | 2026-09-13 |
|---|---|---|
| Colour accuracy | 0.364 (55 cases) | **0.406** (64 cases) |
| Attribute accuracy | 0.351 (57) | **0.394** (66) |
| Repaired | 268 | 276 |
| T2V accuracy | 0.379 (214) | 0.381 (215) |

Accuracy and coverage rose together, which is the direction that matters: the
estimator agreeing with the CLIP probe more often (24.8% → 29.2%) means fewer
rows escalate on estimator conflict, so more repairs are attempted *and* more
of them are right. T2V is flat, as expected — image repair does not consult the
colour estimator for its value.

`v2t_estimator_diagnostics.csv` gained a `pixel_region` column recording how
each estimate was obtained (`foreground` / `center_box` / `flooded`), which is
what makes `B4`'s calibration question answerable on the next run rather than
guessable.

Two earlier improvements carried forward from the 2026-09-12 run: the summary
reports all five outcome statuses rather than two (`E8`), and the estimator
report captures `B6`'s disagreement escalations rather than silently dropping
them (766 rows here, against 289 before that fix).

`arbiter_calibration.json` is derived rather than emitted by the run: it is the
`D1` calibration check on holdout seed 1014 (ECE, the signed gap at γ, per-class
recall) for the shipped balanced router and for an unbalanced refit on the same
seeds. `train-arbiter` now writes the same measurement into the model file as
`calibration_holdout` on every run.

## These predate the colour estimator's repair

Everything under `abo/` was produced before `B2`/`B3`/`B5`/`B8` fixed the V2T
pixel estimator, which on that very run was reading the studio ground rather
than the product (`gray`, `multicolour` and `white` were 73% of its output, at
37.1% / 1.7% / 13.8% correct). The files are an accurate record of that run and
the manifest says exactly what produced them — but they are not the system's
current behaviour, and the repair-side numbers should not be quoted as if they
were. See `honest_limitations.md` §6.

## Figures

`paper_figures/` holds two artifacts this run produced rather than carried:

- `qualitative_grid_final.png` — clean / corrupted / repaired triptychs, built
  by `tiger.cli qualitative-grid` from the run's own per-row frame and outcome
  log. Every earlier version of this file was a hand-made PNG that round-tripped
  through the results zip unchanged; see `code_fixes/FIXES.md` D10.
- `generation_pattern_panel_panel.png` — the generator asked directly, one
  render per pattern with everything else fixed. Needed because the generative
  fallback never fires on a catalogue this size (`paper_concepts.md` §5), so the
  pattern question cannot be answered as a byproduct of a repair run.

## What is still not here

`data/sample/` — the source catalogue and its images. The notebooks now export
the *generated* images (`data/sample/images/generated/`) into the results zip,
but not the imported catalogue, which is reconstructible from ABO via
`tiger.cli import-abo`. Tracked as `E6`.
