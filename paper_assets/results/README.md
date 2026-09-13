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
committed copy is the **2026-09-12 re-run** (executed notebook:
`tiger_abo_d4_check.ipynb`), which is the same pipeline after `D4` landed and
with `D4`'s pass instrumentation added. It supersedes the 2026-09-12 01:28 run
in two ways and is otherwise identical to it:

- `repair_ablations_summary.csv` reports all five outcome statuses
  (`Repaired`/`Escalated`/`Dismissed`/`Acquire Image`/`Unrepaired`) rather than
  two, which is what closed `E8`'s unexplained 17-row denominator gap.
- `v2t_estimator_diagnostics.csv` has 738 rows rather than 289, because `B6`'s
  estimator-disagreement escalations are now captured. Those rows are the whole
  point of the report — a disagreement writes nothing, so before the fix every
  disagreeing row silently vanished from the file meant to attribute
  pixel-vs-probe error.

Ablation counts are unchanged between the two runs (Full System 268 repaired,
No Gamma Gate 498).

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

## What is still not here

`data/sample/` — the generated-image samples. The notebooks do not export it,
so it cannot be committed without a re-run that does. Tracked as `E6`.
