# TIGeR — Code Fix Backlog

Working document. One entry per defect, ordered so that fixes which *gate the
measurement of other fixes* land first.

**Status:** `TODO` · `DOING` · `DONE` · `BLOCKED` · `PARKED` · `WITHDRAWN`
**Marker:** ⚑ = changes the design, not just the numbers (see the ⚑ table below)
**Line references** are current as of branch `docs/fixes-backlog-audit`.
**Verified:** all 50 entries swept against the code on 2026-09-08. 41 held as
written; D3 withdrawn; B1/B2/B5 re-classified BLOCKED (no data in-repo can
exercise them); A2/A3/C3/C5/D12 corrected. Findings are recorded in-entry.

---

## Read this before starting

Two dependency traps:

1. **A4 must land before A1.** All five ablation configs share nested dicts via
   `cfg.copy()` (shallow). The moment A1 writes `cfg["arbiter"]["gamma"] = 0.0`,
   it mutates *every* config including Full System, and all five runs silently
   become γ=0. Fixing A1 first produces confidently wrong numbers.

2. **Section A gates Section B.** Every accuracy fix in B is measured by the
   ablation harness. If the harness is broken, you cannot tell which B fix
   helped. Land A first, re-run once to get a trustworthy baseline, then start B.

3. **D5 must land before D6 and D7.** Those two are the same normalisation
   defect, and both exist because D5 put five colours into the domain *and* into
   the alias map. Fixing the comparisons without fixing the domain leaves the
   probe candidates colliding.

Expected sequence: `A4 → A1 → A6 → A2 → A3 → A5`, plus `D4` and `D5 → D6/D7`
→ baseline run → `B0 (done) → B1…B6`.

`D4` and `D5` join the pre-baseline set because they change **pipeline
behaviour**, not just measurement: the numbers taken before them are not the
numbers the fixed system produces.

---

## ⚑ Three findings that change the design, not the numbers

Everything else in this file is a defect against a sound design — fix it and the
architecture stands. These three say the built system is not the described
system. Each needs a **decision** before it needs a patch.

The sweep split these into two kinds, which the first draft conflated.

**Changes the design — new behaviour that has never existed.**

| ⚑ | Finding | Entry | Why it is a design change |
|---|---|---|---|
| **1** | The repair operator had no ground-truth source, and nothing abstained on *value* | `B6` (with `B1`–`B4`, `E2`, `D6`) — **BUILT 2026-09-11** | The γ-gate abstains on routing; Eq. 27–29 abstains on schema and similarity. Nothing abstained on *"I do not know what colour this is."* A wrong-but-in-domain value that raises CLIP similarity used to pass every gate and be committed silently — the ~9 of 19 in `E2`. Now inserts an abstention: a real disagreement between the pixel and probe estimators escalates instead of committing. See `B6`'s entry for the mechanism. |
| **2** | The closed loop was not closed, and E3 had no behaviour of its own | `D4` — **BUILT 2026-09-12** | Four taxonomy classes, three implemented behaviours. Now: an accepted repair stays `pending` and re-enters the next pass, so a row needing both fixes actually gets both — the general re-diagnosis loop `repair.py`'s own docstring already claimed to have. |

**Changes a claim's scope — the architecture is untouched.**

| ⚑ | Finding | Entry | Why it is not a design change |
|---|---|---|---|
| **3** | Cross-domain generalisation was never tested for image repair | `A6` | The T2V policy gate is *designed* to be configurable. Widening `allowed_categories` is a one-line config edit. What moves is what RQ3 may claim, not the pipeline's shape. |

**Everything else in this file is architecture-preserving.** Sorted by blast
radius: docs/hygiene only (E1, E2, E4–E7, E10, E12, C6, C7, D2, D13); measurement
only (A1, A3, A3b, A4, A5, A8, C2, C3, C5, C8, E8); behaviour moves but the design
is intact (A2, A6, A7, C4, D5–D12, B1–B5, B7).

**Framing consequence.** What exists today is a *high-recall cross-modal error
detector with a calibrated triage layer, and a repair operator that is the
weakest component in the system.* That is a defensible paper on the committed
artifacts. The repair paper needs `B1`/`B2`/`B6` first. `reviewer_defense.md`
Attack 3 already reaches for the triage framing — it should be the accurate
description, not the fallback argument.

---

## A. Measurement correctness — blocks everything else

### A1 · The "No Gamma Gate" ablation never disables the gamma gate
**Severity:** Critical — invalidates a published result
**Where:** `tiger/eval/repair_ablation.py:191` writes · `tiger/arbiter.py:197` reads

```python
cfg_no_gamma["fusion"]["gamma"] = 0.0     # written here
gamma = float(acfg.get("gamma", 0.60))    # read from cfg["arbiter"], not cfg["fusion"]
```

There is no `fusion:` section in `configs/tiger.yaml` at all, so the γ=0 run
executes the **identical configuration** as Full System. This is why the rows
matched (163/269/52.6% on Fashion, 35/444 on ABO) — it is the same run twice.

**Consequence:** `paper_assets/paper_draft_materials.md` §7.5 — the "cascading
safety net / early exit" analysis — explains a phenomenon that never occurred.
The gamma gate has still never been ablated. That section must be withdrawn or
rewritten after a corrected run.

**Fix:** write to `cfg_no_gamma["arbiter"]["gamma"] = 0.0`. **Requires A4 first.**

**Status:** DONE — `2d29713`; regression tests pin both directions.

---

### A2 · VLM judge points at a non-existent model, and the failure mode is a silent veto
**Severity:** Critical
**Where:** `tiger/vlm_judge.py:116`, retry list at `:201`, veto branch at `:208-210`
**Verified:** confirmed — `404` matches none of `429/503/504`, so an invalid model
ID reaches `return False` at `:210`.

Default is `gemini-3.5-flash-lite`, which is not a real Gemini model ID (the
family runs 1.5 → 2.0 → 2.5 → 3). Several commits cycled through
`gemini-3.7-flash` and `gemini-3.1-pro`, which are also not real.

An invalid ID raises `404 NotFound`. That is not in the retry list
(`429/503/504`), so it falls to the fatal branch — which **returns `False`,
i.e. vetoes the repair**. A run with `--vlm-judge` would therefore veto 100% of
repairs while printing one error line per call.

**Fix:**
1. `genai.list_models()` with a live key to get valid IDs; pin a real one.
2. Add `404` / `NotFound` to a **fail-fast** branch — an invalid model is a
   configuration error and must raise at construction, not degrade to a veto.
3. Validate the model at `__init__` rather than on first call.

**Also determine:** whether any number in `paper_assets/` came from a
`--vlm-judge` run. If so it is unusable.

**Update (2026-09-11):** the working default has since moved to
`gemini-2.5-flash` (`tiger/vlm_judge.py`, current default). Verified live
against a fresh key via `genai.list_models()`: `gemini-2.5-flash` is a real,
reachable model ID for this key. Interestingly, `gemini-3.5-flash-lite` — the
original ID that 404'd on 2026-09-08 — **now also appears** in the model list;
Google's catalogue has evidently shipped it since the original finding. The
original finding was accurate for its date; it was overtaken by the model
catalogue moving, not a bug in this repo.

**Not required for the reported pipeline.** Per `project_chronicle.md`, the
Gemini-vs-SigLIP comparison was already run and decided: SigLIP was selected as
the reported Independent Verifier (faster, no rate limits, marginally higher
accuracy). `--vlm-judge` is an optional alternative path, not what the
corrected-run notebooks execute (they call `--independent`, i.e. SigLIP). This
verification was done for completeness/robustness, not because any reported
number depends on it — no re-run of the Gemini-vs-SigLIP comparison is planned
or required (see `E7`, which is a pure documentation-consistency fix, not a
re-measurement).

**Status:** DONE — `53736e6` fail-fasts on a misconfigured judge instead of
vetoing; default model ID confirmed live and working; not load-bearing for any
reported result.

---

### A3 · The random baseline is malformed
**Severity:** High — the 3.2% figure depends on it
**Where:** `tiger/eval/repair_ablation.py:27` · `tiger/arbiter.py:35`

`DummyArbiter` emits an `"E4"` key, but `CLASSES = ["E1","E2","E3","CLEAN"]`.
E4 is produced *by the gamma gate*, never predicted. When `E4` scores highest,
`route()` matches neither `CLEAN` nor `E1`, falls through to the final return,
and is labelled **E3/BOTH** — the final `return` hardcodes `"E3"` regardless of
what `top` was.

**Verified, with one correction:** the mechanism is confirmed, but "~25%" is
loose. Four normalised uniforms rarely produce a max ≥ γ, so the gamma gate
intercepts many rows *before* the E4 fallthrough is reached. The share of routes
affected depends on γ and cannot be stated without a run.

It also hardcodes `"CLEAN": 0.0`, so the baseline can never dismiss a row —
it is not a uniform random baseline over the real outcome space.

**Fix:** sample over the four real classes `["E1","E2","E3","CLEAN"]`; drop the
`E4` key; seed the RNG (see A3b).

**Status:** DONE — `01c469a`.

---

### A3b · `DummyArbiter` is unseeded
**Severity:** High — "No Arbiter" is non-reproducible between runs
**Where:** `tiger/eval/repair_ablation.py:30` (`import random` inside `predict_proba`)

Uses the global `random` module with no seed, so the random-routing row changes
every run and cannot be reproduced for the paper.

**Fix:** hold a `random.Random(seed)` instance on the class; take the seed from
config so it is recorded with the run.

**Status:** DONE — `01c469a`; seed from `cfg["eval"]["random_baseline_seed"]`.

---

### A4 · `cfg.copy()` is shallow — the trap that breaks A1
**Severity:** High (blocking)
**Where:** `tiger/eval/repair_ablation.py:149, 158, 188`

All five ablation configs share the same nested dicts. Nothing leaks *today*
only because the gamma write creates a fresh `fusion` dict that nobody reads.
Correcting A1 without fixing this mutates every config at once.

**Fix:** `import copy` → `copy.deepcopy(cfg)` at all three sites.
**Land this before A1.**

**Status:** DONE — `d6c4936`, landed before A1 as required.

---

### A5 · "Restoration Accuracy" measures colour only, and never scores image repairs
**Severity:** High — the column name overstates what is measured
**Where:** `tiger/eval/repair_ablation.py:83`

```python
for _, r in audit[audit["field"] == "color"].iterrows():
```

Ground truth is built solely from colour rows. Consequences:
- V2T patches to `material` / `pattern` are committed but never scored.
- **T2V (image) repairs are never scored at all** — the ablation says nothing
  about image repair quality.
- The headline is therefore colour-patch accuracy on N=19, not "restoration".

**Fix:**
1. Build ground truth for every audited field, not just colour.
2. Add a separate T2V correctness metric (did the swapped image come from the
   originally-correct product?) — the audit log already records the swap.
3. Rename the column to what it measures, and report N per metric.

**Sweep finding — step 2 as written is unachievable.** `noise.swap_image`
COPIES a donor path over the row's own (`tiger/data/noise.py:158-178`), leaving
the row's true original referenced by no row; `CandidatePool` holds in-use paths
only (`tiger/solver.py:135-141`). The original is absent from the pool *by
design* — that is the F14 held-out protocol applied to the operator. Scoring
recovery of it would report 0% forever.

Implemented instead: a T2V repair counts as restored when the image it installs
depicts a product matching this row's true category and colour
(`image_provenance`). Generated images depict no catalogue product and count as
attempts but never successes — the conservative reading. Steps 1 and 3 landed
as specified.

**Status:** DONE — `804ed68`, with one deviation; see the sweep note above.

---

### A6 ⚑ · The T2V policy allowlist disabled image repair for the whole ABO run
**Severity:** Critical — a cross-domain claim rests on a path that never executed
**Where:** `tiger/arbiter.py:242` reads · `configs/tiger.yaml:73` sets

```python
cat_ok = ev.get("category") in (policy.get("allowed_categories") or [ev.get("category")])
if "T2V" not in allowed_dirs or not cat_ok:
    return Route(..., "HUMAN", "human_review", 3, f"{top} but T2V blocked by policy/modality")
```

`allowed_categories` is `["shirts","shoes","bags","hats"]`. ABO imports map to
`electronics / furniture / kitchen / home_decor` (`tiger/data/import_abo.py:33`).
Every ABO E2 and E3 row was force-escalated by a fashion-only allowlist **before**
schema validation or the gamma gate applied.

**Consequence:** RQ3 — "does TIGeR generalise to a new vertical via lightweight
schema adaptation?" — is unanswered for image repair. H10 blames the escalation
rate on the `color` requirement and H11 on Arbiter underconfidence; this is a
third cause, upstream of both, and unacknowledged in every document. It also
interacts with A1: on ABO, correcting the gamma wiring still cannot move E2/E3
outcomes while this gate is closed.

**Fix:** extend `allowed_categories` to the ABO verticals, or make the policy
schema-driven per domain; echo the effective policy into the run output.
**Land with A1, before the corrected ABO re-run** — otherwise the new numbers get
read the same wrong way.

**Status:** DONE — `a989d10`.

---

### A7 · Precision-floor fusion is never loaded by the live pipeline
**Severity:** High — the headline precision is not the operating point
**Where:** `tiger/cli.py:195` (`cmd_detect`) · `tiger/repair.py:74` (`run_repair_cycle`)

Both call `sieve_mod.apply_thresholds(sig, thr)` with no `fusion=` argument.
`data/thresholds/tiger_fusion.json` is read by `cmd_ablate` and nothing else.

**Consequence:** the advertised **P=0.888 / R=0.882 / F1=0.885** is an offline
ablation row. `detect`, `analyze`, `route`, `repair` and every repair-side number
run on the un-fused detector at **P=0.793 / R=0.924**. `tiger_project_doc.md` §4
presents the fused figure as the Sieve's output.

**Fix:** load the fusion config in `cmd_detect` and `run_repair_cycle` when it
exists, behind an explicit flag, and state per reported number which operating
point produced it. Note fusion trades recall for precision (mutate_text recall
0.853 → 0.773), so enabling it changes what reaches the repair stage.

**Status:** DONE — `2d0690c`; opt-in via `--fusion`.

---

### A8 · The per-signal precision floor measures row dirtiness, not signal correctness
**Severity:** Medium — the 0.85 floor is weaker than it reads
**Where:** `tiger/fusion.py:72`

```python
prec = float(dirty[fired].mean())
```

A row counts as a true positive for a probe whenever the row is dirty **for any
reason**. The material probe firing on a `swap_image` row scores as a hit. The
floor therefore certifies "this signal fires on dirty rows", not "this signal
identifies the error it names" — which is what the precision-floor claim needs.

**Fix:** score each probe against the subtype it claims to detect
(`noise_subtype` is already in the frame) and report both figures. The joint
number stays meaningful for OR-fusion; the per-signal number is the one the
paper's claim rests on.

**Status:** DONE — `233ba2d`.

---

## B. Repair accuracy — the actual goal

The architecture claims *"the image is ground truth; read the true value from
it."* In practice every V2T value comes from `solver._corrected_value`
(`tiger/solver.py:164`), which returns either the HSV histogram estimate or the
CLIP probe argmax. **Neither estimator localises the product.** Routing can be
perfect and the written value will still be wrong roughly half the time.

### B0 · Estimator attribution instrumentation
**Status:** ✅ **DONE** — commit `8ba8d19`

Records both estimator candidates on every repair plus per-estimator
counterfactuals, so operator error can be attributed to the pixel path or the
CLIP path. Emits `data/outputs/v2t_estimator_diagnostics.csv` and a printed
report. Chosen value is bit-identical to previous behaviour.

**Run this before B1–B6** — it tells you which estimator to fix first and the
ceiling any selection rule can reach.

---

### B1 · Skin is counted as product colour
**Severity:** High — systematic bias on fashion photography
**Where:** `tiger/colors.py:26` (`HUE_RANGES`), brown rule inside `_hue_to_name`

`orange` is hue 14–40°, and `brown` is hue < 50° with `v<0.6, s>0.2`. Human skin
sits squarely inside both. On short-sleeved, sleeveless, or full-body model
shots, exposed arms/face/legs can dominate the sampled region — biasing the
estimate toward orange/brown.

**Fix:** mask skin-tone pixels before histogramming (standard HSV skin range,
roughly hue 5–35° with bounded saturation/value), then renormalise. Cheap, and
expected to be one of the larger single wins on Myntra-style imagery.

**Sweep finding — blocked on data.** The mechanism is confirmed (skin hue sits
inside both `orange` 14–40° and the `brown` rule's <50°), but the magnitude
cannot be measured here.
**Correction (2026-09-10): partially unblocked.** The claim that "the Fashion and
ABO datasets are not in the repo" was wrong. The full official ABO release is on
the dev machine at `data/raw/abo/` -- 398,212 images (matching the official
count) and all 16 listings files. `data/raw/` is gitignored, which is why it was
missed; not in the repo is not the same as not available.

B1 specifically remains hard to measure on ABO: it is a furniture, electronics
and homeware catalogue, so model shots with exposed skin are rare. The mechanism
is real but ABO is the wrong corpus to size it on. This one still wants fashion
imagery.
 `synthgen.render_product_image` (`tiger/data/synthgen.py:122`)
draws a flat-fill polygon on a 238–250 grey ground: no skin, no models. The
Fashion and ABO datasets are not in the repo (C5/E6). Changing the estimator with
no data that exercises the failure is editing blind.

**Two updates from the B2 work (2026-09-13), both of which make this worse
rather than better:**

**1. B2's localisation increases the exposure.** The old central 70% box on a
full-body model shot mostly sampled the torso, i.e. the garment. The
background flood keeps *everything* the studio ground cannot reach — arms,
legs and face included. On fashion imagery the new estimator therefore samples
strictly more skin than the one this defect was written against. It changes
nothing today (no fashion vertical), but the severity is higher than recorded
if fashion ever returns.

**2. The stated fix cannot be applied unconditionally — it would delete
wooden furniture.** "Mask hue 5–35° with bounded saturation/value" is the
standard recipe, and on this corpus that range is *wood*:

| | hue | s | v |
|---|---|---|---|
| light skin | 35.7° | 0.48 | 0.95 |
| mid skin | 33.8° | 0.53 | 0.88 |
| deep skin | 28.0° | 0.74 | 0.55 |
| oak | 32.8° | 0.45 | 0.76 |
| walnut | 30.0° | 0.67 | 0.40 |
| synthgen `brown` | 26.5° | 0.62 | 0.49 |

They are not merely adjacent, they interleave. A skin mask on the ABO
furnishing vertical would remove the product from wooden chairs, tables and
stools — the categories where `material` coverage is best and where B2's
localisation just started paying off. So the fix has to be category-conditional
(apply on apparel worn by a model, never on furniture), which makes it a larger
change than "mask and renormalise", and it still cannot be validated without
the imagery.

**Status:** BLOCKED on fashion imagery specifically — ABO is available but is
the wrong corpus for a skin-tone effect, and is actively the wrong corpus to
apply the fix to

---

### B2 · No product localisation — the estimator measures the wrong object
**Severity:** High — affects 3 of 4 categories
**Where:** `tiger/colors.py:80` (fixed central 70% box)

The central box is the model's torso regardless of what the product is. For
**hats** (top of frame), **shoes** (bottom), and **bags** (held to the side),
the estimator is measuring a different object entirely.

**Fix, cheapest first:**
1. Category-conditioned crop regions (hats → upper third, shoes → lower third).
2. Background removal / saliency to isolate the foreground object.
3. CLIP/SigLIP patch-level attention to localise the described product.

**Sweep finding — blocked on data.** Confirmed in code (`lo, hi = 0.15, 0.85`),
but unfalsifiable on the only committed dataset: synthgen centres every shape
within ±4% of frame centre at 36–44% scale, so the central box is *correct* there.
A regression on the synthetic set would prove nothing either way.

**Correction (2026-09-10): partially unblocked.** The claim that "the Fashion and
ABO datasets are not in the repo" was wrong. The full official ABO release is on
the dev machine at `data/raw/abo/` -- 398,212 images (matching the official
count) and all 16 listings files. `data/raw/` is gitignored, which is why it was
missed; not in the repo is not the same as not available.

**B2 is now testable and worth doing.** ABO is real product photography where the
subject is not reliably centred or frame-filling -- a chair, a phone case and a
rug occupy very different regions. The fixed central 70% box is exactly the wrong
assumption for that corpus, and 398k images are available to measure it against.

**Correction (2026-09-13): `data/raw/abo/` is gone from this machine.** The raw
release was deleted after the Kaggle runs (the notebook itself does
`rm -rf /kaggle/working/abo` to free 6 GB, and the local copy went the same
way). ABO images now live only on Kaggle. Sizing on real photography therefore
needs a re-run; the fix itself does not.

**Fix implemented (option 2, background removal).** `tiger/colors.py` now
localises the product by connectivity instead of by position: the studio ground
is flooded inward from the image border and what the flood cannot reach is the
product. The growth tolerance is derived from the border's *own* spread
(`3 × median border distance`, clamped to 0.04–0.18) so a flat ground stays
tight enough not to cross a low-contrast product edge, while a graduated ground
is still absorbed whole. Three outcomes, all recorded on the estimate as
`region` for later attribution:

- `foreground` — a ground was found and something survived it (the normal case).
- `center_box` — the border is not a uniform ground (lifestyle/in-context shot);
  the old central 70% box is used, because a flood would run into the scene.
- `flooded` — the flood reached everything, which means product and ground are
  the same colour (a white product on white); the whole frame is measured, and
  that colour is the answer.

**Measured, on synthgen (the corpus that exists locally).** 480 solid-colour
renders, 11 colours, 4 categories, seed 20260913:

| corpus | old | new |
|---|---|---|
| synthgen as rendered | 68.3% | **97.1%** |
| geometry-perturbed (product cropped out, pasted at a random position/scale onto a random-aspect canvas) | 22.5% | **90.0%** |

The first row is the honest surprise. synthgen centres every shape at 36–44%
scale, so the central box was supposed to be *correct by construction* — and
the old estimator still only scored 68.3%, because the box is 70% of the frame
while the product is under half of it, so the studio ground outvoted the product
and the answer came back `multicolour`. That is the same failure the real ABO
run shows: `multicolour` was the pixel estimator's second most common output
(58/205 rows) and was correct **1.7%** of the time.

The second row is B2 proper. Note that neither number is an estimate of ABO
accuracy — the perturbation was built to contain the failure mode. What they
establish is the mechanism and the direction.

Regression tests: `test_offcentre_product_is_localised`,
`test_bottom_band_product_is_localised`,
`test_localisation_ignores_background_gradient` and
`test_lifestyle_shot_falls_back_to_centre_box` (the guard: no uniform ground
means no flood). All four return the wrong colour on the pre-fix estimator.

**Sized on real ABO photography (2026-09-13 re-run). The mechanism works; the
magnitude does not transfer.** 87% of rows (182/209) were genuinely localised —
the flood found a studio ground and something survived it — so B2 is doing what
it was built to do. But accuracy is flat across the outcomes:

| region | n | pixel estimate correct |
|---|---|---|
| `foreground` (localised) | 182 | 27.5% |
| `center_box` (fallback) | 26 | 26.9% |
| `flooded` | 1 | — |

Localised and unlocalised rows score the same. Overall the pixel path moved
**23.9% → 27.3%**, against +67 points on the geometry-perturbed synthetic
corpus. So the synthetic benchmark measured a real defect that is not the
binding constraint here: ABO product shots are mostly *already* centred and
frame-filling, which is exactly the condition the old central box assumed.

**What the bottleneck actually is**, from the same report — `multicolour` is
now the single most common pixel output (58/209, 28%) and is right **1.7%** of
the time, and `orange` is 0/9 (wood read as orange). That is not a localisation
failure, it is a *vocabulary* failure: a 12-value colour domain cannot describe
a patterned rug or a wood grain, so the estimator correctly reports "no single
colour dominates" and the pipeline has nowhere to put that answer.

**Status:** DONE — mechanism fixed, pinned, and now sized on real data. Keeping
the localisation (it is correct, and it is what makes `pixel_conf` meaningful —
see B4), but further work on *localisation* has low expected value on this
corpus. The `multicolour` bucket is the next target and it is a schema
question, not a computer-vision one.

---

### B3 · The white-discount rule breaks on genuinely white products
**Severity:** Medium — fails an entire common colour class
**Where:** `tiger/colors.py:119`

White is discarded and the remainder renormalised whenever white is below 85%.
A genuinely white shirt at 84% therefore returns whatever shadow noise ranks
second. White is one of the most common fashion colours, so the threshold is
brittle at exactly the common case.

**Fix:** decide white-as-background from *spatial* evidence (is it connected to
the border?) rather than a global proportion threshold. Fall back to the
proportion rule only when the mask is unavailable.

**Fix implemented, exactly as stated** — and it costs nothing extra, because
B2's flood already answers "is this white connected to the border?". When the
flood ran (`region` is `foreground` or `flooded`), background white is already
out of the sample and any white left is on the product, so the 85% rule is not
applied at all. The proportion rule survives only on the `center_box` fallback,
where white background genuinely does still leak into the sample.

The white-on-white case that has no spatial answer is handled by `flooded`:
if the flood reaches everything, product and ground are the same colour, so
measuring the whole frame returns it. That is strictly better than the old
behaviour, which fell through to whatever shadow noise ranked second.

Regression test: `test_white_product_survives_the_shadow_that_used_to_outvote_it`
— a white product with a grey shadow skirt at 79% white inside the old central
box. Pre-fix the 85% rule discounted the white and returned `gray`; post-fix it
returns `white`. Plus `test_white_product_on_white_ground_is_still_white`, which
pins the `flooded` path.

**Status:** DONE

---

### B4 · `pixel_color_confidence` is not a confidence
**Severity:** Medium
**Where:** `tiger/colors.py:126` produces it; `tiger/solver.py:164` gates on `>= 0.55`

The value is the winning colour's **pixel share**. A 60% share of a badly-chosen
region is not a 60% probability of being correct, so the `0.55` gate is a magic
number applied to an uncalibrated quantity.

**Fix:** calibrate against actual correctness using the B0 diagnostics — fit
`P(correct | share, agreement, category)` on scored cases and gate on that.
Requires a B0 run first.

**Measured (2026-09-13) against the corrected ABO run.** The B0 report exists
now, so the question is answerable. `tests/bench_pixel_confidence.py`
reproduces everything below from
`paper_assets/results/abo/v2t_estimator_diagnostics.csv`. 738 report rows →
277 unique estimator outcomes → 205 with both a pixel estimate and a
ground-truth colour.

**The share does not rank correctness.** It separates once, at the bottom, and
carries no information above the gate:

| pixel share | n | would the pixel value have been right |
|---|---|---|
| < 0.55 | 70 | 7.1% |
| 0.55–0.60 | 15 | **40.0%** |
| 0.60–0.70 | 19 | 26.3% |
| 0.70–0.80 | 18 | 27.8% |
| 0.80–0.90 | 27 | 33.3% |
| ≥ 0.90 | 56 | 33.9% |

`corr(share, correct) = 0.233`. A 0.9 share is right a third of the time and
the *least* confident passing bucket is the most accurate one. So the field
name is exactly as wrong as B4 says — but fitting `P(correct | share)` would
be fitting a monotone model to a non-monotone signal, and the honest answer is
that this quantity cannot be turned into a probability. No calibrator was
fitted, on purpose.

**Second finding, which changes what the gate is even for: B6 made it inert.**
Since B6 escalates any genuine pixel-vs-probe disagreement, the gate can only
change an outcome when exactly one estimator produced a value. In the real run
that never happens on the pixel side:

- both estimators produced a value: **593** rows
- pixel only (the gate decides alone): **0** rows
- probe only (nothing to gate): 40 rows
- 450 disagreements → 0 values written; 183 agreements → 183 written

When both agree, both branches of the `if/else` return the same string. So on
this corpus the `0.55` threshold did not change a single written value. Tuning
it would have been tuning a dead number, and any before/after it produced would
have been noise.

**Third finding, and this is the one worth carrying into the paper.** On the
rows that *are* committed — both estimators agreeing — the written colour is
right **49.2%** of the time (n=65), and the share does not rank those either
(66.7% / 57.1% / 45.5% / 33.3% / 41.7% / 56.5% across ascending share bins).
Agreement between two independent estimators roughly quadruples accuracy over
disagreement (12.1%), which is B6 earning its place, but half of what the
system commits is still wrong and no available scalar tells you which half.

**Where the error actually lives** — not in the threshold, in the estimator:

| pixel value | n | correct |
|---|---|---|
| `gray` | 62 | 37.1% |
| `multicolour` | 58 | **1.7%** |
| `white` | 29 | 13.8% |
| `orange` | 11 | 0.0% |

`gray`, `multicolour` and `white` are 73% of all pixel estimates on a furniture
catalogue. Those are the studio ground, the estimator failing to find a
dominant colour, and the studio ground again. That is B2/B3, and it is why the
sequencing goes estimator first, calibration after.

**What was changed:**
1. The gate is no longer a bare magic number. It is `PIXEL_SHARE_MIN`, named a
   *share*, with the measurement above recorded next to it, and it is now a
   floor against a three-way split rather than a pretend probability.
2. A second condition was added that *is* mechanism-based: the pixel estimate
   is refused when `pixel_region == "center_box"`, i.e. when B2's localisation
   could not find a studio ground and fell back to measuring a fixed box. A 0.9
   share of an unknown region is not evidence about the product. (Evidence
   written before B2 has no region field and is treated the same way, since it
   came from the unlocalised estimator by definition.)
3. `pixel_color_region` is plumbed through evidence → `RepairPlan` →
   the repair log → the B0 diagnostics CSV, so the next run can condition
   `P(correct | ·)` on *how* the estimate was obtained — which, unlike the
   share, is a variable with a mechanism behind it.

Tests: `test_pixel_value_used_when_the_product_was_localised` and
`test_pixel_value_refused_when_the_product_was_not_localised`.

**Re-fit after the estimator fix (2026-09-13), and the verdict changes.** The
share was uninformative because the estimator was measuring the wrong region.
Once it measures the product, "how much of the product is this colour" starts
to mean something:

| pixel share | correct — before | correct — after |
|---|---|---|
| < 0.55 | 7.1% | 7.0% |
| 0.55–0.60 | **40.0%** | 24.0% |
| 0.60–0.70 | 26.3% | 33.3% |
| 0.70–0.80 | 27.8% | 41.4% |
| 0.80–0.90 | 33.3% | 45.8% |
| ≥ 0.90 | 33.9% | **53.3%** |
| **corr(share, correct)** | **0.233** | **0.346** |

It is now **monotone above the gate** — every bucket beats the one below it,
where before the least-confident passing bucket was the most accurate. The
0.55 cut also separates harder (7.0% vs 37.7%). So the quantity B4 called "not
a confidence" has been turned into something that ranks correctness, not by
calibrating it but by fixing what it measures. That is worth stating plainly in
the paper: **the confidence was uninformative because the estimator was
broken**, not because pixel share is inherently meaningless.

**Two things still hold, and stop this being closed outright.**

1. *It still does not rank the rows that matter.* On committed repairs (both
   estimators agreeing) accuracy by share is 42.9 / 45.0 / 55.6 / 46.2 / 57.1
   across ascending buckets, n=79 — noisy and near-flat. Once two independent
   estimators agree, the share adds little on top of the agreement itself.
2. *The gate is still inert.* 619 rows had both estimators, **0** had pixel
   only. As before, the threshold changed no written value.

And the headline number barely moved: committed colour accuracy is **49.4%**
(was 49.2%). The estimator got better, agreement rose (24.8% → 29.2%, which is
why more repairs were attempted — 55 → 64 V2T cases), but *what gets committed*
is still right about half the time.

**Status:** DONE — asked and answered twice, and the second answer is the
useful one. A calibrator is now fittable in principle; it is not worth fitting
while the gate it would feed is unreachable and the share is flat on the
decision set. Revisit if `multicolour` (B2) is dealt with, which is what would
give the pixel path more rows to be confident about.

---

### B5 · Aspect ratio is destroyed before cropping
**Severity:** Low
**Where:** `tiger/colors.py:76` — `resize((size, size))`

Squashes tall product images, shifting which body region lands in the centre
box and compounding B2.

**Fix:** resize preserving aspect ratio, then crop.

**Sweep finding — blocked on data.** Confirmed in code, but synthgen emits square
images, so the distortion is identically zero on the committed dataset.

**Correction (2026-09-10): partially unblocked.** The claim that "the Fashion and
ABO datasets are not in the repo" was wrong. The full official ABO release is on
the dev machine at `data/raw/abo/` -- 398,212 images (matching the official
count) and all 16 listings files. `data/raw/` is gitignored, which is why it was
missed; not in the repo is not the same as not available.

**B5 is now testable.** Only 36.2% of ABO images are square (dimensions range
21–2871 px), so `resize((size, size))` distorts nearly two thirds of the corpus
before the centre crop is taken. This compounds B2 on the same data.

**The stated mechanism was wrong, and this matters for how B5 is written up.**
"Shifts which body region lands in the centre box" does not follow: a square
resize is a uniform rescale of each axis, so the central 15–85% box covers the
same *fractional* region of the original image either way. Squashing a 1:3
image does not move the product out of the box. Measured rather than argued: on
200 centred products in frames from 1:2.9 to 2.9:1, the square-resize estimator
and an aspect-preserving one restricted to the same central box agree on
**194/200**. The six disagreements are resampling artefacts (below), not the
product leaving the box.

What the square resize actually costs:
1. **It destroys shape**, which is invisible to a positional crop but not to
   B2's flood: connectivity, the border frame's width, and the geometry of the
   product edge are all computed on distorted pixels once localisation exists.
   A 1:3 product came out 1:1 before it was ever localised.
2. **Asymmetric resampling.** On a 2871×500 image the long axis is downsampled
   30× while the short axis is upsampled 5×, so thin features survive on one
   axis and vanish on the other, and colour proportions shift with the aspect
   ratio rather than with the product.

**Fix implemented:** `_load_rgb` scales the long edge to `size` and lets the
short edge follow, so nothing is distorted. Everything downstream (border
frame, flood, centre-box fallback) works on the true shape.

Regression tests: `test_resize_preserves_aspect_ratio` (a 160×480 image loads
as 32×96, not 96×96) and `test_tall_product_keeps_its_shape_through_the_estimator`
(a 1:3 bar in a 1:3 frame is localised as a 1:3 region — it measured 1:1 before).

**Status:** DONE — severity confirmed Low on its own; it is load-bearing only
because B2's localisation is geometric

---

### B6 ⚑ · The two estimators never cross-check
**Severity:** High — this is the highest-value *architectural* fix
**Where:** `tiger/solver.py:164`

The logic is `if/else`: a confident-but-wrong pixel estimate silently wins and
the CLIP probe is never consulted. **Disagreement between the two is exactly the
signal worth acting on**, and it is currently discarded.

**Fix:** agree → repair; disagree → escalate. This converts silent wrong-writes
into escalations, raising restoration accuracy on the acted-on set by trading
coverage. Report as a **risk–coverage curve**, not a point — that is the
standard form in the selective-prediction literature and makes the trade
explicit rather than looking like threshold tuning.

**⚑ Architectural.** This is the system's only possible abstention on *value*
uncertainty. The γ-gate abstains on routing, Eq. 27–29 on schema and similarity;
nothing today abstains on "I do not know what colour this is", so a wrong-but-
in-domain value that raises CLIP similarity is committed silently (`E2`). Treat
this as a missing pipeline stage between Solver and Verify, not a threshold tweak.

B0's report already quantifies what this would buy before it is built.

**Decision (user, 2026-09-11):** build it — assessed as low-risk and additive
(doesn't touch the γ-gate, Eq. 27–29, or T2V at all; only V2T single-field
patching), so implementing ahead of sizing was accepted as reasonable here,
unlike `D4` (which changes results for an existing row class and was
explicitly told to precede the baseline run).

**Fix implemented** (`tiger/solver.py::plan_repair`, `tiger/repair.py`):
- `_corrected_value`'s if/else is unchanged (still resolves single-estimator
  fields — material, pattern — exactly as before; nothing there disagrees with
  anything, since only `color` has a second, pixel-based estimator).
- Before committing a V2T patch, `plan_repair` now checks whether the pixel
  estimate and the CLIP probe both produced a value *and* differ. If so, the
  row escalates (`plannable=False`) instead of silently committing whichever
  one the old confidence threshold happened to favour.
- The escalation carries the same `value_source`/`pixel_value`/`pixel_conf`/
  `probe_value`/`estimators_agree` diagnostic fields an applied repair would
  (`repair.py`'s unplannable branch previously dropped them) — an
  estimator-disagreement escalation stays attributable in any future analysis,
  not indistinguishable from a generic unplannable row.

Regression tests: `tests/test_solver_planning.py::test_v2t_escalates_on_estimator_disagreement`
(disagreement → escalate, diagnostics preserved) and
`test_v2t_plans_normally_when_only_one_estimator_applies` (a field with no
second estimator — e.g. material — is not mistaken for a disagreement).

**Status:** DONE — 167 tests passing. Not yet sized against real data (that
still needs the Phase 2 run — the risk/coverage trade this converts silent
errors into hasn't been measured on real numbers yet), but the mechanism
itself is built and tested.

---

### B7 · CLIP is the weakest available encoder for attribute binding
**Severity:** Medium — sets the ceiling on the probe path
**Where:** `configs/tiger.yaml:12`, `compare_encoders` at `:17`

ARO (ICLR 2023) measures CLIP at **62%** on attribution where chance is 50%,
against BLIP 88% and XVLM 87%. The pipeline uses CLIP for an attribute-centric
task. The probe implementation itself is sound (category-conditioned,
prompt-ensembled, normalised at `tiger/sieve.py:128`) — the encoder is the limit.

**Fix options:**
1. Swap the probe encoder to BLIP/XVLM — `compare_encoders` already supports it.
2. Apply the Koishigarina et al. (ICLR 2026) linear transform on text
   embeddings, which recovers cross-modal binding **from the existing embedding
   cache** with no re-encoding and no retraining.

**"`compare_encoders` already supports it" was wrong (checked 2026-09-13).**
`compare-encoders` scores candidate encoders offline on per-field probe
accuracy (`tiger/eval/encoder_compare.py`) and prints a table. It never touched
the live pipeline. In the pipeline there was exactly one encoder —
`_encoder(cfg)` built from `models.clip_model_name` — and `sieve.compute_signals`
used that same object for `sim_full`, the title view, the swap check *and* the
per-field probes. There was no swap to perform: changing the probe encoder
meant changing the reported CLIP baseline for every other signal at the same
time, which is not what B7 asks for.

**Option 1 is now actually available.** `models.probe_model_name` (empty by
default, so nothing changes for any existing configuration) puts the probes on
their own encoder while `sim_full`, the swap check and the LOO deltas stay on
`clip_model_name`. The probes then get a second image-embedding pass from the
probe encoder — necessarily, since an image and a caption can only be compared
inside one embedding space. The embedding cache is already keyed per model
name, so the two do not collide.

Resolution lives in one place, `encoders.probe_encoder_from_cfg(cfg, root)`,
memoised per process, and is used by the sieve, by the repair cycle's
re-diagnosis passes (`repair.py` runs `compute_signals` itself, twice) and
therefore by the ablations. Threading it through call sites instead was tried
first and immediately produced the bug it invites: detection probing with one
encoder and re-diagnosis with another. They must agree or the numbers stop
meaning one thing.

**What is not done, and what is not known:**
- **No encoder has actually been swapped.** This is wiring, not a result. The
  swap needs a run, and `torch`/`transformers` are not installed in this
  checkout (nor is there a GPU) — the mechanism is unit-tested with encoder
  doubles, not with a real second model.
- **BLIP specifically may not drop in.** `ClipEncoder` calls
  `AutoModel.get_text_features` / `get_image_features`, the dual-encoder API.
  That holds for CLIP and SigLIP (both already in `compare_encoders`). BLIP's
  retrieval model is an image-text-matching architecture with a different
  interface, and XVLM is not in `transformers` at all. Either will need an
  encoder adapter behind the same two methods. **Unverified** — stated from the
  interface `ClipEncoder` requires, not from having loaded them.
- Option 2 (the Koishigarina text-embedding transform) is untouched. It remains
  the cheaper path precisely because it needs no re-encoding.
- The probe z-thresholds are calibrated per encoder, so a swap requires
  re-running `calibrate` before `detect`. Nothing enforces that yet.

Tests: `tests/test_probe_encoder_split.py` — the probe encoder decides
`probe_*_pred` while the primary still decides `sim_full`; a probe encoder that
cannot read an image drops that row from the probes without marking the row's
image missing; and the resolver returns `None` unless a genuinely different
model is named.

**Status:** DOING — option 1 is wired and tested, and the "already supported"
claim is corrected. Choosing and validating an encoder needs a run.

---

### B8 · Saturation is unreliable at the dark end, so black products are hue-binned
**Severity:** High — found while measuring B2/B5 (2026-09-13), not previously listed
**Where:** `tiger/colors.py` — the achromatic/chromatic split

Saturation is `(max − min) / max`. The denominator is the pixel's own value, so
on a near-black pixel a difference of a few RGB levels is a large saturation:
`(20, 22, 35)` scores `s = 0.43` and sails past `ACHROMATIC_SAT_MAX = 0.18` into
the hue binner, which calls it **blue**. `(18, 30, 22)` becomes green. Only
pixels whose channels happen to be nearly equal — `(28, 28, 30)`, `s = 0.067` —
were ever classified black.

Sensor noise and JPEG chroma subsampling produce exactly this spread in dark
regions of real photographs, so this is not an artefact of the synthetic
renderer that surfaced it.

**Measured:** on 480 synthgen renders the old estimator identified black
products correctly **5%** of the time (2/44) as rendered, and **0%** once the
geometry was perturbed. Every other colour scored 51–100%. Black is one of the
most common furniture and homeware colours, so this is not a corner case.

**Fix:** value decides first. Below `BLACK_V_MAX` a pixel is black regardless of
saturation; the saturation split only applies above it, where saturation means
something. One consequence worth stating: a strongly saturated but very dark
hue (`v ≤ 0.22`, i.e. every channel under 56/255) is now black rather than
"dark navy". That is the right call at that value — and the brown rule, which
lives at `v < 0.6`, is untouched.

After the fix black reaches **84%** on the perturbed corpus (0% before), and
overall accuracy moves 82.7% → 90.0% perturbed and 88.3% → 97.1% as rendered.

Regression tests: `test_noisy_black_product_is_black_not_blue` (jittered
near-black fill, the realistic case) and `test_dark_pixels_do_not_become_hues`
(the rule stated directly). Both return `blue` on the pre-fix estimator.

**Confirmed on real data (2026-09-13 re-run).** `black` does not appear in the
pre-fix run's six most common pixel outputs at all — black products were being
hue-binned into blue, green and pink and so never surfaced as an answer. Post-
fix it is the estimator's fourth most common output at **66.7% correct**
(15 cases), the highest accuracy of any colour it returns. This is the clearest
single-defect win in the estimator work, and it was found by accident while
measuring something else.

**Status:** DONE — and verified on ABO, not only on the renderer that exposed it

---

### B9 · The colour domain cannot describe the corpus, so 28% of estimates are "multicolour"
**Severity:** High — now the largest single bucket of estimator error
**Where:** `configs/schema.yaml` colour domain (12 values) · `tiger/colors.py` `MULTI_DOMINANCE_MIN`
**Found:** 2026-09-13, while sizing B2 on the re-run

With localisation fixed, `multicolour` is the pixel estimator's most common
verdict — **58 of 209** scored rows (28%) — and it is correct **1.7%** of the
time. `orange` is 0/9 (wood grain read as orange). Together that is a third of
the pixel path's output, almost none of it usable.

This is not a vision failure. The estimator is right that no single colour
dominates a patterned rug or a wood grain; it has nowhere to put that answer,
because the schema offers twelve flat colour names and the catalogue's declared
values are things like "Dark Brown" and "Espresso". `multicolour` is
simultaneously a legal domain value *and* the estimator's way of saying "I
decline", and those two meanings are not distinguishable downstream.

**Options, cheapest first:**
1. Separate the two meanings: return `undetermined` (abstain, never written)
   rather than `multicolour` (a real declared value). The B6 machinery already
   escalates rows it cannot resolve; this lets it fire for the right reason.
2. Extend the domain with the neutral/wood family the corpus actually uses
   (beige, tan, natural, espresso) and map them in `surface_forms`. Changes the
   schema, so it changes every probe and every LOO delta — needs a full re-run.
3. Report colour as a distribution rather than a label, and score the repair on
   whether the declared value is in the top-2. `ColorEstimate.top2` already
   carries this; nothing consumes it.

**Fixed 2026-09-14 — and the actual cost was not where this entry predicted.**
`multicolour` is barely ever *written* (4 rows, 0 correct). Its damage was done
somewhere else entirely: it was being compared against the probe's answer as
though it were a value, so `B6`'s conflict check fired and the row escalated
with the reason **"estimators disagree"** when only one estimator had spoken.

On the ABO run that is **175 of 280 unique rows**. The giveaway is that the
pixel/probe "agreement" rate inside that group is **1.7%** — which is what you
get comparing a refusal against an answer, not what a genuine conflict looks
like.

`_corrected_value` now maps `multicolour`/`unknown` from the pixel estimator to
*no opinion* (`pixel_value=""`, `pixel_declined=True`). A colour row where the
estimator declined is then the same situation as `material` or `pattern`, which
have no pixel estimator at all: one opinion, planned normally. `B6`'s escalation
is untouched for real conflicts — two values that differ.

**Expected effect on the next run:** ~175 rows stop escalating as false
conflicts and instead get a probe-decided repair. That is a coverage/accuracy
trade, and it should be reported as one: the probe alone is right 38.7% against
the 49.4% that two agreeing estimators buy. It also repairs `B6`'s headline —
"450 disagreements" was counting refusals as conflicts.

`pixel_declined` is carried through evidence → plan → repair log → the
diagnostics CSV, so the next run can separate the three cases cleanly.

Options 2 (extend the domain) and 3 (report a distribution) are untouched and
remain the real answer to the *vocabulary* problem; this fix stops the
vocabulary gap from also corrupting the escalation bookkeeping.

**Status:** DONE for the bookkeeping half — the domain itself is still too
small for the corpus, which is options 2/3 and needs a schema change plus a run

---

## C. Configuration & reproducibility

### C1 · γ has four different values across the repo
**Severity:** Medium — nobody can say which threshold produced which result
**Where:** `configs/tiger.yaml:67`

| Source | Value |
|---|---|
| `configs/tiger.yaml:67` | **0.40** |
| ABO analysis + `honest_limitations.md` | "default **0.60**" |
| `paper_assets/paper_concepts.md` | **0.85** |
| ABO recalibrated | **0.448** |

The ABO "75.2% fall below the default threshold" finding depends entirely on
0.60 being the value that actually ran. Reading the committed confidence
histogram, at γ=0.40 only ~5% of items fall below — not 75%.

**Fix:** establish which value the ABO run used, correct every document to
match, and record the effective γ in the run output so this cannot recur.

**Resolution (2026-09-11):**
- Which value the ABO run used was already established by the verification
  sweep (`42bcb36`) but never written into this entry: `paper_figures/abo_confidence_plot_clean.png`'s
  legend reads "Gamma threshold (0.6)" — ~70% of mass below 0.60, matching the
  documented 75.2%; only ~5% falls below the config's 0.40 default. **The ABO
  run used an overridden γ=0.60**, not the repo's current default. This is not
  a contradiction to fix — `honest_limitations.md`'s "default γ=0.60" claim is
  correct for that run and needs no change. What was actually wrong is the
  *fourth* value: `paper_concepts.md` described γ as "dynamically calibrated"
  to an "85% precision floor" — that 0.85 is a different parameter entirely
  (`fusion.precision_floor`, the Sieve's separately-calibrated signal
  threshold, not the Arbiter's static γ gate). Fixed — see the corrected
  paragraph in `paper_concepts.md`.
- Recurrence prevention: `--gamma` now exists (`C2`, below) and echoes the
  effective value into `repair`/`ablate-repair` output, so an overridden run
  is no longer silently undocumented.

**Status:** DONE — remaining two values (0.40 config default vs. 0.60 ABO-run
override) are a real, disclosed difference between runs, not an error.

---

### C2 · No `--gamma` CLI flag
**Severity:** Medium
**Where:** `tiger/cli.py` (absent)

Recalibration requires editing `configs/tiger.yaml` in place, which is how C1
arose. Their own note flags the risk of leaving the config wrong between the
fashion and ABO runs.

**Fix:** add `--gamma` to `repair` and `ablate-repair`, overriding config; echo
the effective value into the run output.

**Status:** DONE — applied globally at dispatch time in `main()` (`tiger/cli.py`)
so every command shares one override path; echoed to stdout for `repair` and
`ablate-repair` specifically. 162 tests passing, unchanged.

---

### C3 · Tests deleted but still declared
**Severity:** Medium — reproducibility claim for an applied-venue paper
**Where:** `tests/` (only stale `__pycache__` remains) · `pyproject.toml:41`

`testpaths = ["tests"]` and the `test` extra still ship, so `pytest` collects
nothing. The README previously advertised 68 unit tests pinning the critical-
review fixes (F1/F3/F6/F10/F12, the Eq. 27–29 gates, routing constraints,
fusion quarantine) — that was a genuine credibility asset.

**Sweep finding — recoverable, and it should land first.** The suite was deleted
in `a7c1e72` ("Strip repository to absolute bare minimum"). Recovered contents:
12 files, **73** `def test_` functions (the README's "68" predates the last
additions). Restore with:

```bash
git checkout a7c1e72^ -- tests/ && .venv/bin/python -m pytest -q
```

This is purely additive — no runtime code changes — and it is the instrument that
proves subsequent fixes are architecture-preserving. **Do it before any other fix.**
Tests that fail on today's code are themselves findings and belong in this file.

**Fix:** restore the suite as above,
or remove the pytest config and drop the claim. Do not leave it declared-but-empty.
Also delete the surviving "✅ Unit tests written" line in
`paper_assets/tiger_project_doc.md` §11, which `f050639` missed (see E6).

**Status:** DONE — `62373dc`; 73 restored, all passing. Suite now at 101.

---

### C4 · Hardcoded generated-image path
**Severity:** Low
**Where:** `tiger/solver.py:234,236`

`data/sample/images/generated/` regardless of dataset, so ABO and Fashion
artifacts are written into the synthetic sample tree.

**Fix:** derive from `cfg["data"]` with a per-run subdirectory.

**Status:** DONE — `plan_repair` now takes `sample_dir` (from `cfg["data"]["sample_dir"]`,
threaded through by `run_repair_cycle`) instead of a hardcoded literal.

---

### C5 · Evaluation artifacts exist nowhere in the repo
**Severity:** Medium
**Where:** `data/outputs/` is gitignored; `data/sample/` and `data/thresholds/` are absent

Every number in `paper_assets/` traces to Kaggle CSVs that survive in no
committed form. For a systems paper this is the reproducibility surface.

**Sweep correction:** `data/outputs/` is **not empty** — 41 files exist locally,
0 tracked. The artifacts that back the detection numbers are already on disk and
merely uncommitted, which makes this much cheaper than it reads. `data/sample/`
and `data/thresholds/` genuinely do not exist and must be regenerated (E6).

**Fix:** commit the summary CSVs (not the caches) under `paper_assets/results/`.

**Done (2026-09-13).** `paper_assets/results/` now carries both runs' summary
artifacts plus `README.md`, which records provenance: which notebook produced
each directory, when, and what each file is. Specifically:

- `abo/` refreshed to the **2026-09-12 19:58 re-run** (executed notebook
  `tiger_abo_d4_check.ipynb`), which supersedes the run committed in `387b455`
  on two counts and matches it everywhere else: the summary reports all five
  outcome statuses rather than two (`E8`), and the estimator-attribution report
  has 738 rows rather than 289 because `B6`'s disagreement escalations are now
  captured. Ablation counts are unchanged (Full System 268, No Gamma Gate 498).
- `synthetic/` gained `detection_metrics_sweep.json` (the per-seed confusion
  matrices behind the detection table) and `ablations.json` (the detection
  ablations behind `E9`) — the two artifacts that were still uncommitted.
- Excluded on purpose: embedding caches, per-seed `.npz` arrays, per-seed
  sieve/evidence dumps. They are large and regenerate from the notebook.

`data/sample/` remains absent — the notebooks do not export it, so it needs a
re-run that does. That half stays with `E6`.

**Status:** DONE (except `data/sample/`, which is `E6`'s)

---

### C6 · `requirements.txt` and `pyproject.toml` disagree
**Severity:** Low
**Where:** `requirements.txt` · `pyproject.toml:12-34`

`requirements.txt` pins `opencv-python`, `tqdm`, `requests` and `torchvision` —
none imported anywhere under `tiger/`. `matplotlib` is imported by
`tiger/viz.py:2` and declared in neither file, so a clean `pip install -e ".[dev]"`
cannot run `viz`.

**Fix:** delete `requirements.txt` in favour of the extras (or regenerate it from
them), and add a `viz` extra carrying `matplotlib`.

**Status:** DONE — `requirements.txt` deleted (confirmed nothing referenced it
outside this entry); `pyproject.toml` gained a `viz` extra (`matplotlib`),
also added to `dev`.

---

### C7 · A 73 MB AWS installer is sitting in the project root
**Severity:** Low
**Where:** `awscliv2.zip`, `aws/` (untracked but present)

Alongside untracked `literature_review.md`, `related_work.tex`, `related_work.bib`
and `papers/`. Nothing distinguishes scratch from deliverable.

**Fix:** delete the installer and `aws/`; decide whether the literature files are
tracked deliverables and either commit them or add them to `.gitignore` explicitly.

**Checked (2026-09-11):** not present in this checkout (`awscliv2.zip`, `aws/`
absent; `git status` clean, nothing untracked). This was always untracked
scratch on the original dev machine, so a fresh clone never had it — nothing
to delete here. Still worth checking directly on whichever machine actually
carries it.

**Status:** DISCARDED (2026-09-13, user decision) — the file was scratch
clutter on the original researcher's machine, never part of this repo. Not
applicable to this project going forward; not tracked as open work.

---

### C8 · Repair iterates a `set`, so the provenance log is not reproducible
**Severity:** Low
**Where:** `tiger/repair.py:95` — `for row_id in active_ids:`

Python string hashing is randomised per process. Outcomes are unaffected (the
pass reads `flagged` and `pool`, both fixed at pass start), but the provenance
log — the audit artefact the roadmap cites for rollback under 4.3 — comes out in
a different order on identical inputs.

**Fix:** `for row_id in sorted(active_ids):`

**Status:** DONE — `d1f22c3`.

---

## D. Robustness & design

### D1 · `class_weight="balanced"` undercuts the calibration claim
**Severity:** Medium
**Where:** `tiger/arbiter.py:129`

Class weighting distorts probability calibration, yet the γ-gate depends on
calibrated `predict_proba` and `reviewer_defense.md` Attack 8 claims logistic
regression was chosen *because* it is well-calibrated.

**Fix:** produce a reliability curve (`arbiter.reliability_table` already
exists). If calibration is poor, add Platt/isotonic recalibration on a holdout,
or drop the balancing and handle imbalance in the threshold instead.

**Measured (2026-09-13). The concern is not borne out: calibration is fine.**
Held-out calibration seed 1014 of the corrected ABO run, model trained on
1007–1013, n = 1,653 rows after dropping the rule-routed missing-modality rows:

| | ECE | gap at γ=0.60 | rows ≥ γ | accuracy ≥ γ | E3 recall |
|---|---|---|---|---|---|
| **balanced (shipped)** | **0.026** | **−0.024** | 60.1% | 0.740 | **0.691** |
| unbalanced | 0.025 | +0.060 | 73.9% | 0.795 | 0.064 |

So neither branch of the stated fix applies. Recalibration is not warranted at
ECE 0.026, and dropping the balancing does not improve calibration — the ECEs
are within 0.001 of each other, so class weighting is not what is setting the
calibration here.

The sign at the boundary is worth keeping: at γ the balanced router is
*under*-confident by 0.024, so the gate escalates a few rows it would have
routed correctly. For a gate whose entire job is abstention that is the safe
direction to be wrong in.

**And dropping the balancing would be a clear regression, for a reason that has
nothing to do with calibration.** It raises aggregate accuracy 0.636 → 0.724 —
by predicting the majority classes. E3 recall goes **0.691 → 0.064**: the model
predicts E3 eleven times on a holdout containing 94 of them. E3 is the class
that routes to `BOTH`, the two-pass path `D4` exists to make real. The
balancing is buying minority-class recall, and the aggregate accuracy it costs
is the price of that.

**What was changed:** `arbiter.calibration_report()` (next to the existing
`reliability_table`) computes ECE over equal-width bins plus the signed gap in
a ±0.05 window around γ — the aggregate can look fine while the router is
miscalibrated exactly where the gate cuts, so the boundary is reported
separately. `train-arbiter` now stores it as `calibration_holdout` in the model
JSON and prints it, so the claim is re-measured on every run instead of being
asserted once. The ABO numbers are committed at
`paper_assets/results/abo/arbiter_calibration.json`, and `reviewer_defense.md`
Attack 8 now cites them instead of asserting calibration.

Tests: three in `tests/test_arbiter.py` — a perfectly calibrated router scores
~0, an overconfident one is caught, and a router that is fine in aggregate but
wrong at γ is caught by `gap_at_gamma`.

**Status:** DONE — `class_weight="balanced"` stays, now with the measurement
that justifies it

---

### D2 · Asymmetric failure handling in the independent verifier
**Severity:** Low — intentional, but undocumented in the paper
**Where:** `tiger/verify.py:161` (`check_v2t` returns `True` on unreadable image)
vs `check_t2v` (returns `False` on unreadable candidate)

Deliberate and commented: do not veto on our own read failure, but do not trust
an unverifiable candidate. Sound, and should be stated explicitly rather than
discovered by a reviewer.

**Status:** DONE — stated explicitly in `paper_assets/pipeline_architecture.md`,
Independent Verifier bullet (§ Component Details).

---

### D3 · ~~Position bias in `check_t2v`~~ — WITHDRAWN
**Severity:** ~~Low~~ — does not apply
**Where:** `tiger/verify.py:176` · `tiger/vlm_judge.py:246`

**Original claim:** old/new images are presented in a fixed order, and
MLLM-as-a-Judge (ICML 2024) names position bias as a failure mode present even in
GPT-4V.

**Sweep finding: the premise does not hold for either implementation.**

1. `IndependentVerifier.check_t2v` is a **bi-encoder**. It embeds both images
   independently and compares `imgs[0] @ t` against `imgs[1] @ t`. Cosine
   similarity has no notion of presentation order — there is no position to bias.
2. `GeminiVLMJudge.check_t2v` sends **one image** (the proposed replacement).
   Position bias requires two items in an order; there is no ordering. This was
   already designed out — see `project_chronicle.md` Hiccup 3.

The citation is real and the failure mode is real for VLM judges in general; it
is simply not reachable in this code. Implementing "evaluate both orderings"
would add cost for a bias that cannot occur.

**Status:** WITHDRAWN — no action. Retained so the reasoning is not re-derived.

---

### D4 ⚑ · The two-pass loop never runs, so E3 has no behaviour of its own
**Severity:** High — a documented architectural edge that has never executed
**Where:** `tiger/repair.py:84` (mask) · `tiger/solver.py:224` (plan) · `tiger/repair.py:162` (apply)

```python
active_mask = flagged["flagged"].astype(bool) & ~flagged["row_id"].astype(str).isin(
    [rid for rid, oc in outcomes.items() if oc.final_status != "pending"])
```

An accepted repair sets `final_status = "repaired"` immediately, so pass 2
excludes it; escalated rows are excluded too. Pass 2 has nothing to act on and
`verify.max_passes: 2` is inert.

Downstream of that: `route()` returns E3 → direction `BOTH`; `plan_repair`
handles `("T2V","BOTH")` identically and returns a plan whose direction is
`"T2V"`; `repair.py` applies T2V and stops. **E3 is operationally
indistinguishable from E2** — the taxonomy has four classes and three behaviours.

**Consequence:** the "image first, then re-diagnose text" behaviour described in
`paper_concepts.md` §1, `tiger_project_doc.md` §9 and the E3 row of the taxonomy
table has never run. The re-route arrow in the README architecture diagram is
drawn but not wired. `mixed_swap_color` rows (2% of injected noise) get the image
swapped and keep the wrong colour.

**Fix:** pick a contract and implement it —
1. keep accepted rows `pending` so they re-enter the next pass, terminating when
   they are no longer flagged; or
2. give `BOTH` an explicit two-step plan inside one pass: T2V, re-embed, then V2T
   against the new image.

Either changes results. If adopted, it must land **before** the corrected
baseline run, not after.

**Decision (user, 2026-09-12):** build it. Sized against the real ABO Phase 2
run first — the Arbiter's own routing output showed `{'E1': 494, 'E2': 315,
'E3': 239, 'CLEAN': 655}`: **E3/BOTH is 239 of 1703 routed rows (~14%)**,
comparable in size to E2, not the rare edge case the 2% *synthetic injection
rate* for one noise subtype had suggested. That 2% figure was the rate for
one deliberately-injected noise type; E3 as the Arbiter actually classifies it
is broader and far more common.

**Fix implemented — option 1** ("keep accepted rows pending"), not option 2
(a bespoke two-step BOTH plan). Reading `tiger/repair.py`'s own module
docstring first: *"the cycle re-diagnoses so a row needing two fixes gets two
(capped at two passes, roadmap 2.5)"* — this was already describing option 1
as the intended design. The actual bug was one line: both the V2T and T2V
acceptance branches set `final_status = "repaired"` immediately, which
excluded the row from `active_mask` in every later pass regardless of whether
`max_passes` said there should be one. `verify.max_passes: 2` was accordingly
inert — read from config, never actually able to matter.

- Both acceptance branches now leave the outcome `"pending"` instead.
- A new `_promote_clean_pending(outcomes, flagged)` (module-level, not a
  closure, specifically so it is unit-testable without the full CLIP/encoder
  pipeline) runs at the top of every pass: a `"pending"` row whose fresh sieve
  re-run shows it is no longer flagged gets promoted to `"repaired"`; a row
  still flagged stays `"pending"` and is picked up again by `active_mask`
  next pass — with its already-applied fix (e.g. the new image) now part of
  `working`, so the next pass's diagnosis is against the corrected state.
  Absence from the current pass's frame is deliberately **not** treated as
  evidence of cleanliness (a test caught this ambiguity in the first draft).
- A post-loop promotion check runs once more against the final `working`
  state, so whatever the *last* pass accepted is not lost to the "still
  pending after the cap -> unrepaired" catch-all with no chance to be
  re-checked.
- This required no changes to `solver.py`/`plan_repair` at all — `route()`
  still returns `BOTH` → `T2V` first, same as before; it's the *next* pass's
  ordinary re-diagnosis (now actually reachable) that catches the text half
  against the newly-swapped image, exactly the "image first, then re-diagnose
  text" sequence the paper docs already described.

Regression tests: `tests/test_repair_two_pass.py` (promotion logic in
isolation — promote when genuinely clean, stay pending while still flagged,
never touch already-terminal rows, don't mistake absence for cleanliness).

**Consequence:** the ABO/synthetic Phase 2 runs completed *before* this fix
landed, so their E3-row numbers reflect the old (image-only) behaviour.
Re-running both notebooks is needed to get numbers reflecting the fix;
everything else from those runs (detection numbers, E1/E2-only repairs, A1/A6
validation) is unaffected and stays valid.

**Status:** DONE — 178 tests passing.

**Follow-up: hard-proof instrumentation (2026-09-12).** The corrected-run
tables show accuracy moved in the right direction, but didn't prove D4
specifically fired (B6 landed in the same run). Added: `_evaluate_run` now
tallies each config's `passes_used` distribution and, more directly,
`both_directions_rows` — rows whose log contains both a T2V and a V2T entry,
which is the unambiguous signature of a row getting both fixes across passes
(the thing that could never happen before D4). Surfaced in
`format_repair_ablations`'s printed report, `repair_ablations.json`, and a
dedicated cell in both notebooks (reads the JSON directly, same pattern as
the existing A1/A6 checks) — next Kaggle run gives a real yes/no on real ABO
rows, not just the unit tests.

---

### D5 · Five colours are in the domain *and* in the alias map
**Severity:** High — root cause of D6 and D7; land first
**Where:** `configs/schema.yaml:19-31`

`silver, gold, beige, navy, teal` appear in `color.values` **and** in
`color.aliases`, mapping to `gray, yellow, white, blue, green`. `Schema.in_domain`
normalises both sides so membership still works, but `Schema.domain("color")`
returns 17 raw entries of which 5 are semantic duplicates.

**Consequences:**
- Contrastive probes (`tiger/sieve.py:128`) build one caption per raw entry, so
  "silver" and "gray" compete as separate candidates for the same pixels. A grey
  product can lose its declared value to its own alias and fire a false
  `flag_probe_color`.
- Any comparison of a raw domain value against a normalised one breaks — D6, D7.
- **Found while fixing:** `synthgen` and the noise injector draw colours from
  this list (`synthgen.py:201`, `noise.py:94`) and `COLOR_RGB` has no entry for
  the five. `python -m tiger.cli synthgen` raised `KeyError` on ~79 of 240
  products. **This is why `data/sample/` cannot be regenerated** (C5/E6) — the
  bundled catalogue has been unbuildable since the ABO schema edit. Fixed as a
  consequence; a test now asserts every domain colour is renderable.

**Fix:** keep the ABO colours in `aliases` only, or in `values` only with the
alias removed. Then assert `set(values) ∩ set(aliases) == ∅` at schema load so it
cannot recur.

**Status:** DONE — `03d0546`; load_schema now refuses a reintroduced clash.

---

### D6 · The independent verifier vetoes correct repairs on aliased colours
**Severity:** High — silently suppresses good repairs, in the column carrying the +13.3% claim
**Where:** `tiger/verify.py:174`

```python
pred = domain[int(np.argmax(np.stack(embs) @ img[0]))]   # raw domain value
return pred == self.schema.normalize(field, value)        # normalised value
```

`pred` is raw (`"navy"`); the right-hand side is normalised (`"blue"`). Whenever
the independent encoder's argmax lands on one of the five aliased colours,
`check_v2t` returns `False` and the repair is vetoed as a semantic failure.

**Consequence:** vetoes caused by this bug are indistinguishable in the results
from genuine wrong-direction catches — the exact quantity the Independent
Verifier ablation measures.

**Fix:** `return self.schema.normalize(field, pred) == self.schema.normalize(field, value)`.
**Requires D5** to remove the duplicate candidates as well.

**Status:** DONE — `ff54b41`.

---

### D7 · `_title_color` manufactures false title-contradiction flags
**Severity:** Medium
**Where:** `tiger/sieve.py:174` returns raw · `tiger/sieve.py:165` compares against normalised

A "Navy Shirt" carrying `color: navy` yields `title_color="navy"`,
`declared="blue"`, and fires `flag_title_contradiction` — a signal the ablation
reports at precision 1.000.

**Fix:** normalise `_title_color`'s return value. **Requires D5.**

**Status:** DONE — `d6ecadd`.

---

### D8 · Category singularisation is fashion-only, so ABO captions are malformed
**Severity:** Medium — a competing explanation for the H11 underconfidence finding
**Where:** `tiger/text_views.py:23,35`

`CATEGORY_SINGULAR` covers `shirts/shoes/bags/hats` only. ABO categories fall
through to `rstrip("s")`, producing probe and LOO captions like *"a photo of a
red home_decor"* and *"a photo of a red electronic"* — an underscore token and a
non-noun, fed to CLIP as the whole prompt ensemble.

**Consequence:** every ABO probe margin, LOO delta, and Arbiter feature derived
from them was measured against a degraded prompt. H11 attributes ABO
underconfidence entirely to covariate shift; this is a mechanical alternative
that has not been ruled out.

**Fix:** add the ABO categories with real nouns (`home_decor → "home decoration"`,
`electronics → "electronic device"`), then re-run the confidence diagnostic
before drawing any conclusion from it.

**Status:** DONE (superseded, checked 2026-09-11) — the `home_decor`/`electronics`
category names this entry describes no longer exist anywhere in the pipeline;
Phase 1's actual category migration (`feat(Phase 1)`) picked different,
final vertical names (chair, sofa, table, ottoman, stool, rug, lamp,
light_fixture, wall_art, ring, necklace, earring, handbag, suitcase, hat) and
gave every one of them a real singular noun in `CATEGORY_SINGULAR` from the
start — confirmed by diffing the full `configs/schema.yaml` category list
against `tiger/text_views.py`'s map: zero gaps. The re-run confidence
diagnostic in the fix note is still worth doing as a general Phase 2 sanity
check, but not because of this specific bug — it no longer exists.

---

### D9 · `rstrip("s")` is the wrong primitive for singularisation
**Severity:** Low
**Where:** `tiger/text_views.py:35` · `tiger/generator.py:49`

`rstrip` strips *every* trailing `s`: `"dress" → "dre"`, `"glasses" → "glasse"`.
Harmless for the current four fashion categories, wrong for obvious next-vertical
candidates.

**Fix:** `removesuffix("s")`, or an explicit map with a fallback.

**Status:** DONE — both sites (`tiger/text_views.py`, `tiger/generator.py`)
switched to `removesuffix("s")`. 162 tests passing, unchanged.

---

### D10 · The SDXL prompt drops pattern and material before generation
**Severity:** Medium — a documented limitation is attributed to the wrong cause
**Where:** `tiger/generator.py:49-55`

```python
subject = f"{color} {cat_singular}" if color and cat_singular else caption
```

Only colour and category reach the prompt. `pattern` and `material` are accepted
as arguments and discarded.

**Consequence:** `honest_limitations.md` §2 and `paper_draft_materials.md` §4 both
explain the loss of "striped"/"printed" as diffusion models struggling with
fine-grained pattern adherence. The pattern never enters the prompt. The
limitation is real; the stated cause is a claim about SDXL that this code cannot
support.

**Fix:** include pattern and material in the prompt, regenerate the qualitative
grid, and re-assess whether the limitation survives. Correct §2 and §4 either way.

**Update (2026-09-13): a second defect in the same four lines, and the fix is
now testable without a GPU.**

The prompt was also building its category noun with a local
`category.removesuffix("s")`. `D8` exists because that class of singularisation
produces non-nouns and underscore tokens for the ABO categories, and `D8` gave
every category a real noun in `text_views.CATEGORY_SINGULAR` — but only the
caption/probe/LOO views were moved onto it. The generator kept its own copy, so
the T2V fallback was asking SDXL for a **"blue wall_art"** and a **"brass
light_fixture"**. Those are two of the categories `D8` added nouns *for*, and
the generative fallback is what produces the qualitative grid this item is
about re-generating. Now `text_views.singular()`, the same table.

Prompt construction is extracted into `generator.build_prompt(caption,
category, attrs)`, a pure function with no `diffusers` import, so what the
model is actually asked for can be pinned without a GPU. Five tests in
`tests/test_generator_prompt.py`: material and pattern reach the prompt,
`solid` is not described as a pattern, the category noun comes from the shared
table, and the caption fallback still works. There were no tests on this path
at all before, which is how a `removesuffix` survived `D8` and `D9`.

**The documentation half is done, and did not need the regen.** D10 asks to
"correct §2 and §4 either way", and the correction does not depend on what a
regenerated grid shows: the images in the current grid were produced by a
prompt that never contained the pattern, so they are not evidence about SDXL's
pattern adherence whatever they look like. `honest_limitations.md` §2 and
`paper_draft_materials.md` §4 now report the observation (patterns were lost)
and **withdraw the attribution** (that diffusion models drop them), marking the
cause open pending regeneration.

**The regeneration needs more than a GPU (checked 2026-09-13).** Earlier notes
here said the grid just needed compute. It does not — running either notebook
as-is would not regenerate it, for two independent reasons:

1. **Nothing in the repository builds `qualitative_grid_final.png`.** No Python
   module writes it and no notebook cell references it. It is a committed PNG
   (dated 2026-08-26) that predates the pipeline's current shape. The ABO
   notebook's export cell does `cp -r ... paper_figures /kaggle/working/` and
   zips that, so the file makes a round trip out of the checkout and back —
   which is why the copy in `tiger_abo_corrected_d4_check.zip` is **byte-identical**
   to the committed one despite that run having executed
   `ablate-repair --generative-fallback`.
2. **The generated images are thrown away at the end of every run.** SDXL
   writes to `{sample_dir}/images/generated/{row_id}.jpg` (`solver.py:269`), and
   the export cell copies `data/outputs`, `data/thresholds`, `data/processed`
   and `paper_figures` — not `data/sample`. This is the same gap as `E6`'s
   uncommittable `data/sample/`.

**The plumbing is now built (2026-09-13), so a single run closes this.** Three
pieces, all local:

1. **The run stops discarding its own per-row record.** The corrected notebooks
   only ever call `ablate-repair`, which summarised the five configurations and
   dropped the frames — so a full pipeline run left behind counts and no
   per-row trace at all. `repair_ablation._persist_full_run` now writes the
   Full System run's `repaired_report_seed{N}.parquet` and
   `repair_report_seed{N}.json`, the same filenames the `repair` subcommand
   writes, so a consumer does not care which command produced them.
2. **Something builds the figure.** `viz.build_qualitative_grid` draws
   clean / corrupted / repaired triptychs — image, the error subtype, the
   repair action actually taken, and the attributes at each stage. It draws
   with **PIL only**, not matplotlib: the figure should build wherever the
   pipeline runs, and it is then testable in a checkout with no plotting
   extra. Row selection is deterministic and puts a generated image first, so
   the figure can be compared with the one before it. Exposed as
   `python -m tiger.cli qualitative-grid --seed 7`, which explains what to run
   first if the artifacts are missing rather than raising a traceback.
3. **The generated images leave Kaggle.** Both notebooks' export cells now copy
   `data/sample/images/generated/` into the results zip, and both gained a cell
   that builds the grid and displays it before the export runs.

Tests: six in `tests/test_qualitative_grid.py` against a synthetic completed
run — the figure is written, the generated row is shown first, selection is
deterministic and respects `max_rows`, a run with nothing repaired returns
`None` rather than an empty image, and a missing image is drawn as a labelled
tile rather than raised (an E4 row legitimately has none).

**A fourth finding, which changes the shape of the re-assessment: the
generative fallback has never fired.** Full System and "No Generative Fallback"
are identical in **every column** of the committed ABO ablation — same repaired
count, same escalations, same accuracies — and the same holds on the synthetic
run. Turning the generator off changed nothing because it was never on any
path. The reason is in `solver.py:263`: generation is reachable only when
`pool.best_for_text(...)` returns `None`, i.e. when the catalogue holds no
other usable image in the same category. On a corpus with thousands of rows per
category that never happens, and **that is correct behaviour** — retrieving a
real photo should beat synthesising one.

So the previous run generated nothing, and a re-run will generate nothing
either. The qualitative grid will contain no synthesised image, and D10's
question cannot be answered as a byproduct of the repair run. It has to be put
to the generator directly.

`generate` is the only command that can reach the generator, and it could not
reach the thing D10 fixed: it called `generate(caption, out_path)` with no
category and no attributes, so `build_prompt` fell through to
`subject = caption` and rendered the *caption*, never the constructed prompt.
It now takes `--category` and `--attrs`, prints the prompt it used, and offers
`--pattern-panel solid,striped,dotted`, which renders one image per pattern and
contact-sheets them via `viz.build_generation_panel`. That panel is the actual
experiment §2 needs: the observation (patterns were lost) stands, the cause was
withdrawn because the pattern never entered the prompt, and this is what
decides whether the limitation survives now that it does. The ABO notebook
gained a cell that runs it.

**Run 2026-09-13. Both figures exist, and the pattern question has an answer.**

`paper_figures/generation_pattern_panel_panel.png`: the same product —
`a brown wooden dining chair`, `{"color":"brown","material":"wood"}` — rendered
at `pattern` = solid / striped / dotted, everything else held fixed.

- **The prompt fix works.** All three renders are plainly brown, plainly
  wooden, plainly chairs. Colour, material and the category noun all survive
  into the image, which is what `D8`+`D10`'s prompt work was for.
- **The pattern does not.** None of the three shows a stripe or a dot. The
  "striped" and "dotted" renders differ from "solid" only as different draws of
  the same chair.

So the limitation in `honest_limitations.md` §2 **survives**, and for the first
time there is direct evidence for it rather than an assumption. The attribution
that was withdrawn — diffusion models drop fine-grained pattern — is now
supportable, *with three caveats that must travel with it*:

1. **n = 3, one product, one category.** This is an illustration, not a
   measurement. Do not report a rate.
2. **The category may be doing the work.** A striped *chair* is a semantically
   odd request; a striped *rug* is not. ABO has rugs. Re-running the panel on
   `rug` would separate "the model ignores pattern" from "the model ignores
   implausible pattern", and until that is done the claim is confounded.
3. **SDXL-Turbo at 4 steps, `guidance_scale=0.0`** is a deliberately fast
   configuration, and low guidance is known to weaken prompt adherence. The
   finding is about *this generator as configured*, not about diffusion models.

The qualitative grid (`paper_figures/qualitative_grid_final.png`) also
regenerated, and is doing its job: six rows showing three clean repairs, one
wrong-colour image swap, one wrong material written (`stone` onto a platinum
ring), and one **clean row damaged** — subtype `clean`, flagged, routed to T2V
and given a wooden chair in place of a black leather chaise. That last row is
the honest picture of the 49% committed accuracy, and it is the row a reviewer
will find.

**Status:** DONE for what D10 asked — prompt fixed, pinned, §2/§4 corrected,
and the regeneration done with a real result. The follow-up (the same panel on
`rug`, to de-confound caveat 2) is logged as its own item rather than left
inside a closed one.

---

### D11 · The Gemini prompt cache is keyed on full base64 images and is unbounded
**Severity:** Medium
**Where:** `tiger/vlm_judge.py:167` — `cache_key = json.dumps(parts, sort_keys=True)`

`parts` contains the base64-encoded image, so every key is the size of the image
and every image is retained for the process lifetime.
`project_chronicle.md` Hiccup 2 describes this as "an in-memory LRU cache"; it is
neither LRU nor bounded. On a 1,500-image run it holds the corpus twice.

**Fix:** key on `sha1(image bytes) + sha1(prompt)`; bound it (`functools.lru_cache`
or an explicit cap). Correct the chronicle's description of what was built.

**Status:** DONE — cache key is now a 40-char sha1 digest of the request parts
(image data + prompt text) instead of the raw json-dumped base64 string; cache
is an `OrderedDict` capped at 2048 entries with LRU eviction. Smoke-tested
against a live key.

---

### D12 · `repair` always reports zero flagged products
**Severity:** Low
**Where:** `tiger/cli.py:589` — `total = summary.get("total", 0)`

`run_repair_cycle` emits `n_products`, `by_status` and `max_passes`. There is no
`total` key, so the user-facing summary always reads *"We attempted to repair the
0 flagged products."*

**Sweep note:** this is the *second* instance of the same bug. `8ba8d19` already
fixed the sibling at `tiger/eval/repair_ablation.py:135` (`total_attempted` now
sums `repaired + escalated`) as a drive-by. Only the `cli.py` site remains.

**Fix:** use `n_products`, or sum `by_status`.

**Status:** DONE — `cli.py:614` now reads `summary.get("n_products", 0)`. Both
instances of this bug (this one and the `repair_ablation.py` sibling `8ba8d19`
fixed earlier) are closed.

---

### D13 · Dead allocation in `encode_images`
**Severity:** Trivial
**Where:** `tiger/encoders.py:140-143`

A conditional `np.zeros` whose result is unconditionally overwritten thirty lines
later, with a comment already admitting it (`# simpler: resolve dim lazily below`).

**Fix:** delete it.

**Status:** DONE — removed; the real `out` allocation thirty lines down (using
the correctly-resolved `dim`) is the only one now.

---

### D14 ⚑ · Clean rows cannot be cleared — 1 dismissal in 575
**Severity:** High — this sets the pipeline's automation economics
**Where:** `tiger/arbiter.py::route` (dismiss branch) · `tiger/analyzer.py` (title check)
**Found:** 2026-09-13 · **diagnosed 2026-09-14, and the first diagnosis was wrong**

Of 575 genuinely clean rows that reached the repair cycle, **1 was dismissed**
and **564 escalated to human review**. The dismiss path — the only way the
system can say "the Sieve was wrong, this row is fine" — is effectively dead,
and 98% of clean-but-flagged rows become human work.

**The first diagnosis (recorded here 2026-09-13) was that the dismiss *guard*
cancels itself:** the Sieve flags on probe z ≤ −2.0 and the guard refuses to
dismiss on probe z ≤ −2.0, the same test on the same quantity. That reasoning
is structurally correct and it is **not** the cause. Swept offline over 13,203
flagged calibration rows (`tests/bench_dismiss_guard.py`, which needs no GPU —
dismissal is a pure routing decision over evidence already on disk):

| guard z | dismiss p | clean rows cleared | dirty rows leaked |
|---|---|---|---|
| −2.0 (shipped) | 0.80 | 15 (0.3%) | 1 |
| −3.0 | 0.80 | 63 (1.1%) | 7 |
| guard off | 0.80 | 65 (1.1%) | 9 |
| guard off | 0.70 | 170 (3.0%) | 23 |

Removing the probe guard entirely recovers **1.1%** of clean rows, not most of
them. Relaxing it to −3.0 gets nearly all of that benefit at precision 0.90, so
it is worth doing — but it was never the binding constraint, and the earlier
entry here overstated it.

**The actual cause.** Decomposing the 945 clean rows the router already calls
CLEAN at p ≥ 0.80:

| blocked by | n | share |
|---|---|---|
| `title_contradiction` | 880 | **93.1%** |
| probe z ≤ −2.0 | 52 | 5.5% |
| `text_out_of_domain` | 0 | 0% |
| nothing — dismissable today | 15 | 1.6% |

The router is *good* at this: it calls 69.9% of clean rows CLEAN, and 945 of
them confidently. One noisy signal vetoes nearly all of them. See `D15`, which
is the real defect; this item is the symptom that led to it.

**Fix:** raise the probe guard to z ≤ −3.0 (measured above: 4× the clearances,
precision 0.90) *and* fix `D15`. Neither alone is sufficient — the probe change
alone leaves 93% of the blockage in place.

**Fixed 2026-09-14, both halves.** `dismiss_contrary_z` is configurable and set
to **−3.0**, and `D15`'s text check landed. Re-swept with the title flag
recomputed from the current check (`--recompute-title`, which is honest because
the flag is a pure text function — no re-encoding needed to predict the next
run):

| | clean rows cleared | dirty leaked | precision |
|---|---|---|---|
| before (guard −2.0, old title check) | 15 (0.3%) | 1 | 0.938 |
| title check alone (guard −2.0) | 1,068 (18.6%) | 82 | 0.929 |
| **both (guard −3.0) — shipped** | **1,123 (19.6%)** | 88 (1.2%) | **0.927** |
| guard off entirely | 1,125 (19.6%) | 90 | 0.926 |

**15 → 1,123, a 75× improvement**, at a cost of 88 dirty rows dismissed out of
7,466 (1.2%). −3.0 captures essentially everything disabling the guard would
give, so the guard keeps doing its job on genuinely egregious contrary evidence
while no longer vetoing itself.

The decomposition was right: the title check was the blockage (71× of the 75×),
the guard threshold the remainder.

**Status:** DONE — pending a run to confirm end-to-end

---

### D15 ⚑ · The title-contradiction flag fires on brand names, and it fires on clean rows most
**Severity:** High — a near-anti-signal, and it is both an Arbiter feature and a dismiss veto
**Where:** `tiger/analyzer.py` title check · feature `title_contradiction`
**Found:** 2026-09-14, while diagnosing `D14`

Fire rate by ground truth on an ABO calibration seed:

| row is actually | title_contradiction fires |
|---|---|
| **clean** | **78.4%** |
| `mutate_text` (text really is wrong) | 60.1% |
| `swap_image` (image really is wrong) | 21.9% |

It fires *more often on clean rows than on corrupted ones*. As a detector of
text faults it is worse than a coin flip weighted the wrong way.

**Why**, from the rows themselves — four distinct causes, none a real fault:

| title | attributes | what the flag "found" |
|---|---|---|
| `Stone & Beam Stone Brown Swatch` | `color=gray` | **brand name** — "Stone & Beam" is an Amazon furniture brand; "stone" is a material in Ω |
| `Rivet Modern Geometric Wool Area Rug, Blue, Grey, Brown` | `color=blue` | **legitimate secondary colours** — the title lists all three, the attribute records the primary |
| `Platinum-Plated Sterling Silver ... Rings` | `color=white` | **material words in the product name** — "silver" read as a colour contradicting white |
| `Sterling Silver Genuine Garnet ... Earrings` | `color=red` | **gemstone names** — garnet *is* red; the attribute is right |

`D7` already fixed one class of false title contradictions (`_title_color`
manufacturing them) on the synthetic catalogue. Real marketing titles are a
different problem: they contain brand names, stone names, plating descriptions
and every colour in the product, and any of those collides with a flat domain
vocabulary.

**This compounds the colour vocabulary gap (`B9`)** — both are the same root
cause seen from two sides: a twelve-value flat domain applied to a catalogue
that writes "Stone & Beam", "Walnut" and "Blue, Grey, Brown".

**Fix options, cheapest first:**
1. **Mask the brand prefix** before the check. ABO titles overwhelmingly start
   `Amazon Brand – <Brand> …`; that span should never be scanned.
2. **Require the contradiction to be exclusive**: a title naming *several*
   domain values of the same field is listing, not contradicting. Only fire
   when the title names exactly one value and it differs from the attribute.
3. **Drop the material half of the check**, or restrict it to fields where the
   title's vocabulary and Ω are the same register. "Sterling Silver" is a
   product name, not a material claim.
4. Consider retiring the feature. It is 1 of 14 and currently misinforms both
   routing and dismissal; measuring routing accuracy with it removed is one
   offline run.

**Fixed 2026-09-14.** Two rules, both in `sieve._title_color`:

1. **Exclusivity.** The title must name *exactly one* colour. A title naming
   several is enumerating what is in the picture, not disagreeing with the
   attribute. This also removes an arbitrariness nobody had noticed: the old
   scan returned the first match in *schema iteration order*, so which colour a
   multi-colour title "asserted" depended on the order of `surface_forms`.
2. **Metal cues.** `silver` and `gold` are colour surface forms (→ gray, yellow)
   and also the two commonest words in jewellery product names. When the title
   carries a metal cue — `sterling`, `plated`, `platinum`, `14k`, … — they are
   material claims and do not count as colour evidence.

A third rule was tried and **dropped**: positionally masking the
`Amazon Brand – <Brand>` prefix. Measured over 39,808 rows it moved false
positives by 11 (1,578 vs 1,567 — slightly *worse*) while being able to swallow
a genuine colour word in a product name. Brand names collide with the material
vocabulary, not the colour one, so the colour check never needed it.

**Measured over 39,808 rows, 8 calibration seeds:**

| | before | after |
|---|---|---|
| fires on clean rows | 15.9% | **5.6%** |
| fires on `mutate_text` (true faults) | 34.9% | 20.0% |
| precision as a text-fault detector | 0.251 | **0.347** |

It keeps well over half its true positives while dropping two-thirds of its
false ones. The knock-on effect on dismissal is `D14`: 15 → 1,123 clean rows
cleared.

Tests: `tests/test_title_contradiction.py` — a single contradicting colour
still fires, aliases still normalise (`D7`), multi-colour titles and jewellery
metal names no longer do, and a bare "Silver Chair" with no metal cue still
counts as a colour.

**Status:** DONE — pending a run to confirm end-to-end

---

### D16 · T2V installs whichever image is most category-typical, and the verifier approves
**Severity:** High — it is how a clean row gets damaged
**Where:** `tiger/verify.py` acceptance test · `tiger/solver.py` candidate retrieval
**Found:** 2026-09-14, from the regenerated qualitative grid

10 of 575 clean rows (**1.7%**) were edited by the pipeline — made worse. The
grid contains one: `B000P2068O`, *"Strathwood Aspen Leather Chaise, Brown"*, a
clean row, whose correct black leather chaise image was replaced with a generic
wooden chair. Every check passed:

```
c_before 0.264  ->  c_after 0.341   (delta +0.077, tau 0.225)   accepted
```

**The mechanism is that the acceptance test has no notion of identity.** It asks
whether image-text similarity *improved*, and similarity is trivially improved
by installing a more category-typical photograph: a generic wooden chair matches
the word "chair" better than this specific chaise matches its own long,
specific title. A correct-but-atypical image is therefore always beatable.

**Second, related symptom: hub images.** 304 T2V installs drew on only **114
distinct donors**, and the most popular single image was installed on **18
different rows**. That is textbook retrieval hubness — a few images sit close to
many captions in embedding space and win repeatedly. A donor installed 18 times
is not 18 correct repairs.

**Three fixes were proposed, measured, and two refuted.** The negative results
are more useful than the guesses were:

1. ~~**Gate T2V on the row being a similarity outlier for its category.**~~
   **Refuted.** The rationale was "a clean row is by definition not a
   similarity outlier". Not true here: clean rows routed to T2V have mean
   `sim_z` **−2.86** (median −2.91) against **−4.25** for genuine `swap_image`
   rows — heavily overlapping. Gating at `sim_z >= -2.0` protects 13% of clean
   rows while losing 2.2% of genuine repairs; at −2.5, 21% protected for 7.3%
   lost. A bad trade — and the stated reason was wrong, because the chaise
   looked "healthy" only when 0.264 was compared against a *global* tau, which
   is the very error this item accuses the verifier of making.
2. ~~**CSLS hubness correction on retrieval.**~~ **Measured, no benefit.**
   Replayed over 3,689 `swap_image` rows using the calibration seeds'
   embeddings (`tests/bench_hubness.py`): accuracy **48.0% → 48.4%**, inside
   noise. It does fix the symptom — distinct donors 515 → 634, most-installed
   image 171 → 89 — but a less concentrated wrong answer is still wrong. Not
   shipped, on the same principle that dropped the brand regex in `D15`.
3. **The independent verifier should have caught this** — and structurally
   cannot. Split out as `D17`; it is the real lead here.

**What the hubness data does say.** Accuracy by how many rows a donor image was
installed on: 35.3% (1), 37.5% (2), 46.2% (3–4), 49.0% (5–8), **25.5% (9+)**.
Only extreme hubs are harmful and donors reused 3–8 times are the *best*, which
is why a blanket correction gains nothing. Capping at 9+ would convert 51
repairs at 25.5% into escalations, lifting T2V accuracy 38.6% → 42.7% on the
remainder — a genuine risk–coverage trade, but an install cap is order-dependent
(the 9th row wanting a donor is refused, the 1st is not), so it is not
reproducible in the way this pipeline requires. Left unimplemented deliberately.

**Conclusion.** Installing a category-typical image over a correct-but-atypical
one is not fixable at the routing or retrieval layer with CLIP-family encoders:
on this data the clean and dirty populations are not separable by any
similarity statistic available. The leverage is a product-identity-aware judge
(`D17`) or a stronger encoder (`B7`).

**Status:** TODO — two candidate fixes eliminated by measurement; the remaining
lead is `D17`


---

### D17 ⚑ · The independent verifier cannot reject a T2V repair — 0 of 304
**Severity:** High — it is presented as the pipeline's semantic safety net
**Where:** `tiger/verify.py::IndependentVerifier.check_t2v`
**Found:** 2026-09-14, while looking for what should have stopped `D16`

On the reported run the independent verifier (SigLIP) approved **304 of 304**
image repairs, including all 7 that damaged clean rows. On the text side it is
working normally — it rejected **30 of 107** V2T repairs. The asymmetry is not
a tuning accident, it is structural:

```python
def check_t2v(self, old_image_path, new_image_path, caption):
    ...
    return float(imgs[1] @ t) > old_s      # new image beats old, on caption similarity
```

The candidate was *selected* by maximising caption similarity. Asking a second
encoder whether the winner of that contest wins that contest returns yes
essentially always. The V2T check works precisely because it asks something
different — whether the independent encoder's *own probe* predicts the patched
value — so it can and does disagree.

This is the F7 circularity the independent verifier exists to cure, surviving
in the half nobody measured. `paper_concepts.md` §6 describes it as "a final
semantic safety checkpoint ... catching wrong-direction repairs"; on T2V it
has never caught anything.

**Proposed fix (designed, not shipped — it cannot be validated without a run):**
ask the independent encoder to agree on the *choice*, not on the direction.
Pass it CLIP's top-2 candidates and require the winner to beat the runner-up
under the independent encoder too. Cross-encoder disagreement about *which*
candidate is best is genuine evidence, and it is the same "two estimators must
agree or we abstain" principle the V2T path already uses (`B6`). Cost is one
extra image encode per repair.

Not shipped because `torch`/`transformers` are unavailable in this checkout, so
the change could not be measured — and shipping an unmeasured behaviour change
is what this backlog exists to stop.

**Status:** TODO — diagnosed and designed; needs a run to validate


---

## E. Documentation contradicted by the code

Fix **after** the corrected ablation run — the numbers will move.

### E1 · `reviewer_defense.md` Attack 8: "exactly four continuous evidence metrics"
The `FEATURES` list in `tiger/arbiter.py` has **14**. The rebuttal also argues
from "a 4-dimensional input space". Trivially checkable by any reviewer.
**Status:** DONE — Attack 8 rewritten to cite the real 14-feature list by name.

### E2 · `reviewer_defense.md` Attack 3: the 47.4% were *not* "safely escalated"
52.6% is restoration accuracy among the **163 repaired** rows; escalated rows
are the separate 269. Those ~9 cases were **committed with wrong values**. The
rebuttal presents a real error rate as a safety guarantee — the most dangerous
line in that document.

**Status:** DONE — rewritten. Attack 3 point 2 no longer calls the 47.4%
"safely escalated"; it now states plainly that these are rows TIGeR committed
and got wrong, that the 269 escalated rows are a disjoint population already
excluded from the 163-row denominator, and links to `B6` (built 2026-09-11,
after this entry was written — the finding it references, that nothing
abstained on value-level uncertainty, is now addressed by B6's
disagreement-escalation mechanism going forward) as the honest explanation for
why these errors were committed at all in the run these figures describe.
Exact percentages are flagged as pending re-measurement under Phase 2 — the
qualitative correction
does not depend on the exact numbers.

### E3 · "Perfectly overlapped" was never demonstrated
Identical aggregate counts do not prove identical *sets*; two different sets of
444 items produce the same total. Given A1, the claim is moot — but if the
early-exit story survives a corrected run, prove it by intersecting the
escalated `row_id` sets.

**Resolution (2026-09-12):** didn't survive a corrected run — no need for the
set-intersection proof. The corrected ABO run shows Full System and No Gamma
Gate producing genuinely different counts (268 vs. 498 repaired), so there's
no identity left to demonstrate anything about. `paper_draft_materials.md`
§7.5 formally withdrawn.

**Status:** DONE

### E4 · Attack 1 argues cost where the durable argument is reliability
The "why not just use a VLM" rebuttal rests on cost, throughput, and rate
limits. Cost arguments age badly. MLLM-as-a-Judge (ICML 2024) shows judges
diverge from humans on *absolute scoring* and exhibit position, egocentric, and
length bias plus hallucination even in GPT-4V — a reliability argument that does
not expire.
**Status:** DONE — Attack 1 rewritten to lead with the MLLM-as-a-Judge
reliability argument; cost/throughput retained as secondary.

### E5 · Attack 6 does not answer the ARO objection
The model-agnostic defence answers "why not a newer model" but not "why the
model measured worst at your exact task" (CLIP 62% vs BLIP 88% on attribution).
Pairs with B7.
**Status:** DONE — Attack 6 now cites the ARO 62%/87%/88% figures directly and
names the two mitigations already available without retraining
(`compare_encoders`, the Koishigarina et al. linear correction).

---

### E6 · The docs route readers to six paths that do not exist
**Severity:** Medium — first-contact credibility, and it is the reproducibility surface

| Referenced | By | Reality |
|---|---|---|
| `data/sample/` "committed" | `README.md`, `tiger_project_doc.md` §2 | absent, untracked |
| `data/thresholds/` "committed" | `README.md`, `tiger_project_doc.md` §2 | absent, untracked |
| `scripts/` legacy MVP "kept for reference" | `README.md`, `tiger_project_doc.md` §2 | deleted |
| `.github/workflows/ci.yml` | `ROADMAP_PROGRESS.md` 4.2 (marked ✅) | absent |
| `kaggle_workflow.ipynb` "in the repository root" | `README.md:132`, `tiger_project_doc.md` §7 Option B | absent; the real notebooks are `tiger.ipynb` and `tiger_abo.ipynb` |
| `[ROADMAP_PROGRESS.md](ROADMAP_PROGRESS.md)` | `README.md:17` | broken link; the file is in `paper_assets/` |

The two directories advertised as committed are precisely the two that would make
the repo reproducible. Pairs with C5.

**Fix:** commit the sample catalogue and locked thresholds (both small), correct
every path, drop the `scripts/` and `ci.yml` claims, and remove the surviving
"✅ Unit tests written" line in `tiger_project_doc.md` §11 (C3).

**Status:** DONE (partial) — `data/thresholds/` committed with real artifacts
(excluding `tiger_fusion.json`, which must stay absent by design — a test
depends on `_load_fusion` failing loudly when fusion hasn't been calibrated).
`data/sample/` still not committed: the notebooks don't export it (checked —
neither zip's export cell includes it), so it needs a dedicated export step
added first, not available from this session's run artifacts. README.md and
`tiger_project_doc.md` corrected: `scripts/` marked deleted not "kept for
reference", `kaggle_workflow.ipynb` (doesn't exist) replaced with the real
notebook names, broken `ROADMAP_PROGRESS.md` link fixed, "✅ Unit tests
written" line removed. `ci.yml` claim not found in current docs to remove.

---

### E7 · Three documents name three different final verifiers
**Severity:** Medium

`pipeline_architecture.md` §5 says "Gemini 3.7 Flash or Local SigLIP";
`project_chronicle.md` §2 and `ROADMAP_PROGRESS.md` decision 6 say SigLIP was
selected *over* Gemini; `README.md`'s end-to-end example is
`repair --seed 7 --vlm-judge`, i.e. Gemini. The README's own worked example does
not run the configuration the paper reports — and per A2 it currently cannot run
at all. "Gemini 3.7 Flash" is not a real model.

**Fix:** name SigLIP as the reported verifier everywhere, change the README
example to `--independent`, and describe Gemini as an evaluated alternative.

**Status:** DONE — README's end-to-end example changed to `--independent`
with a note that `--vlm-judge` was evaluated but SigLIP is reported.
`pipeline_architecture.md` already fixed earlier this session (removed the
fake "Gemini 3.7 Flash" model name). `project_chronicle.md` and
`ROADMAP_PROGRESS.md` checked — both already correctly say SigLIP was
selected over Gemini; no fix needed there.

---

### E8 · Ablation-table denominators move by 17 rows with no explanation
**Severity:** Medium — a reviewer will subtract these
**Where:** `paper_assets/paper_draft_materials.md` §1

| Config | Repaired | Escalated | Sum |
|---|---|---|---|
| No Arbiter | 168 | 266 | 434 |
| No Independent Verifier | 176 | 256 | 432 |
| No Generative Fallback | 148 | 269 | **417** |
| No Gamma Gate | 163 | 269 | 432 |
| Full System | 163 | 269 | 432 |

Same sample, five different totals. The cause is benign — `_evaluate_run`
(`tiger/eval/repair_ablation.py:96`) reports only `repaired` and `escalated`,
silently dropping `dismissed`, `acquire_image` and `unrepaired` — but the table
presents the two columns as exhaustive.

**Fix:** report all statuses, or add a `Total` column with a footnote. Regenerate
after A1.

**Status:** DONE — `_evaluate_run` now extracts `dismissed`/`acquire_image`/
`unrepaired` from `by_status` (previously discarded), and
`save_repair_ablations_csv` reports all of them plus a `Total (All Statuses)`
column alongside the pre-existing `Total Attempted`. `paper_draft_materials.md`
§1's table marked invalidated (see `E3`) rather than patched — it's Fashion
data from the pre-A1 harness, not something to regenerate against ABO.

---

### E9 · The ablation credits LOO masking for the contrastive probes' result
**Severity:** Medium — the paper mis-credits its own strongest contribution
**Where:** `tiger/eval/ablation.py:91,140` · `paper_assets/paper_concepts.md` §2

`CONFIGS["probes_only"]` is `flag_probe_{color,material,pattern}` and
`CONFIGS["no_loo"]` is everything *except* those — both are the per-field
contrastive probes. They print as "LOO Probes Only" and "No LOO Masking", and the
takeaway line reads "Adding LOO Masking: +X F1".

Eq. 18 leave-one-out lives in `tiger/analyzer.py` and runs only on already-flagged
rows. **It contributes nothing to detection.** The mechanism that produces the
project's strongest verified result — mutate_text recall 0.267 → 0.853 — is the
contrastive probes. `paper_concepts.md` §2 presents LOO masking as the headline
detection contribution.

**Fix:** rename the ablation rows to "Probes Only" / "No Probes"; rewrite
`paper_concepts.md` §2 to credit the per-field contrastive probes for detection
and describe LOO as field attribution for routing. This correction runs in the
project's favour — the probe result is stronger and more novel than the LOO story.

**Status:** DONE — `tiger/eval/ablation.py`'s display labels renamed exactly
as specified (internal dict keys `probes_only`/`no_loo` left as-is — already
accurate at the code level, only the printed labels and the "Adding LOO
Masking" takeaway line were wrong). `paper_concepts.md` §2 rewritten to credit
contrastive probes for detection and reframe LOO as routing-side field
attribution. Also caught and fixed a second, previously-unknown bug while
here: §1 of the same document had E1/E2 defined backwards relative to
`tiger/arbiter.py`'s authoritative convention.

---

### E10 · `tiger_project_doc.md` §12 line counts are stale
**Severity:** Low

`cli.py` 505 → 732 · `vlm_judge.py` ~165 → 255 · `solver.py` 209 → 251 ·
`schema.py` 112 → 118. The table omits `generator.py`, `viz.py` and
`eval/repair_ablation.py` entirely.

**Fix:** regenerate, or drop the line-count column — it ages on every commit.

**Status:** DONE — regenerated via `wc -l` (2026-09-12); added the three
previously-omitted files (`generator.py`, `viz.py`, `eval/repair_ablation.py`).
Still ages on every commit as noted in the fix — no code change can prevent
that, only re-running `wc -l` before each doc update.

---

### E11 · A planted always-flagged row sits inside the reported synthetic metrics
**Severity:** Medium — undisclosed evaluation artefact
**Where:** `tiger/data/synthgen.py:238` · `tiger/data/noise.py:274`

`forced_gen_000` is appended to the catalogue with sentinel `category: "uniforms"`
(not in `schema.categories`), `color: magenta` and `material: velvet` (neither in
Ω), hard-assigned to the **report** split, then unconditionally image-blanked by
the injector regardless of seed or configured rate.

It is guaranteed flagged (unknown category → `flag_text_out_of_domain`),
guaranteed to have no T2V candidate (its category is unique), and guaranteed to
fail Eq. 27 at verify. It exists to force the generative-fallback branch to run.

**Consequence:** defensible as a smoke test, but it is a hand-placed row inside
every reported detection number for the synthetic catalogue, and no document
mentions it.

**Fix:** move it to a fixture behind a smoke test, exclude it from reported
metrics, or disclose it in the evaluation setup — then confirm the synthetic
numbers are unchanged.

**Status:** DONE (disclosure only) — added as `honest_limitations.md` §5, the
cheapest of the three fix options. Moving it to a dedicated fixture (removing
it from the shared catalogue entirely) is still open if wanted, but no longer
undisclosed.

---

### E12 · `swap_image` recall is quoted at two granularities as one number
**Severity:** Trivial

`README.md` quotes `swap_image 0.975` — the coarse **label**, and correct.
`tiger_project_doc.md` §8 places 0.975 in a table of **subtypes**, where the
verified value is 0.983 (59/60), with `swap_image_same_category` as the separate
0.950 row.

**Fix:** one number per granularity, each labelled with which it is.

**Status:** DONE — `tiger_project_doc.md` §8's subtype table now shows 0.983
(59/60) labelled "subtype-level", distinct from README's 0.975 coarse label.

---

## F. Found during the Phase 2 Kaggle dry run (2026-09-11)

Not in the original 50-item audit. Both notebooks were run on Kaggle and both
crashed early enough that every downstream cell ran on empty/partial data —
these are new, higher-severity findings than most of Sections A-E, discovered
only once a real run was attempted.

### F1 · `synthgen` crashes on the first non-fashion category — blocks the entire synthetic notebook
**Severity:** Critical — blocks Phase 2 entirely for `tiger_corrected_run.ipynb`
**Where:** `tiger/data/synthgen.py` — `MATERIALS`, `SIZES`, `TITLE_NOUNS` dicts

Phase 1 added 15 ABO categories to `configs/schema.yaml`'s category list, and
`synthgen.generate()` already iterates `schema.categories` directly (matching
the pattern already used for `color_domain = schema.domain("color")`) — but
`MATERIALS`, `SIZES`, and `TITLE_NOUNS` were never extended past the original 4
fashion categories (`shirts`, `shoes`, `bags`, `hats`). The loop hit `chair`
(the first category in the schema's list) on its very first iteration and
raised `KeyError: 'chair'` before writing a single row. Everything downstream
in the Kaggle run — `calibrate`, `train-arbiter`, `sweep`, `ablate`, and the
`ablate-repair` cell F2/this session added — executed against an empty
`data/sample/` directory. None of the numbers from that run are usable.

**Fix:**
- `MATERIALS`: categories outside the curated fashion dict fall back to
  `schema.domain("material")` — the schema's own canonical 17-value material
  list, not invented.
- `SIZES`: categories outside the fashion dict get no size (`""`) — matching
  `schema.yaml`'s own comment that "the ABO furnishing and accessory verticals
  carry no size enum, so they are unconstrained."
- `TITLE_NOUNS`: falls back to `text_views.singular(category).title()`,
  reusing the utility already fixed for `D9` instead of a fourth hand-written
  dict.

Regression test added: `tests/test_synthgen_categories.py` asserts every
schema category generates at least one row, and that non-fashion categories
get an empty size.

**Status:** DONE — 165 tests passing (162 + 3 new).

---

### F2 · ABO import falls back to non-English titles, which can exceed CLIP's token budget
**Severity:** Critical — blocks `tiger_abo_corrected_run.ipynb` at `calibrate`
**Where:** `tiger/data/import_abo.py` — `_extract_english_value`

`calibrate` crashed with `ValueError: 105 text(s) exceed the CLIP token limit
(77)`, first offender a Japanese product title. `_extract_english_value`
prefers an "en*"-tagged name but falls back to the first available language
when no English variant exists — a reasonable-looking choice that collides
with `assert_token_budget` (the guardrail `F1`/`F2` from the original roadmap
built specifically to catch silent CLIP truncation): CJK text tokenizes far
less efficiently than English, so even a modest-length Japanese title can blow
past 77 tokens. 105 such products existed in the imported slice, and the first
one aborted the entire run.

**Decision (user, 2026-09-11):** exclude products with no English title,
rather than truncating them. Simpler, avoids producing a mangled mid-word
title, and costs ~105 products out of the imported set.

**Fix:** `_extract_english_value` gained a `require_english` flag; the title
call site (only) passes `require_english=True`, so a product with no English
name returns `None` and is skipped by the existing `if not title: continue`.
Color/material/product_type extraction elsewhere in the same file are
unchanged — they go through further schema-based matching downstream and
aren't subject to the same hard token-length constraint.

Regression test added: `tests/test_import_abo_formats.py::test_non_english_only_title_is_excluded`.

**Status:** DONE — 165 tests passing.

---

## Summary

| Section | Items | Done | Open | Parked / blocked / withdrawn |
|---|---|---|---|---|
| A. Measurement correctness | 9 | 9 | — | — |
| B. Repair accuracy | 9 | 7 | B7 (wired, needs a run to choose an encoder) | B1 needs fashion imagery |
| C. Config & reproducibility | 8 | 7 | — | C7 discarded (not applicable) |
| D. Robustness & design | 13 | 11 | D10 (prompt fixed and pinned, regen needs a GPU) | D3 withdrawn |
| E. Documentation | 12 | 12 | — | — |
| F. Found in Phase 2 dry run | 2 | 2 | — | — |
| **Total** | **53** | **48** | **2** | 0 parked · 1 blocked · 1 withdrawn · 1 discarded |

**Counts re-verified 2026-09-13** by parsing every `### <id>` header and its
next `**Status:**` line programmatically (not hand-tallying, and not a plain
grep count either — this run had 4 status values that aren't literally
`DONE`/`TODO`: `WITHDRAWN`/`BLOCKED`/`DISCARDED` counted separately, B0's
`✅ **DONE**` counted as done, `DOING` counted as open). Sums to exactly 53:
52 plus `B8`, found while measuring `B2`/`B5`.

**Both remaining open items are open for the same reason, and it is not a code
reason.** `B7` needs a run to choose an encoder; `D10` needs a GPU to
regenerate a figure. Everything decidable from code or from the data in this
repository is decided.

**Section A is fully closed** — A2's model pin is verified against a live key
as of 2026-09-11 (see A2's entry).
The measurement instrument is now trustworthy, so B and the remaining sections
can be measured against a baseline that means something.

**Test suite: 203 passing** (re-verified 2026-09-13, `.venv/bin/python -m pytest`;
178 + 25 from the `B2`/`B3`/`B5`/`B8` estimator work, `B4`'s region gate, `D1`'s
calibration report, `B7`'s probe-encoder split and `D10`'s prompt builder).
**Section E fully closed** the same day — see each entry above; the only
partial one is `E6` (data/sample still not committed — the notebooks don't
export it).
Every code fix in this pass and the 2026-09-11 housekeeping pass below was
verified against it; none changed behaviour the suite did not already pin.

---

## Housekeeping pass — 2026-09-11

Everything the roadmap's "Housekeeping (any time)" list flagged as independent
of the Phase 2 corrected run, plus `E2` (explicitly exempted from waiting on
Phase 2), done in one pass: `A2` (verified), `C1`, `C2`, `C4`, `C6`, `D2`, `D9`,
`D10` (partial), `D11`, `D12`, `D13`, `E2`. Also resolved the ABO licence
discrepancy noted in `ROADMAP.md` 4.6 (cite CC BY 4.0 — see that entry).

Left alone and why:
- **C5** — needs `data/outputs/` artifacts that exist only on the original dev
  machine, not this checkout.
- **C7** — the 73MB installer isn't present in this checkout to delete.
- **D1** — needs a trained Arbiter model to produce a reliability curve; that
  only exists after Phase 2.
- **D8, D3-family Section E items (E1, E3, E4–E12 except E2)** — left as
  sequenced: the roadmap deliberately holds most of Section E until Phase 2
  produces final numbers, so as not to correct documentation twice. E2 was the
  one explicit exception (a qualitative correction independent of any number).
- **B2, B3, B5, B7** — Section B is sequenced behind the B0 estimator-
  attribution report, which Phase 2 produces; B2/B5 are additionally blocked on
  local access to `data/raw/abo/`, not present in this checkout.
**2026-09-12:** both remaining ⚑ decisions are now closed. `B6` was built the
same day as this pass (see its entry) — judged low-risk enough to implement
ahead of sizing. `D4` was sized against the real Phase 2 run (E3 = 239/1703
routed rows, ~14% — not rare) and then built too; see its entry for the fix
and why it required re-running both notebooks.

171 tests passing throughout (165 after the housekeeping pass above, +2 for
B6, +4 for D4); no behaviour changed that the suite did not already pin.

**Done in this pass:** `C3` → `A4` → `A1` → `A3`/`A3b` → `A5` → `A6` → `A8` →
`A7` → `A2` (partial) → `C8`. Section A is closed bar A2's model pin.

**Next action:** the pre-baseline set is now complete — `A4`, `A1`, `A3`/`A3b`,
`A5`, `A6`, `A8`, `A7`, `D5`, `D6`, `D7` have all landed, and `synthgen` builds
again. Run, in order:

```bash
python -m tiger.cli synthgen && python -m tiger.cli calibrate
python -m tiger.cli train-arbiter && python -m tiger.cli calibrate-fusion
python -m tiger.cli ablate-repair --independent --generative-fallback
```

Then read the B0 estimator attribution report: it decides whether `B7` (encoder
path) is worth pursuing while B1/B2 stay blocked on data.

Then the numbers move, and `E2`, `E3`, `E8` and the ⚑ decisions can be settled
against figures that mean something. Everything in `E` should wait for that run;
the current values in `paper_assets/` are the ones this pass invalidated.

`D4` and `B6` were **parked** as of this writing (2026-09-08/10) — the two ⚑
design changes excluded from the fix pass by decision, not by oversight.
**Update: both have since been built** — `B6` on 2026-09-11, `D4` on
2026-09-12 once the real Phase 2 run showed E3 was ~14% of routed rows, not
the rare case the original 2% synthetic-injection figure suggested. See each
entry for the fix.
