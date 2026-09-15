# TIGeR: Honest Limitations & Future Work

No system is perfect, and peer reviewers expect a critical, honest appraisal of a system's boundaries. This document outlines the true limitations of TIGeR, which you can use to write a strong, scientifically mature "Limitations & Future Work" section in your paper.

---

### 1. Schema Rigidity and the Open-World Problem
**The Limitation:** TIGeR relies on a predefined `schema.yaml` to dictate what attributes exist and which are required (e.g., `color`, `material`). As demonstrated by our initial ABO run, if the pipeline encounters a product category that breaks these assumptions (e.g., trying to enforce a `color` requirement on a motherboard), the system will safely escalate, but it will fail to automate the repair.
**Why it matters:** TIGeR cannot dynamically discover or infer *new* attributes that are not in its schema.
**Future Work:** Future iterations could integrate an LLM to dynamically generate or adapt the schema by reading the raw textual descriptions in a zero-shot manner, bridging the gap between structured databases and unstructured text.

### 2. Generative Fallback Detail Loss
**The Limitation:** SDXL-Turbo is highly effective at generating macro-level replacements (e.g., generating a picture of a "red polo shirt"), rescuing 15 products from being complete dead-ends in our evaluation. The generated images did lose fine-grained pattern: asked for a striped shirt, the grid shows solid ones.
**The stated cause is withdrawn (2026-09-13).** This was previously attributed to diffusion models struggling with fine-grained pattern adherence. Our own code cannot support that attribution: the prompt builder passed only colour and category to SDXL and discarded `pattern` and `material` before generation, so the model was never asked for the stripes it is blamed for dropping. It was also asking for a `"wall_art"` and a `"light_fixture"` — underscore tokens rather than nouns — on two ABO categories. Both are fixed (`FIXES.md` D10), but every generated image currently in the qualitative grid predates the fix, so it is evidence about neither SDXL's pattern adherence nor this pipeline's. The limitation is real and observed; its cause is open until the grid is regenerated.
**Why it matters:** The generated image might fulfill the schema's basic constraints to pass the VLM Judge, but it loses the high-fidelity commercial nuance of the original product. And a limitation attributed to the wrong component is a limitation nobody fixes.
**Future Work:** Regenerate the grid with the corrected prompt and re-assess. Independently of the outcome, this validates our design decision to use text-to-vision (T2V) k-NN image swapping from the existing database *first*, using Generative Fallback strictly as a last resort; if pattern loss survives the corrected prompt, control-nets or mask-based inpainting are the next step.

### 3. Covariate Shift and Classifier Underconfidence
**The Limitation:** The Arbiter is a Logistic Regression model trained on a specific distribution of simulated noise, and its confidence on out-of-domain data sits almost exactly on the operating threshold. Measured on the held-out ABO calibration seed (n = 1,318): mean max-p **0.615**, median **0.609**, against γ = 0.60. **47.7%** of flagged rows fall below the gate.
*(An earlier version of this entry said "over 75%". That figure came from a pre-correction run and is superseded; see `paper_draft_materials.md` §7.4.)*
**Why it matters:** with the gate sitting at the mode of the confidence distribution, roughly half of all flagged rows escalate on confidence alone. That is a defensible operating point for a system built around abstention — but it should be presented as a *choice*, with the distribution shown, not as a tuned optimum.
**Future Work:** replace the point-estimate confidence with a mechanism that adapts to shifting distributions — conformal prediction or a Bayesian treatment — so the operating point transfers across domains rather than needing per-domain recalibration.

### 4. Dependency on Calibration Data Quality
**The Limitation:** Both the Sieve (for multimodal similarity thresholds) and the Arbiter (for routing probabilities) require a calibration dataset. TIGeR assumes that this calibration data (even when corrupted synthetically) is representative of the actual catalogue.
**Why it matters:** If a real-world catalog is *already* so massively corrupted that the "clean" baseline is noisy, the Sieve thresholds will become extremely wide, leading to false negatives (failing to flag actual anomalies). 
**Future Work:** Exploring unsupervised or self-supervised anomaly detection methods that do not require clean calibration splits would make the pipeline more resilient to heavily degraded starting states.

### 5. A planted smoke-test row is present in the synthetic catalogue's reported metrics
**The Limitation:** `tiger/data/synthgen.py` appends one hand-placed row
(`forced_gen_000`) to the synthetic catalogue, assigned a sentinel category
(`"uniforms"`, not in `schema.categories`) with a colour/material combination
outside the domain, hard-assigned to the report split, and unconditionally
image-blanked regardless of noise seed or configured rate. It exists to force
the generative-fallback code path to exercise during evaluation, since no
other guaranteed trigger for that branch otherwise exists — it is guaranteed
flagged, guaranteed to have no T2V candidate, and guaranteed to fail the Eq.
27 schema gate.
**Why it matters:** it is defensible as a smoke test, but it was never
previously disclosed anywhere in the documentation, and it sits inside every
reported detection number computed on the synthetic report split.
**Disclosure, not yet fixed in code:** at the scale of a 570+ product
synthetic catalogue, one row's effect on pooled precision/recall/F1 is not
expected to be material, but the honest position is to say so explicitly
rather than let a reviewer discover an undisclosed planted row themselves.
Cleaner options for a future pass: move it behind a dedicated fixture/smoke
test instead of the shared catalogue, or exclude it from reported metrics
explicitly and confirm the numbers are unchanged either way.

### 6. The committed repair numbers predate the colour estimator's repair
**The Limitation:** Every V2T repair writes a value produced by
`tiger/colors.py`. On the corrected ABO run that estimator was reading the
studio background rather than the product: `gray`, `multicolour` and `white`
were 73% of its output on a furniture catalogue, correct 37.1%, 1.7% and 13.8%
of the time respectively. Four defects were behind it — no product
localisation, a global white-discount rule, an aspect-destroying resize, and a
saturation test that mis-binned near-black pixels as blue (`FIXES.md`
B2/B3/B5/B8). All four are fixed and pinned by tests.
**RESOLVED (2026-09-16).** The committed numbers are now post-fix, and the
honest result is mixed — worth reporting as such rather than as a clean win.

On the synthetic renderer the fixes are decisive: **68.3% → 97.1%** as
rendered, **22.5% → 90.0%** with the product moved off-centre into a non-square
frame (`tests/bench_color_estimator.py`).

**On real ABO photography the gain is much smaller: 23.9% → 27.3%.** The
localisation works — 87% of rows were genuinely localised — but it does not
help, because localised and unlocalised rows score the same (27.5% vs 26.9%).
ABO product shots are mostly *already* centred and frame-filling, which is
precisely the assumption the old fixed crop made. The synthetic benchmark
measured a real defect that is not the binding constraint on this corpus.

The one fix that clearly transferred is the near-black one: `black` does not
appear in the pre-fix run's six most common pixel outputs at all — black
products were being hue-binned as blue and green — and post-fix it is the
fourth most common at **66.7%** correct, the highest of any colour.

**What the estimator's remaining error actually is:** `multicolour` is the most
common pixel verdict (28% of estimates) and is right **1.7%** of the time, and
`orange` is 0/9 (wood grain). That is a *vocabulary* limit — a twelve-value
flat colour domain cannot describe a patterned rug or a wood grain — not a
localisation one, and it bounds what any estimator behind that schema can
achieve (`FIXES.md` B9).


### 7. Dismissal is implemented, measured, and switched off

**The Limitation:** The Arbiter can in principle recognise a Sieve false
positive and dismiss it, sparing a human the review. Measured end to end on
ABO, that path was right **39%** of the time — and got *worse* as its threshold
rose (0.375 at p ≥ 0.85, 0.250 at p ≥ 0.90).

**The cause is specific and worth reporting.** The rule reads `p(CLEAN)` as
"probability this row is clean". On held-out data that quantity does not rank
cleanliness: rows the router calls CLEAN are actually clean **68–83%** of the
time whether it states 0.55 or 0.92. The head is flat, so no threshold can
separate the populations. The aggregate calibration hides this completely —
overall holdout ECE is a healthy **0.027**.

**Why it matters:** a wrongly dismissed dirty row leaves the pipeline unseen by
anyone, which is the single outcome this system is built to prevent. A wrongly
escalated clean row costs a reviewer seconds and is recoverable. Given that
asymmetry, a 61%-wrong silent-drop path is not a capability, so
`arbiter.dismiss_enabled` ships **false** and every flagged clean row escalates
(`FIXES.md` D19).

**What this costs:** the honest human-review figure includes rows the system
could plausibly have cleared. We prefer to report a larger review burden than a
smaller one obtained by silently discarding data.

**Status:** the path is intact, tested, and one config line from returning if a
future corpus calibrates better. The calibration report now emits **per-class**
figures so a flat head cannot hide behind a healthy aggregate again.

### 8. Two encoders can agree on the same wrong image

**The Limitation:** 9 of 313 clean ABO rows (**2.9%**) were edited by the
pipeline — made worse. The qualitative grid contains one: a correct photograph
of a black leather chaise replaced by a generic wooden chair, with every check
passing.

**The mechanism:** image candidates are selected by caption similarity, and a
category-typical photograph matches the words better than a specific, atypical
product matches its own long title. **Being typical beats being correct.**

**Three fixes were tried; two were refuted by measurement.** Gating replacement
on the row being a similarity outlier fails because the damaged rows *are*
outliers (mean `sim_z` −2.86, against −4.25 for genuinely broken rows). CSLS
hubness correction halves donor concentration (most-reused image 171 → 89) and
does not improve correctness (48.0% → 48.4%, inside noise). Requiring the
independent encoder to also prefer the chosen image over the runner-up did
help — damage fell 10 → 7 — but the residue is rows where **both encoders agree
on the same wrong candidate**, and no question posed in similarity space
separates those.

**Why it matters:** this is a discrimination ceiling of CLIP-family encoders on
fine-grained product identity, not a routing bug, and it will not be fixed by
threshold work. The untried lever is an identity-aware judge (a multimodal LLM,
wired as its own ablation row but deliberately left off given its rate limits)
or a stronger encoder.
