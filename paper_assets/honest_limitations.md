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
**The Limitation:** The Arbiter is a Logistic Regression model trained on a specific distribution of simulated noise. When evaluated on out-of-domain data (ABO), the model exhibited severe underconfidence, with over 75% of predictions falling below the default $\gamma=0.60$ threshold. 
**Why it matters:** If deployed blindly to a new domain without recalibration, the pipeline loses its automation capabilities, escalating almost every flagged item back to humans. 
**Future Work:** While we mitigated this via a lightweight 25th-percentile recalibration step, future architectures could replace the rigid Logistic Regression model with a more robust uncertainty-quantification mechanism, such as conformal prediction or Bayesian neural networks, to naturally adapt to shifting confidence distributions.

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
**Why it matters:** the repair-side numbers in `paper_assets/results/abo/` were
all produced *before* those fixes, so they measure the old estimator. They are
not wrong as a record of that run — the run manifest and diagnostics are
committed — but they are not the system's current behaviour and should not be
quoted as such.
**Status:** the fixes are measured on the synthetic renderer (68.3% → 97.1% as
rendered, 22.5% → 90.0% with the product moved off-centre into a non-square
frame; `tests/bench_color_estimator.py`). Real photography needs a re-run: the
raw ABO release is no longer on the development machine and lives only on
Kaggle.
