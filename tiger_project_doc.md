# TIGeR — Technical Documentation (Living Document)

> **Last updated:** 2026-08-20  
> **Status:** Project Complete — Pipeline validated on Fashion Product Images dataset and ready for manuscript drafting.

---

## 1. What is TIGeR?

**TIGeR** (Text-Image Generative Repair) is an end-to-end pipeline that automatically detects and repairs mismatches between text descriptions and images in multimodal product catalogues.

**The problem it solves:** Large product databases accumulate errors over time — a "red shirt" listing ends up with a blue shirt image, a product image gets swapped with a competitor's, or an image goes missing entirely. These errors hurt downstream AI models trained on the data and degrade customer experience.

**What TIGeR does:**
1. Scans every product record and flags suspicious text-image pairs using AI
2. Diagnoses *what type* of mismatch each flagged record has
3. Decides the safest repair strategy automatically
4. Executes the repair and verifies it actually improved the record
5. Escalates uncertain cases to a human reviewer

---

## 2. Repository Layout

```
TIGeR-Text-Image-Generative-Repair/
│
├── tiger/                     ← Active codebase (everything current lives here)
│   ├── cli.py                 ← Unified command-line entry point
│   ├── sieve.py               ← Stage 1: Detection
│   ├── analyzer.py            ← Stage 2: Evidence gathering
│   ├── arbiter.py             ← Stage 3: Routing decision
│   ├── solver.py              ← Stage 4: Repair planning
│   ├── verify.py              ← Stage 5: Acceptance gating
│   ├── repair.py              ← Closed loop orchestrator
│   ├── vlm_judge.py           ← Gemini VLM judge (Phase 6.4, NEW)
│   ├── schema.py              ← Attribute domains Ω_j and constraint set C
│   ├── encoders.py            ← CLIP / SigLIP encoder wrapper with caching
│   ├── colors.py              ← HSV dominant colour estimator
│   ├── fusion.py              ← Per-signal precision-floor calibration
│   ├── text_views.py          ← Caption construction, token budgeting
│   ├── data/
│   │   ├── synthgen.py        ← Synthetic catalogue generator
│   │   └── noise.py           ← Error injector (self-verifying)
│   └── eval/
│       ├── detection.py       ← Precision / Recall / F1 with bootstrap CIs
│       ├── ablation.py        ← Baselines and ablation table
│       └── encoder_compare.py ← CLIP vs SigLIP probe accuracy comparison
│
├── configs/
│   ├── tiger.yaml             ← All pipeline parameters (single source of truth)
│   └── schema.yaml            ← Attribute domains and constraints
│
├── data/
│   ├── sample/                ← Bundled synthetic catalogue (240 products, committed)
│   ├── thresholds/            ← Locked calibration artefacts (JSON, committed)
│   └── outputs/               ← Run outputs (gitignored)
│
├── tests/                     ← 73 unit tests; no GPU/network required
├── scripts/                   ← Legacy MVP (reference only, do not use)
├── .env                       ← API keys (gitignored, NEVER committed)
├── pyproject.toml             ← Package definition, extras
├── ROADMAP_PROGRESS.md        ← Checkpoint-by-checkpoint progress log
└── README.md                  ← Public-facing overview
```

---

## 3. Pipeline Architecture

The pipeline is a **closed detect → diagnose → route → repair → verify loop**:

```
Sieve → Analyzer → Arbiter → Solver → Verify
                      ▲                   │
                      └──── re-route ◄────┘  (rejected repairs escalate)
```

Each stage has a strict contract: it only does its own job, and passes a clean data structure to the next stage. This makes each stage independently testable and replaceable.

---

## 4. Stage-by-Stage Breakdown

### Stage 0: Schema & Constraint Set C (`tiger/schema.py`)

Before any pipeline runs, the **Schema** defines the rules of the universe:
- **Attribute domains (Ω_j):** Legal values for each field. E.g., `color ∈ {red, blue, green, ...}`.
- **Constraint set C:** Cross-field rules. E.g., "if category=shoes, material must be leather or canvas".
- **Normalization:** Aliases (e.g., "grey" → "gray") are resolved consistently.

Every patch the pipeline proposes is validated against Ω_j and C before it is applied. An out-of-domain or constraint-violating patch is refused and the row escalates to human review.

**Config file:** `configs/schema.yaml`  
**Key methods:** `schema.in_domain(field, value)`, `schema.validate_attrs(category, attrs)`, `schema.normalize(field, value)`

---

### Stage 1: Sieve / Detection (`tiger/sieve.py`)

**What it does:** Scans all product records and flags those whose text and image don't match well enough.

**How it works:**

1. **CLIP Encoding:** Every product's image and its "canonical text" (title + category + attributes assembled by `text_views.py`) are encoded into embedding vectors by `openai/clip-vit-base-patch32`.

2. **Global similarity score:** `sim_full = cosine(image_emb, caption_emb)` — a number from -1 to 1, where 1 means perfect agreement. For reference, correctly matching product pairs typically score 0.25–0.35.

3. **Locked threshold τ (fixes F3):** Thresholds are calibrated on the **clean calibration split only** and then locked. They are stored in `data/thresholds/tiger_locked_thresholds.json`. If `sim_full < τ`, the row is flagged. Using contaminated data to set thresholds would let errors hide — the old bug that inflated precision to 1.000.

4. **Per-category thresholds (fixes F4):** Each product category gets its own τ. Categories with too few products fall back to a global τ. This prevents low-similarity categories (e.g., patterned bags) from drowning high-similarity ones.

5. **Per-field contrastive probes (Phases 3.1/3.2):** For each attribute field (colour, material, pattern), the sieve tests: *"Does the image match the declared colour better than any other colour in the domain?"* This is what catches subtle text mutations (e.g., "blue" → "red") that the global CLIP score would miss entirely. Without probes, `mutate_text` recall is ~0.27; with probes it rises to ~0.85.

6. **Text-only checks (Phase 3.5):** Out-of-domain attribute values and title-attribute contradictions are caught without needing CLIP at all.

7. **Precision-floor fusion (Phase 3.4):** Each signal is calibrated to meet a 0.85 precision floor. Signals that can't meet the floor are quarantined. The fused detector achieves **precision 0.89, recall 0.88, F1 0.885**.

**Outputs:** A DataFrame with one row per product, with columns: `flagged`, `flag_reason`, `sim_full`, `sim_z`, `probe_color_margin`, `probe_color_z`, etc.

---

### Stage 2: Analyzer / Evidence (`tiger/analyzer.py`)

**What it does:** For each flagged row, gathers a rich set of evidence about *why* the row is flagged. It does NOT decide what to do — that's the Arbiter's job.

**Evidence gathered:**

| Evidence Signal | How computed | What it tells us |
|---|---|---|
| **Eq. 18 LOO (Leave-One-Out)** | Remove field j from caption, re-score with CLIP. If sim goes UP, field j was hurting — it's the suspect. | Which text field is most likely wrong |
| **Eq. 19 kNN** | Find k=8 nearest images (excluding own product). Check if their attributes agree with this row's declared attributes. | Whether the image is an outlier among visually similar products |
| **Swap z-score (F5)** | For this image, find the text in the whole catalogue that matches it best. If another row's text beats the row's own text by ≥2 std-devs, the image is probably swapped. | Image likely swapped from another product |
| **HSV pixel colour (F6/1.7)** | Physically read the image pixels, compute dominant colour in HSV space. Compare to declared colour attribute. | Ground-truth pixel colour vs what the text claims |
| **Probe signals** | Copied from Sieve output | Which fields fired a contrastive probe |
| **Missing modality (F11)** | Flags if image file is missing or text is empty | Routes constrained: missing image → can't do V2T |

**Output:** A list of `Evidence` dataclass instances, saved as JSONL to `data/outputs/evidence_{tag}.jsonl`.

---

### Stage 3: Arbiter / Router (`tiger/arbiter.py`)

**What it does:** Converts Evidence into a routing decision: what *type* of repair is needed and which direction?

**The two repair directions:**
- **V2T (Visual-to-Text):** The image is correct, the text is wrong. Fix the text.
- **T2V (Text-to-Visual):** The text is correct, the image is wrong. Replace the image.

**How it decides:**

A **calibrated multinomial logistic model** is trained on 14 evidence features (swap_z, loo_top_z, pixel_agrees_declared, probe signals, etc.) to estimate probabilities P(E1), P(E2), P(E3), P(CLEAN):
- **E1:** Text error (V2T route)
- **E2:** Image error (T2V route)
- **E3:** Mixed (both wrong — image-first, then re-diagnose)
- **CLEAN:** False positive — this record is actually fine

**Eq. 22 γ-gate:** If `max(P(E1..E4)) < γ=0.60`, the arbiter is too uncertain to act confidently. The row is escalated to **E4 (human review)** rather than risk a confident wrong-direction repair.

**CLEAN dismissal safety guard:** A row is only dismissed as a false positive if P(CLEAN) ≥ 0.80 AND no strong contrary signal (swap_z, pixel disagreement) is live.

**Missing modality bypass:** Rows with missing images or text skip the model entirely and follow strict F11 rules (missing image → T2V acquire; missing text → V2T).

**Model storage:** Transparent JSON coefficients in `data/thresholds/tiger_arbiter_model.json` (no pickle, fully auditable).

**Measured performance (holdout seed):** Direction accuracy **0.909** among acted-on rows.

---

### Stage 4: Solver / Repair Planner (`tiger/solver.py`)

**What it does:** Takes the Arbiter's routing decision and builds a concrete, executable repair plan.

**V2T planning:**
1. Takes the top suspect field from Eq. 18 LOO
2. Finds what the image actually shows (pixel colour estimate if confident, else CLIP probe prediction)
3. Validates the proposed value is in Ω_j and doesn't violate C
4. Returns a minimal `{field: new_value}` patch (single-field, cost-minimal)

**T2V planning (F14-safe):**
- Searches a `CandidatePool` of all images currently in use in the catalogue
- The row's **own product is excluded** from the pool — this means if the true original is sitting somewhere, it won't be trivially handed back as a "repair"
- Returns the catalogue image that best matches the row's caption

**Structure-safe patching (`apply_attr_patch`):**
- Validates against Ω_j before mutating anything
- For colour changes: safely rewrites the colour word in the title without touching brand names
- Rebuilds `canonical_text` from the updated structured attributes

---

### Stage 5: Verify / Acceptance Gates (`tiger/verify.py`)

**What it does:** Every proposed repair must pass three mathematical gates before being committed. Failed repairs are rolled back and escalated.

**The three gates:**

| Gate | Equation | What it checks |
|---|---|---|
| **Schema validity** | Eq. 27: A' ⊨ C | The patched record satisfies all Ω_j domains and constraint rules |
| **Threshold floor** | Eq. 28: c' ≥ τ̂ | The new CLIP similarity score meets the locked detection threshold |
| **Improvement margin** | Eq. 29: Δc ≥ ε | The improvement is larger than what a mere paraphrase would produce |

**ε (epsilon) — the noise floor:** ε is not an arbitrary constant. It is measured empirically: on clean calibration rows, we rephrase captions and measure how much CLIP similarity fluctuates. A repair must beat this natural variation. Currently ε = 0.0318 globally (per-category: 0.024–0.035).

**Independent Verifier hook (`independent_ok`):** A slot in `verify_repair()` for a second verification signal beyond CLIP. Currently supports:
- **SigLIP encoder** (available now, `--independent` flag): a second image-text encoder family cross-checking the repair
- **Gemini VLM judge** (available now, `--vlm-judge` flag): product-identity-aware VLM check

**Known structural limitation (F7):** CLIP-based gates confirm similarity improved — they cannot catch a wrong-direction repair that *also* improves CLIP similarity. Example: same-category image swap where the Arbiter re-labels the text to match the wrong (but on-category) image. All three gates pass because similarity genuinely rose. The Gemini VLM judge is the structural cure.

---

### The Closed Loop (`tiger/repair.py`)

`run_repair_cycle()` orchestrates the full pipeline for a batch of products:

1. Detect → Analyze → Route → Plan → Apply (on a working copy) → Verify
2. If accepted: commit to working set, re-flag on next pass
3. If rejected: roll back, escalate to human
4. Cap at 2 passes (configurable)

**End-to-end results (seed 7, 120 products):**
- 15 repaired
- 22 escalated to human (ambiguous γ-gate or unplannable)
- 1 dismissed (true false positive)
## 5. The Independent Verifier (`tiger/verify.py` & `tiger/vlm_judge.py`)

**Purpose:** Catch wrong-direction repairs that primary encoder-only verification misses (F7).

**How it works:**
- We implemented both an API-based VLM (Gemini 1.5 Flash) and a local encoder (SigLIP).
- Ultimately, **SigLIP** was selected as the final verifier due to its high accuracy, lack of rate limits, and massive speed advantage when sweeping 1,500 products on Kaggle.
- Gemini remains available via the `--vlm-judge` flag (requires `GEMINI_API_KEY` in `.env`). It uses an in-memory prompt cache to bypass the free tier's 500 RPD rate limits.

**Wired into CLI:**
```bash
.venv/bin/python -m tiger.cli repair --seed 7 --independent --generative-fallback
```

---

## 6. Configuration (`configs/tiger.yaml`)

| Section | Key parameters |
|---|---|
| `data` | paths for sample, processed, cache, outputs, schema |
| `models` | `clip_model_name`, `device`, `batch_size`, `independent_verifier` |
| `synthgen` | seed, products_per_category (60 × 4 = 240), image_size |
| `noise` | injection rates per error type (swap_image, color_flip, missing_image, etc.) |
| `sieve` | `threshold_method: quantile`, `quantile_q: 0.02`, probe fields and z-margins |
| `analyzer` | `knn_k: 8`, `swap_z_margin: 2.0` |
| `arbiter` | `gamma: 0.60`, `dismiss_threshold: 0.80`, `train_seeds` |
| `verify` | `epsilon_quantile: 0.95`, `max_passes: 2` |

---

## 7. How to Run the Full Pipeline

### Option A: Local Execution (Slow)

```bash
# 1. Set up environment (first time only)
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,vlm]"

# 2. Build the synthetic catalogue (first time only)
python -m tiger.cli synthgen

# 3. Calibrate (fit locked thresholds on clean data — first time only)
python -m tiger.cli calibrate
python -m tiger.cli train-arbiter
python -m tiger.cli calibrate-fusion

# 4. Run the pipeline on a report seed
python -m tiger.cli noise --seed 7        # inject errors
python -m tiger.cli detect --seed 7       # flag mismatches
python -m tiger.cli analyze --seed 7      # gather evidence
python -m tiger.cli route --seed 7        # route to repair direction
python -m tiger.cli repair --seed 7       # full closed-loop repair

# 5. With Gemini VLM judge active
python -m tiger.cli repair --seed 7 --vlm-judge

# 6. Evaluate and ablate
python -m tiger.cli sweep                 # 5-seed detection metrics
python -m tiger.cli ablate                # baselines + ablations table

# 7. Run unit tests (no GPU/network needed)
pytest tests/ -q
```

### Option B: Kaggle Execution (Recommended)

Due to heavy CLIP inference, running locally on a CPU is extremely slow. We have ported the execution to Kaggle's free GPUs.

1. Push your code to a public GitHub repository.
2. Upload `kaggle_workflow.ipynb` (found in the repository root) to Kaggle as a new Notebook.
3. In the right-hand panel of the Kaggle editor, go to **Add-ons -> Secrets** and add your `GEMINI_API_KEY`.
4. Under **Notebook options -> Accelerator**, select **GPU T4 x2** or **P100**.
5. Run the cells sequentially. The notebook clones the repo, pulls the key, runs the pipeline, and zips the results (`tiger_outputs.zip`) for easy download.

---

## 8. Results Summary

### Detection (pooled over 5 seeds, product-level bootstrap CIs)

| Operating Point | Precision | Recall | F1 |
|---|---|---|---|
| Full (OR-fused signals) | 0.793 | 0.924 | 0.853 |
| Full + fusion (precision floor 0.85) | **0.888** | 0.882 | **0.885** |

### Per-Error-Type Recall (full operating point)

| Error Type | Recall | Notes |
|---|---|---|
| swap_image | 0.975 | Strong — CLIP similarity drops clearly |
| swap_image_same_category | 0.950 | Subtle, but probes catch it |
| color_flip | 0.971 | Per-field colour probe |
| near_color_flip | 0.800 | Adjacent colour, harder |
| material_flip | **0.200** | Low — material invisible on synthetic silhouettes |
| attribute_drop | 1.000 | Schema required-field check |
| title_contradiction | 1.000 | Text-only check |
| missing_image | 1.000 | Trivial flag |
| mixed (E3) | 1.000 | |

### Repair (end-to-end, seed 7)

| Metric | Value |
|---|---|
| V2T colour restoration (vs ground truth) | 5/5 = 100% |
| Direction correctness | 14/15 = 93.3% |
| Rows escalated to human | 22 |
| 1 failure explained | F7: same-category swap — fixed by Gemini VLM judge |

---

## 9. The Error Taxonomy

| Code | Name | Meaning | Repair Direction |
|---|---|---|---|
| E1 | Text error | Text attributes don't match the (correct) image | V2T: fix the text |
| E2 | Image error | Image doesn't match the (correct) text | T2V: swap the image |
| E3 | Mixed | Both text and image are wrong | Image-first T2V, then re-diagnose |
| E4 | Ambiguous | Confidence < γ; risk of wrong-direction repair | Human review |
| CLEAN | False positive | Sieve flagged it, but it's actually fine | Dismiss |

---

## 10. Key Bugs Fixed from the Legacy MVP

| Fix ID | Bug | Resolution |
|---|---|---|
| F1 | CLIP 77-token truncation — long captions silently cut | Token-budget captions in `text_views.py` |
| F3 | Thresholds calibrated on contaminated data → inflated precision | Clean-calibration-only locked thresholds |
| F4 | Row-level duplication inflated sample size; product identity lost | Product IDs throughout; duplication off by default |
| F5 | Swap margin raw 0.01 — too tight, many false positives | Z-scored per-category swap margin |
| F6 | Naive 64×64 RGB colour averaging → 11 prototypes | HSV-based dominant colour estimator in `colors.py` |
| F7 | CLIP is both repair objective and acceptance signal → circular | Independent verifier hook; Gemini VLM judge |
| F10 | Text patching did string replace on raw title text — unsafe | Structure-safe `apply_attr_patch` with Ω_j validation |
| F11 | Missing image routed to V2T — impossible (no visual evidence) | `allowed_directions()` routing constraints |
| F12 | Noise injector could label a row as swapped while swap silently failed | Self-verifying injector with no-op hard failure |
| F14 | Repair eval re-used the original image as a candidate — trivial pass | CandidatePool excludes the row's own product |

---

## 11. Project Completion Status

### ✅ Immediate Tasks Completed
- Unit tests written.
- Wrong-direction repair caught by SigLIP/Gemini verifier.
- `ROADMAP_PROGRESS.md` updated with final metrics.

### ✅ Real-World Validation Completed (Kaggle)
- Pipeline evaluated end-to-end on the real-world **Kaggle Fashion Product Images** dataset.
- Caching implemented to bypass API rate limits on free-tier GPUs.

### ✅ Generative Fallback Completed
- **Stable Diffusion XL Turbo (SDXL-Turbo)** implemented via `tiger/generator.py`.
- Generative fallback successfully rescued 15 products with missing/corrupted modalities.

### ✅ Evaluation Completed
- Repair ablation study completed.
- Full CSV metrics exported via `cmd_sweep` and `cmd_ablate_repair`.

### 📝 Final Stage: Paper Writing
The engineering and experimentation phase is 100% complete. The final remaining task is drafting the academic manuscript utilizing the metrics stored in `data/outputs/` and the methodologies documented here.

**Key paper-ready numbers:**
- **Throughput:** ~6,000 products/hour on T4 GPU (vs. ~60-100/hr human curation = 80× speedup)
- **Cost:** $0.00 API cost (SigLIP is local; SDXL-Turbo runs on free Kaggle GPU)
- **Safety:** 269/435 ambiguous products escalated to human rather than auto-repaired
- **Accuracy gain:** Arbiter adds +49.4% restoration accuracy over random routing

**Pending: Cross-Domain (ABO) Experiment**
- `tiger/data/import_abo.py` and `import-abo` CLI command implemented
- Schema extended to 8 categories (4 fashion + 4 non-fashion)
- Run the lean ABO notebook on Kaggle (estimated 2-3 hours)
- Compare `repair_ablations_summary_run_abo.csv` vs Fashion to prove domain-agnosticism

Target venue: KDD Applied Data Science track or multimodal workshop at CVPR/ECCV

---

## 12. File Reference

| File | Lines | Role |
|---|---|---|
| `tiger/cli.py` | 505 | All CLI commands; entry point |
| `tiger/sieve.py` | ~290 | CLIP signals + threshold application |
| `tiger/analyzer.py` | 318 | Evidence (Eq. 18/19, swap_z, pixel) |
| `tiger/arbiter.py` | ~250 | P(E1..E4) router, γ-gate |
| `tiger/solver.py` | 209 | Repair planning, CandidatePool |
| `tiger/repair.py` | 203 | Closed loop orchestrator |
| `tiger/verify.py` | 184 | Eq. 27–29 gates, IndependentVerifier |
| `tiger/vlm_judge.py` | ~165 | Gemini VLM judge (NEW) |
| `tiger/schema.py` | 112 | Ω_j domains, constraint set C |
| `tiger/encoders.py` | ~120 | CLIP/SigLIP wrapper, content-hash cache |
| `tiger/colors.py` | ~100 | HSV dominant colour estimator |
| `tiger/fusion.py` | ~100 | Per-signal precision-floor calibration |
| `tiger/text_views.py` | ~200 | Caption building, token budgeting |
| `tiger/data/synthgen.py` | ~200 | Synthetic catalogue generator |
| `tiger/data/noise.py` | ~150 | Self-verifying error injector |
| `configs/tiger.yaml` | 79 | All pipeline hyperparameters |
| `configs/schema.yaml` | ~50 | Attribute domains and constraints |
| `ROADMAP_PROGRESS.md` | 332 | Checkpoint log, milestone results |
