# TIGeR — Text-Image Generative Repair

A multimodal catalogue data-cleaning pipeline that detects and repairs
image↔text mismatches in product records (wrong image, wrong colour, missing
modality, mixed errors). Built as a five-module detect → diagnose → route →
repair → accept loop:

```
Sieve → Analyzer → Arbiter → Solver → Verify
                     ▲                    │
                     └──── re-route ◄──────┘   (rejected repairs escalate)
```

This repository is the `tiger/` reference implementation. It grew out of a
critical review of an earlier script-based MVP; the fixes and the paper-alignment
work are tracked checkpoint-by-checkpoint in
[`ROADMAP_PROGRESS.md`](ROADMAP_PROGRESS.md).

## Architecture

```mermaid
graph TD
    %% Styling
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    classDef phase fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
    classDef gate fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef action fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef fallback fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    classDef human fill:#ffebee,stroke:#f44336,stroke-width:2px

    %% Inputs
    RawDB[("Raw Catalogue<br/>(Images & Text Attributes)")]:::input

    %% Phase 1: Sieve
    subgraph Phase1 ["Phase 1: The Sieve (Detection)"]
        Enc[Multimodal Encoder<br/>CLIP / SigLIP]
        QGate{"Confidence<br/>Quantile Gate<br/>(Thresholding)"}:::gate
    end

    RawDB --> Enc
    Enc --> |Cross-Modal Similarity| QGate
    QGate -->|High Confidence<br/>(Clean)| CleanData[("Clean Catalogue")]
    QGate -->|Low Confidence<br/>(Flagged Anomaly)| Phase2

    %% Phase 2: Analyzer
    subgraph Phase2 ["Phase 2: The Analyzer (Evidence Gathering)"]
        LOO[Leave-One-Out (LOO)<br/>Text Embeddings]
        KNN[K-NN Visual Search<br/>(Find Candidate Images)]
        Evid[Compile Evidence:<br/>Suspect Fields & Donors]
    end

    LOO --> Evid
    KNN --> Evid
    Phase2 --> Phase3

    %% Phase 3: Arbiter
    subgraph Phase3 ["Phase 3: The Arbiter (Routing)"]
        LogReg[Logistic Regression Router]
        GammaGate{"Gamma Gate<br/>(Confidence > 0.40?)"}:::gate
    end

    Evid --> LogReg
    LogReg --> GammaGate

    GammaGate -->|Low Confidence| Esc1[Escalate to Human]:::human
    GammaGate -->|High Confidence| RouteSplit{Route Decision}:::gate

    %% Phase 4: Solver
    subgraph Phase4 ["Phase 4: The Solver (Repair Execution)"]
        V2T[V2T Repair<br/>Patch Text Attributes]:::action
        T2V[T2V Repair<br/>Swap Catalogue Image]:::action
        GenFallback[Generative Fallback<br/>SDXL-Turbo Synthesis]:::fallback
        PoolCheck{"Candidate<br/>Available?"}:::gate
    end

    RouteSplit -->|V2T| V2T
    RouteSplit -->|T2V| PoolCheck
    PoolCheck -->|Yes| T2V
    PoolCheck -->|No| GenFallback

    %% Phase 5: Verification
    subgraph Phase5 ["Phase 5: Independent Verifier"]
        Judge{"VLM / SigLIP Judge<br/>(Final Validation)"}:::gate
    end

    V2T --> Judge
    T2V --> Judge
    GenFallback --> Judge

    Judge -->|YES (Approved)| Commit[("Repaired Catalogue")]:::action
    Judge -->|NO (Vetoed)| Esc2[Escalate to Human]:::human
```

> The legacy `scripts/*.py` MVP is kept for reference only. Everything current
> lives under `tiger/` and is driven by `python -m tiger.cli`.

## Modules

| Module | File | Responsibility |
| --- | --- | --- |
| Sieve | `tiger/sieve.py` | detection signals + contamination-robust locked thresholds; per-field contrastive probes |
| Analyzer | `tiger/analyzer.py` | evidence only — Eq. 18 leave-one-out attribution, Eq. 19 kNN, z-scored swap margin, pixel colour |
| Arbiter | `tiger/arbiter.py` | calibrated p(E1..E4), γ confidence gate, direction + tier routing, T→V policy gate |
| Solver | `tiger/solver.py` | structure-safe patch construction, cost-minimal V2T patch, F14-safe T2V candidate pool |
| Verify | `tiger/verify.py` | per-repair acceptance gates Eq. 27–29 with a measured ε; rollback hook |
| — | `tiger/repair.py` | the closed loop: apply → verify → accept/reject → re-route (2-pass cap) |
| — | `tiger/schema.py` | attribute domains Ω_j and constraint set C |
| — | `tiger/fusion.py` | per-signal precision-floor calibration + quarantine |
| — | `tiger/encoders.py` | CLIP wrapper with content-hash embedding cache |
| — | `tiger/data/` | seeded synthetic catalogue + self-verifying error injection |
| — | `tiger/eval/` | detection metrics, product-level bootstrap CIs, baselines/ablations |

## Install

```bash
python -m venv .venv && . .venv/bin/activate      # or: uv venv .venv
pip install -e ".[dev,vlm]"    # dev = torch/transformers; vlm = google-generativeai
# unit tests only, no heavy deps:
pip install -e ".[test]"
```

**Kaggle Workflow:** A fully configured `kaggle_workflow.ipynb` is included in the root directory to run this pipeline on Kaggle's free GPUs (T4x2/P100), bypassing local hardware limits.

CPU torch is sufficient. The first CLIP-using command downloads
`openai/clip-vit-base-patch32` (~600 MB).

## End-to-end pipeline

```bash
python -m tiger.cli synthgen           # build the seeded sample catalogue (data/sample/)
python -m tiger.cli calibrate          # lock thresholds + LOO + epsilon on the CLEAN calibration split
python -m tiger.cli train-arbiter      # fit p(E1..E4) on separately-seeded calibration-split noise
python -m tiger.cli calibrate-fusion   # tune per-signal margins to a precision floor

python -m tiger.cli noise --seed 7     # inject errors into the report split
python -m tiger.cli detect --seed 7    # sieve with locked thresholds
python -m tiger.cli analyze --seed 7   # Eq. 18/19 evidence for flagged rows
python -m tiger.cli route  --seed 7    # Arbiter routing plan
python -m tiger.cli repair --seed 7 --vlm-judge # full closed-loop repair + VLM independent cross-check

python -m tiger.cli sweep              # 5-seed detection metrics with product-level CIs
python -m tiger.cli ablate             # baselines + ablations table
```

Locked calibration artefacts live in `data/thresholds/`; run outputs in
`data/outputs/`. All calibration is done on the CALIBRATION product split and
locked; the REPORT split is never used to fit anything.

## Results (synthetic sample catalogue)

Detection, pooled over 5 report seeds, product-level bootstrap CIs:

| operating point | Precision | Recall | F1 |
| --- | --- | --- | --- |
| full (OR-fused signals) | 0.793 | 0.924 | 0.853 |
| full + fusion (precision floor) | 0.888 | 0.882 | 0.885 |

Per-error-type recall (full): swap_image 0.975 · mutate_text 0.853 ·
mixed 1.000 · missing_image 1.000. The ablation table (`tiger.cli ablate`)
shows a CLIP-similarity-only detector catches just **0.27** of text mutations —
the per-field contrastive probes are what lift that to 0.85.

**Repair Evaluation (Real-World Data)**
The repair pipeline was fully validated on 1,500 products from the **Kaggle Fashion Product Images** dataset.
- The Arbiter's routing logic provided a **+49.4%** improvement in restoration accuracy over random guessing.
- Generative Fallback (SDXL-Turbo) successfully synthesized missing images for 15 dead-end products.
- The Independent Verifier blocked semantic "wrong direction" repairs, increasing accuracy by +13.3%.

## Honest caveats

- **CLIP-based verification is not correctness (F7).** Verify's Eq. 27–29 gates
  confirm similarity improved; they cannot catch a *wrong-direction* repair that
  also improves CLIP similarity (reproduced live: a same-category image swap
  text-patched to match the wrong image, all gates passed). **Cure implemented:**
  The `IndependentVerifier` (Phase 6.4) hooks into the pipeline via `--independent` (SigLIP) or `--vlm-judge` (Gemini) to check semantic alignment and successfully vetoes this class of error.
- **material_flip recall is low** on synthetic silhouettes because material is
  not visible in a flat coloured shape; expected to improve on real photos.
- **Precision is not 1.000.** The earlier MVP's 1.000 was an artefact of
  thresholds calibrated on contaminated data (F3); the honest number is ~0.79
  (0.89 with fusion).

## Tests

```bash
pytest            # 68 unit tests; no network/model download required
```

Regression tests pin the critical review's fixes (F1/F3/F6/F10/F12, the
Eq. 27–29 gates, the routing constraints, the fusion quarantine).

## Mapping to the paper

Every equation the manuscript claims now exists in code and is exercised:
Eq. 18 (`analyzer.calibrate_loo` / LOO in `analyzer.analyze`), Eq. 19 (kNN in
`analyzer.analyze`), Eq. 22 (`arbiter.route` γ gate + policy), Eq. 27–29
(`verify.verify_repair`), constraint set C (`schema.py`). See
`ROADMAP_PROGRESS.md` for the checkpoint-by-checkpoint trace.
