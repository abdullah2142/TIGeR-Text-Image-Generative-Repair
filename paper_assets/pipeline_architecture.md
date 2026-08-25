# TIGeR Pipeline Architecture

The following diagram illustrates the complete, end-to-end workflow of the **Text-Image Generative Repair (TIGeR)** pipeline. It details the journey of a product record from raw input, through anomaly detection (Sieve), evidence gathering (Analyzer), routing (Arbiter), repair execution (Solver with Generative Fallback), and final verification (VLM/SigLIP Judge).

You can render this diagram directly in markdown editors that support Mermaid (like GitHub, Notion, or Obsidian), or paste it into a live editor like [Mermaid Live](https://mermaid.live/) to export it as an SVG/PNG for your research paper.

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
        Enc["Multimodal Encoder<br/>CLIP / SigLIP"]
        QGate{"Confidence<br/>Quantile Gate<br/>(Thresholding)"}:::gate
    end

    RawDB --> Enc
    Enc -- "Cross-Modal Similarity" --> QGate
    QGate -- "High Confidence<br/>(Clean)" --> CleanData[("Clean Catalogue")]
    QGate -- "Low Confidence<br/>(Flagged Anomaly)" --> Phase2

    %% Phase 2: Analyzer
    subgraph Phase2 ["Phase 2: The Analyzer (Evidence Gathering)"]
        LOO["Leave-One-Out (LOO)<br/>Text Embeddings"]
        KNN["K-NN Visual Search<br/>(Find Candidate Images)"]
        Evid["Compile Evidence:<br/>Suspect Fields & Donors"]
    end

    LOO --> Evid
    KNN --> Evid
    Phase2 --> Phase3

    %% Phase 3: Arbiter
    subgraph Phase3 ["Phase 3: The Arbiter (Routing)"]
        LogReg["Logistic Regression Router"]
        GammaGate{"Gamma Gate<br/>(Confidence > 0.40?)"}:::gate
    end

    Evid --> LogReg
    LogReg --> GammaGate

    GammaGate -- "Low Confidence" --> Esc1["Escalate to Human"]:::human
    GammaGate -- "High Confidence" --> RouteSplit{"Route Decision"}:::gate

    %% Phase 4: Solver
    subgraph Phase4 ["Phase 4: The Solver (Repair Execution)"]
        V2T["V2T Repair<br/>Patch Text Attributes"]:::action
        T2V["T2V Repair<br/>Swap Catalogue Image"]:::action
        GenFallback["Generative Fallback<br/>SDXL-Turbo Synthesis"]:::fallback
        PoolCheck{"Candidate<br/>Available?"}:::gate
    end

    RouteSplit -- "V2T" --> V2T
    RouteSplit -- "T2V" --> PoolCheck
    PoolCheck -- "Yes" --> T2V
    PoolCheck -- "No" --> GenFallback

    %% Phase 5: Verification
    subgraph Phase5 ["Phase 5: Independent Verifier"]
        Judge{"VLM / SigLIP Judge<br/>(Final Validation)"}:::gate
    end

    V2T --> Judge
    T2V --> Judge
    GenFallback --> Judge

    Judge -- "YES (Approved)" --> Commit[("Repaired Catalogue")]:::action
    Judge -- "NO (Vetoed)" --> Esc2["Escalate to Human"]:::human
```

### Component Details for the Paper:

1. **The Sieve:** Acts as the high-recall anomaly detector using contrastive embeddings (CLIP/SigLIP) to filter out clearly correct (clean) product rows.
2. **The Analyzer:** Generates actionable evidence by masking out individual text attributes (Leave-One-Out) and querying the visual catalogue (k-NN) for alternative donor images.
3. **The Arbiter:** A logistic regression model trained on simulated noise. It uses a **Gamma Gate** (adjustable confidence threshold, $\gamma = 0.40$) to decide whether it is safe to automate a repair, or if human escalation is required.
4. **The Solver:** Executes the structure-safe repair.
    - **V2T (Visual-to-Text):** Corrects erroneous text attributes based on visual evidence.
    - **T2V (Text-to-Visual):** Swaps a corrupted image with a valid donor.
    - **Generative Fallback:** If the catalogue lacks a donor, **SDXL-Turbo** synthesizes a new, attribute-weighted product image in 4 inference steps.
5. **Independent Verifier:** A secondary guardrail (Gemini 3.7 Flash or Local SigLIP) that checks the final proposed pair. It prevents "same-category wrong-direction" swaps from entering the database.
