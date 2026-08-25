# TIGeR Pipeline Architecture

The following diagram illustrates the complete, end-to-end workflow of the **Text-Image Generative Repair (TIGeR)** pipeline. It details the journey of a product record from raw input, through anomaly detection (Sieve), evidence gathering (Analyzer), routing (Arbiter), repair execution (Solver with Generative Fallback), and final verification (VLM/SigLIP Judge).

You can render this diagram directly in markdown editors that support Mermaid (like GitHub, Notion, or Obsidian), or paste it into a live editor like [Mermaid Live](https://mermaid.live/) to export it as an SVG/PNG for your research paper.

```mermaid
flowchart TD
    %% Styling
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5,color:#000
    classDef phase fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px,color:#000
    classDef gate fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000
    classDef action fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#000
    classDef fallback fill:#fce4ec,stroke:#e91e63,stroke-width:2px,color:#000
    classDef human fill:#ffebee,stroke:#f44336,stroke-width:2px,color:#000

    %% Main Input
    RawDB[("Raw Catalogue (Images & Text)")]:::input

    %% Phase 1
    subgraph Phase1 ["Phase 1: The Sieve (Detection)"]
        Enc["Multimodal Encoder (CLIP / SigLIP)"]
        QGate{"Confidence Quantile Gate"}:::gate
    end

    %% Phase 2
    subgraph Phase2 ["Phase 2: The Analyzer (Evidence Gathering)"]
        LOO["Leave-One-Out (LOO) Text Embeddings"]
        KNN["K-NN Visual Search (Find Candidate Images)"]
        Evid["Compile Evidence: Suspect Fields & Donors"]
    end

    %% Phase 3
    subgraph Phase3 ["Phase 3: The Arbiter (Routing)"]
        LogReg["Logistic Regression Router"]
        GammaGate{"Gamma Gate (Confidence > 0.40?)"}:::gate
        RouteSplit{"Route Decision"}:::gate
    end

    %% Phase 4
    subgraph Phase4 ["Phase 4: The Solver (Repair Execution)"]
        V2T["V2T Repair (Patch Text)"]:::action
        PoolCheck{"Candidate Available?"}:::gate
        T2V["T2V Repair (Swap Catalogue Image)"]:::action
        GenFallback["Generative Fallback (SDXL-Turbo)"]:::fallback
    end

    %% Phase 5
    subgraph Phase5 ["Phase 5: Independent Verifier"]
        Judge{"VLM / SigLIP Judge (Final Validation)"}:::gate
    end

    %% External Outcomes
    CleanData[("Clean Catalogue")]:::input
    Esc1["Escalate to Human"]:::human
    Esc2["Escalate to Human"]:::human
    Commit[("Repaired Catalogue")]:::action

    %% Graph Connections
    RawDB --> Enc
    Enc -- "Cross-Modal Similarity" --> QGate
    
    QGate -- "High Confidence (Clean)" --> CleanData
    QGate -- "Low Confidence (Flagged Anomaly)" --> LOO
    QGate -- "Low Confidence (Flagged Anomaly)" --> KNN

    LOO --> Evid
    KNN --> Evid
    
    Evid --> LogReg
    LogReg --> GammaGate

    GammaGate -- "Low Confidence" --> Esc1
    GammaGate -- "High Confidence" --> RouteSplit

    RouteSplit -- "V2T Route" --> V2T
    RouteSplit -- "T2V Route" --> PoolCheck
    
    PoolCheck -- "Yes" --> T2V
    PoolCheck -- "No" --> GenFallback

    V2T --> Judge
    T2V --> Judge
    GenFallback --> Judge

    Judge -- "YES (Approved)" --> Commit
    Judge -- "NO (Vetoed)" --> Esc2
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
