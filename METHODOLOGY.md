# Technical Schematic: Resolution Artifact Correction

This document details the methodology for the **VLSV-JAX Neural Corrector**, specifically focusing on how the framework bridges the gap between heterogeneous coarse resolutions and high-fidelity fine-resolution targets.

## 1. The Core Philosophy
Directly correcting a distribution function $f$ is difficult because of its massive dynamic range ($10^{-5}$ to $0.5$). Instead, our model learns a **Log-Residual Mapping**:
$$f_{fine}(x, v) = f_{coarse}(x, v) \cdot \exp(\text{MLP}(x, v))$$
Or in log-space:
$$\log f_{fine} = \log f_{coarse} + \Delta \log f$$

## 2. The Resolution Adapter (Multi-Grid Logic)
To ensure the MLP can handle any velocity grid (16^3, 32^3, etc.), we introduce a **Canonical Resolution Adapter**.

### The Mapping Workflow:
```mermaid
graph TD
    A["Coarse Data (Arbitrary NV^3)"] --> B{"Resolution Adapter"}
    B -- "NV < 32" --> C["Tri-linear Upsampling"]
    B -- "NV = 32" --> D["Identity"]
    B -- "NV > 32" --> E["Box-Downsampling"]
    C --> F["Canonical Grid (32^3)"]
    D --> F
    E --> F
    F --> G["ResMLP Corrector"]
    G --> H["Log-Residual (32^3)"]
    H --> I{"Inverse Adapter"}
    I --> J["Final Corrected f (Original NV^3)"]
```

## 3. Architecture & Training Flow (Detailed Schema)

The diagram below visualizes how the **Fine Resolution Simulation** acts as the high-fidelity oracle to define the target for the MLP.

```mermaid
graph TD
    classDef physical fill:#ffffff,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    classDef ml fill:#ffffff,stroke:#01579b,stroke-width:2px;
    classDef target fill:#ffffff,stroke:#fbc02d,stroke-width:2px;

    %% Data Generation Branch
    subgraph "Phase 1: Resolution Artifact Correction"
        Fine["High-Res Simulation (64^3)"]:::physical
        Coarse["Coarse Simulation (32^3)"]:::physical
    end

    %% The Ground Truth Pipeline
    Fine -->|log transform| LogFine["log(f_fine)"]
    Coarse -->|log transform| LogCoarse["log(f_coarse)"]
    
    LogFine & LogCoarse --> TargetCalc(["Target Calculation: Delta = log(f_fine) - log(f_coarse)"]):::target

    %% The ML Inference Pipeline
    Coarse -->|Features| Inputs["Input Samples (32780x) <br/> [f_coarse, E, B, grad]"]:::ml
    Inputs --> MLP["Pure MLP (3-Layer Baseline)"]:::ml
    MLP --> |Predict| PredDelta["Predicted Delta-Log-f"]:::ml

    %% The Optimization
    TargetCalc --- Loss{Physics-Aware Loss}
    PredDelta --- Loss
    
    Loss -->|Backprop| MLP
```

### Key Differences in this Architecture:
1.  **Direct vs Residual**: The MLP doesn't predict the distribution; it predicts the **Residual** between the coarse and fine regimes.
2.  **The Fine-Res Anchor**: The "Ground Truth" for every training sample comes from a high-fidelity $64^3$ simulation that has been box-downsampled to the coarse grid.
3.  **Logarithmic Stability**: By calculating `Delta = log(f_fine) - log(f_coarse)`, we allow the model to learn corrections for both the dense thermal core (high values) and the kinetic beams (very low values) simultaneously.

## 4. Solver-in-the-loop: The Hybrid Integration

This schematic shows how the Neural Corrector is injected into the **Hybrid Vlasov-Maxwell** solver loop.

```mermaid
sequenceDiagram
    participant S as Vlasov Solver (Physics)
    participant M as Maxwell Solver (Fields)
    participant N as ResMLP Corrector (AI)
    
    Note over S,M: Time Step (t -> t + dt)
    S->>S: Advection / Acceleration (Standard)
    M->>M: Field Update (E, B)
    
    rect rgb(230, 240, 255)
    Note right of N: The Neural Hook
    S-->>N: f_coarse(x, v)
    M-->>N: Local Fields (E, B, grad)
    N->>N: Infer Delta-Log-f
    N-->>S: Neural Correction Layer
    S->>S: f_new = f * exp(Delta_log_f)
    end
    
    Note over S,M: Conserved Moments & Diagnostics
```
