# Visual Assets

Architecture diagrams and visual materials for EdgeBit-350M.

## Diagrams (ASCII/Text)

The following diagrams are embedded in the documentation:

- **Architecture overview**: `docs/architecture.md` — Full model block diagram
- **Memory layout**: `docs/runtime.md` — Runtime memory budget visualization
- **Training curriculum**: `docs/training_pipeline.md` — Phase transition diagram
- **Decoder block**: `docs/architecture.md` — Attention + FFN block detail

## Generating Visual Diagrams

To create publication-quality diagrams, use these descriptions with any diagramming tool (draw.io, Excalidraw, Figma):

### 1. Architecture Diagram
```
Top to bottom:
  Input tokens → Embedding (NF4) → [Decoder Block x 24] → RMSNorm → LM Head → Output logits
  
Decoder Block detail:
  Input → RMSNorm → GQA Attention (BitLinear Q,K,V,O + QK Norm + SubLN) → Residual
       → RMSNorm → SwiGLU FFN (BitLinear gate,up,down + SubLN) → Residual → Output
```

### 2. Quantization Flow
```
Left to right:
  Float32 weights → Group (128) → Absmean scale → Round → Clamp [-1,+1] → Ternary {-1,0,+1}
  Store: 2-bit packed (4 per byte) + FP16 scale per group
```

### 3. Training Pipeline
```
Timeline:
  [BF16 10%] → [INT8 20%] → [INT4 25%] → [Ternary 45%]
  Loss curve: starts high, drops, spikes at each transition, recovers
```

### 4. Memory Comparison
```
Bar chart:
  FP32: 1334 MB
  FP16:  913 MB
  Packed: 156 MB (5.9x compression)
  
  Breakdown of 156 MB:
    Embedding (NF4): 82 MB
    BitLinear (2-bit): 35 MB
    KV Cache (INT8): 24 MB
    Other: 15 MB
```

### 5. Deployment Targets
```
Device compatibility matrix:
  Raspberry Pi 5 (4GB) → 156 MB model → fits with 3.8 GB headroom
  Laptop (8GB) → 156 MB model → fits with 7.8 GB headroom
  Cloud CPU (2GB) → 156 MB model → fits with 1.8 GB headroom
```

## Color Scheme

For consistent branding:
- Primary: #2563EB (blue) — model components
- Secondary: #059669 (green) — quantization/compression
- Accent: #DC2626 (red) — warnings/critical paths
- Neutral: #6B7280 (gray) — backgrounds/borders
