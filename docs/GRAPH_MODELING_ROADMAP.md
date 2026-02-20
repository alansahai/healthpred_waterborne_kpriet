# Graph-Based Spatiotemporal Modeling Roadmap

> **Document Status:** Architectural Roadmap (Future Planning)  
> **Last Updated:** February 2026  
> **Author:** System Architecture Team

---

## Executive Summary

This document outlines the architectural roadmap for evolving the Disease Outbreak Early Warning System from **spatially-aware tabular ML** to **graph-based spatiotemporal learning**.

### Key Principle

> **No current production functionality will be modified.**  
> This roadmap is additive and modular.

---

## Current Architecture (Production)

### Modeling Assumption

The current XGBoost model predicts outbreak probability using ward-level tabular features:

$$P(\text{outbreak}_{i,t}) = f(X_{i,t-1})$$

Where:
- $X_{i,t-1}$ = ward-level features (environmental, temporal, lag features)
- $f$ = XGBoost classifier with isotonic calibration

### Spatial Enhancements (Already Implemented)

| Feature | Description | Status |
|---------|-------------|--------|
| Neighbor Lag Features | Average cases/ecoli/rainfall from adjacent wards | ✅ Production |
| Peripheral Ward Flag | Binary indicator for boundary wards | ✅ Production |
| Distance from Center | Euclidean distance to city centroid | ✅ Production |
| Ring Road Exposure | Proximity to major transport corridors | ✅ Production |
| Synthetic Spread Simulation | Visual diffusion simulation (dashboard only) | ✅ Production |

These features represent **first-order spatial awareness** within the tabular framework.

---

## Future Architecture Options

### Option A: Spatial Lag Regression (Short-Term)

**Timeline:** 3-6 months  
**Complexity:** Moderate  
**Dependencies:** PySAL, statsmodels

#### Mathematical Formulation

Spatial Autoregressive (SAR) Model:

$$y = \rho W y + X \beta + \epsilon$$

Where:
- $W$ = row-normalized adjacency matrix
- $\rho$ = spatial dependence coefficient
- $X$ = feature matrix
- $\beta$ = regression coefficients
- $\epsilon$ = error term

#### Implementation Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    SPATIAL LAG REGRESSION                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Build Adjacency Matrix W from ward_adjacency.json       │
│                           ↓                                 │
│  2. Row-normalize: W_norm = D^(-1) W                        │
│                           ↓                                 │
│  3. Compute Spatial Lag: Wy = W_norm @ y                    │
│                           ↓                                 │
│  4. Fit SAR Model: y ~ ρ(Wy) + Xβ                           │
│                           ↓                                 │
│  5. Estimate ρ via Maximum Likelihood                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Potential File Structure

```
model/
├── spatial_sar.py          # Spatial autoregressive model (NEW)
├── train.py                # Unchanged
├── predict.py              # Unchanged
utils/
├── spatial_matrix.py       # Adjacency matrix utilities (NEW)
```

#### Integration Strategy

- SAR model runs as **parallel experiment**, not replacement
- Compare SAR vs XGBoost performance on held-out data
- Production deployment only if statistically superior

---

### Option B: Graph Neural Network (Long-Term)

**Timeline:** 12-18 months  
**Complexity:** High  
**Dependencies:** PyTorch Geometric / DGL / Spektral

#### Mathematical Formulation

Graph Convolutional Network (GCN):

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} \Theta^{(l)})$$

Where:
- $\tilde{A} = A + I$ (adjacency with self-loops)
- $\tilde{D}$ = degree matrix of $\tilde{A}$
- $H^{(l)}$ = node features at layer $l$
- $\Theta^{(l)}$ = learnable weight matrix
- $\sigma$ = non-linear activation

#### Graph Structure

```
Ward Graph G = (V, E, X)
━━━━━━━━━━━━━━━━━━━━━━━━

V (Nodes):
  - 100 wards (W01 to W100)
  - Each node has feature vector X_i

E (Edges):
  - Adjacency connections from ward_adjacency.json
  - Optional: weighted by distance or shared boundary length

X (Node Features):
  - Temporal: week, month, monsoon_flag
  - Environmental: rainfall, turbidity, ecoli_index
  - Lag: cases_last_week, cases_2w_avg
  - Epidemiological: outbreak_count_2w
```

#### Architecture Options

| Model | Description | Use Case |
|-------|-------------|----------|
| GCN | Standard graph convolution | Baseline graph model |
| GraphSAGE | Sampling + aggregation | Scalable training |
| GAT | Attention-weighted neighbors | Learn neighbor importance |
| STGCN | Spatiotemporal GCN | Time-series + graph |

#### Potential File Structure

```
model/
├── graph/
│   ├── __init__.py
│   ├── graph_builder.py    # Ward graph construction
│   ├── gcn_model.py        # GCN architecture
│   ├── gat_model.py        # Graph Attention Network
│   ├── train_graph.py      # Graph model training
│   └── predict_graph.py    # Graph model inference
```

#### Integration Strategy

- Graph model runs as **experimental pipeline**
- A/B test against production XGBoost
- Gradual rollout with feature flags

---

## Implementation Phases

### Phase 1: Current State (Complete ✅)

```
                    ┌─────────────────────┐
                    │   Raw CSV Data      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Feature Engineering │
                    │  (Spatial-Aware)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  XGBoost Classifier │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Operational Dashboard│
                    └─────────────────────┘
```

**Capabilities:**
- Tabular ML with 32 features (25 base + 7 spatial)
- Neighbor lag features capture first-order spatial dependence
- Synthetic spread simulation for visualization

---

### Phase 2: Spatial Lag Regression (Future)

```
                    ┌─────────────────────┐
                    │   Raw CSV Data      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────────┐     │     ┌──────────▼──────────┐
    │ Adjacency Matrix  │     │     │ Feature Engineering │
    │   Builder (W)     │     │     │   (Spatial-Aware)   │
    └─────────┬─────────┘     │     └──────────┬──────────┘
              │               │                │
              └───────────────┼────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Model Selection   │
                   │  ┌────────┬───────┐ │
                   │  │XGBoost │  SAR  │ │
                   │  └────────┴───────┘ │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │ Ensemble / Compare  │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │ Operational Dashboard│
                   └─────────────────────┘
```

**New Capabilities:**
- Explicit spatial autocorrelation modeling
- Estimate spatial dependence coefficient ρ
- Compare SAR vs XGBoost performance

---

### Phase 3: Graph Neural Network (Future)

```
                    ┌─────────────────────┐
                    │   Raw CSV Data      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────────┐     │     ┌──────────▼──────────┐
    │  Graph Builder    │     │     │ Feature Engineering │
    │   (Nodes + Edges) │     │     │   (Node Features)   │
    └─────────┬─────────┘     │     └──────────┬──────────┘
              │               │                │
              └───────────────┼────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Graph Encoder      │
                   │  (GCN / GAT / SAGE) │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Node Classifier   │
                   │   (Per-Ward Risk)   │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │ Operational Dashboard│
                   └─────────────────────┘
```

**New Capabilities:**
- Multi-hop propagation (2-3 layer GNN)
- Learned neighbor importance (GAT)
- End-to-end graph representation learning

---

## Backward Compatibility Guarantees

| Guarantee | Description |
|-----------|-------------|
| ✅ XGBoost Default | Production model remains XGBoost |
| ✅ Artifact Validity | Existing .pkl artifacts remain valid |
| ✅ Inference Unchanged | predict.py signature preserved |
| ✅ UI Independence | Dashboard works without graph models |
| ✅ No New Dependencies | No forced installation of PyTorch/DGL |
| ✅ No Forced Retraining | Current model deployable indefinitely |

---

## Dependency Matrix (Future Only)

| Phase | Core Dependencies | Additional Libraries |
|-------|-------------------|----------------------|
| Current | pandas, xgboost, scikit-learn | geopandas, shapely |
| Phase 2 (SAR) | + pysal, libpysal | esda, spreg |
| Phase 3 (GNN) | + torch, torch-geometric | networkx, dgl (optional) |

> **Note:** Phase 2 and 3 dependencies are NOT required for current production.

---

## Performance Expectations

### Theoretical Benefits of Graph-Based Modeling

| Aspect | Tabular ML | Spatial Lag | Graph Neural Network |
|--------|------------|-------------|----------------------|
| Local Features | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| 1-Hop Neighbors | ✅ Manual Features | ✅ Automatic (ρ) | ✅ Automatic |
| 2+ Hop Neighbors | ❌ Not Captured | ⚠️ Higher-Order W | ✅ Multi-Layer |
| Diffusion Dynamics | ❌ Simulated Only | ⚠️ Static | ✅ Learned |
| Interpretability | ✅ SHAP/Feature Importance | ✅ ρ Coefficient | ⚠️ Attention Weights |

### When to Upgrade

Consider graph-based modeling when:
- Spatial autocorrelation is statistically significant
- Outbreaks frequently propagate across ward boundaries
- Multi-hop spread patterns are observed
- Sufficient labeled data for GNN training (> 1000 samples)

---

## Risk Assessment

### Low Risk (Recommended)

- **Option A (SAR)**: Well-established methodology, interpretable coefficients
- **Hybrid Approach**: Use SAR features as input to XGBoost

### Medium Risk

- **Option B (GCN)**: Requires careful hyperparameter tuning
- **Data Requirements**: GNNs benefit from larger datasets

### Mitigation Strategies

1. **A/B Testing**: Always compare against XGBoost baseline
2. **Feature Ablation**: Validate spatial features add predictive value
3. **Staged Rollout**: Deploy to subset of wards first
4. **Fallback**: Automatic revert to XGBoost if graph model degrades

---

## Summary

### Current State

The system has **spatially-aware tabular ML** with:
- Neighbor lag features (4 features)
- Static spatial features (3 features)
- Synthetic spread simulation (visualization)

### Future State

The architecture is **ready to evolve** toward:
- Spatial autoregressive modeling (SAR)
- Graph neural networks (GCN, GAT, GraphSAGE)
- End-to-end spatiotemporal learning

### Key Message

> The system is modular and future-ready.  
> Graph-based modeling can be added **without replacing** current infrastructure.

---

## File Reference

### Current Implementation (Production)

| File | Purpose |
|------|---------|
| `utils/spatial.py` | Core spatial utilities (zone loading, adjacency) |
| `utils/spatial_viz.py` | Visualization helpers + simulation |
| `utils/feature_engineering.py` | Spatial feature generation |
| `data/ward_adjacency.json` | Ward-level adjacency map |
| `data/peripheral_wards.json` | Peripheral ward list |
| `spatial_metadata.json` | Comprehensive spatial metadata |

### Future Implementation (Not Yet Created)

| File | Purpose | Phase |
|------|---------|-------|
| `utils/spatial_matrix.py` | Adjacency matrix utilities | Phase 2 |
| `model/spatial_sar.py` | Spatial autoregressive model | Phase 2 |
| `model/graph/graph_builder.py` | Ward graph construction | Phase 3 |
| `model/graph/gcn_model.py` | Graph convolutional network | Phase 3 |

---

*This document serves as architectural guidance. Implementation decisions should be validated against production requirements and available data.*
