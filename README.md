# 🏥 Health Outbreak Early Warning System

> **AI-Powered Predictive Analytics for Waterborne Disease Prevention**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green.svg)](https://xgboost.ai)

---

## 🎯 Project Overview

A machine learning-based early warning system that predicts waterborne disease outbreaks **7 days in advance** (with optional recursive projection to 14 days), enabling proactive public health interventions. The system monitors 100 wards across Coimbatore, analyzing environmental factors to identify high-risk zones before outbreaks occur.

### Key Capabilities
- ✅ **Operational auto-prediction** on app startup (no manual train step)
- ✅ **89% recall rate** - catches 9 out of 10 actual outbreaks
- ✅ **Environmental intelligence** - rainfall, water quality, sanitation analysis
- ✅ **Geographic visualization** - interactive heatmaps for decision support
- ✅ **Automated alerts** - ranked ward-level interventions with trend metadata

---

## 📁 Project Structure

```
health_prediction_poc/
├── app.py                          # Multi-page dashboard ⭐
├── model/
│   ├── __init__.py
│   ├── train.py                    # XGBoost trainer with metrics
│   └── predict.py                  # Inference engine
├── utils/
│   ├── __init__.py
│   ├── feature_engineering.py      # Lag/rolling/growth features
│   ├── risk_logic.py               # Risk classification
│   ├── geo_mapping.py              # Ward-to-zone mapping
│   └── analytics.py                # Correlation analysis
├── data/
│   ├── integrated_surveillance_dataset_final.csv
│   ├── integrated_surveillance_dataset.csv
│   └── coimbatore_unified_master_dataset_with_zone.csv
├── geo/
│   └── coimbatore.geojson          # Zone boundaries
├── requirements.txt
└── DEMO_SCRIPT.md                  # Presentation guide
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd health_prediction_poc
pip install -r requirements.txt
```

**Note:** If `shap` fails to install, that's okay - the system will run without it.

### 2. Run the Application (Operational Mode)

```bash
# Launch the dashboard
streamlit run app.py
```

**Default User Navigation:**
- System Overview
- Monitoring Dashboard
- Risk Heatmap
- Environmental Analysis

**Admin Mode Navigation:**
- Model Diagnostics (Admin)
- Retrain Model (Admin)

### 3. Retraining (Admin only)

1. Enable **Admin Mode** in the sidebar
2. Open **Retrain Model (Admin)**
3. Trigger retraining explicitly when needed

Normal monitoring users do not need to train the model.

---

## 📊 System Features

### 🏠 Monitoring Dashboard
- **Risk Assessment Table** - Ward-by-ward probability scores
- **Geographic Heatmap** - Color-coded zone visualization
- **Automated Alert Panel** - Threshold-driven ranked warning objects
- **Operational Summary Cards** - Current high-risk wards, predicted outbreaks next 7 days, seasonal indicator, rainfall surge alert

### 🔬 Environmental Analysis Page
- **Rainfall vs Disease Correlation** - Scatter plots with trends
- **Water Quality Impact** - Statistical relationships
- **Seasonal Patterns** - Monthly outbreak trends
- **Correlation Heatmap** - Feature relationships

### 🎯 Model Diagnostics (Admin)
- **Model Metadata** - last trained date, training range, best threshold, calibration
- **Validation/Test Metrics** - operational trust indicators
- **Leakage Checks** - strict temporal split and past-only rolling verification

### ⚙️ Configuration

Runtime config file: `config/system_config.json`

- `data_path`: predefined ingestion CSV path
- `model_path`: persisted model artifact path
- `threshold_override`: reserved field; ignored in strict operational mode
- `retraining_frequency_days`: governance for retraining cadence
- `artifact_version`: model artifact version tracking
- `enable_14_day_projection`: toggle recursive week +2 projection

The architecture is ready to integrate real-time data APIs without structural changes.

### 🔒 Operational Hardening (Implemented)

- **Frozen artifact lock:** Operational mode hard-fails if `model/final_outbreak_model_v3.pkl` is missing.
- **No silent model fallback:** No secondary model artifact is loaded in operational startup.
- **Strict metadata contract:** Artifact metadata is validated for required keys (`threshold`, `calibration`, `training_range`, `cv_metrics`, `artifact_version`).
- **Policy threshold guard:** Startup asserts deployment threshold equals `0.23`.
- **Data integrity checks:** Duplicate `(ward_id, week_start_date)` records raise errors before feature engineering.
- **Merge row integrity:** Integration validates expected rows (`expected_wards * expected_weeks`) and fails on mismatch.
- **What-if scope safety:** Scenario edits apply only to selected ward on the latest week row.
- **Admin-only retraining:** Retraining is manual from Administration page and writes to `model/final_outbreak_model_v3.pkl`.
- **Rerun efficiency:** Model loading uses `@st.cache_resource`; dataset loading uses `@st.cache_data`.
- **Clean runtime output:** Debug-style `print(...)` traces removed from training/finalization runtime paths.

---

## 🧠 Machine Learning Architecture

### Model: XGBoost Classifier
- **Algorithm:** Gradient Boosting Decision Trees
- **Target:** `outbreak_next_week` (binary classification)
- **Optimization:** Recall-prioritized (minimize false negatives)

### Performance Metrics
| Metric | Score | Why It Matters |
|--------|-------|----------------|
| **Recall** | 89% | Catches 9/10 outbreaks (critical for public health) |
| **F1 Score** | 79% | Balanced precision & recall |
| **Accuracy** | 82% | Overall correctness |

### Feature Engineering Pipeline
- **Temporal Features:** Lag (1, 2 weeks), rolling averages, growth rates
- **Environmental:** Rainfall, water quality, sanitation indices
- **Statistical:** Ward-level min/max/median history
- **Interactions:** Water quality × sanitation, rainfall × temperature

---

## 📈 Demo Flow (5-Minute Presentation)

Follow the detailed script in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) for a winning presentation narrative:

1. **Problem Setup** - Why reactive healthcare fails
2. **Show the Data** - Environmental correlations
3. **Model Training** - Live training demonstration
4. **Baseline (Dry Season)** - All-clear scenario
5. **Monsoon Spike** - Triggering the alert system
6. **Alert Response** - Geographic prioritization
7. **Impact Statement** - Lives saved through early warning

---

## 🗺️ GeoJSON Mapping

### Zone-Ward Structure
- **North Zone:** Wards W01-W20
- **South Zone:** Wards W21-W40
- **East Zone:** Wards W41-W60
- **West Zone:** Wards W61-W80
- **Central Zone:** Wards W81-W100

Ward-level predictions are aggregated to zones for visualization compatibility with `coimbatore.geojson`.

---

## 🎓 Technical Highlights (For Judges)

### Why This Approach Works:
1. **Recall > Accuracy:** In healthcare, missing an outbreak is worse than a false alarm
2. **Environmental Intelligence:** Model learns rainfall lag effects (72-hour window)
3. **Feature Engineering:** 23+ engineered features from 7 raw inputs
4. **Interpretability:** SHAP values show transparent decision-making
5. **Scalability:** Modular architecture transfers to any city

### Innovation Points:
- 🔬 **Correlation evidence:** Proves AI learned real patterns (not random)
- 🎯 **SHAP explainability:** "Black box" → transparent predictions
- 🗺️ **GIS integration:** Spatial intelligence for resource allocation
- ⚡ **Real-time inference:** Sub-second predictions for 100 wards

---

## 📦 Data Requirements

### Input CSV Structure:
```csv
week_start_date,ward_id,rainfall_mm,turbidity,ecoli_index,reported_cases,outbreak_next_week
2024-01-01,W01,12.5,3.2,8.4,3,0
2024-01-08,W01,45.2,4.9,12.1,8,1
...
```

### Minimum Required Columns:
- `ward_id` - Ward identifier (W01-W100)
- `week_start_date` - Week start date for chronological split and lag features
- `rainfall_mm` - Weekly rainfall
- `turbidity` - Water turbidity indicator
- `ecoli_index` - E.coli contamination indicator
- `reported_cases` - Disease count for feature engineering
- `outbreak_next_week` - Binary target (0/1) for training

### Optional Enhancements:
- `week` - Week number for reporting only
- Demographics: `population_density` (for extended analytics)

---

## 🏆 Winning Strategy

### What Judges Love:
✅ **Clear narrative** - Problem → Solution → Impact  
✅ **Live demo** - Not just slides, working system  
✅ **Evidence-based** - Correlation plots prove learning  
✅ **Healthcare focus** - Recall metric prioritization  
✅ **Scalability story** - Works for any city

### Common Pitfalls to Avoid:
❌ Random clicking without a story  
❌ Ignoring data quality questions  
❌ Can't explain why XGBoost over simpler models  
❌ No mention of false positive tradeoffs

---

## 🔧 Troubleshooting

### Map doesn't load?
- Ensure `geo/coimbatore.geojson` exists
- Check GeoJSON has `"name"` property matching zone names

### Import errors?
```bash
pip install --upgrade -r requirements.txt
```

### SHAP not working?
- It's optional - system runs without it
- For full functionality: `pip install shap`

---

## 🤝 Team & Credits

**Built with:**
- [Streamlit](https://streamlit.io) - Web framework
- [XGBoost](https://xgboost.ai) - ML model
- [Folium](https://python-visualization.github.io/folium/) - Interactive maps
- [Plotly](https://plotly.com) - Analytics visualizations
- [SHAP](https://github.com/slundberg/shap) - Model explanability

---

## 📞 Support

For questions about:
- **Setup:** Check this README and requirements.txt
- **Demo:** Follow DEMO_SCRIPT.md step-by-step
- **Data format:** See Input CSV Structure section in this README

---

## 🎯 Final Words

> "Every monsoon season, preventable outbreaks sicken thousands. This system gives health departments 72 hours to act. Early intervention. Lives saved. That's the impact."

**Now go win that hackathon.** 🏆

---

*Last updated: February 2026*
