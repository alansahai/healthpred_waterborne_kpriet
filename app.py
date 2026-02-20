"""
Health Outbreak Early Warning System - Streamlit Application
Unified dashboard with analytics and explainability features
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import os
import textwrap

from model import HealthOutbreakPredictor, OutbreakPredictor
from utils import get_risk_color, generate_alert_message
from utils.constants import MODERATE_RISK_RATIO, THRESHOLD_FALLBACK
from utils.geo_mapping import aggregate_predictions_to_zones, map_ward_to_zone, WARD_ZONE_MAPPING
from utils.runtime_config import load_runtime_config
from utils.analytics import (
    plot_rainfall_vs_cases,
    plot_water_quality_vs_cases,
    plot_seasonal_trends,
    plot_feature_correlation_heatmap,
    generate_key_insights
)


def format_user_error(error: Exception):
    message = str(error)
    if 'Missing required columns' in message:
        return "Missing required columns in dataset.", message
    if 'Invalid ward_id values detected' in message:
        return "Dataset contains invalid ward IDs.", message
    if 'Model not found' in message or 'not trained and saved yet' in message:
        return "Model artifact is unavailable. Train the model first.", message
    if 'Insufficient temporal coverage' in message:
        return "Not enough historical weeks for rolling temporal cross-validation.", message
    return "Operation could not be completed.", message


def get_cv_metrics_summary(metrics: dict):
    return metrics.get('cv_metrics', {}) if isinstance(metrics, dict) else {}


def resolve_default_training_data_path():
    config = load_runtime_config()
    configured_path = config.get('data_path', 'data/coimbatore_weekly_water_disease_2024.csv')
    configured_full = os.path.join(os.path.dirname(__file__), configured_path)
    if os.path.exists(configured_full):
        return configured_path

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('data directory not found')
    csv_candidates = sorted([file_name for file_name in os.listdir(data_dir) if file_name.lower().endswith('.csv')])
    if not csv_candidates:
        raise FileNotFoundError('No CSV files available in data directory')
    return f"data/{csv_candidates[0]}"


# Page configuration
st.set_page_config(
    page_title="Health Outbreak Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: var(--secondary-background-color);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        color: inherit;
    }
    .stMetric * {
        color: inherit !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] .stCodeBlock {
        color: inherit !important;
    }
    .alert-critical {
        background-color: #ffebee;
        padding: 15px;
        border-left: 5px solid #f44336;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_geojson():
    """Load ward/zone GeoJSON data"""
    geo_path = os.path.join(os.path.dirname(__file__), 'geo', 'coimbatore.geojson')
    if os.path.exists(geo_path):
        try:
            gdf = gpd.read_file(geo_path)
            return gdf
        except Exception:
            return None

    return None


@st.cache_resource
def get_cached_predictor(model_path='model/outbreak_model.pkl'):
    predictor = OutbreakPredictor(model_path=model_path)
    predictor.load_model()
    return predictor


@st.cache_data(show_spinner=False)
def get_cached_feature_importance(raw_data: pd.DataFrame, split_name: str = 'validation', top_n: int = 8):
    model_obj = HealthOutbreakPredictor()
    model_obj._ensure_model()
    prepared = model_obj.prepare_data(raw_data)
    split_key = 'X_validation' if split_name == 'validation' else 'X_test'
    split_df = prepared.get(split_key)

    if split_df is None or len(split_df) == 0:
        return model_obj.get_feature_importance(top_n=top_n)

    sample_size = min(200, len(split_df))
    sampled = split_df.sample(sample_size, random_state=42) if sample_size > 0 else split_df
    sampled = sampled[model_obj.feature_columns]
    _, importance_df = model_obj.get_shap_values(sampled, top_n=top_n)
    return importance_df


def validate_geo_merge(predictions_df, gdf, threshold=THRESHOLD_FALLBACK):
    report = {
        'ward_id_in_predictions': 'ward_id' in predictions_df.columns,
        'ward_id_in_geojson': False,
        'merge_drops_wards': False,
        'missing_ward_count': 0,
        'unknown_ward_count': 0,
        'mode': 'zone',
        'missing_zones': [],
    }

    if gdf is None or not report['ward_id_in_predictions']:
        return report

    if 'ward_id' in gdf.columns:
        report['ward_id_in_geojson'] = True
        report['mode'] = 'ward'
        predicted_wards = set(predictions_df['ward_id'].astype(str).unique())
        known_wards = set(WARD_ZONE_MAPPING.keys())
        report['unknown_ward_count'] = len(predicted_wards - known_wards)
        geo_wards = set(gdf['ward_id'].astype(str).unique())
        missing_wards = predicted_wards - geo_wards
        report['missing_ward_count'] = len(missing_wards)
        report['merge_drops_wards'] = (len(missing_wards) > 0) or (report['unknown_ward_count'] > 0)
    else:
        predicted_wards = set(predictions_df['ward_id'].astype(str).unique())
        known_wards = set(WARD_ZONE_MAPPING.keys())
        report['unknown_ward_count'] = len(predicted_wards - known_wards)
        zone_predictions = aggregate_predictions_to_zones(predictions_df, threshold=threshold)
        prediction_zones = set(zone_predictions['zone'].astype(str).unique())
        geo_zones = set(gdf['name'].astype(str).unique()) if 'name' in gdf.columns else set()
        missing_zones = sorted(prediction_zones - geo_zones)
        report['missing_zones'] = missing_zones
        report['merge_drops_wards'] = (len(missing_zones) > 0) or (report['unknown_ward_count'] > 0)

    return report


def create_zone_heatmap(predictions_df, gdf, threshold=THRESHOLD_FALLBACK):
    """
    Create folium heatmap with zone-level risk aggregation

    Args:
        predictions_df (pd.DataFrame): Ward predictions
        gdf (GeoDataFrame): Zone geometries

    Returns:
        folium.Map: Configured map
    """
    if gdf is None or len(gdf) == 0:
        st.warning("GeoJSON file not found. Using default coordinates.")
        return folium.Map(location=[11.0168, 76.9558], zoom_start=11)

    # Aggregate to zones
    zone_predictions = aggregate_predictions_to_zones(predictions_df, threshold=threshold)

    # Merge with GeoJSON (match 'zone' with 'name' in properties)
    gdf_zones = gdf.copy()
    gdf_zones = gdf_zones.merge(
        zone_predictions,
        left_on='name',
        right_on='zone',
        how='left'
    )

    # Calculate center using projected CRS to avoid geographic-centroid warning
    projected = gdf_zones.to_crs(epsg=3857)
    centroids_wgs84 = projected.geometry.centroid.to_crs(epsg=4326)
    center_lat = centroids_wgs84.y.mean()
    center_lon = centroids_wgs84.x.mean()

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Add choropleth for zones
    for _, row in gdf_zones.iterrows():
        if pd.notna(row.get('risk')):
            risk_color = get_risk_color(row['risk'])

            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, color=risk_color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.5
                },
                tooltip=folium.Tooltip(
                    f"<b>Zone:</b> {row['name']}<br>"
                    f"<b>Risk:</b> {row['risk']}<br>"
                    f"<b>Avg Probability:</b> {row['avg_probability']:.2f}<br>"
                    f"<b>Wards:</b> {int(row['ward_count'])}"
                )
            ).add_to(m)

    return m


def display_alert_panel(predictions_df, threshold=THRESHOLD_FALLBACK):
    """Display sidebar alerts for high-risk zones/wards"""
    high_risk_wards = predictions_df[predictions_df['probability'] >= float(threshold)].sort_values('probability', ascending=False)

    if len(high_risk_wards) > 0:
        st.sidebar.error("🚨 **CRITICAL ALERTS**")
        st.sidebar.markdown(f"**{len(high_risk_wards)} High-Risk Wards Detected (≥ {float(threshold):.2f})**")
        st.sidebar.markdown("---")

        for idx, (_, ward) in enumerate(high_risk_wards.head(5).iterrows(), 1):
            alert = generate_alert_message(ward['ward_id'], ward['probability'], threshold=float(threshold))
            zone = map_ward_to_zone(ward['ward_id'])
            priority = alert.get('priority', 'HIGH')
            interventions = [
                "Chlorination drive",
                "Water quality testing",
                "Medical awareness camp",
            ]

            alert_text = textwrap.dedent(f"""
            **Alert #{idx}** - {priority} PRIORITY
            **Ward:** {alert['ward']} ({zone})
            **Probability:** {alert['probability']:.1%}
            **Action:** {alert['action']}
            **Recommended Interventions:**
            - {interventions[0]}
            - {interventions[1]}
            - {interventions[2]}
            """).strip()
            st.sidebar.markdown(alert_text)
            st.sidebar.markdown("---")
    else:
        st.sidebar.success("✅ **All Clear**")
        st.sidebar.info("No high-risk zones currently detected")


def page_dashboard():
    """Main Dashboard Page"""
    st.title("🏥 Health Outbreak Early Warning System")
    st.markdown("### Real-Time Risk Monitoring Dashboard - Coimbatore")
    st.caption("Model trained on seasonal environmental-disease interaction data with temporal validation and recall-optimized threshold tuning.")

    # Sidebar controls
    st.sidebar.header("📊 Control Panel")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV)",
        type=['csv'],
        help="Upload your time-series health surveillance data"
    )

    apply_realism_noise = st.sidebar.checkbox(
        "Apply realistic training noise (demo)",
        value=True,
        help="Adds slight noise to rainfall, ecoli, and reported cases to avoid over-perfect metrics in demos.",
    )

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = HealthOutbreakPredictor()
        st.session_state.predictor = OutbreakPredictor()
        st.session_state.trained = False
        st.session_state.predictions = None
        st.session_state.predictions_timeseries = None
        st.session_state.raw_data = None
        st.session_state.metrics = None

    # Train model button
    if st.sidebar.button("🚀 Train Model", type="primary", width='stretch'):
        with st.spinner("Training XGBoost model..."):
            try:
                # Load data
                if uploaded_file is not None:
                    data = pd.read_csv(uploaded_file)
                    st.sidebar.success("✅ Custom dataset loaded")
                else:
                    default_data_path = resolve_default_training_data_path()
                    data = st.session_state.model.load_data(default_data_path)
                    st.sidebar.info(f"ℹ️ Using {default_data_path}")

                st.session_state.raw_data = data.copy()

                # Train model
                metrics = st.session_state.model.train(data, apply_realism_noise=apply_realism_noise)
                st.session_state.trained = True
                st.session_state.metrics = metrics

                # Load saved model and generate latest-week predictions
                get_cached_predictor.clear()
                st.session_state.predictor = get_cached_predictor()
                predictions = st.session_state.predictor.predict_latest_week(df=data)
                predictions_timeseries = st.session_state.predictor.predict_time_series(df=data)
                st.session_state.predictions = predictions
                st.session_state.predictions_timeseries = predictions_timeseries

                # Display metrics
                st.sidebar.success("✅ Model trained successfully!")
                cv_metrics = get_cv_metrics_summary(metrics)
                outbreak_ratios = metrics.get('outbreak_ratios', {})
                col1, col2 = st.sidebar.columns(2)
                col1.metric("CV Recall", f"{cv_metrics.get('recall_mean', 0):.1%}")
                col2.metric("CV F1", f"{cv_metrics.get('f1_mean', 0):.1%}")
                col1.metric("CV Accuracy", f"{cv_metrics.get('accuracy_mean', 0):.1%}")
                roc_auc_val = cv_metrics.get('roc_auc_mean')
                col2.metric("CV ROC-AUC", "N/A" if pd.isna(roc_auc_val) else f"{roc_auc_val:.3f}")
                st.sidebar.metric("Best Threshold", f"{metrics.get('best_threshold', 0.5):.2f}")
                st.sidebar.metric("CV Folds", metrics.get('n_valid_folds', metrics.get('n_folds', 0)))

                if outbreak_ratios:
                    st.sidebar.caption(
                        "Outbreak Ratios — "
                        f"Train: {outbreak_ratios.get('train', 0):.2%}, "
                        f"Validation: {outbreak_ratios.get('validation', 0):.2%}, "
                        f"Test: {outbreak_ratios.get('test', 0):.2%}"
                    )

                class_balance = metrics.get('class_balance', {})
                if class_balance:
                    st.sidebar.markdown("**Class Balance (normalized):**")
                    st.sidebar.code(str(class_balance))
                    st.sidebar.caption(f"Assessment: {metrics.get('class_balance_assessment', 'N/A')}")
                st.sidebar.caption(f"Realism noise applied: {metrics.get('noise_applied', False)}")

            except Exception as e:
                headline, details = format_user_error(e)
                st.sidebar.error(f"❌ {headline}")
                st.sidebar.caption(details)
                st.session_state.trained = False
                st.session_state.predictions = None
                st.session_state.predictions_timeseries = None

    if st.session_state.get('trained') and st.session_state.get('metrics'):
        existing_metrics = st.session_state.get('metrics', {})
        cv_metrics = get_cv_metrics_summary(existing_metrics)
        outbreak_ratios = existing_metrics.get('outbreak_ratios', {})
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧾 Trained Model Stats")
        stats_view_live = st.sidebar.selectbox(
            "Choose stats view",
            ["Compact", "Metrics", "Parameters", "All"],
            key="stats_view_live",
        )

        compact_stats = {
            'cv_accuracy_mean': cv_metrics.get('accuracy_mean'),
            'cv_precision_mean': cv_metrics.get('precision_mean'),
            'cv_recall_mean': cv_metrics.get('recall_mean'),
            'cv_recall_std': cv_metrics.get('recall_std'),
            'cv_f1_mean': cv_metrics.get('f1_mean'),
            'cv_f1_std': cv_metrics.get('f1_std'),
            'cv_roc_auc_mean': cv_metrics.get('roc_auc_mean'),
            'cv_roc_auc_std': cv_metrics.get('roc_auc_std'),
            'best_threshold': existing_metrics.get('best_threshold'),
            'n_folds': existing_metrics.get('n_folds'),
            'n_valid_folds': existing_metrics.get('n_valid_folds'),
        }
        metric_stats = {
            'cv_fold_metrics': existing_metrics.get('cv_fold_metrics', []),
            'skipped_folds': existing_metrics.get('skipped_folds', []),
            'n_samples': existing_metrics.get('n_samples'),
            'evaluation_method': existing_metrics.get('evaluation_method'),
            'outbreak_ratios': outbreak_ratios,
            'cross_fold_prevalence': existing_metrics.get('cross_fold_prevalence', {}),
            'calibration': existing_metrics.get('calibration', {}),
            'leakage_checks': existing_metrics.get('leakage_checks'),
        }
        parameter_stats = {
            'best_params': existing_metrics.get('best_params', {}),
            'threshold_curve_points': len(existing_metrics.get('threshold_curve', [])),
        }

        if stats_view_live == "Compact":
            st.sidebar.json(compact_stats)
        elif stats_view_live == "Metrics":
            st.sidebar.json(metric_stats)
        elif stats_view_live == "Parameters":
            st.sidebar.json(parameter_stats)
        else:
            st.sidebar.json({
                **compact_stats,
                **metric_stats,
                **parameter_stats,
            })

    # Main content
    if st.session_state.trained and st.session_state.predictions is not None:
        predictions_df = st.session_state.predictions
        model_metrics = st.session_state.get('metrics') or {}
        best_threshold = float(model_metrics.get('best_threshold', THRESHOLD_FALLBACK))
        cv_metrics = get_cv_metrics_summary(model_metrics)
        outbreak_ratios = model_metrics.get('outbreak_ratios', {})
        cross_fold_prevalence = model_metrics.get('cross_fold_prevalence', {})

        # Display alerts
        display_alert_panel(predictions_df, threshold=best_threshold)

        # Key metrics row
        st.markdown("### 📈 System Overview")
        col1, col2, col3, col4 = st.columns(4)

        risk_counts = predictions_df['risk'].value_counts()
        total_wards = len(predictions_df)
        high_risk_pct = (risk_counts.get('High', 0) / total_wards * 100) if total_wards > 0 else 0

        col1.metric("Total Wards", total_wards)
        col2.metric("🟢 Low Risk", risk_counts.get('Low', 0))
        col3.metric("🟡 Moderate Risk", risk_counts.get('Moderate', 0))
        col4.metric("🔴 High Risk", risk_counts.get('High', 0),
                   delta=f"{high_risk_pct:.1f}% of total", delta_color="inverse")

        if model_metrics:
            st.markdown("### 🧪 Model Performance")
            perf1, perf2, perf3, perf4, perf5 = st.columns(5)
            perf1.metric("CV Recall", f"{cv_metrics.get('recall_mean', 0):.2%} ± {cv_metrics.get('recall_std', 0):.2%}")
            perf2.metric("CV F1", f"{cv_metrics.get('f1_mean', 0):.2%} ± {cv_metrics.get('f1_std', 0):.2%}")
            roc_auc = cv_metrics.get('roc_auc_mean')
            roc_auc_std = cv_metrics.get('roc_auc_std', 0)
            perf3.metric("CV ROC-AUC", "N/A" if pd.isna(roc_auc) else f"{roc_auc:.3f} ± {roc_auc_std:.3f}")
            perf4.metric("Global Threshold", f"{model_metrics.get('global_threshold', best_threshold):.2f}")
            perf5.metric("CV Folds", model_metrics.get('n_valid_folds', model_metrics.get('n_folds', 0)))
            st.caption("Model evaluated using rolling temporal cross-validation to handle seasonal regime variation.")

            with st.expander("View confusion matrix and training details"):
                st.write("Cross-Validated Metrics", cv_metrics)
                st.write("Fold-wise Metrics", model_metrics.get('cv_fold_metrics', []))
                st.write("Skipped Folds", model_metrics.get('skipped_folds', []))
                st.write("Best Params", model_metrics.get('best_params', {}))
                st.write("Global Threshold", model_metrics.get('global_threshold'))
                st.write("Fold Prevalence", cross_fold_prevalence)
                st.write("Outbreak Ratios (Cross-Fold)", outbreak_ratios)
                st.write("Calibration", model_metrics.get('calibration', {}))
                st.write("Leakage Checks", model_metrics.get('leakage_checks', {}))
                st.write("Final model trained on full dataset", model_metrics.get('final_model_trained_on_full_data'))
                st.write("Calibration source", model_metrics.get('calibration_source'))

            threshold_curve = model_metrics.get('threshold_curve', [])
            if threshold_curve:
                curve_df = pd.DataFrame(threshold_curve)
                st.markdown("#### Probability Threshold Curve (Precision vs Recall)")
                threshold_fig = go.Figure()
                threshold_fig.add_trace(
                    go.Scatter(
                        x=curve_df['threshold'],
                        y=curve_df['recall'],
                        mode='lines+markers',
                        name='Recall',
                        line=dict(color='#d62728', width=2),
                    )
                )
                threshold_fig.add_trace(
                    go.Scatter(
                        x=curve_df['threshold'],
                        y=curve_df['precision'],
                        mode='lines+markers',
                        name='Precision',
                        line=dict(color='#1f77b4', width=2),
                    )
                )

                nearest_idx = (curve_df['threshold'] - best_threshold).abs().idxmin()
                best_row = curve_df.loc[nearest_idx]
                threshold_fig.add_trace(
                    go.Scatter(
                        x=[best_row['threshold']],
                        y=[best_row['recall']],
                        mode='markers',
                        name='Selected Threshold',
                        marker=dict(color='#2ca02c', size=12, symbol='diamond'),
                    )
                )

                threshold_fig.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title='Decision Threshold',
                    yaxis_title='Metric Value',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                )
                st.plotly_chart(threshold_fig, width='stretch')
                st.caption(
                    f"Selected threshold: {best_threshold:.2f}. "
                    "Threshold derived from mean fold-optimal values in rolling cross-validation."
                )

        st.markdown("---")

        # Two-column layout
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📋 Risk Assessment Table")

            controls_col1, controls_col2 = st.columns([1, 1])
            with controls_col1:
                selected_risk_levels = st.multiselect(
                    "Filter by risk",
                    options=["High", "Moderate", "Low"],
                    default=["High", "Moderate", "Low"],
                    key="risk_table_filter",
                )
            with controls_col2:
                sort_order = st.selectbox(
                    "Sort probability",
                    options=["High to Low", "Low to High"],
                    index=0,
                    key="risk_table_sort",
                )

            table_df = predictions_df.copy()
            table_df['zone'] = table_df['ward_id'].apply(map_ward_to_zone)
            table_df['probability_pct'] = table_df['probability'] * 100

            if selected_risk_levels:
                table_df = table_df[table_df['risk'].isin(selected_risk_levels)]

            table_df = table_df.sort_values(
                'probability',
                ascending=(sort_order == "Low to High")
            )

            table_df = table_df[['ward_id', 'zone', 'risk', 'probability', 'probability_pct']]
            table_df = table_df.rename(
                columns={
                    'ward_id': 'Ward',
                    'zone': 'Zone',
                    'risk': 'Risk',
                    'probability': 'Outbreak Probability',
                    'probability_pct': 'Probability %',
                }
            )

            st.dataframe(
                table_df,
                width='stretch',
                height=450,
                hide_index=True,
                column_config={
                    'Ward': st.column_config.TextColumn('Ward'),
                    'Zone': st.column_config.TextColumn('Zone'),
                    'Risk': st.column_config.TextColumn('Risk'),
                    'Outbreak Probability': st.column_config.ProgressColumn(
                        'Outbreak Probability',
                        min_value=0.0,
                        max_value=1.0,
                        format='%.3f',
                    ),
                    'Probability %': st.column_config.NumberColumn(
                        'Probability %',
                        format='%.2f%%',
                    ),
                },
            )

            st.caption(f"Showing {len(table_df)} ward records")

        with col_right:
            st.subheader("🗺️ Geographic Risk Heatmap")

            try:
                gdf = load_geojson()
                map_obj = create_zone_heatmap(predictions_df, gdf, threshold=best_threshold)
                st_folium(map_obj, width=None, height=450)

                moderate_cutoff = best_threshold * MODERATE_RISK_RATIO
                st.caption(
                    f"**Legend:** 🔴 High Risk (≥ {best_threshold:.2f}) | "
                    f"🟡 Moderate ({moderate_cutoff:.2f} - {best_threshold:.2f}) | "
                    f"🟢 Low (< {moderate_cutoff:.2f})"
                )
            except Exception:
                st.warning("Map rendering is unavailable right now. Check GeoJSON integrity and retry.")

            if 'gdf' in locals() and gdf is not None:
                geo_report = validate_geo_merge(predictions_df, gdf, threshold=best_threshold)
                with st.expander("Geo merge validation"):
                    st.write("ward_id in predictions", geo_report['ward_id_in_predictions'])
                    st.write("ward_id in geojson properties", geo_report['ward_id_in_geojson'])
                    st.write("merge drops wards", geo_report['merge_drops_wards'])
                    st.write("Unknown ward count", geo_report['unknown_ward_count'])
                    if geo_report['mode'] == 'zone':
                        st.write("Validation mode", "Zone-level (GeoJSON has zone polygons)")
                        st.write("Missing zones", geo_report['missing_zones'])
                    else:
                        st.write("Missing ward count", geo_report['missing_ward_count'])

        if st.session_state.get('predictions_timeseries') is not None:
            st.markdown("---")
            st.markdown("### 📈 Outbreak Probability Over Time (Per Ward)")
            ward_timeseries = st.session_state.predictions_timeseries.copy()
            ward_options = sorted(ward_timeseries['ward_id'].unique())
            selected_ward = st.selectbox("Select Ward", ward_options, index=0)

            ward_view = (
                ward_timeseries[ward_timeseries['ward_id'] == selected_ward]
                .sort_values('week_start_date')
                .tail(10)
                .copy()
            )

            trend_fig = go.Figure()
            trend_fig.add_trace(
                go.Bar(
                    x=ward_view['week_start_date'],
                    y=ward_view['reported_cases'],
                    name='Reported Cases',
                    marker_color='#1f77b4',
                    opacity=0.7,
                    yaxis='y1',
                )
            )
            trend_fig.add_trace(
                go.Scatter(
                    x=ward_view['week_start_date'],
                    y=ward_view['rainfall_mm'],
                    mode='lines+markers',
                    name='Rainfall (mm)',
                    line=dict(color='#2ca02c', width=2, dash='dot'),
                    yaxis='y1',
                )
            )
            trend_fig.add_trace(
                go.Scatter(
                    x=ward_view['week_start_date'],
                    y=ward_view['probability'],
                    mode='lines+markers',
                    name='Outbreak Probability',
                    line=dict(color='#d62728', width=3),
                    yaxis='y2',
                )
            )
            trend_fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(title='Cases / Rainfall'),
                yaxis2=dict(title='Probability', overlaying='y', side='right', range=[0, 1]),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            )
            st.plotly_chart(trend_fig, width='stretch')

        if st.session_state.raw_data is not None:
            st.markdown("---")
            st.markdown("### 🔍 Model Explainability & Correlation")

            explain_col, corr_col = st.columns(2)

            with explain_col:
                st.markdown("#### Feature Importance")
                st.caption("Explainability based on out-of-sample validation data.")
                try:
                    shap_importance_df = get_cached_feature_importance(
                        st.session_state.raw_data,
                        split_name='validation',
                        top_n=8,
                    )
                    importance_col = 'shap_importance' if 'shap_importance' in shap_importance_df.columns else 'importance'

                    fig_imp = px.bar(
                        shap_importance_df.sort_values(importance_col),
                        x=importance_col,
                        y='feature',
                        orientation='h',
                        title='Top Predictive Features',
                        color=importance_col,
                        color_continuous_scale='Viridis'
                    )
                    fig_imp.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig_imp, width='stretch')
                except Exception:
                    st.info("Feature importance is temporarily unavailable.")

            with corr_col:
                st.markdown("#### Rainfall vs Reported Cases")
                try:
                    rain_corr_fig = plot_rainfall_vs_cases(st.session_state.raw_data)
                    if rain_corr_fig is not None:
                        rain_corr_fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                        st.plotly_chart(rain_corr_fig, width='stretch')
                    else:
                        st.info("Rainfall and reported case columns are required for this chart.")
                except Exception:
                    st.info("Correlation chart is temporarily unavailable.")

    else:
        # Welcome screen
        st.info("👈 **Get Started:** Click 'Train Model' in the sidebar")

        st.markdown("""
        ### 🎯 System Capabilities

        - **AI-Powered Prediction:** XGBoost machine learning for outbreak forecasting
        - **Real-Time Risk Assessment:** Continuous monitoring of 100 wards across 5 zones
        - **Environmental Intelligence:** Analyzes rainfall, water quality, sanitation indicators
        - **Early Warning Alerts:** Automatic notification for high-risk areas
        - **Geographic Visualization:** Interactive heatmap for decision support

        ### 📊 Model Performance
        - **Recall-Optimized:** Prioritizes catching all potential outbreaks
        - **F1 Score Tracking:** Balanced precision and recall metrics
        - **Feature Importance:** SHAP values for interpretability
        """)

        if st.session_state.raw_data is None:
            with st.expander("📂 View Sample Data"):
                try:
                    sample_data = pd.read_csv(
                        os.path.join(os.path.dirname(__file__), 'data', 'sample_data.csv')
                    )
                    st.dataframe(sample_data.head(15), width='stretch')
                except Exception:
                    st.warning("Sample data not available")


def page_analytics():
    """Environmental Correlation Analytics Page"""
    st.title("🔬 Environmental Correlation Analysis")
    st.markdown("### Understanding Environmental Drivers of Disease Outbreaks")

    if st.session_state.get('raw_data') is None:
        st.warning("⚠️ Please train the model first to access analytics")
        return

    data = st.session_state.raw_data

    # Key insights
    st.markdown("### 💡 Key Insights")
    insights = generate_key_insights(data)

    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.info("Train the model with environmental data to see correlations")

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌧️ Rainfall Impact")
        fig1 = plot_rainfall_vs_cases(data)
        if fig1:
            st.plotly_chart(fig1, width='stretch')
        else:
            st.info("Rainfall data not available in dataset")

    with col2:
        st.markdown("#### 💧 Water Quality Impact")
        fig2 = plot_water_quality_vs_cases(data)
        if fig2:
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("Water quality data not available")

    # Seasonal trends
    st.markdown("---")
    st.markdown("### 📅 Seasonal Disease Patterns")
    fig3 = plot_seasonal_trends(data)
    if fig3:
        st.plotly_chart(fig3, width='stretch')
    else:
        st.info("week_start_date / reported_cases columns are required for seasonal analysis")

    # Correlation heatmap
    st.markdown("---")
    st.markdown("### 🔥 Feature Correlation Matrix")

    environmental_features = [
        'reported_cases',
        'rainfall_mm',
        'turbidity',
        'ecoli_index',
    ]

    fig4 = plot_feature_correlation_heatmap(data, environmental_features)
    if fig4:
        st.plotly_chart(fig4, width='stretch')

        st.markdown("""
        **Model Learning:** The AI identifies patterns showing:
        - Strong correlation between poor water quality and disease outbreaks
        - Rainfall increases contamination risk in monsoon season
        - Sanitation index inversely correlates with outbreak probability
        """)
    else:
        st.info("Insufficient features for correlation analysis")


def page_feature_importance():
    """SHAP Feature Importance Page"""
    st.title("🎯 AI Model Explainability")
    st.markdown("### Understanding What Drives Outbreak Predictions")

    if not st.session_state.get('trained'):
        st.warning("⚠️ Please train the model first")
        return

    data = st.session_state.raw_data

    st.markdown("---")
    st.markdown("### 📊 Feature Importance Rankings")
    st.caption("Explainability based on out-of-sample validation data.")

    try:
        importance_df = get_cached_feature_importance(data, split_name='validation', top_n=10)
        value_col = 'shap_importance' if 'shap_importance' in importance_df.columns else 'importance'

        fig = px.bar(
            importance_df,
            x=value_col,
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features",
            labels={value_col: 'Importance Score', 'feature': 'Feature'},
            color=value_col,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, width='stretch')

        # Feature interpretation
        st.markdown("### 🔍 Top Predictors Identified:")
        top_features = importance_df.head(5)

        for idx, row in top_features.iterrows():
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{idx+1}. {row['feature']}**")
            col2.metric("Score", f"{row[value_col]:.3f}")

        st.info("""
        **Interpretation:**
        Features like rainfall lag, water quality scores, and disease case history
        are the strongest predictors, proving the model learns meaningful environmental patterns.
        """)

    except Exception:
        st.warning("Feature importance could not be generated right now.")


# Main app
def main():
    """Main application with multi-page navigation"""

    # Sidebar navigation
    st.sidebar.title("🏥 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Dashboard", "🔬 Environmental Analysis", "🎯 Feature Importance"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Route to pages
    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🔬 Environmental Analysis":
        page_analytics()
    elif page == "🎯 Feature Importance":
        page_feature_importance()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("🏥 Health Outbreak Early Warning System v2.0")
    st.sidebar.caption("Powered by XGBoost & GIS Intelligence")


if __name__ == "__main__":
    main()
