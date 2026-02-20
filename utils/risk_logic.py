"""
Risk Classification and Alert Logic
"""
from .constants import MODERATE_RISK_RATIO, SCATTER_POINT_OPACITY, THRESHOLD_FALLBACK


def classify_risk(probability, threshold=THRESHOLD_FALLBACK):
    """
    Classify outbreak risk based on probability
    
    Args:
        probability (float): Prediction probability (0-1)
    
    Returns:
        str: Risk level (Low, Moderate, High)
    """
    threshold = float(threshold)
    moderate_cutoff = float(threshold) * MODERATE_RISK_RATIO

    if probability < moderate_cutoff:
        return "Low"
    elif probability < threshold:
        return "Moderate"
    else:
        return "High"


def get_risk_color(risk_level):
    """
    Get color code for risk level visualization
    
    Args:
        risk_level (str): Risk level (Low, Moderate, High)
    
    Returns:
        str: Hex color code
    """
    color_map = {
        "Low": "#00FF00",       # Green
        "Moderate": "#FFFF00",   # Yellow
        "High": "#FF0000"        # Red
    }
    return color_map.get(risk_level, "#CCCCCC")


def get_risk_color_rgba(risk_level, opacity=SCATTER_POINT_OPACITY):
    """
    Get RGBA color for risk level
    
    Args:
        risk_level (str): Risk level
        opacity (float): Opacity value (0-1)
    
    Returns:
        str: RGBA color string
    """
    rgba_map = {
        "Low": f"rgba(0, 255, 0, {opacity})",
        "Moderate": f"rgba(255, 255, 0, {opacity})",
        "High": f"rgba(255, 0, 0, {opacity})"
    }
    return rgba_map.get(risk_level, f"rgba(200, 200, 200, {opacity})")


def generate_alerts(df, threshold=THRESHOLD_FALLBACK, ward_col='ward_id', prob_col='probability'):
    """
    Generate alert list for high-risk wards
    
    Args:
        df (pd.DataFrame): Predictions with probabilities
        threshold (float): Decision threshold for outbreak alert
        ward_col (str): Ward ID column name
        prob_col (str): Probability column name
    
    Returns:
        pd.DataFrame: High-risk wards only
    """
    del ward_col
    high_risk_wards = df[df[prob_col] >= float(threshold)].copy()
    
    # Sort by probability descending
    high_risk_wards = high_risk_wards.sort_values(prob_col, ascending=False)
    
    return high_risk_wards


def generate_alert_message(
    ward_id,
    probability,
    threshold=THRESHOLD_FALLBACK,
    action="Immediate Water Quality Intervention",
):
    """
    Generate structured alert message
    
    Args:
        ward_id (str): Ward identifier
        probability (float): Risk probability
        action (str): Recommended action
    
    Returns:
        dict: Alert information
    """
    return {
        "ward": ward_id,
        "probability": probability,
        "risk_level": classify_risk(probability, threshold=threshold),
        "action": action,
        "priority": "CRITICAL" if probability >= float(threshold) * 1.2 else "HIGH"
    }


def get_risk_summary(df, risk_col='risk'):
    """
    Get risk distribution summary
    
    Args:
        df (pd.DataFrame): Predictions with risk levels
        risk_col (str): Risk column name
    
    Returns:
        dict: Risk counts and percentages
    """
    risk_counts = df[risk_col].value_counts()
    total = len(df)
    
    summary = {
        "low_count": risk_counts.get("Low", 0),
        "moderate_count": risk_counts.get("Moderate", 0),
        "high_count": risk_counts.get("High", 0),
        "low_pct": (risk_counts.get("Low", 0) / total * 100) if total > 0 else 0,
        "moderate_pct": (risk_counts.get("Moderate", 0) / total * 100) if total > 0 else 0,
        "high_pct": (risk_counts.get("High", 0) / total * 100) if total > 0 else 0,
        "total_wards": total
    }
    
    return summary


def get_intervention_recommendations(risk_level, probability):
    """
    Get recommended interventions based on risk level
    
    Args:
        risk_level (str): Risk classification
        probability (float): Risk probability
    
    Returns:
        list: List of recommended actions
    """
    recommendations = {
        "High": [
            "🚨 Deploy immediate field investigation team",
            "💧 Conduct urgent water quality testing",
            "🏥 Alert local health facilities",
            "📢 Issue public health advisory",
            "🔬 Collect environmental samples"
        ],
        "Moderate": [
            "⚠️ Schedule preventive inspections",
            "💧 Monitor water quality indicators",
            "📊 Increase surveillance frequency",
            "🧪 Review sanitation infrastructure"
        ],
        "Low": [
            "✅ Continue routine monitoring",
            "📈 Track trend indicators",
            "🔍 Maintain data quality"
        ]
    }
    
    return recommendations.get(risk_level, [])


def calculate_trend(df, ward_id, prob_col='probability', week_col='week_start_date'):
    """
    Calculate risk trend for a specific ward
    
    Args:
        df (pd.DataFrame): Historical predictions
        ward_id (str): Ward to analyze
        prob_col (str): Probability column
        week_col (str): Week column
    
    Returns:
        str: Trend direction (Increasing, Decreasing, Stable)
    """
    ward_data = df[df['ward_id'] == ward_id].sort_values(week_col)
    
    if len(ward_data) < 2:
        return "Insufficient Data"
    
    recent_prob = ward_data[prob_col].iloc[-1]
    previous_prob = ward_data[prob_col].iloc[-2]
    
    diff = recent_prob - previous_prob
    
    if diff > 0.1:
        return "📈 Increasing"
    elif diff < -0.1:
        return "📉 Decreasing"
    else:
        return "➡️ Stable"


def _severity_from_probability(probability: float, threshold: float) -> str:
    ratio = float(probability) / max(float(threshold), 1e-6)
    if ratio >= 1.5:
        return "Critical"
    if ratio >= 1.2:
        return "High"
    if ratio >= 1.0:
        return "Watch"
    return "Info"


def _recommended_intervention(severity: str) -> str:
    if severity == "Critical":
        return "Immediate chlorination, rapid field testing, and ward-level health advisory."
    if severity == "High":
        return "Deploy targeted water quality testing and preventive medical outreach within 24 hours."
    if severity == "Watch":
        return "Increase surveillance frequency and pre-position sanitation response teams."
    return "Continue routine monitoring and data quality checks."


def generate_alert_objects(predictions_df, threshold=THRESHOLD_FALLBACK, timeseries_df=None):
    high_risk = predictions_df[predictions_df['probability'] >= float(threshold)].copy()
    if len(high_risk) == 0:
        return []

    high_risk['zone'] = high_risk['ward_id'].apply(lambda ward: __import__('utils.geo_mapping', fromlist=['map_ward_to_zone']).map_ward_to_zone(ward))
    high_risk = high_risk.sort_values('probability', ascending=False).reset_index(drop=True)

    alerts = []
    for rank, row in high_risk.iterrows():
        trend = "➡️ Stable"
        if timeseries_df is not None and len(timeseries_df) > 0:
            trend = calculate_trend(timeseries_df, row['ward_id'])

        severity = _severity_from_probability(row['probability'], float(threshold))
        alerts.append(
            {
                'rank': int(rank + 1),
                'ward': row['ward_id'],
                'zone': row['zone'],
                'probability': float(row['probability']),
                'trend': trend,
                'severity': severity,
                'recommended_intervention': _recommended_intervention(severity),
            }
        )
    return alerts
