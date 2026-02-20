"""
Analytics and Visualization Utilities
Environmental correlation analysis for presentations
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr

from importlib import import_module

from .constants import INSIGHT_CORRELATION_CUTOFF, SCATTER_POINT_OPACITY

try:
    import_module('statsmodels.api')
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def calculate_correlation_matrix(df, features):
    """
    Calculate correlation matrix for selected features
    
    Args:
        df (pd.DataFrame): Input data
        features (list): List of feature columns
    
    Returns:
        pd.DataFrame: Correlation matrix
    """
    return df[features].corr()


def _paired_numeric(df, x_col, y_col):
    paired = df[[x_col, y_col]].copy()
    paired[x_col] = pd.to_numeric(paired[x_col], errors='coerce')
    paired[y_col] = pd.to_numeric(paired[y_col], errors='coerce')
    paired = paired.dropna(subset=[x_col, y_col])
    return paired


def plot_rainfall_vs_cases(df, rainfall_col='rainfall_mm', cases_col='reported_cases'):
    """
    Create scatter plot: Rainfall vs Disease Cases
    
    Args:
        df (pd.DataFrame): Input data
        rainfall_col (str): Rainfall column name
        cases_col (str): Disease cases column name
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if rainfall_col not in df.columns or cases_col not in df.columns:
        return None
    
    paired = _paired_numeric(df, rainfall_col, cases_col)
    if len(paired) < 3:
        return None

    # Calculate correlation
    corr, p_value = pearsonr(paired[rainfall_col], paired[cases_col])
    trendline_mode = "ols" if HAS_STATSMODELS else None
    
    fig = px.scatter(
        paired,
        x=rainfall_col, 
        y=cases_col,
        trendline=trendline_mode,
        title=f"Rainfall vs Reported Cases<br><sub>Correlation: {corr:.3f} (p={p_value:.4f})</sub>",
        labels={rainfall_col: "Rainfall (mm)", cases_col: "Reported Cases"}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=SCATTER_POINT_OPACITY))
    fig.update_layout(height=400)
    
    return fig


def plot_water_quality_vs_cases(df, water_col='turbidity', cases_col='reported_cases'):
    """
    Create scatter plot: Water Quality vs Disease Cases
    
    Args:
        df (pd.DataFrame): Input data
        water_col (str): Water quality column name
        cases_col (str): Disease cases column name
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if water_col not in df.columns or cases_col not in df.columns:
        return None
    
    paired = _paired_numeric(df, water_col, cases_col)
    if len(paired) < 3:
        return None

    # Calculate correlation
    corr, p_value = pearsonr(paired[water_col], paired[cases_col])
    trendline_mode = "ols" if HAS_STATSMODELS else None
    
    fig = px.scatter(
        paired,
        x=water_col, 
        y=cases_col,
        trendline=trendline_mode,
        title=f"Turbidity vs Reported Cases<br><sub>Correlation: {corr:.3f} (p={p_value:.4f})</sub>",
        labels={water_col: "Turbidity", cases_col: "Reported Cases"},
        color=cases_col,
        color_continuous_scale="Reds"
    )
    
    fig.update_traces(marker=dict(size=8, opacity=SCATTER_POINT_OPACITY))
    fig.update_layout(height=400)
    
    return fig


def plot_seasonal_trends(df, date_col='week_start_date', cases_col='reported_cases'):
    """
    Plot seasonal disease trends
    
    Args:
        df (pd.DataFrame): Input data with week_start_date column
        date_col (str): Date-like column name
        cases_col (str): Reported cases column name
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if date_col not in df.columns or cases_col not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy['month'] = df_copy[date_col].dt.month_name()
    
    # Aggregate by month
    monthly_avg = df_copy.groupby('month')[cases_col].mean().reset_index()
    
    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_avg['month'] = pd.Categorical(monthly_avg['month'], categories=month_order, ordered=True)
    monthly_avg = monthly_avg.sort_values('month')
    
    fig = px.bar(
        monthly_avg,
        x='month',
        y=cases_col,
        title="Seasonal Disease Pattern (Monthly Average)",
        labels={'month': 'Month', cases_col: 'Avg Reported Cases'},
        color=cases_col,
        color_continuous_scale="RdYlGn_r"
    )
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    
    return fig


def plot_feature_correlation_heatmap(df, features):
    """
    Plot correlation heatmap for features
    
    Args:
        df (pd.DataFrame): Input data
        features (list): List of feature columns to include
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 2:
        return None
    
    corr_matrix = df[available_features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Environmental Factors Correlation Matrix",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig


def generate_key_insights(df):
    """
    Generate key analytical insights from data
    
    Args:
        df (pd.DataFrame): Input data
    
    Returns:
        list: List of insight strings
    """
    insights = []
    
    # Rainfall correlation
    if 'rainfall_mm' in df.columns and 'reported_cases' in df.columns:
        paired = _paired_numeric(df, 'rainfall_mm', 'reported_cases')
        if len(paired) >= 3:
            corr, _ = pearsonr(paired['rainfall_mm'], paired['reported_cases'])
            if corr > INSIGHT_CORRELATION_CUTOFF:
                insights.append(f"🌧️ Strong positive correlation ({corr:.2f}) between rainfall and reported cases")
            elif corr < -INSIGHT_CORRELATION_CUTOFF:
                insights.append(f"☀️ Negative correlation ({corr:.2f}) between rainfall and reported cases")
    
    # Water quality correlation
    if 'turbidity' in df.columns and 'reported_cases' in df.columns:
        paired = _paired_numeric(df, 'turbidity', 'reported_cases')
        if len(paired) >= 3:
            corr, _ = pearsonr(paired['turbidity'], paired['reported_cases'])
            if corr > INSIGHT_CORRELATION_CUTOFF:
                insights.append(f"💧 Higher turbidity is associated ({corr:.2f}) with increased reported cases")

    if 'ecoli_index' in df.columns and 'reported_cases' in df.columns:
        paired = _paired_numeric(df, 'ecoli_index', 'reported_cases')
        if len(paired) >= 3:
            corr, _ = pearsonr(paired['ecoli_index'], paired['reported_cases'])
            if corr > INSIGHT_CORRELATION_CUTOFF:
                insights.append(f"🧫 Elevated E.coli index correlates ({corr:.2f}) with reported cases")
    
    # Seasonal patterns
    if 'week_start_date' in df.columns and 'reported_cases' in df.columns:
        df_copy = df.copy()
        df_copy['week_start_date'] = pd.to_datetime(df_copy['week_start_date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['week_start_date'])
        df_copy['month'] = df_copy['week_start_date'].dt.month
        monthly_avg = df_copy.groupby('month')['reported_cases'].mean()
        peak_month = monthly_avg.idxmax()
        month_names = {6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October'}
        if peak_month in month_names:
            insights.append(f"📊 Peak outbreak season: {month_names.get(peak_month, peak_month)} (Monsoon period)")
    
    return insights
