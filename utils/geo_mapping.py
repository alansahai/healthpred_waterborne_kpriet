"""
GeoJSON Mapping Utilities
Handles ward-to-zone mapping for visualization
"""
import pandas as pd


# Zone-to-Ward mapping based on Coimbatore administrative structure
ZONE_WARD_MAPPING = {
    "North Zone": [f"W{i:02d}" for i in range(1, 21)],      # W01-W20
    "South Zone": [f"W{i:02d}" for i in range(21, 41)],     # W21-W40
    "East Zone": [f"W{i:02d}" for i in range(41, 61)],      # W41-W60
    "West Zone": [f"W{i:02d}" for i in range(61, 81)],      # W61-W80
    "Central Zone": [f"W{i:02d}" for i in range(81, 101)]   # W81-W100
}

# Reverse mapping: Ward -> Zone
WARD_ZONE_MAPPING = {}
for zone, wards in ZONE_WARD_MAPPING.items():
    for ward in wards:
        WARD_ZONE_MAPPING[ward] = zone


def map_ward_to_zone(ward_id):
    """
    Map ward ID to zone name
    
    Args:
        ward_id (str): Ward identifier (e.g., 'W01')
    
    Returns:
        str: Zone name
    """
    return WARD_ZONE_MAPPING.get(ward_id, "Unknown Zone")


def aggregate_predictions_to_zones(predictions_df, threshold=0.5):
    """
    Aggregate ward-level predictions to zone level
    
    Args:
        predictions_df (pd.DataFrame): Ward-level predictions
    
    Returns:
        pd.DataFrame: Zone-level aggregated predictions
    """
    # Add zone column on a copy (avoid mutating caller DataFrame)
    predictions_with_zone = predictions_df.copy()
    predictions_with_zone['zone'] = predictions_with_zone['ward_id'].apply(map_ward_to_zone)
    
    # Aggregate by zone
    zone_agg = predictions_with_zone.groupby('zone').agg({
        'probability': 'mean',  # Average probability
        'ward_id': 'count'      # Number of wards
    }).reset_index()
    
    zone_agg.columns = ['zone', 'avg_probability', 'ward_count']
    
    # Apply risk classification to zone average
    from .risk_logic import classify_risk
    zone_agg['risk_level'] = zone_agg['avg_probability'].apply(lambda probability: classify_risk(probability, threshold=threshold))
    
    return zone_agg


def get_high_risk_wards_by_zone(predictions_df):
    """
    Get high-risk wards grouped by zone
    
    Args:
        predictions_df (pd.DataFrame): Ward-level predictions with risk
    
    Returns:
        dict: Zone -> list of high-risk wards
    """
    high_risk = predictions_df[predictions_df['risk_level'] == 'High'].copy()
    high_risk['zone'] = high_risk['ward_id'].apply(map_ward_to_zone)
    
    zone_dict = {}
    for zone in high_risk['zone'].unique():
        zone_wards = high_risk[high_risk['zone'] == zone]['ward_id'].tolist()
        zone_dict[zone] = zone_wards
    
    return zone_dict


def validate_geojson_mapping(gdf, predictions_df):
    """
    Validate that GeoJSON zones match prediction data
    
    Args:
        gdf (GeoDataFrame): GeoJSON data
        predictions_df (pd.DataFrame): Predictions with ward_id
    
    Returns:
        dict: Validation report
    """
    geojson_zones = set(gdf['name'].unique()) if 'name' in gdf.columns else set()
    prediction_zones = set(predictions_df['ward_id'].apply(map_ward_to_zone).unique())
    
    return {
        'geojson_zones': sorted(geojson_zones),
        'prediction_zones': sorted(prediction_zones),
        'matched': geojson_zones == prediction_zones,
        'missing_in_geojson': prediction_zones - geojson_zones,
        'missing_in_predictions': geojson_zones - prediction_zones
    }
