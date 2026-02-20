"""
Spatial Visualization Helpers

Pure post-processing visualization utilities for spatial spread analysis.
These functions do NOT modify:
- ML logic
- Risk classification
- Feature engineering
- Training pipeline
- Prediction outputs

They only consume existing predictions and spatial metadata for visualization.
"""
import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

logger = logging.getLogger(__name__)


def _project_root() -> str:
    """Get project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json_safe(filepath: str) -> Optional[Dict]:
    """Load JSON file safely, returning None if missing or invalid."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load {filepath}: {e}")
    return None


def load_ward_adjacency() -> Dict[str, List[str]]:
    """Load ward adjacency mapping.
    
    Returns:
        Dict mapping ward_id -> list of neighbor ward_ids
        Empty dict if file missing
    """
    filepath = os.path.join(_project_root(), 'data', 'ward_adjacency.json')
    data = _load_json_safe(filepath)
    if data and isinstance(data, dict):
        return data
    return {}


def load_peripheral_wards() -> set:
    """Load set of peripheral ward IDs.
    
    Returns:
        Set of ward_ids that are peripheral
        Empty set if file missing
    """
    filepath = os.path.join(_project_root(), 'data', 'peripheral_wards.json')
    data = _load_json_safe(filepath)
    if data and 'peripheral_wards' in data:
        return set(data['peripheral_wards'])
    return set()


def load_ward_centroids() -> Dict[str, Dict[str, float]]:
    """Load ward centroid coordinates.
    
    Returns:
        Dict mapping ward_id -> {'lat': float, 'lon': float}
        Empty dict if file missing
    """
    filepath = os.path.join(_project_root(), 'spatial_metadata.json')
    data = _load_json_safe(filepath)
    if data and 'centroids' in data:
        return data['centroids']
    return {}


def compute_peripheral_indicator(predictions_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Compute percentage of high-risk wards that are peripheral.
    
    This is purely descriptive — no impact on risk logic.
    
    Args:
        predictions_df: Prediction DataFrame with 'ward_id', 'risk' columns
                       and optionally 'is_peripheral_ward' column
    
    Returns:
        Dict with:
            - peripheral_high_count: int
            - high_risk_count: int
            - percentage: float (0-100)
            - available: bool
        None if data insufficient
    """
    if predictions_df is None or len(predictions_df) == 0:
        return None
    
    if 'risk' not in predictions_df.columns:
        return None
    
    high_risk = predictions_df[predictions_df['risk'] == 'High']
    high_risk_count = len(high_risk)
    
    if high_risk_count == 0:
        return {
            'peripheral_high_count': 0,
            'high_risk_count': 0,
            'percentage': 0.0,
            'available': True,
            'message': 'No high-risk wards currently'
        }
    
    # Check if is_peripheral_ward column exists
    if 'is_peripheral_ward' in predictions_df.columns:
        peripheral_high = high_risk[high_risk['is_peripheral_ward'] == 1]
        peripheral_high_count = len(peripheral_high)
    else:
        # Fallback: load peripheral wards from file and check ward_id
        peripheral_wards = load_peripheral_wards()
        if not peripheral_wards:
            return {
                'available': False,
                'message': 'Peripheral ward data not available'
            }
        
        peripheral_high = high_risk[high_risk['ward_id'].isin(peripheral_wards)]
        peripheral_high_count = len(peripheral_high)
    
    percentage = (peripheral_high_count / high_risk_count) * 100
    
    return {
        'peripheral_high_count': peripheral_high_count,
        'high_risk_count': high_risk_count,
        'percentage': round(percentage, 1),
        'available': True,
        'message': f'{peripheral_high_count} of {high_risk_count} high-risk wards are peripheral'
    }


def compute_infection_influence_arrows(
    predictions_df: pd.DataFrame,
    threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """Compute arrows from high-risk peripheral wards to their neighbors.
    
    Influence_i→j = P_i if j ∈ N(i)
    
    Only draws arrows for wards where:
    - risk == "High"
    - is_peripheral == 1
    
    Args:
        predictions_df: Prediction DataFrame with 'ward_id', 'risk', 'probability'
        threshold: Risk threshold for "High" classification
    
    Returns:
        List of arrow definitions:
            [{
                'from_ward': str,
                'to_ward': str,
                'from_coords': (lat, lon),
                'to_coords': (lat, lon),
                'influence': float (source probability)
            }, ...]
        Empty list if data insufficient
    """
    if predictions_df is None or len(predictions_df) == 0:
        return []
    
    # Load spatial data
    adjacency = load_ward_adjacency()
    if not adjacency:
        return []
    
    peripheral_wards = load_peripheral_wards()
    centroids = load_ward_centroids()
    if not centroids:
        return []
    
    # Find high-risk peripheral wards
    high_risk = predictions_df[predictions_df['risk'] == 'High'].copy()
    
    if len(high_risk) == 0:
        return []
    
    # Determine peripheral status
    if 'is_peripheral_ward' in predictions_df.columns:
        high_risk_peripheral = high_risk[high_risk['is_peripheral_ward'] == 1]
    else:
        high_risk_peripheral = high_risk[high_risk['ward_id'].isin(peripheral_wards)]
    
    arrows = []
    
    for _, row in high_risk_peripheral.iterrows():
        ward_id = row['ward_id']
        probability = row['probability']
        
        # Get neighbors
        neighbors = adjacency.get(ward_id, [])
        
        # Get source centroid
        if ward_id not in centroids:
            continue
        from_coords = (centroids[ward_id]['lat'], centroids[ward_id]['lon'])
        
        for neighbor in neighbors:
            if neighbor not in centroids:
                continue
            
            to_coords = (centroids[neighbor]['lat'], centroids[neighbor]['lon'])
            
            arrows.append({
                'from_ward': ward_id,
                'to_ward': neighbor,
                'from_coords': from_coords,
                'to_coords': to_coords,
                'influence': float(probability)
            })
    
    return arrows


def compute_neighbor_risk_overlay(
    predictions_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Compute average neighbor risk for each ward.
    
    NeighborRisk_i = (1/|N(i)|) * Σ_{j∈N(i)} P_j
    
    This is for visualization only — does NOT alter base risk classification.
    
    Args:
        predictions_df: Prediction DataFrame with 'ward_id', 'probability'
    
    Returns:
        Dict mapping ward_id -> {
            'neighbor_avg_risk': float,
            'neighbor_count': int,
            'high_risk_neighbor_count': int,
            'intensity': float (0-1 for visualization)
        }
        Empty dict if data insufficient
    """
    if predictions_df is None or len(predictions_df) == 0:
        return {}
    
    adjacency = load_ward_adjacency()
    if not adjacency:
        return {}
    
    # Build probability lookup
    prob_lookup = dict(zip(predictions_df['ward_id'], predictions_df['probability']))
    risk_lookup = dict(zip(predictions_df['ward_id'], predictions_df.get('risk', ['Low'] * len(predictions_df))))
    
    result = {}
    
    for ward_id in predictions_df['ward_id'].unique():
        neighbors = adjacency.get(ward_id, [])
        
        if not neighbors:
            result[ward_id] = {
                'neighbor_avg_risk': 0.0,
                'neighbor_count': 0,
                'high_risk_neighbor_count': 0,
                'intensity': 0.0
            }
            continue
        
        neighbor_probs = [prob_lookup.get(n, 0.0) for n in neighbors if n in prob_lookup]
        neighbor_risks = [risk_lookup.get(n, 'Low') for n in neighbors if n in risk_lookup]
        
        if neighbor_probs:
            avg_risk = sum(neighbor_probs) / len(neighbor_probs)
            high_risk_count = sum(1 for r in neighbor_risks if r == 'High')
        else:
            avg_risk = 0.0
            high_risk_count = 0
        
        # Intensity for visualization (0-1 scale)
        # Higher intensity when surrounded by high-risk neighbors
        intensity = min(1.0, avg_risk * 1.5)  # Scale up slightly for visibility
        
        result[ward_id] = {
            'neighbor_avg_risk': round(avg_risk, 4),
            'neighbor_count': len(neighbors),
            'high_risk_neighbor_count': high_risk_count,
            'intensity': round(intensity, 3)
        }
    
    return result


def get_spatial_viz_summary(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary of spatial visualization data availability.
    
    Args:
        predictions_df: Prediction DataFrame
    
    Returns:
        Dict with availability flags and counts
    """
    adjacency = load_ward_adjacency()
    peripheral = load_peripheral_wards()
    centroids = load_ward_centroids()
    
    return {
        'adjacency_available': bool(adjacency),
        'adjacency_ward_count': len(adjacency),
        'peripheral_available': bool(peripheral),
        'peripheral_ward_count': len(peripheral),
        'centroids_available': bool(centroids),
        'centroid_ward_count': len(centroids),
        'all_available': all([adjacency, peripheral, centroids]),
        'predictions_count': len(predictions_df) if predictions_df is not None else 0
    }


def simulate_spatial_spread(
    pred_df: pd.DataFrame,
    threshold: float = 0.3,
    alpha: float = 0.12
) -> Optional[pd.DataFrame]:
    """Simulate synthetic spatial diffusion for visualization only.
    
    This is a SYNTHETIC SIMULATION for demonstration purposes.
    It does NOT:
    - Modify model outputs
    - Overwrite prediction probabilities
    - Modify risk classification logic
    - Change thresholds
    - Affect stored artifacts
    - Feed back into training
    - Write to any persistent file
    
    Rule:
        If a peripheral ward has high-risk prediction,
        → Increase neighbor outbreak probability by α * source_probability
    
    Mathematical model:
        P_j,t^sim = P_j,t + α * P_i,t  if i ∈ Peripheral ∧ j ∈ N(i)
        
    Where:
        α ∈ [0.10, 0.15] (default 0.12)
        Values capped at 1.0
    
    Args:
        pred_df: Original prediction DataFrame with 'ward_id', 'probability', 'risk'
        threshold: Risk threshold (for recomputing simulated risk class)
        alpha: Diffusion coefficient (0.10 to 0.15 recommended)
    
    Returns:
        Simulated DataFrame with additional columns:
            - simulated_probability: adjusted probability
            - simulated_risk: recomputed risk class based on simulation
            - spread_source: list of source wards that influenced this ward
        Returns None if data insufficient
    """
    if pred_df is None or len(pred_df) == 0:
        return None
    
    # Load spatial metadata
    adjacency = load_ward_adjacency()
    peripheral_wards = load_peripheral_wards()
    
    # Graceful degradation if metadata missing
    if not adjacency or not peripheral_wards:
        logger.warning("Simulation skipped: adjacency or peripheral data missing")
        return None
    
    # Create a COPY - do NOT modify original
    sim_df = pred_df.copy()
    
    # Initialize simulated columns
    sim_df['simulated_probability'] = sim_df['probability'].copy()
    sim_df['spread_source'] = [[] for _ in range(len(sim_df))]
    
    # Build lookup for ward_id -> row index
    ward_to_idx = {row['ward_id']: idx for idx, row in sim_df.iterrows()}
    
    # Identify high-risk peripheral wards (sources of spread)
    high_risk_peripheral = sim_df[
        (sim_df['risk'] == 'High') & 
        (sim_df['ward_id'].isin(peripheral_wards))
    ]
    
    # Apply diffusion from each high-risk peripheral ward
    for _, source_row in high_risk_peripheral.iterrows():
        source_ward = source_row['ward_id']
        source_prob = source_row['probability']
        
        # Get neighbors
        neighbors = adjacency.get(source_ward, [])
        
        # Increase neighbor probabilities
        for neighbor in neighbors:
            if neighbor in ward_to_idx:
                neighbor_idx = ward_to_idx[neighbor]
                
                # Apply diffusion: P_j += α * P_i
                spread_amount = alpha * source_prob
                sim_df.at[neighbor_idx, 'simulated_probability'] += spread_amount
                
                # Track spread source
                sim_df.at[neighbor_idx, 'spread_source'].append(source_ward)
    
    # Cap probabilities at 1.0
    sim_df['simulated_probability'] = sim_df['simulated_probability'].clip(upper=1.0)
    
    # Compute simulated risk class (for visualization only)
    # This does NOT replace the original 'risk' column
    def classify_simulated_risk(prob, thresh):
        if prob >= thresh:
            return 'High'
        elif prob >= thresh * 0.5:  # Moderate cutoff at 50% of threshold
            return 'Moderate'
        return 'Low'
    
    sim_df['simulated_risk'] = sim_df['simulated_probability'].apply(
        lambda p: classify_simulated_risk(p, threshold)
    )
    
    # Compute spread delta for visualization
    sim_df['spread_delta'] = sim_df['simulated_probability'] - sim_df['probability']
    
    return sim_df


def get_spread_simulation_summary(sim_df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for spatial spread simulation.
    
    Args:
        sim_df: DataFrame from simulate_spatial_spread()
    
    Returns:
        Dict with simulation statistics
    """
    if sim_df is None or 'simulated_probability' not in sim_df.columns:
        return {'available': False}
    
    affected_wards = sim_df[sim_df['spread_delta'] > 0.001]
    new_high_risk = sim_df[
        (sim_df['simulated_risk'] == 'High') & 
        (sim_df['risk'] != 'High')
    ]
    
    return {
        'available': True,
        'total_wards': len(sim_df),
        'affected_wards_count': len(affected_wards),
        'new_high_risk_count': len(new_high_risk),
        'max_spread_delta': round(sim_df['spread_delta'].max(), 4),
        'avg_spread_delta': round(affected_wards['spread_delta'].mean(), 4) if len(affected_wards) > 0 else 0,
        'spread_sources': list(set(
            src for sources in sim_df['spread_source'] for src in sources
        )),
    }
