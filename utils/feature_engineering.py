"""Feature engineering pipeline for ward-level outbreak prediction."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .constants import (
    ECOLI_COL,
    RAINFALL_COL,
    REPORTED_CASES_COL,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COL,
    TURBIDITY_COL,
    WARD_ID_COL,
    WEEK_START_DATE_COL,
)


REQUIRED_COLUMNS = REQUIRED_INPUT_COLUMNS

logger = logging.getLogger(__name__)

# Default path for adjacency map (relative to project root)
_DEFAULT_ADJACENCY_PATH = "data/ward_adjacency.json"
_DEFAULT_PERIPHERAL_PATH = "data/peripheral_wards.json"
_DEFAULT_SPATIAL_METADATA_PATH = "spatial_metadata.json"


def load_adjacency_map(path: str = None) -> Dict[str, List[str]]:
    """
    Load ward adjacency map from JSON file.
    
    Args:
        path: Path to adjacency JSON file. If None, uses default path.
        
    Returns:
        Dictionary mapping ward_id to list of neighbor ward_ids.
        Returns empty dict if file is missing (graceful degradation).
        
    Note:
        This function never raises exceptions - it returns empty dict on any error
        to ensure pipeline continues without spatial features.
    """
    if path is None:
        # Try to find file relative to project structure
        # Check multiple possible locations
        possible_paths = [
            Path(_DEFAULT_ADJACENCY_PATH),
            Path(__file__).parent.parent / _DEFAULT_ADJACENCY_PATH,
        ]
        adjacency_path = None
        for p in possible_paths:
            if p.exists():
                adjacency_path = p
                break
        if adjacency_path is None:
            logger.info("Adjacency file not found - skipping spatial features")
            return {}
    else:
        adjacency_path = Path(path)
        if not adjacency_path.exists():
            logger.info(f"Adjacency file not found at {path} - skipping spatial features")
            return {}
    
    try:
        with open(adjacency_path, 'r', encoding='utf-8') as f:
            adjacency = json.load(f)
        logger.info(f"Loaded adjacency map with {len(adjacency)} areas")
        return adjacency
    except Exception as e:
        logger.warning(f"Failed to load adjacency map: {e} - skipping spatial features")
        return {}


def load_peripheral_metadata(path: str = None) -> set:
    """
    Load peripheral ward metadata from JSON file.
    
    Args:
        path: Path to peripheral wards JSON file. If None, uses default path.
        
    Returns:
        Set of peripheral ward_ids.
        Returns empty set if file is missing (graceful degradation).
        
    Note:
        This function never raises exceptions - it returns empty set on any error
        to ensure pipeline continues without peripheral features.
    """
    if path is None:
        possible_paths = [
            Path(_DEFAULT_PERIPHERAL_PATH),
            Path(__file__).parent.parent / _DEFAULT_PERIPHERAL_PATH,
        ]
        peripheral_path = None
        for p in possible_paths:
            if p.exists():
                peripheral_path = p
                break
        if peripheral_path is None:
            logger.info("Peripheral wards file not found - skipping is_peripheral_ward feature")
            return set()
    else:
        peripheral_path = Path(path)
        if not peripheral_path.exists():
            logger.info(f"Peripheral wards file not found at {path} - skipping feature")
            return set()
    
    try:
        with open(peripheral_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        peripheral_list = data.get('peripheral_wards', [])
        logger.info(f"Loaded {len(peripheral_list)} peripheral wards")
        return set(str(w) for w in peripheral_list)
    except Exception as e:
        logger.warning(f"Failed to load peripheral metadata: {e} - skipping feature")
        return set()


def load_spatial_centroids(path: str = None) -> Dict[str, Dict[str, float]]:
    """
    Load ward centroids from spatial metadata JSON file.
    
    Args:
        path: Path to spatial metadata JSON file. If None, uses default path.
        
    Returns:
        Dictionary mapping ward_id to {"lat": float, "lon": float}.
        Returns empty dict if file is missing (graceful degradation).
        
    Note:
        This function never raises exceptions - it returns empty dict on any error
        to ensure pipeline continues without centroid-based features.
    """
    if path is None:
        possible_paths = [
            Path(_DEFAULT_SPATIAL_METADATA_PATH),
            Path(__file__).parent.parent / _DEFAULT_SPATIAL_METADATA_PATH,
        ]
        metadata_path = None
        for p in possible_paths:
            if p.exists():
                metadata_path = p
                break
        if metadata_path is None:
            logger.info("Spatial metadata file not found - skipping centroid-based features")
            return {}
    else:
        metadata_path = Path(path)
        if not metadata_path.exists():
            logger.info(f"Spatial metadata file not found at {path} - skipping features")
            return {}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        centroids = data.get('centroids', {})
        logger.info(f"Loaded centroids for {len(centroids)} areas")
        return centroids
    except Exception as e:
        logger.warning(f"Failed to load spatial centroids: {e} - skipping features")
        return {}


def add_static_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add static geographic exposure feature based on ward location.
    
    Features added (when data available):
    - is_peripheral_ward (0/1): Whether ward touches district boundary
    
    Args:
        df: DataFrame with ward-level data (must have 'ward_id' column).
        
    Returns:
        DataFrame with is_peripheral_ward feature added.
        Skips feature silently if peripheral_wards.json is missing.
        
    Note:
        This feature is time-independent and based solely on geography.
        No temporal leakage possible.
    """
    data = df.copy()
    
    # ===== IS_PERIPHERAL_WARD =====
    peripheral_set = load_peripheral_metadata()
    if peripheral_set:
        data['is_peripheral_ward'] = data[WARD_ID_COL].apply(
            lambda w: 1 if str(w) in peripheral_set else 0
        )
        logger.info("Added is_peripheral_ward feature")
    
    return data


def add_neighbor_lag_features(
    df: pd.DataFrame, 
    adjacency: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Add neighbor-based lagged features for spatial correlation modeling.
    
    Computes average of neighbors' T-1 values (no leakage):
    - neighbor_avg_cases_last_week
    - neighbor_outbreak_rate_last_week
    
    Args:
        df: DataFrame with ward-level features (must have lag columns already).
        adjacency: Dictionary mapping ward_id to list of neighbor ward_ids.
        
    Returns:
        DataFrame with additional neighbor features added.
        Wards with no neighbors get NaN for neighbor features.
        
    Note:
        Uses only lagged (T-1) values to prevent data leakage.
        Does not modify rows or drop any data.
    """
    if not adjacency:
        return df
    
    data = df.copy()
    
    # Ensure sorted by ward and date for consistent lag alignment
    data = data.sort_values([WARD_ID_COL, WEEK_START_DATE_COL]).reset_index(drop=True)
    
    # Ensure lag columns exist for neighbor lookup
    grouped = data.groupby(WARD_ID_COL, group_keys=False)
    if 'cases_last_week' not in data.columns:
        data['cases_last_week'] = grouped[REPORTED_CASES_COL].shift(1)
    
    # Also need outbreak lag if target exists
    if TARGET_COL in data.columns and 'outbreak_last_week' not in data.columns:
        data['outbreak_last_week'] = grouped[TARGET_COL].shift(1)
    
    # Build lookup: (ward_id, week_start_date) -> lag values
    lag_lookup_cols = ['cases_last_week']
    if 'outbreak_last_week' in data.columns:
        lag_lookup_cols.append('outbreak_last_week')
    
    # Create indexed lookup for fast neighbor queries
    lookup_df = data[[WARD_ID_COL, WEEK_START_DATE_COL] + lag_lookup_cols].copy()
    lookup_df = lookup_df.set_index([WARD_ID_COL, WEEK_START_DATE_COL])
    
    # Initialize neighbor feature columns (exactly 2)
    data['neighbor_avg_cases_last_week'] = np.nan
    data['neighbor_outbreak_rate_last_week'] = np.nan
    
    # Get unique weeks for batch processing
    unique_weeks = data[WEEK_START_DATE_COL].unique()
    
    # Process by week for efficiency
    for week in unique_weeks:
        week_mask = data[WEEK_START_DATE_COL] == week
        week_indices = data[week_mask].index
        
        for idx in week_indices:
            ward_id = data.loc[idx, WARD_ID_COL]
            neighbors = adjacency.get(str(ward_id), [])
            
            if not neighbors:
                # Ward has no neighbors - leave as NaN
                continue
            
            # Collect neighbor lag-1 values for this week
            neighbor_cases = []
            neighbor_outbreak = []
            
            for neighbor_id in neighbors:
                try:
                    neighbor_data = lookup_df.loc[(str(neighbor_id), week)]
                    
                    if pd.notna(neighbor_data['cases_last_week']):
                        neighbor_cases.append(neighbor_data['cases_last_week'])
                    if 'outbreak_last_week' in neighbor_data.index and pd.notna(neighbor_data['outbreak_last_week']):
                        neighbor_outbreak.append(neighbor_data['outbreak_last_week'])
                except KeyError:
                    # Neighbor not found for this week - skip
                    continue
            
            # Compute averages (mean of neighbors' t-1 values)
            if neighbor_cases:
                data.loc[idx, 'neighbor_avg_cases_last_week'] = np.mean(neighbor_cases)
            if neighbor_outbreak:
                data.loc[idx, 'neighbor_outbreak_rate_last_week'] = np.mean(neighbor_outbreak)
    
    return data


def validate_feature_schema(dataframe: pd.DataFrame, feature_columns: list) -> None:
    missing_features = [column for column in feature_columns if column not in dataframe.columns]
    if missing_features:
        raise ValueError(f"Missing engineered feature columns for inference: {missing_features}")


def prepare_outbreak_data(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Clean and standardize outbreak input data before feature engineering."""
    data = df.copy()
    required_columns = REQUIRED_COLUMNS if require_target else [
        WEEK_START_DATE_COL,
        WARD_ID_COL,
        RAINFALL_COL,
        TURBIDITY_COL,
        ECOLI_COL,
        REPORTED_CASES_COL,
    ]
    missing_cols = [column for column in required_columns if column not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data[WEEK_START_DATE_COL] = pd.to_datetime(data[WEEK_START_DATE_COL], errors='coerce')
    data = data.dropna(subset=[WEEK_START_DATE_COL, WARD_ID_COL])
    data[WARD_ID_COL] = data[WARD_ID_COL].astype(str).str.strip()

    invalid_ward_mask = ~data[WARD_ID_COL].str.match(r'^W(\d{2}|100)$')
    if invalid_ward_mask.any():
        invalid_values = sorted(data.loc[invalid_ward_mask, WARD_ID_COL].unique().tolist())
        preview = invalid_values[:5]
        raise ValueError(f"Invalid ward_id values detected: {preview}")

    duplicate_mask = data.duplicated(subset=[WARD_ID_COL, WEEK_START_DATE_COL], keep=False)
    if duplicate_mask.any():
        duplicate_rows = data.loc[duplicate_mask, [WARD_ID_COL, WEEK_START_DATE_COL]].head(10)
        raise ValueError(
            "Duplicate (ward_id, week_start_date) rows detected before feature engineering: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    numeric_columns = [RAINFALL_COL, TURBIDITY_COL, ECOLI_COL, REPORTED_CASES_COL]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data[column] = data.groupby(WARD_ID_COL)[column].transform(lambda values: values.fillna(values.median()))
        data[column] = data[column].fillna(data[column].median())

    if require_target:
        data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors='coerce').fillna(0)
        data[TARGET_COL] = (data[TARGET_COL] > 0).astype(int)

    data = data.sort_values([WARD_ID_COL, WEEK_START_DATE_COL]).reset_index(drop=True)
    return data


def engineer_outbreak_features(
    df: pd.DataFrame,
    dropna_lag_rows: bool = True,
    require_target: bool = True,
) -> pd.DataFrame:
    """Create leakage-safe lag, rolling, growth and interaction features per ward."""
    data = prepare_outbreak_data(df, require_target=require_target)
    grouped = data.groupby(WARD_ID_COL, group_keys=False)

    # ===== LAG FEATURES (1-week, 2-week) =====
    data['cases_last_week'] = grouped[REPORTED_CASES_COL].shift(1)
    data['cases_last_2_weeks'] = grouped[REPORTED_CASES_COL].shift(2)
    data['rainfall_last_week'] = grouped[RAINFALL_COL].shift(1)
    data['turbidity_last_week'] = grouped[TURBIDITY_COL].shift(1)
    data['ecoli_last_week'] = grouped[ECOLI_COL].shift(1)

    # ===== ROLLING AVERAGES (2-week and 3-week) =====
    data['rainfall_2w_avg'] = grouped[RAINFALL_COL].transform(
        lambda values: values.shift(1).rolling(window=2, min_periods=2).mean()
    )
    data['rainfall_3w_avg'] = grouped[RAINFALL_COL].transform(
        lambda values: values.shift(1).rolling(window=3, min_periods=3).mean()
    )
    data['cases_2w_avg'] = grouped[REPORTED_CASES_COL].transform(
        lambda values: values.shift(1).rolling(window=2, min_periods=2).mean()
    )
    data['cases_3w_avg'] = grouped[REPORTED_CASES_COL].transform(
        lambda values: values.shift(1).rolling(window=3, min_periods=3).mean()
    )
    data['ecoli_2w_avg'] = grouped[ECOLI_COL].transform(
        lambda values: values.shift(1).rolling(window=2, min_periods=2).mean()
    )
    data['turbidity_2w_avg'] = grouped[TURBIDITY_COL].transform(
        lambda values: values.shift(1).rolling(window=2, min_periods=2).mean()
    )

    # ===== GROWTH RATES (acceleration) =====
    # Case growth rate: (current - last week) / last week
    previous_cases = data['cases_last_week'].replace(0, np.nan)
    data['case_growth_rate'] = (data[REPORTED_CASES_COL] - data['cases_last_week']) / previous_cases
    data['case_growth_rate'] = data['case_growth_rate'].replace([np.inf, -np.inf], np.nan)

    # Rainfall acceleration: (current - last week) / (last week + 1 to avoid division by zero)
    rainfall_accel = (data[RAINFALL_COL] - data['rainfall_last_week']) / (data['rainfall_last_week'] + 1.0)
    data['rainfall_acceleration'] = rainfall_accel.replace([np.inf, -np.inf], np.nan)

    # Contamination growth rate: change in ecoli levels
    previous_ecoli = data['ecoli_last_week'].replace(0, np.nan)
    data['contamination_growth_rate'] = (data[ECOLI_COL] - data['ecoli_last_week']) / (previous_ecoli + 0.1)
    data['contamination_growth_rate'] = data['contamination_growth_rate'].replace([np.inf, -np.inf], np.nan)

    # Turbidity growth rate
    previous_turbidity = data['turbidity_last_week'].replace(0, np.nan)
    data['turbidity_growth_rate'] = (data[TURBIDITY_COL] - data['turbidity_last_week']) / (previous_turbidity + 0.1)
    data['turbidity_growth_rate'] = data['turbidity_growth_rate'].replace([np.inf, -np.inf], np.nan)

    # ===== INTERACTION FEATURES =====
    data['rainfall_ecoli_interaction'] = data[RAINFALL_COL] * data[ECOLI_COL]
    data['rainfall_2w_avg_turbidity_interaction'] = data['rainfall_2w_avg'] * data[TURBIDITY_COL]
    data['rainfall_3w_avg_ecoli_interaction'] = data['rainfall_3w_avg'] * data[ECOLI_COL]

    # ===== PAST OUTBREAK INDICATOR =====
    # Flag if outbreak occurred in previous week
    data['outbreak_last_week'] = grouped[TARGET_COL].shift(1) if require_target else 0
    # Create a rolling count of outbreaks in past 2 weeks
    if require_target:
        data['outbreak_count_2w'] = grouped[TARGET_COL].transform(
            lambda values: values.shift(1).rolling(window=2, min_periods=1).sum()
        )
    else:
        data['outbreak_count_2w'] = 0

    # ===== TEMPORAL FEATURES =====
    data['month'] = data[WEEK_START_DATE_COL].dt.month
    data['iso_week'] = data[WEEK_START_DATE_COL].dt.isocalendar().week.astype(int)
    data['monsoon_flag'] = data['month'].isin([6, 7, 8, 9, 10, 11]).astype(int)

    # ===== SPATIAL LAG FEATURES (Optional) =====
    # Only added if adjacency file is available
    adjacency = load_adjacency_map()
    if adjacency:
        data = add_neighbor_lag_features(data, adjacency)
        logger.info("Added spatial neighbor lag features")

    # ===== STATIC SPATIAL FEATURES (Optional) =====
    # Add peripheral ward flag and distance-based features
    data = add_static_spatial_features(data)

    if dropna_lag_rows:
        required_lag_cols = [
            'cases_last_week',
            'cases_last_2_weeks',
            'rainfall_last_week',
            'turbidity_last_week',
            'ecoli_last_week',
            'rainfall_2w_avg',
            'rainfall_3w_avg',
            'cases_2w_avg',
            'cases_3w_avg',
            'ecoli_2w_avg',
            'turbidity_2w_avg',
            'case_growth_rate',
            'rainfall_acceleration',
            'contamination_growth_rate',
            'turbidity_growth_rate',
        ]
        data = data.dropna(subset=required_lag_cols).reset_index(drop=True)
    else:
        data = data.fillna(0)

    return data


def get_spatial_feature_columns() -> list:
    """Return spatial feature columns for experimental training.
    
    These are optionally included when include_spatial_features=True.
    Returns 2 lagged neighbor features + 1 static peripheral flag.
    """
    return [
        'neighbor_avg_cases_last_week',
        'neighbor_outbreak_rate_last_week',
        'is_peripheral_ward',
    ]


def get_model_feature_columns(include_spatial: bool = False) -> list:
    """Return feature list used consistently for training and inference.
    
    Args:
        include_spatial: If True, include spatial features (experimental).
    """
    base_features = [
        # Raw environmental features
        'rainfall_mm',
        'turbidity',
        'ecoli_index',
        'reported_cases',
        # Lag features (1-week, 2-week)
        'cases_last_week',
        'cases_last_2_weeks',
        'rainfall_last_week',
        'turbidity_last_week',
        'ecoli_last_week',
        # Rolling averages (2-week and 3-week)
        'rainfall_2w_avg',
        'rainfall_3w_avg',
        'cases_2w_avg',
        'cases_3w_avg',
        'ecoli_2w_avg',
        'turbidity_2w_avg',
        # Growth rates and acceleration
        'case_growth_rate',
        'rainfall_acceleration',
        'contamination_growth_rate',
        'turbidity_growth_rate',
        # Interaction features
        'rainfall_ecoli_interaction',
        'rainfall_2w_avg_turbidity_interaction',
        'rainfall_3w_avg_ecoli_interaction',
        # Temporal features
        'month',
        'iso_week',
        'monsoon_flag',
    ]
    
    if include_spatial:
        base_features.extend(get_spatial_feature_columns())
    
    return base_features


# Backward-compatible exports used by earlier app paths
def engineer_all_features(df, target_col='reported_cases'):
    del target_col
    return engineer_outbreak_features(df)


def create_lag_features(df, target_col='reported_cases', lags=None, group_by='ward_id'):
    lags = lags or [1, 2]
    data = df.copy()
    for lag in lags:
        data[f'{target_col}_lag_{lag}'] = data.groupby(group_by)[target_col].shift(lag)
    return data


def create_rolling_features(df, target_col='reported_cases', windows=None, group_by='ward_id'):
    windows = windows or [2]
    data = df.copy()
    for window in windows:
        data[f'{target_col}_rolling_mean_{window}'] = (
            data.groupby(group_by)[target_col].shift(1).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
    return data
