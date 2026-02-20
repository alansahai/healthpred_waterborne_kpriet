"""Feature engineering pipeline for ward-level outbreak prediction."""

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


def get_model_feature_columns() -> list:
    """Return feature list used consistently for training and inference."""
    return [
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
