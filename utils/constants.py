"""Shared constants for schema, split strategy, thresholding and analytics."""

WEEK_START_DATE_COL = 'week_start_date'
WARD_ID_COL = 'ward_id'
RAINFALL_COL = 'rainfall_mm'
TURBIDITY_COL = 'turbidity'
ECOLI_COL = 'ecoli_index'
REPORTED_CASES_COL = 'reported_cases'
TARGET_COL = 'outbreak_next_week'

REQUIRED_INPUT_COLUMNS = [
    WEEK_START_DATE_COL,
    WARD_ID_COL,
    RAINFALL_COL,
    TURBIDITY_COL,
    ECOLI_COL,
    REPORTED_CASES_COL,
    TARGET_COL,
]

TRAIN_SPLIT_RATIO = 7 / 10
VALIDATION_SPLIT_RATIO = 85 / 100

THRESHOLD_FALLBACK = 1 / 2
MODERATE_RISK_RATIO = 3 / 5

GLOBAL_THRESHOLD = 23 / 100
ALERT_NORMAL_MAX = 1 / 5
ALERT_ELEVATED_MAX = 7 / 20
FORECAST_HORIZON_DAYS = 7

SELECTION_RECALL_WEIGHT = 7 / 10
SELECTION_F1_WEIGHT = 3 / 10

SCATTER_POINT_OPACITY = 3 / 5
INSIGHT_CORRELATION_CUTOFF = 3 / 10

ALERT_LOAD_NORMAL_MAX = 1 / 5
ALERT_LOAD_ELEVATED_MAX = 7 / 20

CONTRIBUTING_FACTOR_THRESHOLDS = {
    'ecoli_index': 4.0,
    'turbidity': 5.0,
    'rainfall_mm': 80.0,
    'reported_cases': 15.0,
    'monsoon_flag': 1.0,
}

INTERVENTION_SIMULATION_REDUCTION_PCT = {
    'ecoli_index': -40,
    'rainfall_mm': -50,
    'reported_cases': -30,
    'turbidity': -25,
}

OPERATIONAL_MODE_LABEL = 'Operational Mode'
ARTIFACT_VERSION_LABEL = '3.0-final'