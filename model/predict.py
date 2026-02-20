"""Inference module for ward-level outbreak predictions."""

import logging
import os
import warnings
from datetime import timedelta

import numpy as np

import joblib
import pandas as pd

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = UserWarning

from utils.constants import GLOBAL_THRESHOLD
from utils.feature_engineering import engineer_outbreak_features, validate_feature_schema
from utils.risk_logic import classify_risk
from utils.runtime_config import load_runtime_config


logger = logging.getLogger(__name__)


class OutbreakPredictor:
    """Load the persisted model and run inference on latest ward records."""

    def __init__(self, model_path=None):
        self.model = None
        self.feature_columns = None
        self.metrics = {}
        self.best_threshold = float(GLOBAL_THRESHOLD)
        self.calibration = {'method': 'none'}
        runtime_cfg = load_runtime_config()
        configured_model_path = runtime_cfg.get('model_artifact_path')
        configured_data_path = runtime_cfg.get('data_path')
        if not configured_model_path:
            raise KeyError("Runtime config missing required key: model_artifact_path")
        if not configured_data_path:
            raise KeyError("Runtime config missing required key: data_path")
        self.default_data_path = str(configured_data_path)
        self.model_path = str(model_path) if model_path is not None else str(configured_model_path)
        self.model_metadata = {}
        self.is_loaded = False
        self.latest_engineered = None

    @property
    def project_root(self):
        return os.path.dirname(os.path.dirname(__file__))

    def _resolve_path(self, relative_path):
        return os.path.join(self.project_root, relative_path)

    def load_model(self):
        full_model_path = self._resolve_path(self.model_path)
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model not found at {full_model_path}")

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            model_data = joblib.load(full_model_path)

        version_mismatch_detected = any(
            isinstance(record.message, InconsistentVersionWarning)
            for record in captured_warnings
        )
        if version_mismatch_detected:
            logger.warning(
                "Detected sklearn model persistence version mismatch while loading artifact %s. "
                "Persisted isotonic calibrator will be disabled for runtime safety.",
                full_model_path,
            )

        self.model = model_data['model']
        self.feature_columns = model_data.get('feature_columns', model_data.get('feature_list'))
        if self.feature_columns is None:
            raise ValueError('Model artifact missing feature_columns/feature_list')

        self.metrics = model_data.get('metrics', {})
        artifact_threshold = model_data.get('global_threshold', self.metrics.get('global_threshold'))
        if artifact_threshold is None:
            artifact_threshold = model_data.get('best_threshold', self.metrics.get('best_threshold'))
        if artifact_threshold is not None and not np.isclose(float(artifact_threshold), float(GLOBAL_THRESHOLD), atol=1e-9):
            raise RuntimeError(
                f"Threshold policy drift detected in artifact: artifact={float(artifact_threshold):.6f}, "
                f"expected={float(GLOBAL_THRESHOLD):.6f}."
            )
        self.best_threshold = float(GLOBAL_THRESHOLD)
        self.calibration = model_data.get('calibration', self.metrics.get('calibration', {'method': 'none'}))
        if version_mismatch_detected and isinstance(self.calibration, dict) and self.calibration.get('method') == 'isotonic':
            self.calibration = {'method': 'none', 'reason': 'sklearn-version-mismatch'}
        if (
            isinstance(self.calibration, dict)
            and self.calibration.get('method') == 'isotonic'
            and self.calibration.get('calibrator') is None
        ):
            self.calibration = {'method': 'none', 'reason': 'missing-isotonic-calibrator'}

        cv_metrics = model_data.get('cv_metrics', self.metrics.get('cv_metrics', {}))
        training_data_range = model_data.get('training_data_range', self.metrics.get('training_data_range', {}))
        trained_at = model_data.get(
            'generated_at_utc',
            model_data.get(
                'trained_at_utc',
                self.metrics.get('trained_at_utc', pd.Timestamp(os.path.getmtime(full_model_path), unit='s', tz='UTC').isoformat()),
            ),
        )
        calibration_meta = (
            {k: v for k, v in self.calibration.items() if k != 'calibrator'}
            if isinstance(self.calibration, dict)
            else {'method': 'none'}
        )
        if not calibration_meta:
            calibration_meta = {'method': 'none'}

        if calibration_meta.get('method') == 'none':
            calibration_source = calibration_meta.get('reason', 'runtime')
        else:
            calibration_source = self.metrics.get(
                'calibration_source',
                model_data.get('calibration', {}).get('fitted_on', calibration_meta.get('fitted_on', 'N/A')),
            )

        self.model_metadata = {
            'artifact_version': model_data.get('artifact_version', self.metrics.get('model_artifact_version', 'unknown')),
            'trained_at_utc': trained_at,
            'training_data_range': training_data_range,
            'training_range': training_data_range,
            'validation_metrics': self.metrics.get('validation_metrics', {}),
            'test_metrics': self.metrics.get('test_metrics', self.metrics),
            'best_threshold': self.best_threshold,
            'threshold': float(GLOBAL_THRESHOLD),
            'calibration': calibration_meta if calibration_meta else {'method': 'none'},
            'leakage_checks': self.metrics.get('leakage_checks', {}),
            'outbreak_ratios': self.metrics.get('outbreak_ratios', {}),
            'distribution_shift': self.metrics.get('distribution_shift', {}),
            'cv_metrics': cv_metrics,
            'n_folds': self.metrics.get('n_folds', len(cv_metrics.get('fold_metrics', [])) if isinstance(cv_metrics, dict) else None),
            'global_threshold': float(GLOBAL_THRESHOLD),
            'n_samples': self.metrics.get('n_samples', model_data.get('n_samples', 'N/A')),
            'n_features': self.metrics.get('n_features', len(self.feature_columns) if self.feature_columns is not None else 'N/A'),
            'retraining_frequency_days': model_data.get('retraining_frequency_days', self.metrics.get('retraining_frequency_days')),
            'best_params': model_data.get('best_params', self.metrics.get('best_params', {})),
            'calibration_source': calibration_source,
        }

        required_keys = [
            'threshold',
            'calibration',
            'training_range',
            'cv_metrics',
            'artifact_version',
        ]
        for key in required_keys:
            if key not in self.model_metadata:
                raise RuntimeError(f"Artifact metadata missing required key: {key}")

        self.is_loaded = True
        return self.metrics

    def load_latest_data(self, data_path=None):
        effective_data_path = self.default_data_path if data_path is None else str(data_path)
        full_path = self._resolve_path(effective_data_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found: {full_path}")
        return pd.read_csv(full_path)

    def _ensure_loaded(self):
        if not self.is_loaded:
            self.load_model()

    def _prepare_latest_rows(self, df):
        engineered = engineer_outbreak_features(df, dropna_lag_rows=True, require_target=False)
        latest = (
            engineered.sort_values('week_start_date')
            .groupby('ward_id', as_index=False)
            .tail(1)
            .copy()
        )
        return latest

    def get_effective_threshold(self, threshold_override=None):
        del threshold_override
        return float(GLOBAL_THRESHOLD)

    def get_model_metadata(self):
        self._ensure_loaded()
        return self.model_metadata

    def _apply_probability_adjustment(self, probabilities):
        calibration = self.calibration or {'method': 'none'}
        method = calibration.get('method', 'none')

        if method == 'isotonic' and calibration.get('calibrator') is not None:
            calibrated = calibration['calibrator'].transform(probabilities)
            return calibrated.clip(0.0, 1.0)

        if method == 'smoothing':
            alpha = float(calibration.get('alpha', 0.1))
            target_rate = float(calibration.get('target_rate', 0.0))
            smoothed = ((1.0 - alpha) * probabilities) + (alpha * target_rate)
            return smoothed.clip(0.0, 1.0)

        return probabilities

    def predict_latest_week(self, df=None, data_path=None, threshold_override=None):
        self._ensure_loaded()
        threshold = self.get_effective_threshold(threshold_override=threshold_override)
        source_df = df if df is not None else self.load_latest_data(data_path)
        latest_df = self._prepare_latest_rows(source_df)
        validate_feature_schema(latest_df, self.feature_columns)
        self.latest_engineered = latest_df

        probabilities = self.model.predict_proba(latest_df[self.feature_columns])[:, 1]
        probabilities = self._apply_probability_adjustment(probabilities)

        result = pd.DataFrame(
            {
                'week_start_date': latest_df['week_start_date'].values,
                'ward_id': latest_df['ward_id'].values,
                'probability': probabilities,
            }
        )
        result['risk_level'] = result['probability'].apply(lambda probability: classify_risk(probability, threshold=threshold))
        result['prediction'] = (result['probability'] >= threshold).astype(int)
        result['risk_score'] = (result['probability'] * 100).round(2)
        result = result.sort_values('ward_id').reset_index(drop=True)
        return result

    def predict_time_series(self, df=None, data_path=None, threshold_override=None):
        self._ensure_loaded()
        threshold = self.get_effective_threshold(threshold_override=threshold_override)
        source_df = df if df is not None else self.load_latest_data(data_path)
        engineered = engineer_outbreak_features(source_df, dropna_lag_rows=True, require_target=False)
        validate_feature_schema(engineered, self.feature_columns)
        probabilities = self.model.predict_proba(engineered[self.feature_columns])[:, 1]
        probabilities = self._apply_probability_adjustment(probabilities)

        result = engineered[['week_start_date', 'ward_id', 'reported_cases', 'rainfall_mm']].copy()
        result['probability'] = probabilities
        result['prediction'] = (result['probability'] >= threshold).astype(int)
        result['risk_level'] = result['probability'].apply(lambda probability: classify_risk(probability, threshold=threshold))
        return result.sort_values(['ward_id', 'week_start_date']).reset_index(drop=True)

    def predict_forward_projection(self, df=None, data_path=None, horizon_weeks=2, threshold_override=None):
        self._ensure_loaded()
        threshold = self.get_effective_threshold(threshold_override=threshold_override)
        source_df = (df if df is not None else self.load_latest_data(data_path)).copy()

        latest_input = self._prepare_latest_rows(source_df)
        validate_feature_schema(latest_input, self.feature_columns)

        forward_outputs = []
        latest_week = pd.to_datetime(source_df['week_start_date']).max()
        working_df = source_df.copy()

        for step in range(1, int(horizon_weeks) + 1):
            current_latest = self._prepare_latest_rows(working_df)
            probabilities = self.model.predict_proba(current_latest[self.feature_columns])[:, 1]
            probabilities = self._apply_probability_adjustment(probabilities)

            projection_week = latest_week + timedelta(days=7 * step)
            predicted_cases = np.rint(np.clip(probabilities * current_latest['reported_cases'].to_numpy() * 1.15, 0, None)).astype(int)

            step_df = pd.DataFrame(
                {
                    'projection_week_start_date': projection_week,
                    'forecast_horizon_week': step,
                    'ward_id': current_latest['ward_id'].values,
                    'probability': probabilities,
                    'prediction': (probabilities >= threshold).astype(int),
                    'risk_level': [classify_risk(value, threshold=threshold) for value in probabilities],
                }
            )
            forward_outputs.append(step_df)

            synthetic_rows = current_latest[['ward_id', 'rainfall_mm', 'turbidity', 'ecoli_index', 'reported_cases']].copy()
            synthetic_rows['week_start_date'] = projection_week
            synthetic_rows['reported_cases'] = predicted_cases
            if 'outbreak_next_week' in working_df.columns:
                synthetic_rows['outbreak_next_week'] = (probabilities >= threshold).astype(int)
            working_df = pd.concat([working_df, synthetic_rows], ignore_index=True, sort=False)

        return pd.concat(forward_outputs, ignore_index=True).sort_values(['forecast_horizon_week', 'ward_id']).reset_index(drop=True)

    # Backward compatibility wrappers
    def predict_proba(self, df):
        self._ensure_loaded()
        latest_df = self._prepare_latest_rows(df)
        return self.model.predict_proba(latest_df[self.feature_columns])[:, 1]

    def predict_with_metadata(self, df):
        return self.predict_latest_week(df=df)

    def batch_predict(self, df, ward_col='ward_id'):
        del ward_col
        return self.predict_latest_week(df=df)
