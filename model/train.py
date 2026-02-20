"""Training module for weekly outbreak prediction using XGBoost."""

import os
from datetime import datetime, timezone
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from utils.feature_engineering import (
    engineer_outbreak_features,
    get_model_feature_columns,
    prepare_outbreak_data,
    validate_feature_schema,
)
from utils.runtime_config import load_runtime_config
from utils.constants import (
    SELECTION_F1_WEIGHT,
    SELECTION_RECALL_WEIGHT,
    THRESHOLD_FALLBACK,
)

try:
    shap = import_module('shap')
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


class OutbreakModelTrainer:
    """Train and evaluate an XGBoost classifier for outbreak_next_week."""

    def __init__(self, model_path='model/outbreak_model.pkl'):
        self.model = None
        self.model_path = model_path
        self.feature_columns = get_model_feature_columns()
        self.metrics = {}
        self.best_params = {}
        self.best_threshold = THRESHOLD_FALLBACK
        self.calibration = {'method': 'none'}

    @property
    def project_root(self):
        return os.path.dirname(os.path.dirname(__file__))

    def _resolve_path(self, relative_path):
        return os.path.join(self.project_root, relative_path)

    def load_data(self, data_path='data/coimbatore_weekly_water_disease_2024.csv'):
        full_path = self._resolve_path(data_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found: {full_path}")
        return pd.read_csv(full_path)

    def prepare_data(self, df, target_col='outbreak_next_week'):
        engineered = engineer_outbreak_features(df, dropna_lag_rows=True)
        if target_col not in engineered.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        engineered = engineered.sort_values(['week_start_date', 'ward_id']).reset_index(drop=True)
        unique_dates = np.array(sorted(pd.to_datetime(engineered['week_start_date']).unique()))

        if len(unique_dates) < 8:
            raise ValueError('Insufficient temporal coverage for rolling TimeSeriesSplit cross-validation.')

        return {
            'engineered': engineered,
            'X': engineered[self.feature_columns],
            'y': engineered[target_col].astype(int),
            'unique_dates': unique_dates,
        }

    def _build_time_folds(self, engineered_df, n_splits=5):
        unique_dates = np.array(sorted(pd.to_datetime(engineered_df['week_start_date']).unique()))
        if len(unique_dates) < 3:
            raise ValueError('At least 3 unique weeks are required for rolling cross-validation.')

        effective_splits = min(int(n_splits), len(unique_dates) - 1)
        if effective_splits < 2:
            raise ValueError('Not enough unique weeks to create at least 2 rolling CV folds.')

        tscv = TimeSeriesSplit(n_splits=effective_splits)
        folds = []
        for fold_id, (train_idx, validation_idx) in enumerate(tscv.split(unique_dates), start=1):
            train_dates = set(unique_dates[train_idx])
            validation_dates = set(unique_dates[validation_idx])

            train_df = engineered_df[engineered_df['week_start_date'].isin(train_dates)].copy()
            validation_df = engineered_df[engineered_df['week_start_date'].isin(validation_dates)].copy()

            folds.append(
                {
                    'fold_id': int(fold_id),
                    'train_df': train_df,
                    'validation_df': validation_df,
                    'train_end': str(pd.Timestamp(max(train_dates)).date()),
                    'validation_start': str(pd.Timestamp(min(validation_dates)).date()),
                    'validation_end': str(pd.Timestamp(max(validation_dates)).date()),
                }
            )
        return folds

    def _scale_pos_weight(self, y):
        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        if positives == 0:
            return 1.0
        return max(1.0, negatives / positives)

    def _evaluate(self, y_true, y_pred, y_pred_proba):
        roc_auc = np.nan
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc) if not pd.isna(roc_auc) else np.nan,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

    def _score_for_selection(self, metrics):
        return (SELECTION_RECALL_WEIGHT * metrics['recall']) + (SELECTION_F1_WEIGHT * metrics['f1_score'])

    def _tune_threshold(self, y_true, y_proba):
        best_threshold = THRESHOLD_FALLBACK
        best_metrics = self._evaluate(y_true, (y_proba >= THRESHOLD_FALLBACK).astype(int), y_proba)
        best_score = self._score_for_selection(best_metrics)

        curve = []
        for threshold in np.arange(0.20, 0.81, 0.02):
            y_pred = (y_proba >= threshold).astype(int)
            metrics = self._evaluate(y_true, y_pred, y_proba)
            score = self._score_for_selection(metrics)
            curve.append(
                {
                    'threshold': round(float(threshold), 2),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1_score': float(metrics['f1_score']),
                }
            )

            if score > best_score or (np.isclose(score, best_score) and metrics['recall'] > best_metrics['recall']):
                best_score = score
                best_threshold = float(threshold)
                best_metrics = metrics

        return best_threshold, best_metrics, curve

    def _class_balance(self, engineered_df, target_col='outbreak_next_week'):
        distribution = engineered_df[target_col].value_counts(normalize=True).sort_index()
        return {int(label): float(value) for label, value in distribution.items()}

    def _class_balance_assessment(self, class_balance):
        positive_rate = float(class_balance.get(1, 0.0))
        if positive_rate < 0.05:
            return 'unstable-positive-rate-below-5-percent'
        if positive_rate > 0.50:
            return 'unrealistic-positive-rate-above-50-percent'
        if 0.10 <= positive_rate <= 0.25:
            return 'ideal-range-10-to-25-percent'
        return 'acceptable-but-outside-ideal-range'

    def _inject_realism_noise(self, df):
        noisy = df.copy()
        random_generator = np.random.default_rng(42)

        if 'rainfall_mm' in noisy.columns:
            rainfall = pd.to_numeric(noisy['rainfall_mm'], errors='coerce').fillna(0)
            rainfall_jitter = random_generator.normal(loc=0.0, scale=0.03, size=len(noisy))
            noisy['rainfall_mm'] = (rainfall * (1 + rainfall_jitter)).clip(lower=0)

        if 'ecoli_index' in noisy.columns:
            ecoli = pd.to_numeric(noisy['ecoli_index'], errors='coerce').fillna(0)
            ecoli_jitter = random_generator.normal(loc=0.0, scale=0.04, size=len(noisy))
            noisy['ecoli_index'] = (ecoli * (1 + ecoli_jitter)).clip(lower=0)

        if 'reported_cases' in noisy.columns:
            cases = pd.to_numeric(noisy['reported_cases'], errors='coerce').fillna(0)
            case_jitter = random_generator.normal(loc=0.0, scale=0.10, size=len(noisy))
            noisy['reported_cases'] = np.rint((cases * (1 + case_jitter)).clip(lower=0)).astype(int)

        return noisy

    def _run_leakage_checks_cv(self, engineered_df, raw_df, fold_metrics, target_col='outbreak_next_week'):
        target_not_in_features = target_col not in self.feature_columns
        no_future_rolling = True

        pre_lag_data = prepare_outbreak_data(raw_df)
        recomputed = (
            pre_lag_data.sort_values(['ward_id', 'week_start_date'])
            .groupby('ward_id')['rainfall_mm']
            .transform(lambda values: values.shift(1).rolling(window=2, min_periods=2).mean())
        )
        pre_lag_data = pre_lag_data[['ward_id', 'week_start_date']].copy()
        pre_lag_data['recomputed_rainfall_2w_avg'] = recomputed

        merged = engineered_df.merge(pre_lag_data, on=['ward_id', 'week_start_date'], how='left')
        comparable = merged['rainfall_2w_avg'].notna() & merged['recomputed_rainfall_2w_avg'].notna()
        if comparable.any():
            no_future_rolling = np.allclose(
                merged.loc[comparable, 'rainfall_2w_avg'].to_numpy(),
                merged.loc[comparable, 'recomputed_rainfall_2w_avg'].to_numpy(),
                atol=1e-8,
            )

        strict_time_split_respected = True
        for fold in fold_metrics:
            if fold.get('skipped', False):
                continue
            if fold['train_end'] >= fold['validation_start']:
                strict_time_split_respected = False
                break

        return {
            'target_not_in_features': bool(target_not_in_features),
            'rolling_uses_only_past_rows': bool(no_future_rolling),
            'strict_time_split_respected': bool(strict_time_split_respected),
        }

    def _fit_calibrator_from_oof(self, y_true, y_proba):
        if len(y_true) == 0:
            self.calibration = {'method': 'none'}
            return

        if len(np.unique(y_true)) > 1:
            try:
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_proba, y_true)
                self.calibration = {
                    'method': 'isotonic',
                    'calibrator': calibrator,
                    'fitted_on': 'oof_cv',
                }
                return
            except Exception:
                pass

        alpha = 0.1
        target_rate = float(np.mean(y_true))
        self.calibration = {
            'method': 'smoothing',
            'alpha': float(alpha),
            'target_rate': float(target_rate),
            'fitted_on': 'oof_cv',
        }

    def _summarize_cv_metrics(self, fold_metrics):
        valid = [fold for fold in fold_metrics if not fold.get('skipped', False)]
        if not valid:
            raise ValueError('No valid CV folds available for metric summarization.')

        recalls = np.array([fold['recall'] for fold in valid], dtype=float)
        f1_scores = np.array([fold['f1_score'] for fold in valid], dtype=float)
        roc_values = np.array([fold['roc_auc'] for fold in valid], dtype=float)

        return {
            'accuracy_mean': float(np.mean([fold['accuracy'] for fold in valid])),
            'accuracy_std': float(np.std([fold['accuracy'] for fold in valid])),
            'precision_mean': float(np.mean([fold['precision'] for fold in valid])),
            'precision_std': float(np.std([fold['precision'] for fold in valid])),
            'recall_mean': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_mean': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
            'roc_auc_mean': float(np.nanmean(roc_values)) if len(roc_values) else np.nan,
            'roc_auc_std': float(np.nanstd(roc_values)) if len(roc_values) else np.nan,
        }

    def _evaluate_param_with_cv(self, engineered_df, folds, params, target_col='outbreak_next_week'):
        fold_metrics = []
        fold_thresholds = []
        fold_curves = []
        oof_rows = []
        skipped_folds = []

        for fold in folds:
            train_df = fold['train_df']
            validation_df = fold['validation_df']

            y_train = train_df[target_col].astype(int)
            y_validation = validation_df[target_col].astype(int)

            if len(train_df) == 0 or len(validation_df) == 0:
                skipped_folds.append({'fold_id': fold['fold_id'], 'reason': 'empty_train_or_validation'})
                fold_metrics.append({'fold_id': fold['fold_id'], 'skipped': True, 'reason': 'empty_train_or_validation'})
                continue

            if len(np.unique(y_train)) < 2:
                skipped_folds.append({'fold_id': fold['fold_id'], 'reason': 'single_class_train'})
                fold_metrics.append({'fold_id': fold['fold_id'], 'skipped': True, 'reason': 'single_class_train'})
                continue

            if len(np.unique(y_validation)) < 2:
                skipped_folds.append({'fold_id': fold['fold_id'], 'reason': 'single_class_validation'})
                fold_metrics.append({'fold_id': fold['fold_id'], 'skipped': True, 'reason': 'single_class_validation'})
                continue

            model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                reg_lambda=params['reg_lambda'],
                scale_pos_weight=self._scale_pos_weight(y_train),
            )

            X_train = train_df[self.feature_columns]
            X_validation = validation_df[self.feature_columns]
            model.fit(X_train, y_train, verbose=False)

            y_validation_proba = model.predict_proba(X_validation)[:, 1]
            tuned_threshold, tuned_metrics, threshold_curve = self._tune_threshold(y_validation, y_validation_proba)
            y_validation_pred = (y_validation_proba >= tuned_threshold).astype(int)
            metrics = self._evaluate(y_validation, y_validation_pred, y_validation_proba)

            fold_record = {
                'fold_id': int(fold['fold_id']),
                'skipped': False,
                'train_end': fold['train_end'],
                'validation_start': fold['validation_start'],
                'validation_end': fold['validation_end'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'best_threshold': float(tuned_threshold),
                'train_prevalence': float((y_train == 1).mean()),
                'validation_prevalence': float((y_validation == 1).mean()),
                'selection_score': float(self._score_for_selection(tuned_metrics)),
            }
            fold_metrics.append(fold_record)
            fold_thresholds.append(float(tuned_threshold))
            fold_curves.append({'fold_id': int(fold['fold_id']), 'curve': threshold_curve})

            oof_rows.append(
                pd.DataFrame(
                    {
                        'row_index': validation_df.index.to_numpy(),
                        'y_true': y_validation.to_numpy(),
                        'y_proba': y_validation_proba,
                        'fold_id': int(fold['fold_id']),
                    }
                )
            )

        valid_folds = [fold for fold in fold_metrics if not fold.get('skipped', False)]
        mean_selection_score = float(np.mean([fold['selection_score'] for fold in valid_folds])) if valid_folds else -np.inf

        oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(columns=['row_index', 'y_true', 'y_proba', 'fold_id'])
        return {
            'params': params,
            'fold_metrics': fold_metrics,
            'fold_thresholds': fold_thresholds,
            'fold_threshold_curves': fold_curves,
            'valid_fold_count': int(len(valid_folds)),
            'skipped_folds': skipped_folds,
            'mean_selection_score': mean_selection_score,
            'oof_predictions': oof_df,
        }

    def _bundle_score(self, fold_metrics):
        valid = [fold for fold in fold_metrics if not fold.get('skipped', False)]
        if not valid:
            return -np.inf

        scores = []
        for fold in valid:
            roc_auc = fold.get('roc_auc', np.nan)
            roc_component = 0.5 if pd.isna(roc_auc) else float(roc_auc)
            recall_component = float(fold.get('recall', 0.0))
            f1_component = float(fold.get('f1_score', 0.0))
            scores.append((0.5 * roc_component) + (0.3 * recall_component) + (0.2 * f1_component))
        return float(np.mean(scores))

    def _build_feature_set_candidates(self):
        full_features = get_model_feature_columns()
        candidates = [
            {'name': 'full_feature_set', 'columns': full_features},
            {
                'name': 'drop_interaction_feature',
                'columns': [column for column in full_features if column != 'rainfall_ecoli_interaction'],
            },
        ]
        dedup = []
        seen = set()
        for candidate in candidates:
            key = tuple(candidate['columns'])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(candidate)
        return dedup

    def train(self, df, target_col='outbreak_next_week', apply_realism_noise=False):
        runtime_cfg = load_runtime_config()
        training_df = self._inject_realism_noise(df) if apply_realism_noise else df.copy()

        if target_col in training_df.columns:
            raw_target = pd.to_numeric(training_df[target_col], errors='coerce').fillna(0)
            raw_target = (raw_target > 0).astype(int)
            print("Raw target distribution: df['outbreak_next_week'].value_counts(normalize=True)")
            print(raw_target.value_counts(normalize=True))

        prepared = self.prepare_data(training_df, target_col=target_col)
        engineered = prepared['engineered']
        folds = self._build_time_folds(engineered, n_splits=5)

        param_grid = [
            {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 220, 'min_child_weight': 2, 'reg_lambda': 1.0, 'subsample': 0.9, 'colsample_bytree': 0.9},
            {'max_depth': 4, 'learning_rate': 0.04, 'n_estimators': 300, 'min_child_weight': 3, 'reg_lambda': 1.5, 'subsample': 0.85, 'colsample_bytree': 0.85},
            {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 360, 'min_child_weight': 3, 'reg_lambda': 2.0, 'subsample': 0.8, 'colsample_bytree': 0.8},
        ]

        best_bundle = None
        search_results = []
        total_candidates = len(self._build_feature_set_candidates()) * len(param_grid)
        candidate_counter = 0
        for feature_candidate in self._build_feature_set_candidates():
            self.feature_columns = feature_candidate['columns']
            for params in param_grid:
                candidate_counter += 1
                print(
                    f"CV Search {candidate_counter}/{total_candidates} | "
                    f"feature_set={feature_candidate['name']} | params={params}"
                )
                bundle = self._evaluate_param_with_cv(engineered, folds, params, target_col=target_col)
                bundle['feature_set_name'] = feature_candidate['name']
                bundle['feature_columns'] = list(feature_candidate['columns'])
                bundle['bundle_score'] = self._bundle_score(bundle['fold_metrics'])
                search_results.append(
                    {
                        'feature_set_name': feature_candidate['name'],
                        'params': params,
                        'bundle_score': bundle['bundle_score'],
                        'valid_fold_count': bundle['valid_fold_count'],
                    }
                )
                if best_bundle is None or bundle['bundle_score'] > best_bundle['bundle_score']:
                    best_bundle = bundle

        if best_bundle['valid_fold_count'] < 2:
            raise ValueError('Insufficient valid CV folds after skipping single-class folds.')

        self.best_params = best_bundle['params']
        self.feature_columns = best_bundle['feature_columns']
        self.best_threshold = float(np.mean(best_bundle['fold_thresholds'])) if best_bundle['fold_thresholds'] else THRESHOLD_FALLBACK

        oof_df = best_bundle['oof_predictions'].sort_values('row_index').reset_index(drop=True)
        y_oof = oof_df['y_true'].to_numpy(dtype=int)
        y_oof_proba = oof_df['y_proba'].to_numpy(dtype=float)
        self._fit_calibrator_from_oof(y_oof, y_oof_proba)

        y_full = engineered[target_col].astype(int)
        X_full = engineered[self.feature_columns]
        self.model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            max_depth=self.best_params['max_depth'],
            learning_rate=self.best_params['learning_rate'],
            n_estimators=self.best_params['n_estimators'],
            subsample=self.best_params['subsample'],
            colsample_bytree=self.best_params['colsample_bytree'],
            min_child_weight=self.best_params['min_child_weight'],
            reg_lambda=self.best_params['reg_lambda'],
            scale_pos_weight=self._scale_pos_weight(y_full),
        )
        self.model.fit(X_full, y_full, verbose=False)

        fold_metrics = best_bundle['fold_metrics']
        cv_metrics = self._summarize_cv_metrics(fold_metrics)
        valid_fold_metrics = [fold for fold in fold_metrics if not fold.get('skipped', False)]

        prevalence_values = [fold['validation_prevalence'] for fold in valid_fold_metrics]
        cross_fold_prevalence = {
            'validation_prevalence_mean': float(np.mean(prevalence_values)) if prevalence_values else 0.0,
            'validation_prevalence_std': float(np.std(prevalence_values)) if prevalence_values else 0.0,
            'validation_prevalence_min': float(np.min(prevalence_values)) if prevalence_values else 0.0,
            'validation_prevalence_max': float(np.max(prevalence_values)) if prevalence_values else 0.0,
        }

        class_balance = self._class_balance(engineered, target_col=target_col)
        class_balance_assessment = self._class_balance_assessment(class_balance)
        leakage_checks = self._run_leakage_checks_cv(engineered, raw_df=training_df, fold_metrics=fold_metrics, target_col=target_col)

        self.metrics = {
            'cv_metrics': cv_metrics,
            'cv_fold_metrics': fold_metrics,
            'cv_thresholds': [float(value) for value in best_bundle['fold_thresholds']],
            'global_threshold': float(self.best_threshold),
            'best_threshold': float(self.best_threshold),
            'n_folds': int(len(folds)),
            'n_valid_folds': int(best_bundle['valid_fold_count']),
            'skipped_folds': best_bundle['skipped_folds'],
            'best_params': self.best_params,
            'selected_feature_set': best_bundle.get('feature_set_name'),
            'model_search_results': search_results,
            'n_samples': int(len(engineered)),
            'n_features': int(len(self.feature_columns)),
            'noise_applied': bool(apply_realism_noise),
            'class_balance': class_balance,
            'class_balance_assessment': class_balance_assessment,
            'leakage_checks': leakage_checks,
            'cross_fold_prevalence': cross_fold_prevalence,
            'outbreak_ratios': {
                'train': float(np.mean([fold['train_prevalence'] for fold in valid_fold_metrics])) if valid_fold_metrics else 0.0,
                'validation': float(cross_fold_prevalence['validation_prevalence_mean']),
                'test': float(cross_fold_prevalence['validation_prevalence_mean']),
            },
            'calibration': {
                key: value for key, value in self.calibration.items() if key != 'calibrator'
            },
            'oof_prediction_count': int(len(oof_df)),
            'threshold_curve': best_bundle['fold_threshold_curves'][0]['curve'] if best_bundle['fold_threshold_curves'] else [],
            'training_data_range': {
                'start': str(pd.to_datetime(engineered['week_start_date']).min().date()),
                'end': str(pd.to_datetime(engineered['week_start_date']).max().date()),
            },
            'trained_at_utc': datetime.now(timezone.utc).isoformat(),
            'model_artifact_version': runtime_cfg.get('artifact_version', 'unknown'),
            'evaluation_method': 'rolling_timeseries_cv',
            'final_model_trained_on_full_data': True,
            'calibration_source': 'out_of_fold_validation_predictions',
            'validation_metrics': {
                'accuracy': cv_metrics['accuracy_mean'],
                'precision': cv_metrics['precision_mean'],
                'recall': cv_metrics['recall_mean'],
                'f1_score': cv_metrics['f1_mean'],
                'roc_auc': cv_metrics['roc_auc_mean'],
            },
            'test_metrics': {
                'accuracy': cv_metrics['accuracy_mean'],
                'precision': cv_metrics['precision_mean'],
                'recall': cv_metrics['recall_mean'],
                'f1_score': cv_metrics['f1_mean'],
                'roc_auc': cv_metrics['roc_auc_mean'],
                'confusion_matrix': [],
            },
            'distribution_shift': {
                'train_ratio': float(np.mean([fold['train_prevalence'] for fold in valid_fold_metrics])) if valid_fold_metrics else 0.0,
                'validation_ratio': float(cross_fold_prevalence['validation_prevalence_mean']),
                'test_ratio': float(cross_fold_prevalence['validation_prevalence_mean']),
                'max_abs_shift': float(cross_fold_prevalence['validation_prevalence_max'] - cross_fold_prevalence['validation_prevalence_min']),
                'severe_shift': bool(cross_fold_prevalence['validation_prevalence_std'] >= 0.10),
            },
        }

        self.save_model()

        print("\n" + "=" * 60)
        print("OUTBREAK MODEL TRAINING RESULTS - ROLLING CV")
        print("=" * 60)
        print(f"Selected feature set: {best_bundle.get('feature_set_name')}")
        print(f"Selected params: {self.best_params}")
        print(f"Bundle score (ROC/Recall/F1 blended): {best_bundle.get('bundle_score'):.4f}")
        print("Fold-wise metrics:")
        for fold in fold_metrics:
            if fold.get('skipped', False):
                print(f"Fold {fold['fold_id']}: SKIPPED ({fold['reason']})")
                continue
            print(
                f"Fold {fold['fold_id']} [{fold['validation_start']} → {fold['validation_end']}] | "
                f"Recall={fold['recall']:.2%}, F1={fold['f1_score']:.2%}, ROC-AUC={fold['roc_auc']:.4f}, "
                f"Threshold={fold['best_threshold']:.2f}"
            )

        print("\nMean CV metrics:")
        print(f"Recall  : {cv_metrics['recall_mean']:.2%} ± {cv_metrics['recall_std']:.2%}")
        print(f"F1      : {cv_metrics['f1_mean']:.2%} ± {cv_metrics['f1_std']:.2%}")
        if pd.notna(cv_metrics['roc_auc_mean']):
            print(f"ROC-AUC : {cv_metrics['roc_auc_mean']:.4f} ± {cv_metrics['roc_auc_std']:.4f}")
        else:
            print("ROC-AUC : N/A")

        print(f"\nGlobal threshold (mean fold threshold): {self.best_threshold:.2f}")
        print("Final model trained on full dataset: YES")
        print("Calibration fitted on out-of-fold predictions: YES")
        print("=" * 60 + "\n")

        return self.metrics

    def save_model(self):
        if self.model is None:
            raise ValueError('No trained model to save.')
        full_model_path = self._resolve_path(self.model_path)
        os.makedirs(os.path.dirname(full_model_path), exist_ok=True)
        joblib.dump(
            {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'feature_list': self.feature_columns,
                'metrics': self.metrics,
                'best_params': self.best_params,
                'best_threshold': self.best_threshold,
                'calibration': self.calibration,
                'artifact_version': self.metrics.get('model_artifact_version', 'unknown'),
                'trained_at_utc': self.metrics.get('trained_at_utc'),
            },
            full_model_path,
        )
        return full_model_path

    def _ensure_model(self):
        if self.model is None:
            full_model_path = self._resolve_path(self.model_path)
            if not os.path.exists(full_model_path):
                raise ValueError('Model not trained and saved yet.')
            payload = joblib.load(full_model_path)
            self.model = payload['model']
            self.feature_columns = payload.get('feature_columns', payload.get('feature_list', self.feature_columns))
            self.metrics = payload.get('metrics', {})
            self.best_params = payload.get('best_params', {})
            self.best_threshold = payload.get('best_threshold', self.metrics.get('best_threshold', THRESHOLD_FALLBACK))
            self.calibration = payload.get('calibration', self.metrics.get('calibration', {'method': 'none'}))

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

    def predict(self, df):
        self._ensure_model()
        engineered = engineer_outbreak_features(df, dropna_lag_rows=True)
        latest = engineered.sort_values('week_start_date').groupby('ward_id', as_index=False).tail(1).copy()
        validate_feature_schema(latest, self.feature_columns)
        X_latest = latest[self.feature_columns]

        probabilities = self.model.predict_proba(X_latest)[:, 1]
        probabilities = self._apply_probability_adjustment(probabilities)
        predictions = (probabilities >= self.best_threshold).astype(int)

        return pd.DataFrame(
            {
                'week_start_date': latest['week_start_date'].values,
                'ward_id': latest['ward_id'].values,
                'probability': probabilities,
                'prediction': predictions,
            }
        ).sort_values('ward_id').reset_index(drop=True)

    def get_feature_importance(self, top_n=10):
        self._ensure_model()
        importance_df = pd.DataFrame(
            {'feature': self.feature_columns, 'importance': self.model.feature_importances_}
        )
        return importance_df.sort_values('importance', ascending=False).head(top_n)

    def get_shap_values(self, X_sample, top_n=10):
        self._ensure_model()
        if not SHAP_AVAILABLE:
            return None, self.get_feature_importance(top_n)
        try:
            sample = X_sample[self.feature_columns]
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(sample)
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_importance_df = pd.DataFrame(
                {'feature': self.feature_columns, 'shap_importance': mean_shap}
            ).sort_values('shap_importance', ascending=False).head(top_n)
            return shap_values, shap_importance_df
        except Exception:
            return None, self.get_feature_importance(top_n)
