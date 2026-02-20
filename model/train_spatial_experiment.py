"""
Spatial Features Experiment Training Script

This script trains two models in comparison:
1. Baseline Model: Standard feature set (25 features)
2. Spatial Model: Extended with spatial features (32 features)

The models are evaluated using TimeSeriesSplit CV with recall-weighted scoring.
Results are compared and both artifacts are saved for further analysis.

Usage:
    python -m model.train_spatial_experiment
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.train import OutbreakModelTrainer
from utils.feature_engineering import (
    get_model_feature_columns,
    get_spatial_feature_columns,
)


def load_training_data():
    """Load the unified dataset for training."""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'coimbatore_unified_master_dataset_with_zone.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['week_start_date'])
    return df


def extract_key_metrics(metrics):
    """Extract key metrics for comparison."""
    cv = metrics.get('cv_metrics', {})
    return {
        'accuracy': round(cv.get('accuracy_mean', 0), 4),
        'precision': round(cv.get('precision_mean', 0), 4),
        'recall': round(cv.get('recall_mean', 0), 4),
        'f1_score': round(cv.get('f1_mean', 0), 4),
        'roc_auc': round(cv.get('roc_auc_mean', 0), 4),
        'n_features': metrics.get('n_features', 0),
        'n_valid_folds': metrics.get('n_valid_folds', 0),
        'selected_feature_set': metrics.get('selected_feature_set', 'unknown'),
        'best_params': metrics.get('best_params', {}),
    }


def print_comparison_table(baseline, spatial):
    """Print a comparison table of metrics."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS: Baseline vs Spatial Model")
    print("=" * 70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'n_features']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Features']
    
    print(f"\n{'Metric':<15} {'Baseline':>12} {'Spatial':>12} {'Delta':>12} {'Winner':>10}")
    print("-" * 61)
    
    for metric, label in zip(metrics, labels):
        b_val = baseline.get(metric, 0)
        s_val = spatial.get(metric, 0)
        delta = s_val - b_val
        
        if metric == 'n_features':
            winner = '-'
            delta_str = f"+{int(delta)}"
        else:
            delta_str = f"{delta:+.4f}"
            if abs(delta) < 0.001:
                winner = 'Tie'
            elif delta > 0:
                winner = 'Spatial'
            else:
                winner = 'Baseline'
        
        print(f"{label:<15} {b_val:>12.4f} {s_val:>12.4f} {delta_str:>12} {winner:>10}")
    
    print("-" * 61)
    
    # Overall assessment
    improvements = 0
    for metric in ['recall', 'f1_score', 'roc_auc']:
        if spatial.get(metric, 0) > baseline.get(metric, 0):
            improvements += 1
    
    print(f"\nSpatial Model improved on {improvements}/3 key metrics (Recall, F1, ROC-AUC)")
    
    recall_delta = spatial.get('recall', 0) - baseline.get('recall', 0)
    if recall_delta > 0.01:
        print(f"✓ Recall improvement: +{recall_delta:.2%} (critical for early warning)")
    elif recall_delta < -0.01:
        print(f"⚠ Recall degradation: {recall_delta:.2%} (spatial features may need tuning)")
    else:
        print(f"→ Recall stable: {recall_delta:+.2%}")
    
    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("SPATIAL FEATURES EXPERIMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Load data
    print("\n[1/4] Loading training data...")
    df = load_training_data()
    print(f"      Loaded {len(df)} rows, {len(df['ward_id'].unique())} wards")
    
    # Ensure artifacts directory exists
    artifacts_dir = os.path.join(PROJECT_ROOT, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Train baseline model
    print("\n[2/4] Training BASELINE model (no spatial features)...")
    print(f"      Features: {len(get_model_feature_columns(include_spatial=False))}")
    
    baseline_trainer = OutbreakModelTrainer(
        model_path='artifacts/outbreak_model_baseline.pkl',
        include_spatial_features=False
    )
    baseline_metrics = baseline_trainer.train(df)
    baseline_summary = extract_key_metrics(baseline_metrics)
    print(f"      ✓ Baseline trained: Recall={baseline_summary['recall']:.4f}, ROC-AUC={baseline_summary['roc_auc']:.4f}")
    
    # Train spatial model
    print("\n[3/4] Training SPATIAL model (with spatial features)...")
    print(f"      Features: {len(get_model_feature_columns(include_spatial=True))}")
    print(f"      Spatial columns: {get_spatial_feature_columns()}")
    
    spatial_trainer = OutbreakModelTrainer(
        model_path='artifacts/outbreak_model_spatial_experiment.pkl',
        include_spatial_features=True
    )
    spatial_metrics = spatial_trainer.train(df)
    spatial_summary = extract_key_metrics(spatial_metrics)
    print(f"      ✓ Spatial trained: Recall={spatial_summary['recall']:.4f}, ROC-AUC={spatial_summary['roc_auc']:.4f}")
    
    # Compare results
    print("\n[4/4] Comparing results...")
    print_comparison_table(baseline_summary, spatial_summary)
    
    # Save experiment summary
    experiment_summary = {
        'experiment': 'spatial_features_v2',
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'model_path': 'artifacts/outbreak_model_baseline.pkl',
            'metrics': baseline_summary,
            'spatial_enabled': False,
        },
        'spatial': {
            'model_path': 'artifacts/outbreak_model_spatial_experiment.pkl',
            'metrics': spatial_summary,
            'spatial_enabled': True,
            'spatial_features': get_spatial_feature_columns(),
        },
        'comparison': {
            'recall_delta': round(spatial_summary['recall'] - baseline_summary['recall'], 4),
            'roc_auc_delta': round(spatial_summary['roc_auc'] - baseline_summary['roc_auc'], 4),
            'f1_delta': round(spatial_summary['f1_score'] - baseline_summary['f1_score'], 4),
        }
    }
    
    summary_path = os.path.join(artifacts_dir, 'spatial_experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"\nExperiment summary saved to: {summary_path}")
    
    print(f"\nCompleted: {datetime.now().isoformat()}")
    
    return experiment_summary


if __name__ == '__main__':
    main()
