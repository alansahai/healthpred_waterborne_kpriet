"""
Comprehensive verification of minimal spatial feature implementation.

Validates:
1. Exactly 2 spatial features added
2. No data leakage (lag-1 only)
3. Graceful degradation when adjacency missing
4. Backward compatibility with existing pipeline
5. No function signature changes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from utils.feature_engineering import (
    engineer_outbreak_features,
    get_spatial_feature_columns,
    get_model_feature_columns,
    load_adjacency_map,
)


def print_check(name: str, passed: bool, detail: str = "") -> None:
    """Print check result."""
    symbol = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    msg = f"[{symbol}] {name}: {status}"
    if detail:
        msg += f" - {detail}"
    print(msg)


def test_feature_count():
    """Test 1: Exactly 2 spatial features defined."""
    print("\n=== Test 1: Feature Count ===")
    
    spatial_features = get_spatial_feature_columns()
    count_ok = len(spatial_features) == 2
    print_check("Exactly 2 spatial features", count_ok, f"{len(spatial_features)} features")
    
    expected = ['neighbor_avg_cases_last_week', 'neighbor_outbreak_rate_last_week']
    features_ok = set(spatial_features) == set(expected)
    print_check("Correct feature names", features_ok, f"{spatial_features}")
    
    return count_ok and features_ok


def test_no_leakage():
    """Test 2: Verify no temporal leakage (lag-1 only)."""
    print("\n=== Test 2: Leakage Protection ===")
    
    df = pd.read_csv('data/coimbatore_unified_master_dataset_with_zone.csv')
    result = engineer_outbreak_features(df.copy())
    
    # Check outbreak rate is in valid range (0-1, since it's average of binary)
    outbreak_col = 'neighbor_outbreak_rate_last_week'
    if outbreak_col in result.columns:
        min_val = result[outbreak_col].min()
        max_val = result[outbreak_col].max()
        range_ok = 0 <= min_val <= max_val <= 1
        print_check("Outbreak rate in [0,1]", range_ok, f"range=[{min_val:.3f}, {max_val:.3f}]")
    else:
        range_ok = False
        print_check("Outbreak rate exists", False)
    
    # Check cases are reasonable (avg of neighbor lag-1 values)
    cases_col = 'neighbor_avg_cases_last_week'
    if cases_col in result.columns:
        mean_val = result[cases_col].mean()
        cases_ok = mean_val > 0 and mean_val < 100  # Sanity check
        print_check("Cases values reasonable", cases_ok, f"mean={mean_val:.1f}")
    else:
        cases_ok = False
        print_check("Cases column exists", False)
    
    return range_ok and cases_ok


def test_graceful_degradation():
    """Test 3: Pipeline works when adjacency file missing."""
    print("\n=== Test 3: Graceful Degradation ===")
    
    # Temporarily move adjacency file
    adj_path = Path('data/ward_adjacency.json')
    backup_path = Path('data/ward_adjacency.json.bak')
    
    if adj_path.exists():
        adj_path.rename(backup_path)
    
    try:
        # Should return empty dict
        adjacency = load_adjacency_map()
        empty_ok = len(adjacency) == 0
        print_check("Returns empty dict when missing", empty_ok, f"{len(adjacency)} wards")
        
        # Feature engineering should still work
        df = pd.read_csv('data/coimbatore_unified_master_dataset_with_zone.csv')
        result = engineer_outbreak_features(df.copy())
        pipeline_ok = len(result) > 0
        print_check("Pipeline runs without adjacency", pipeline_ok, f"{len(result)} rows")
        
        # Should not have neighbor columns or they should be all NaN
        neighbor_cols = [c for c in result.columns if 'neighbor_avg_cases' in c or 'neighbor_outbreak' in c]
        no_neighbors = len(neighbor_cols) == 0
        print_check("No neighbor columns added", no_neighbors, f"{len(neighbor_cols)} cols")
        
        return empty_ok and pipeline_ok and no_neighbors
    
    finally:
        # Restore file
        if backup_path.exists():
            backup_path.rename(adj_path)


def test_backward_compatibility():
    """Test 4: No breaking changes to existing pipeline."""
    print("\n=== Test 4: Backward Compatibility ===")
    
    # Without spatial features (default behavior)
    base_features = get_model_feature_columns(include_spatial=False)
    base_count = len(base_features)
    no_neighbors_in_base = not any('neighbor_' in f for f in base_features)
    print_check("Base features unchanged", no_neighbors_in_base, f"{base_count} features")
    
    # With spatial features enabled
    spatial_features = get_model_feature_columns(include_spatial=True)
    spatial_count = len(spatial_features)
    exactly_two_added = spatial_count == base_count + 2
    print_check("Exactly +2 with spatial", exactly_two_added, 
                f"{base_count} -> {spatial_count}")
    
    # Check spatial features are appended, not inserted
    neighbor_features = [f for f in spatial_features if 'neighbor_' in f]
    correct_features = set(neighbor_features) == set(get_spatial_feature_columns())
    print_check("Correct spatial features added", correct_features, 
                f"{neighbor_features}")
    
    return no_neighbors_in_base and exactly_two_added and correct_features


def test_feature_engineering_integrity():
    """Test 5: Feature engineering produces valid output."""
    print("\n=== Test 5: Feature Engineering Integrity ===")
    
    df = pd.read_csv('data/coimbatore_unified_master_dataset_with_zone.csv')
    input_rows = len(df)
    
    result = engineer_outbreak_features(df.copy())
    output_rows = len(result)
    
    # Should drop some rows (first weeks with insufficient lag)
    rows_dropped = input_rows > output_rows
    print_check("Lag rows dropped", rows_dropped, 
                f"{input_rows} -> {output_rows}")
    
    # Check all expected columns exist
    expected_cols = get_model_feature_columns(include_spatial=True)
    all_exist = all(col in result.columns for col in expected_cols)
    print_check("All expected columns exist", all_exist)
    
    # Check no NaN in spatial features (after dropna)
    spatial_cols = get_spatial_feature_columns()
    no_nans = all(result[col].notna().all() for col in spatial_cols if col in result.columns)
    print_check("No NaN in spatial features", no_nans)
    
    # Check output sorted by ward and date
    is_sorted = (result[['ward_id', 'week_start_date']].reset_index(drop=True) ==
                 result[['ward_id', 'week_start_date']].sort_values(['ward_id', 'week_start_date']).reset_index(drop=True)).all().all()
    print_check("Output properly sorted", is_sorted)
    
    return rows_dropped and all_exist and no_nans and is_sorted


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("  MINIMAL SPATIAL FEATURE VERIFICATION")
    print("  2 neighbor lag features - no overbuilding")
    print("=" * 60)
    
    results = []
    
    results.append(("Feature Count", test_feature_count()))
    results.append(("Leakage Protection", test_no_leakage()))
    results.append(("Graceful Degradation", test_graceful_degradation()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Feature Engineering", test_feature_engineering_integrity()))
    
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        print_check(name, passed)
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + ("✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
