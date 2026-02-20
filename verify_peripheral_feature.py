"""
Verification of is_peripheral_ward feature implementation.

Validates:
1. Exactly 1 new static feature added
2. Binary values (0 or 1)
3. Graceful degradation when peripheral_wards.json missing
4. Time-independent (constant per ward)
5. No breaking changes
"""

import sys
from pathlib import Path
import pandas as pd

from utils.feature_engineering import (
    engineer_outbreak_features,
    get_spatial_feature_columns,
    get_model_feature_columns,
    load_peripheral_metadata,
)


def print_check(name: str, passed: bool, detail: str = "") -> None:
    """Print check result."""
    symbol = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    msg = f"[{symbol}] {name}: {status}"
    if detail:
        msg += f" - {detail}"
    print(msg)


def test_feature_definition():
    """Test 1: Feature is properly defined in spatial features."""
    print("\n=== Test 1: Feature Definition ===")
    
    spatial_features = get_spatial_feature_columns()
    has_peripheral = 'is_peripheral_ward' in spatial_features
    print_check("is_peripheral_ward in spatial features", has_peripheral)
    
    expected_count = 3  # 2 neighbor + 1 peripheral
    count_ok = len(spatial_features) == expected_count
    print_check(f"Total spatial features = {expected_count}", count_ok, 
                f"{len(spatial_features)} features")
    
    return has_peripheral and count_ok


def test_feature_values():
    """Test 2: Feature has correct binary values."""
    print("\n=== Test 2: Feature Values ===")
    
    df = pd.read_csv('data/coimbatore_unified_master_dataset_with_zone.csv')
    result = engineer_outbreak_features(df.copy())
    
    # Check feature exists
    exists = 'is_peripheral_ward' in result.columns
    print_check("Feature created", exists)
    
    if not exists:
        return False
    
    # Check binary values
    unique_vals = set(result['is_peripheral_ward'].unique())
    is_binary = unique_vals.issubset({0, 1})
    print_check("Binary values (0 or 1)", is_binary, f"values={unique_vals}")
    
    # Check no NaN
    no_nan = not result['is_peripheral_ward'].isna().any()
    print_check("No NaN values", no_nan)
    
    # Check time-independent (same value for all weeks per ward)
    max_unique_per_ward = result.groupby('ward_id')['is_peripheral_ward'].nunique().max()
    is_static = max_unique_per_ward == 1
    print_check("Time-independent (constant per ward)", is_static, 
                f"max_unique={max_unique_per_ward}")
    
    # Check distribution
    counts = result['is_peripheral_ward'].value_counts()
    print(f"  Distribution: {dict(counts)}")
    
    return exists and is_binary and no_nan and is_static


def test_graceful_degradation():
    """Test 3: Pipeline works when peripheral_wards.json missing."""
    print("\n=== Test 3: Graceful Degradation ===")
    
    peripheral_path = Path('data/peripheral_wards.json')
    backup_path = Path('data/peripheral_wards.json.bak')
    
    if peripheral_path.exists():
        peripheral_path.rename(backup_path)
    
    try:
        # Load function should return empty set
        peripheral_set = load_peripheral_metadata()
        empty_ok = len(peripheral_set) == 0
        print_check("Returns empty set when missing", empty_ok)
        
        # Pipeline should continue
        df = pd.read_csv('data/coimbatore_unified_master_dataset_with_zone.csv')
        result = engineer_outbreak_features(df.copy())
        pipeline_ok = len(result) > 0
        print_check("Pipeline runs without error", pipeline_ok, f"{len(result)} rows")
        
        # Feature should not be added
        not_added = 'is_peripheral_ward' not in result.columns
        print_check("Feature not added when missing", not_added)
        
        return empty_ok and pipeline_ok and not_added
    
    finally:
        if backup_path.exists():
            backup_path.rename(peripheral_path)


def test_backward_compatibility():
    """Test 4: No breaking changes to existing functionality."""
    print("\n=== Test 4: Backward Compatibility ===")
    
    # Without spatial features
    base_features = get_model_feature_columns(include_spatial=False)
    base_count = len(base_features)
    no_peripheral_in_base = 'is_peripheral_ward' not in base_features
    print_check("Base features unchanged", no_peripheral_in_base, 
                f"{base_count} features")
    
    # With spatial features
    spatial_features = get_model_feature_columns(include_spatial=True)
    spatial_count = len(spatial_features)
    expected_increase = 3  # 2 neighbor + 1 peripheral
    correct_increase = spatial_count == base_count + expected_increase
    print_check(f"Spatial adds exactly +{expected_increase}", correct_increase, 
                f"{base_count} -> {spatial_count}")
    
    # Check is_peripheral_ward is included
    has_peripheral = 'is_peripheral_ward' in spatial_features
    print_check("is_peripheral_ward in spatial mode", has_peripheral)
    
    return no_peripheral_in_base and correct_increase and has_peripheral


def test_no_overbuilding():
    """Test 5: No unwanted features added."""
    print("\n=== Test 5: No Overbuilding ===")
    
    df = pd.read_csv('data/coimbatore_unified_master_dataset_with_zone.csv')
    result = engineer_outbreak_features(df.copy())
    
    # Check that distance features are NOT added
    unwanted_features = [
        'distance_from_city_center',
        'ring_road_exposure_score',
    ]
    
    none_added = True
    for feature in unwanted_features:
        exists = feature in result.columns
        if exists:
            print_check(f"{feature} NOT added", False, "FOUND")
            none_added = False
        else:
            print_check(f"{feature} NOT added", True)
    
    # Check spatial features count
    spatial_cols = [c for c in result.columns if c in get_spatial_feature_columns()]
    correct_count = len(spatial_cols) == 3
    print_check("Exactly 3 spatial features in output", correct_count, 
                f"{len(spatial_cols)} found: {spatial_cols}")
    
    return none_added and correct_count


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("  is_peripheral_ward FEATURE VERIFICATION")
    print("  Static binary flag - no overbuilding")
    print("=" * 60)
    
    results = []
    
    results.append(("Feature Definition", test_feature_definition()))
    results.append(("Feature Values", test_feature_values()))
    results.append(("Graceful Degradation", test_graceful_degradation()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("No Overbuilding", test_no_overbuilding()))
    
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
