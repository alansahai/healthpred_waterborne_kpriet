"""
Diagnostic script to check why spatial features aren't showing in the heatmap
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model.predict import OutbreakPredictor
from utils.spatial_viz import (
    load_ward_adjacency, 
    load_peripheral_wards, 
    load_ward_centroids,
    compute_infection_influence_arrows,
    compute_neighbor_risk_overlay,
    get_spatial_viz_summary
)
import pandas as pd

print("=" * 80)
print("SPATIAL FEATURES DIAGNOSTIC")
print("=" * 80)

# Step 1: Check spatial data files
print("\n1. Checking spatial data files...")
adjacency = load_ward_adjacency()
peripheral = load_peripheral_wards()
centroids = load_ward_centroids()

print(f"   ✓ Ward adjacency: {len(adjacency)} wards")
print(f"   ✓ Peripheral wards: {len(peripheral)} wards")
print(f"   ✓ Centroids: {len(centroids)} wards")

if adjacency:
    sample_ward = list(adjacency.keys())[0]
    print(f"   Sample: {sample_ward} has {len(adjacency[sample_ward])} neighbors: {adjacency[sample_ward][:3]}...")

# Step 2: Load current predictions
print("\n2. Loading current predictions...")
predictor = OutbreakPredictor()
predictor.load_model()
predictions = predictor.predict_latest_week()

print(f"   ✓ Total wards: {len(predictions)}")
print(f"   Risk level distribution:")
for risk_level in ['Low', 'Moderate', 'High']:
    count = len(predictions[predictions['risk_level'] == risk_level])
    print(f"     - {risk_level}: {count} wards")

# Step 3: Check for high-risk wards
print("\n3. Checking high-risk wards...")
high_risk = predictions[predictions['risk_level'] == 'High']
print(f"   High-risk wards: {len(high_risk)}")
if len(high_risk) > 0:
    print(f"   Ward IDs: {high_risk['ward_id'].tolist()[:10]}")
    
    # Check which are peripheral
    high_risk_ward_ids = set(high_risk['ward_id'].tolist())
    high_risk_peripheral = high_risk_ward_ids.intersection(peripheral)
    print(f"   High-risk AND peripheral: {len(high_risk_peripheral)} wards")
    if high_risk_peripheral:
        print(f"   Examples: {list(high_risk_peripheral)[:5]}")
else:
    print("   ⚠️ NO HIGH-RISK WARDS FOUND - This is likely why features don't show!")

# Step 4: Test spillover arrows function
print("\n4. Testing spillover arrows computation...")
threshold = 0.23  # Default threshold
arrows = compute_infection_influence_arrows(predictions, threshold=threshold)
print(f"   Arrows generated: {len(arrows)}")
if len(arrows) > 0:
    print(f"   Sample arrows:")
    for arrow in arrows[:3]:
        print(f"     {arrow['from_ward']} → {arrow['to_ward']} (influence: {arrow['influence']:.3f})")
else:
    print("   ⚠️ NO ARROWS GENERATED")

# Step 5: Test neighbor risk overlay
print("\n5. Testing neighbor risk overlay...")
neighbor_overlay = compute_neighbor_risk_overlay(predictions)
print(f"   Wards with neighbor data: {len(neighbor_overlay)}")
high_intensity = [w for w, d in neighbor_overlay.items() if d.get('intensity', 0) > 0.3]
print(f"   Wards with elevated neighbor risk (intensity > 0.3): {len(high_intensity)}")
if high_intensity:
    print(f"   Examples: {high_intensity[:5]}")

# Step 6: Check spatial viz summary
print("\n6. Spatial visualization summary...")
summary = get_spatial_viz_summary(predictions)
for key, value in summary.items():
    status = "✓" if value else "✗"
    print(f"   {status} {key}: {value}")

# Step 7: Recommendations
print("\n" + "=" * 80)
print("DIAGNOSIS & RECOMMENDATIONS")
print("=" * 80)

if len(high_risk) == 0:
    print("\n⚠️  PRIMARY ISSUE: No high-risk wards in current predictions")
    print("   → Spillover arrows require high-risk peripheral wards to show")
    print("   → Neighbor risk requires high-risk wards to compute intensity")
    print("\n   SOLUTIONS:")
    print("   1. Lower the threshold to create some high-risk predictions")
    print("   2. Wait for real high-risk conditions in the data")
    print("   3. Use what-if simulator to test features with synthetic high risk")
    print("   4. Generate test predictions with artificially elevated risk")
elif len(arrows) == 0:
    print("\n⚠️  ISSUE: High-risk wards exist but no arrows generated")
    print("   → Check if high-risk wards are peripheral")
    print("   → Check if adjacency data maps correctly to ward IDs")
elif not summary['all_available']:
    print("\n⚠️  ISSUE: Spatial data incomplete")
    print("   → Need to regenerate adjacency/peripheral/centroid files")
else:
    print("\n✓ All spatial data available and arrows should render")
    print("  → Issue may be in the map rendering logic")
    print("  → Check browser console for JavaScript errors")

print("\n" + "=" * 80)
