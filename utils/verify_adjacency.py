"""
Ward Adjacency Verification Utility

Standalone verification script for spatial adjacency computation.
Does NOT modify any existing files or import ML/dashboard modules.

Validates:
1. GeoJSON loads correctly with proper structure
2. Zone-level adjacency (touches/intersects) is computed correctly
3. Ward-level adjacency derivation is symmetric and correct
4. No self-neighbors, no null keys, sorted neighbor lists

Run: python utils/verify_adjacency.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
from shapely.ops import unary_union


# === Constants ===
PROJECT_ROOT = Path(__file__).parent.parent
GEOJSON_PATH = PROJECT_ROOT / "geo" / "coimbatore.geojson"
WARD_ZONE_MAPPING_PATH = PROJECT_ROOT / "data" / "ward_zone_mapping.json"
WARD_ADJACENCY_PATH = PROJECT_ROOT / "data" / "ward_adjacency.json"
PERIPHERAL_WARDS_PATH = PROJECT_ROOT / "data" / "peripheral_wards.json"


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_check(name: str, passed: bool, detail: str = "") -> None:
    """Print check result with symbol."""
    symbol = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    msg = f"[{symbol}] {name}: {status}"
    if detail:
        msg += f" - {detail}"
    print(msg)


# === Step 1: GeoJSON Loading Verification ===
def verify_geojson_loading() -> Tuple[bool, gpd.GeoDataFrame]:
    """Verify GeoJSON loads correctly with required structure."""
    print_header("Step 1: GeoJSON Loading Verification")
    
    all_passed = True
    gdf = None
    
    # Check file exists
    exists = GEOJSON_PATH.exists()
    print_check("GeoJSON file exists", exists, str(GEOJSON_PATH))
    if not exists:
        return False, None
    
    # Load GeoDataFrame
    try:
        gdf = gpd.read_file(GEOJSON_PATH)
        print_check("GeoJSON parsable", True, f"{len(gdf)} features loaded")
    except Exception as e:
        print_check("GeoJSON parsable", False, str(e))
        return False, None
    
    # Check not empty
    not_empty = len(gdf) > 0
    print_check("Contains features", not_empty, f"{len(gdf)} zones")
    all_passed &= not_empty
    
    # Check 'name' field exists (zone identifier)
    has_name = 'name' in gdf.columns
    print_check("'name' field exists", has_name)
    all_passed &= has_name
    
    # Check geometry column
    has_geometry = 'geometry' in gdf.columns and not gdf.geometry.isna().all()
    print_check("Geometry column valid", has_geometry)
    all_passed &= has_geometry
    
    # Check CRS
    has_crs = gdf.crs is not None
    crs_detail = str(gdf.crs) if has_crs else "None (will default to EPSG:4326)"
    print_check("CRS defined", has_crs, crs_detail)
    
    # Check for null geometries
    null_geom_count = gdf.geometry.isna().sum()
    no_null_geoms = null_geom_count == 0
    print_check("No null geometries", no_null_geoms, f"{null_geom_count} null")
    all_passed &= no_null_geoms
    
    # Check for duplicate zone names
    if has_name:
        duplicate_names = gdf['name'].duplicated().sum()
        no_duplicates = duplicate_names == 0
        print_check("No duplicate zone names", no_duplicates, f"{duplicate_names} duplicates")
        all_passed &= no_duplicates
    
    # Print zone summary
    if has_name:
        print(f"\n  Zones found: {list(gdf['name'].values)}")
    
    return all_passed, gdf


# === Step 2: Zone-Level Adjacency Computation ===
def verify_zone_adjacency(gdf: gpd.GeoDataFrame) -> Tuple[bool, Dict[str, List[str]]]:
    """Compute and verify zone-level adjacency from GeoJSON."""
    print_header("Step 2: Zone-Level Adjacency Computation")
    
    all_passed = True
    zone_adjacency: Dict[str, List[str]] = {}
    
    # Initialize adjacency for all zones
    zone_names = [str(row['name']) for _, row in gdf.iterrows()]
    for zone in zone_names:
        zone_adjacency[zone] = []
    
    # Compute adjacency using touches/intersects
    adjacency_pairs = []
    for i, row_i in gdf.iterrows():
        zone_i = str(row_i['name'])
        geom_i = row_i.geometry
        
        for j, row_j in gdf.iterrows():
            if i >= j:
                continue  # Skip self and already-checked pairs
            
            zone_j = str(row_j['name'])
            geom_j = row_j.geometry
            
            # Check if zones touch or intersect
            if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                adjacency_pairs.append((zone_i, zone_j))
                zone_adjacency[zone_i].append(zone_j)
                zone_adjacency[zone_j].append(zone_i)
    
    # Sort adjacency lists
    for zone in zone_adjacency:
        zone_adjacency[zone] = sorted(zone_adjacency[zone])
    
    print_check("Adjacency computation complete", True, f"{len(adjacency_pairs)} pairs found")
    
    # Print adjacency pairs
    print(f"\n  Adjacent zone pairs:")
    for z1, z2 in adjacency_pairs:
        print(f"    {z1} <-> {z2}")
    
    # Verify no self-neighbors
    self_neighbors = [z for z, neighbors in zone_adjacency.items() if z in neighbors]
    no_self = len(self_neighbors) == 0
    print_check("No self-neighbors", no_self, f"{len(self_neighbors)} found")
    all_passed &= no_self
    
    # Verify symmetry
    symmetry_ok = True
    for zone, neighbors in zone_adjacency.items():
        for neighbor in neighbors:
            if zone not in zone_adjacency.get(neighbor, []):
                symmetry_ok = False
                print(f"    Asymmetry: {zone} -> {neighbor} but not reverse")
    print_check("Symmetry holds", symmetry_ok)
    all_passed &= symmetry_ok
    
    # Print summary statistics
    neighbor_counts = [len(n) for n in zone_adjacency.values()]
    if neighbor_counts:
        print(f"\n  Zone neighbor counts:")
        for zone, neighbors in sorted(zone_adjacency.items()):
            print(f"    {zone}: {len(neighbors)} neighbors -> {neighbors}")
        print(f"\n  Statistics: Min={min(neighbor_counts)}, Max={max(neighbor_counts)}, Avg={sum(neighbor_counts)/len(neighbor_counts):.1f}")
    
    return all_passed, zone_adjacency


# === Step 3: Ward-Zone Mapping Verification ===
def verify_ward_zone_mapping() -> Tuple[bool, Dict]:
    """Verify ward-zone mapping file structure."""
    print_header("Step 3: Ward-Zone Mapping Verification")
    
    all_passed = True
    mapping = None
    
    # Check file exists
    exists = WARD_ZONE_MAPPING_PATH.exists()
    print_check("Mapping file exists", exists, str(WARD_ZONE_MAPPING_PATH))
    if not exists:
        return False, None
    
    # Load mapping
    try:
        with open(WARD_ZONE_MAPPING_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print_check("Mapping JSON parsable", True)
    except Exception as e:
        print_check("Mapping JSON parsable", False, str(e))
        return False, None
    
    # Check required keys
    has_zone_to_wards = 'zone_to_wards' in mapping
    has_ward_to_zone = 'ward_to_zone' in mapping
    print_check("'zone_to_wards' key exists", has_zone_to_wards)
    print_check("'ward_to_zone' key exists", has_ward_to_zone)
    all_passed &= has_zone_to_wards and has_ward_to_zone
    
    if not all_passed:
        return all_passed, mapping
    
    zone_to_wards = mapping['zone_to_wards']
    ward_to_zone = mapping['ward_to_zone']
    
    # Count wards and zones
    total_wards = sum(len(wards) for wards in zone_to_wards.values())
    zone_count = len(zone_to_wards)
    print_check("Ward count valid", total_wards > 0, f"{total_wards} wards across {zone_count} zones")
    
    # Verify ward_to_zone is inverse of zone_to_wards
    inverse_ok = True
    for zone, wards in zone_to_wards.items():
        for ward in wards:
            if ward_to_zone.get(ward) != zone:
                inverse_ok = False
                print(f"    Mismatch: {ward} should map to {zone}")
    print_check("Mapping consistency (inverse)", inverse_ok)
    all_passed &= inverse_ok
    
    # Print zone distribution
    print(f"\n  Ward distribution by zone:")
    for zone, wards in sorted(zone_to_wards.items()):
        print(f"    {zone}: {len(wards)} wards ({wards[0]}-{wards[-1]})")
    
    return all_passed, mapping


# === Step 4: Ward Adjacency File Verification ===
def verify_ward_adjacency_file(zone_adjacency: Dict[str, List[str]], mapping: Dict) -> bool:
    """Verify the ward_adjacency.json file."""
    print_header("Step 4: Ward Adjacency File Verification")
    
    all_passed = True
    
    # Check file exists
    exists = WARD_ADJACENCY_PATH.exists()
    print_check("Ward adjacency file exists", exists, str(WARD_ADJACENCY_PATH))
    if not exists:
        return False
    
    # Load adjacency
    try:
        with open(WARD_ADJACENCY_PATH, 'r', encoding='utf-8') as f:
            ward_adjacency = json.load(f)
        print_check("Ward adjacency JSON parsable", True, f"{len(ward_adjacency)} wards")
    except Exception as e:
        print_check("Ward adjacency JSON parsable", False, str(e))
        return False
    
    zone_to_wards = mapping['zone_to_wards']
    ward_to_zone = mapping['ward_to_zone']
    
    # Count expected wards
    expected_wards = set(ward_to_zone.keys())
    actual_wards = set(ward_adjacency.keys())
    
    all_wards_present = expected_wards == actual_wards
    print_check("All wards present", all_wards_present, 
                f"Expected {len(expected_wards)}, Got {len(actual_wards)}")
    all_passed &= all_wards_present
    
    if not all_wards_present:
        missing = expected_wards - actual_wards
        extra = actual_wards - expected_wards
        if missing:
            print(f"    Missing: {sorted(missing)[:5]}...")
        if extra:
            print(f"    Extra: {sorted(extra)[:5]}...")
    
    # Check no null keys
    null_keys = [k for k in ward_adjacency.keys() if k is None or k == ""]
    no_null_keys = len(null_keys) == 0
    print_check("No null keys", no_null_keys, f"{len(null_keys)} null keys")
    all_passed &= no_null_keys
    
    # Check no self-neighbors
    self_neighbors = [w for w, neighbors in ward_adjacency.items() if w in neighbors]
    no_self = len(self_neighbors) == 0
    print_check("No self-neighbors", no_self, f"{len(self_neighbors)} found")
    all_passed &= no_self
    
    # Check symmetry
    asymmetric_pairs = []
    for ward, neighbors in ward_adjacency.items():
        for neighbor in neighbors:
            if neighbor in ward_adjacency:
                if ward not in ward_adjacency[neighbor]:
                    asymmetric_pairs.append((ward, neighbor))
    symmetry_ok = len(asymmetric_pairs) == 0
    print_check("Symmetry holds", symmetry_ok, f"{len(asymmetric_pairs)} asymmetric pairs")
    all_passed &= symmetry_ok
    
    # Check neighbor lists are sorted
    unsorted = [w for w, neighbors in ward_adjacency.items() if neighbors != sorted(neighbors)]
    all_sorted = len(unsorted) == 0
    print_check("Neighbor lists sorted", all_sorted, f"{len(unsorted)} unsorted")
    all_passed &= all_sorted
    
    # Verify derivation logic: neighbors should be in adjacent zones
    derivation_errors = []
    for ward, neighbors in ward_adjacency.items():
        ward_zone = ward_to_zone.get(ward)
        if not ward_zone:
            continue
        
        allowed_zones = set(zone_adjacency.get(ward_zone, []))
        
        for neighbor in neighbors:
            neighbor_zone = ward_to_zone.get(neighbor)
            if neighbor_zone and neighbor_zone not in allowed_zones:
                derivation_errors.append((ward, neighbor, ward_zone, neighbor_zone))
    
    derivation_ok = len(derivation_errors) == 0
    print_check("Derivation logic correct", derivation_ok, 
                f"{len(derivation_errors)} cross-zone violations")
    all_passed &= derivation_ok
    
    # Statistics
    neighbor_counts = [len(n) for n in ward_adjacency.values()]
    print(f"\n  Ward neighbor statistics:")
    print(f"    Total wards: {len(ward_adjacency)}")
    print(f"    Min neighbors: {min(neighbor_counts)}")
    print(f"    Max neighbors: {max(neighbor_counts)}")
    print(f"    Avg neighbors: {sum(neighbor_counts)/len(neighbor_counts):.1f}")
    print(f"    Total neighbor pairs: {sum(neighbor_counts) // 2}")
    
    # Sample output
    sample_wards = sorted(ward_adjacency.keys())[:3]
    print(f"\n  Sample ward adjacency:")
    for ward in sample_wards:
        neighbors = ward_adjacency[ward]
        print(f"    {ward}: {len(neighbors)} neighbors -> {neighbors[:5]}{'...' if len(neighbors) > 5 else ''}")
    
    return all_passed


# === Step 5: Peripheral Wards Verification ===
def verify_peripheral_wards(gdf: gpd.GeoDataFrame, mapping: Dict) -> bool:
    """Verify peripheral wards file and computation."""
    print_header("Step 5: Peripheral Wards Verification")
    
    all_passed = True
    
    # Compute expected peripheral zones from GeoJSON
    district_geometry = unary_union(gdf.geometry)
    district_boundary = district_geometry.boundary
    
    expected_peripheral_zones = []
    for _, row in gdf.iterrows():
        zone_name = str(row['name'])
        if row.geometry.boundary.intersects(district_boundary):
            expected_peripheral_zones.append(zone_name)
    expected_peripheral_zones = sorted(expected_peripheral_zones)
    
    print_check("Peripheral zones computed", True, f"{len(expected_peripheral_zones)} zones")
    print(f"    Peripheral zones: {expected_peripheral_zones}")
    
    # Map to expected peripheral wards
    zone_to_wards = mapping['zone_to_wards']
    expected_peripheral_wards = []
    for zone in expected_peripheral_zones:
        expected_peripheral_wards.extend(zone_to_wards.get(zone, []))
    expected_peripheral_wards = sorted(expected_peripheral_wards)
    
    # Check peripheral wards file
    if not PERIPHERAL_WARDS_PATH.exists():
        print_check("Peripheral wards file exists", False)
        return False
    
    try:
        with open(PERIPHERAL_WARDS_PATH, 'r', encoding='utf-8') as f:
            peripheral_data = json.load(f)
        # Handle nested structure: file may have {"peripheral_wards": [...]} or just [...]
        if isinstance(peripheral_data, dict) and 'peripheral_wards' in peripheral_data:
            actual_peripheral_wards = peripheral_data['peripheral_wards']
        elif isinstance(peripheral_data, list):
            actual_peripheral_wards = peripheral_data
        else:
            actual_peripheral_wards = list(peripheral_data.keys()) if isinstance(peripheral_data, dict) else []
        print_check("Peripheral wards JSON parsable", True, f"{len(actual_peripheral_wards)} wards")
    except Exception as e:
        print_check("Peripheral wards JSON parsable", False, str(e))
        return False
    
    # Compare lists
    match = set(expected_peripheral_wards) == set(actual_peripheral_wards)
    print_check("Peripheral wards match expected", match)
    all_passed &= match
    
    if not match:
        missing = set(expected_peripheral_wards) - set(actual_peripheral_wards)
        extra = set(actual_peripheral_wards) - set(expected_peripheral_wards)
        if missing:
            print(f"    Missing: {sorted(missing)[:5]}...")
        if extra:
            print(f"    Extra: {sorted(extra)[:5]}...")
    
    print(f"\n  Peripheral ward count: {len(actual_peripheral_wards)} / {len(mapping['ward_to_zone'])} total")
    
    return all_passed


# === Main Entry Point ===
def main() -> int:
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("  WARD ADJACENCY VERIFICATION UTILITY")
    print("  Standalone verification - No ML/dashboard impact")
    print("=" * 60)
    
    all_passed = True
    
    # Step 1: GeoJSON loading
    step1_passed, gdf = verify_geojson_loading()
    all_passed &= step1_passed
    
    if gdf is None:
        print("\n[!] Cannot proceed without valid GeoJSON")
        return 1
    
    # Step 2: Zone adjacency
    step2_passed, zone_adjacency = verify_zone_adjacency(gdf)
    all_passed &= step2_passed
    
    # Step 3: Ward-zone mapping
    step3_passed, mapping = verify_ward_zone_mapping()
    all_passed &= step3_passed
    
    if mapping is None:
        print("\n[!] Cannot proceed without valid ward-zone mapping")
        return 1
    
    # Step 4: Ward adjacency file
    step4_passed = verify_ward_adjacency_file(zone_adjacency, mapping)
    all_passed &= step4_passed
    
    # Step 5: Peripheral wards
    step5_passed = verify_peripheral_wards(gdf, mapping)
    all_passed &= step5_passed
    
    # Final summary
    print_header("VERIFICATION SUMMARY")
    
    results = [
        ("GeoJSON Loading", step1_passed),
        ("Zone Adjacency", step2_passed),
        ("Ward-Zone Mapping", step3_passed),
        ("Ward Adjacency File", step4_passed),
        ("Peripheral Wards", step5_passed),
    ]
    
    for name, passed in results:
        print_check(name, passed)
    
    overall = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    symbol = "✓" if all_passed else "✗"
    print(f"\n[{symbol}] {overall}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
