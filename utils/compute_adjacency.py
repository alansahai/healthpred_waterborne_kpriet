"""
Ward/Zone Adjacency Computation Utility

A standalone utility module that computes adjacency relationships from GeoJSON
and persists the result as JSON. Operates independently of ML, prediction, and UI layers.

This module is opt-in and does not execute automatically.
Run manually: python utils/compute_adjacency.py

Output: data/ward_adjacency.json
"""

import json
from pathlib import Path
from typing import Dict, List

import geopandas as gpd


def load_geojson(path: str, id_field: str = "name") -> gpd.GeoDataFrame:
    """
    Load GeoJSON file with validation and CRS normalization.
    
    Args:
        path: Path to the GeoJSON file.
        id_field: Name of the identifier field (default: "name" for zones,
                  use "ward_id" when ward-level data is available).
        
    Returns:
        GeoDataFrame with validated geometries in EPSG:4326.
        
    Raises:
        FileNotFoundError: If the GeoJSON file does not exist.
        ValueError: If required fields are missing or data is invalid.
    """
    geojson_path = Path(path)
    
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")
    
    if not geojson_path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        raise ValueError(f"Failed to parse GeoJSON file: {e}")
    
    if gdf.empty:
        raise ValueError("GeoJSON file contains no features")
    
    # Validate required columns
    if id_field not in gdf.columns:
        available_cols = [c for c in gdf.columns if c != 'geometry']
        raise ValueError(
            f"Missing required column: '{id_field}'. "
            f"Available columns: {available_cols}"
        )
    
    if 'geometry' not in gdf.columns:
        raise ValueError("Missing required column: 'geometry'")
    
    # Check for null geometries
    null_geom_count = gdf.geometry.isna().sum()
    if null_geom_count > 0:
        raise ValueError(f"Found {null_geom_count} features with null geometry")
    
    # Check for null identifiers
    null_id_count = gdf[id_field].isna().sum()
    if null_id_count > 0:
        raise ValueError(f"Found {null_id_count} features with null {id_field}")
    
    # Standardize to 'ward_id' column for internal processing
    gdf = gdf.copy()
    gdf['ward_id'] = gdf[id_field].astype(str)
    
    # Handle CRS - ensure EPSG:4326
    if gdf.crs is None:
        # Missing CRS - set to EPSG:4326
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        # Different CRS - convert to EPSG:4326
        gdf = gdf.to_crs("EPSG:4326")
    
    return gdf


def compute_adjacency(gdf: gpd.GeoDataFrame) -> Dict[str, List[str]]:
    """
    Compute adjacency graph from GeoDataFrame.
    
    Two areas are neighbors if their boundaries touch:
        A ~ B ⟺ Boundary(A) ∩ Boundary(B) ≠ ∅
    
    Uses spatial index (R-tree) for efficient neighbor lookup.
    
    Args:
        gdf: GeoDataFrame with 'ward_id' and 'geometry' columns.
        
    Returns:
        Dictionary mapping each ward_id to sorted list of adjacent ward_ids.
        All wards included, even if isolated (empty neighbor list).
        Adjacency is symmetric: if A neighbors B, then B neighbors A.
        No self-references.
    """
    adjacency: Dict[str, List[str]] = {}
    
    # Initialize all wards with empty neighbor lists (handles isolated wards)
    for ward_id in gdf['ward_id']:
        adjacency[str(ward_id)] = []
    
    # Use spatial index for efficient bounding box queries
    sindex = gdf.sindex
    
    for idx, row in gdf.iterrows():
        ward_id = str(row['ward_id'])
        geometry = row.geometry
        
        # Query spatial index for candidate neighbors (bounding box intersection)
        candidate_indices = list(sindex.intersection(geometry.bounds))
        
        for candidate_idx in candidate_indices:
            # Skip self-comparison
            if candidate_idx == idx:
                continue
            
            candidate_row = gdf.iloc[candidate_idx]
            candidate_ward_id = str(candidate_row['ward_id'])
            
            # Check if boundaries actually touch
            # touches() returns True if geometries have at least one point in common
            # but interiors do not intersect
            if geometry.touches(candidate_row.geometry):
                # Add neighbor (avoid duplicates)
                if candidate_ward_id not in adjacency[ward_id]:
                    adjacency[ward_id].append(candidate_ward_id)
                # Ensure symmetry
                if ward_id not in adjacency[candidate_ward_id]:
                    adjacency[candidate_ward_id].append(ward_id)
    
    # Sort all neighbor lists for consistent output
    for ward_id in adjacency:
        adjacency[ward_id] = sorted(adjacency[ward_id])
    
    return adjacency


def save_adjacency(adjacency: Dict[str, List[str]], output_path: str) -> None:
    """
    Persist adjacency dictionary to JSON file.
    
    Args:
        adjacency: Dictionary mapping ward_id to list of neighbor ward_ids.
        output_path: Path where JSON file will be written.
        
    Raises:
        IOError: If write operation fails.
    """
    output_file = Path(output_path)
    
    # Create parent directory if missing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(adjacency, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to write adjacency file: {e}")


def build_ward_adjacency(
    geojson_path: str, 
    output_path: str,
    id_field: str = "name"
) -> Dict[str, List[str]]:
    """
    Orchestrate adjacency computation and persistence.
    
    Loads GeoJSON, computes adjacency graph, and saves to JSON.
    
    Args:
        geojson_path: Path to input GeoJSON file.
        output_path: Path for output JSON file.
        id_field: Name of the identifier field in GeoJSON properties.
        
    Returns:
        Computed adjacency dictionary.
        
    Raises:
        FileNotFoundError: If GeoJSON file not found.
        ValueError: If GeoJSON is invalid.
        IOError: If output write fails.
    """
    # Load and validate
    gdf = load_geojson(geojson_path, id_field=id_field)
    
    # Compute adjacency
    adjacency = compute_adjacency(gdf)
    
    # Persist result
    save_adjacency(adjacency, output_path)
    
    return adjacency


if __name__ == "__main__":
    # Manual execution entry point
    # Run: python utils/compute_adjacency.py
    
    # Determine paths relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    default_geojson = project_root / "geo" / "coimbatore.geojson"
    default_output = project_root / "data" / "ward_adjacency.json"
    
    print("=" * 60)
    print("Ward/Zone Adjacency Computation Utility")
    print("=" * 60)
    print(f"Input:  {default_geojson}")
    print(f"Output: {default_output}")
    print("-" * 60)
    
    try:
        # Use 'name' field for zone-based GeoJSON
        # Change to 'ward_id' when ward-level boundaries are available
        adjacency = build_ward_adjacency(
            str(default_geojson),
            str(default_output),
            id_field="name"
        )
        
        total_wards = len(adjacency)
        total_edges = sum(len(neighbors) for neighbors in adjacency.values())
        avg_neighbors = total_edges / total_wards if total_wards > 0 else 0
        
        # Count isolated wards (no neighbors)
        isolated_count = sum(1 for neighbors in adjacency.values() if len(neighbors) == 0)
        
        print("Adjacency graph created successfully.")
        print(f"Total wards/zones: {total_wards}")
        print(f"Average neighbors: {avg_neighbors:.1f}")
        if isolated_count > 0:
            print(f"Isolated areas: {isolated_count}")
        print("-" * 60)
        print(f"✓ Saved to: {default_output}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        exit(1)
    except ValueError as e:
        print(f"✗ Validation error: {e}")
        exit(1)
    except IOError as e:
        print(f"✗ Write error: {e}")
        exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        exit(1)
