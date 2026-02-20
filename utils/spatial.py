"""
Spatial Intelligence Utility Module

A standalone module for computing spatial relationships between wards.
This module operates independently of ML, prediction, and UI layers.

Features:
- Load and validate ward polygons from GeoJSON
- Compute ward adjacency relationships
- Detect peripheral wards (boundary wards)
- Generate spatial metadata as JSON

This module is opt-in and does not execute automatically.
Run manually: python utils/spatial.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import geopandas as gpd
from shapely.ops import unary_union


def load_wards(geojson_path: str, id_field: str = "name") -> gpd.GeoDataFrame:
    """
    Load zone/ward polygons from a GeoJSON file with validation.
    
    Args:
        geojson_path: Path to the GeoJSON file containing geometries.
        id_field: Name of the field to use as identifier (default: "name" for zones,
                  use "ward_id" if ward-level data is available).
        
    Returns:
        GeoDataFrame with validated geometries and standardized 'area_id' column.
        
    Raises:
        FileNotFoundError: If the GeoJSON file does not exist.
        ValueError: If required fields are missing or data is invalid.
    """
    path = Path(geojson_path)
    
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {geojson_path}")
    
    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as e:
        raise ValueError(f"Failed to parse GeoJSON file: {e}")
    
    if gdf.empty:
        raise ValueError("GeoJSON file contains no features")
    
    # Validate required fields
    if id_field not in gdf.columns:
        available_fields = [c for c in gdf.columns if c != 'geometry']
        raise ValueError(
            f"Missing identifier field: '{id_field}'. "
            f"Available fields: {available_fields}"
        )
    
    if 'geometry' not in gdf.columns:
        raise ValueError("Missing required field: 'geometry'")
    
    # Check for null geometries
    null_geom_count = gdf.geometry.isna().sum()
    if null_geom_count > 0:
        raise ValueError(f"Found {null_geom_count} features with null geometry")
    
    # Check for null identifiers
    null_id_count = gdf[id_field].isna().sum()
    if null_id_count > 0:
        raise ValueError(f"Found {null_id_count} features with null {id_field}")
    
    # Check for duplicate identifiers
    duplicate_count = gdf[id_field].duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate {id_field} values")
    
    # Standardize identifier column name to 'area_id' for internal use
    gdf = gdf.copy()
    gdf['area_id'] = gdf[id_field].astype(str)
    
    # Ensure CRS exists, default to EPSG:4326 if missing
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def compute_adjacency(gdf: gpd.GeoDataFrame) -> Dict[str, List[str]]:
    """
    Compute adjacency relationships between zones/wards.
    
    Two areas are considered adjacent if their geometries touch or intersect
    (excluding containment).
    
    Args:
        gdf: GeoDataFrame with geometries and 'area_id' column.
        
    Returns:
        Dictionary mapping each area_id to a list of adjacent area_ids.
        The relationship is symmetric (if A is adjacent to B, B is adjacent to A).
        No self-references are included.
        
    Note:
        Uses spatial index for efficient computation.
        Does not modify the input GeoDataFrame.
    """
    adjacency: Dict[str, List[str]] = {}
    
    # Initialize empty adjacency lists for all areas
    for area_id in gdf['area_id']:
        adjacency[str(area_id)] = []
    
    # Use spatial index for efficient neighbor lookup
    sindex = gdf.sindex
    
    for idx, row in gdf.iterrows():
        area_id = str(row['area_id'])
        geometry = row.geometry
        
        # Get candidate neighbors using spatial index (bounding box intersection)
        possible_matches_idx = list(sindex.intersection(geometry.bounds))
        
        for match_idx in possible_matches_idx:
            if match_idx == idx:
                # Skip self
                continue
            
            other_row = gdf.iloc[match_idx]
            other_area_id = str(other_row['area_id'])
            
            # Check if geometries actually touch (share boundary) or intersect
            if geometry.touches(other_row.geometry) or geometry.intersects(other_row.geometry):
                # Avoid duplicates and ensure symmetry
                if other_area_id not in adjacency[area_id]:
                    adjacency[area_id].append(other_area_id)
                if area_id not in adjacency[other_area_id]:
                    adjacency[other_area_id].append(area_id)
    
    # Sort adjacency lists for consistent output
    for area_id in adjacency:
        adjacency[area_id] = sorted(adjacency[area_id])
    
    return adjacency


def detect_peripheral_wards(gdf: gpd.GeoDataFrame) -> List[str]:
    """
    Detect peripheral zones/wards whose boundaries touch the district outer boundary.
    
    An area is considered peripheral if its boundary intersects with the 
    outer boundary of the entire district (computed via unary_union).
    
    Args:
        gdf: GeoDataFrame with geometries and 'area_id' column.
        
    Returns:
        List of area_ids that are on the periphery of the district.
        
    Note:
        Does not modify the input GeoDataFrame.
    """
    # Compute the district boundary (union of all geometries)
    district_geometry = unary_union(gdf.geometry)
    district_boundary = district_geometry.boundary
    
    peripheral_areas: List[str] = []
    
    for _, row in gdf.iterrows():
        area_id = str(row['area_id'])
        area_boundary = row.geometry.boundary
        
        # Check if area boundary intersects with district boundary
        if area_boundary.intersects(district_boundary):
            peripheral_areas.append(area_id)
    
    return sorted(peripheral_areas)


def build_spatial_metadata(
    geojson_path: str, 
    output_path: str,
    id_field: str = "name"
) -> Dict[str, Any]:
    """
    Orchestrate spatial computation and generate metadata JSON file.
    
    Computes adjacency relationships, peripheral areas, and centroids,
    then persists all metadata to a JSON file.
    
    Args:
        geojson_path: Path to the GeoJSON file containing geometries.
        output_path: Path where the spatial metadata JSON will be written.
        id_field: Name of the field to use as identifier (default: "name" for zones).
        
    Returns:
        Dictionary containing all computed spatial metadata.
        
    Raises:
        ValueError: If the GeoJSON is invalid or contains no areas.
        
    Output JSON structure:
        {
            "adjacency": {"East Zone": ["North Zone", "Central Zone"], ...},
            "peripheral_wards": ["East Zone", "West Zone", ...],
            "centroids": {"East Zone": {"lat": 11.01, "lon": 76.96}, ...},
            "metadata": {"area_count": 5, "id_field": "name"}
        }
    """
    # Load and validate areas
    gdf = load_wards(geojson_path, id_field=id_field)
    
    area_count = len(gdf)
    if area_count == 0:
        raise ValueError("GeoJSON contains no areas")
    
    # Compute adjacency relationships
    adjacency = compute_adjacency(gdf)
    
    # Detect peripheral areas
    peripheral_areas = detect_peripheral_wards(gdf)
    
    # Compute centroids
    centroids: Dict[str, Dict[str, float]] = {}
    for _, row in gdf.iterrows():
        area_id = str(row['area_id'])
        centroid = row.geometry.centroid
        centroids[area_id] = {
            "lat": round(centroid.y, 6),
            "lon": round(centroid.x, 6)
        }
    
    # Build metadata structure
    spatial_metadata: Dict[str, Any] = {
        "adjacency": adjacency,
        "peripheral_wards": peripheral_areas,
        "centroids": centroids,
        "metadata": {
            "area_count": area_count,
            "id_field": id_field
        }
    }
    
    # Write to output file with safe overwrite
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(spatial_metadata, f, indent=2, ensure_ascii=False)
    
    return spatial_metadata


if __name__ == "__main__":
    # Manual execution entry point
    # Run: python utils/spatial.py
    
    from pathlib import Path
    
    # Determine paths relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    default_geojson = project_root / "geo" / "coimbatore.geojson"
    default_output = project_root / "spatial_metadata.json"
    
    print("=" * 60)
    print("Spatial Intelligence Utility Module")
    print("=" * 60)
    print(f"GeoJSON source: {default_geojson}")
    print(f"Output target:  {default_output}")
    print("-" * 60)
    
    try:
        # Use 'name' field for zone-based GeoJSON
        # Change to 'ward_id' if ward-level data becomes available
        metadata = build_spatial_metadata(
            str(default_geojson),
            str(default_output),
            id_field="name"
        )
        
        area_count = metadata["metadata"]["area_count"]
        id_field = metadata["metadata"]["id_field"]
        peripheral_count = len(metadata["peripheral_wards"])
        
        # Count total adjacency relationships (divide by 2 since symmetric)
        total_edges = sum(len(neighbors) for neighbors in metadata["adjacency"].values()) // 2
        
        print(f"✓ Loaded {area_count} zones (using '{id_field}' field)")
        print(f"✓ Computed {total_edges} adjacency relationships")
        print(f"✓ Identified {peripheral_count} peripheral zones")
        print(f"✓ Generated centroids for all zones")
        print("-" * 60)
        print(f"✓ Spatial metadata saved to: {default_output}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        exit(1)
    except ValueError as e:
        print(f"✗ Validation error: {e}")
        exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        exit(1)
