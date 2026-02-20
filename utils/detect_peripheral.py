"""
Peripheral Ward/Zone Detection Utility

A standalone utility that identifies areas touching the outer city boundary.
Operates independently of ML, prediction, and UI layers.

This module is opt-in and does not execute automatically.
Run manually: python utils/detect_peripheral.py

Output: data/peripheral_wards.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


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


def compute_district_boundary(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """
    Compute the outer boundary of the district.
    
    Steps:
        1. Compute union of all ward polygons
        2. Extract the boundary (outer ring)
    
    Args:
        gdf: GeoDataFrame with ward geometries.
        
    Returns:
        Geometry representing the district's outer boundary.
        
    Note:
        Does not modify input GeoDataFrame.
    """
    # Compute union of all ward geometries
    district_union = unary_union(gdf.geometry)
    
    # Extract outer boundary
    district_boundary = district_union.boundary
    
    return district_boundary


def detect_peripheral_wards(
    gdf: gpd.GeoDataFrame, 
    district_boundary: BaseGeometry
) -> List[str]:
    """
    Identify wards whose boundaries touch the district outer boundary.
    
    A ward is peripheral if:
        Boundary(Ward) ∩ Boundary(District) ≠ ∅
    
    Args:
        gdf: GeoDataFrame with 'ward_id' and 'geometry' columns.
        district_boundary: Geometry of the district's outer boundary.
        
    Returns:
        Sorted list of ward_ids that are on the periphery.
        
    Note:
        Pure boundary comparison - no geometry mutation.
        Handles multipolygons robustly via .intersects().
    """
    peripheral_wards: List[str] = []
    
    for _, row in gdf.iterrows():
        ward_id = str(row['ward_id'])
        ward_geometry = row.geometry
        
        # Get ward boundary (handles both Polygon and MultiPolygon)
        ward_boundary = ward_geometry.boundary
        
        # Check if ward boundary intersects district boundary
        if ward_boundary.intersects(district_boundary):
            peripheral_wards.append(ward_id)
    
    return sorted(peripheral_wards)


def save_peripheral_list(
    peripheral_list: List[str], 
    total_ward_count: int,
    output_path: str
) -> None:
    """
    Persist peripheral ward list to JSON file with metadata.
    
    Args:
        peripheral_list: List of peripheral ward_ids.
        total_ward_count: Total number of wards in dataset.
        output_path: Path where JSON file will be written.
        
    Raises:
        IOError: If write operation fails.
        
    Output format:
        {
            "peripheral_wards": ["W01", "W05", ...],
            "metadata": {
                "ward_count": 100,
                "peripheral_count": 14
            }
        }
    """
    output_file = Path(output_path)
    
    # Create parent directory if missing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data: Dict[str, Any] = {
        "peripheral_wards": peripheral_list,
        "metadata": {
            "ward_count": total_ward_count,
            "peripheral_count": len(peripheral_list)
        }
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to write peripheral wards file: {e}")


def build_peripheral_metadata(
    geojson_path: str, 
    output_path: str,
    id_field: str = "name"
) -> Dict[str, Any]:
    """
    Orchestrate peripheral ward detection and persistence.
    
    Loads GeoJSON, computes district boundary, detects peripheral wards,
    and saves result to JSON.
    
    Args:
        geojson_path: Path to input GeoJSON file.
        output_path: Path for output JSON file.
        id_field: Name of the identifier field in GeoJSON properties.
        
    Returns:
        Dictionary containing peripheral ward list and metadata.
        
    Raises:
        FileNotFoundError: If GeoJSON file not found.
        ValueError: If GeoJSON is invalid.
        IOError: If output write fails.
    """
    # Load and validate
    gdf = load_geojson(geojson_path, id_field=id_field)
    
    total_ward_count = len(gdf)
    
    # Compute district boundary
    district_boundary = compute_district_boundary(gdf)
    
    # Detect peripheral wards
    peripheral_wards = detect_peripheral_wards(gdf, district_boundary)
    
    # Persist result
    save_peripheral_list(peripheral_wards, total_ward_count, output_path)
    
    return {
        "peripheral_wards": peripheral_wards,
        "metadata": {
            "ward_count": total_ward_count,
            "peripheral_count": len(peripheral_wards)
        }
    }


if __name__ == "__main__":
    # Manual execution entry point
    # Run: python utils/detect_peripheral.py
    
    # Determine paths relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    default_geojson = project_root / "geo" / "coimbatore.geojson"
    default_output = project_root / "data" / "peripheral_wards.json"
    
    print("=" * 60)
    print("Peripheral Ward/Zone Detection Utility")
    print("=" * 60)
    print(f"Input:  {default_geojson}")
    print(f"Output: {default_output}")
    print("-" * 60)
    
    try:
        # Use 'name' field for zone-based GeoJSON
        # Change to 'ward_id' when ward-level boundaries are available
        result = build_peripheral_metadata(
            str(default_geojson),
            str(default_output),
            id_field="name"
        )
        
        total_wards = result["metadata"]["ward_count"]
        peripheral_count = result["metadata"]["peripheral_count"]
        
        print("Peripheral detection complete.")
        print(f"Total wards/zones: {total_wards}")
        print(f"Peripheral wards/zones detected: {peripheral_count}")
        
        if peripheral_count > 0:
            print("-" * 60)
            print("Peripheral areas:")
            for ward_id in result["peripheral_wards"]:
                print(f"  - {ward_id}")
        
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
