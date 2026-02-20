"""
Ward-Level Spatial Metadata Generator

Generates ward-level spatial metadata by combining:
1. Zone-level GeoJSON boundaries
2. Ward-to-zone mapping

This creates ward-level approximations for:
- Adjacency (based on zone adjacency + intra-zone connectivity)
- Peripheral status (wards in peripheral zones)
- Centroids (zone centroid as ward centroid approximation)

Run manually: python utils/generate_ward_spatial.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set

import geopandas as gpd
from shapely.ops import unary_union


def load_zone_geojson(geojson_path: str) -> gpd.GeoDataFrame:
    """Load zone-level GeoJSON with validation."""
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError("GeoJSON contains no features")
    
    if 'name' not in gdf.columns:
        raise ValueError("GeoJSON missing 'name' field for zone identification")
    
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def load_ward_zone_mapping(mapping_path: str) -> Dict[str, Any]:
    """Load ward-zone mapping from JSON."""
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_zone_adjacency(gdf: gpd.GeoDataFrame) -> Dict[str, List[str]]:
    """Compute which zones are adjacent to each other."""
    adjacency: Dict[str, Set[str]] = {str(row['name']): set() for _, row in gdf.iterrows()}
    
    for i, row_i in gdf.iterrows():
        zone_i = str(row_i['name'])
        geom_i = row_i.geometry
        
        for j, row_j in gdf.iterrows():
            if i >= j:
                continue
            
            zone_j = str(row_j['name'])
            geom_j = row_j.geometry
            
            if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                adjacency[zone_i].add(zone_j)
                adjacency[zone_j].add(zone_i)
    
    return {k: sorted(list(v)) for k, v in adjacency.items()}


def compute_zone_centroids(gdf: gpd.GeoDataFrame) -> Dict[str, Dict[str, float]]:
    """Compute centroid for each zone."""
    centroids = {}
    for _, row in gdf.iterrows():
        zone_name = str(row['name'])
        centroid = row.geometry.centroid
        centroids[zone_name] = {
            "lat": round(centroid.y, 6),
            "lon": round(centroid.x, 6)
        }
    return centroids


def detect_peripheral_zones(gdf: gpd.GeoDataFrame) -> List[str]:
    """Detect zones touching the district boundary."""
    district_geometry = unary_union(gdf.geometry)
    district_boundary = district_geometry.boundary
    
    peripheral = []
    for _, row in gdf.iterrows():
        zone_name = str(row['name'])
        if row.geometry.boundary.intersects(district_boundary):
            peripheral.append(zone_name)
    
    return sorted(peripheral)


def generate_ward_adjacency(
    zone_adjacency: Dict[str, List[str]],
    zone_to_wards: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Generate ward-level adjacency based on zone adjacency.
    
    Rules:
    1. Wards within the same zone are potentially adjacent to each other
    2. Wards in adjacent zones may be adjacent (we mark all cross-zone pairs)
    
    This is an approximation since we don't have exact ward boundaries.
    """
    ward_adjacency: Dict[str, List[str]] = {}
    
    # Initialize all wards
    for zone, wards in zone_to_wards.items():
        for ward in wards:
            ward_adjacency[ward] = []
    
    # For each zone, determine potential neighbors
    for zone, wards in zone_to_wards.items():
        adjacent_zones = zone_adjacency.get(zone, [])
        
        # Wards in adjacent zones are potential neighbors
        adjacent_zone_wards = []
        for adj_zone in adjacent_zones:
            adjacent_zone_wards.extend(zone_to_wards.get(adj_zone, []))
        
        # Each ward in this zone can neighbor wards in adjacent zones
        for ward in wards:
            # Add wards from adjacent zones as potential neighbors
            ward_adjacency[ward].extend(adjacent_zone_wards)
    
    # Sort and deduplicate
    for ward in ward_adjacency:
        ward_adjacency[ward] = sorted(list(set(ward_adjacency[ward])))
    
    return ward_adjacency


def generate_ward_centroids(
    zone_centroids: Dict[str, Dict[str, float]],
    zone_to_wards: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Assign zone centroid to each ward as approximation.
    
    Note: This is a rough approximation. All wards in a zone
    share the same centroid (the zone's centroid).
    """
    ward_centroids = {}
    for zone, wards in zone_to_wards.items():
        zone_centroid = zone_centroids.get(zone, {"lat": 0.0, "lon": 0.0})
        for ward in wards:
            ward_centroids[ward] = zone_centroid.copy()
    return ward_centroids


def generate_peripheral_wards(
    peripheral_zones: List[str],
    zone_to_wards: Dict[str, List[str]]
) -> List[str]:
    """
    Identify peripheral wards (wards in peripheral zones).
    
    Note: This approximates that all wards in a peripheral zone
    are themselves peripheral. True peripheral detection would
    require ward-level boundaries.
    """
    peripheral_wards = []
    for zone in peripheral_zones:
        peripheral_wards.extend(zone_to_wards.get(zone, []))
    return sorted(peripheral_wards)


def build_ward_spatial_metadata(
    geojson_path: str,
    mapping_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Generate all ward-level spatial metadata files.
    
    Outputs:
    - data/ward_adjacency.json (ward-level adjacency)
    - data/peripheral_wards.json (ward-level peripheral list)
    - spatial_metadata.json (comprehensive metadata)
    """
    # Load input data
    gdf = load_zone_geojson(geojson_path)
    mapping = load_ward_zone_mapping(mapping_path)
    
    zone_to_wards = mapping['zone_to_wards']
    ward_to_zone = mapping['ward_to_zone']
    
    # Compute zone-level spatial properties
    zone_adjacency = compute_zone_adjacency(gdf)
    zone_centroids = compute_zone_centroids(gdf)
    peripheral_zones = detect_peripheral_zones(gdf)
    
    # Generate ward-level equivalents
    ward_adjacency = generate_ward_adjacency(zone_adjacency, zone_to_wards)
    ward_centroids = generate_ward_centroids(zone_centroids, zone_to_wards)
    peripheral_wards = generate_peripheral_wards(peripheral_zones, zone_to_wards)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save ward_adjacency.json
    adjacency_file = output_path / "ward_adjacency.json"
    with open(adjacency_file, 'w', encoding='utf-8') as f:
        json.dump(ward_adjacency, f, indent=2, ensure_ascii=False)
    
    # Save peripheral_wards.json
    peripheral_file = output_path / "peripheral_wards.json"
    peripheral_data = {
        "peripheral_wards": peripheral_wards,
        "metadata": {
            "ward_count": len(ward_to_zone),
            "peripheral_count": len(peripheral_wards)
        }
    }
    with open(peripheral_file, 'w', encoding='utf-8') as f:
        json.dump(peripheral_data, f, indent=2, ensure_ascii=False)
    
    # Save comprehensive spatial_metadata.json at project root
    spatial_metadata = {
        "adjacency": ward_adjacency,
        "peripheral_wards": peripheral_wards,
        "centroids": ward_centroids,
        "ward_to_zone": ward_to_zone,
        "zone_adjacency": zone_adjacency,
        "zone_centroids": zone_centroids,
        "peripheral_zones": peripheral_zones,
        "metadata": {
            "area_count": len(ward_to_zone),
            "zone_count": len(zone_to_wards),
            "id_field": "ward_id",
            "approximation_note": "Ward centroids approximated from zone centroids"
        }
    }
    
    # Save to project root
    project_root = output_path.parent
    metadata_file = project_root / "spatial_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(spatial_metadata, f, indent=2, ensure_ascii=False)
    
    return {
        "ward_adjacency": ward_adjacency,
        "peripheral_wards": peripheral_wards,
        "ward_centroids": ward_centroids,
        "files_created": [
            str(adjacency_file),
            str(peripheral_file),
            str(metadata_file)
        ]
    }


if __name__ == "__main__":
    # Manual execution entry point
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    default_geojson = project_root / "geo" / "coimbatore.geojson"
    default_mapping = project_root / "data" / "ward_zone_mapping.json"
    default_output = project_root / "data"
    
    print("=" * 60)
    print("Ward-Level Spatial Metadata Generator")
    print("=" * 60)
    print(f"Zone GeoJSON: {default_geojson}")
    print(f"Ward Mapping: {default_mapping}")
    print(f"Output Dir:   {default_output}")
    print("-" * 60)
    
    try:
        result = build_ward_spatial_metadata(
            str(default_geojson),
            str(default_mapping),
            str(default_output)
        )
        
        ward_count = len(result["ward_adjacency"])
        peripheral_count = len(result["peripheral_wards"])
        
        # Compute average neighbors
        total_neighbors = sum(len(n) for n in result["ward_adjacency"].values())
        avg_neighbors = total_neighbors / ward_count if ward_count > 0 else 0
        
        print(f"✓ Generated adjacency for {ward_count} wards")
        print(f"✓ Average neighbors per ward: {avg_neighbors:.1f}")
        print(f"✓ Peripheral wards identified: {peripheral_count}")
        print("-" * 60)
        print("Files created:")
        for f in result["files_created"]:
            print(f"  - {f}")
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
