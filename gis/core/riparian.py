"""
Riparian zone detection using Timor-Leste Waterways dataset.

"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd

from config.settings import WATERWAYS_PATH

# ============================================================================
# CONSTANTS
# ============================================================================

# UTM Zone 52S — correct projected CRS for Timor-Leste.
# Distance calculations require a projected CRS; WGS84 (EPSG:4326) is NOT suitable.
_UTM_EPSG = 32752


# ============================================================================
# AC1 — DATASET INGESTION AND INDEXING
# ============================================================================


@lru_cache(maxsize=1)
def _load_waterways() -> gpd.GeoDataFrame:
    """
    Load the Timor-Leste waterways lines dataset and build a spatial index.

    - Called once per process; result is cached for all subsequent queries.
    - Reprojects from WGS84 → UTM 52S so distances are in metres.
    - Spatial index (.sindex) is built on load; GeoPandas uses STRtree internally.

    Raises:
        FileNotFoundError: If WATERWAYS_PATH does not exist.
    """
    path = Path(WATERWAYS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Waterways dataset not found at '{WATERWAYS_PATH}'.\n"
            "Download 'hotosm_tls_waterways_lines_geojson.zip' from:\n"
            "  MS Teams → Planting Optimisation Tool → Datasets → GIS → Timor Leste Waterways\n"
            "Then set WATERWAYS_PATH in your environment or config/settings.py."
        )

    # GeoPandas reads GeoPackage (.gpkg), GeoJSON, and Shapefile natively.
    # For .gpkg with multiple layers, we pass layer=0 to always get the first layer.
    kwargs = {"layer": 0} if path.suffix.lower() == ".gpkg" else {}
    gdf = gpd.read_file(path, **kwargs).to_crs(epsg=_UTM_EPSG)

    # Trigger spatial index construction. GeoPandas builds an STRtree lazily;
    # accessing .sindex here forces it to build during warm-up, not at query time.
    _ = gdf.sindex

    return gdf


# ============================================================================
# GEOMETRY CONVERSION
# ============================================================================


def _to_shapely_projected(geometry):
    """
    Convert any supported geometry input to a projected Shapely geometry (UTM 52S).

    Accepts the same formats as core/geometry_parser.py:
      - (lat, lon) tuple           → Point
      - [(lat, lon), ...]          → MultiPoint
      - [[(lat, lon), ...], ...]   → Polygon (list of rings)
      - Shapely geometry object    → used directly (assumed WGS84)

    Args:
        geometry: Raw geometry input.

    Returns:
        Shapely geometry projected to EPSG:32752.

    Raises:
        ValueError: If the geometry format is not recognised.
    """
    from shapely.geometry import MultiPoint, Point, Polygon

    # (lat, lon) tuple → Point
    if isinstance(geometry, tuple) and len(geometry) == 2:
        lat, lon = geometry
        geom = Point(lon, lat)  # Shapely convention: (x=lon, y=lat)

    # List of (lat, lon) tuples → MultiPoint
    elif isinstance(geometry, list) and all(isinstance(p, tuple) and len(p) == 2 for p in geometry):
        geom = MultiPoint([(lon, lat) for lat, lon in geometry])

    # List of rings → Polygon
    elif isinstance(geometry, list) and all(isinstance(r, list) for r in geometry):
        outer = [(lon, lat) for lat, lon in geometry[0]]
        holes = [[(lon, lat) for lat, lon in ring] for ring in geometry[1:]]
        geom = Polygon(outer, holes)

    # Already a Shapely geometry (assumed WGS84)
    elif hasattr(geometry, "geom_type"):
        geom = geometry

    else:
        raise ValueError(f"Unsupported geometry format for riparian check: {type(geometry)}. Expected (lat, lon) tuple, list of tuples, list of rings, or Shapely geometry.")

    # Project WGS84 → UTM 52S
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    return gdf.to_crs(epsg=_UTM_EPSG).geometry.iloc[0]


# ============================================================================
# AC2 + AC3 — GEOSPATIAL INTERSECTION CHECK / PUBLIC API
# ============================================================================


def get_riparian_flags(
    geometry,
    buffer_m: float | None = None,
) -> dict:
    """
    Check if a farm geometry falls within a riparian zone.

    Works for both existing farms and candidate new farm boundaries —
    the function is stateless with respect to whether the farm exists yet.

    Uses a two-phase spatial query:
      1. Spatial index (STRtree) to filter candidate waterway features by bounding box.
      2. Exact distance calculation only against those candidates.

    Args:
        geometry:  Farm geometry — same formats as geometry_parser.py.
                   (lat/lon tuple, list of tuples, list of rings, or Shapely geometry)
        buffer_m:  Riparian buffer distance in metres.
                   Defaults to settings.RIPARIAN_BUFFER_M if not provided.

    Returns:
        {
            "is_riparian": bool | None,
            "distance_to_nearest_waterway_m": float | None,
        }

        Returns None values (not False) if the waterways dataset is unavailable,
        so callers can distinguish "not riparian" from "check not performed".
    """
    from config.settings import RIPARIAN_BUFFER_M

    buffer_m = buffer_m if buffer_m is not None else RIPARIAN_BUFFER_M

    try:
        waterways = _load_waterways()
        farm_geom = _to_shapely_projected(geometry)

        # Phase 1: bounding-box filter via spatial index (fast)
        candidate_indices = list(
            waterways.sindex.intersection(
                farm_geom.buffer(buffer_m * 10).bounds  # generous bbox — exact check follows
            )
        )

        if candidate_indices:
            # Phase 2: exact distance against spatially nearby candidates only
            candidates = waterways.iloc[candidate_indices]
            distance_m = float(farm_geom.distance(candidates.geometry.union_all()))
        else:
            # No candidates within wide bbox — compute against full dataset as fallback
            # (handles edge cases where farm is far from all waterways)
            distance_m = float(farm_geom.distance(waterways.geometry.union_all()))

        is_riparian = distance_m <= buffer_m

        return {
            "is_riparian": bool(is_riparian),
            "distance_to_nearest_waterway_m": round(distance_m, 1),
        }

    except FileNotFoundError as e:
        # Dataset not available — return None sentinel rather than crash the profile build.
        # Callers should treat None as "check not performed", not as False.
        import warnings

        warnings.warn(str(e), stacklevel=2)
        return {
            "is_riparian": None,
            "distance_to_nearest_waterway_m": None,
        }
