"""
One-time script to load farm_boundaries.gpkg, clean geometries,
and export a minimal farm boundaries with:

- farm_id
- polygon_geometry (true Shapely geometry)
"""

import geopandas as gpd
from shapely.geometry import shape, mapping
import os


def to_2d(geom):
    """Drop Z coordinates (3D → 2D)."""
    try:
        return shape(mapping(geom))
    except Exception:
        return geom


def fix_geometry(geom):
    """Fix invalid geometries using buffer(0)."""
    try:
        if not geom.is_valid:
            return geom.buffer(0)
        return geom
    except Exception:
        return geom.buffer(0)


def build_farm_geolocation():
    base_dir = os.path.dirname(__file__)
    gpkg_path = os.path.join(base_dir, "../assets/farm_boundaries.gpkg")
    docs_dir = os.path.join(base_dir, "../docs")

    print("Loading:", gpkg_path)

    gdf = gpd.read_file(gpkg_path)

    # Ensure CRS exists
    if gdf.crs is None:
        raise ValueError("Missing CRS. GPKG must be in EPSG:4326.")

    # Convert to EPSG:4326 if needed
    if gdf.crs.to_epsg() != 4326:
        print(f"Reprojecting {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)

    # Clean geometry (remove Z + fix issues)
    gdf["polygon_geometry"] = gdf.geometry.apply(to_2d).apply(fix_geometry)

    # Build final GeoDataFrame with true geometry
    df = gpd.GeoDataFrame(
        {
            "farm_id": gdf["Name"],  # or whichever is the unique ID
            "polygon_geometry": gdf["polygon_geometry"],
        },
        geometry="polygon_geometry",
        crs="EPSG:4326",
    )

    os.makedirs(docs_dir, exist_ok=True)

    output_file = os.path.join(docs_dir, "farm_boundaries_minimal.gpkg")
    df.to_file(output_file, driver="GPKG")

    print("\nPreview:")
    print(df.head())

    print("\nSaved:")
    print(f"- {output_file}")


if __name__ == "__main__":
    build_farm_geolocation()
