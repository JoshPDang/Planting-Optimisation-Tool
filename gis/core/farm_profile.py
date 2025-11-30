from core.geometry_parser import parse_geometry
from core.extract_data import get_rainfall

def build_farm_profile(farm_id: str, geom_raw, years: list[int]):
    """
    Build a rainfall-only profile for a farm.
    """
    # 1. Parse geometry using universal parser
    geom = parse_geometry(geom_raw)

    # 2. Compute rainfall for each year
    annual_rainfall = []
    for y in years:
        value = get_rainfall(geom_raw, y)   # ← uses parse_geometry inside get_rainfall
        annual_rainfall.append({
            "year": y,
            "rainfall_mm": value
        })

    # 3. Return structured profile
    return {
        "farm_id": farm_id,
        "geometry": geom_raw,
        "rainfall": annual_rainfall
    }
