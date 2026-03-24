"""
demo_riparian.py

Quick demo of the riparian zone detection feature (US-018).
Run from the gis/ folder:

    python demo_riparian.py
"""

from core.riparian import get_riparian_flags

DIVIDER = "-" * 55


def print_result(label: str, result: dict):
    status = "🔴 RIPARIAN" if result["is_riparian"] else "🟢 NOT RIPARIAN"
    dist = result["distance_to_nearest_waterway_m"]
    print(f"  {label}")
    print(f"    → {status}")
    print(f"    → Distance to nearest waterway: {dist}m")
    print()


# ============================================================================
# 1. Single point check
# ============================================================================
print(DIVIDER)
print("1. SINGLE POINT CHECK")
print(DIVIDER)

point = (-8.569, 126.676)
result = get_riparian_flags(point)
print_result(f"Point {point}", result)


# ============================================================================
# 2. Polygon check (farm boundary)
# ============================================================================
print(DIVIDER)
print("2. FARM BOUNDARY (POLYGON) CHECK")
print(DIVIDER)

farm_boundary = [[
    (-8.55, 125.57),
    (-8.56, 125.57),
    (-8.56, 125.58),
    (-8.55, 125.58),
    (-8.55, 125.57),  # closed ring
]]
result = get_riparian_flags(farm_boundary)
print_result("Farm polygon", result)


# ============================================================================
# 3. Effect of different buffer distances
# ============================================================================
print(DIVIDER)
print("3. SAME FARM, DIFFERENT BUFFER DISTANCES")
print(DIVIDER)

for buffer in [10, 30, 50, 100]:
    r = get_riparian_flags(point, buffer_m=buffer)
    flag = "RIPARIAN" if r["is_riparian"] else "not riparian"
    print(f"  buffer={buffer:>4}m  →  {flag}  (distance={r['distance_to_nearest_waterway_m']}m)")
print()


# ============================================================================
# 4. Batch — multiple farms at once
# ============================================================================
print(DIVIDER)
print("4. BATCH — MULTIPLE FARMS")
print(DIVIDER)

farms = [
    {"id": 1, "name": "Hillside Farm",   "geometry": (-8.55, 125.57)},
    {"id": 2, "name": "River Edge Farm", "geometry": (-8.569, 126.676)},
    {"id": 3, "name": "Plateau Farm",    "geometry": (-8.60, 125.90)},
]

print(f"  {'ID':<4} {'Name':<20} {'Riparian':<12} {'Distance (m)'}")
print(f"  {'-'*4} {'-'*20} {'-'*12} {'-'*12}")

for farm in farms:
    r = get_riparian_flags(farm["geometry"])
    flag = "YES" if r["is_riparian"] else "NO"
    dist = r["distance_to_nearest_waterway_m"]
    print(f"  {farm['id']:<4} {farm['name']:<20} {flag:<12} {dist}")

print()


# ============================================================================
# 5. Integration with build_farm_profile
# ============================================================================
print(DIVIDER)
print("5. FULL FARM PROFILE (riparian fields highlighted)")
print(DIVIDER)

from core.gee_client import init_gee
from core.farm_profile import build_farm_profile

try:
    init_gee()
    profile = build_farm_profile(geometry=(-8.569, 126.676), farm_id=1, year=2024)

    if profile["status"] == "success":
        print(f"  Farm ID:              {profile['id']}")
        print(f"  Rainfall:             {profile['rainfall_mm']} mm")
        print(f"  Elevation:            {profile['elevation_m']} m")
        print(f"  Coastal:              {profile['coastal']}")
        print()
        print(f"  *** is_riparian:      {profile['is_riparian']} ***")
        print(f"  *** distance:         {profile['distance_to_nearest_waterway_m']} m ***")
    else:
        print(f"  Profile failed: {profile.get('error')}")

except Exception as e:
    print(f"  GEE not available ({e})")
    print("  (Riparian check still works independently — see examples 1–4 above)")

print()
print(DIVIDER)
print("Done.")
print(DIVIDER)
