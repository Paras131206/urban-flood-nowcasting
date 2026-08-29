"""Height Above Nearest Drainage, from ASF's global 30 m dataset.

What HAND is
------------
For any point, HAND is the vertical distance to the nearest point on the
drainage network that the point flows into, measured *along the flow path*
rather than as the crow flies. It normalises topography to the local drainage
rather than to sea level.

That distinction is the whole reason this file exists. Two of the least
defensible numbers in flood_engine.py — `retention` and `max_pond_cm` — were
both computed from elevation **above sea level**, when the question they are
actually asking is "how far above the nearest drainage is this?". Those are
different quantities:

    a junction 8 m above sea level but 0.4 m above its drain  -> floods
    a junction 3 m above sea level but 3 m above its drain     -> does not

Absolute elevation cannot tell those apart. HAND is precisely the number that
can, and it is a published, peer-reviewed terrain descriptor (Rennó et al.
2008; Nobre et al. 2011) rather than a coefficient invented for this project.

Where the data comes from
-------------------------
ASF (Alaska Satellite Facility, a NASA DAAC) publishes a global 30 m HAND
raster derived from the 2021 Copernicus GLO-30 DEM, under CC0. Two ways in:

  * **ImageServer identify** — a point query, no key, no GDAL. Ten drains means
    ten small requests. This is what `fetch_hand.py` uses, and it is the right
    tool for sampling a handful of locations.
  * **The COG tiles on S3** — `glo-30-hand`, 1 x 1 degree Cloud Optimized
    GeoTIFFs. The right tool if you ever want HAND as a continuous raster for
    the whole map, and it needs rasterio.

What HAND does NOT know
-----------------------
It is derived from natural topography. It has never heard of a storm drain.
Mumbai floods because an engineered pipe network is undersized, silted and
tide-locked, and none of that appears in a DEM. So HAND on its own would be a
*worse* predictor here than the network model already in this project.

The combination is what is worth having: HAND says how flood-prone the ground
is, the network model says whether the pipes can cope. Terrain plus hydraulics
beats either alone, and saying so plainly is more defensible than implying a
NASA dataset settles the question.

Two more caveats worth keeping in view. GLO-30 is derived from a surface
model, so in dense low-rise Mumbai it partly sees rooftops rather than
streets, and urban HAND is noisier than rural. And 30 m pixels are coarse for
a single junction — fine for a catchment, approximate for a kerb.
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Optional, Tuple

HAND_CSV = "hand_values.csv"

# ASF's ArcGIS ImageServer. No key, no account, no GDAL.
IDENTIFY_URL = ("https://gis.asf.alaska.edu/arcgis/rest/services/"
                "GlobalHAND/GLO30_HAND/ImageServer/identify")

# The same data as Cloud Optimized GeoTIFFs, if you ever want a full raster.
COG_BASE = "https://glo-30-hand.s3.amazonaws.com/v1/2021"

# How quickly ponding potential falls away as ground rises above its drainage.
# An exponential rather than the old step function: there is no physical reason
# for a cliff edge at exactly 2 m, and a step made the model jump when a drain's
# elevation was nudged by a tenth of a metre.
RETENTION_SCALE_M = 6.0      # retention falls to 1/e at 6 m above drainage
POND_SCALE_M = 5.0           # so does the ponding ceiling, a little faster
MAX_POND_AT_DRAINAGE_CM = 150.0
MIN_POND_CM = 25.0
MIN_RETENTION = 0.15


# --------------------------------------------------------------------------- #
# Turning HAND into the two things the engine needs
# --------------------------------------------------------------------------- #
def retention_from_hand(hand_m: float) -> float:
    """How much of an overflow stays put rather than running off further.

    At drainage level there is nowhere for it to go, so nearly all of it
    stays. Well above drainage it sheds freely.
    """
    hand_m = max(hand_m, 0.0)
    return round(max(math.exp(-hand_m / RETENTION_SCALE_M), MIN_RETENTION), 3)


def max_pond_from_hand(hand_m: float) -> float:
    """How deep water gets here before it spreads overland instead.

    A spot sitting at the level of its own drainage is a basin and genuinely
    goes knee-deep. A spot ten metres above one cannot hold that much water no
    matter how hard it rains — it runs off first.
    """
    hand_m = max(hand_m, 0.0)
    depth = MAX_POND_AT_DRAINAGE_CM * math.exp(-hand_m / POND_SCALE_M)
    return round(max(depth, MIN_POND_CM), 1)


def describe(hand_m: float) -> str:
    if hand_m < 1.0:
        return "at drainage level — a basin, water has nowhere to go"
    if hand_m < 3.0:
        return "barely above drainage — floods readily"
    if hand_m < 8.0:
        return "moderately above drainage — ponds, then drains"
    return "well above drainage — sheds water"


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def sample_point(lat: float, lon: float, timeout: float = 20.0) -> dict:
    """HAND in metres at one coordinate, or an explanation of why not.

    Returns {"ok": bool, "hand_m": float|None, "error": str|None}. It never
    raises and never guesses — a missing value has to stay missing, or the
    model silently starts trusting a number nobody measured.
    """
    try:
        import requests
    except ImportError:
        return {"ok": False, "hand_m": None,
                "error": "The requests package is not installed."}

    geometry = json.dumps({"x": float(lon), "y": float(lat),
                           "spatialReference": {"wkid": 4326}})
    try:
        response = requests.get(
            IDENTIFY_URL,
            params={"geometry": geometry,
                    "geometryType": "esriGeometryPoint",
                    "returnGeometry": "false",
                    "f": "json"},
            timeout=timeout,
            headers={"User-Agent": "urban-flood-nowcasting/1.0 (SIH 2026)"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:                            # noqa: BLE001
        return {"ok": False, "hand_m": None,
                "error": f"{exc.__class__.__name__}: {exc}"}

    if "error" in payload:
        return {"ok": False, "hand_m": None,
                "error": str(payload["error"])[:200]}

    raw = payload.get("value")
    if raw in (None, "", "NoData"):
        return {"ok": False, "hand_m": None,
                "error": "No HAND value at this location (outside coverage, "
                         "or over water)."}
    try:
        return {"ok": True, "hand_m": round(float(raw), 2), "error": None}
    except (TypeError, ValueError):
        return {"ok": False, "hand_m": None,
                "error": f"Unexpected value from the service: {raw!r}"}


def cog_url_for(lat: float, lon: float) -> str:
    """The 1x1 degree COG tile covering a point, for the raster path."""
    northing = f"{'N' if lat >= 0 else 'S'}{int(abs(math.floor(lat))):02d}_00"
    easting = f"{'E' if lon >= 0 else 'W'}{int(abs(math.floor(lon))):03d}_00"
    return (f"{COG_BASE}/Copernicus_DSM_COG_10_{northing}_{easting}_HAND.tif")


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
# Read once, not once per property access. `retention` and `max_pond_cm` are
# evaluated inside every timestep for every drain, so an uncached file read
# there took a forecast from 6 ms to 36 ms and the nine-run confidence ensemble
# to nearly 300 ms — on every rerun of the dashboard. Keyed on modification
# time so re-fetching the CSV still takes effect without a restart.
_CACHE: Dict[str, Tuple[float, Dict[str, float]]] = {}


def load(path: Optional[str] = None) -> Dict[str, float]:
    """Cached HAND per drain, or {} if it has not been fetched.

    Absent is a supported state, not an error: the engine falls back to the
    elevation heuristic and the app says which one it is using.

    `path` resolves at call time rather than through a default argument.
    Python binds defaults once, at definition, so `path=HAND_CSV` froze
    whatever the module constant was when this file was first imported and
    quietly ignored any later change to it.
    """
    path = path or HAND_CSV
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        _CACHE.pop(path, None)
        return {}

    cached = _CACHE.get(path)
    if cached and cached[0] == stamp:
        return cached[1]

    out: Dict[str, float] = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                value = (row.get("hand_m") or "").strip()
                if value:
                    out[row["drain_id"]] = float(value)
    except Exception:                                   # noqa: BLE001
        return {}

    _CACHE[path] = (stamp, out)
    return out


def forget(path: Optional[str] = None) -> None:
    """Drop the cache. Only needed by tests that rewrite the file quickly."""
    if path is None:
        _CACHE.clear()
    else:
        _CACHE.pop(path, None)


def save(values: Dict[str, dict], path: Optional[str] = None) -> None:
    path = path or HAND_CSV
    forget(path)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["drain_id", "name", "lat", "lon",
                            "elevation_m", "hand_m", "source"])
        writer.writeheader()
        for drain_id, row in sorted(values.items()):
            writer.writerow({"drain_id": drain_id, **row})


def available(path: Optional[str] = None) -> bool:
    return bool(load(path))


# --------------------------------------------------------------------------- #
# Comparing the two
# --------------------------------------------------------------------------- #
def comparison(drains, path: Optional[str] = None) -> list:
    """Side by side: what the elevation heuristic says, what HAND says.

    This is the honest way to present the upgrade — not "we use NASA data" but
    "here is the number that changed, and by how much".
    """
    values = load(path)
    rows = []
    for drain_id, drain in drains.items():
        hand_m = values.get(drain_id)
        row = {
            "Drain_ID": drain_id,
            "Segment_Name": drain.name,
            "Elevation_m": drain.elevation_m,
            "HAND_m": hand_m,
            "Retention_elevation": round(
                min(max(1.0 - drain.elevation_m / 20.0, MIN_RETENTION), 1.0), 3),
            "Retention_HAND": retention_from_hand(hand_m) if hand_m is not None else None,
            "Max_pond_elevation_cm": _legacy_max_pond(drain.elevation_m),
            "Max_pond_HAND_cm": max_pond_from_hand(hand_m) if hand_m is not None else None,
            "Ground": describe(hand_m) if hand_m is not None else "not sampled",
        }
        rows.append(row)
    return rows


def _legacy_max_pond(elevation_m: float) -> float:
    """The elevation step function, kept so the two can be compared."""
    if elevation_m < 2.0:
        return 150.0
    if elevation_m < 4.0:
        return 90.0
    if elevation_m < 8.0:
        return 55.0
    return 30.0
