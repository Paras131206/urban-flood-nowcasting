"""AquaGrid engine, running on the Bandra drainage data.

The original predict_risk_level() scored each drain on its own: runoff over
its own catchment, divided by its own capacity, plus a penalty for sitting low.
That is a reasonable first cut, and it is wrong in one important way - a drain
does not only receive the rain that falls on it. It also receives whatever the
drains uphill could not carry.

So this module treats the ten drains as a network:

    Tertiary  ->  the nearest lower Secondary or Primary
    Secondary ->  the nearest lower Primary
    Primary   ->  the outfall (Mithi mouth / the sea)

and routes water through it, step by step. Each drain carries what it can
actually carry; the rest stays on the surface as ponded water, reported as a
depth in centimetres - a number a person can act on in a way that a
dimensionless score is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CSV_PATH = "bandra_capacity.csv"

RUNOFF_C = 0.9              # dense urban, mostly paved
HORIZON_MIN = 180           # how far ahead we predict
STEP_MIN = 15               # forecast resolution

LEVELS = [("SEVERE", 45.0), ("HIGH", 25.0), ("MEDIUM", 10.0), ("LOW", 0.0)]
LEVEL_COLOUR = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "SEVERE": "darkred"}
LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "SEVERE"]

ROAD_FRACTION = 0.12
MAX_POND_AREA_M2 = 40_000.0


@dataclass
class Drain:
    drain_id: str
    name: str
    kind: str
    catchment_m2: float
    capacity_m3s: float
    lat: float
    lon: float
    elevation_m: float
    blockage: float

    downstream: Optional[str] = None
    pumped: bool = False
    upstream_area_m2: float = field(default=0.0, init=False)

    @property
    def effective_capacity_m3s(self) -> float:
        """What the drain actually carries, once silt is accounted for.

        Blockage removes cross-section, and discharge scales with area^(5/3),
        so a drain 75% choked carries about 10% of its design flow, not 25%.
        This is the single biggest reason paper capacities mislead.
        """
        b = min(max(self.blockage, 0.0), 0.95)
        return self.capacity_m3s * (1.0 - b) ** (5.0 / 3.0)

    @property
    def total_area_m2(self) -> float:
        return self.catchment_m2 + self.upstream_area_m2

    @property
    def road_area_m2(self) -> float:
        """The surface an overflow actually spreads across.

        Overflow does not spread evenly across a whole catchment; it runs to
        the low ground and concentrates there, so the pond area saturates.
        """
        spread = self.total_area_m2 * ROAD_FRACTION
        return min(max(spread, 2_000.0), MAX_POND_AREA_M2)

    @property
    def max_pond_cm(self) -> float:
        """How deep water gets here before it spreads somewhere else."""
        if self.elevation_m < 2.0:
            return 150.0
        if self.elevation_m < 4.0:
            return 90.0
        if self.elevation_m < 8.0:
            return 55.0
        return 30.0

    @property
    def retention(self) -> float:
        """How much of an overflow stays put rather than running on.

        Elevation is the only terrain signal in the CSV, so it stands in for
        slope here; a DEM would do this properly.
        """
        return min(max(1.0 - self.elevation_m / 20.0, 0.15), 1.0)


RANK = {"Tertiary": 0, "Secondary": 1, "Primary": 2}


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def load_drains(csv_path: str = CSV_PATH) -> Dict[str, Drain]:
    import csv as _csv

    drains: Dict[str, Drain] = {}
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            drains[row["Drain_ID"]] = Drain(
                drain_id=row["Drain_ID"],
                name=row["Segment_Name"].replace("_", " "),
                kind=row["Drain_Type"],
                catchment_m2=float(row["Catchment_Area_sqm"]),
                capacity_m3s=float(row["Max_Flow_Capacity_m3s"]),
                lat=float(row["Latitude"]),
                lon=float(row["Longitude"]),
                elevation_m=float(row["Elevation_m"]),
                blockage=float(row["Blockage_Pct"]) / 100.0,
            )
    _build_topology(drains)
    return drains


def _build_topology(drains: Dict[str, Drain]) -> None:
    """Work out what drains into what, from rank, elevation and distance."""
    for drain in drains.values():
        candidates = [
            other for other in drains.values()
            if other.drain_id != drain.drain_id
            and RANK[other.kind] >= RANK[drain.kind]
            and other.elevation_m < drain.elevation_m - 0.05
        ]
        if drain.kind == "Primary":
            drain.downstream = None
            continue
        if not candidates:
            # Nowhere downhill to go. In Bandra that means a pump, which is
            # exactly what sits at the real low points.
            drain.downstream = None
            drain.pumped = True
            continue
        nearest = min(
            candidates,
            key=lambda o: haversine_m((drain.lat, drain.lon), (o.lat, o.lon)),
        )
        drain.downstream = nearest.drain_id

    for drain in drains.values():
        cursor, guard = drain.downstream, 0
        while cursor and guard < 50:
            drains[cursor].upstream_area_m2 += drain.catchment_m2
            cursor = drains[cursor].downstream
            guard += 1


def topo_order(drains: Dict[str, Drain]) -> List[str]:
    """Highest first, so a drain is processed before its outlet."""
    return sorted(drains, key=lambda d: -drains[d].elevation_m)


def level_for_depth(depth_cm: float) -> str:
    for name, floor in LEVELS:
        if depth_cm >= floor:
            return name
    return "LOW"


def _runoff_m3s(drain: Drain, intensity_mm_hr: float) -> float:
    """Rational method: Q = C i A."""
    return RUNOFF_C * (intensity_mm_hr / 1000.0 / 3600.0) * drain.catchment_m2


def _step(drains: Dict[str, Drain], order: List[str],
          ponded: Dict[str, float], intensity_mm_hr: float, dt_s: float) -> None:
    arriving = {d: 0.0 for d in drains}

    for did in order:
        drain = drains[did]
        capacity = drain.effective_capacity_m3s
        inflow = _runoff_m3s(drain, intensity_mm_hr) + arriving[did]

        drain_back = 0.0
        if inflow < capacity and ponded.get(did, 0.0) > 0:
            drain_back = min(capacity - inflow, ponded[did] / dt_s)

        moved = min(inflow + drain_back, capacity)
        spill = max(inflow - capacity, 0.0)

        if drain.downstream:
            arriving[drain.downstream] += moved

        volume = max(ponded.get(did, 0.0) + (spill - drain_back) * dt_s, 0.0)
        ponded[did] = min(volume, volume_for_depth(drain, drain.max_pond_cm))

    # Backflow: a spilling drain stops the one above it discharging freely.
    for did in reversed(order):
        drain = drains[did]
        if not drain.downstream:
            continue
        below = ponded.get(drain.downstream, 0.0)
        if below <= 0:
            continue
        penalty = min(below / max(drains[drain.downstream].effective_capacity_m3s * dt_s, 1e-6), 1.0)
        held_back = ponded[did] * 0.15 * penalty
        ponded[did] = min(ponded[did] + held_back,
                          volume_for_depth(drain, drain.max_pond_cm))


def depth_cm(drain: Drain, volume_m3: float) -> float:
    return max(volume_m3 * drain.retention / drain.road_area_m2, 0.0) * 100.0


def volume_for_depth(drain: Drain, target_cm: float) -> float:
    return target_cm / 100.0 * drain.road_area_m2 / max(drain.retention, 1e-6)


def forecast(drains: Dict[str, Drain], series: List[Tuple[int, float]]) -> Dict[str, List[Tuple[int, float]]]:
    order = topo_order(drains)
    ponded: Dict[str, float] = {d: 0.0 for d in drains}
    out: Dict[str, List[Tuple[int, float]]] = {d: [] for d in drains}
    dt_s = STEP_MIN * 60.0

    for minutes in range(0, HORIZON_MIN + 1, STEP_MIN):
        _step(drains, order, ponded, intensity_at(series, minutes), dt_s)
        for did, drain in drains.items():
            out[did].append((minutes, round(depth_cm(drain, ponded[did]), 1)))
    return out


def steady_depths(drains: Dict[str, Drain], intensity_mm_hr: float,
                  duration_min: int = 60) -> Dict[str, float]:
    order = topo_order(drains)
    ponded: Dict[str, float] = {d: 0.0 for d in drains}
    dt_s = STEP_MIN * 60.0
    for _ in range(max(1, duration_min // STEP_MIN)):
        _step(drains, order, ponded, intensity_mm_hr, dt_s)
    return {d: depth_cm(drains[d], ponded[d]) for d in drains}


def floods_at_mm_hr(drains: Dict[str, Drain], drain_id: str,
                    duration_min: int = 60, target_cm: float = 15.0) -> float:
    """How much rain before this spot goes under 15 cm.

    Solved by bisection through the whole network rather than in closed form,
    because a drain floods on water that fell somewhere uphill.
    """
    lo, hi = 0.0, 400.0
    if steady_depths(drains, hi, duration_min).get(drain_id, 0.0) < target_cm:
        return float("inf")
    for _ in range(30):
        mid = (lo + hi) / 2
        if steady_depths(drains, mid, duration_min).get(drain_id, 0.0) >= target_cm:
            hi = mid
        else:
            lo = mid
    return round(hi, 1)


def intensity_at(series: List[Tuple[int, float]], minutes: int) -> float:
    if not series:
        return 0.0
    if minutes <= series[0][0]:
        return series[0][1]
    if minutes >= series[-1][0]:
        return series[-1][1]
    for (m0, v0), (m1, v1) in zip(series, series[1:]):
        if m0 <= minutes <= m1:
            return v0 + (v1 - v0) * ((minutes - m0) / (m1 - m0) if m1 != m0 else 0)
    return series[-1][1]


def flat_series(intensity_mm_hr: float) -> List[Tuple[int, float]]:
    return [(m, intensity_mm_hr) for m in range(0, HORIZON_MIN + 1, STEP_MIN)]


def live_series(lat: float = 19.0544, lon: float = 72.8402) -> dict:
    """Real 15-minute rainfall forecast from Open-Meteo. No API key needed.

    On failure it returns an empty series and says why, rather than quietly
    inventing rain.
    """
    from datetime import datetime

    try:
        import requests
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "minutely_15": "precipitation",
                "forecast_minutely_15": 16,
                "timezone": "auto",
            },
            timeout=8,
        )
        response.raise_for_status()
        block = response.json().get("minutely_15", {})
        times, values = block.get("time", []), block.get("precipitation", [])
    except Exception as exc:
        return {"series": [], "error": f"{exc.__class__.__name__}: {exc}", "peak": 0.0}

    now = datetime.now().astimezone()
    merged: Dict[int, float] = {}
    for iso, mm in zip(times, values):
        if mm is None:
            continue
        stamp = datetime.fromisoformat(iso)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=now.tzinfo)
        minutes = (stamp - now).total_seconds() / 60.0
        if minutes < -STEP_MIN or minutes > HORIZON_MIN:
            continue
        # Providers publish on quarter-hour boundaries while "now" sits inside
        # one, so snap to our grid rather than claiming minute precision.
        snapped = int(round(minutes / STEP_MIN) * STEP_MIN)
        snapped = min(max(snapped, 0), HORIZON_MIN)
        merged[snapped] = max(merged.get(snapped, 0.0), round(float(mm) * 4, 2))

    series = sorted(merged.items())
    return {
        "series": series,
        "error": None if series else "No precipitation data in the response.",
        "peak": max((v for _, v in series), default=0.0),
    }


def assess(drains: Dict[str, Drain], series: List[Tuple[int, float]],
           at_minutes: int = 0) -> List[dict]:
    """A row per drain: depth now, peak depth, level, and where it flows."""
    timeline = forecast(drains, series)
    rows = []
    for did, drain in drains.items():
        points = timeline[did]
        depth_now = dict(points).get(at_minutes, points[0][1])
        peak_min, peak = max(points, key=lambda p: p[1])
        rows.append({
            "Drain_ID": did,
            "Segment_Name": drain.name,
            "Drain_Type": drain.kind,
            "Latitude": drain.lat,
            "Longitude": drain.lon,
            "Elevation_m": drain.elevation_m,
            "Blockage_Pct": round(drain.blockage * 100),
            "Design_Capacity_m3s": drain.capacity_m3s,
            "Real_Capacity_m3s": round(drain.effective_capacity_m3s, 2),
            "Drains_To": drains[drain.downstream].name if drain.downstream else
                         ("PUMPED" if drain.pumped else "Outfall"),
            "Upstream_Area_ha": round(drain.upstream_area_m2 / 10_000, 1),
            "Depth_cm": round(depth_now, 1),
            "Peak_Depth_cm": round(peak, 1),
            "Minutes_To_Peak": peak_min,
            "Risk_Level": level_for_depth(depth_now),
            "Peak_Level": level_for_depth(peak),
            "Color": LEVEL_COLOUR[level_for_depth(depth_now)],
            "Timeline": points,
        })
    rows.sort(key=lambda r: -r["Peak_Depth_cm"])
    return rows
