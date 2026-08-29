"""AquaGrid engine, running on the Bandra drainage data.

What this replaces
------------------
The original `predict_risk_level()` scored each drain on its own: runoff over
its own catchment, divided by its own capacity, plus a penalty for sitting low.
That is a reasonable first cut, and it is wrong in one important way — a drain
does not only receive the rain that falls on it. It also receives whatever the
drains uphill could not carry. Chimbai floods partly because Chimbai's own
gully is choked, and partly because Hill Road and Pali Hill are pushing water
down towards it.

So this module treats the ten drains as a network:

    Tertiary  ->  the nearest lower Secondary or Primary
    Secondary ->  the nearest lower Primary
    Primary   ->  the outfall (Mithi mouth / the sea)

and routes water through it, time step by time step. Each drain carries what
it can actually carry; the rest stays on the surface as ponded water and is
reported as a depth in centimetres, which is a number a person can act on in a
way that a dimensionless score is not.

Nothing here needs a database, an API key or a network connection — except the
optional live-rainfall helper at the bottom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CSV_PATH = "bandra_capacity.csv"

# Fallback runoff coefficient. Each drain now composes its own from the land
# cover of its catchment (see terrain.py); this is only used if that lookup
# fails, and it is the dense-urban value the model used to apply everywhere.
RUNOFF_C = 0.9
HORIZON_MIN = 180           # how far ahead we predict
STEP_MIN = 15               # forecast resolution

# The intensity Mumbai's existing storm water drains were designed for:
# 25 mm/hr at low tide (MCGM / BRIMSTOWAD). Everything the model says is
# ultimately measured against this number, because it is the city's own.
DESIGN_STANDARD_MM_HR = 25.0

# Depth at which a road is treated as flooded when solving for a threshold.
FLOOD_DEPTH_CM = 15.0

# Tide. The design standard is not "25 mm/hr", it is "25 mm/hr AT LOW TIDE",
# and that qualifier is the whole story of Mumbai flooding. A storm drain
# discharges by gravity through an outfall in the sea wall. When the sea is
# higher than the outfall the water has nowhere to go, the drain backs up, and
# rain that would have drained away in an hour sits on the road until the tide
# turns. In August 2025 the Mithi came within 10 cm of its danger mark while
# the harbour line stayed shut for fifteen hours — that is this term at work.
TIDE_LOW_M = 0.5            # outfalls fully clear
TIDE_DROWNED_M = 4.0        # outfalls submerged, gravity discharge gone
DEFAULT_TIDE_M = 1.2        # a middling tide, the everyday case
MIN_OUTFALL_FRACTION = 0.05  # what pumps and floodgates still manage

# One definition of the risk-factor scale, used by the dashboard, the
# navigation page and the glossary. Two copies would drift, and a legend that
# disagrees with the number beside it is worse than no legend.
RISK_FACTOR_BANDS = [
    (70, "Critical", "#7B1E1E",
     "Deep water over a long stretch. Do not travel unless you have to."),
    (45, "Serious", "#C62828",
     "Passable with care for the right vehicle, not for everyone."),
    (20, "Moderate", "#EF6C00",
     "Standing water on part of it. Slower than usual."),
    (0, "Minor", "#2E7D32",
     "Little or no standing water. Normal conditions."),
]


def risk_band(score: float):
    """(name, colour, meaning) for a 0-100 risk factor."""
    for floor, name, colour, meaning in RISK_FACTOR_BANDS:
        if score >= floor:
            return name, colour, meaning
    return RISK_FACTOR_BANDS[-1][1:]


# Surcharge. When the drain below is holding this much water on the surface,
# the drain above it has lost all the capacity it can lose.
SURCHARGE_FULL_CM = 40.0
SURCHARGE_FLOOR = 0.25       # some flow always squeezes through


def outfall_factor(tide_m: float) -> float:
    """How much of its capacity a sea-discharging drain still has.

    Linear between a clear outfall and a drowned one. The real relationship is
    a submerged-orifice curve, but the linear form carries the behaviour that
    matters — discharge collapses as the sea rises — without pretending to a
    precision the tide table here does not have.
    """
    if tide_m <= TIDE_LOW_M:
        return 1.0
    span = TIDE_DROWNED_M - TIDE_LOW_M
    fraction = 1.0 - (tide_m - TIDE_LOW_M) / span
    return max(min(fraction, 1.0), MIN_OUTFALL_FRACTION)


def capacity_now(drain: "Drain", tide_m: float = DEFAULT_TIDE_M) -> float:
    """Effective capacity after silt, and after the tide if it meets the sea."""
    capacity = drain.effective_capacity_m3s
    if drain.downstream is None:
        # Discharges to the sea or to a pump, so the tide reaches it.
        capacity *= outfall_factor(tide_m)
    return capacity

# Depth at which each level begins, in centimetres.
LEVELS = [("SEVERE", 45.0), ("HIGH", 25.0), ("MEDIUM", 10.0), ("LOW", 0.0)]
LEVEL_COLOUR = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "SEVERE": "darkred"}
LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "SEVERE"]

# Share of a catchment that is road surface, which is where water shows up.
ROAD_FRACTION = 0.12
# Ceiling on how wide a single pool gets before depth takes over from spread.
MAX_POND_AREA_M2 = 40_000.0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Drain:
    drain_id: str
    name: str
    kind: str                       # Primary | Secondary | Tertiary
    catchment_m2: float
    capacity_m3s: float             # design capacity, as surveyed
    lat: float
    lon: float
    elevation_m: float
    blockage: float                 # 0.0 clear .. 0.95 choked

    downstream: Optional[str] = None
    pumped: bool = False            # a low point with no gravity outlet
    upstream_area_m2: float = field(default=0.0, init=False)

    @property
    def effective_capacity_m3s(self) -> float:
        """What the drain actually carries, once silt is accounted for.

        Blockage removes cross-section, and discharge scales with area^(5/3),
        so a drain 75% choked carries about 10% of its design flow — not 25%.
        This is the single biggest reason paper capacities mislead.
        """
        b = min(max(self.blockage, 0.0), 0.95)
        return self.capacity_m3s * (1.0 - b) ** (5.0 / 3.0)

    @property
    def total_area_m2(self) -> float:
        """Everything that drains through here, its own land plus uphill."""
        return self.catchment_m2 + self.upstream_area_m2

    @property
    def road_area_m2(self) -> float:
        """The surface an overflow actually spreads across.

        It has to be the whole contributing area, not just this drain's own
        patch. Water arriving from three streets uphill does not pile up on
        one junction; it spreads back across the low ground it came through.
        """
        spread = max(self.total_area_m2 * ROAD_FRACTION, 2_000.0)
        if spread <= MAX_POND_AREA_M2:
            return spread
        # Beyond this, overflow does not spread evenly across the whole
        # catchment - it runs to the low ground and concentrates there. But it
        # does not stop spreading either. A hard cap made a 300-hectare primary
        # catchment pond in the same puddle as a two-hectare lane, and report a
        # metre and a half of water on an ordinary monsoon shower. Square-root
        # growth is the compromise: a catchment ten times larger floods about
        # three times the area, and correspondingly deeper rather than 10x.
        return MAX_POND_AREA_M2 * math.sqrt(spread / MAX_POND_AREA_M2)

    @property
    def hand_m(self) -> Optional[float]:
        """Height above the nearest drainage, in metres, if it was sampled.

        None when hand_values.csv has not been fetched, which is a supported
        state — the two properties below then fall back to elevation.
        """
        try:
            import hand
            return hand.load().get(self.drain_id)
        except Exception:                               # noqa: BLE001
            return None

    @property
    def max_pond_cm(self) -> float:
        """How deep water can get here before it spreads somewhere else.

        A low basin like Chimbai genuinely goes knee-deep and stays there; a
        hillside sheds long before that. Without this ceiling the model stacks
        water up indefinitely and reports depths no street has ever seen.

        Height above the nearest drainage is the right measure of "basin",
        and elevation above sea level is only a proxy for it — a junction 8 m
        above the sea but half a metre above its drain is a basin, and the
        step function below cannot see that. So HAND is used when it has been
        fetched, and the steps remain as the fallback.
        """
        above_drainage = self.hand_m
        if above_drainage is not None:
            import hand
            return hand.max_pond_from_hand(above_drainage)

        if self.elevation_m < 2.0:
            return 150.0
        if self.elevation_m < 4.0:
            return 90.0
        if self.elevation_m < 8.0:
            return 55.0
        return 30.0

    @property
    def runoff_c(self) -> float:
        """Runoff coefficient composed from what this catchment is made of.

        A hillside under trees turns roughly half its rain into surface flow.
        A concrete retail street turns nearly all of it. Using one number for
        both was the model's largest unforced error.
        """
        try:
            import terrain
            return terrain.runoff_coefficient(self.drain_id)
        except Exception:                               # noqa: BLE001
            return RUNOFF_C

    @property
    def retention(self) -> float:
        """How much of an overflow stays put rather than running off further.

        A low, flat basin holds nearly everything that arrives. A hillside
        sheds it. With HAND fetched this is measured against the local
        drainage, which is the quantity that actually governs it; without it,
        elevation above sea level stands in, as it always did.
        """
        above_drainage = self.hand_m
        if above_drainage is not None:
            import hand
            return hand.retention_from_hand(above_drainage)
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
            drain.downstream = None            # primaries discharge to the sea
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

    # How much land drains through each point, including everything above it.
    for drain in drains.values():
        cursor, guard = drain.downstream, 0
        while cursor and guard < 50:
            drains[cursor].upstream_area_m2 += drain.catchment_m2
            cursor = drains[cursor].downstream
            guard += 1


def topo_order(drains: Dict[str, Drain]) -> List[str]:
    """Highest first, so a drain is always processed before its outlet."""
    return sorted(drains, key=lambda d: -drains[d].elevation_m)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def level_for_depth(depth_cm: float) -> str:
    for name, floor in LEVELS:
        if depth_cm >= floor:
            return name
    return "LOW"


def _runoff_m3s(drain: Drain, intensity_mm_hr: float) -> float:
    """Rational method: Q = C i A, with C composed from the land cover."""
    return drain.runoff_c * (intensity_mm_hr / 1000.0 / 3600.0) * drain.catchment_m2


def _step(drains: Dict[str, Drain], order: List[str],
          ponded: Dict[str, float], intensity_mm_hr: float, dt_s: float,
          couple: bool = True, backflow: bool = True,
          tide_m: float = DEFAULT_TIDE_M,
          trace: Optional[Dict[str, dict]] = None) -> None:
    """Advance the whole network one time step.

    `couple` routes what a drain carries into the drain below it. `backflow`
    additionally holds water back above a drain that is already surcharged.
    Turning either off is not a simplification for its own sake — it is how
    the page decomposes a depth into "own rain", "water from uphill" and
    "cannot discharge because downstream is full".
    """
    arriving = {d: 0.0 for d in drains}
    if trace is not None:
        trace.clear()

    # Surcharge, computed from where the water stood at the end of the last
    # step. A drain discharging into one that is already under water cannot
    # discharge freely — the pipe below is running full, so the hydraulic
    # grade line backs up into this one and it loses capacity it nominally
    # has. This is why a clean gully in a good street still goes under: the
    # trunk it feeds is drowned. Without this term the network is only a
    # bookkeeping exercise for volumes and not a network at all.
    throttle: Dict[str, float] = {}
    for did in order:
        below = drains[did].downstream
        if not below or not couple or not backflow:
            throttle[did] = 1.0
            continue
        depth_below = depth_cm(drains[below], ponded.get(below, 0.0))
        throttle[did] = max(1.0 - depth_below / SURCHARGE_FULL_CM, SURCHARGE_FLOOR)

    for did in order:
        drain = drains[did]
        capacity = capacity_now(drain, tide_m) * throttle[did]
        own = _runoff_m3s(drain, intensity_mm_hr)
        from_above = arriving[did] if couple else 0.0
        inflow = own + from_above

        # Water already on the street drains back once there is room for it.
        drain_back = 0.0
        if inflow < capacity and ponded.get(did, 0.0) > 0:
            drain_back = min(capacity - inflow, ponded[did] / dt_s)

        moved = min(inflow + drain_back, capacity)
        spill = max(inflow - capacity, 0.0)

        if couple and drain.downstream:
            arriving[drain.downstream] += moved

        volume = max(ponded.get(did, 0.0) + (spill - drain_back) * dt_s, 0.0)
        # Anything above the ceiling has left this spot overland.
        ceiling = volume_for_depth(drain, drain.max_pond_cm)
        ponded[did] = min(volume, ceiling)

        if trace is not None:
            trace[did] = {
                "own_runoff_m3s": round(own, 3),
                "from_upstream_m3s": round(from_above, 3),
                "inflow_m3s": round(inflow, 3),
                "design_capacity_m3s": round(drain.capacity_m3s, 3),
                "silt_capacity_m3s": round(capacity_now(drain, tide_m), 3),
                "effective_capacity_m3s": round(capacity, 3),
                "surcharge_throttle": round(throttle[did], 3),
                "tide_factor": round(outfall_factor(tide_m), 3)
                                if drain.downstream is None else 1.0,
                "capacity_lost_m3s": round(drain.capacity_m3s - capacity, 3),
                "blockage_loss_m3s": round(
                    drain.capacity_m3s - drain.effective_capacity_m3s, 3),
                "carried_m3s": round(moved, 3),
                "spill_m3s": round(spill, 3),
                "drained_back_m3s": round(drain_back, 3),
                "at_ceiling": volume >= ceiling - 1e-9 and volume > 0,
                "held_by_downstream_m3": 0.0,
            }

    if not backflow:
        return

    # Backflow: if a drain is spilling, the one above it cannot discharge freely.
    for did in reversed(order):
        drain = drains[did]
        if not drain.downstream:
            continue
        below = ponded.get(drain.downstream, 0.0)
        if below <= 0:
            continue
        penalty = min(below / max(capacity_now(drains[drain.downstream], tide_m) * dt_s, 1e-6), 1.0)
        held_back = ponded[did] * 0.15 * penalty
        # Re-apply the ceiling: backflow can push a spot over it otherwise.
        before = ponded[did]
        ponded[did] = min(ponded[did] + held_back,
                          volume_for_depth(drain, drain.max_pond_cm))
        if trace is not None and did in trace:
            trace[did]["held_by_downstream_m3"] = round(ponded[did] - before, 2)
            trace[did]["downstream_surcharged"] = True


def depth_cm(drain: Drain, volume_m3: float) -> float:
    return max(volume_m3 * drain.retention / drain.road_area_m2, 0.0) * 100.0


def volume_for_depth(drain: Drain, target_cm: float) -> float:
    """Inverse of depth_cm, used to cap how much water a spot can hold."""
    return target_cm / 100.0 * drain.road_area_m2 / max(drain.retention, 1e-6)


def forecast(drains: Dict[str, Drain], series: List[Tuple[int, float]],
             couple: bool = True, backflow: bool = True,
             tide_m: float = DEFAULT_TIDE_M) -> Dict[str, List[Tuple[int, float]]]:
    """Depth in cm for every drain at every forecast step."""
    order = topo_order(drains)
    ponded: Dict[str, float] = {d: 0.0 for d in drains}
    out: Dict[str, List[Tuple[int, float]]] = {d: [] for d in drains}
    dt_s = STEP_MIN * 60.0

    for minutes in range(0, HORIZON_MIN + 1, STEP_MIN):
        _step(drains, order, ponded, intensity_at(series, minutes), dt_s,
              couple=couple, backflow=backflow, tide_m=tide_m)
        for did, drain in drains.items():
            out[did].append((minutes, round(depth_cm(drain, ponded[did]), 1)))
    return out


def steady_depths(drains: Dict[str, Drain], intensity_mm_hr: float,
                  duration_min: int = 60,
                  tide_m: float = DEFAULT_TIDE_M,
                  couple: bool = True, backflow: bool = True) -> Dict[str, float]:
    """Depth reached if it rains at a constant intensity for this long."""
    order = topo_order(drains)
    ponded: Dict[str, float] = {d: 0.0 for d in drains}
    dt_s = STEP_MIN * 60.0
    for _ in range(max(1, duration_min // STEP_MIN)):
        _step(drains, order, ponded, intensity_mm_hr, dt_s, tide_m=tide_m,
              couple=couple, backflow=backflow)
    return {d: depth_cm(drains[d], ponded[d]) for d in drains}


def floods_at_mm_hr(drains: Dict[str, Drain], drain_id: str,
                    duration_min: int = 60, target_cm: float = FLOOD_DEPTH_CM,
                    tide_m: float = DEFAULT_TIDE_M) -> float:
    """How much rain before this spot goes under 15 cm.

    Solved by bisection through the whole network rather than in closed form,
    because a drain floods on water that fell somewhere uphill.
    """
    lo, hi = 0.0, 400.0
    if steady_depths(drains, hi, duration_min, tide_m).get(drain_id, 0.0) < target_cm:
        return float("inf")
    for _ in range(30):
        mid = (lo + hi) / 2
        if steady_depths(drains, mid, duration_min, tide_m).get(drain_id, 0.0) >= target_cm:
            hi = mid
        else:
            lo = mid
    return round(hi, 1)


# --------------------------------------------------------------------------- #
# Rainfall
# --------------------------------------------------------------------------- #
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

    Returns {"series": [...], "error": str|None, "peak": float}. On failure it
    returns an empty series and says why, rather than quietly inventing rain.
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
    except Exception as exc:                        # noqa: BLE001
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
        # mm per 15 minutes -> mm per hour
        merged[snapped] = max(merged.get(snapped, 0.0), round(float(mm) * 4, 2))

    series = sorted(merged.items())
    return {
        "series": series,
        "error": None if series else "No precipitation data in the response.",
        "peak": max((v for _, v in series), default=0.0),
    }


# --------------------------------------------------------------------------- #
# One call that gives the dashboard everything
# --------------------------------------------------------------------------- #
def assess(drains: Dict[str, Drain], series: List[Tuple[int, float]],
           at_minutes: int = 0, tide_m: float = DEFAULT_TIDE_M) -> List[dict]:
    """A row per drain: depth now, peak depth, level, and the rain it takes."""
    timeline = forecast(drains, series, tide_m=tide_m)
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
            "Real_Capacity_m3s": round(capacity_now(drain, tide_m), 2),
            "Runoff_C": drain.runoff_c,
            "HAND_m": drain.hand_m,
            "Terrain_source": "HAND" if drain.hand_m is not None else "elevation",
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


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
# Every number on the dashboard is older than it looks, and saying so is part
# of being trustworthy. These are the four places time is lost between rain
# falling and a person reading a depth.
LATENCY_BUDGET = [
    ("Observation to provider", 2, 5,
     "A radar sweep or gauge reading has to be collected and quality-checked "
     "before anyone can fetch it."),
    ("Provider publish interval", 0, 15,
     "The feed updates on a 15-minute grid, so a reading can already be up to "
     "a quarter of an hour old when it appears."),
    ("Our fetch and cache", 0, 5,
     "The forecast is cached for five minutes so the page does not hammer the "
     "API on every interaction."),
    ("Model run", 0, 1,
     "Routing three hours of rain through the network takes well under a "
     "second for ten drains."),
]


def latency_range_min() -> Tuple[int, int]:
    """Best and worst case age of what is on screen, in minutes."""
    return (sum(lo for _, lo, _, _ in LATENCY_BUDGET),
            sum(hi for _, _, hi, _ in LATENCY_BUDGET))


def latency_note() -> str:
    low, high = latency_range_min()
    return (f"Readings are typically {low}-{high} minutes behind real time. "
            "Treat a depth as the situation a few minutes ago, not this second.")


# --------------------------------------------------------------------------- #
# Why is this spot flooding?
# --------------------------------------------------------------------------- #
def diagnose(drains: Dict[str, Drain], series: List[Tuple[int, float]],
             at_minutes: int = 0, tide_m: float = DEFAULT_TIDE_M) -> Dict[str, dict]:
    """Run the model to a moment and hand back the working, not just the answer.

    Every figure the explanation quotes comes from the same step the depth came
    from, so the arithmetic on screen is the arithmetic that was actually done.
    """
    order = topo_order(drains)
    ponded: Dict[str, float] = {d: 0.0 for d in drains}
    dt_s = STEP_MIN * 60.0
    trace: Dict[str, dict] = {}

    for minutes in range(0, HORIZON_MIN + 1, STEP_MIN):
        _step(drains, order, ponded, intensity_at(series, minutes), dt_s,
              tide_m=tide_m, trace=trace)
        if minutes >= at_minutes:
            break

    for did, drain in drains.items():
        entry = trace.setdefault(did, {})
        entry["depth_cm"] = round(depth_cm(drain, ponded[did]), 1)
        entry["ponded_m3"] = round(ponded[did])
        entry["pond_area_m2"] = round(drain.road_area_m2)
        entry["retention"] = round(drain.retention, 2)
        entry["ceiling_cm"] = drain.max_pond_cm
        entry["runoff_c"] = drain.runoff_c
        entry["upstream_names"] = sorted(
            o.name for o in drains.values() if o.downstream == did
        )
    return trace


def explain(drains: Dict[str, Drain], drain_id: str, trace: Dict[str, dict],
            intensity_mm_hr: float) -> List[str]:
    """The chain of reasons this spot is or is not under water, in plain words."""
    drain = drains[drain_id]
    t = trace.get(drain_id, {})
    lines: List[str] = []

    own = t.get("own_runoff_m3s", 0.0)
    above = t.get("from_upstream_m3s", 0.0)
    design = t.get("design_capacity_m3s", drain.capacity_m3s)
    # Capacity after silt only. The backwater and tide losses are reported
    # separately below, so quoting the fully throttled figure here would blame
    # the blockage for all three.
    real = t.get("silt_capacity_m3s", t.get("effective_capacity_m3s", 0.0))
    spill = t.get("spill_m3s", 0.0)
    depth = t.get("depth_cm", 0.0)

    lines.append(
        f"**Rain landing here.** {intensity_mm_hr:.0f} mm/hr over "
        f"{drain.catchment_m2 / 10_000:.0f} ha, and this catchment is "
        f"{round(drain.runoff_c * 100)}% runoff "
        f"({round((1 - drain.runoff_c) * 100)}% soaks in or is intercepted), "
        f"which makes **{own:.2f} m³/s**."
    )

    if above > 0.001:
        feeders = t.get("upstream_names") or []
        who = ", ".join(feeders) if feeders else "drains uphill"
        lines.append(
            f"**Water arriving from above.** {who} cannot hold what falls on "
            f"them either, so a further **{above:.2f} m³/s** arrives here — "
            f"{above / max(own + above, 1e-9) * 100:.0f}% of everything this "
            "drain has to deal with fell somewhere else."
        )
    else:
        lines.append(
            "**Nothing arriving from above.** This is a high point, so it "
            "handles only its own rain."
        )

    if drain.blockage > 0.01:
        lines.append(
            f"**Capacity lost to silt.** Rated at {design:.2f} m³/s, but "
            f"{round(drain.blockage * 100)}% blocked. Flow scales with "
            f"area^(5/3), so it really carries **{real:.2f} m³/s** — "
            f"{round((1 - real / max(design, 1e-9)) * 100)}% gone."
        )
    else:
        lines.append(f"**Clear pipe.** Carrying its full {real:.2f} m³/s.")

    throttle = t.get("surcharge_throttle", 1.0)
    if throttle < 0.99 and drain.downstream:
        below = drains[drain.downstream].name
        lines.append(
            f"**Backed up from below.** {below} is already surcharged, so this "
            f"drain can only discharge at {round(throttle * 100)}% of what it "
            "otherwise would. Nothing about this pipe changed — the one it "
            "empties into is full."
        )

    tide_factor = t.get("tide_factor", 1.0)
    if tide_factor < 0.99:
        lines.append(
            f"**Tide on the outfall.** This drain discharges to the sea, and "
            f"the sea is high enough to leave it {round(tide_factor * 100)}% "
            "of its gravity discharge."
        )

    if spill > 0.001:
        lines.append(
            f"**The surplus.** {spill:.2f} m³/s has nowhere to go and stays on "
            f"the surface. Spread over {t.get('pond_area_m2', 0):,} m² of low "
            f"ground, with {round(t.get('retention', 1) * 100)}% of it staying "
            f"put rather than running off further, that is **{depth:.0f} cm**."
        )
    elif depth > 0.5:
        lines.append(
            f"**Draining down.** Inflow is now within capacity, so the "
            f"{depth:.0f} cm still on the road is falling."
        )
    else:
        lines.append("**No surplus.** Everything arriving fits down the pipe.")

    if depth >= drain.max_pond_cm - 0.5 and depth > 0:
        lines.append(
            f"**At the ceiling.** {drain.max_pond_cm:.0f} cm is as deep as this "
            "spot gets before water spreads overland into neighbouring streets "
            "instead of getting deeper."
        )
    return lines


# --------------------------------------------------------------------------- #
# What the drain below does to the drain above
# --------------------------------------------------------------------------- #
def surcharge_report(drains: Dict[str, Drain], series: List[Tuple[int, float]],
                     at_minutes: int = 0,
                     tide_m: float = DEFAULT_TIDE_M) -> List[dict]:
    """Split each depth into own rain, water from uphill, and backing up.

    Three runs of the same model:

      isolated   every drain gets only its own rain and passes nothing on
      routed     water flows downhill, but a full drain does not hold back
                 the one above it
      full       the real model, including surcharge

    The differences are the two network effects, separated. This is what
    answers "when a drain runs full, what does it do to the one before it" —
    the answer is a number of centimetres, per spot.
    """
    def depths(**kw) -> Dict[str, float]:
        tl = forecast(drains, series, tide_m=tide_m, **kw)
        return {did: dict(pts).get(at_minutes, pts[0][1]) for did, pts in tl.items()}

    isolated = depths(couple=False, backflow=False)
    routed = depths(couple=True, backflow=False)
    full = depths(couple=True, backflow=True)

    rows = []
    for did, drain in drains.items():
        from_uphill = routed[did] - isolated[did]
        from_backup = full[did] - routed[did]
        rows.append({
            "Drain_ID": did,
            "Segment_Name": drain.name,
            "Own_rain_cm": round(isolated[did], 1),
            "From_uphill_cm": round(from_uphill, 1),
            "From_backup_cm": round(from_backup, 1),
            "Total_cm": round(full[did], 1),
            "Network_share_pct": round(
                (from_uphill + from_backup) / max(full[did], 1e-9) * 100
            ) if full[did] > 0.1 else 0,
            "Drains_Into": drains[drain.downstream].name if drain.downstream else
                           ("Pumped out" if drain.pumped else "Sea outfall"),
        })
    rows.sort(key=lambda r: -r["Total_cm"])
    return rows


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
# How wrong the two least certain inputs could plausibly be. Blockage is an
# estimate rather than a CCTV survey; a 15-point error either way is modest.
# Short-range rainfall nowcasts are routinely 40% out on intensity.
BLOCKAGE_UNCERTAINTY = 0.15
RAIN_SCALES = (0.6, 1.0, 1.4)


def _clone_with_blockage(drains: Dict[str, Drain], delta: float) -> Dict[str, Drain]:
    clone: Dict[str, Drain] = {}
    for did, d in drains.items():
        clone[did] = Drain(
            drain_id=d.drain_id, name=d.name, kind=d.kind,
            catchment_m2=d.catchment_m2, capacity_m3s=d.capacity_m3s,
            lat=d.lat, lon=d.lon, elevation_m=d.elevation_m,
            blockage=min(max(d.blockage + delta, 0.0), 0.95),
        )
    _build_topology(clone)
    return clone


def confidence(drains: Dict[str, Drain], series: List[Tuple[int, float]],
               at_minutes: int = 0,
               tide_m: float = DEFAULT_TIDE_M) -> Dict[str, dict]:
    """How much the answer moves when the shaky inputs are shaken.

    Nine runs: blockage low / as-recorded / high, crossed with rainfall at
    60% / 100% / 140% of forecast. If a spot is under water in all nine, the
    model is confident. If it flips between them, it says so instead of
    printing one number and hoping.

    This is an honest measure of *model* uncertainty. It cannot capture what
    the model does not represent at all — a collapsed culvert, a lorry parked
    over a gully — so it is an upper bound on how much to trust a figure, not
    a probability the street will be wet.
    """
    members: Dict[str, List[float]] = {}
    corners: Dict[str, Dict[str, float]] = {}

    for blockage_delta in (-BLOCKAGE_UNCERTAINTY, 0.0, BLOCKAGE_UNCERTAINTY):
        model = drains if blockage_delta == 0.0 else _clone_with_blockage(drains, blockage_delta)
        for scale in RAIN_SCALES:
            scaled = [(m, v * scale) for m, v in series] or flat_series(0.0)
            timeline = forecast(model, scaled, tide_m=tide_m)
            for did, points in timeline.items():
                value = dict(points).get(at_minutes, points[0][1])
                members.setdefault(did, []).append(value)
                corners.setdefault(did, {})[f"b{blockage_delta:+.2f}_r{scale}"] = value

    out: Dict[str, dict] = {}
    for did, values in members.items():
        ordered = sorted(values)
        low, high = ordered[1], ordered[-2]      # trim the two extremes
        middle = ordered[len(ordered) // 2]
        spread = high - low

        # What actually matters is whether the runs agree on the call, not
        # whether they agree on the centimetre. Nine runs that all say "no
        # standing water" are worth trusting even if they range 0-11 cm;
        # nine that straddle the flooding threshold are not, however tight
        # they look. So agreement about the risk band carries most of the
        # weight, and the numeric spread refines it.
        bands = [level_for_depth(v) for v in values]
        agreement = max(bands.count(b) for b in set(bands)) / len(bands)

        relative = spread / max(middle, FLOOD_DEPTH_CM)
        base = 100.0 * (0.65 * agreement + 0.35 * math.exp(-0.7 * relative))

        # Skill decays with lead time. At three hours a 15-minute nowcast is
        # doing considerably worse than it is doing now.
        lead_penalty = max(1.0 - 0.0016 * at_minutes, 0.6)
        pct = int(round(max(min(base * lead_penalty, 96.0), 15.0)))

        corner = corners[did]
        blockage_effect = abs(corner["b+0.15_r1.0"] - corner["b-0.15_r1.0"])
        rain_effect = abs(corner["b+0.00_r1.4"] - corner["b+0.00_r0.6"])

        straddles = low < FLOOD_DEPTH_CM < high
        if agreement >= 0.99:
            driver = (f"All nine runs agree this is "
                      f"{level_for_depth(middle).lower()} risk.")
        elif straddles:
            driver = ("This spot sits right on the line — some runs flood it "
                      "and some do not.")
        elif blockage_effect > rain_effect * 1.3:
            driver = ("Most of the uncertainty is how blocked this drain "
                      "really is, not how hard it will rain.")
        elif rain_effect > blockage_effect * 1.3:
            driver = "Most of the uncertainty is in the rainfall forecast."
        elif spread < 1.0:
            driver = "Every run agrees to within a centimetre."
        else:
            driver = "Rainfall and blockage contribute about equally."

        if at_minutes >= 120:
            driver += f" Confidence is also reduced for a {at_minutes}-minute lead time."

        out[did] = {
            "pct": pct,
            "low_cm": round(low, 1),
            "high_cm": round(high, 1),
            "middle_cm": round(middle, 1),
            "spread_cm": round(spread, 1),
            "agreement": round(agreement, 2),
            "straddles_threshold": straddles,
            "blockage_effect_cm": round(blockage_effect, 1),
            "rain_effect_cm": round(rain_effect, 1),
            "why": driver,
            "band": ("High" if pct >= 75 else "Moderate" if pct >= 50 else "Low"),
        }
    return out


# --------------------------------------------------------------------------- #
# Risk factor
# --------------------------------------------------------------------------- #
def rate_of_rise_cm(points: List[Tuple[int, float]], at_minutes: int = 0) -> float:
    """Centimetres gained per 15 minutes over the half hour after `at_minutes`."""
    depths = dict(points)
    here = depths.get(at_minutes)
    if here is None:
        return 0.0
    later = depths.get(at_minutes + 2 * STEP_MIN)
    if later is None:
        later = depths.get(at_minutes + STEP_MIN, here)
        return round(later - here, 2)
    return round((later - here) / 2.0, 2)


def minutes_above(points: List[Tuple[int, float]], cm: float = FLOOD_DEPTH_CM) -> int:
    return sum(STEP_MIN for _, depth in points if depth >= cm)


def risk_factor(drain: Drain, points: List[Tuple[int, float]],
                at_minutes: int, biggest_area_m2: float) -> dict:
    """Depth is not the same thing as danger.

    Peak percentage answers "how deep". This answers "how bad", which also
    depends on how fast it arrives (whether anyone gets warning), how long it
    stays (whether it strands people rather than delaying them) and how much
    ground drains through the spot (whether closing it closes a district or a
    lane).
    """
    depths = dict(points)
    depth = depths.get(at_minutes, points[0][1])
    peak = max(v for _, v in points)
    rise = rate_of_rise_cm(points, at_minutes)
    stuck = minutes_above(points)

    depth_part = min(peak / 50.0, 1.0)
    rise_part = min(max(rise, 0.0) / 12.0, 1.0)
    persist_part = min(stuck / float(HORIZON_MIN), 1.0)
    exposure_part = min(drain.total_area_m2 / max(biggest_area_m2, 1.0), 1.0)

    score = (0.40 * depth_part + 0.25 * rise_part
             + 0.20 * persist_part + 0.15 * exposure_part) * 100.0
    score = round(score)

    drivers = sorted([
        ("Depth", 0.40 * depth_part,
         f"peaks at {peak:.0f} cm"),
        ("Speed of onset", 0.25 * rise_part,
         f"rising {rise:.1f} cm every 15 min" if rise > 0 else "not rising"),
        ("How long it lasts", 0.20 * persist_part,
         f"over {FLOOD_DEPTH_CM:.0f} cm for {stuck} min of the next three hours"
         if stuck else "clears within the forecast"),
        ("Ground it serves", 0.15 * exposure_part,
         f"{drain.total_area_m2 / 10_000:.0f} ha drains through here"),
    ], key=lambda item: -item[1])

    band, colour, meaning = risk_band(score)

    top = [d for d in drivers if d[1] > 0.02][:2]
    if top:
        why = f"{band}: mainly " + " and ".join(f"{label.lower()} ({text})"
                                                for label, _, text in top) + "."
    else:
        why = "Minor: nothing here is close to flooding."

    return {
        "score": score,
        "band": band,
        "colour": colour,
        "meaning": meaning,
        "depth_cm": round(depth, 1),
        "peak_cm": round(peak, 1),
        "rise_cm_per_15min": rise,
        "minutes_above_flood": stuck,
        "drivers": drivers,
        "why": why,
    }
