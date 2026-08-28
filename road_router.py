"""Real road navigation that steers around flooding.

Routes come from OSRM, the open-source engine behind a lot of OpenStreetMap
routing. It is free, needs no key, follows actual streets and will hand back
several alternative routes for the same journey.

The part that matters here is what we do with those alternatives. OSRM ranks
them by driving time and knows nothing about water. So each candidate route is
sampled along its length, scored against the predicted flood depth near every
sample, and the one that stays driest is chosen — even when it is slower. If
the fastest route is flooded and a slower one is not, that is precisely the
decision worth making, and it is the decision a navigation app cannot make
today because nobody tells it where the water is.

If OSRM cannot be reached, the caller falls back to the straight-line planner
in route_planner.py, and the page says so rather than pretending.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from flood_engine import Drain, haversine_m
from route_planner import advice_for, risk_pct

OSRM = "https://router.project-osrm.org/route/v1/driving"

# How far from a drain a stretch of road still counts as affected by it.
INFLUENCE_M = 400.0
# Distance between the points we sample along a route when scoring it.
SAMPLE_EVERY_M = 120.0
REQUEST_TIMEOUT_S = 12.0

# Worst-point risk at which a route stops being slow and starts being one
# nobody should attempt. 90% of the 50 cm full-risk scale is 45 cm — over a
# car's door sill and past the wading depth of everything but a truck.
IMPASSABLE_RISK_PCT = 90.0

# A sanity clamp on the arithmetic, not a judgement about the road: however
# much water there is, the estimate never exceeds this multiple of the
# free-flow driving time.
MAX_SLOWDOWN_FACTOR = 4.0


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def fetch_routes(origin: Tuple[float, float], destination: Tuple[float, float]) -> dict:
    """Ask OSRM for the driving routes between two (lat, lon) points.

    Returns {"routes": [...], "error": str | None}. Each route carries real
    road geometry, a distance in metres, a duration in seconds, and OSRM's
    turn-by-turn steps.
    """
    try:
        import requests
    except ImportError:
        return {"routes": [], "error": "The requests package is not installed."}

    # OSRM wants lon,lat.
    path = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
    url = f"{OSRM}/{path}"

    try:
        response = requests.get(
            url,
            params={
                "alternatives": "true",
                "overview": "full",
                "geometries": "geojson",
                "steps": "true",
            },
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": "urban-flood-nowcasting/1.0 (SIH 2026 project)"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:                        # noqa: BLE001
        return {"routes": [], "error": f"{exc.__class__.__name__}: {exc}"}

    if payload.get("code") != "Ok" or not payload.get("routes"):
        return {"routes": [], "error": payload.get("message", "No route found.")}

    routes = []
    for index, route in enumerate(payload["routes"]):
        coords = route.get("geometry", {}).get("coordinates", [])
        routes.append({
            "index": index,
            "coords_latlon": [(lat, lon) for lon, lat in coords],
            "distance_m": route.get("distance", 0.0),
            "duration_s": route.get("duration", 0.0),
            "steps": _flatten_steps(route),
        })
    return {"routes": routes, "error": None}


def _flatten_steps(route: dict) -> List[dict]:
    """OSRM nests steps inside legs. Flatten them into plain instructions."""
    out: List[dict] = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            manoeuvre = step.get("maneuver", {})
            out.append({
                "instruction": _describe(manoeuvre, step.get("name", "")),
                "road": step.get("name", "") or "unnamed road",
                "distance_m": step.get("distance", 0.0),
                "location": tuple(reversed(manoeuvre.get("location", [0, 0]))),
            })
    return out


TURNS = {
    "left": "Turn left", "right": "Turn right",
    "sharp left": "Sharp left", "sharp right": "Sharp right",
    "slight left": "Bear left", "slight right": "Bear right",
    "straight": "Continue straight", "uturn": "Make a U-turn",
}


def _describe(manoeuvre: dict, road: str) -> str:
    kind = manoeuvre.get("type", "")
    modifier = manoeuvre.get("modifier", "")
    where = f" onto {road}" if road else ""

    if kind == "depart":
        return f"Start out{' on ' + road if road else ''}"
    if kind == "arrive":
        return "Arrive at your destination"
    if kind == "roundabout":
        exit_no = manoeuvre.get("exit")
        return f"At the roundabout take exit {exit_no}{where}" if exit_no else f"Take the roundabout{where}"
    if kind in ("merge", "fork", "end of road", "new name", "continue", "turn"):
        return f"{TURNS.get(modifier, 'Continue')}{where}"
    return f"{TURNS.get(modifier, 'Continue')}{where}"


# --------------------------------------------------------------------------- #
# Scoring a route against the flood forecast
# --------------------------------------------------------------------------- #
def _sample(coords: List[Tuple[float, float]], every_m: float) -> List[Tuple[float, float]]:
    """Thin a dense polyline down to points roughly every_m apart."""
    if not coords:
        return []
    picked = [coords[0]]
    travelled = 0.0
    for previous, current in zip(coords, coords[1:]):
        travelled += haversine_m(previous, current)
        if travelled >= every_m:
            picked.append(current)
            travelled = 0.0
    if picked[-1] != coords[-1]:
        picked.append(coords[-1])
    return picked


def score_route(
    coords: List[Tuple[float, float]],
    drains: Dict[str, Drain],
    depths: Dict[str, float],
    threshold_pct: float,
) -> dict:
    """How wet is this route, and where.

    Each sampled point takes the risk of the nearest drain, fading to nothing
    at INFLUENCE_M away. A road 50 m from a flooded gully is wet; a road 400 m
    away is not.
    """
    samples = _sample(coords, SAMPLE_EVERY_M)
    if not samples:
        return {"max_pct": 0.0, "flooded_m": 0, "hotspots": [], "risk_at": [], "samples": []}

    by_name = {d.name: d for d in drains.values()}
    risk_at: List[float] = []
    hotspots: Dict[str, float] = {}

    for point in samples:
        worst_here = 0.0
        culprit = None
        for name, drain in by_name.items():
            distance = haversine_m(point, (drain.lat, drain.lon))
            if distance > INFLUENCE_M:
                continue
            fade = 1.0 - (distance / INFLUENCE_M)
            here = risk_pct(depths.get(name, 0.0)) * fade
            if here > worst_here:
                worst_here, culprit = here, name
        risk_at.append(round(worst_here, 1))
        if culprit and worst_here >= threshold_pct:
            hotspots[culprit] = max(hotspots.get(culprit, 0.0), worst_here)

    # Roughly how much of the route is above the threshold.
    over = sum(1 for r in risk_at if r >= threshold_pct)
    flooded_m = over * SAMPLE_EVERY_M

    return {
        "max_pct": round(max(risk_at), 1),
        "flooded_m": round(flooded_m),
        "hotspots": sorted(((v, k) for k, v in hotspots.items()), reverse=True),
        "risk_at": risk_at,
        "samples": samples,
    }


# --------------------------------------------------------------------------- #
# Building an alternate, and choosing between them
# --------------------------------------------------------------------------- #
def fetch_via(origin: Tuple[float, float],
              via: Tuple[float, float],
              destination: Tuple[float, float]) -> dict:
    """One route that is forced to pass through a waypoint.

    This is how a genuine alternate gets built. OSRM's own alternatives are a
    bonus, not a guarantee - for many journeys it returns a single route, and
    if that one is flooded there is nothing to choose from. Sending the request
    through a dry waypoint forces a different road, which is what a driver
    means by "show me another way".
    """
    try:
        import requests
    except ImportError:
        return {"routes": [], "error": "The requests package is not installed."}

    path = (f"{origin[1]},{origin[0]};"
            f"{via[1]},{via[0]};"
            f"{destination[1]},{destination[0]}")

    try:
        response = requests.get(
            f"{OSRM}/{path}",
            params={"overview": "full", "geometries": "geojson", "steps": "true"},
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": "urban-flood-nowcasting/1.0 (SIH 2026 project)"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {"routes": [], "error": f"{exc.__class__.__name__}: {exc}"}

    if payload.get("code") != "Ok" or not payload.get("routes"):
        return {"routes": [], "error": payload.get("message", "No route found.")}

    route = payload["routes"][0]
    coords = route.get("geometry", {}).get("coordinates", [])
    return {
        "routes": [{
            "index": 0,
            "coords_latlon": [(lat, lon) for lon, lat in coords],
            "distance_m": route.get("distance", 0.0),
            "duration_s": route.get("duration", 0.0),
            "steps": _flatten_steps(route),
        }],
        "error": None,
    }


def detour_waypoints(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    places: Dict[str, Tuple[float, float]],
    risk_by_place: Dict[str, float],
    threshold_pct: float,
    limit: int = 4,
    max_detour_ratio: float = 2.2,
) -> List[Tuple[str, Tuple[float, float]]]:
    """Dry places worth routing through, nearest-first.

    A waypoint is only worth trying if it is dry itself and does not send the
    journey wildly out of the way. Ranking by added distance means the first
    detour tried is the least painful one.
    """
    straight = haversine_m(origin, destination) or 1.0
    candidates = []

    for name, point in places.items():
        if risk_by_place.get(name, 0.0) >= threshold_pct:
            continue                       # no sense detouring through water
        through = haversine_m(origin, point) + haversine_m(point, destination)
        if through <= straight * 1.02:
            continue                       # already on the direct line
        if through > straight * max_detour_ratio:
            continue                       # too far out of the way to be useful
        candidates.append((through, name, point))

    candidates.sort()
    return [(name, point) for _, name, point in candidates[:limit]]


def _broadly_same(a: dict, b: dict, tolerance_m: float = 60.0) -> bool:
    """Are these two routes the same road, give or take?

    Compares sampled points pairwise. Two routes that share almost every point
    are one route with a different label, and offering both as "alternatives"
    is a lie the user notices immediately.
    """
    left = _sample(a["coords_latlon"], 250.0)
    right = _sample(b["coords_latlon"], 250.0)
    if not left or not right:
        return False
    if abs(a["distance_m"] - b["distance_m"]) > 250:
        return False
    near = sum(1 for point in left
               if min(haversine_m(point, other) for other in right) <= tolerance_m)
    return near / len(left) >= 0.9


def plan_journey(
    drains: Dict[str, Drain],
    depths: Dict[str, float],
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    places: Dict[str, Tuple[float, float]],
    risk_by_place: Dict[str, float],
    threshold_pct: float = 40.0,
    max_detours: int = 4,
    prefer: str = "safety",
    min_alternatives: int = 2,
    fetch=None,
    fetch_via_fn=None,
) -> dict:
    """Find the driest way there, building detours if nothing on offer is clear.

    Three stages, stopping as soon as something clear turns up:

      1. Ask OSRM for the direct route and its own alternatives.
      2. If all of them cross water, force routes through dry waypoints.
      3. If still nothing is clear, return the driest and say so plainly.

    `prefer` decides how to rank what is on offer, and it is genuinely
    different per traveller rather than the same route with a different label:

      "time"      quickest, wet or not. A fire tender wades where a car cannot,
                  and an ambulance held up by a diversion has its own body count.
      "safety"    quickest of the routes under the threshold. The default, and
                  what a car wants.
      "driest"    lowest worst-point, even if it is slower. A scooter rider
                  cannot see a pothole or an open manhole through muddy water,
                  so avoiding the water matters more than saving five minutes.
      "shortest"  least distance among the passable ones. On foot, an extra
                  kilometre is worse than an extra five minutes in a car.

    `min_alternatives` guarantees there is always something to compare against.
    OSRM often returns a single route, and "here is your only option" is not
    navigation — so if fewer than this many come back, detours are built through
    dry waypoints until there are enough, whether or not the first one was wet.

    `fetch` and `fetch_via_fn` exist so a caller can supply cached versions of
    the two network calls. The scoring and the choosing must stay uncached —
    they depend on the forecast, which changes every time the user moves the
    rainfall slider, and caching them silently freezes the answer.
    """
    fetch = fetch or fetch_routes
    fetch_via_fn = fetch_via_fn or fetch_via

    fetched = fetch(origin, destination)
    if fetched["error"]:
        return {"ok": False, "error": fetched["error"], "routes": []}

    routes = []
    for index, route in enumerate(fetched["routes"]):
        route["score"] = score_route(route["coords_latlon"], drains, depths, threshold_pct)
        route["label"] = "Direct" if index == 0 else f"Alternative {index}"
        route["via"] = None
        routes.append(route)

    fastest = min(routes, key=lambda r: r["duration_s"])
    clear = [r for r in routes if r["score"]["max_pct"] < threshold_pct]
    tried_detours = []

    # Two reasons to go and build a detour: nothing on offer is dry, or there
    # is nothing to compare against. The second one used to be skipped, which
    # meant that on a journey OSRM answered with a single route the page said
    # "1 route considered" and offered no choice at all.
    needs_dry = not clear
    needs_choice = len(routes) < min_alternatives

    if needs_dry or needs_choice:
        # Budget the network calls. Hunting for a dry route is worth up to
        # max_detours attempts; merely wanting something to compare against is
        # worth exactly one. Firing five serial OSRM requests on a journey that
        # was already fine is what made this page sluggish before.
        budget = max_detours if needs_dry else 2
        attempts = 0
        # More candidates than budget, because a waypoint that turns out to be
        # unreachable or to retrace the direct route should not cost us the
        # chance to try the next one. The budget counts network calls, which is
        # what actually costs time — offering extra candidates is free.
        for name, point in detour_waypoints(
            origin, destination, places, risk_by_place, threshold_pct,
            limit=budget + 3,
            max_detour_ratio=2.2 if needs_dry else 2.6,
        ):
            if attempts >= budget:
                break
            attempts += 1
            tried_detours.append(name)
            attempt = fetch_via_fn(origin, point, destination)
            if attempt["error"] or not attempt["routes"]:
                continue
            route = attempt["routes"][0]
            route["score"] = score_route(route["coords_latlon"], drains, depths, threshold_pct)
            route["label"] = f"Detour via {name}"
            route["via"] = name

            # A detour that retraces the route we already have is not an
            # alternative, however differently it was built.
            if any(_broadly_same(route, existing) for existing in routes):
                tried_detours[-1] = f"{name} (same roads)"
                continue
            routes.append(route)

            dry_now = route["score"]["max_pct"] < threshold_pct
            enough = len(routes) >= min_alternatives
            if enough and (dry_now or not needs_dry):
                break

        clear = [r for r in routes if r["score"]["max_pct"] < threshold_pct]

    pool = clear or routes
    if prefer == "time":
        chosen = min(routes, key=lambda r: (r["duration_s"], r["score"]["max_pct"]))
    elif prefer == "driest":
        chosen = min(pool, key=lambda r: (r["score"]["max_pct"], r["duration_s"]))
    elif prefer == "shortest":
        chosen = min(pool, key=lambda r: (r["distance_m"], r["score"]["max_pct"]))
    elif clear:                             # "safety": quickest passable route
        chosen = min(clear, key=lambda r: r["duration_s"])
    else:
        # Nothing is passable, so the least-bad option is the driest one.
        # Collapsing this into a single sort over the whole pool looked tidier
        # and quietly made "safety" behave exactly like "time".
        chosen = min(routes, key=lambda r: (r["score"]["max_pct"], r["duration_s"]))

    return {
        "ok": True,
        "error": None,
        "routes": routes,
        "chosen": chosen,
        "fastest": fastest,
        "all_clear": bool(clear),
        "rerouted": chosen is not fastest,
        "detours_tried": tried_detours,
        "threshold_pct": threshold_pct,
        "prefer": prefer,
        "alternatives": max(len(routes) - 1, 0),
    }


def summarise(route: dict) -> dict:
    return {
        "distance_km": round(route["distance_m"] / 1000, 1),
        "minutes": round(route["duration_s"] / 60),
        "max_pct": route["score"]["max_pct"],
        "flooded_m": route["score"]["flooded_m"],
        "condition": advice_for(route["score"]["max_pct"]),
    }


def check() -> str:
    """Quick connectivity test: Bandra Fort to BKC."""
    result = fetch_routes((19.0430, 72.8190), (19.0660, 72.8690))
    if result["error"]:
        return f"OSRM unreachable: {result['error']}"
    first = result["routes"][0]
    return (
        f"OSRM works. {len(result['routes'])} route(s), "
        f"first is {first['distance_m'] / 1000:.1f} km with "
        f"{len(first['coords_latlon'])} geometry points and "
        f"{len(first['steps'])} turns."
    )


# --------------------------------------------------------------------------- #
# How long will it actually take?
# --------------------------------------------------------------------------- #
# OSRM's duration is free-flow: dry road, no traffic, no water. Neither half of
# that is true in a Mumbai monsoon, and a navigation app that says "19 min"
# while you crawl through a foot of water for half an hour is worse than one
# that says nothing.
#
# So the free-flow time is stretched twice. Once for the rain itself, which
# slows everyone down whether or not the road is flooded. Once per sampled
# segment, for the water standing on it.

def rain_slowdown(intensity_mm_hr: float) -> float:
    """Speed multiplier for driving in rain on a road that is not flooded.

    Visibility, spray and everyone else braking. Heavy monsoon rain costs
    roughly a third of your speed before a single puddle is involved.
    """
    if intensity_mm_hr <= 0.5:
        return 1.0
    # 1.0 in the dry, about 0.85 in light rain, 0.66 in a downpour.
    return max(1.0 - 0.18 * math.log1p(intensity_mm_hr / 6.0), 0.55)


def water_slowdown(risk_pct_here: float) -> float:
    """Speed multiplier for a stretch with standing water on it.

    Falls away fast, because driving through water is not a linear penalty:
    at ankle depth you slow down, at knee depth you are in first gear behind
    someone who has stopped.
    """
    if risk_pct_here <= 1.0:
        return 1.0
    return max(1.0 / (1.0 + (risk_pct_here / 25.0) ** 2), 0.06)


def eta(route: dict, intensity_mm_hr: float = 0.0,
        depart_in_min: int = 0, now=None) -> dict:
    """A realistic arrival estimate, as a range rather than a false precision.

    The range is the honest part. A single number implies the app knows the
    traffic, and it does not. The spread widens with the amount of water on
    the route, because that is exactly where the estimate gets least reliable.
    """
    from datetime import datetime, timedelta

    free_flow_min = route["duration_s"] / 60.0
    risk_at = route["score"]["risk_at"] or [0.0]
    segments = max(len(risk_at), 1)

    # Split the free-flow time evenly across the sampled segments, then stretch
    # each one by what is standing on it.
    per_segment = free_flow_min / segments
    rain_factor = rain_slowdown(intensity_mm_hr)

    wet_minutes = 0.0
    total = 0.0
    for risk in risk_at:
        factor = rain_factor * water_slowdown(risk)
        stretched = per_segment / max(factor, 0.05)
        total += stretched
        if risk >= 20.0:
            wet_minutes += stretched

    # Two different things, which an earlier version ran together.
    #
    # "Impassable" is about DEPTH, not time. A route whose worst point is over
    # 45 cm is one nobody should attempt, however long the arithmetic says it
    # takes. Judging it by time instead labelled a genuine 50-minute monsoon
    # crawl impassable while it was merely slow.
    impassable = route["score"]["max_pct"] >= IMPASSABLE_RISK_PCT

    # The cap is separate, and only stops the arithmetic running away: a
    # five-kilometre trip does not have a useful "284 minutes" answer.
    capped = total > free_flow_min * MAX_SLOWDOWN_FACTOR
    if capped:
        scale = (free_flow_min * MAX_SLOWDOWN_FACTOR) / max(total, 1e-9)
        total *= scale
        wet_minutes *= scale        # or the share below exceeds 1 and the
                                    # window collapses to a negative low bound

    delay = total - free_flow_min
    # Uncertainty: 15% baseline for traffic, widening as the route gets wetter,
    # but bounded — an estimate of "20 to 300 minutes" is not an estimate.
    wet_share = min(wet_minutes / max(total, 1e-9), 1.0)
    spread = min(0.15 + 0.35 * wet_share, 0.5)

    low = max(total * (1 - spread), 1.0)
    high = total * (1 + spread)

    now = now or datetime.now()
    depart = now + timedelta(minutes=depart_in_min)

    return {
        "free_flow_min": round(free_flow_min),
        "minutes_low": max(1, round(low)),
        "minutes_high": max(1, round(high)),
        "minutes_mid": max(1, round(total)),
        "delay_min": round(delay),
        "wet_minutes": round(wet_minutes),
        "spread_pct": round(spread * 100),
        "impassable": impassable,
        "capped": capped,
        "depart_at": depart.strftime("%H:%M"),
        "arrive_low": (depart + timedelta(minutes=low)).strftime("%H:%M"),
        "arrive_high": (depart + timedelta(minutes=high)).strftime("%H:%M"),
        "rain_factor": round(rain_factor, 2),
    }


def describe_eta(estimate: dict) -> str:
    if estimate.get("impassable"):
        return ("impassable — the worst point is over 45 cm, deep enough that "
                "no arrival time is worth quoting")
    if estimate["minutes_low"] == estimate["minutes_high"]:
        window = f"{estimate['minutes_mid']} min"
    else:
        window = f"{estimate['minutes_low']}-{estimate['minutes_high']} min"
    if estimate.get("capped"):
        return (f"over {estimate['minutes_low']} min — major delays, the "
                f"estimate stops being meaningful beyond this")
    if estimate["delay_min"] >= 2:
        return (f"{window} — arriving {estimate['arrive_low']}-"
                f"{estimate['arrive_high']}, about {estimate['delay_min']} min "
                f"more than a clear day")
    return f"{window} — arriving {estimate['arrive_low']}-{estimate['arrive_high']}"


# --------------------------------------------------------------------------- #
# Risk factor for a whole journey
# --------------------------------------------------------------------------- #
# The drain-level risk factor answers "how bad is this spot". This answers
# "how bad is this trip", which is a different question: a single deep puddle
# on a short hop matters more than the same puddle on a long drive, and a
# route that is 40% past a scooter's limit is worse than one 40% past a fire
# tender's.

# The same scale the dashboard uses, imported rather than restated so a legend
# can never disagree with the number printed next to it.
from flood_engine import RISK_FACTOR_BANDS as RISK_BANDS      # noqa: E402


from flood_engine import risk_band                            # noqa: E402,F401


def route_risk_factor(route: dict, threshold_pct: float,
                      estimate: Optional[dict] = None) -> dict:
    """0-100 for the journey, with the reasoning kept alongside it.

    Four parts, because depth alone is not danger:

      depth    (35%)  how deep the worst point gets
      extent   (25%)  how much of the route is under water, not just one spot
      exposure (20%)  how much of the trip is spent in it
      margin   (20%)  how far past this traveller's own limit it goes
    """
    score_block = route["score"]
    worst = score_block["max_pct"]
    flooded_m = score_block["flooded_m"]
    distance_m = max(route["distance_m"], 1.0)

    depth_part = min(worst / 100.0, 1.0)
    extent_part = min(flooded_m / distance_m, 1.0)

    if estimate and estimate["minutes_mid"]:
        exposure_part = min(estimate["wet_minutes"] / estimate["minutes_mid"], 1.0)
    else:
        exposure_part = extent_part

    # How far past the line for whoever is travelling. At the threshold this is
    # 0; at twice the threshold it is 1.
    margin_part = min(max((worst - threshold_pct) / max(threshold_pct, 1.0), 0.0), 1.0)

    score = round((0.35 * depth_part + 0.25 * extent_part
                   + 0.20 * exposure_part + 0.20 * margin_part) * 100)
    name, colour, meaning = risk_band(score)

    reasons = []
    if worst > 0:
        reasons.append(f"deepest point {worst / 100 * 50:.0f} cm ({worst:.0f}%)")
    if flooded_m:
        reasons.append(f"{flooded_m} m of the {distance_m / 1000:.1f} km is wet")
    if estimate and estimate["wet_minutes"] >= 1:
        reasons.append(f"about {estimate['wet_minutes']} min spent in water")
    if margin_part > 0:
        reasons.append("past the limit for this vehicle")

    return {
        "score": score,
        "band": name,
        "colour": colour,
        "meaning": meaning,
        "why": (name + ": " + ", ".join(reasons) + ".") if reasons
               else "Minor: no standing water on this route.",
        "parts": {
            "Depth of the worst point": round(0.35 * depth_part * 100),
            "How much of the route is wet": round(0.25 * extent_part * 100),
            "Time spent in water": round(0.20 * exposure_part * 100),
            "Past this vehicle's limit": round(0.20 * margin_part * 100),
        },
    }
