"""Real road navigation that steers around flooding.

Routes come from OSRM, the open-source engine behind a lot of OpenStreetMap
routing. It is free, needs no key, follows actual streets and will hand back
several alternative routes for the same journey.

The part that matters is what we do with those alternatives. OSRM ranks them
by driving time and knows nothing about water. So each candidate is sampled
along its length, scored against the predicted flood depth near every sample,
and the driest one is chosen - even when it is slower. If the fastest route is
flooded and a slower one is not, that is precisely the decision worth making,
and it is the decision a navigation app cannot make today because nobody tells
it where the water is.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from flood_engine import Drain, haversine_m
from route_planner import advice_for, risk_pct

OSRM = "https://router.project-osrm.org/route/v1/driving"

INFLUENCE_M = 400.0
SAMPLE_EVERY_M = 120.0
REQUEST_TIMEOUT_S = 12.0


def fetch_routes(origin: Tuple[float, float], destination: Tuple[float, float]) -> dict:
    """Ask OSRM for the driving routes between two (lat, lon) points."""
    try:
        import requests
    except ImportError:
        return {"routes": [], "error": "The requests package is not installed."}

    # OSRM wants lon,lat.
    path = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"

    try:
        response = requests.get(
            f"{OSRM}/{path}",
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
    except Exception as exc:
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
        return (f"At the roundabout take exit {exit_no}{where}" if exit_no
                else f"Take the roundabout{where}")
    return f"{TURNS.get(modifier, 'Continue')}{where}"


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

    over = sum(1 for r in risk_at if r >= threshold_pct)

    return {
        "max_pct": round(max(risk_at), 1),
        "flooded_m": round(over * SAMPLE_EVERY_M),
        "hotspots": sorted(((v, k) for k, v in hotspots.items()), reverse=True),
        "risk_at": risk_at,
        "samples": samples,
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
    """Dry places worth routing through, least painful first.

    A waypoint is only worth trying if it is dry itself and does not send the
    journey wildly out of the way.
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


def plan_journey(
    drains: Dict[str, Drain],
    depths: Dict[str, float],
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    places: Dict[str, Tuple[float, float]],
    risk_by_place: Dict[str, float],
    threshold_pct: float = 40.0,
    max_detours: int = 4,
    fetch=None,
    fetch_via_fn=None,
) -> dict:
    """Find the driest way there, building detours if nothing on offer is clear.

    Three stages, stopping as soon as something clear turns up:

      1. Ask OSRM for the direct route and its own alternatives.
      2. If all of them cross water, force routes through dry waypoints.
      3. If still nothing is clear, return the driest and say so plainly.

    fetch and fetch_via_fn exist so a caller can supply cached versions of the
    two network calls. The scoring and the choosing must stay uncached - they
    depend on the forecast, which changes every time the rainfall slider moves,
    and caching them silently freezes the answer.
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

    if not clear:
        # Nothing OSRM offered is dry, so go and build something that is.
        for name, point in detour_waypoints(
            origin, destination, places, risk_by_place, threshold_pct, limit=max_detours
        ):
            tried_detours.append(name)
            attempt = fetch_via_fn(origin, point, destination)
            if attempt["error"] or not attempt["routes"]:
                continue
            route = attempt["routes"][0]
            route["score"] = score_route(route["coords_latlon"], drains, depths, threshold_pct)
            route["label"] = f"Detour via {name}"
            route["via"] = name
            routes.append(route)
            if route["score"]["max_pct"] < threshold_pct:
                break                       # found a dry way; stop asking

        clear = [r for r in routes if r["score"]["max_pct"] < threshold_pct]

    if clear:
        chosen = min(clear, key=lambda r: r["duration_s"])
    else:
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
    }
