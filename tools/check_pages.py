"""Execute every Streamlit page against a fake Streamlit and report crashes.

    python3 tools/check_pages.py

Run this before every push. It will not tell you the layout looks right; it
will tell you the page runs at all, which is the failure mode that ruins a
demo. Network calls are stubbed out too, so it works offline and never
depends on OSRM being up.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import stub_streamlit                                   # noqa: E402

PAGES = ["app.py", "pages/1_Safe_Route.py", "pages/2_How_It_Works.py"]

# Widget values that steer the pages down branches the defaults never reach.
# Every one of these was a real place a bug could hide: the emergency briefing,
# the drowned-outfall case, the manual what-if path, the zero-rain path.
SCENARIOS = {
    "defaults": {},
    "emergency at high tide": {
        "Vehicle": "Emergency (fire / ambulance)",
        "Sea level at the outfalls": 4.0,
        "Tide during the storm": 4.0,
        "Tide": 4.0,
    },
    "two-wheeler, manual heavy rain": {
        "Vehicle": "Two-wheeler",
        "Source": "Manual (what-if)",
        "Rainfall": "Manual (what-if)",
        "Intensity (mm/hr)": 150,
        "Rainfall (mm/hr)": 150,
    },
    "no rain at all": {
        "Source": "Manual (what-if)",
        "Rainfall": "Manual (what-if)",
        "Intensity (mm/hr)": 0,
        "Rainfall (mm/hr)": 0,
    },
    "moderate rain, dry detour available": {
        # Exercises the route-comparison branch with more than one option on
        # offer. Note it cannot produce "shortest flooded, alternate clear":
        # these fake routes are straight lines, so a detour through a dry
        # waypoint still starts and ends beside the same drains and scores the
        # same. That case is pinned in test_flood_engine.py instead.
        "Source": "Manual (what-if)",
        "Rainfall": "Manual (what-if)",
        "Intensity (mm/hr)": 30,
        "Rainfall (mm/hr)": 30,
        "Vehicle": "Car",
    },
    "walking, custom threshold, three hours out": {
        "Vehicle": "Walking",
        "Set the threshold myself": True,
        "Treat as flooded above (%)": 15,
        "Leaving in": 180,
        "Looking ahead": 180,
        "After this much rain": 180,
        "After": 180,
    },
}


def stub_network() -> None:
    """No OSRM, no Open-Meteo. Offline must be a supported state, not a crash."""
    import flood_engine as fe
    import road_router as rr

    fe.live_series = lambda *a, **k: {
        "series": fe.flat_series(35.0), "error": None, "peak": 35.0}

    def _line(origin, destination, bow=0.0, steps=8):
        """A polyline from A to B, optionally bowed sideways.

        The bow matters: two routes with identical geometry get rescored
        identically and deduplicated, so the page only ever saw a single
        option and the "several routes" branch went unexercised. That is
        precisely how a name collision in that branch reached the browser.
        """
        points = []
        for i in range(steps + 1):
            t = i / steps
            lat = origin[0] + (destination[0] - origin[0]) * t
            lon = origin[1] + (destination[1] - origin[1]) * t
            # A sine bulge, zero at both ends so the endpoints still match.
            offset = bow * (t * (1 - t) * 4)
            points.append((lat + offset, lon - offset))
        return points

    def _route(coords, distance_m, duration_s):
        return {"index": 0, "coords_latlon": coords,
                "distance_m": distance_m, "duration_s": duration_s,
                "steps": [
                    {"instruction": "Start out on Hill Road", "road": "Hill Road",
                     "distance_m": 400.0, "location": coords[0]},
                    {"instruction": "Turn left onto Linking Road",
                     "road": "Linking Road", "distance_m": 900.0,
                     "location": coords[len(coords) // 2]},
                    {"instruction": "Arrive at your destination", "road": "",
                     "distance_m": 0.0, "location": coords[-1]},
                ]}

    def fake_routes(origin, destination):
        return {"routes": [
            _route(_line(origin, destination, 0.0), 5200.0, 900.0),
            _route(_line(origin, destination, 0.012), 7100.0, 1250.0),
        ], "error": None}

    def fake_via(origin, via, destination):
        """Actually pass through the waypoint, as the real one does.

        Ignoring `via` and returning a fixed bow meant the detour was scored
        against the same drains as the direct route, so it was always just as
        wet. The "shortest is flooded, here is the dry way round" branch —
        which is the whole point of the page — was never executed.
        """
        return {"routes": [
            _route(_line(origin, via, 0.0, steps=5)
                   + _line(via, destination, 0.0, steps=5)[1:],
                   8400.0, 1500.0),
        ], "error": None}

    rr.fetch_routes = fake_routes
    rr.fetch_via = fake_via


def run(path: str) -> bool:
    os.chdir(ROOT)
    source = open(path).read()
    namespace = {"__name__": "__main__", "__file__": os.path.join(ROOT, path)}
    try:
        exec(compile(source, path, "exec"), namespace)
    except stub_streamlit.Stop:
        print(f"  {path:32} OK (stopped early, which is a valid path)")
        return True
    except Exception:
        print(f"  {path:32} CRASHED")
        print("    " + "\n    ".join(traceback.format_exc().splitlines()[-12:]))
        return False
    print(f"  {path:32} OK")
    return True


def main() -> int:
    results = []
    for name, overrides in SCENARIOS.items():
        print(f"--- {name} ---")
        stub_streamlit.install(overrides)
        stub_network()
        for page in PAGES:
            results.append(run(page))
        print()

    failures = results.count(False)
    if not failures:
        print(f"All {len(PAGES)} pages run clean under "
              f"{len(SCENARIOS)} scenarios ({len(results)} runs).")
        return 0
    print(f"{failures} of {len(results)} runs crashed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
