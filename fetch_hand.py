"""Sample Height Above Nearest Drainage for every drain, once.

Run this on a machine with internet:

    python3 fetch_hand.py

It writes `hand_values.csv`. The model picks it up automatically and switches
`retention` and `max_pond_cm` from the elevation heuristic to HAND. Commit the
CSV and the app then needs no network for it — which matters, because a demo
should never depend on conference wifi.

Where the data comes from
-------------------------
ASF's global 30 m HAND raster, derived from the 2021 Copernicus GLO-30 DEM and
published under CC0. This uses the ArcGIS ImageServer `identify` endpoint: one
small request per point, no API key, no AWS account, no GDAL. For ten drains
that is ten requests and about five seconds.

If you later want HAND as a continuous raster rather than at ten points, the
same data is on S3 as Cloud Optimized GeoTIFFs — `--show-tiles` prints the URLs
of the tiles covering your drains. Reading those needs rasterio.
"""
from __future__ import annotations

import sys
import time

import flood_engine as fe
import hand


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    try:
        drains = fe.load_drains()
    except Exception as exc:                            # noqa: BLE001
        print(f"Could not read bandra_capacity.csv — {exc}")
        return 1

    if "--show-tiles" in argv:
        print("HAND COG tiles covering these drains (CC0, no credentials):\n")
        for url in sorted({hand.cog_url_for(d.lat, d.lon) for d in drains.values()}):
            print(f"  {url}")
        return 0

    try:
        import requests                                 # noqa: F401
    except ImportError:
        print("Install requests first:  pip install requests")
        return 1

    print(f"Sampling ASF HAND at {len(drains)} drain locations.\n")
    print(f"  {'drain':<24}{'elev m':>8}{'HAND m':>9}   ground")

    values, failures = {}, []
    for drain_id, drain in sorted(drains.items()):
        result = hand.sample_point(drain.lat, drain.lon)
        if not result["ok"]:
            failures.append((drain.name, result["error"]))
            print(f"  {drain.name:<24}{drain.elevation_m:>8.1f}{'—':>9}   "
                  f"{result['error'][:44]}")
            continue

        metres = result["hand_m"]
        values[drain_id] = {
            "name": drain.name, "lat": drain.lat, "lon": drain.lon,
            "elevation_m": drain.elevation_m, "hand_m": metres,
            "source": "ASF GLO-30 HAND v1 (2021), ImageServer identify",
        }
        print(f"  {drain.name:<24}{drain.elevation_m:>8.1f}{metres:>9.2f}   "
              f"{hand.describe(metres)}")
        time.sleep(0.3)                 # be polite to a free public service

    if not values:
        print("\nNothing sampled, so nothing written — the model keeps using "
              "the elevation heuristic and the app will say so.")
        if failures:
            print(f"First failure: {failures[0][1]}")
        return 1

    hand.save(values)
    print(f"\nWrote {hand.HAND_CSV} — {len(values)} of {len(drains)} drains.")
    if failures:
        print(f"{len(failures)} could not be sampled and will fall back to "
              "elevation. That is fine; the app reports which is which.")

    # The point of the exercise: show what actually changed.
    print("\nWhat this changes in the model:\n")
    print(f"  {'drain':<24}{'retention':>22}{'max ponding':>22}")
    print(f"  {'':<24}{'elevation -> HAND':>22}{'elevation -> HAND':>22}")
    for row in hand.comparison(drains):
        if row["HAND_m"] is None:
            continue
        print(f"  {row['Segment_Name']:<24}"
              f"{row['Retention_elevation']:>10.2f} -> {row['Retention_HAND']:<8.2f}"
              f"{row['Max_pond_elevation_cm']:>10.0f} -> {row['Max_pond_HAND_cm']:<8.0f}")

    print("\nCommit hand_values.csv so the app works offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
