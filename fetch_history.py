"""Download the real hourly rainfall for Mumbai's recorded flood days.

Run this once, on a machine with internet:

    python3 fetch_history.py

It writes `historical_rainfall.csv`, which the justification page picks up
automatically. Commit that file and the page then works with no network at
all — which matters, because a demo should never depend on a conference
wifi connection holding up.

The source is the Open-Meteo historical archive, which serves ERA5
reanalysis: the same dataset climate scientists use, gridded and free, no
key needed. It goes back to 1940, so every Mumbai flood on record is
reachable.

One note worth keeping in view. ERA5 is a reanalysis on a roughly 25 km grid,
not a single rain gauge, so it will not reproduce a sharply localised cell.
The published gauge figures in history.py are the authority for how much fell;
this file is the authority for how it was distributed through the day, which
is what the model needs and what a daily total cannot tell you.
"""
from __future__ import annotations

import csv
import sys
import time

BANDRA_LAT, BANDRA_LON = 19.0544, 72.8402
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OUT = "historical_rainfall.csv"

# The dates history.py knows about, plus a few ordinary wet days to act as
# controls — a model that only ever sees disasters has not been tested.
DATES = [
    # The events the model is scored on — recent, so they reflect the drainage
    # network as it stands rather than as it was twenty years ago.
    ("2025-08-20", "20 August 2025 - harbour line down 15 hours"),
    ("2025-05-26", "26 May 2025 - earliest monsoon onset in 75 years"),
    ("2024-07-08", "8 July 2024 deluge"),
    ("2024-09-25", "25 September 2024 heavy rain"),
    ("2025-07-15", "July 2025 monsoon spell"),
    # Historical context, kept clearly separate.
    ("2017-08-29", "29 August 2017 flood"),
    # Controls: monsoon days with rain but no recorded city-wide disruption.
    # A model that has only ever been shown disasters has not been tested.
    ("2024-08-14", "control - ordinary monsoon day"),
    ("2025-09-05", "control - ordinary monsoon day"),
    ("2024-06-20", "control - ordinary monsoon day"),
]


def fetch_one(session, date: str) -> list:
    response = session.get(
        ARCHIVE,
        params={
            "latitude": BANDRA_LAT,
            "longitude": BANDRA_LON,
            "start_date": date,
            "end_date": date,
            "hourly": "precipitation",
            "timezone": "Asia/Kolkata",
        },
        timeout=30,
    )
    response.raise_for_status()
    block = response.json().get("hourly", {})
    times = block.get("time", [])
    values = block.get("precipitation", [])
    return [(t, 0.0 if v is None else float(v)) for t, v in zip(times, values)]


def main() -> int:
    try:
        import requests
    except ImportError:
        print("Install requests first:  pip install requests")
        return 1

    session = requests.Session()
    rows = []
    for date, label in DATES:
        try:
            hourly = fetch_one(session, date)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {date}  FAILED  {exc.__class__.__name__}: {exc}")
            continue
        total = sum(mm for _, mm in hourly)
        peak = max((mm for _, mm in hourly), default=0.0)
        print(f"  {date}  {total:7.1f} mm total, {peak:6.1f} mm peak hour   {label}")
        for stamp, mm in hourly:
            rows.append({"date": date, "time": stamp,
                         "precipitation_mm": round(mm, 2), "label": label})
        time.sleep(0.4)                 # be polite to a free API

    if not rows:
        print("\nNothing downloaded. Check your connection and try again.")
        return 1

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", "time",
                                                "precipitation_mm", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUT} — {len(rows)} hourly readings across "
          f"{len({r['date'] for r in rows})} dates.")
    print("Commit it, and the justification page will use real hourly rainfall "
          "instead of flat averages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
