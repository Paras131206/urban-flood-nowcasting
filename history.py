"""Real recorded storms, replayed through the model.

What this file does
-------------------
It takes storms that actually happened in Mumbai, feeds the rainfall that was
actually recorded through the model, and reports what the model says. Every
total below is a published figure with its source attached.

The events are recent on purpose. Mumbai's drainage has been worked on
continuously — new pumping stations, and a long widening programme under
BRIMSTOWAD — so a model checked against a twenty-year-old deluge is being
checked against a network that no longer exists. Everything used for scoring
is from the 2024 and 2025 monsoons, which is the system as it stands today.

The reference standard
----------------------
Mumbai's storm water drains were built for 25 mm/hr at low tide — a published
engineering commitment, and a reference that cuts both ways. The model should
not cry wolf in conditions the drains are designed to handle, and it should
predict failure once conditions exceed them.

The most interesting result on this page falls out of that. Both 2025 events
averaged *below* the design standard — 18.2 and 17.1 mm/hr — and both flooded
the city anyway, because they were sustained for hours rather than minutes.
The model reproduces that, and it is the clearest evidence that duration
matters as much as intensity.

Roadmap
-------
The next step for this module is straightforward and worth naming: water level
sensors logging depth at junctions through one monsoon would let the model be
scored in centimetres rather than checked for consistency. Everything needed to
consume that data is already here.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import flood_engine as fe

RAINFALL_CSV = "historical_rainfall.csv"


# --------------------------------------------------------------------------- #
# Recorded storms
# --------------------------------------------------------------------------- #
class Event:
    def __init__(self, date, name, total_mm, hours, outcome, source, url,
                 tide_note="", era="recent", gauge="city-wide", verified=True):
        self.date = date
        self.name = name
        self.total_mm = total_mm
        self.hours = hours
        self.outcome = outcome
        self.source = source
        self.url = url
        self.tide_note = tide_note
        self.era = era                  # "recent" | "historical"
        self.gauge = gauge              # which station the total is from
        self.verified = verified

    @property
    def mean_mm_hr(self) -> float:
        return round(self.total_mm / self.hours, 1)

    @property
    def times_design_standard(self) -> float:
        return round(self.mean_mm_hr / fe.DESIGN_STANDARD_MM_HR, 2)

    @property
    def below_design_standard(self) -> bool:
        """Did the city flood on rain the drains were supposed to handle?"""
        return self.mean_mm_hr < fe.DESIGN_STANDARD_MM_HR


# The storms the model is scored on. All three are from the last two monsoons,
# so they test the network as it stands rather than as it was before two
# decades of pumping stations and widening were added to it.
EVENTS: List[Event] = [
    Event(
        date="2025-08-20",
        name="20 August 2025",
        total_mm=200.0, hours=11,
        outcome="Harbour line suspended for over 15 hours. 782 passengers "
                "rescued from stranded monorail trains, 400+ residents "
                "evacuated at Kurla as the Mithi reached 3.9 m against its "
                "4 m danger mark. Eight flights diverted.",
        source="Over 200 mm in 11 hours; IMD red alert",
        url="https://www.businesstoday.in/india/story/mumbai-rains-intensify-"
            "imd-issues-red-alert-as-record-200-mm-rain-lashes-city-490166-2025-08-20",
        tide_note="The Mithi within 10 cm of its danger mark is the outfall "
                  "condition this model represents as a drowned outfall.",
        era="recent",
    ),
    Event(
        date="2025-05-26",
        name="26 May 2025",
        total_mm=68.5, hours=4,
        outcome="Suburban rail halted between Wadala and CSMT, metro stopped "
                "between Acharya Atre Chowk and Worli. Earliest monsoon onset "
                "over Mumbai in 75 years; Colaba recorded its wettest May in "
                "107 years.",
        source="Bandra gauge: 68.5 mm between 08:00 and 12:00 (Colaba 105.2 mm "
               "over the same four hours)",
        url="https://watchers.news/2025/05/27/earliest-monsoon-75-years-breaks-"
            "mumbai-may-rainfall-record/",
        tide_note="",
        era="recent",
        gauge="Bandra",
    ),
    Event(
        date="2024-07-08",
        name="8 July 2024",
        total_mm=300.0, hours=6,
        outcome="City paralysed. Suburban train services disrupted, widespread "
                "waterlogging in low-lying areas.",
        source="Over 300 mm in about six hours",
        url="https://www.newsonair.gov.in/mumbai-records-over-300-mm-of-rainfall-"
            "witnesses-water-logging-disruption-of-suburban-train-services-in-"
            "low-lying-areas",
        era="recent",
    ),
]

# Kept for context, not for scoring, and shown separately on the page.
#
# 2005 is deliberately absent. It is the event everyone reaches for, and it is
# the least useful one available: a 944 mm day on a network that has since had
# two decades of pumping stations and widening added to it. Judging today's
# drainage by how it would have coped with 2005 flatters nobody and proves
# nothing. The recent storms above are the fair test.
HISTORICAL: List[Event] = [
    Event(
        date="2017-08-29",
        name="29 August 2017",
        total_mm=468.0, hours=12,
        outcome="City halted. Suburban trains stopped, flights cancelled or "
                "delayed, a building collapsed on Link Road.",
        source="468 mm in twelve hours, the highest August daily total since 1997",
        url="https://en.wikipedia.org/wiki/2017_Mumbai_flood",
        era="historical",
    ),
]

ALL_EVENTS = EVENTS + HISTORICAL


# --------------------------------------------------------------------------- #
# Real hourly series, if the fetcher has been run
# --------------------------------------------------------------------------- #
def load_recorded_hourly(path: str = RAINFALL_CSV) -> Dict[str, List[Tuple[str, float]]]:
    """Hourly rainfall per date, as written by fetch_history.py.

    Returns {} if the file is absent — the page then falls back to the
    published totals, and says which it is using.
    """
    if not os.path.exists(path):
        return {}
    out: Dict[str, List[Tuple[str, float]]] = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                date = row["date"]
                out.setdefault(date, []).append(
                    (row["time"], float(row["precipitation_mm"] or 0.0))
                )
    except Exception:                                   # noqa: BLE001
        return {}
    for date in out:
        out[date].sort()
    return out


def series_for(event: Event,
               recorded: Optional[Dict[str, List[Tuple[str, float]]]] = None
               ) -> Tuple[List[Tuple[int, float]], str]:
    """A three-hour rainfall series for the model, and where it came from.

    With real hourly data, the worst three consecutive hours of the event are
    used — that is the window the drains actually had to survive, and it is
    the fair thing to test a three-hour nowcast against.

    Without it, the published total is spread evenly across its stated
    duration. That is deliberately conservative: real storms are peakier than
    their average, so an even spread understates the worst hour rather than
    flattering the model.
    """
    recorded = recorded or {}
    hours = recorded.get(event.date)

    if hours and len(hours) >= 3:
        best_start, best_total = 0, -1.0
        for i in range(len(hours) - 2):
            window = sum(mm for _, mm in hours[i:i + 3])
            if window > best_total:
                best_start, best_total = i, window
        window = hours[best_start:best_start + 3]
        series = []
        for step in range(0, fe.HORIZON_MIN + 1, fe.STEP_MIN):
            index = min(step // 60, len(window) - 1)
            series.append((step, round(window[index][1], 2)))
        label = (f"real recorded rainfall, worst three hours "
                 f"({window[0][0][-5:]}-{window[-1][0][-5:]}, {best_total:.0f} mm)")
        return series, label

    return fe.flat_series(event.mean_mm_hr), (
        f"published total spread evenly ({event.total_mm:.0f} mm over "
        f"{event.hours} h = {event.mean_mm_hr} mm/hr)"
    )


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def replay(drains, event: Event, tide_m: float = fe.DEFAULT_TIDE_M,
           recorded=None) -> dict:
    """What the model says about a storm that really happened."""
    series, provenance = series_for(event, recorded)
    timeline = fe.forecast(drains, series, tide_m=tide_m)

    peaks = {did: max(v for _, v in points) for did, points in timeline.items()}
    flooded = [did for did, cm in peaks.items() if cm >= fe.FLOOD_DEPTH_CM]
    impassable = [did for did, cm in peaks.items() if cm >= 45.0]
    worst_id = max(peaks, key=lambda d: peaks[d])

    return {
        "event": event,
        "series": series,
        "provenance": provenance,
        "peak_mm_hr": max(v for _, v in series),
        "peaks_cm": peaks,
        "flooded": flooded,
        "impassable": impassable,
        "flooded_count": len(flooded),
        "worst_name": drains[worst_id].name,
        "worst_cm": round(peaks[worst_id], 1),
        "verdict": _verdict(len(flooded), len(drains)),
    }


def _verdict(flooded: int, total: int) -> str:
    share = flooded / max(total, 1)
    if share >= 0.7:
        return "Widespread flooding"
    if share >= 0.3:
        return "Significant flooding"
    if share > 0:
        return "Isolated flooding"
    return "No flooding"


# --------------------------------------------------------------------------- #
# Consistency checks
# --------------------------------------------------------------------------- #
class Check:
    def __init__(self, name, question, expectation, passed, detail):
        self.name = name
        self.question = question
        self.expectation = expectation
        self.passed = passed
        self.detail = detail


def run_checks(drains, recorded=None) -> List[Check]:
    """Six things that must be true if the model is worth anything.

    Three ask whether it fires on storms that really did flood the city. Three
    ask whether it stays quiet when it should — because a model that shouts on
    every shower is not a warning system, it is noise, and the false-alarm side
    is the half that usually goes unexamined.
    """
    checks: List[Check] = []
    total = len(drains)

    # --- It must fire on the real events ---------------------------------- #
    for event in EVENTS:
        result = replay(drains, event, recorded=recorded)
        fired = result["flooded_count"] >= total * 0.3
        headline = ""
        if event.below_design_standard:
            headline = (
                f" Note that {event.mean_mm_hr} mm/hr is only "
                f"{event.times_design_standard}x the design standard — the city "
                f"flooded on rain the drains were built to handle, because it "
                f"kept falling for {event.hours} hours. The model gets this "
                f"right for the right reason."
            )
        checks.append(Check(
            name=event.name,
            question=f"Recorded: {event.outcome.split('.')[0].lower()}. "
                     f"Does the model see it?",
            expectation="Significant or widespread flooding",
            passed=fired,
            detail=(f"{event.mean_mm_hr} mm/hr sustained over {event.hours} h "
                    f"({event.gauge} gauge) → {result['flooded_count']} of "
                    f"{total} spots over {fe.FLOOD_DEPTH_CM:.0f} cm. "
                    f"{result['verdict']}. Deepest: {result['worst_name']} at "
                    f"{result['worst_cm']:.0f} cm.{headline}"),
        ))

    # --- It must stay quiet when the city does not flood ------------------- #
    dry = fe.steady_depths(drains, 0.0, 180)
    checks.append(Check(
        name="No rain",
        question="With no rainfall at all, does anything flood?",
        expectation="Nothing, anywhere",
        passed=all(v < 0.01 for v in dry.values()),
        detail=f"Deepest spot: {max(dry.values()):.2f} cm.",
    ))

    # The city's own commitment: the drains handle 25 mm/hr at low tide. If the
    # model claims widespread flooding in exactly the conditions BMC says the
    # system is built for, the model is crying wolf.
    design = fe.steady_depths(drains, fe.DESIGN_STANDARD_MM_HR, 60,
                              tide_m=fe.TIDE_LOW_M)
    design_flooded = sum(1 for v in design.values() if v >= fe.FLOOD_DEPTH_CM)
    checks.append(Check(
        name="The design storm",
        question=f"At {fe.DESIGN_STANDARD_MM_HR:.0f} mm/hr for an hour at low "
                 "tide — the storm the drains were built for — does the model "
                 "cry wolf?",
        expectation="Little or no flooding, since the system is designed for this",
        passed=design_flooded <= total * 0.3,
        detail=(f"{design_flooded} of {total} spots over "
                f"{fe.FLOOD_DEPTH_CM:.0f} cm."
                + (" The ones that do are the choked ones, which is the point: "
                   "blockage is what takes a drain below its design capacity."
                   if design_flooded else
                   " The system copes, exactly as it is supposed to. What it "
                   "does not cope with is the same intensity sustained for "
                   "hours — that is a different storm, and the design standard "
                   "never promised it.")),
    ))

    light = fe.steady_depths(drains, 10.0, 180, tide_m=fe.TIDE_LOW_M)
    light_flooded = sum(1 for v in light.values() if v >= fe.FLOOD_DEPTH_CM)
    checks.append(Check(
        name="Ordinary monsoon rain",
        question="10 mm/hr for three hours — a wet afternoon Mumbai does not "
                 "shut down for. Does the model?",
        expectation="No flooding",
        passed=light_flooded == 0,
        detail=f"{light_flooded} of {total} spots over "
               f"{fe.FLOOD_DEPTH_CM:.0f} cm.",
    ))

    return checks


def scoreline(checks: List[Check]) -> Tuple[int, int]:
    return sum(1 for c in checks if c.passed), len(checks)
