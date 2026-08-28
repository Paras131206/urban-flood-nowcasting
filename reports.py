"""Photographs from the street, and what the model does with them.

A resident standing in the water knows something the model does not. This
turns that into an input.

The loop, honestly described
----------------------------
Someone uploads a photo of a flooded road. Three things happen:

1. **Where.** If the phone wrote GPS into the image, it is read straight out
   of the EXIF. Most messaging apps strip it, so there is a manual fallback.
2. **What.** If a Gemini API key is configured, the image is described: is
   there standing water, roughly how deep against the usual references — kerb,
   wheel, knee. Without a key the reporter says so themselves.
3. **So what.** The report is compared against what the model predicted for
   that spot at that time. A consistent gap is evidence about the input the
   model is least sure of: how blocked the drain is.

What this is not
----------------
This does not train a neural network, and calling it that would be a lie. It
does something narrower and more defensible: it solves for the blockage figure
that would have reproduced the depth people actually reported, and offers that
as a correction to the CSV. That is inverse calibration — the same thing a
hydrologist does by hand, done automatically and shown with its working.

Genuine learning would need thousands of labelled photographs with surveyed
depths. This is the mechanism that would collect them.
"""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import flood_engine as fe

REPORTS_PATH = "field_reports.json"

# How much a photo can move the blockage estimate. Even a confident report is
# one person's view of one puddle, so it nudges rather than overrides — and it
# takes several agreeing reports to move the number far.
MAX_BLOCKAGE_SHIFT = 0.25
REPORTS_FOR_FULL_WEIGHT = 5


# --------------------------------------------------------------------------- #
# Where was it taken
# --------------------------------------------------------------------------- #
def read_exif_gps(data: bytes) -> Optional[Tuple[float, float]]:
    """Pull latitude and longitude out of a photo, if the camera wrote them.

    Returns None rather than raising for anything at all — a missing tag, a
    stripped header, no Pillow installed. A photo without coordinates is
    normal, not an error.
    """
    try:
        import io
        from PIL import Image, ExifTags
    except Exception:                                   # noqa: BLE001
        return None

    try:
        image = Image.open(io.BytesIO(data))
        exif = image.getexif()
        if not exif:
            return None
        gps_tag = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), None)
        gps = exif.get_ifd(gps_tag) if gps_tag else None
        if not gps:
            return None

        def to_degrees(value) -> float:
            d, m, s = (float(x) for x in value)
            return d + m / 60.0 + s / 3600.0

        lat = to_degrees(gps[2])
        if str(gps.get(1, "N")).upper().startswith("S"):
            lat = -lat
        lon = to_degrees(gps[4])
        if str(gps.get(3, "E")).upper().startswith("W"):
            lon = -lon
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return round(lat, 6), round(lon, 6)
    except Exception:                                   # noqa: BLE001
        return None


def nearest_drain(drains: Dict[str, fe.Drain], lat: float, lon: float
                  ) -> Tuple[str, float]:
    """Which drain this photo belongs to, and how far away it is in metres."""
    best = min(drains.values(),
               key=lambda d: fe.haversine_m((lat, lon), (d.lat, d.lon)))
    return best.drain_id, round(fe.haversine_m((lat, lon), (best.lat, best.lon)))


# --------------------------------------------------------------------------- #
# What is in the picture
# --------------------------------------------------------------------------- #
GEMINI_PROMPT = """You are looking at a photograph of an Indian city street,
sent in by a resident reporting waterlogging.

Answer ONLY with a JSON object, no prose and no markdown fence:

{
  "is_flooded": true or false,
  "depth_cm": integer best estimate of standing water depth in centimetres,
  "reference": "what you judged the depth against, e.g. kerb height, car wheel, knee",
  "confidence": "high" | "medium" | "low",
  "notes": "one short sentence a control room would find useful"
}

Judging depth: a kerb is about 15 cm, a car wheel centre about 35 cm, an adult
knee about 50 cm, a car bonnet about 75 cm. If there is no standing water set
is_flooded false and depth_cm 0. If the picture is too dark or too close to
judge, say so in notes and set confidence low."""


def describe_with_gemini(image_bytes: bytes, mime_type: str = "image/jpeg",
                         api_key: Optional[str] = None,
                         model: Optional[str] = None) -> dict:
    """Ask Gemini what the photo shows. Degrades to a clear message, never a guess."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"ok": False,
                "error": "No GEMINI_API_KEY set, so the photo was not analysed."}

    try:
        import requests
    except ImportError:
        return {"ok": False, "error": "The requests package is not installed."}

    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")

    payload = {
        "contents": [{
            "parts": [
                {"text": GEMINI_PROMPT},
                {"inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300},
    }

    try:
        response = requests.post(url, params={"key": api_key}, json=payload,
                                 timeout=25)
        response.raise_for_status()
        body = response.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:                            # noqa: BLE001
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned.strip())
    except Exception:                                   # noqa: BLE001
        return {"ok": False, "error": "Gemini replied, but not with usable JSON.",
                "raw": text[:400]}

    return {
        "ok": True,
        "is_flooded": bool(parsed.get("is_flooded")),
        "depth_cm": int(parsed.get("depth_cm") or 0),
        "reference": str(parsed.get("reference", "")),
        "confidence": str(parsed.get("confidence", "low")),
        "notes": str(parsed.get("notes", "")),
        "model": model,
    }


# --------------------------------------------------------------------------- #
# Storing what people send
# --------------------------------------------------------------------------- #
def load_reports(path: str = REPORTS_PATH) -> List[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:                                   # noqa: BLE001
        return []


def save_report(report: dict, path: str = REPORTS_PATH) -> List[dict]:
    reports = load_reports(path)
    report.setdefault("at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    reports.append(report)
    try:
        with open(path, "w") as fh:
            json.dump(reports, fh, indent=2)
    except Exception:                                   # noqa: BLE001
        pass                        # a read-only deployment must not crash
    return reports


def reports_for(reports: List[dict], drain_id: str) -> List[dict]:
    return [r for r in reports if r.get("drain_id") == drain_id]


# --------------------------------------------------------------------------- #
# Learning from them
# --------------------------------------------------------------------------- #
def blockage_that_explains(drains: Dict[str, fe.Drain], drain_id: str,
                           observed_cm: float, intensity_mm_hr: float,
                           duration_min: int = 60,
                           tide_m: float = fe.DEFAULT_TIDE_M) -> Optional[float]:
    """What blockage would the drain need for the model to match the photo?

    Bisection on the blockage of one drain, holding everything else fixed and
    routing through the whole network each time — because changing one drain
    changes what reaches the ones below it.

    Returns None when no blockage explains the observation. That is a useful
    answer, not a failure: it means the gap is something the model does not
    represent at all, and a nudge to the blockage figure would be papering
    over it.
    """
    if intensity_mm_hr <= 0 or observed_cm < 0:
        return None

    def depth_at(blockage: float) -> float:
        trial = fe._clone_with_blockage(drains, 0.0)
        trial[drain_id].blockage = min(max(blockage, 0.0), 0.95)
        return fe.steady_depths(trial, intensity_mm_hr, duration_min,
                                tide_m=tide_m).get(drain_id, 0.0)

    lowest, highest = depth_at(0.0), depth_at(0.95)
    if not (lowest - 0.5 <= observed_cm <= highest + 0.5):
        return None

    lo, hi = 0.0, 0.95
    for _ in range(24):
        mid = (lo + hi) / 2
        if depth_at(mid) < observed_cm:
            lo = mid
        else:
            hi = mid
    return round(hi, 3)


def calibration(drains: Dict[str, fe.Drain], reports: List[dict],
                drain_id: str) -> Optional[dict]:
    """A suggested blockage correction for one drain, with its working shown.

    Only reports that recorded the rainfall at the time can be used — a depth
    without the rain that caused it says nothing about the pipe.
    """
    usable = [r for r in reports_for(reports, drain_id)
              if r.get("observed_cm") is not None and r.get("intensity_mm_hr")]
    if not usable:
        return None

    drain = drains[drain_id]
    solved = []
    for report in usable:
        answer = blockage_that_explains(
            drains, drain_id,
            float(report["observed_cm"]), float(report["intensity_mm_hr"]),
            duration_min=int(report.get("duration_min", 60)),
        )
        if answer is not None:
            solved.append(answer)

    if not solved:
        return {
            "drain_id": drain_id,
            "name": drain.name,
            "reports": len(usable),
            "explained": 0,
            "recorded": drain.blockage,
            "implied": None,
            "suggested": drain.blockage,
            "weight": 0.0,
            "note": ("No blockage value reproduces what was reported. The gap "
                     "is something the model does not represent — an obstructed "
                     "outfall, a pump that was off, or a depth judged from a "
                     "photograph that does not show the deepest point. Worth a "
                     "human looking at, not worth silently changing a number "
                     "for."),
        }

    implied = sum(solved) / len(solved)
    # Weight by how many reports agree, capped so a single photo cannot rewrite
    # a survey figure.
    weight = min(len(solved) / REPORTS_FOR_FULL_WEIGHT, 1.0)
    shift = max(min(implied - drain.blockage, MAX_BLOCKAGE_SHIFT),
                -MAX_BLOCKAGE_SHIFT) * weight
    suggested = round(min(max(drain.blockage + shift, 0.0), 0.95), 3)

    return {
        "drain_id": drain_id,
        "name": drain.name,
        "reports": len(usable),
        "explained": len(solved),
        "recorded": drain.blockage,
        "implied": round(implied, 3),
        "suggested": suggested,
        "weight": round(weight, 2),
        "note": (
            f"{len(solved)} report(s) imply about "
            f"{round(implied * 100)}% blockage against the "
            f"{round(drain.blockage * 100)}% on file. With "
            f"{len(solved)} of {REPORTS_FOR_FULL_WEIGHT} reports needed for "
            f"full weight, that moves the estimate to "
            f"{round(suggested * 100)}%."
        ),
    }


def all_calibrations(drains: Dict[str, fe.Drain],
                     reports: List[dict]) -> List[dict]:
    out = []
    for drain_id in drains:
        result = calibration(drains, reports, drain_id)
        if result:
            out.append(result)
    return out
