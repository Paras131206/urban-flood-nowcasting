"""Flood warnings sent to a phone.

The honest position on SMS
--------------------------
Sending a real text message costs money and needs an account with a gateway.
There is no free path, and any project claiming otherwise is either not sending
messages or not telling you what it costs. So this module does three things:

1. **Composes** the message properly — short, specific, actionable, and worth
   the 160 characters it occupies. That work is independent of who delivers it.
2. **Sends** it through Twilio or an Indian gateway (MSG91, Fast2SMS) if
   credentials are configured in the environment.
3. **Queues it visibly** when they are not, so the app demonstrates exactly
   what would go out, to whom, and when — rather than a button that silently
   does nothing.

Point 3 matters. A demo where the SMS button appears to work but sends nothing
is worse than one that says "this is the message, here is where it would go,
add a gateway key to send it". The second is honest and shows the same
engineering.

What stops it becoming spam
---------------------------
A warning system that texts every five minutes gets muted, and a muted warning
system is worse than none. So a subscriber is only messaged when the situation
gets *worse* than the last thing they were told — an escalation, not a repeat.
Going from HIGH back to MEDIUM sends nothing; MEDIUM to SEVERE sends.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import flood_engine as fe

SUBSCRIBERS_PATH = "sms_subscribers.json"
OUTBOX_PATH = "sms_outbox.json"

# Only these levels are worth a text. Nobody wants a message about a puddle.
ALERTABLE = ("HIGH", "SEVERE")
SEVERITY = {name: index for index, name in enumerate(fe.LEVEL_ORDER)}


# --------------------------------------------------------------------------- #
# Composing
# --------------------------------------------------------------------------- #
def compose(rows: List[dict], risks: Dict[str, dict], minutes_ahead: int = 0,
            area: str = "Bandra") -> Optional[dict]:
    """The message itself, or None when there is nothing worth sending.

    It alerts on the **peak within the next three hours**, not on the water
    already on the road. That is the entire point of a nowcast: a text saying
    "SV Road is flooded" that arrives when you are already sitting in it is not
    a warning, it is a weather report. Using the current depth also meant the
    message went quiet at minute zero, when the model has barely begun routing
    water and every level still reads LOW.

    Written to be read on a lock screen in the rain: what, where, how deep,
    how long until it happens. No preamble, no branding, no "please be
    advised".
    """
    serious = [r for r in rows if r["Peak_Level"] in ALERTABLE]
    if not serious:
        return None

    serious.sort(key=lambda r: -r["Peak_Depth_cm"])
    worst = serious[0]
    level = max(serious, key=lambda r: SEVERITY[r["Peak_Level"]])["Peak_Level"]

    # How long until the first of them goes under, so the message leads with
    # the lead time rather than burying it.
    soonest = min(
        (next((m for m, cm in r["Timeline"] if cm >= fe.FLOOD_DEPTH_CM), 999)
         for r in serious),
        default=999,
    )
    if soonest <= 0:
        when = "now"
    elif soonest < 900:
        when = f"in {soonest} min"
    else:
        when = "within 3h"

    names = [r["Segment_Name"] for r in serious[:3]]
    where = ", ".join(names)
    if len(serious) > 3:
        where += f" +{len(serious) - 3} more"

    body = (f"{area} flood alert ({level}) {when}: {where}. "
            f"Deepest {worst['Segment_Name']} {worst['Peak_Depth_cm']:.0f}cm")

    rising = [r for r in serious
              if risks.get(r["Drain_ID"], {}).get("rise_cm_per_15min", 0) >= 3]
    if rising:
        body += f". Rising fast at {rising[0]['Segment_Name']}"
    body += ". Avoid these roads."

    return {
        "level": level,
        "body": body,
        "spots": [r["Segment_Name"] for r in serious],
        "worst": worst["Segment_Name"],
        "worst_cm": round(worst["Peak_Depth_cm"]),
        "lead_time_min": None if soonest >= 900 else soonest,
        "minutes_ahead": minutes_ahead,
        "characters": len(body),
        "segments": (len(body) // 160) + 1,
    }


# --------------------------------------------------------------------------- #
# Who gets it
# --------------------------------------------------------------------------- #
def _read(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:                                   # noqa: BLE001
        return []


def _write(path: str, rows: list) -> None:
    try:
        with open(path, "w") as fh:
            json.dump(rows, fh, indent=2)
    except Exception:                                   # noqa: BLE001
        pass                        # a read-only deployment must not crash


def subscribers(path: str = SUBSCRIBERS_PATH) -> List[dict]:
    return _read(path)


def subscribe(number: str, label: str = "", path: str = SUBSCRIBERS_PATH) -> List[dict]:
    number = number.strip()
    people = subscribers(path)
    if not any(p["number"] == number for p in people):
        people.append({"number": number, "label": label.strip(),
                       "last_level": None, "last_sent": None})
        _write(path, people)
    return people


def unsubscribe(number: str, path: str = SUBSCRIBERS_PATH) -> List[dict]:
    people = [p for p in subscribers(path) if p["number"] != number.strip()]
    _write(path, people)
    return people


def looks_like_a_phone_number(number: str) -> bool:
    """Loose on purpose — this validates shape, not that anyone is there.

    A gateway will reject a bad number properly. The point here is to catch
    an obvious typo before spending a message credit on it.
    """
    cleaned = number.strip().replace(" ", "").replace("-", "")
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    return cleaned.isdigit() and 8 <= len(cleaned) <= 15


def should_send(person: dict, level: str) -> bool:
    """Only on escalation. A warning that repeats itself gets muted."""
    previous = person.get("last_level")
    if previous is None:
        return True
    return SEVERITY.get(level, 0) > SEVERITY.get(previous, 0)


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
def configured_gateway() -> Optional[str]:
    """Which gateway the environment is set up for, if any."""
    if os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("TWILIO_AUTH_TOKEN"):
        return "twilio"
    if os.environ.get("MSG91_AUTHKEY"):
        return "msg91"
    if os.environ.get("FAST2SMS_API_KEY"):
        return "fast2sms"
    return None


def _send_twilio(number: str, body: str) -> dict:
    import requests

    sid = os.environ["TWILIO_ACCOUNT_SID"]
    token = os.environ["TWILIO_AUTH_TOKEN"]
    sender = os.environ.get("TWILIO_FROM_NUMBER", "")
    if not sender:
        return {"ok": False, "error": "TWILIO_FROM_NUMBER is not set."}

    response = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
        auth=(sid, token),
        data={"From": sender, "To": number, "Body": body},
        timeout=15,
    )
    if response.status_code >= 400:
        return {"ok": False, "error": f"Twilio {response.status_code}: "
                                      f"{response.text[:200]}"}
    return {"ok": True, "id": response.json().get("sid", ""), "gateway": "twilio"}


def _send_msg91(number: str, body: str) -> dict:
    import requests

    response = requests.post(
        "https://api.msg91.com/api/v2/sendsms",
        headers={"authkey": os.environ["MSG91_AUTHKEY"],
                 "Content-Type": "application/json"},
        json={"sender": os.environ.get("MSG91_SENDER", "AQGRID"),
              "route": "4",
              "country": "91",
              "sms": [{"message": body, "to": [number.lstrip("+")]}]},
        timeout=15,
    )
    if response.status_code >= 400:
        return {"ok": False, "error": f"MSG91 {response.status_code}: "
                                      f"{response.text[:200]}"}
    return {"ok": True, "id": "", "gateway": "msg91"}


def _send_fast2sms(number: str, body: str) -> dict:
    import requests

    response = requests.post(
        "https://www.fast2sms.com/dev/bulkV2",
        headers={"authorization": os.environ["FAST2SMS_API_KEY"]},
        data={"route": "q", "message": body,
              "numbers": number.lstrip("+").removeprefix("91")},
        timeout=15,
    )
    if response.status_code >= 400:
        return {"ok": False, "error": f"Fast2SMS {response.status_code}: "
                                      f"{response.text[:200]}"}
    return {"ok": True, "id": "", "gateway": "fast2sms"}


def send_one(number: str, body: str) -> dict:
    """Deliver one message, or explain precisely why it was not delivered."""
    gateway = configured_gateway()
    if gateway is None:
        return {"ok": False, "queued": True, "gateway": None,
                "error": "No SMS gateway configured, so the message was "
                         "queued rather than sent."}
    try:
        import requests                                 # noqa: F401
    except ImportError:
        return {"ok": False, "queued": True, "gateway": gateway,
                "error": "The requests package is not installed."}

    try:
        if gateway == "twilio":
            return _send_twilio(number, body)
        if gateway == "msg91":
            return _send_msg91(number, body)
        return _send_fast2sms(number, body)
    except Exception as exc:                            # noqa: BLE001
        return {"ok": False, "queued": True, "gateway": gateway,
                "error": f"{exc.__class__.__name__}: {exc}"}


def dispatch(message: dict, people: Optional[List[dict]] = None,
             force: bool = False,
             subscribers_path: str = SUBSCRIBERS_PATH,
             outbox_path: str = OUTBOX_PATH) -> dict:
    """Send to everyone the situation has got worse for.

    Returns a report rather than raising: which numbers were sent to, which
    were skipped because they had already been told, and which failed. `force`
    overrides the escalation rule, for the demo button.
    """
    people = subscribers(subscribers_path) if people is None else people
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    sent, skipped, failed = [], [], []
    outbox = _read(outbox_path)

    for person in people:
        if not force and not should_send(person, message["level"]):
            skipped.append({"number": person["number"],
                            "why": f"already told at {person.get('last_level')}"})
            continue

        result = send_one(person["number"], message["body"])
        record = {
            "at": stamp,
            "number": person["number"],
            "label": person.get("label", ""),
            "level": message["level"],
            "body": message["body"],
            "delivered": bool(result.get("ok")),
            "gateway": result.get("gateway"),
            "error": result.get("error"),
        }
        outbox.append(record)

        if result.get("ok"):
            sent.append(record)
        else:
            failed.append(record)
        person["last_level"] = message["level"]
        person["last_sent"] = stamp

    _write(subscribers_path, people)
    _write(outbox_path, outbox[-200:])          # keep the log from growing forever

    return {
        "message": message,
        "sent": sent,
        "skipped": skipped,
        "failed": failed,
        "gateway": configured_gateway(),
        "outbox_size": len(outbox),
    }


def outbox(path: str = OUTBOX_PATH) -> List[dict]:
    return list(reversed(_read(path)))


def reset_escalation(path: str = SUBSCRIBERS_PATH) -> None:
    """Forget who has been told what, so a demo can be run twice."""
    people = subscribers(path)
    for person in people:
        person["last_level"] = None
    _write(path, people)


SETUP_HELP = """
**To send real messages**, set these before starting Streamlit and restart it.

*Twilio* — free trial credit, sends to numbers you have verified:

    export TWILIO_ACCOUNT_SID=ACxxxxxxxx
    export TWILIO_AUTH_TOKEN=xxxxxxxx
    export TWILIO_FROM_NUMBER=+15005550006

*MSG91* (Indian gateway):

    export MSG91_AUTHKEY=xxxxxxxx
    export MSG91_SENDER=AQGRID

*Fast2SMS* (Indian gateway):

    export FAST2SMS_API_KEY=xxxxxxxx

With none of these set the app still composes every message and shows exactly
what would go out, to whom — it just does not spend money doing it.
"""
