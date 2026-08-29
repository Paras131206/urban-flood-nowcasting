"""Bandra Predictive Drainage Monitor.

The prediction lives in flood_engine.py; this file is the dashboard.

What changed from the first version, and why:

  * Risk used to be a dimensionless score per drain. It is now a water depth
    in centimetres, because "34 cm at Chimbai in 45 minutes" is something a
    person can act on and "score 0.7" is not.

  * Drains used to be scored independently. They are now a network — a drain
    receives what the drains uphill could not carry, and a drain that is full
    strangles the one above it. That is why a clean gully can still go under.

  * Rain used to come only from a slider. It now comes from a live 15-minute
    forecast by default, with the slider kept for what-if questions.

  * Every depth now carries a confidence and a risk factor, because a number
    with no error bar invites more trust than it has earned.
"""
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

import alerts
import flood_engine as fe
import hand
import terrain

st.set_page_config(page_title="Bandra Drainage Monitor", layout="wide")
st_autorefresh(interval=300_000, key="risk_timer")     # re-check every 5 minutes

BANDRA = (19.0544, 72.8402)


def render_map(fmap, height: int = 480):
    """Draw a folium map that fits its column.

    A fixed pixel width wider than the browser column gets clipped on the
    right, which is what was hiding most of the layer-control box in the top
    corner. Newer streamlit-folium takes use_container_width; older versions
    do not, so fall back to a width that fits a laptop screen rather than
    crashing.
    """
    try:
        return st_folium(fmap, height=height, use_container_width=True,
                         returned_objects=[])
    except TypeError:
        return st_folium(fmap, height=height, width=1000, returned_objects=[])


def speak(message: str) -> None:
    """Browsers block speech until the user has interacted with the page."""
    safe = (message.replace("\\", "")
                   .replace("'", "")
                   .replace('"', "")
                   .replace("\n", " "))
    components.html(
        f"<script>"
        f"window.speechSynthesis.cancel();"
        f"var m = new SpeechSynthesisUtterance('{safe}');"
        f"m.rate = 0.95; window.speechSynthesis.speak(m);"
        f"</script>",
        height=0,
    )


def spoken_briefing(rows, risks, limit: int = 8) -> str:
    """Read out every place that is going under, not just the worst one.

    The earlier version named a single location, which is exactly the wrong
    thing for someone who cannot look at the screen: if the one road it names
    is not yours, the warning has told you nothing. This walks the whole list
    worst-first, with the depth and how long you have.
    """
    serious = [r for r in rows if r["Peak_Level"] in ("HIGH", "SEVERE")]
    if not serious:
        shallow = [r for r in rows if r["Peak_Depth_cm"] >= 10]
        if not shallow:
            return ("No flooding expected anywhere in Bandra over the next "
                    "three hours.")
        return (f"No roads expected to close. {len(shallow)} spots will have "
                f"shallow standing water, the deepest at "
                f"{shallow[0]['Segment_Name']}.")

    serious.sort(key=lambda r: -r["Peak_Depth_cm"])
    parts = [f"Flood warning for {len(serious)} "
             f"location{'s' if len(serious) != 1 else ''} in Bandra."]

    for row in serious[:limit]:
        crossing = next((m for m, cm in row["Timeline"]
                         if cm >= fe.FLOOD_DEPTH_CM), None)
        if crossing is None:
            timing = ""
        elif crossing == 0:
            timing = ", already under water"
        else:
            timing = f", in {crossing} minutes"

        rise = risks.get(row["Drain_ID"], {}).get("rise_cm_per_15min", 0)
        fast = ", rising fast" if rise >= 4 else ""
        parts.append(
            f"{row['Segment_Name']}, {row['Peak_Depth_cm']:.0f} "
            f"centimetres{timing}{fast}."
        )

    if len(serious) > limit:
        parts.append(f"And {len(serious) - limit} more locations.")
    parts.append("Avoid these roads.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Control panel")
    voice_on = st.checkbox("🔈 Voice alerts", value=False)
    st.caption("Browsers block speech until you click something on the page.")

    st.divider()
    st.header("Rainfall")
    source = st.radio(
        "Source", ["Live forecast", "Manual (what-if)"], index=0,
        help="Live pulls a real 15-minute forecast for Bandra. Manual lets you "
             "ask what would happen at an intensity that is not falling today.",
    )
    manual_intensity = st.slider("Intensity (mm/hr)", 0, 200, 45,
                                 disabled=(source == "Live forecast"))

    st.divider()
    st.header("Tide")
    tide_m = st.select_slider(
        "Sea level at the outfalls",
        options=[0.5, 1.2, 2.0, 3.0, 4.0],
        value=fe.DEFAULT_TIDE_M,
        format_func=lambda t: {
            0.5: "0.5 m - low, outfalls clear",
            1.2: "1.2 m - ordinary",
            2.0: "2.0 m - rising",
            3.0: "3.0 m - high",
            4.0: "4.0 m - outfalls drowned",
        }[t],
        help="Mumbai's drains are rated for 25 mm/hr AT LOW TIDE. When the sea "
             "is higher than the outfall there is nowhere for the water to go, "
             "and the same rain does far more damage.",
    )
    st.caption(
        f"Sea-discharging drains keep "
        f"**{fe.outfall_factor(tide_m) * 100:.0f}%** of their capacity."
    )

    st.divider()
    minutes_ahead = st.select_slider(
        "Looking ahead", options=list(range(0, fe.HORIZON_MIN + 1, fe.STEP_MIN)),
        value=0, format_func=lambda m: "Now" if m == 0 else f"+{m} min",
    )


# --------------------------------------------------------------------------- #
# Rainfall
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300)
def cached_live():
    return fe.live_series(*BANDRA)


if source == "Live forecast":
    live = cached_live()
    if live["error"]:
        st.warning(
            f"Live forecast unavailable ({live['error']}). Showing zero rainfall — "
            "switch to Manual to demonstrate the model."
        )
        series = fe.flat_series(0.0)
    else:
        series = live["series"]
else:
    series = fe.flat_series(manual_intensity)

intensity_now = fe.intensity_at(series, minutes_ahead)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
st.title("Bandra Predictive Drainage Monitor")

low_delay, high_delay = fe.latency_range_min()
st.caption(
    f"⏱️ **Data delay {low_delay}-{high_delay} min.** {fe.latency_note()} "
    "The full breakdown is on *How it works → Where the data comes from*."
)

try:
    drains = fe.load_drains()
except Exception as exc:                            # noqa: BLE001
    st.error(f"Could not read bandra_capacity.csv — {exc}")
    st.stop()

rows = fe.assess(drains, series, at_minutes=minutes_ahead, tide_m=tide_m)
confidences = fe.confidence(drains, series, minutes_ahead, tide_m=tide_m)
biggest_area = max(d.total_area_m2 for d in drains.values())
risks = {
    row["Drain_ID"]: fe.risk_factor(drains[row["Drain_ID"]], row["Timeline"],
                                    minutes_ahead, biggest_area)
    for row in rows
}

df = pd.DataFrame(rows)
flooded = df[df["Depth_cm"] >= 10]
critical = df[df["Depth_cm"] >= 25]
worst = rows[0]
worst_risk_id = max(risks, key=lambda k: risks[k]["score"])
worst_risk = risks[worst_risk_id]
mean_confidence = round(sum(c["pct"] for c in confidences.values()) / len(confidences))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rain", f"{intensity_now:.1f} mm/hr",
          "live" if source == "Live forecast" else "what-if", delta_color="off")
c2.metric("Spots with standing water", len(flooded), f"of {len(df)}",
          delta_color="off")
c3.metric("Deepest", f"{worst['Peak_Depth_cm']:.0f} cm", worst["Segment_Name"],
          delta_color="off")
c4.metric("Highest risk factor", f"{worst_risk['score']}",
          f"{worst_risk['band']} - {drains[worst_risk_id].name}",
          delta_color="off")
c5.metric("Model confidence", f"{mean_confidence}%",
          "average across all spots", delta_color="off")

st.markdown("**Risk factor scale**")
scale_cols = st.columns(len(fe.RISK_FACTOR_BANDS))
for column, (floor, name, colour, meaning) in zip(scale_cols,
                                                  reversed(fe.RISK_FACTOR_BANDS)):
    ceiling = next((f - 1 for f, *_ in reversed(fe.RISK_FACTOR_BANDS) if f > floor), 100)
    count = sum(1 for r in risks.values() if r["band"] == name)
    column.markdown(
        f"<div style='border-left:6px solid {colour};padding:3px 10px;"
        f"background:{'#00000010' if count else 'transparent'}'>"
        f"<b>{floor}-{ceiling} &nbsp; {name}</b> "
        f"<span style='opacity:0.7'>({count} spot{'s' if count != 1 else ''})</span>"
        f"<br><span style='font-size:0.85em'>{meaning}</span></div>",
        unsafe_allow_html=True,
    )

with st.expander("What are 'risk factor' and 'confidence'?"):
    st.markdown(
        """
**Risk factor** is not depth. Depth answers *how deep*; risk factor answers
*how bad*, and combines how deep it gets (40%), how fast it arrives (25%) —
water that rises in ten minutes gives nobody time to move — how long it stays
(20%), and how much ground drains through that spot (15%), because closing a
trunk junction closes a district while closing a lane closes a lane.

**Confidence** is how much the answer moves when the two least certain inputs
are shaken: blockage 15 points either side of what is on file, crossed with
rainfall at 60%, 100% and 140% of forecast. Nine runs in all, and confidence is
mostly how many of them agree on the risk level rather than on the centimetre.
It measures uncertainty *inside* the model — it cannot account for a collapsed
culvert or a lorry parked over a gully, because the model does not know those
exist.
        """
    )

# --------------------------------------------------------------------------- #
# Alert
# --------------------------------------------------------------------------- #
if len(critical):
    ordered = critical.sort_values("Depth_cm", ascending=False)
    names = ", ".join(ordered["Segment_Name"].head(3))
    top = ordered.iloc[0]
    top_conf = confidences[top["Drain_ID"]]
    st.error(
        f"🚨 **Avoid: {names}** — over 25 cm of water predicted. "
        f"Deepest is {top['Segment_Name']} at {top['Depth_cm']:.0f} cm "
        f"({top_conf['pct']}% confidence, "
        f"{top_conf['low_cm']:.0f}-{top_conf['high_cm']:.0f} cm across runs)."
    )
elif len(flooded):
    st.warning(f"{len(flooded)} spots have standing water, none deep enough to "
               "close a road yet.")
else:
    st.success("No standing water predicted anywhere in the next three hours.")

# The spoken briefing covers every affected location, so it runs once here
# rather than inside whichever alert branch happened to fire.
briefing = spoken_briefing(rows, risks)
if voice_on:
    speak(briefing)
with st.expander("What the voice alert says"):
    st.write(briefing)
    st.caption(
        "Every location that goes under 15 cm in the next three hours, worst "
        "first, with the depth and how long you have. Tick **Voice alerts** in "
        "the sidebar to hear it — browsers block speech until you have clicked "
        "something on the page."
    )

# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
st.subheader("Predicted flood depth")

fmap = folium.Map(location=BANDRA, zoom_start=14, tiles="OpenStreetMap")

layer_network = folium.FeatureGroup(name="Drain network", show=True)
layer_depth = folium.FeatureGroup(name="Predicted depth", show=True)
layer_elevation = folium.FeatureGroup(name="Ground elevation", show=False)
layer_terrain = folium.FeatureGroup(name="Terrain / ground cover", show=False)


def elevation_colour(metres: float) -> str:
    """Low ground is where water ends up, so low ground is the dark colour."""
    if metres < 2:
        return "#08306B"
    if metres < 4:
        return "#2171B5"
    if metres < 8:
        return "#6BAED6"
    if metres < 15:
        return "#A1D99B"
    return "#238B45"


def sealed_colour(fraction: float) -> str:
    """Redder means more concrete, which means more runoff."""
    if fraction >= 0.85:
        return "#67000D"
    if fraction >= 0.70:
        return "#CB181D"
    if fraction >= 0.55:
        return "#FB6A4A"
    return "#238B45"


# Draw the network first, so the markers sit on top of it.
for row in rows:
    drain = drains[row["Drain_ID"]]
    if drain.downstream:
        other = drains[drain.downstream]
        folium.PolyLine(
            [(drain.lat, drain.lon), (other.lat, other.lon)],
            color="#5A6B7F", weight=2, opacity=0.55, dash_array="6",
        ).add_to(layer_network)

for row in rows:
    depth = row["Depth_cm"]
    did = row["Drain_ID"]
    drain = drains[did]
    conf = confidences[did]
    risk = risks[did]
    mix = terrain.mix_for(did)
    sealed = terrain.sealed_fraction(did)

    # --- Depth ------------------------------------------------------------ #
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8 + min(depth, 100) / 6,             # bigger pool, bigger circle
        color=row["Color"],
        fill=True, fill_opacity=0.65, weight=2,
        tooltip=folium.Tooltip(
            f"<b>{row['Segment_Name']}</b><br>"
            f"{depth:.0f} cm now &middot; peak {row['Peak_Depth_cm']:.0f} cm "
            f"in {row['Minutes_To_Peak']} min<br>"
            f"<b>Risk factor {risk['score']}</b> ({risk['band']})<br>"
            f"Confidence {conf['pct']}% "
            f"({conf['low_cm']:.0f}-{conf['high_cm']:.0f} cm)<br>"
            + (f"HAND {drain.hand_m:.1f} m above drainage &middot; "
               if drain.hand_m is not None else "")
            + f"Ground {drain.elevation_m:.1f} m &middot; "
            f"{sealed * 100:.0f}% sealed &middot; C={row['Runoff_C']}<br>"
            f"{row['Drain_Type']} &middot; {row['Blockage_Pct']}% blocked<br>"
            f"drains to {row['Drains_To']} &middot; "
            f"serves {row['Upstream_Area_ha']} ha above"
        ),
    ).add_to(layer_depth)

    # --- Elevation -------------------------------------------------------- #
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=14,
        color=elevation_colour(drain.elevation_m),
        fill=True, fill_opacity=0.75, weight=2,
        tooltip=folium.Tooltip(
            f"<b>{drain.name}</b><br>"
            f"Ground elevation <b>{drain.elevation_m:.1f} m</b><br>"
            f"Water can stand up to {drain.max_pond_cm:.0f} cm here before it "
            f"spreads overland<br>"
            f"{round(drain.retention * 100)}% of an overflow stays put rather "
            f"than running off"
        ),
    ).add_to(layer_elevation)

    # --- Terrain, as percentages ------------------------------------------ #
    breakdown = "<br>".join(
        f"&nbsp;&nbsp;{terrain.LABELS[k]}: <b>{v * 100:.0f}%</b>"
        for k, v in sorted(mix.items(), key=lambda kv: -kv[1]) if v >= 0.01
    )
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=14,
        color=sealed_colour(sealed),
        fill=True, fill_opacity=0.75, weight=2,
        tooltip=folium.Tooltip(
            f"<b>{drain.name}</b><br>"
            f"<b>{sealed * 100:.0f}% sealed surface</b><br>{breakdown}<br>"
            f"Runoff coefficient <b>{drain.runoff_c}</b> — of every 100 mm of "
            f"rain, {drain.runoff_c * 100:.0f} mm runs off"
        ),
    ).add_to(layer_terrain)

for layer in (layer_elevation, layer_terrain, layer_network, layer_depth):
    layer.add_to(fmap)
folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)

render_map(fmap, height=520)

legend_a, legend_b = st.columns(2)
with legend_a:
    st.markdown("**Ground elevation** (switch the layer on in the map)")
    st.markdown(
        " &nbsp; ".join(
            f"<span style='background:{c};color:#fff;padding:2px 7px;"
            f"border-radius:3px;font-size:0.8em'>{label}</span>"
            for c, label in [("#08306B", "under 2 m"), ("#2171B5", "2-4 m"),
                             ("#6BAED6", "4-8 m"), ("#A1D99B", "8-15 m"),
                             ("#238B45", "over 15 m")]),
        unsafe_allow_html=True)
    st.caption("Dark blue is low ground — where the water ends up.")
with legend_b:
    st.markdown("**Ground cover** (percentages in the tooltip)")
    st.markdown(
        " &nbsp; ".join(
            f"<span style='background:{c};color:#fff;padding:2px 7px;"
            f"border-radius:3px;font-size:0.8em'>{label}</span>"
            for c, label in [("#67000D", "85%+ sealed"), ("#CB181D", "70-85%"),
                             ("#FB6A4A", "55-70%"), ("#238B45", "under 55%")]),
        unsafe_allow_html=True)
    st.caption("Redder is more concrete, and more concrete means more runoff.")

st.caption(
    "Dashed lines show which drain flows into which; circle size is depth. "
    "Use the box on the map to switch between depth, elevation and ground cover."
)

# --------------------------------------------------------------------------- #
# SMS
# --------------------------------------------------------------------------- #
st.subheader("SMS alerts")

message = alerts.compose(rows, risks, minutes_ahead)
gateway = alerts.configured_gateway()
people = alerts.subscribers()

if gateway:
    st.success(f"Gateway configured: **{gateway}**. Messages will be sent for real.")
else:
    st.info(
        "**No SMS gateway configured**, so messages are composed and queued "
        "rather than sent. Everything below is exactly what would go out — "
        "add a gateway key and the same code delivers it."
    )

left, right = st.columns([1, 1])

with left:
    st.markdown("**The message**")
    if message is None:
        st.write("Nothing worth sending. No location reaches HIGH in the next "
                 "three hours.")
    else:
        st.code(message["body"], language=None)
        m1, m2, m3 = st.columns(3)
        m1.metric("Level", message["level"])
        m2.metric("Warning time",
                  "already" if message["lead_time_min"] == 0
                  else f"{message['lead_time_min']} min"
                  if message["lead_time_min"] is not None else "3 h",
                  delta_color="off")
        m3.metric("Length", f"{message['characters']} chars",
                  f"{message['segments']} SMS", delta_color="off")
        st.caption(
            "It warns on the **peak in the next three hours**, not the water "
            "already on the road. A text that arrives once you are sitting in "
            "it is a weather report, not a warning."
        )

with right:
    st.markdown("**Who gets it**")
    number = st.text_input("Phone number", placeholder="+91 98765 43210")
    label = st.text_input("Label (optional)", placeholder="Control room")

    add, send, reset = st.columns(3)
    if add.button("Subscribe"):
        if alerts.looks_like_a_phone_number(number):
            alerts.subscribe(number, label)
            st.success(f"{number} subscribed.")
            people = alerts.subscribers()
        else:
            st.error("That does not look like a phone number.")

    if send.button("Send now", type="primary", disabled=message is None):
        if not people:
            st.warning("Nobody is subscribed yet.")
        else:
            report = alerts.dispatch(message, force=True)
            if report["sent"]:
                st.success(f"Sent to {len(report['sent'])} number(s) "
                           f"via {report['gateway']}.")
            if report["failed"]:
                st.warning(
                    f"{len(report['failed'])} queued rather than delivered — "
                    + report["failed"][0]["error"]
                )

    if reset.button("Reset"):
        alerts.reset_escalation()
        st.info("Escalation history cleared, so the next alert sends again.")

    if people:
        st.dataframe(
            pd.DataFrame([{
                "Number": p["number"], "Label": p.get("label", ""),
                "Last told": p.get("last_level") or "never",
            } for p in people]),
            hide_index=True, width="stretch",
        )
    else:
        st.caption("No subscribers yet.")

with st.expander("How the alerting avoids becoming noise"):
    st.markdown(
        """
A warning system that texts every five minutes gets muted, and a muted warning
system is worse than none. So a subscriber is messaged only when the situation
gets **worse** than the last thing they were told — MEDIUM to SEVERE sends,
SEVERE back to HIGH does not. **Send now** overrides that for demonstrations,
and **Reset** clears the history so it can be shown twice.

Only HIGH and SEVERE are worth a message. Nobody wants a text about a puddle.
        """
    )
    st.markdown(alerts.SETUP_HELP)

    log = alerts.outbox()
    if log:
        st.markdown("**Recently sent or queued**")
        st.dataframe(
            pd.DataFrame([{
                "When": e["at"].replace("T", " ")[:16],
                "To": e["number"], "Level": e["level"],
                "Delivered": "yes" if e["delivered"] else "queued",
                "Message": e["body"],
            } for e in log[:15]]),
            hide_index=True, width="stretch",
        )

# --------------------------------------------------------------------------- #
# The next three hours
# --------------------------------------------------------------------------- #
st.subheader("The next three hours")

left, right = st.columns([1, 1])

with left:
    st.markdown("**Rain forecast**")
    st.line_chart(
        pd.DataFrame(series, columns=["Minutes ahead", "mm/hr"]
                     ).set_index("Minutes ahead"),
        height=220,
    )
    st.caption(
        "15-minute resolution, three hours out. "
        + ("Live from Open-Meteo." if source == "Live forecast"
           else "Flat what-if intensity from the slider.")
    )

with right:
    st.markdown("**Predicted depth at the four worst spots**")
    depth_frame = pd.DataFrame(
        {r["Segment_Name"]: dict(r["Timeline"]) for r in rows[:4]}
    )
    depth_frame.index.name = "Minutes ahead"
    st.line_chart(depth_frame, height=220)
    st.caption("Depth in centimetres. The flooding line is 15 cm.")

st.markdown("**When does each spot go under?**")
timing = []
for row in rows:
    crossing = next((m for m, cm in row["Timeline"] if cm >= fe.FLOOD_DEPTH_CM), None)
    risk = risks[row["Drain_ID"]]
    timing.append({
        "Spot": row["Segment_Name"],
        "Goes under 15 cm": "not in 3 h" if crossing is None else
                            ("already" if crossing == 0 else f"in {crossing} min"),
        "Peak depth": f"{row['Peak_Depth_cm']:.0f} cm",
        "Peak at": f"+{row['Minutes_To_Peak']} min",
        "Rising": f"{risk['rise_cm_per_15min']:+.1f} cm / 15 min",
        "Time above 15 cm": f"{risk['minutes_above_flood']} min",
        "Risk factor": risk["score"],
        "Confidence": f"{confidences[row['Drain_ID']]['pct']}%",
    })
st.dataframe(pd.DataFrame(timing), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Why
# --------------------------------------------------------------------------- #
st.subheader("Why is this happening?")

pick = st.selectbox(
    "Spot", [r["Drain_ID"] for r in rows],
    format_func=lambda d: f"{drains[d].name} - risk factor {risks[d]['score']}",
)

trace = fe.diagnose(drains, series, minutes_ahead, tide_m=tide_m)
picked_risk = risks[pick]
picked_conf = confidences[pick]

a, b, c = st.columns(3)
a.metric("Depth", f"{picked_risk['depth_cm']:.0f} cm",
         f"peak {picked_risk['peak_cm']:.0f} cm", delta_color="off")
b.metric("Risk factor", picked_risk["score"], picked_risk["band"],
         delta_color="off")
c.metric("Confidence", f"{picked_conf['pct']}%",
         f"{picked_conf['low_cm']:.0f}-{picked_conf['high_cm']:.0f} cm "
         "across 9 runs", delta_color="off")

st.markdown(f"**Risk:** {picked_risk['why']}")
st.markdown(f"**Confidence:** {picked_conf['why']}")

st.markdown("**Why the water is there:**")
for line in fe.explain(drains, pick, trace, intensity_now):
    st.markdown(f"- {line}")

st.caption(
    f"Ground cover here: {terrain.describe(pick)} — "
    f"runoff coefficient {drains[pick].runoff_c}."
)

# --------------------------------------------------------------------------- #
# Detail
# --------------------------------------------------------------------------- #
with st.expander("Every drain, in full"):
    table = df[[
        "Segment_Name", "Drain_Type", "Blockage_Pct", "Runoff_C",
        "Design_Capacity_m3s", "Real_Capacity_m3s", "Drains_To",
        "Depth_cm", "Peak_Depth_cm", "Risk_Level",
    ]].copy()
    table["Risk factor"] = [risks[r["Drain_ID"]]["score"] for r in rows]
    table["Confidence"] = [f"{confidences[r['Drain_ID']]['pct']}%" for r in rows]
    st.dataframe(
        table.rename(columns={
            "Segment_Name": "Drain", "Drain_Type": "Type",
            "Blockage_Pct": "Blocked %", "Runoff_C": "Runoff C",
            "Design_Capacity_m3s": "Design m³/s",
            "Real_Capacity_m3s": "Real m³/s",
            "Drains_To": "Flows into", "Depth_cm": "Now (cm)",
            "Peak_Depth_cm": "Peak (cm)", "Risk_Level": "Level",
        }),
        hide_index=True, width="stretch",
    )
    st.caption(
        "**Real m³/s** is after silt and after the tide. At the current tide "
        f"({tide_m} m) a sea-discharging drain keeps "
        f"{fe.outfall_factor(tide_m) * 100:.0f}% of what silt left it."
    )

with st.expander("How much rain does each spot take?"):
    st.caption(
        "The intensity at which each spot goes under 15 cm, solved through the "
        "whole network — a drain floods on water that fell uphill. Duration "
        "matters as much as intensity, which is why both columns are here."
    )
    thresholds = []
    for row in rows:
        one_hour = fe.floods_at_mm_hr(drains, row["Drain_ID"], 60, tide_m=tide_m)
        three_hour = fe.floods_at_mm_hr(drains, row["Drain_ID"], 180, tide_m=tide_m)
        thresholds.append({
            "Drain": row["Segment_Name"],
            "Blocked %": row["Blockage_Pct"],
            "Floods after 1 h of": "—" if one_hour == float("inf") else f"{one_hour:.0f} mm/hr",
            "Floods after 3 h of": "—" if three_hour == float("inf") else f"{three_hour:.0f} mm/hr",
            "Share of design standard": (
                "—" if three_hour == float("inf")
                else f"{three_hour / fe.DESIGN_STANDARD_MM_HR:.2f}"
            ),
        })
    st.dataframe(pd.DataFrame(thresholds), hide_index=True, width="stretch")
    st.info(
        f"Mumbai's drains are rated for **{fe.DESIGN_STANDARD_MM_HR:.0f} mm/hr "
        "at low tide**. Several of these spots survive that comfortably for an "
        "hour and fail well below it over three — which is the real shape of "
        "the problem. Both 2025 storms averaged *below* the design standard "
        "and still flooded the city, because the rain kept falling."
    )

with st.expander("What this model is built on"):
    st.markdown(
        """
**The engineering.** The rational method for runoff and the area^(5/3)
capacity reduction for blockage are standard drainage design. On top of those
sit the parts that make this a network model rather than a scorecard: water
routed downhill drain by drain, surcharge propagating back upstream when a
drain runs full, the tide closing the outfalls, and a depth in centimetres
rather than a dimensionless score.

**The live data.** A genuine 15-minute rainfall forecast from Open-Meteo, and
a runoff coefficient composed from each catchment's actual land cover rather
than assumed constant across the whole suburb.

**How it is checked.** Against storms that really happened in the 2024 and
2025 monsoons, and against Mumbai's own published design standard of 25 mm/hr
at low tide. Six checks, all passing — see *How it works*. The result worth
knowing: both 2025 storms flooded the city on rain **below** that standard,
because it fell for hours, and the model catches both.

**Where it goes next.** Water level sensors at twenty junctions through one
monsoon would turn every consistency check into a scored error in centimetres.
Municipal CCTV records would replace the estimated blockage figures, and a
land-cover raster the estimated terrain mix. All three drop into the existing
code without changing any of the logic above.
        """
    )
