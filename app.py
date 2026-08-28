"""Bandra Predictive Drainage Monitor.

The prediction lives in flood_engine.py; this file is the dashboard.

What changed from the first version, and why:

  * Risk used to be a dimensionless score per drain. It is now a water depth
    in centimetres, because "34 cm at Chimbai in 45 minutes" is something a
    person can act on and "score 0.7" is not.

  * Drains used to be scored independently. They are now a network - a drain
    receives what the drains uphill could not carry. That is why a clean gully
    can still go under.

  * Rain used to come only from a slider. It now comes from a live 15-minute
    forecast by default, with the slider kept for what-if questions.
"""
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

import flood_engine as fe

st.set_page_config(page_title="Bandra Drainage Monitor", layout="wide")
st_autorefresh(interval=300_000, key="risk_timer")

BANDRA = (19.0544, 72.8402)


def speak(message: str) -> None:
    """Browsers block speech until the user has interacted with the page."""
    safe = message.replace("'", "").replace("\n", " ")
    components.html(
        f"<script>"
        f"var m = new SpeechSynthesisUtterance('{safe}');"
        f"m.rate = 0.92; window.speechSynthesis.speak(m);"
        f"</script>",
        height=0,
    )


with st.sidebar:
    st.header("Control panel")
    voice_on = st.checkbox("Voice alerts", value=False)
    st.caption("Browsers block speech until you click something on the page.")

    st.divider()
    st.header("Rainfall")
    source = st.radio(
        "Source", ["Live forecast", "Manual (what-if)"], index=0,
        help="Live pulls a real 15-minute forecast for Bandra. Manual lets you "
             "ask what would happen at an intensity that is not falling today.",
    )
    manual_intensity = st.slider("Intensity (mm/hr)", 0, 150, 45,
                                 disabled=(source == "Live forecast"))

    st.divider()
    minutes_ahead = st.select_slider(
        "Looking ahead", options=list(range(0, fe.HORIZON_MIN + 1, fe.STEP_MIN)),
        value=0, format_func=lambda m: "Now" if m == 0 else f"+{m} min",
    )


@st.cache_data(ttl=300)
def cached_live():
    return fe.live_series(*BANDRA)


if source == "Live forecast":
    live = cached_live()
    if live["error"]:
        st.warning(
            f"Live forecast unavailable ({live['error']}). Showing zero rainfall - "
            "switch to Manual to demonstrate the model."
        )
        series = fe.flat_series(0.0)
    else:
        series = live["series"]
else:
    series = fe.flat_series(manual_intensity)

intensity_now = fe.intensity_at(series, minutes_ahead)

st.title("Bandra Predictive Drainage Monitor")

try:
    drains = fe.load_drains()
except Exception as exc:
    st.error(f"Could not read bandra_capacity.csv - {exc}")
    st.stop()

rows = fe.assess(drains, series, at_minutes=minutes_ahead)
df = pd.DataFrame(rows)

flooded = df[df["Depth_cm"] >= 10]
critical = df[df["Depth_cm"] >= 25]
worst = rows[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rain", f"{intensity_now:.1f} mm/hr",
          "live" if source == "Live forecast" else "what-if")
c2.metric("Spots with standing water", len(flooded), f"of {len(df)}")
c3.metric("Avoid / impassable", len(critical))
c4.metric("Deepest", f"{worst['Peak_Depth_cm']:.0f} cm", worst["Segment_Name"])

st.subheader("Predicted flood depth")

fmap = folium.Map(location=BANDRA, zoom_start=14, tiles="OpenStreetMap")

for row in rows:
    drain = drains[row["Drain_ID"]]
    if drain.downstream:
        other = drains[drain.downstream]
        folium.PolyLine(
            [(drain.lat, drain.lon), (other.lat, other.lon)],
            color="#5A6B7F", weight=2, opacity=0.55, dash_array="6",
        ).add_to(fmap)

for row in rows:
    depth = row["Depth_cm"]
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8 + min(depth, 100) / 6,
        color=row["Color"],
        fill=True,
        fill_opacity=0.65,
        weight=2,
        tooltip=folium.Tooltip(
            f"<b>{row['Segment_Name']}</b><br>"
            f"{depth:.0f} cm now, peak {row['Peak_Depth_cm']:.0f} cm "
            f"in {row['Minutes_To_Peak']} min<br>"
            f"{row['Drain_Type']}, {row['Blockage_Pct']}% blocked<br>"
            f"drains to {row['Drains_To']}, serves {row['Upstream_Area_ha']} ha above"
        ),
    ).add_to(fmap)

st_folium(fmap, width=1200, height=480, returned_objects=[])
st.caption("Dashed lines show which drain flows into which. Circle size is depth.")

if len(critical):
    names = ", ".join(critical.sort_values("Depth_cm", ascending=False)["Segment_Name"].head(3))
    st.error(f"Avoid: {names} - over 25 cm of water predicted.")
    if voice_on:
        top = critical.sort_values("Depth_cm", ascending=False).iloc[0]
        speak(
            f"Flood warning. {top['Segment_Name']} is expected to have "
            f"{int(top['Depth_cm'])} centimetres of water. Avoid this route."
        )
elif len(flooded):
    st.warning(f"{len(flooded)} spots have standing water, none deep enough to close a road yet.")
else:
    st.success("No standing water predicted anywhere in the next three hours.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Every drain")
    st.dataframe(
        df[[
            "Segment_Name", "Drain_Type", "Blockage_Pct", "Design_Capacity_m3s",
            "Real_Capacity_m3s", "Drains_To", "Depth_cm", "Peak_Depth_cm", "Risk_Level",
        ]].rename(columns={
            "Segment_Name": "Drain", "Drain_Type": "Type", "Blockage_Pct": "Blocked %",
            "Design_Capacity_m3s": "Design m3/s", "Real_Capacity_m3s": "Real m3/s",
            "Drains_To": "Flows into", "Depth_cm": "Now (cm)",
            "Peak_Depth_cm": "Peak (cm)", "Risk_Level": "Level",
        }),
        hide_index=True, use_container_width=True,
    )

with right:
    st.subheader("How much rain does it take?")
    st.caption(
        "The intensity at which each spot goes under 15 cm, solved through the "
        "whole network - a drain floods on water that fell uphill."
    )
    thresholds = []
    for row in rows:
        t = fe.floods_at_mm_hr(drains, row["Drain_ID"])
        thresholds.append({
            "Drain": row["Segment_Name"],
            "Blocked %": row["Blockage_Pct"],
            "Floods at (mm/hr)": 999 if t == float("inf") else t,
        })
    st.dataframe(
        pd.DataFrame(thresholds).sort_values("Floods at (mm/hr)"),
        hide_index=True, use_container_width=True,
    )

st.subheader(f"Next three hours - {worst['Segment_Name']}")
timeline = pd.DataFrame(worst["Timeline"], columns=["Minutes ahead", "Depth (cm)"])
st.line_chart(timeline.set_index("Minutes ahead"), height=200)

with st.expander("What this model does, and what it does not"):
    st.markdown(
        """
**Real.** The rational method for runoff and the capacity reduction for
blockage are standard drainage engineering. The network routing, the
backflow term and the depth calculation are the substance of the model.
The live forecast is a genuine 15-minute prediction from Open-Meteo.

**Not real yet.** Open-Meteo is a numerical weather model, not IMD radar -
a deployment would use radar nowcasts. Catchment areas in the CSV are
derived from each drain's rated capacity at Mumbai's 25 mm/hr design
standard, not surveyed. Blockage percentages are estimates, not CCTV
inspection records. The depth figures are physically reasoned but have not
been checked against observed flood marks, which is the single most
valuable thing that could be added next.
        """
    )
