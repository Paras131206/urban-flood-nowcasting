"""Turn-by-turn navigation that routes around water.

Real roads, from OSRM. It returns several ways to make the same journey and
ranks them by driving time, knowing nothing about flooding. We re-rank them by
how much standing water each one crosses and send you the driest, then show
the one you would otherwise have taken and what it would have cost you.
"""
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

import flood_engine as fe
import road_router as rr
import route_planner as rp

st.set_page_config(page_title="Safe Route", layout="wide")

BANDRA = (19.0544, 72.8402)


def colour_for(pct: float) -> str:
    if pct >= 60:
        return "#C62828"
    if pct >= 40:
        return "#EF6C00"
    if pct >= 20:
        return "#F9A825"
    return "#2E7D32"


def speak(message: str) -> None:
    safe = message.replace("'", "").replace("\n", " ")
    components.html(
        f"<script>var m = new SpeechSynthesisUtterance('{safe}');"
        f"m.rate = 0.92; window.speechSynthesis.speak(m);</script>",
        height=0,
    )


st.title("Navigation")
st.caption(
    "Routes follow real streets. When the quickest way crosses standing water, "
    "the app sends you the driest alternative instead and shows you what it cost."
)

drains = fe.load_drains()
places = dict(rp.LANDMARKS)

with st.sidebar:
    st.header("Journey")
    names = sorted(places)
    origin_name = st.selectbox("From", names,
                               index=names.index("Bandra Fort") if "Bandra Fort" in names else 0)
    destination_name = st.selectbox("To", names,
                                    index=names.index("Bandra Kurla Complex")
                                    if "Bandra Kurla Complex" in names else 1)

    st.divider()
    st.header("Conditions")
    source = st.radio("Rainfall", ["Live forecast", "Manual (what-if)"], index=0)
    manual = st.slider("Intensity (mm/hr)", 0, 150, 45,
                       disabled=(source == "Live forecast"))
    minutes_ahead = st.select_slider(
        "Leaving in", options=list(range(0, fe.HORIZON_MIN + 1, fe.STEP_MIN)),
        value=0, format_func=lambda m: "Now" if m == 0 else f"{m} min",
    )

    st.divider()
    threshold = st.slider("Treat as flooded above (%)", 10, 90, 40, step=5)
    voice_on = st.checkbox("Read the first instruction aloud", value=False)


@st.cache_data(ttl=300)
def cached_live():
    return fe.live_series(*BANDRA)


if source == "Live forecast":
    live = cached_live()
    if live["error"]:
        st.warning(f"Live forecast unavailable ({live['error']}). Using zero rainfall.")
        series = fe.flat_series(0.0)
    else:
        series = live["series"]
else:
    series = fe.flat_series(manual)

timeline = fe.forecast(drains, series)
depths = {
    drains[did].name: dict(points).get(minutes_ahead, points[0][1])
    for did, points in timeline.items()
}

if origin_name == destination_name:
    st.info("Pick two different places.")
    st.stop()


origin = places[origin_name]
destination = places[destination_name]

# Everywhere the router may use as a waypoint, and how wet each one is.
# Landmarks have no drain of their own, so they take the risk of the nearest
# one, fading with distance.
all_places = dict(places)
all_places.update({d.name: (d.lat, d.lon) for d in drains.values()})

risk_by_place = {}
for name, point in all_places.items():
    if name in depths:
        risk_by_place[name] = rp.risk_pct(depths[name])
        continue
    nearest = min(drains.values(), key=lambda d: fe.haversine_m(point, (d.lat, d.lon)))
    distance = fe.haversine_m(point, (nearest.lat, nearest.lon))
    fade = max(0.0, 1.0 - distance / 600.0)
    risk_by_place[name] = round(rp.risk_pct(depths.get(nearest.name, 0.0)) * fade, 1)


# Only the network calls are cached, and only on their coordinates. Scoring
# and choosing run fresh on every rerun, because they depend on the forecast -
# caching those would freeze the route while the rainfall slider moved.
@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_routes(a, b):
    return rr.fetch_routes(a, b)


@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_via(a, via, b):
    return rr.fetch_via(a, via, b)


with st.spinner("Finding routes..."):
    fetched = rr.plan_journey(
        drains, depths, origin, destination, all_places, risk_by_place,
        threshold_pct=float(threshold),
        fetch=cached_fetch_routes,
        fetch_via_fn=cached_fetch_via,
    )

if not fetched["ok"]:
    st.error(
        f"Road routing is unavailable ({fetched['error']}). "
        "Falling back to straight-line planning between known points - the "
        "distances below are as the crow flies, not driving distances."
    )
    fallback = rp.plan(drains, depths, origin_name, destination_name,
                       avoid_above_pct=float(threshold))
    if fallback["found"]:
        st.write(fallback["message"])
        st.dataframe(
            pd.DataFrame([{
                "Head": leg["heading"], "For": f"{leg['metres']} m",
                "Towards": leg["to"], "Risk": f"{leg['risk_pct']:.0f}%",
            } for leg in fallback["legs"]]),
            hide_index=True, width="stretch",
        )
    else:
        st.warning(fallback["message"])
    st.stop()

routes = fetched["routes"]
chosen = fetched["chosen"]
fastest = fetched["fastest"]
rerouted = fetched["rerouted"]
clear = fetched["all_clear"]

summary = rr.summarise(chosen)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Distance", f"{summary['distance_km']} km")
c2.metric("Time", f"{summary['minutes']} min")
c3.metric("Worst point", f"{summary['max_pct']:.0f}%", summary["condition"])
c4.metric("Through water", f"{summary['flooded_m']} m" if summary["flooded_m"] else "none")

if not clear:
    st.error(
        f"Every route is affected. This is the driest of {len(routes)}, peaking at "
        f"{chosen['score']['max_pct']:.0f}% over {chosen['score']['flooded_m']} m. "
        "Consider waiting."
    )
elif rerouted:
    extra_min = round((chosen["duration_s"] - fastest["duration_s"]) / 60)
    extra_km = (chosen["distance_m"] - fastest["distance_m"]) / 1000
    st.warning(
        f"**Rerouted.** The quickest way crosses {fastest['score']['flooded_m']} m of "
        f"standing water, peaking at {fastest['score']['max_pct']:.0f}%. "
        f"This route avoids it for {extra_km:.1f} km and about {extra_min} min more."
        + (f" It was built by routing through {chosen['via']}." if chosen.get("via") else "")
    )
else:
    st.success("The quickest route is also clear of flooding.")

if voice_on and chosen["steps"]:
    first = chosen["steps"][0]
    lead = "Rerouting to avoid flooding. " if rerouted else ""
    speak(f"{lead}{first['instruction']}. Then continue for "
          f"{round(first['distance_m'])} metres.")

fmap = folium.Map(location=BANDRA, zoom_start=13, tiles="OpenStreetMap")

# Separate layers so the two routes can be compared, and either switched off.
# A detour usually shares most of its roads with the original, so drawing the
# original underneath at a heavier weight lets it show through where they run
# together and stand alone where they diverge.
layer_original = folium.FeatureGroup(name="Original route", show=True)
layer_others = folium.FeatureGroup(name="Other alternatives", show=True)
layer_chosen = folium.FeatureGroup(name="Recommended route", show=True)
layer_water = folium.FeatureGroup(name="Flooded stretches", show=True)
layer_drains = folium.FeatureGroup(name="Drains", show=True)

for route in routes:
    if route is chosen:
        continue
    if route is fastest:
        folium.PolyLine(
            route["coords_latlon"],
            color="#C62828", weight=11, opacity=0.85,
            tooltip=(
                f"ORIGINAL - {route.get('label', 'Direct')}: "
                f"{route['distance_m'] / 1000:.1f} km, "
                f"{round(route['duration_s'] / 60)} min, "
                f"peak {route['score']['max_pct']:.0f}%, "
                f"{route['score']['flooded_m']} m through water"
            ),
        ).add_to(layer_original)
    else:
        folium.PolyLine(
            route["coords_latlon"],
            color="#8894A0", weight=5, opacity=0.6, dash_array="8",
            tooltip=(
                f"{route.get('label', 'Alternative')}: "
                f"{route['distance_m'] / 1000:.1f} km, "
                f"{round(route['duration_s'] / 60)} min, "
                f"peak {route['score']['max_pct']:.0f}%"
            ),
        ).add_to(layer_others)

folium.PolyLine(
    chosen["coords_latlon"], color="#1F6FC5", weight=6, opacity=0.95,
    tooltip=(
        f"RECOMMENDED - {chosen.get('label', 'Route')}: "
        f"{summary['distance_km']} km, {summary['minutes']} min, "
        f"peak {summary['max_pct']:.0f}%"
    ),
).add_to(layer_chosen)

for route in routes:
    for point, risk in zip(route["score"]["samples"], route["score"]["risk_at"]):
        if risk >= threshold:
            folium.CircleMarker(
                point, radius=7, color="#C62828", fill=True,
                fill_opacity=0.45, weight=1,
                tooltip=f"{risk:.0f}% here - {rp.advice_for(risk)}",
            ).add_to(layer_water)

for drain in drains.values():
    pct = rp.risk_pct(depths.get(drain.name, 0.0))
    folium.CircleMarker(
        (drain.lat, drain.lon),
        radius=6 + min(pct, 100) / 14,
        color=colour_for(pct), fill=True, fill_opacity=0.55, weight=2,
        tooltip=f"<b>{drain.name}</b><br>{pct:.0f}% - {rp.advice_for(pct)}"
                f"<br>{depths.get(drain.name, 0):.0f} cm",
    ).add_to(layer_drains)

# Order matters: original goes down first so the recommendation sits on top.
layer_original.add_to(fmap)
layer_others.add_to(fmap)
layer_chosen.add_to(fmap)
layer_water.add_to(fmap)
layer_drains.add_to(fmap)

for label, point in (("A", origin), ("B", destination)):
    folium.Marker(point, tooltip=label, icon=folium.DivIcon(html=(
        f"<div style='background:#10375C;color:#fff;border-radius:50%;width:26px;"
        f"height:26px;line-height:26px;text-align:center;font-weight:700;"
        f"border:2px solid #fff'>{label}</div>"))).add_to(fmap)

folium.LayerControl(collapsed=False).add_to(fmap)

st_folium(fmap, width=1200, height=520, returned_objects=[])
legend = (
    "**Thick red** is the original route. **Blue** is the recommendation, drawn on "
    "top - where only blue shows, both take the same road; where red shows alone, "
    "that is the stretch being avoided. Grey dashes are other options, red dots are "
    "flooded stretches. Use the box on the map to switch layers off."
    if rerouted else
    "**Blue** is the recommended route, and it is also the quickest. "
    "Red dots would mark flooded stretches if there were any."
)
st.caption(legend)

st.subheader("Routes considered")
if fetched["detours_tried"]:
    st.caption("Nothing OSRM offered was dry, so detours were built through: "
               + ", ".join(fetched["detours_tried"]))
st.dataframe(
    pd.DataFrame([{
        "Route": r.get("label", f"Option {i + 1}"),
        "Distance": f"{r['distance_m'] / 1000:.1f} km",
        "Time": f"{round(r['duration_s'] / 60)} min",
        "Worst point": f"{r['score']['max_pct']:.0f}%",
        "Through water": f"{r['score']['flooded_m']} m" if r["score"]["flooded_m"] else "none",
        "Verdict": "Take this" if r is chosen else (
            "Flooded" if r["score"]["max_pct"] >= threshold else "Clear, but slower"),
    } for i, r in enumerate(routes)]),
    hide_index=True, width="stretch",
)

st.subheader("Directions")
st.dataframe(
    pd.DataFrame([{
        "#": i,
        "Instruction": step["instruction"],
        "For": f"{round(step['distance_m'])} m" if step["distance_m"] >= 1 else "",
    } for i, step in enumerate(chosen["steps"], start=1)]),
    hide_index=True, width="stretch", height=340,
)

hotspots = chosen["score"]["hotspots"]
if hotspots:
    st.subheader("Still wet on this route")
    st.dataframe(
        pd.DataFrame([{"Drain": name, "Risk": f"{pct:.0f}%", "Condition": rp.advice_for(pct)}
                      for pct, name in hotspots]),
        hide_index=True, width="stretch",
    )

with st.expander("How the route is chosen"):
    st.markdown(
        """
OSRM returns several ways to make the journey and ranks them by driving time.
It knows nothing about flooding.

Each route is then sampled every 120 metres. Every sample takes the risk of
the nearest drain, fading to nothing 400 metres away - a road 50 m from a
flooded gully is wet, a road 400 m away is not. A route's score is its worst
sample, and its "through water" figure is how much of its length sits above
the threshold.

The quickest **clear** route wins. If nothing is clear, the driest one wins and
the page says so rather than implying the journey is safe.

The limitation worth knowing: risk is inferred from ten drain locations, so
the spatial resolution of the flood map is coarse. More drains, or a proper
inundation raster, would sharpen it. The routing logic does not change.
        """
    )
