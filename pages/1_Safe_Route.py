"""Turn-by-turn navigation that routes around water.

Real roads, from OSRM. It returns several ways to make the same journey and
ranks them by driving time, knowing nothing about flooding. We re-rank them by
how much standing water each one crosses and recommend the driest.

The recommendation is a default, not a verdict. Every route that was
considered can be selected, and selecting one redraws the map and rebuilds the
statistics for that route — so the person can see for themselves what the
detour is buying and decide whether it is worth it.
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

# Who is travelling changes what counts as impassable, and it changes it a
# lot. A fire engine has half a metre of ground clearance and a crew who have
# trained for this; a scooter is in trouble at ten centimetres. Sending both
# the same way is the mistake — one gets needlessly delayed, the other gets
# needlessly drowned.
#
# `wade_cm` is the depth at which that vehicle is genuinely in difficulty.
# It maps onto the risk scale as wade_cm / 50 cm * 100.
PROFILES = {
    "Car": {
        "wade_cm": 20,
        "prefer": "safety",          # quickest route that stays passable
        "note": "A hatchback starts to struggle at about 20 cm and floats "
                "before it stalls. Routed around anything deeper.",
    },
    "Two-wheeler": {
        "wade_cm": 10,
        "prefer": "driest",          # avoid the water even if it costs time
        "note": "A scooter exhaust is under water at about 10 cm, and the "
                "rider cannot see potholes or open manholes through it. So "
                "this profile picks the driest route on offer even when it is "
                "slower — the most cautious of the four.",
    },
    "Walking": {
        "wade_cm": 15,
        "prefer": "shortest",        # on foot an extra km hurts more than 5 min
        "note": "Depth is not the only hazard on foot — an open drain under "
                "opaque water is. Ranked by distance among the passable "
                "routes, because an extra kilometre on foot costs more than "
                "an extra five minutes in a car.",
    },
    "Emergency (fire / ambulance)": {
        "wade_cm": 45,
        "prefer": "time",
        "note": "A fire tender wades where a car cannot, and a delayed "
                "ambulance has its own body count. So this profile takes the "
                "quickest route that is genuinely passable and warns about "
                "the water rather than driving around it.",
    },
}


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


def colour_for(pct: float) -> str:
    if pct >= 60:
        return "#C62828"
    if pct >= 40:
        return "#EF6C00"
    if pct >= 20:
        return "#F9A825"
    return "#2E7D32"


def speak(message: str) -> None:
    safe = (message.replace("\\", "")
                   .replace("'", "")
                   .replace('"', "")
                   .replace("\n", " "))
    components.html(
        f"<script>window.speechSynthesis.cancel();"
        f"var m = new SpeechSynthesisUtterance('{safe}');"
        f"m.rate = 0.95; window.speechSynthesis.speak(m);</script>",
        height=0,
    )


st.title("Navigation")
st.caption(
    "Routes follow real streets. When the quickest way crosses standing water, "
    "the app recommends the driest alternative — but you can select any route "
    "below and see its own numbers."
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
    manual = st.slider("Intensity (mm/hr)", 0, 200, 45,
                       disabled=(source == "Live forecast"))
    minutes_ahead = st.select_slider(
        "Leaving in", options=list(range(0, fe.HORIZON_MIN + 1, fe.STEP_MIN)),
        value=0, format_func=lambda m: "Now" if m == 0 else f"{m} min",
    )
    tide_m = st.select_slider(
        "Tide", options=[0.5, 1.2, 2.0, 3.0, 4.0], value=fe.DEFAULT_TIDE_M,
        format_func=lambda t: f"{t} m",
        help="Storm drains discharge to the sea by gravity. A high tide "
             "drowns the outfalls, and roads that would drain in an hour stay "
             "under until it turns.",
    )

    st.divider()
    st.header("Who is travelling")
    profile_name = st.selectbox("Vehicle", list(PROFILES), index=0)
    profile = PROFILES[profile_name]
    suggested = int(round(profile["wade_cm"] / rp.DEPTH_AT_FULL_RISK_CM * 100))

    st.caption(profile["note"])
    st.caption(f"Impassable for this profile at about "
               f"**{profile['wade_cm']} cm** of water ({suggested}% risk).")

    override = st.checkbox("Set the threshold myself", value=False)
    threshold = st.slider("Treat as flooded above (%)", 10, 90, suggested, step=5,
                          disabled=not override)
    if not override:
        threshold = suggested

    voice_on = st.checkbox("Read the route aloud", value=False)


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

timeline = fe.forecast(drains, series, tide_m=tide_m)
depths = {
    drains[did].name: dict(points).get(minutes_ahead, points[0][1])
    for did, points in timeline.items()
}

intensity_at_departure = fe.intensity_at(series, minutes_ahead)

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
# and choosing run fresh on every rerun, because they depend on the forecast —
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
        prefer=profile["prefer"],
        fetch=cached_fetch_routes,
        fetch_via_fn=cached_fetch_via,
    )

# --------------------------------------------------------------------------- #
# If OSRM is unreachable, fall back and say so rather than showing nothing.
# --------------------------------------------------------------------------- #
if not fetched["ok"]:
    st.error(
        f"Road routing is unavailable ({fetched['error']}). "
        "Falling back to straight-line planning between known points — the "
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

# --------------------------------------------------------------------------- #
# Every route that was considered
# --------------------------------------------------------------------------- #
routes = fetched["routes"]
chosen = fetched["chosen"]
fastest = fetched["fastest"]
rerouted = fetched["rerouted"]
clear = fetched["all_clear"]


def role_of(route) -> str:
    """What this route is, in one word, for labels and tables."""
    if route is chosen and route is fastest:
        return "recommended, quickest"
    if route is chosen:
        return "recommended"
    if route is fastest:
        return "original"
    return "alternative"


def option_label(index: int, route) -> str:
    # The number keeps every label unique even if two routes happen to share a
    # name and the same distance and time — st.radio matches on the string.
    name = route.get("label") or f"Option {index + 1}"
    water = (f"{route['score']['flooded_m']} m through water"
             if route["score"]["flooded_m"] else "clear")
    when = rr.eta(route, intensity_at_departure, minutes_ahead)
    return (f"{index + 1}. {name} ({role_of(route)}) — "
            f"{route['distance_m'] / 1000:.1f} km, "
            f"{when['minutes_low']}-{when['minutes_high']} min, {water}")


labels = [option_label(i, r) for i, r in enumerate(routes)]
# Identity, not equality — two routes with identical contents must not collapse.
default_index = next((i for i, r in enumerate(routes) if r is chosen), 0)

st.subheader("Choose a route")
picked = st.radio(
    "Every route considered. The recommendation is pre-selected.",
    labels,
    index=default_index,
    key=f"route_pick::{origin_name}->{destination_name}",
)
selected = routes[labels.index(picked)]
is_recommended = selected is chosen

summary = rr.summarise(selected)
chosen_summary = rr.summarise(chosen)

# Arrival time, not a stopwatch figure. OSRM's duration is free-flow on a dry
# road; this stretches it for the rain and for every stretch of standing water
# the route crosses, and reports a window rather than implying the app knows
# the traffic.
estimate = rr.eta(selected, intensity_at_departure, minutes_ahead)
chosen_estimate = rr.eta(chosen, intensity_at_departure, minutes_ahead)
journey_risk = rr.route_risk_factor(selected, float(threshold), estimate)

# --------------------------------------------------------------------------- #
# Statistics for whichever route is selected
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Distance", f"{summary['distance_km']} km",
    None if is_recommended
    else f"{(selected['distance_m'] - chosen['distance_m']) / 1000:+.1f} km vs recommended",
)
c2.metric(
    "Arrive",
    "—" if estimate["impassable"]
    else f"{estimate['arrive_low']}-{estimate['arrive_high']}",
    rr.describe_eta(estimate) if (estimate["impassable"] or estimate["capped"])
    else (f"{estimate['minutes_low']}-{estimate['minutes_high']} min"
          + (f", {estimate['delay_min']} min lost to rain"
             if estimate["delay_min"] >= 2 else "")),
    delta_color="off",
    help="Not a stopwatch figure. OSRM gives the free-flow driving time on a "
         "dry road; this stretches it for the rain (spray, visibility, "
         "everyone else braking) and again for every stretch of standing "
         "water on the route. The window is the honest part — the app does "
         "not know the traffic, and the range widens the wetter the route "
         "gets.",
)
c3.metric(
    "Risk factor", journey_risk["score"], journey_risk["band"],
    delta_color="off",
    help="How bad the journey is, not how deep the water is. Combines the "
         "depth of the worst point (35%), how much of the route is wet (25%), "
         "how long you spend in it (20%), and how far past this vehicle's own "
         "limit it goes (20%).",
)
c4.metric(
    "Worst point", f"{summary['max_pct']:.0f}%", summary["condition"],
    help="The wettest single point along this route, not an average and not a "
         "probability. The route is sampled every 120 m and each sample takes "
         "the risk of the nearest drain; this is the worst of them. So 52% "
         "means there is one stretch at 52% risk (about 26 cm of water) and "
         "everywhere else on the route is better than that. It is a worst case "
         "on purpose — an average would hide a single impassable 100 m stretch "
         "inside an otherwise dry 10 km drive.",
)
c5.metric(
    "Through water",
    f"{summary['flooded_m']} m" if summary["flooded_m"] else "none",
    None if is_recommended
    else f"{selected['score']['flooded_m'] - chosen['score']['flooded_m']:+d} m vs recommended",
    delta_color="inverse",
)

with st.expander(f"Risk factor scale — this journey scores "
                 f"{journey_risk['score']} ({journey_risk['band']})", expanded=True):
    scale = st.columns(len(rr.RISK_BANDS))
    for column, (floor, name, colour, meaning) in zip(scale, reversed(rr.RISK_BANDS)):
        ceiling = next((f - 1 for f, *_ in reversed(rr.RISK_BANDS) if f > floor), 100)
        here = journey_risk["band"] == name
        column.markdown(
            f"<div style='border-left:6px solid {colour};padding:2px 10px;"
            f"background:{'#00000012' if here else 'transparent'};'>"
            f"<b>{floor}-{ceiling} &nbsp;{name}</b>"
            f"{' &nbsp;&larr; you are here' if here else ''}<br>"
            f"<span style='font-size:0.85em'>{meaning}</span></div>",
            unsafe_allow_html=True,
        )
    st.caption(f"**This journey:** {journey_risk['why']}")
    st.dataframe(
        pd.DataFrame([{"Contributes": k, "Points": v}
                      for k, v in journey_risk["parts"].items()]),
        hide_index=True, width="stretch",
    )

st.caption(
    f"**Reading the worst point:** {summary['max_pct']:.0f}% means the wettest "
    f"120-metre sample on this route sits at {summary['max_pct']:.0f}% risk, "
    f"which is about {summary['max_pct'] / 100 * rp.DEPTH_AT_FULL_RISK_CM:.0f} cm "
    f"of water. Everywhere else is shallower. It is not a chance of flooding, "
    f"and it is deliberately a worst case rather than an average — for "
    f"**{profile_name.lower()}** the line is drawn at {threshold}% "
    f"({profile['wade_cm']} cm)."
)

# --- Emergency briefing: go fast, but know what you are driving into ------ #
if profile["prefer"] == "time":
    wet_spots = selected["score"]["hotspots"]
    if wet_spots:
        worst_pct, worst_name = wet_spots[0]
        st.warning(
            f"🚒 **Emergency routing — fastest passable route, not the driest.** "
            f"This route crosses {selected['score']['flooded_m']} m of standing "
            f"water, deepest around **{worst_name}** at "
            f"{worst_pct / 100 * rp.DEPTH_AT_FULL_RISK_CM:.0f} cm. That is "
            f"within a fire tender's wading depth but above a car's. Slow to a "
            f"walking pace through it, keep the revs up, and do not follow a "
            f"civilian vehicle in."
        )
    else:
        st.success(
            "🚒 **Emergency routing.** The quickest route is also clear of "
            "standing water — no trade-off to make."
        )
    if selected["score"]["max_pct"] >= 90:
        st.error(
            "Part of this route is at or near 100% risk — deep enough that "
            "even a high-clearance vehicle should not attempt it. Check the "
            "alternatives below before committing."
        )

if not is_recommended:
    if selected["score"]["max_pct"] >= threshold:
        st.warning(
            f"You are looking at **{selected.get('label', 'this route')}**, which crosses "
            f"{selected['score']['flooded_m']} m of standing water and peaks at "
            f"{selected['score']['max_pct']:.0f}%. It is shown for comparison — the "
            f"recommendation is **{chosen.get('label', 'the other route')}**, "
            f"{chosen_summary['distance_km']} km and {chosen_summary['minutes']} min."
        )
    else:
        st.info(
            f"**{selected.get('label', 'This route')}** is clear too, but it is "
            f"{abs(round((selected['duration_s'] - chosen['duration_s']) / 60))} min "
            f"{'slower' if selected['duration_s'] > chosen['duration_s'] else 'quicker'} "
            f"than the recommendation. Either is safe at the current forecast."
        )
elif not clear:
    st.error(
        f"Every route is affected. This is the driest of {len(routes)}, peaking at "
        f"{chosen['score']['max_pct']:.0f}% over {chosen['score']['flooded_m']} m. "
        "Consider waiting."
    )
elif rerouted:
    extra_min = round((chosen["duration_s"] - fastest["duration_s"]) / 60)
    extra_km = (chosen["distance_m"] - fastest["distance_m"]) / 1000
    built = f" It was built by routing through {chosen['via']}." if chosen.get("via") else ""
    st.warning(
        f"**Rerouted.** The quickest way crosses {fastest['score']['flooded_m']} m of "
        f"standing water, peaking at {fastest['score']['max_pct']:.0f}%. "
        f"This route avoids it for {extra_km:.1f} km and about {extra_min} min more."
        + built
    )
else:
    st.success("The quickest route is also clear of flooding.")

def spoken_route(route, risk, when, wet_spots, rerouted_now: bool,
                 max_steps: int = 8) -> str:
    """Read out the whole journey: the verdict, the timing, every wet spot,
    then the turns.

    Reading only the first instruction was the wrong call for the one person
    who most needs this — somebody holding a phone in the rain who cannot look
    at it. They need to know what they are driving into before they know which
    way to turn out of the car park.
    """
    parts = []
    if rerouted_now:
        parts.append("Rerouting to avoid flooding.")

    parts.append(f"{profile_name}, {origin_name} to {destination_name}.")
    parts.append(f"Risk factor {risk['score']}, {risk['band'].lower()}.")

    if when.get("impassable"):
        parts.append("This route is impassable. Do not attempt it.")
    else:
        parts.append(f"Arriving between {when['arrive_low']} "
                     f"and {when['arrive_high']}.")
        if when["delay_min"] >= 2:
            parts.append(f"About {when['delay_min']} minutes more than a clear day.")

    if wet_spots:
        named = ". ".join(
            f"{name}, {pct / 100 * rp.DEPTH_AT_FULL_RISK_CM:.0f} centimetres"
            for pct, name in wet_spots[:5])
        parts.append(f"Standing water at {named}.")
        if len(wet_spots) > 5:
            parts.append(f"And {len(wet_spots) - 5} more.")
    else:
        parts.append("No standing water on this route.")

    parts.append("Directions.")
    for index, step in enumerate(route["steps"][:max_steps], start=1):
        distance = (f", then continue for {round(step['distance_m'])} metres"
                    if step["distance_m"] >= 20 else "")
        parts.append(f"{index}. {step['instruction']}{distance}.")
    if len(route["steps"]) > max_steps:
        parts.append(f"And {len(route['steps']) - max_steps} more turns.")

    return " ".join(parts)


route_briefing = spoken_route(
    selected, journey_risk, estimate, selected["score"]["hotspots"],
    is_recommended and rerouted,
)
if voice_on:
    speak(route_briefing)
with st.expander("What the voice guidance says"):
    st.write(route_briefing)
    st.caption(
        "The whole journey — the verdict, the arrival window, every stretch of "
        "standing water on it, then the turns. Tick the box in the sidebar to "
        "hear it; browsers block speech until you have clicked something."
    )

# --------------------------------------------------------------------------- #
# Map — the selected route on top, everything else beneath it
# --------------------------------------------------------------------------- #
fmap = folium.Map(location=BANDRA, zoom_start=13, tiles="OpenStreetMap")

# Colour says one thing only: can you get through. Blue is passable, red is
# not. That holds for the shortest route and for every alternate, so a flooded
# alternate can never be mistaken for a safe one just because it is the
# alternate — and the whole point of the page reads off the map at a glance:
# red line through the water, blue line around it.
SAFE_COLOUR = "#1565C0"          # bold blue: passable for this vehicle
FLOODED_COLOUR = "#C62828"       # red: worst point is past the line

# Which route is "the shortest" — the one someone would take without this app.
# Detours are excluded: a route we built by forcing a waypoint is an
# alternative by construction, so calling it the shortest would be circular.
direct_offers = [r for r in routes if not r.get("via")] or routes
shortest = min(direct_offers, key=lambda r: r["distance_m"])

layer_shortest = folium.FeatureGroup(name="Shortest route", show=True)
layer_alternates = folium.FeatureGroup(name="Alternate routes", show=True)
layer_selected = folium.FeatureGroup(name="Selected route", show=True)
layer_water = folium.FeatureGroup(name="Flooded stretches", show=True)
layer_drains = folium.FeatureGroup(name="Drains", show=True)


def is_passable(route) -> bool:
    return route["score"]["max_pct"] < threshold


def route_colour(route) -> str:
    return SAFE_COLOUR if is_passable(route) else FLOODED_COLOUR


def route_tooltip(route, index) -> str:
    when = rr.eta(route, intensity_at_departure, minutes_ahead)
    kind = "SHORTEST" if route is shortest else "ALTERNATE"
    state = "passable" if is_passable(route) else "FLOODED"
    if route is selected:
        kind = "SELECTED - " + kind
    return (
        f"{kind} ({state}) — {route.get('label', f'Option {index + 1}')}: "
        f"{route['distance_m'] / 1000:.1f} km, "
        f"{when['minutes_low']}-{when['minutes_high']} min, "
        f"peak {route['score']['max_pct']:.0f}%, "
        f"{route['score']['flooded_m']} m through water"
    )


for i, route in enumerate(routes):
    if route is selected:
        continue
    folium.PolyLine(
        route["coords_latlon"],
        color=route_colour(route),
        weight=8,                       # bold, whichever it is
        opacity=0.75,
        tooltip=route_tooltip(route, i),
    ).add_to(layer_shortest if route is shortest else layer_alternates)

# The selection gets a white casing underneath and a heavier line on top, so
# it stands out without borrowing a colour that already means "passable".
folium.PolyLine(
    selected["coords_latlon"], color="#FFFFFF", weight=16, opacity=0.9,
).add_to(layer_selected)
folium.PolyLine(
    selected["coords_latlon"],
    color=route_colour(selected),
    weight=9, opacity=1.0,
    tooltip=route_tooltip(selected,
                          routes.index(selected) if selected in routes else 0),
).add_to(layer_selected)

# Mark the stretches that are wet, on whichever routes cross them.
for route in routes:
    for point, risk in zip(route["score"]["samples"], route["score"]["risk_at"]):
        if risk >= threshold:
            folium.CircleMarker(
                point, radius=7, color="#C62828", fill=True,
                fill_opacity=0.45, weight=1,
                tooltip=f"{risk:.0f}% here — {rp.advice_for(risk)}",
            ).add_to(layer_water)

for drain in drains.values():
    pct = rp.risk_pct(depths.get(drain.name, 0.0))
    folium.CircleMarker(
        (drain.lat, drain.lon),
        radius=6 + min(pct, 100) / 14,
        color=colour_for(pct), fill=True, fill_opacity=0.55, weight=2,
        tooltip=f"<b>{drain.name}</b><br>{pct:.0f}% — {rp.advice_for(pct)}"
                f"<br>{depths.get(drain.name, 0):.0f} cm",
    ).add_to(layer_drains)

# Order matters: the unselected routes go down first so the selection, with
# its white casing, sits on top of them.
layer_shortest.add_to(fmap)
layer_alternates.add_to(fmap)
layer_selected.add_to(fmap)
layer_water.add_to(fmap)
layer_drains.add_to(fmap)

for label, point in (("A", origin), ("B", destination)):
    folium.Marker(point, tooltip=label, icon=folium.DivIcon(html=(
        f"<div style='background:#10375C;color:#fff;border-radius:50%;width:26px;"
        f"height:26px;line-height:26px;text-align:center;font-weight:700;"
        f"border:2px solid #fff'>{label}</div>"))).add_to(fmap)

folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)

render_map(fmap, height=560)
safe_alternates = [r for r in routes if r is not shortest and is_passable(r)]
st.caption(
    "**Colour means one thing: can you get through.** Bold blue is passable "
    "for this vehicle, red is not — and that applies to the shortest route and "
    "to every alternate alike, so a flooded alternate never looks safe just "
    "because it is the alternate. The route with the white outline is the one "
    "you have selected. Small red dots are flooded stretches; switch any layer "
    "off with the box on the map."
)
if not is_passable(shortest) and safe_alternates:
    st.success(
        f"**The shortest route is red — it crosses "
        f"{shortest['score']['flooded_m']} m of standing water.** The blue "
        f"line is the way around it: "
        f"{safe_alternates[0].get('label', 'the alternate')}, "
        f"{safe_alternates[0]['distance_m'] / 1000:.1f} km."
    )
elif not is_passable(shortest):
    st.error(
        "The shortest route is flooded and no alternate on offer is clear "
        "either. Every line on the map is red. Consider waiting."
    )

# --------------------------------------------------------------------------- #
# Risk along the selected route
# --------------------------------------------------------------------------- #
samples = selected["score"]["samples"]
risks = selected["score"]["risk_at"]
if len(samples) > 1:
    st.subheader("Risk along this route")
    travelled = 0.0
    # NOT `profile` — that name already holds the traveller's settings, and
    # reusing it here quietly replaced the dict with a list of chart rows.
    # Everything below this point that read profile["prefer"] then died with
    # "list indices must be integers", several screens further down the page.
    risk_profile = []
    for i, (point, risk) in enumerate(zip(samples, risks)):
        if i:
            travelled += fe.haversine_m(samples[i - 1], point)
        risk_profile.append({"Distance (km)": round(travelled / 1000, 2),
                             "Risk %": risk})
    st.line_chart(pd.DataFrame(risk_profile).set_index("Distance (km)"), height=200)
    st.caption(
        f"Every 120 m along the route, taking the risk of the nearest drain. "
        f"The flooded threshold is {threshold}% — anything above that line is water "
        "deep enough to matter."
    )

# --------------------------------------------------------------------------- #
# The comparison, which is the argument for the detour
# --------------------------------------------------------------------------- #
st.subheader("Routes considered")
if fetched.get("alternatives", 0) == 0:
    st.caption(
        "Only one route could be built between these two points — every "
        "detour tried came back on the same roads."
    )
else:
    st.caption(
        f"{len(routes)} routes, so there is always something to compare "
        f"against. Ranked for **{profile_name.lower()}**: "
        + {"time": "quickest, wet or not.",
           "safety": "quickest route that stays passable.",
           "driest": "least water, even if it is slower.",
           "shortest": "least distance among the passable ones."}[profile["prefer"]]
    )
if fetched["detours_tried"]:
    st.caption(
        "Nothing OSRM offered was dry, so detours were built through: "
        + ", ".join(fetched["detours_tried"])
    )
st.dataframe(
    pd.DataFrame([{
        "Route": r.get("label", f"Option {i + 1}"),
        "Status": " · ".join(filter(None, [
            "Selected" if r is selected else "",
            "Recommended" if r is chosen else "",
            "Original" if r is fastest else "",
        ])) or "",
        "Distance": f"{r['distance_m'] / 1000:.1f} km",
        "Arrive": (lambda e: f"{e['arrive_low']}-{e['arrive_high']}")(
            rr.eta(r, intensity_at_departure, minutes_ahead)),
        "Est. time": (lambda e: f"{e['minutes_low']}-{e['minutes_high']} min")(
            rr.eta(r, intensity_at_departure, minutes_ahead)),
        "Risk factor": rr.route_risk_factor(
            r, float(threshold),
            rr.eta(r, intensity_at_departure, minutes_ahead))["score"],
        "Worst point": f"{r['score']['max_pct']:.0f}%",
        "Through water": f"{r['score']['flooded_m']} m" if r["score"]["flooded_m"] else "none",
        "Verdict": "Flooded" if r["score"]["max_pct"] >= threshold else "Clear",
    } for i, r in enumerate(routes)]),
    hide_index=True, width="stretch",
)

# --------------------------------------------------------------------------- #
# Turn by turn, for the selected route
# --------------------------------------------------------------------------- #
st.subheader(f"Directions — {selected.get('label', 'selected route')}")
st.dataframe(
    pd.DataFrame([{
        "#": i,
        "Instruction": step["instruction"],
        "For": f"{round(step['distance_m'])} m" if step["distance_m"] >= 1 else "",
    } for i, step in enumerate(selected["steps"], start=1)]),
    hide_index=True, width="stretch", height=340,
)

hotspots = selected["score"]["hotspots"]
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
the nearest drain, fading to nothing 400 metres away — a road 50 m from a
flooded gully is wet, a road 400 m away is not. A route's score is its worst
sample, and its "through water" figure is how much of its length sits above
the threshold.

The quickest **clear** route is the one recommended. If nothing is clear, the
driest one is recommended and the page says so rather than implying the
journey is safe. You can still select any of the others — the numbers above,
the map and the directions all follow the selection, not the recommendation.

The limitation worth knowing: risk is inferred from ten drain locations, so
the spatial resolution of the flood map is coarse. More drains, or a proper
inundation raster, would sharpen it. The routing logic does not change.
        """
    )
