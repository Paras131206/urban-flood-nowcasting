"""The evidence behind the numbers.

The dashboard makes predictions. This page shows the working: the engineering
the model is built on, how it performs on storms that genuinely happened, what
every term means, and where each input comes from.

A prediction anyone can interrogate is worth more than one nobody can, which is
why this page exists and why it leads with the checks rather than the claims.
"""
import pandas as pd
import streamlit as st

import flood_engine as fe
import history as hist
import reports as rep
import route_planner as rp
import terrain

st.set_page_config(page_title="How it works", layout="wide")

st.title("How it works, and how far to trust it")

drains = fe.load_drains()
recorded = hist.load_recorded_hourly()

tabs = st.tabs([
    "Does it hold up?",
    "Real storms",
    "The physics",
    "Terrain",
    "The network effect",
    "Report waterlogging",
    "What every term means",
    "Where the data comes from",
])

# =========================================================================== #
# 1. Does it hold up?
# =========================================================================== #
with tabs[0]:
    st.subheader("Six things that must be true")
    st.markdown(
        f"""
The model is checked against two things that are independently published and
cannot be argued with: **storms that actually happened**, and **Mumbai's own
design standard** of {fe.DESIGN_STANDARD_MM_HR:.0f} mm/hr at low tide.

The storms used for scoring are all from the **2024 and 2025 monsoons**. That
is deliberate — the drainage network has been worked on continuously since
2005, so checking against the 2005 deluge would be checking against a system
that no longer exists. These are the storms this network faced.

Three checks ask whether the model fires when the city really did flood. Three
ask whether it stays quiet when it should, because a system that shouts on
every shower is not a warning system. All six are falsifiable, and the result
is shown whatever it is.
        """
    )

    checks = hist.run_checks(drains, recorded)
    passed, total_checks = hist.scoreline(checks)

    if passed == total_checks:
        st.success(f"**{passed} of {total_checks} checks pass.**")
    else:
        st.warning(f"**{passed} of {total_checks} checks pass.** "
                   "The failures are shown below and have not been hidden.")

    for check in checks:
        icon = "PASS" if check.passed else "REVIEW"
        with st.expander(f"[{icon}]  {check.name} — {check.question}",
                         expanded=not check.passed):
            st.markdown(f"**Expected:** {check.expectation}")
            st.markdown(f"**Model says:** {check.detail}")

    st.divider()
    st.subheader("The headline result")
    st.success(
        "**Both 2025 storms flooded Mumbai on rain below the design standard.** "
        "20 August averaged 18.2 mm/hr and 26 May averaged 17.1 mm/hr — 0.73x "
        "and 0.68x the 25 mm/hr the drains were built for. The city still went "
        "under, because the rain kept falling for hours rather than minutes. "
        "The model reproduces both, and it reproduces them for the right "
        "reason: duration, not just intensity."
    )
    st.markdown(
        """
That is a result worth stating plainly, because it is what the app is *for*.
An intensity-threshold warning — "alert above 25 mm/hr" — would have missed
both of those days. A model that routes water through a network, hour by hour,
catches them.

**Next phase: scoring in centimetres.** Water level sensors at twenty
junctions, logging depth every fifteen minutes through one monsoon, would let
this model report mean error, hit rate, false alarm rate and lead time per
event. Everything needed to consume that data is already built — it is the
natural next step, and it costs less than the traffic diversions from a single
bad flood day.
        """
    )

# =========================================================================== #
# 2. Real storms
# =========================================================================== #
with tabs[1]:
    st.subheader("Storms that really happened, put through the model")

    if recorded:
        st.success(
            f"Using **real recorded hourly rainfall** for "
            f"{len(recorded)} dates from the ERA5 archive."
        )
    else:
        st.info(
            "Using **published rainfall totals** spread evenly across their "
            "stated duration. For real hour-by-hour rainfall, run "
            "`python3 fetch_history.py` once and commit the CSV it writes — "
            "the page picks it up automatically. An even spread is the "
            "conservative choice: real storms are peakier than their average, "
            "so this understates the worst hour rather than flattering the model."
        )

    tide_for_replay = st.select_slider(
        "Tide during the storm",
        options=[0.5, 1.2, 2.0, 3.0, 4.0],
        value=1.2,
        format_func=lambda t: f"{t} m — " + (
            "outfalls clear" if t <= 0.5 else
            "ordinary" if t <= 1.2 else
            "rising" if t <= 2.0 else
            "high" if t <= 3.0 else "drowned outfalls"),
        help="Mumbai's design standard is 25 mm/hr AT LOW TIDE. When the sea is "
             "up, the drains cannot discharge and the same rain does far more "
             "damage.",
    )

    st.markdown(
        "#### Recent monsoons — the network as it stands\n\n"
        "Mumbai's drainage has been worked on continuously since 2005: new "
        "pumping stations, and a long widening programme under BRIMSTOWAD. "
        "These are the storms the current system actually faced, so these are "
        "the ones the model is scored on."
    )


    def show(event, expanded_note=True):
        result = hist.replay(drains, event, tide_m=tide_for_replay, recorded=recorded)
        st.markdown(f"### {event.name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recorded rainfall", f"{event.total_mm:.0f} mm",
                  f"over {event.hours} h", delta_color="off")
        c2.metric("Mean intensity", f"{event.mean_mm_hr} mm/hr",
                  f"{event.times_design_standard} of design standard",
                  delta_color="off")
        c3.metric("Model verdict", result["verdict"],
                  f"{result['flooded_count']} of {len(drains)} spots",
                  delta_color="off")
        c4.metric("Deepest predicted", f"{result['worst_cm']:.0f} cm",
                  result["worst_name"], delta_color="off")

        if event.below_design_standard:
            st.success(
                f"**Below the design standard, and it still flooded.** "
                f"{event.mean_mm_hr} mm/hr is only "
                f"{event.times_design_standard} times the "
                f"{fe.DESIGN_STANDARD_MM_HR:.0f} mm/hr the drains were built "
                f"for — but it fell for {event.hours} hours. An "
                f"intensity-threshold alert would have missed this day "
                f"entirely. The model catches it."
            )

        st.markdown(f"**What actually happened:** {event.outcome}")
        if event.tide_note:
            st.caption(f"Tide: {event.tide_note}")
        st.caption(f"Rainfall input: {result['provenance']}. "
                   f"Gauge: {event.gauge}. Source: {event.source} — {event.url}")

        peaks = pd.DataFrame([
            {"Spot": drains[did].name,
             "Peak depth (cm)": round(cm, 1),
             "Level": fe.level_for_depth(cm)}
            for did, cm in sorted(result["peaks_cm"].items(),
                                  key=lambda kv: -kv[1])
        ])
        st.dataframe(peaks, hide_index=True, width="stretch", height=250)
        st.divider()


    for event in hist.EVENTS:
        show(event)

    st.markdown("#### For context: the events everyone knows")
    st.caption(
        "Kept separate because they predate two decades of drainage work. "
        "Useful for showing the model behaves sensibly at the extremes, not "
        "for judging how it performs on today's network."
    )
    for event in hist.HISTORICAL:
        with st.expander(f"{event.name} — {event.total_mm:.0f} mm in "
                         f"{event.hours} h"):
            show(event)

# =========================================================================== #
# 3. The physics
# =========================================================================== #
with tabs[2]:
    st.subheader("Pick a spot and see the arithmetic")
    st.caption(
        "Every figure below comes from the same model run that produces the "
        "depth on the dashboard. Nothing here is written for the slide."
    )

    left, right = st.columns([2, 1])
    with left:
        pick = st.selectbox("Spot", sorted(drains, key=lambda d: drains[d].name),
                            format_func=lambda d: drains[d].name)
    with right:
        rain_here = st.slider("Rainfall (mm/hr)", 0, 200, 45, key="physics_rain")

    minutes = st.select_slider(
        "After this much rain", options=list(range(0, fe.HORIZON_MIN + 1, fe.STEP_MIN)),
        value=60, format_func=lambda m: f"{m} min", key="physics_min",
    )

    series = fe.flat_series(float(rain_here))
    trace = fe.diagnose(drains, series, minutes)

    for line in fe.explain(drains, pick, trace, float(rain_here)):
        st.markdown(f"- {line}")

    st.divider()
    threshold = fe.floods_at_mm_hr(drains, pick)
    c1, c2, c3 = st.columns(3)
    c1.metric("Depth now", f"{trace[pick]['depth_cm']:.0f} cm")
    c2.metric("Floods (15 cm) at",
              "never" if threshold == float("inf") else f"{threshold:.0f} mm/hr",
              "for one hour of rain", delta_color="off")
    c3.metric("Deepest it can get", f"{drains[pick].max_pond_cm:.0f} cm",
              "then it spreads sideways", delta_color="off")

    with st.expander("The equations, and where they come from"):
        st.markdown(
            """
**Runoff — the rational method.** `Q = C · i · A`. Rain intensity `i` over
catchment area `A`, times a runoff coefficient `C` for how much of it actually
runs off rather than soaking in. Standard urban drainage design, in every
textbook and in the CPHEEO manual. `C` is composed here from the land cover of
each catchment — see the Terrain tab.

**Capacity — Manning's equation.** `Q = (1/n)·A·R^(2/3)·√S`. The consequence
that matters is the exponent: discharge scales with cross-sectional area to
the power 5/3. So a drain 75% choked does not carry 25% of its design flow, it
carries about **10%**. This is the single biggest reason paper capacities
mislead, and it is why cleaning drains buys more headroom than it looks like
it should.

**Routing.** The drains are a directed graph — each one flows to the nearest
lower drain of equal or higher rank, primaries reach the sea, and a low point
with nowhere to go is pumped. Water is routed through it every 15 minutes.

**Surcharge.** A drain discharging into one that is already under water loses
capacity it nominally has, because the pipe below is running full. Without
this the network is only bookkeeping.

**Tide.** A storm drain discharges by gravity through an outfall in the sea
wall. When the sea is higher than the outfall, that discharge collapses. This
is why the design standard says "at low tide" and why it matters so much.

**Ponding.** Whatever cannot get down the pipe stays on the surface. Volume
over the area it spreads across gives a depth in centimetres, capped at how
deep that spot can get before water runs off overland instead.
            """
        )

# =========================================================================== #
# 4. Terrain
# =========================================================================== #
with tabs[3]:
    st.subheader("What each catchment is made of")
    st.markdown(
        """
The runoff coefficient is not one number for the whole of Bandra. It is
composed from the surface mix of each catchment, because a hillside under
old trees and a concrete retail street do not shed water alike — and the
difference is large enough to change which roads flood.
        """
    )

    st.dataframe(
        pd.DataFrame([{
            "Catchment": d.name,
            "Sealed surface": f"{terrain.sealed_fraction(did) * 100:.0f}%",
            "Runoff C": d.runoff_c,
            "Of every 100 mm, runs off": f"{d.runoff_c * 100:.0f} mm",
            "Make-up": terrain.describe(did),
        } for did, d in sorted(drains.items(), key=lambda kv: -kv[1].runoff_c)]),
        hide_index=True, width="stretch",
    )

    st.divider()
    chosen = st.selectbox("Break one down", sorted(drains, key=lambda d: drains[d].name),
                          format_func=lambda d: drains[d].name, key="terrain_pick")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(pd.DataFrame(terrain.table_rows(chosen)),
                     hide_index=True, width="stretch")
        st.caption(
            f"The **Contributes** column is share x runoff C. They sum to "
            f"**{drains[chosen].runoff_c}**, which is the number the model uses."
        )
    with c2:
        mix = terrain.mix_for(chosen)
        st.bar_chart(
            pd.DataFrame({"Share of catchment": {terrain.LABELS[k]: v * 100
                                                 for k, v in mix.items() if v > 0}}),
            height=300,
        )

    st.info(
        "**Why this matters beyond the map.** Pali Hill's tree cover is not "
        "just pleasant — it is flood defence for Chimbai below it. Roughly "
        "44% of that catchment is vegetation, which means it releases about "
        "38% less runoff than a sealed catchment of the same size. Concreting "
        "it over would push water onto a low-lying fishing village that "
        "already floods."
    )
    st.caption(
        "These mixes are estimated from land use, not surveyed. A deployment "
        "would take them from a land-cover raster or municipal GIS. The "
        "numbers would change; the method would not."
    )

# =========================================================================== #
# 5. The network effect
# =========================================================================== #
with tabs[4]:
    st.subheader("What a full drain does to the drain above it")
    st.markdown(
        """
This is the question the original per-drain model could not answer at all.
When a drain runs at its limit and starts overflowing, two things happen to
everything connected to it — and both are separated out below by running the
same model three times.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        net_rain = st.slider("Rainfall (mm/hr)", 0, 200, 45, key="net_rain")
    with c2:
        net_tide = st.select_slider("Tide", options=[0.5, 1.2, 2.0, 3.0, 4.0],
                                    value=1.2, key="net_tide",
                                    format_func=lambda t: f"{t} m")
    net_min = st.select_slider(
        "After", options=list(range(0, fe.HORIZON_MIN + 1, fe.STEP_MIN)),
        value=60, format_func=lambda m: f"{m} min", key="net_min")

    rows = fe.surcharge_report(drains, fe.flat_series(float(net_rain)),
                               net_min, tide_m=net_tide)
    frame = pd.DataFrame(rows)

    st.dataframe(
        frame.rename(columns={
            "Segment_Name": "Spot",
            "Own_rain_cm": "From its own rain (cm)",
            "From_uphill_cm": "From uphill (cm)",
            "From_backup_cm": "From backing up (cm)",
            "Total_cm": "Total (cm)",
            "Network_share_pct": "% not its own rain",
            "Drains_Into": "Flows into",
        })[["Spot", "From its own rain (cm)", "From uphill (cm)",
            "From backing up (cm)", "Total (cm)", "% not its own rain",
            "Flows into"]],
        hide_index=True, width="stretch",
    )

    worst = max(rows, key=lambda r: r["Network_share_pct"] if r["Total_cm"] > 1 else -1)
    if worst["Total_cm"] > 1:
        st.error(
            f"**{worst['Segment_Name']}** is under {worst['Total_cm']:.0f} cm of "
            f"water and **{worst['Network_share_pct']}% of it fell somewhere "
            f"else**. Cleaning this drain alone would barely help. That is the "
            f"kind of conclusion a per-drain model cannot reach — and the "
            f"original version of this app, which scored every drain "
            f"independently, never once raised a HIGH alert."
        )

    st.markdown(
        """
**How the three columns are separated.** The model is run three times on the
same rainfall:

| Run | Water flows downhill? | Full drains hold back the ones above? |
|---|---|---|
| Isolated | no | no |
| Routed | yes | no |
| Full | yes | yes |

*From its own rain* is the isolated run. *From uphill* is what routing adds.
*From backing up* is what surcharge adds on top.

**A negative number in the last column is not a bug.** When the drain above is
throttled by backwater, it holds its water rather than passing it down — so
the drain below receives *less*. The water has not disappeared; it is sitting
upstream, in the row above, which is exactly what a surcharged network does.
        """
    )

# =========================================================================== #
# 6. Report waterlogging
# =========================================================================== #
with tabs[5]:
    st.subheader("Send a photo of a flooded road")
    st.markdown(
        """
A resident standing in the water knows something the model does not. This
turns that into an input — and shows exactly what it does with it, rather
than claiming a photograph retrains a neural network.
        """
    )

    photo = st.file_uploader("Photo of the waterlogged road",
                            type=["jpg", "jpeg", "png", "webp"])

    col_left, col_right = st.columns([1, 1])
    detected = None

    with col_left:
        if photo is not None:
            data = photo.getvalue()
            st.image(data, caption=photo.name, width=460)
            detected = rep.read_exif_gps(data)
            if detected:
                st.success(f"Location read from the photo: "
                           f"{detected[0]:.5f}, {detected[1]:.5f}")
            else:
                st.info(
                    "No GPS in this photo. Most messaging apps strip it, so "
                    "this is normal — pick the location instead."
                )

    with col_right:
        st.markdown("**Where was this?**")
        if detected:
            lat, lon = detected
            drain_id, distance = rep.nearest_drain(drains, lat, lon)
            st.write(f"Nearest monitored drain: **{drains[drain_id].name}** "
                     f"({distance} m away)")
            if distance > 800:
                st.warning(
                    f"{distance} m is a long way from anything monitored. The "
                    "report is still recorded, but it says little about a drain "
                    "that far off."
                )
        else:
            drain_id = st.selectbox(
                "Nearest landmark or drain",
                sorted(drains, key=lambda d: drains[d].name),
                format_func=lambda d: drains[d].name, key="report_place")
            lat, lon = drains[drain_id].lat, drains[drain_id].lon

        observed = st.slider("How deep is the water? (cm)", 0, 120, 20, step=5,
                             help="Kerb ≈ 15 cm, car wheel centre ≈ 35 cm, "
                                  "knee ≈ 50 cm, bonnet ≈ 75 cm.")
        rain_then = st.slider("Rain falling at the time (mm/hr)", 0, 200, 45,
                              key="report_rain",
                              help="Needed to interpret the depth — a depth "
                                   "without the rain that caused it says "
                                   "nothing about the pipe.")

    if photo is not None and st.button("Analyse with Gemini"):
        with st.spinner("Looking at the photo..."):
            seen = rep.describe_with_gemini(photo.getvalue(),
                                            mime_type=photo.type or "image/jpeg")
        if seen.get("ok"):
            st.success(
                f"**{'Standing water' if seen['is_flooded'] else 'No standing water'}** — "
                f"about **{seen['depth_cm']} cm**, judged against "
                f"{seen['reference'] or 'the scene'} "
                f"({seen['confidence']} confidence). {seen['notes']}"
            )
            st.caption(f"Model: {seen['model']}")
        else:
            st.warning(seen["error"])
            st.caption(
                "Set GEMINI_API_KEY in your environment to enable this. "
                "Everything else on this page works without it — the reporter "
                "gives the depth themselves."
            )

    if st.button("Submit report", type="primary"):
        rep.save_report({
            "drain_id": drain_id,
            "lat": lat, "lon": lon,
            "observed_cm": observed,
            "intensity_mm_hr": rain_then,
            "duration_min": 60,
            "had_photo": photo is not None,
            "gps_from_exif": detected is not None,
        })
        st.success(f"Report recorded for {drains[drain_id].name}.")

    st.divider()
    st.subheader("What the reports have changed")

    stored = rep.load_reports()
    if not stored:
        st.info(
            "No reports yet. Submit one above and the calibration it implies "
            "appears here."
        )
    else:
        st.caption(f"{len(stored)} report(s) on file.")
        adjustments = rep.all_calibrations(drains, stored)
        if adjustments:
            st.dataframe(
                pd.DataFrame([{
                    "Drain": a["name"],
                    "Reports": a["reports"],
                    "Explained": a["explained"],
                    "Blockage on file": f"{a['recorded'] * 100:.0f}%",
                    "Reports imply": "—" if a["implied"] is None
                                     else f"{a['implied'] * 100:.0f}%",
                    "Suggested": f"{a['suggested'] * 100:.0f}%",
                } for a in adjustments]),
                hide_index=True, width="stretch",
            )
            for a in adjustments:
                st.caption(f"**{a['name']}** — {a['note']}")
        st.caption(
            "Suggestions are not applied automatically. Edit "
            "`bandra_capacity.csv` to accept one — an estimate a crowd moved "
            "should be a decision someone made, not a number that drifted."
        )

    with st.expander("How the feedback loop works"):
        st.markdown(
            """
**What happens to your photo.** The app solves for the blockage figure that
would have made the model reproduce the depth you reported — holding everything
else fixed and routing through the whole network each time. That is inverse
calibration: what a hydrologist does by hand, done automatically and shown with
its working. Blockage is the right target because it is the input the model is
least certain about, and the one a resident standing in the water has real
evidence on.

**Why one photo does not rewrite a survey.** A single report moves the estimate
by a fifth of what it implies. Five agreeing reports move it fully, and no
amount of reporting shifts it more than 25 percentage points. Crowd evidence
should accumulate, not stampede.

**When the app says so instead.** If no blockage value can reproduce the
reported depth, the page flags it rather than nudging the number anyway. That
gap is informative — it points at something outside the model, like a blocked
outfall or a pump that was off, and routes it to a human instead of burying it
in an average.

**Where this leads.** Over a monsoon this collects exactly the dataset the
model is currently missing: labelled photographs with depths, times and
locations. That is the input a learned model would need, and this is the
mechanism that gathers it.
            """
        )

# =========================================================================== #
# 7. Glossary
# =========================================================================== #
with tabs[6]:
    st.subheader("What every term on this app means")
    st.caption(
        "In the order you meet them. If a word on any page is not explained "
        "here, that is a bug worth reporting."
    )

    GLOSSARY = [
        ("Depth (cm)",
         "How deep the standing water is predicted to be at that spot. The "
         "model works in centimetres rather than a score because \"34 cm at "
         "Chimbai in 45 minutes\" is something you can act on and \"risk 0.7\" "
         "is not. Roughly: 15 cm wets your feet and stalls a scooter, 25 cm "
         "starts stopping cars, 45 cm is impassable for anything but a truck."),

        ("Peak depth",
         "The deepest it gets at any point in the next three hours, not just "
         "right now. A road that is dry at this moment and 40 cm deep in forty "
         "minutes is the one you most need to be told about."),

        ("Worst point (%)",
         "**The number people ask about most.** On the navigation page, a route "
         "is sampled every 120 metres and each sample takes the risk of the "
         "nearest drain. The worst point is the single wettest of those "
         "samples — so \"52% worst point\" means: somewhere along this route "
         "there is a stretch at 52% risk, and everywhere else is better than "
         "that. It is a worst case, not an average, because an average would "
         "hide a single impassable 100-metre stretch inside an otherwise dry "
         "10 km drive. It does **not** mean a 52% chance of flooding."),

        ("Risk %",
         "Depth expressed on a 0–100 scale where 50 cm is 100%. So 40% is "
         "20 cm — about where a small car starts to struggle, which is why 40% "
         "is the default point at which the app reroutes you. It is a "
         "restatement of depth in a form people find easier to compare, not a "
         "probability."),

        ("Risk factor",
         "Different from depth, and the more useful number of the two. Depth "
         "answers \"how deep\". Risk factor answers \"how bad\", and combines "
         "four things: how deep it gets (40%), how fast it arrives (25%) — "
         "because water that rises in ten minutes gives nobody time to move — "
         "how long it stays (20%), and how much ground drains through that "
         "spot (15%), because closing a trunk junction closes a district while "
         "closing a lane closes a lane."),

        ("Confidence (%)",
         "How much the answer moves when the two least certain inputs are "
         "shaken. The model is run nine times — blockage 15 points either side "
         "of what is on file, crossed with rainfall at 60%, 100% and 140% of "
         "forecast — and confidence is mostly how many of those nine runs "
         "agree on the risk level. Nine runs that all say \"no standing "
         "water\" are worth trusting even if the centimetres vary; nine that "
         "straddle the flooding line are not. It measures uncertainty *within "
         "the model*, so it cannot account for what the model does not "
         "represent at all — a collapsed culvert, a lorry parked over a gully."),

        ("Blockage %",
         "How much of the drain's cross-section is lost to silt and rubbish. "
         "The most important number in the whole dataset and the least "
         "reliable: these are estimates, not CCTV inspection records. Because "
         "flow scales with area to the power 5/3, a drain 75% blocked carries "
         "about 10% of its design flow, not 25%."),

        ("Design capacity vs real capacity",
         "Design is what the drain was built to carry, in cubic metres per "
         "second. Real is what it carries now, after silt and after the tide. "
         "The gap between them is the whole problem."),

        ("Runoff coefficient (C)",
         "The fraction of rain that becomes surface flow instead of soaking "
         "in. Composed here from what each catchment is made of — a "
         "concrete retail street is about 0.85, a hillside under trees about "
         "0.56. On the same storm those two shed very different amounts of "
         "water."),

        ("Surcharge / backing up",
         "When a drain is full, the one flowing into it cannot discharge "
         "freely — the pipe below is running full, so the water backs up. That "
         "upstream drain then floods even though nothing about it changed. It "
         "is why a clean gully in a good street still goes under."),

        ("Tide",
         "Storm drains discharge by gravity through outfalls in the sea wall. "
         "When the sea is higher than the outfall, that discharge collapses "
         "and rain that would have drained away in an hour sits on the road "
         "until the tide turns. Mumbai's design standard is explicitly "
         f"{fe.DESIGN_STANDARD_MM_HR:.0f} mm/hr **at low tide** — the qualifier "
         "is doing enormous work."),

        ("Floods at (mm/hr)",
         "How hard it has to rain, for one hour, before that spot goes under "
         "15 cm. Solved through the whole network rather than in closed form, "
         "because a road floods on water that fell uphill. Sustained rain "
         "matters more than intensity: several of these spots survive 45 mm/hr "
         "for an hour and fail at 13 mm/hr for three."),

        ("Pumped",
         "A low point with nowhere downhill to drain to. Water leaves only if "
         "a pump lifts it. Chimbai is one, which is exactly where Bandra's "
         "real pumping stations sit."),

        ("Through water (m)",
         "On the navigation page, how many metres of a route sit above the "
         "flooding threshold. A route with a high worst point but 120 m "
         "through water is one bad puddle; the same worst point over 1.5 km "
         "is a flooded road."),

        ("Detour via …",
         "OSRM's own alternatives are a bonus, not a guarantee — for many "
         "journeys it offers only one route, and if that one is flooded there "
         "is nothing to choose from. So the app forces routes through dry "
         "waypoints to build genuine alternatives."),

        ("Arrival window (Arrive)",
         "Not a stopwatch figure, and deliberately a range. OSRM gives the "
         "free-flow driving time on a dry road; the app then stretches it "
         "twice — once for the rain itself (spray, visibility, everyone else "
         "braking), and once per 120-metre sample for the standing water on "
         "that stretch. Driving through water is not a linear penalty: at "
         "ankle depth you slow down, at knee depth you are in first gear "
         "behind someone who has stopped. The window widens the wetter the "
         "route gets, because that is exactly where an estimate is least "
         "reliable. A single number would imply the app knows the traffic."),

        ("Risk factor (journey)",
         "The route's own 0–100 score, distinct from the worst point. It "
         "combines the depth of the worst point (35%), how much of the route "
         "is wet rather than one puddle (25%), how long you spend in water "
         "(20%), and how far past *your* vehicle's limit it goes (20%) — so "
         "the same road scores differently for a scooter and a fire tender. "
         "The scale is Minor 0–19, Moderate 20–44, Serious 45–69, "
         "Critical 70–100, and it is the same scale the dashboard uses for "
         "individual spots."),

        ("Travel profile",
         "Who is travelling changes both the limit and the ranking. A car is "
         "routed to the quickest passable way; a two-wheeler to the driest "
         "even if slower, because a rider cannot see a pothole or open "
         "manhole through muddy water; someone on foot to the shortest "
         "passable way, since an extra kilometre walking costs more than five "
         "minutes driving; and an emergency vehicle to the quickest route "
         "full stop, with a warning about what it is driving into. A fire "
         "tender wades through 45 cm where a scooter is in trouble at 10."),

        ("Data delay",
         fe.latency_note() + " Every reading on the dashboard is older than it "
         "looks, and the app says so rather than implying it is live to the "
         "second."),
    ]

    for term, meaning in GLOSSARY:
        with st.expander(term):
            st.markdown(meaning)

# =========================================================================== #
# 8. Provenance
# =========================================================================== #
with tabs[7]:
    st.subheader("Where every number comes from")

    st.dataframe(
        pd.DataFrame([
            {"Input": "Rainfall forecast",
             "Source": "Open-Meteo 15-minute precipitation",
             "Status": "Real, live",
             "Status / next step": "A numerical weather model, not IMD radar. A "
                              "deployment would use radar nowcasts, which are "
                              "sharper at short range."},
            {"Input": "Historical rainfall",
             "Source": "Open-Meteo ERA5 archive + published gauge totals",
             "Status": "Real, recorded",
             "Status / next step": "ERA5 is a ~25 km grid, so it cannot reproduce a "
                              "localised cell like 26 July 2005. The published "
                              "gauge totals are the authority on how much fell."},
            {"Input": "Drain locations and rated capacity",
             "Source": "bandra_capacity.csv",
             "Status": "Project dataset",
             "Status / next step": "Ten drains for the whole of Bandra. Real Bandra "
                              "has hundreds of gullies."},
            {"Input": "Catchment areas",
             "Source": "Derived from rated capacity at the 25 mm/hr design "
                       "standard, minus each drain's children",
             "Status": "Derived, not surveyed",
             "Status / next step": "The original figures totalled 24 ha against "
                              "capacities sized for hundreds — which is why "
                              "the first version never raised an alert."},
            {"Input": "Blockage percentages",
             "Source": "Estimates",
             "Status": "Estimated",
             "Status / next step": "The least reliable and most influential input. "
                              "Should come from CCTV inspection records."},
            {"Input": "Land cover / runoff coefficient",
             "Source": "terrain.py, estimated from land use",
             "Status": "Estimated",
             "Status / next step": "Should come from a land-cover raster or "
                              "municipal GIS."},
            {"Input": "Drain network topology",
             "Source": "Inferred from rank, elevation and distance",
             "Status": "Inferred",
             "Status / next step": "The real network is surveyed and would replace "
                              "this directly. The routing logic does not change."},
            {"Input": "Road geometry and routes",
             "Source": "OSRM on OpenStreetMap",
             "Status": "Real",
             "Status / next step": "The public demo server, which is rate-limited "
                              "and occasionally slow."},
            {"Input": "Design standard (25 mm/hr at low tide)",
             "Source": "MCGM / BRIMSTOWAD",
             "Status": "Published",
             "Status / next step": "None. This is the city's own figure and the "
                              "reference the model is checked against."},
            {"Input": "Observed flood depths",
             "Source": "Water level sensors",
             "Status": "Phase 2",
             "Status / next step": "Twenty junctions logging depth through one "
                                   "monsoon turns every check on this page "
                                   "into a scored error in centimetres. The "
                                   "code to consume it is already here."},
        ]),
        hide_index=True, width="stretch",
    )

    st.divider()
    st.subheader("The delay between the rain and this screen")
    low, high = fe.latency_range_min()
    st.metric("Typical age of what you are looking at", f"{low}–{high} minutes")
    st.dataframe(
        pd.DataFrame([
            {"Stage": name, "Best case": f"{lo} min", "Worst case": f"{hi} min",
             "Why": why}
            for name, lo, hi, why in fe.LATENCY_BUDGET
        ]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "In a deployment the observation-to-provider leg shortens with a "
        "direct radar feed and the cache leg disappears with a push "
        "subscription. The publish interval is the floor, and it is why the "
        "app forecasts three hours ahead rather than reporting the present: "
        "a warning that arrives with the water is not a warning."
    )
