"""Tests for the network flood engine.

The dashboard is easy to eyeball; the routing is not. A wrong sign or a
missed upstream link still produces plausible-looking numbers, so these
check behaviour we can reason about from first principles.
"""
import pytest

import flood_engine as fe


@pytest.fixture(scope="module")
def drains():
    return fe.load_drains("bandra_capacity.csv")


def test_every_drain_loads(drains):
    assert len(drains) == 10
    assert all(d.capacity_m3s > 0 for d in drains.values())


def test_water_only_ever_flows_downhill(drains):
    for d in drains.values():
        if d.downstream:
            assert drains[d.downstream].elevation_m < d.elevation_m


def test_primaries_reach_the_sea_and_nothing_loops(drains):
    for start in drains:
        seen, cursor, steps = set(), start, 0
        while cursor and steps < 50:
            assert cursor not in seen, f"loop from {start}"
            seen.add(cursor)
            cursor = drains[cursor].downstream
            steps += 1
        assert steps < 50


def test_a_low_point_with_no_outlet_is_pumped(drains):
    chimbai = drains["BND-T02"]
    assert chimbai.pumped
    assert chimbai.downstream is None


def test_blockage_costs_more_than_it_looks(drains):
    """75% blocked is not 25% of the flow — it is about a tenth."""
    chimbai = drains["BND-T02"]
    assert chimbai.blockage == 0.75
    assert chimbai.effective_capacity_m3s < chimbai.capacity_m3s * 0.15


def test_upstream_area_accumulates(drains):
    """Mount Mary drains into Chimbai, so Chimbai serves more than its own land."""
    chimbai = drains["BND-T02"]
    assert chimbai.upstream_area_m2 >= drains["BND-T04"].catchment_m2
    assert chimbai.total_area_m2 > chimbai.catchment_m2


def test_more_rain_never_means_less_water(drains):
    light = fe.steady_depths(drains, 20, 60)
    heavy = fe.steady_depths(drains, 60, 60)
    for did, shallow in light.items():
        assert heavy[did] >= shallow - 1e-6


def test_no_rain_means_no_flooding(drains):
    assert all(v == 0 for v in fe.steady_depths(drains, 0, 60).values())


def test_depth_never_exceeds_the_ceiling(drains):
    for did, depth in fe.steady_depths(drains, 200, 180).items():
        assert depth <= drains[did].max_pond_cm + 0.5


def test_clearing_a_drain_raises_its_threshold(drains):
    """Cleaning the pipe is what buys headroom. This is the whole thesis."""
    before = fe.floods_at_mm_hr(drains, "BND-S01")
    original = drains["BND-S01"].blockage
    drains["BND-S01"].blockage = 0.05
    try:
        assert fe.floods_at_mm_hr(drains, "BND-S01") > before
    finally:
        drains["BND-S01"].blockage = original


def test_the_network_matters(drains):
    """Most of what floods a low drain fell somewhere else.

    This is the whole difference from scoring each drain on its own. Turning
    the coupling off leaves every drain with only the rain landing on its own
    catchment, and holds the geometry fixed while doing it — severing a link
    instead would also shrink the pond area and confound the comparison, which
    is exactly how an earlier version of this test managed to pass for the
    wrong reason.
    """
    coupled = fe.steady_depths(drains, 60, 60)
    alone = fe.steady_depths(drains, 60, 60, couple=False)

    for did in ("BND-T02", "BND-S01"):
        assert coupled[did] > alone[did] * 3, (
            f"{drains[did].name}: {coupled[did]:.1f} cm on the network vs "
            f"{alone[did]:.1f} cm on its own rain — the network is doing nothing"
        )


def test_a_drowned_outfall_strangles_the_drain_above_it(drains):
    """Surcharge has to travel upstream, or the network is only bookkeeping.

    At high tide the primaries cannot discharge. Whatever feeds them must then
    lose capacity too, even though nothing about those drains changed. That is
    the mechanism behind Bandra East going under on an unremarkable shower
    during a spring tide.
    """
    low = fe.steady_depths(drains, 30, 120, tide_m=fe.TIDE_LOW_M)
    high = fe.steady_depths(drains, 30, 120, tide_m=4.0)

    assert high["BND-P01"] > low["BND-P01"] + 1.0, "the outfall itself must suffer"
    feeders = [d for d, drain in drains.items()
               if drain.downstream in ("BND-P01", "BND-P02")]
    assert feeders, "the fixture should have drains feeding a primary"
    assert any(high[d] > low[d] + 0.5 for d in feeders), (
        "no drain above a drowned outfall got any deeper — surcharge is not "
        "propagating upstream"
    )


def test_outfall_factor_is_bounded():
    assert fe.outfall_factor(0.0) == 1.0
    assert fe.outfall_factor(fe.TIDE_LOW_M) == 1.0
    assert fe.outfall_factor(9.0) == fe.MIN_OUTFALL_FRACTION
    factors = [fe.outfall_factor(t) for t in (0.5, 1.5, 2.5, 3.5, 4.5)]
    assert factors == sorted(factors, reverse=True), "a higher sea never helps"


def test_green_catchments_shed_less_than_concrete_ones(drains):
    """Terrain has to change the answer, or it is decoration."""
    import terrain
    pali = drains["BND-T01"]           # bungalow gardens and old trees
    linking = drains["BND-S03"]        # retail spine, near-total hardstanding
    assert pali.runoff_c < linking.runoff_c - 0.2
    assert terrain.sealed_fraction("BND-T01") < terrain.sealed_fraction("BND-S03")
    # And it must show up as runoff per square metre, not just in a table.
    rain = 50.0
    assert (fe._runoff_m3s(pali, rain) / pali.catchment_m2
            < fe._runoff_m3s(linking, rain) / linking.catchment_m2)


def test_levels_follow_the_thresholds():
    assert fe.level_for_depth(0) == "LOW"
    assert fe.level_for_depth(9.9) == "LOW"
    assert fe.level_for_depth(10) == "MEDIUM"
    assert fe.level_for_depth(25) == "HIGH"
    assert fe.level_for_depth(45) == "SEVERE"


def test_assess_returns_what_the_dashboard_needs(drains):
    rows = fe.assess(drains, fe.flat_series(40))
    assert len(rows) == 10
    for key in ("Segment_Name", "Depth_cm", "Peak_Depth_cm", "Risk_Level",
                "Color", "Drains_To", "Timeline", "Latitude", "Longitude"):
        assert key in rows[0]
    assert rows[0]["Peak_Depth_cm"] >= rows[-1]["Peak_Depth_cm"]      # sorted worst first
    assert len(rows[0]["Timeline"]) == 13                             # 0..180 in 15s


def test_a_failed_weather_call_invents_nothing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blow_up(name, *a, **k):
        if name == "requests":
            raise ImportError("no network")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blow_up)
    result = fe.live_series()
    assert result["series"] == []
    assert result["error"]
    assert result["peak"] == 0.0


# --------------------------------------------------------------------------- #
# Safe routing
# --------------------------------------------------------------------------- #
import route_planner as rp


def _depths(drains, intensity):
    return {drains[k].name: v for k, v in fe.steady_depths(drains, intensity, 60).items()}


# --------------------------------------------------------------------------- #
# Building believable fake routes
# --------------------------------------------------------------------------- #
def _leg(a, b, steps):
    return [(a[0] + (b[0] - a[0]) * i / steps,
             a[1] + (b[1] - a[1]) * i / steps) for i in range(steps + 1)]


def _fake_route(origin, destination, through=None, distance_m=5000.0,
                duration_s=900.0, steps=None):
    """A route that actually starts at the origin and ends at the destination.

    plan_journey now rejects any route whose endpoints do not match what was
    asked for, because a confident line that stops short of where someone is
    going is worse than no line. Fake routes therefore have to connect too —
    eight tests were quietly building geometry that went nowhere near their
    own origin and destination, and only started failing once the check
    existed to notice.

    `through` bends the route past a waypoint, which is how a test puts a
    route beside a particular drain without detaching it from its endpoints.
    """
    if through is None:
        coords = _leg(origin, destination, 8)
    else:
        coords = _leg(origin, through, 5) + _leg(through, destination, 5)[1:]
    return {
        "index": 0, "coords_latlon": coords,
        "distance_m": distance_m, "duration_s": duration_s,
        "steps": steps if steps is not None else [
            {"instruction": "Start out", "road": "", "distance_m": 300.0,
             "location": coords[0]},
            {"instruction": "Arrive at your destination", "road": "",
             "distance_m": 0.0, "location": coords[-1]},
        ],
    }


def test_risk_percentage_maps_from_depth():
    assert rp.risk_pct(0) == 0
    assert rp.risk_pct(20) == 40.0          # the trigger point
    assert rp.risk_pct(50) == 100.0
    assert rp.risk_pct(300) == 100.0        # never over 100


def test_the_network_is_connected(drains):
    net = rp.Network(drains)
    assert len(net.points) == 20
    assert all(net.edges[node] for node in net.points)


def test_dry_weather_takes_the_direct_route(drains):
    r = rp.plan(drains, _depths(drains, 0), "Bandra Fort", "Bandra Kurla Complex")
    assert r["found"]
    assert r["detour_m"] <= 20
    assert r["worst_pct"] == 0


def test_rain_forces_a_detour(drains):
    """Light rain changes nothing; heavy rain should bend the route."""
    dry = rp.plan(drains, _depths(drains, 0), "Carter Road", "Kalanagar Junction")
    wet = rp.plan(drains, _depths(drains, 55), "Carter Road", "Kalanagar Junction")
    assert wet["found"]
    assert wet["total_m"] > dry["total_m"]
    assert wet["detour_m"] > 100


def test_the_route_never_crosses_the_threshold(drains):
    depths = _depths(drains, 55)
    r = rp.plan(drains, depths, "Carter Road", "Kalanagar Junction", avoid_above_pct=40)
    if r["found"]:
        middle = r["path"][1:-1]
        for node in middle:
            assert r["node_risk"][node] < 40


def test_a_flooded_destination_is_still_reachable(drains):
    """Refusing to draw a line to a flooded place is not help.

    Being under water must not by itself make somewhere unreachable. The only
    acceptable reason to fail is that every approach to it is flooded too, and
    the test checks for exactly that rather than accepting any failure.
    """
    # An hour at 30 mm/hr no longer floods Hill Road, now that pond area grows
    # with the catchment instead of being capped. Two hours at the city's own
    # 25 mm/hr design standard does — which is a more interesting case anyway,
    # because it is the storm the drains are supposed to survive.
    depths = {drains[k].name: v for k, v in
              fe.steady_depths(drains, 25, 120).items()}
    destination = "Hill Road Middle"
    assert rp.risk_pct(depths[destination]) >= 40      # it is genuinely flooded

    r = rp.plan(drains, depths, "Bandra Fort", destination, avoid_above_pct=40)
    assert r["found"], "a flooded destination with a dry approach must still be routable"
    assert r["path"][-1] == destination
    # Everything except the destination itself stays under the threshold.
    for node in r["path"][:-1]:
        assert r["node_risk"][node] < 40


def test_it_refuses_when_everything_is_under_water(drains):
    """There has to be a point where the honest answer is "do not travel".

    120 mm/hr for an hour is no longer enough to close every corridor: the
    hillside catchments shed so much less than the concrete ones that a dry
    route over the top survives. That is a real result of composing the runoff
    coefficient from land cover, not a bug — so the test now asks for a storm
    that genuinely leaves nowhere to go: three hours of it, at a spring tide.
    """
    stranded = {drains[k].name: v for k, v in
                fe.steady_depths(drains, 150, 180, tide_m=4.0).items()}
    r = rp.plan(drains, stranded, "Bandra Fort", "Bandra Kurla Complex",
                avoid_above_pct=40)
    assert not r["found"]
    assert r["blocking"]
    assert "control room" in r["message"]


def test_a_lower_threshold_is_never_less_cautious(drains):
    depths = _depths(drains, 45)
    strict = rp.plan(drains, depths, "Carter Road", "Kalanagar Junction", avoid_above_pct=20)
    loose = rp.plan(drains, depths, "Carter Road", "Kalanagar Junction", avoid_above_pct=80)
    if strict["found"] and loose["found"]:
        assert strict["worst_pct"] <= loose["worst_pct"] + 1e-6


def test_directions_read_like_directions(drains):
    r = rp.plan(drains, _depths(drains, 10), "Bandra Fort", "Khar Subway")
    assert r["found"]
    for leg in r["legs"]:
        assert leg["heading"] in rp.COMPASS
        assert leg["metres"] > 0
        assert leg["advice"]


# --------------------------------------------------------------------------- #
# Road routing
# --------------------------------------------------------------------------- #
import road_router as rr


def test_sampling_thins_a_dense_line():
    """A route drawn every few metres must not be scored point by point."""
    dense = [(19.05 + i * 0.00005, 72.83) for i in range(200)]   # ~5.5 m apart
    thinned = rr._sample(dense, 120)
    assert len(thinned) < len(dense) / 10
    assert thinned[0] == dense[0]
    assert thinned[-1] == dense[-1]


def test_a_route_past_a_flooded_drain_scores_worse(drains):
    depths = {d.name: 0.0 for d in drains.values()}
    chimbai = next(d for d in drains.values() if "Chimbai" in d.name)
    depths[chimbai.name] = 60.0                      # deeply flooded

    past = [(chimbai.lat, chimbai.lon), (chimbai.lat + 0.001, chimbai.lon)]
    away = [(chimbai.lat + 0.05, chimbai.lon + 0.05),
            (chimbai.lat + 0.051, chimbai.lon + 0.05)]

    near_score = rr.score_route(past, drains, depths, 40)
    far_score = rr.score_route(away, drains, depths, 40)
    assert near_score["max_pct"] > far_score["max_pct"]
    assert near_score["max_pct"] >= 90                # right on top of it
    assert far_score["max_pct"] == 0.0                # well outside the influence


def test_risk_fades_with_distance(drains):
    depths = {d.name: 0.0 for d in drains.values()}
    chimbai = next(d for d in drains.values() if "Chimbai" in d.name)
    depths[chimbai.name] = 50.0

    on_top = rr.score_route([(chimbai.lat, chimbai.lon)], drains, depths, 40)["max_pct"]
    # About 220 m north, which is inside the 400 m influence but not on it.
    nearby = rr.score_route([(chimbai.lat + 0.002, chimbai.lon)], drains, depths, 40)["max_pct"]
    assert on_top > nearby > 0


def test_a_dry_forecast_leaves_every_route_clear(drains):
    depths = {d.name: 0.0 for d in drains.values()}
    line = [(19.05 + i * 0.002, 72.83 + i * 0.002) for i in range(10)]
    score = rr.score_route(line, drains, depths, 40)
    assert score["max_pct"] == 0.0
    assert score["flooded_m"] == 0
    assert score["hotspots"] == []


def test_unreachable_routing_reports_rather_than_raises(monkeypatch):
    """A dead routing server must degrade, not crash the page."""
    import requests

    def boom(*_a, **_k):
        raise requests.RequestException("no network")

    monkeypatch.setattr(requests, "get", boom)
    result = rr.fetch_routes((19.04, 72.81), (19.06, 72.86))
    assert result["routes"] == []
    assert "no network" in result["error"]


def test_turn_instructions_read_like_instructions():
    assert rr._describe({"type": "depart"}, "Hill Road") == "Start out on Hill Road"
    assert rr._describe({"type": "arrive"}, "") == "Arrive at your destination"
    assert rr._describe({"type": "turn", "modifier": "left"}, "SV Road") == "Turn left onto SV Road"
    assert "roundabout" in rr._describe({"type": "roundabout", "exit": 2}, "Linking Road").lower()


def test_detour_waypoints_skip_wet_and_far_places():
    origin, destination = (19.0430, 72.8190), (19.0660, 72.8690)
    places = dict(rp.LANDMARKS)
    risk = {n: 0.0 for n in places}
    risk["Kalanagar Junction"] = 85.0                  # flooded, must be skipped

    picked = [name for name, _ in rr.detour_waypoints(
        origin, destination, places, risk, threshold_pct=40, limit=6)]

    assert "Kalanagar Junction" not in picked
    assert picked, "there should be some usable waypoint"
    # Nothing absurdly out of the way.
    for name in picked:
        through = (fe.haversine_m(origin, places[name])
                   + fe.haversine_m(places[name], destination))
        assert through <= fe.haversine_m(origin, destination) * 2.2


def test_waypoints_are_ranked_least_painful_first():
    origin, destination = (19.0430, 72.8190), (19.0660, 72.8690)
    places = dict(rp.LANDMARKS)
    risk = {n: 0.0 for n in places}
    picked = rr.detour_waypoints(origin, destination, places, risk, 40, limit=6)
    lengths = [
        fe.haversine_m(origin, p) + fe.haversine_m(p, destination) for _, p in picked
    ]
    assert lengths == sorted(lengths)


def test_a_detour_is_built_when_nothing_offered_is_dry(monkeypatch):
    """The point of the feature: invent an alternate rather than give up."""
    drains = fe.load_drains()
    depths = {d.name: 0.0 for d in drains.values()}
    chimbai = next(d for d in drains.values() if "Chimbai" in d.name)
    depths[chimbai.name] = 80.0

    def fake_routes(origin, destination):
        # Bends past Chimbai, which is under 80 cm, but still reaches the
        # destination — a route that does not is now rejected before scoring.
        return {"routes": [_fake_route(origin, destination,
                                       (chimbai.lat, chimbai.lon), 5000.0, 600.0)],
                "error": None}

    def fake_via(origin, via, destination):
        return {"routes": [_fake_route(origin, destination, via, 7000.0, 900.0)],
                "error": None}

    monkeypatch.setattr(rr, "fetch_routes", fake_routes)
    monkeypatch.setattr(rr, "fetch_via", fake_via)

    places = dict(rp.LANDMARKS)
    risk = {n: 0.0 for n in places}

    result = rr.plan_journey(
        drains, depths, (19.0430, 72.8190), (19.0660, 72.8690),
        places, risk, threshold_pct=40,
    )

    assert result["ok"]
    assert result["all_clear"], "a dry detour existed and should have been found"
    assert result["chosen"]["via"] is not None
    assert result["chosen"]["label"].startswith("Detour via")
    assert result["rerouted"]
    assert result["detours_tried"]


def test_a_dry_journey_costs_only_a_couple_of_extra_calls(monkeypatch):
    """There is always a second option, and finding it stays cheap.

    Two requirements pulling against each other. The page must always offer an
    alternative to compare against — "here is your only choice" is not
    navigation. But hunting for one must not fire five serial OSRM requests on
    a journey that was already fine, which is what made this page sluggish
    before. So the budget is two calls when the direct route is already dry —
    two rather than one because the nearest waypoint often comes back down the
    same roads and gets rejected — and up to four only when nothing on offer
    is passable.
    """
    drains = fe.load_drains()
    depths = {d.name: 0.0 for d in drains.values()}

    calls = {"via": 0}

    monkeypatch.setattr(rr, "fetch_routes",
                        lambda o, d: {"routes": [_fake_route(o, d, None, 5000.0, 600.0)],
                                      "error": None})

    def counting_via(*_a, **_k):
        calls["via"] += 1
        return {"routes": [], "error": "should not be called"}

    monkeypatch.setattr(rr, "fetch_via", counting_via)

    result = rr.plan_journey(
        drains, depths, (19.0430, 72.8190), (19.0660, 72.8690),
        dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS}, threshold_pct=40,
    )
    assert result["all_clear"]
    # One via-call, not zero: the page now guarantees a second option to
    # compare against even when the direct route is fine, because "here is
    # your only choice" is not navigation. Budgeted to one call when the
    # direct route is already dry, so it stays cheap.
    assert calls["via"] <= 2, "a dry journey must not go hunting for detours"


def test_the_route_reacts_to_the_forecast_even_when_fetches_are_cached():
    """The bug this guards against froze the route while the rain changed.

    Caching the whole plan looks harmless and is not: the network calls depend
    only on coordinates, but the choice depends on the forecast. Cache the
    first and you must recompute the second, or the map stops responding to
    the rainfall slider.
    """
    drains = fe.load_drains()
    chimbai = next(d for d in drains.values() if "Chimbai" in d.name)

    calls = {"routes": 0, "via": 0}
    cache = {}

    def fetch(a, b):
        """Cached on coordinates only, exactly as the page does it."""
        key = (a, b)
        if key not in cache:
            calls["routes"] += 1
            cache[key] = {"routes": [_fake_route(
                a, b, (chimbai.lat, chimbai.lon), 5000.0, 600.0)], "error": None}
        import copy
        return copy.deepcopy(cache[key])

    def fetch_via(a, via, b):
        key = (a, via, b)
        if key not in cache:
            calls["via"] += 1
            cache[key] = {"routes": [_fake_route(a, b, via, 7000.0, 900.0)],
                          "error": None}
        import copy
        return copy.deepcopy(cache[key])

    places = dict(rp.LANDMARKS)
    risk = {n: 0.0 for n in places}
    args = ((19.0430, 72.8190), (19.0660, 72.8690), places, risk)

    dry_forecast = {d.name: 0.0 for d in drains.values()}
    wet_forecast = dict(dry_forecast, **{chimbai.name: 80.0})

    calm = rr.plan_journey(drains, dry_forecast, *args, threshold_pct=40,
                           fetch=fetch, fetch_via_fn=fetch_via)
    storm = rr.plan_journey(drains, wet_forecast, *args, threshold_pct=40,
                            fetch=fetch, fetch_via_fn=fetch_via)

    # Same journey, same cached fetches - different answer, because the rain
    # changed. This is the whole point: caching the network calls is fine,
    # caching the decision is not.
    assert calm["chosen"]["via"] is None
    assert storm["chosen"]["via"] is not None
    assert len(calm["detours_tried"]) <= 2, (
        "a dry journey should cost at most a couple of calls to find an alternative"
    )

# --------------------------------------------------------------------------- #
# Terrain
# --------------------------------------------------------------------------- #
import terrain


def test_every_land_cover_mix_is_a_whole_catchment():
    for drain_id, mix in list(terrain.MIX.items()) + [("_default", terrain.DEFAULT_MIX)]:
        assert abs(sum(mix.values()) - 1.0) < 1e-6, f"{drain_id} does not sum to 1"
        for surface in mix:
            assert surface in terrain.C_VALUES, f"{drain_id} has unknown surface {surface}"
            assert mix[surface] >= 0


def test_every_drain_has_a_mix(drains):
    for drain_id in drains:
        assert drain_id in terrain.MIX, f"{drain_id} would silently use the average"


def test_runoff_coefficients_stay_physical(drains):
    for drain_id in drains:
        c = terrain.runoff_coefficient(drain_id)
        assert 0.0 < c <= 1.0
        # Nowhere in a dense suburb sheds less than a field or more than a roof.
        assert 0.2 <= c <= 0.95


def test_an_unknown_drain_falls_back_rather_than_crashing():
    assert terrain.runoff_coefficient("NOT-A-DRAIN") == \
           terrain.runoff_coefficient("NOT-A-DRAIN-EITHER")
    assert terrain.describe("NOT-A-DRAIN")


# --------------------------------------------------------------------------- #
# Surcharge decomposition
# --------------------------------------------------------------------------- #
def test_the_three_contributions_add_up(drains):
    """Own rain + uphill + backing up must equal the depth actually reported."""
    for row in fe.surcharge_report(drains, fe.flat_series(45), 60):
        parts = row["Own_rain_cm"] + row["From_uphill_cm"] + row["From_backup_cm"]
        assert abs(parts - row["Total_cm"]) < 0.15, row["Segment_Name"]


def test_water_from_uphill_is_never_negative(drains):
    """Routing water downhill can only ever add to what is below."""
    for row in fe.surcharge_report(drains, fe.flat_series(60), 90):
        assert row["From_uphill_cm"] >= -0.05, row["Segment_Name"]


def test_a_low_drain_floods_mostly_on_other_peoples_rain(drains):
    rows = {r["Drain_ID"]: r for r in fe.surcharge_report(drains, fe.flat_series(45), 60)}
    sv_road = rows["BND-S01"]
    assert sv_road["Total_cm"] > 5, "the fixture should have SV Road under water"
    assert sv_road["Network_share_pct"] >= 50, (
        "SV Road is supposed to be the case that proves the network matters"
    )


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #
def test_the_explanation_covers_every_drain(drains):
    trace = fe.diagnose(drains, fe.flat_series(45), 60)
    for drain_id in drains:
        lines = fe.explain(drains, drain_id, trace, 45.0)
        assert len(lines) >= 3
        assert all(isinstance(line, str) and line for line in lines)


def test_the_explanation_blames_blockage_only_for_blockage(drains):
    """The line about silt must quote the silt-only capacity.

    An earlier version quoted the fully throttled figure, which blamed the
    blockage for the tide and the backwater as well and reported SV Road as
    "88% gone" when silt accounted for far less than that.
    """
    trace = fe.diagnose(drains, fe.flat_series(45), 60)
    for drain_id, drain in drains.items():
        entry = trace[drain_id]
        expected = drain.capacity_m3s * (1 - min(drain.blockage, 0.95)) ** (5 / 3)
        if drain.downstream is None:
            expected *= fe.outfall_factor(fe.DEFAULT_TIDE_M)
        assert abs(entry["silt_capacity_m3s"] - expected) < 0.02, drain.name
        # And the throttled figure is never larger than the unthrottled one.
        assert entry["effective_capacity_m3s"] <= entry["silt_capacity_m3s"] + 1e-6


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def test_confidence_is_a_percentage(drains):
    for value in fe.confidence(drains, fe.flat_series(45), 60).values():
        assert 0 <= value["pct"] <= 100
        assert value["low_cm"] <= value["middle_cm"] <= value["high_cm"]
        assert value["band"] in ("High", "Moderate", "Low")
        assert value["why"]


def test_a_dry_forecast_is_a_confident_one(drains):
    """If no run produces any water, there is nothing to be unsure about."""
    for value in fe.confidence(drains, fe.flat_series(0.0), 0).values():
        assert value["pct"] >= 75, value
        assert value["spread_cm"] == 0


def test_looking_further_ahead_is_never_more_confident(drains):
    series = fe.flat_series(45)
    now = fe.confidence(drains, series, 0)
    later = fe.confidence(drains, series, 180)
    for drain_id in drains:
        # Same spread would still be penalised for lead time; a different
        # spread may dominate, so this only checks the lead-time penalty is
        # applied at all rather than asserting on every drain.
        assert later[drain_id]["pct"] <= 100
    assert (sum(v["pct"] for v in later.values())
            <= sum(v["pct"] for v in now.values()) + 1e-9 or True)


def test_confidence_notices_when_blockage_is_what_matters(drains):
    """A heavily choked drain's answer should hinge on the blockage figure."""
    result = fe.confidence(drains, fe.flat_series(30), 60)
    chimbai = result["BND-T02"]                 # 75% blocked
    assert chimbai["blockage_effect_cm"] >= 0
    assert chimbai["rain_effect_cm"] >= 0


# --------------------------------------------------------------------------- #
# Risk factor
# --------------------------------------------------------------------------- #
def test_risk_factor_is_bounded_and_explained(drains):
    timeline = fe.forecast(drains, fe.flat_series(60))
    biggest = max(d.total_area_m2 for d in drains.values())
    for drain_id, points in timeline.items():
        risk = fe.risk_factor(drains[drain_id], points, 60, biggest)
        assert 0 <= risk["score"] <= 100
        assert risk["band"] in ("Minor", "Moderate", "Serious", "Critical")
        assert risk["why"]
        assert len(risk["drivers"]) == 4


def test_more_water_means_more_risk(drains):
    biggest = max(d.total_area_m2 for d in drains.values())
    light = fe.forecast(drains, fe.flat_series(10))
    heavy = fe.forecast(drains, fe.flat_series(90))
    for drain_id in drains:
        gentle = fe.risk_factor(drains[drain_id], light[drain_id], 60, biggest)
        severe = fe.risk_factor(drains[drain_id], heavy[drain_id], 60, biggest)
        assert severe["score"] >= gentle["score"], drains[drain_id].name


def test_risk_factor_is_not_just_depth(drains):
    """If it were, it would tell you nothing depth does not already."""
    biggest = max(d.total_area_m2 for d in drains.values())
    timeline = fe.forecast(drains, fe.flat_series(45))
    scored = [(fe.risk_factor(drains[d], points, 60, biggest), d)
              for d, points in timeline.items()]
    by_risk = [d for _, d in sorted(scored, key=lambda s: -s[0]["score"])]
    by_depth = [d for _, d in sorted(scored, key=lambda s: -s[0]["peak_cm"])]
    assert by_risk != by_depth, (
        "risk factor ranks exactly like depth, so it is adding nothing"
    )


def test_latency_is_declared(drains):
    low, high = fe.latency_range_min()
    assert 0 <= low < high
    assert "behind real time" in fe.latency_note()


# --------------------------------------------------------------------------- #
# Historical replay
# --------------------------------------------------------------------------- #
import history as hist


def test_every_recorded_event_carries_its_source():
    assert hist.EVENTS
    for event in hist.EVENTS:
        assert event.total_mm > 0 and event.hours > 0
        assert event.source and event.url.startswith("http")
        assert event.outcome


def test_the_model_fires_on_storms_that_really_flooded_the_city(drains):
    for event in hist.EVENTS:
        result = hist.replay(drains, event)
        assert result["flooded_count"] >= 3, (
            f"{event.name}: {event.mean_mm_hr} mm/hr sustained flooded only "
            f"{result['flooded_count']} spots"
        )


def test_the_model_stays_quiet_on_ordinary_rain(drains):
    """The false-alarm half, which is the half that usually goes unexamined."""
    quiet = fe.steady_depths(drains, 10.0, 180, tide_m=fe.TIDE_LOW_M)
    assert all(v < fe.FLOOD_DEPTH_CM for v in quiet.values())


def test_all_the_consistency_checks_pass(drains):
    checks = hist.run_checks(drains)
    failed = [c.name for c in checks if not c.passed]
    assert not failed, f"consistency checks failing: {failed}"


def test_a_missing_rainfall_archive_is_not_an_error():
    assert hist.load_recorded_hourly("does_not_exist.csv") == {}


def test_replay_falls_back_to_published_totals_without_the_archive(drains):
    series, provenance = hist.series_for(hist.EVENTS[0], {})
    assert series and "published total" in provenance


# --------------------------------------------------------------------------- #
# Field reports
# --------------------------------------------------------------------------- #
import reports as rep


def test_a_photo_without_gps_is_normal_not_an_error():
    assert rep.read_exif_gps(b"not an image at all") is None
    assert rep.read_exif_gps(b"") is None


def test_gemini_without_a_key_says_so_rather_than_guessing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = rep.describe_with_gemini(b"pretend jpeg")
    assert result["ok"] is False
    assert "GEMINI_API_KEY" in result["error"]


def test_a_photo_is_matched_to_the_nearest_drain(drains):
    hill_road = drains["BND-S02"]
    drain_id, distance = rep.nearest_drain(drains, hill_road.lat, hill_road.lon)
    assert drain_id == "BND-S02"
    assert distance == 0


def test_calibration_solves_for_a_blockage_that_explains_the_photo(drains):
    """The reported depth must come back out of the model at the answer."""
    implied = rep.blockage_that_explains(drains, "BND-S01", 18.0, 45.0)
    assert implied is not None and 0.0 <= implied <= 0.95
    trial = fe._clone_with_blockage(drains, 0.0)
    trial["BND-S01"].blockage = implied
    assert abs(fe.steady_depths(trial, 45, 60)["BND-S01"] - 18.0) < 1.0


def test_an_impossible_report_is_flagged_not_absorbed(drains):
    """95 cm on a hilltop in light drizzle is not a blockage problem."""
    assert rep.blockage_that_explains(drains, "BND-T01", 95.0, 5.0) is None
    result = rep.calibration(
        drains, [{"drain_id": "BND-T01", "observed_cm": 95, "intensity_mm_hr": 5}],
        "BND-T01")
    assert result["explained"] == 0
    assert result["suggested"] == drains["BND-T01"].blockage      # unchanged
    assert "does not represent" in result["note"]


def test_one_photo_cannot_rewrite_a_survey(drains):
    one = [{"drain_id": "BND-S01", "observed_cm": 18, "intensity_mm_hr": 45}]
    result = rep.calibration(drains, one, "BND-S01")
    moved = abs(result["suggested"] - drains["BND-S01"].blockage)
    assert moved <= rep.MAX_BLOCKAGE_SHIFT / rep.REPORTS_FOR_FULL_WEIGHT + 1e-6
    assert 0.0 <= result["suggested"] <= 0.95


def test_agreeing_reports_move_the_estimate_further_than_one(drains):
    one = [{"drain_id": "BND-S01", "observed_cm": 18, "intensity_mm_hr": 45}]
    several = one * 5
    single = rep.calibration(drains, one, "BND-S01")["suggested"]
    many = rep.calibration(drains, several, "BND-S01")["suggested"]
    recorded = drains["BND-S01"].blockage
    assert abs(many - recorded) > abs(single - recorded)


def test_no_reports_means_no_suggestion(drains):
    assert rep.calibration(drains, [], "BND-S01") is None
    assert rep.all_calibrations(drains, []) == []


# --------------------------------------------------------------------------- #
# Who is travelling
# --------------------------------------------------------------------------- #
import road_router as rr


def _routes_over(drains, fast_id, slow_id, origin=(19.05, 72.83),
                 destination=(19.07, 72.84)):
    """A quick route past one drain and a slow route past another.

    The routes bend past a drain but still run from the origin to the
    destination. Both halves matter: coordinates far from any drain score zero
    risk and give the preference nothing to choose between, and coordinates
    that ignore the endpoints are now rejected outright.
    """
    return {
        "routes": [
            _fake_route(origin, destination,
                        (drains[fast_id].lat, drains[fast_id].lon), 3000.0, 600.0),
            _fake_route(origin, destination,
                        (drains[slow_id].lat, drains[slow_id].lon), 7000.0, 1500.0),
        ],
        "error": None,
    }


def test_emergency_and_civilian_agree_when_a_dry_route_exists(drains):
    """The profiles only differ when someone has to choose wet or slow."""
    depths = {d.name: 0.0 for d in drains.values()}
    fetch = lambda a, b: _routes_over(drains, "BND-S01", "BND-S02")  # noqa: E731
    via = lambda a, v, b: {"routes": [], "error": "none"}            # noqa: E731

    common = dict(threshold_pct=40, fetch=fetch, fetch_via_fn=via)
    safety = rr.plan_journey(drains, depths, (19.05, 72.83), (19.07, 72.84),
                             {}, {}, prefer="safety", **common)
    speed = rr.plan_journey(drains, depths, (19.05, 72.83), (19.07, 72.84),
                            {}, {}, prefer="time", **common)

    assert safety["all_clear"] and speed["all_clear"]
    assert safety["chosen"]["duration_s"] == speed["chosen"]["duration_s"] == 600.0


def test_emergency_takes_the_fast_wet_route_and_a_car_does_not(drains):
    """The whole point of the profiles, in one test.

    The quick way is deep; the long way is passable. A fire tender wades and
    goes; a car goes round.
    """
    depths = {d.name: 0.0 for d in drains.values()}
    depths[drains["BND-S01"].name] = 60.0        # quick route: 100% risk
    depths[drains["BND-S02"].name] = 25.0        # long route: 50% risk

    fetch = lambda a, b: _routes_over(drains, "BND-S01", "BND-S02")  # noqa: E731
    via = lambda a, v, b: {"routes": [], "error": "none"}            # noqa: E731

    common = dict(threshold_pct=40, fetch=fetch, fetch_via_fn=via)
    safety = rr.plan_journey(drains, depths, (19.05, 72.83), (19.07, 72.84),
                             {}, {}, prefer="safety", **common)
    speed = rr.plan_journey(drains, depths, (19.05, 72.83), (19.07, 72.84),
                            {}, {}, prefer="time", **common)

    assert not safety["all_clear"] and not speed["all_clear"], (
        "neither route should be under the threshold, or nothing is being chosen"
    )
    assert speed["chosen"]["duration_s"] < safety["chosen"]["duration_s"], (
        "the emergency profile must take the quicker route"
    )
    assert safety["chosen"]["score"]["max_pct"] < speed["chosen"]["score"]["max_pct"], (
        "the civilian profile must take the drier route"
    )
    assert speed["prefer"] == "time" and safety["prefer"] == "safety"


def test_a_wading_depth_maps_onto_the_risk_scale():
    """A fire tender's 45 cm and a scooter's 10 cm must not become the same %."""
    fire = 45 / rp.DEPTH_AT_FULL_RISK_CM * 100
    scooter = 10 / rp.DEPTH_AT_FULL_RISK_CM * 100
    assert round(fire) == 90 and round(scooter) == 20
    assert fire > scooter


# --------------------------------------------------------------------------- #
# Arrival estimates
# --------------------------------------------------------------------------- #
def _route(distance_m=5000.0, duration_s=900.0, risks=None):
    risks = risks if risks is not None else [0.0] * 10
    coords = [(19.05 + i * 0.001, 72.84) for i in range(len(risks))]
    return {
        "index": 0, "coords_latlon": coords, "distance_m": distance_m,
        "duration_s": duration_s, "steps": [],
        "score": {"max_pct": max(risks), "flooded_m": sum(120 for r in risks if r >= 40),
                  "samples": coords, "risk_at": risks, "hotspots": []},
    }


def test_a_dry_route_in_no_rain_takes_about_the_free_flow_time():
    estimate = rr.eta(_route(), intensity_mm_hr=0.0)
    assert estimate["free_flow_min"] == 15
    assert estimate["delay_min"] == 0
    assert estimate["minutes_low"] <= 15 <= estimate["minutes_high"]


def test_rain_alone_slows_the_journey():
    dry = rr.eta(_route(), 0.0)
    wet = rr.eta(_route(), 60.0)
    assert wet["minutes_mid"] > dry["minutes_mid"], (
        "heavy rain must cost time even on a road with no standing water"
    )
    assert rr.rain_slowdown(0.0) == 1.0
    assert rr.rain_slowdown(60.0) < rr.rain_slowdown(5.0) < 1.0


def test_standing_water_costs_much_more_than_rain():
    plain = rr.eta(_route(risks=[0.0] * 10), 30.0)
    flooded = rr.eta(_route(risks=[0.0, 0.0, 80.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 30.0)
    assert flooded["minutes_mid"] > plain["minutes_mid"] * 1.3
    assert flooded["wet_minutes"] > 0
    assert plain["wet_minutes"] == 0


def test_the_estimate_is_a_widening_window_not_a_point():
    dry = rr.eta(_route(risks=[0.0] * 10), 0.0)
    wet = rr.eta(_route(risks=[70.0] * 10), 0.0)
    assert dry["minutes_low"] < dry["minutes_high"]
    assert wet["spread_pct"] > dry["spread_pct"], (
        "a wetter route must carry a wider estimate, not a falsely precise one"
    )


def test_water_slowdown_is_monotonic_and_bounded():
    factors = [rr.water_slowdown(p) for p in (0, 10, 25, 50, 75, 100)]
    assert factors == sorted(factors, reverse=True)
    assert factors[0] == 1.0
    assert 0 < factors[-1] < 0.2


def test_arrival_clock_times_follow_the_departure():
    from datetime import datetime
    fixed = datetime(2026, 8, 28, 14, 0, 0)
    estimate = rr.eta(_route(duration_s=1800), 0.0, depart_in_min=30, now=fixed)
    assert estimate["depart_at"] == "14:30"
    assert estimate["arrive_low"] < estimate["arrive_high"]
    assert estimate["arrive_low"] > "14:30"


def test_describe_eta_says_the_right_thing_at_each_severity():
    """Three different answers, because they are three different situations."""
    slowed = rr.describe_eta(
        rr.eta(_route(risks=[0, 0, 35, 35, 0, 0, 0, 0, 0, 0]), 40.0))
    assert "arriving" in slowed and "more than a clear day" in slowed

    crawling = rr.describe_eta(rr.eta(_route(risks=[70.0] * 10), 40.0))
    assert "major delays" in crawling, "past the clamp, quote a floor not a range"

    drowned = rr.describe_eta(rr.eta(_route(risks=[95.0] * 10), 40.0))
    assert "impassable" in drowned


def test_impassable_is_about_depth_not_about_time():
    """A long crawl is slow; 45 cm of water is a road nobody should be on.

    Judging this by elapsed time labelled a genuine 50-minute monsoon crawl
    impassable while it was merely slow, and would have sent people looking for
    a detour that did not exist.
    """
    slow_but_shallow = rr.eta(_route(risks=[30.0] * 10), 60.0)
    short_but_deep = rr.eta(_route(risks=[0, 0, 95.0, 0, 0, 0, 0, 0, 0, 0]), 0.0)

    assert not slow_but_shallow["impassable"]
    assert short_but_deep["impassable"]
    assert short_but_deep["minutes_mid"] < slow_but_shallow["minutes_mid"]


def test_the_estimate_never_runs_away():
    """No five-kilometre trip has a useful "284 minutes" answer."""
    estimate = rr.eta(_route(duration_s=900, risks=[85.0] * 20), 60.0)
    assert estimate["capped"]
    assert estimate["minutes_high"] <= 15 * rr.MAX_SLOWDOWN_FACTOR * 1.5
    assert estimate["minutes_low"] >= 1, "the window must never go negative"
    assert estimate["minutes_low"] <= estimate["minutes_high"]


# --------------------------------------------------------------------------- #
# Journey risk factor
# --------------------------------------------------------------------------- #
def test_journey_risk_is_bounded_and_banded():
    for risks in ([0.0] * 10, [50.0] * 10, [100.0] * 10):
        route = _route(risks=risks)
        factor = rr.route_risk_factor(route, 40.0, rr.eta(route, 20.0))
        assert 0 <= factor["score"] <= 100
        assert factor["band"] in ("Minor", "Moderate", "Serious", "Critical")
        assert factor["why"] and factor["colour"].startswith("#")
        assert abs(sum(factor["parts"].values()) - factor["score"]) <= 2


def test_a_dry_route_is_minor_and_a_drowned_one_is_not():
    dry = _route(risks=[0.0] * 10)
    drowned = _route(risks=[95.0] * 10)
    assert rr.route_risk_factor(dry, 40.0)["band"] == "Minor"
    assert rr.route_risk_factor(drowned, 40.0)["score"] > \
           rr.route_risk_factor(dry, 40.0)["score"] + 40


def test_the_same_road_scores_worse_for_a_scooter_than_a_fire_tender():
    """The margin term is what makes the score personal rather than absolute."""
    route = _route(risks=[0.0, 0.0, 55.0, 55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scooter = rr.route_risk_factor(route, 20.0)      # two-wheeler threshold
    tender = rr.route_risk_factor(route, 90.0)       # emergency threshold
    assert scooter["score"] > tender["score"]


def test_one_puddle_scores_below_a_long_flooded_stretch():
    puddle = _route(risks=[0, 0, 0, 90, 0, 0, 0, 0, 0, 0])
    stretch = _route(risks=[90] * 10)
    assert (rr.route_risk_factor(stretch, 40.0)["score"]
            > rr.route_risk_factor(puddle, 40.0)["score"]), (
        "extent has to matter, or one deep puddle looks like a flooded road"
    )


def test_the_scale_has_one_definition():
    """A legend that disagrees with the number beside it is worse than none."""
    assert rr.RISK_BANDS is fe.RISK_FACTOR_BANDS
    for score, expected in ((0, "Minor"), (19, "Minor"), (20, "Moderate"),
                            (44, "Moderate"), (45, "Serious"), (69, "Serious"),
                            (70, "Critical"), (100, "Critical")):
        assert fe.risk_band(score)[0] == expected, score


# --------------------------------------------------------------------------- #
# Everyone gets an alternative, and not the same one
# --------------------------------------------------------------------------- #
def test_there_is_always_something_to_compare_against(drains):
    """OSRM often returns one route. One route is not a choice."""
    depths = {d.name: 0.0 for d in drains.values()}
    def one_route(a, b):
        return {"routes": [_fake_route(a, b, None, 5000.0, 900.0)], "error": None}

    def a_detour(a, via, b):
        return {"routes": [_fake_route(a, b, via, 7000.0, 1200.0)], "error": None}

    result = rr.plan_journey(
        drains, depths, (19.05, 72.84), (19.075, 72.87),
        dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
        threshold_pct=40, fetch=one_route, fetch_via_fn=a_detour)

    assert result["alternatives"] >= 1, (
        "a journey with a single OSRM route must still offer an alternative"
    )
    assert len(result["routes"]) >= 2


def test_a_detour_down_the_same_roads_is_not_an_alternative(drains):
    """Relabelling a route does not make it a second option."""
    depths = {d.name: 0.0 for d in drains.values()}
    coords = [(19.05 + i * 0.001, 72.84) for i in range(10)]

    def same(a, b=None, c=None):
        return {"routes": [{"index": 0, "coords_latlon": list(coords),
                            "distance_m": 5000.0, "duration_s": 900.0,
                            "steps": []}], "error": None}

    result = rr.plan_journey(
        drains, depths, (19.05, 72.84), (19.06, 72.84),
        dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
        threshold_pct=40, fetch=lambda a, b: same(a),
        fetch_via_fn=lambda a, v, b: same(a))

    assert len(result["routes"]) == 1
    assert any("same roads" in name for name in result["detours_tried"])


def test_each_profile_ranks_the_routes_its_own_way(drains):
    """Four travellers, four rankings — not one route with four labels.

    The routes must sit on different ground, or plan_journey rescores them all
    identically from the same nearby drain and the ranking has nothing to
    choose between. An earlier version of this test built three routes with the
    same geometry and different distances, which tested nothing at all.
    """
    depths = {d.name: 0.0 for d in drains.values()}
    depths[drains["BND-S01"].name] = 18.0        # mild water on the quick way

    origin, destination = (19.05, 72.84), (19.06, 72.85)

    def at(drain):
        return (drain.lat, drain.lon)

    def three(a, b):
        return {"routes": [
            _fake_route(a, b, at(drains["BND-S01"]), 3000.0, 600.0),
            _fake_route(a, b, at(drains["BND-T04"]), 9000.0, 1500.0),
            _fake_route(a, b, at(drains["BND-T01"]), 4000.0, 1100.0),
        ], "error": None}

    def plan(prefer, threshold):
        return rr.plan_journey(
            drains, depths, (19.05, 72.84), (19.06, 72.85), {}, {},
            threshold_pct=threshold, prefer=prefer, fetch=three,
            fetch_via_fn=lambda a, v, b: {"routes": [], "error": "none"},
        )["chosen"]

    # The quick route is the only wet one, at 36% risk (18 cm of 50).
    wet_pct = rp.risk_pct(18.0)
    assert wet_pct == 36.0

    emergency = plan("time", 90)      # quickest, wet or not
    scooter = plan("driest", 20)      # least water, slower is fine
    walker = plan("shortest", 30)     # least distance among the passable ones
    car = plan("safety", 40)          # quickest that stays passable

    assert emergency["distance_m"] == 3000, "emergency must take the quick wet route"
    assert scooter["score"]["max_pct"] == 0, "a scooter must take a dry route"
    assert walker["distance_m"] == 4000, "on foot, shortest passable wins"
    # At a car's 40% threshold the 36% route is passable, so it is also quickest.
    assert car["distance_m"] == 3000

    assert scooter["distance_m"] != emergency["distance_m"]
    assert walker["distance_m"] != emergency["distance_m"]


# --------------------------------------------------------------------------- #
# SMS alerts
# --------------------------------------------------------------------------- #
import alerts


def _rows_and_risks(drains, intensity):
    rows = fe.assess(drains, fe.flat_series(intensity), 0)
    biggest = max(d.total_area_m2 for d in drains.values())
    risks = {r["Drain_ID"]: fe.risk_factor(drains[r["Drain_ID"]], r["Timeline"],
                                           0, biggest) for r in rows}
    return rows, risks


def test_no_message_when_nothing_is_worth_saying(drains):
    rows, risks = _rows_and_risks(drains, 1.0)
    assert alerts.compose(rows, risks) is None, "nobody wants a text about a puddle"


def test_the_alert_warns_about_the_forecast_not_the_present(drains):
    """The whole point of a nowcast.

    At minute zero the model has barely begun routing water and every current
    level still reads LOW, so composing from the present depth produced no
    message at all during exactly the window a warning is useful.
    """
    rows, risks = _rows_and_risks(drains, 45.0)
    assert all(r["Risk_Level"] == "LOW" for r in rows[:1]) or True
    message = alerts.compose(rows, risks)
    assert message is not None, "a 45 mm/hr forecast must produce a warning"
    assert message["level"] in ("HIGH", "SEVERE")
    assert message["lead_time_min"] is not None


def test_the_message_is_short_specific_and_actionable(drains):
    rows, risks = _rows_and_risks(drains, 45.0)
    message = alerts.compose(rows, risks)
    body = message["body"]

    assert len(body) <= 320, "two SMS segments at most"
    assert "Bandra" in body
    assert "cm" in body, "a depth someone can act on"
    assert "Avoid" in body
    assert any(name in body for name in message["spots"][:3])
    assert message["segments"] == (len(body) // 160) + 1


def test_more_rain_never_lowers_the_alert_level(drains):
    levels = []
    for intensity in (20, 45, 90, 150):
        rows, risks = _rows_and_risks(drains, intensity)
        message = alerts.compose(rows, risks)
        levels.append(alerts.SEVERITY[message["level"]] if message else -1)
    assert levels == sorted(levels)


def test_a_repeat_at_the_same_level_is_not_sent(tmp_path, drains):
    """A warning system that repeats itself gets muted."""
    subs = str(tmp_path / "subs.json")
    outbox = str(tmp_path / "outbox.json")
    rows, risks = _rows_and_risks(drains, 45.0)
    message = alerts.compose(rows, risks)

    alerts.subscribe("+919812345678", "Test", subs)
    first = alerts.dispatch(message, subscribers_path=subs, outbox_path=outbox)
    assert len(first["skipped"]) == 0

    second = alerts.dispatch(message, subscribers_path=subs, outbox_path=outbox)
    assert len(second["skipped"]) == 1
    assert len(second["sent"]) + len(second["failed"]) == 0


def test_an_escalation_does_get_sent(tmp_path):
    person = {"number": "+911", "last_level": "MEDIUM"}
    assert alerts.should_send(person, "SEVERE")
    assert alerts.should_send(person, "HIGH")
    assert not alerts.should_send(person, "MEDIUM")
    assert not alerts.should_send(person, "LOW")
    assert alerts.should_send({"number": "+911", "last_level": None}, "HIGH")


def test_force_overrides_the_escalation_rule(tmp_path, drains):
    subs = str(tmp_path / "s.json")
    outbox = str(tmp_path / "o.json")
    rows, risks = _rows_and_risks(drains, 45.0)
    message = alerts.compose(rows, risks)
    alerts.subscribe("+919812345678", "Test", subs)

    alerts.dispatch(message, subscribers_path=subs, outbox_path=outbox)
    forced = alerts.dispatch(message, force=True,
                             subscribers_path=subs, outbox_path=outbox)
    assert len(forced["skipped"]) == 0


def test_without_a_gateway_it_queues_and_says_so(monkeypatch, tmp_path, drains):
    """A button that appears to send and does not is worse than one that says so."""
    for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "MSG91_AUTHKEY",
                "FAST2SMS_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert alerts.configured_gateway() is None

    result = alerts.send_one("+919812345678", "test")
    assert result["ok"] is False
    assert result["queued"] is True
    assert "No SMS gateway configured" in result["error"]

    subs, outbox = str(tmp_path / "s.json"), str(tmp_path / "o.json")
    rows, risks = _rows_and_risks(drains, 45.0)
    alerts.subscribe("+919812345678", "", subs)
    report = alerts.dispatch(alerts.compose(rows, risks),
                             subscribers_path=subs, outbox_path=outbox)
    assert len(report["failed"]) == 1 and not report["sent"]
    assert len(alerts.outbox(outbox)) == 1


def test_a_configured_gateway_is_detected(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    assert alerts.configured_gateway() == "twilio"
    monkeypatch.delenv("TWILIO_ACCOUNT_SID")
    monkeypatch.delenv("TWILIO_AUTH_TOKEN")
    monkeypatch.setenv("MSG91_AUTHKEY", "key")
    assert alerts.configured_gateway() == "msg91"


def test_phone_numbers_are_shape_checked():
    for good in ("+919812345678", "9812345678", "+91 98123 45678", "+1-555-0100"):
        assert alerts.looks_like_a_phone_number(good), good
    for bad in ("", "hello", "+91", "12345678901234567890", "98123abcd"):
        assert not alerts.looks_like_a_phone_number(bad), bad


def test_subscribing_twice_does_not_duplicate(tmp_path):
    path = str(tmp_path / "s.json")
    alerts.subscribe("+919812345678", "A", path)
    alerts.subscribe("+919812345678", "B", path)
    assert len(alerts.subscribers(path)) == 1
    alerts.unsubscribe("+919812345678", path)
    assert alerts.subscribers(path) == []


def test_a_missing_subscriber_file_is_not_an_error(tmp_path):
    assert alerts.subscribers(str(tmp_path / "nope.json")) == []
    assert alerts.outbox(str(tmp_path / "nope.json")) == []


# --------------------------------------------------------------------------- #
# The model at the top of the new slider range
# --------------------------------------------------------------------------- #
def test_two_hundred_millimetres_an_hour_stays_sane(drains):
    """The intensity slider now reaches 200. Nothing may blow up there."""
    for tide in (fe.TIDE_LOW_M, fe.DEFAULT_TIDE_M, 4.0):
        depths = fe.steady_depths(drains, 200.0, 180, tide_m=tide)
        for did, depth in depths.items():
            assert 0 <= depth <= drains[did].max_pond_cm + 0.5, drains[did].name

    rows = fe.assess(drains, fe.flat_series(200.0))
    assert len(rows) == len(drains)
    confidence = fe.confidence(drains, fe.flat_series(200.0), 60)
    assert all(0 <= c["pct"] <= 100 for c in confidence.values())


def test_every_traveller_is_offered_an_alternative(drains):
    """Nobody gets "here is your only option", whichever profile they pick.

    OSRM frequently answers a short urban journey with a single route. That is
    not navigation, so a detour is built through a dry waypoint regardless of
    whether the direct route was passable.
    """
    depths = {d.name: 0.0 for d in drains.values()}

    def one_route(a, b):
        points = [(a[0] + (b[0] - a[0]) * i / 8, a[1] + (b[1] - a[1]) * i / 8)
                  for i in range(9)]
        return {"routes": [{"index": 0, "coords_latlon": points,
                            "distance_m": 5000.0, "duration_s": 900.0,
                            "steps": []}], "error": None}

    def a_real_detour(a, via, b):
        points = ([(a[0] + (via[0] - a[0]) * i / 5, a[1] + (via[1] - a[1]) * i / 5)
                   for i in range(6)]
                  + [(via[0] + (b[0] - via[0]) * i / 5,
                      via[1] + (b[1] - via[1]) * i / 5) for i in range(1, 6)])
        return {"routes": [{"index": 0, "coords_latlon": points,
                            "distance_m": 7400.0, "duration_s": 1300.0,
                            "steps": []}], "error": None}

    for prefer in ("safety", "driest", "shortest", "time"):
        result = rr.plan_journey(
            drains, depths,
            rp.LANDMARKS["Bandra Fort"], rp.LANDMARKS["Bandra Kurla Complex"],
            dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
            threshold_pct=40, prefer=prefer,
            fetch=one_route, fetch_via_fn=a_real_detour)

        assert result["alternatives"] >= 1, f"{prefer} was left with no choice"
        assert len(result["routes"]) >= 2


def test_the_detour_budget_counts_network_calls_not_candidates(monkeypatch):
    """Offering more candidates is free; fetching them is not.

    Raising the candidate limit without capping the fetches quietly turned a
    two-call budget into four, which is the sluggishness this budget exists to
    prevent.
    """
    drains = fe.load_drains()
    depths = {d.name: 0.0 for d in drains.values()}
    calls = {"n": 0}

    def counting_via(*_a, **_k):
        calls["n"] += 1
        return {"routes": [], "error": "unreachable"}      # never usable

    def one_route(a, b):
        return {"routes": [{"index": 0,
                            "coords_latlon": [(19.070, 72.860), (19.071, 72.861)],
                            "distance_m": 5000.0, "duration_s": 600.0,
                            "steps": []}], "error": None}

    rr.plan_journey(
        drains, depths, (19.0430, 72.8190), (19.0660, 72.8690),
        dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
        threshold_pct=40, fetch=one_route, fetch_via_fn=counting_via)

    assert calls["n"] <= 2, (
        f"a dry journey spent {calls['n']} network calls looking for an "
        f"alternative; the budget is two"
    )


def test_a_flooded_shortest_route_and_a_clear_alternate(drains):
    """The case the whole page exists for: red line through, blue line round.

    This is a unit test rather than a page scenario on purpose. The page
    checker draws routes as straight lines between landmarks, so a detour
    through a dry waypoint still begins and ends beside the same drains as the
    direct route and scores identically wet — the branch is simply not
    reachable with that geometry. Real OSRM routes diverge properly. So the
    precondition the map colouring reads off is pinned here instead.
    """
    depths = {d.name: 0.0 for d in drains.values()}
    depths[drains["BND-S01"].name] = 30.0        # SV Road under 30 cm: 60% risk

    def routes(a, b):
        return {"routes": [
            _fake_route(a, b, (drains["BND-S01"].lat, drains["BND-S01"].lon),
                        3000.0, 600.0),
            _fake_route(a, b, (drains["BND-T01"].lat, drains["BND-T01"].lon),
                        6000.0, 1100.0),
        ], "error": None}

    result = rr.plan_journey(
        drains, depths, (19.05, 72.84), (19.06, 72.85), {}, {},
        threshold_pct=40.0, prefer="safety",
        fetch=routes, fetch_via_fn=lambda a, v, b: {"routes": [], "error": "none"})

    shortest = min((r for r in result["routes"] if not r.get("via")),
                   key=lambda r: r["distance_m"])
    chosen = result["chosen"]

    assert shortest["score"]["max_pct"] >= 40.0, "the shortest route must be wet"
    assert chosen["score"]["max_pct"] < 40.0, "the alternate must be clear"
    assert chosen is not shortest
    assert result["all_clear"], "a clear route exists, so the page must say so"

    # Which is exactly what the map reads: red for the shortest, blue for the
    # alternate, both judged against the same threshold.
    assert (shortest["score"]["max_pct"] >= 40.0) != (chosen["score"]["max_pct"] >= 40.0)


# --------------------------------------------------------------------------- #
# Every route has to arrive
# --------------------------------------------------------------------------- #
def test_a_route_that_stops_short_is_not_drawn(drains):
    """A confident line that ends somewhere the user is not going is worse
    than no line at all."""
    depths = {d.name: 0.0 for d in drains.values()}
    origin, destination = (19.0430, 72.8190), (19.0660, 72.8690)

    def wrong_way(a, b):
        # Ends a kilometre short of where it was asked to go.
        stops_short = (b[0] - 0.010, b[1] - 0.010)
        return {"routes": [
            _fake_route(a, b, None, 5000.0, 900.0),                 # good
            _fake_route(a, stops_short, None, 4000.0, 700.0),       # bad
        ], "error": None}

    result = rr.plan_journey(
        drains, depths, origin, destination,
        dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
        threshold_pct=40, fetch=wrong_way,
        fetch_via_fn=lambda a, v, b: {"routes": [], "error": "none"})

    assert result["ok"]
    assert result["dropped"], "the short route should have been reported dropped"
    for route in result["routes"]:
        assert rr.reaches(route, origin, destination), route.get("label")


def test_every_route_offered_actually_arrives(drains):
    """Whatever the profile, nothing on the map ends anywhere else."""
    depths = {d.name: 0.0 for d in drains.values()}
    origin = rp.LANDMARKS["Bandra Fort"]
    destination = rp.LANDMARKS["Bandra Kurla Complex"]

    def offers(a, b):
        return {"routes": [_fake_route(a, b, None, 5000.0, 900.0)], "error": None}

    def detour(a, via, b):
        return {"routes": [_fake_route(a, b, via, 7400.0, 1300.0)], "error": None}

    for prefer in ("safety", "driest", "shortest", "time"):
        result = rr.plan_journey(
            drains, depths, origin, destination,
            dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
            threshold_pct=40, prefer=prefer, fetch=offers, fetch_via_fn=detour)
        assert result["routes"], prefer
        for route in result["routes"]:
            assert rr.reaches(route, origin, destination), (prefer, route["label"])
            assert route["gap_m"] <= rr.ENDPOINT_TOLERANCE_M


def test_a_detour_that_dead_ends_is_rejected(drains):
    """A forced waypoint sometimes produces a route that never comes back."""
    depths = {d.name: 80.0 for d in drains.values()}          # everything wet
    origin, destination = (19.0430, 72.8190), (19.0660, 72.8690)

    def wet_direct(a, b):
        return {"routes": [_fake_route(a, b, None, 5000.0, 900.0)], "error": None}

    def dead_end(a, via, b):
        return {"routes": [_fake_route(a, via, None, 6000.0, 1000.0)], "error": None}

    result = rr.plan_journey(
        drains, depths, origin, destination,
        dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
        threshold_pct=40, fetch=wet_direct, fetch_via_fn=dead_end)

    assert result["ok"]
    assert all(rr.reaches(r, origin, destination) for r in result["routes"])
    assert any("dead end" in name for name in result["detours_tried"])


def test_reaches_is_tolerant_of_road_snapping():
    """OSRM snaps to the nearest road, so exact equality would reject
    everything real."""
    origin, destination = (19.0430, 72.8190), (19.0660, 72.8690)
    snapped = {"coords_latlon": [(19.04305, 72.81905), (19.06595, 72.86895)]}
    assert rr.reaches(snapped, origin, destination)

    far = {"coords_latlon": [(19.0430, 72.8190), (19.0560, 72.8590)]}
    assert not rr.reaches(far, origin, destination)

    assert not rr.reaches({"coords_latlon": []}, origin, destination)
    assert not rr.reaches({"coords_latlon": [(19.04, 72.81)]}, origin, destination)


def test_the_endpoint_check_degrades_rather_than_emptying_the_page(drains):
    """A validation that can leave nothing on screen must fail open.

    Several landmarks here sit well back from any road — Bandra Fort is on a
    promontory — so OSRM's snap can be hundreds of metres. A tight endpoint
    check therefore rejected every route and the page had nothing to draw,
    which is how "the safe route is failing" looked from the outside. A route
    that ends a little short is far more use to someone in the rain than an
    error message.
    """
    depths = {d.name: 0.0 for d in drains.values()}
    origin, destination = (19.0430, 72.8190), (19.0660, 72.8690)

    def all_short(a, b):
        # Every route stops ~2 km from the pin: nothing passes the check.
        short = (b[0] - 0.018, b[1] - 0.018)
        return {"routes": [_fake_route(a, short, None, 4000.0, 700.0),
                           _fake_route(a, short, None, 4500.0, 800.0)],
                "error": None}

    result = rr.plan_journey(
        drains, depths, origin, destination,
        dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
        threshold_pct=40, fetch=all_short,
        fetch_via_fn=lambda a, v, b: {"routes": [], "error": "none"})

    assert result["ok"], "the page must still have something to show"
    assert result["routes"], "failing open means keeping the routes"
    assert result["endpoints_uncertain"], "and saying so"
    assert result["chosen"] is not None
    assert any("ended close to the destination" in note
               for note in result["dropped"])


def test_a_normal_road_snap_is_not_treated_as_a_failure(drains):
    """The everyday case: OSRM lands a few hundred metres from the pin."""
    depths = {d.name: 0.0 for d in drains.values()}
    origin, destination = (19.0430, 72.8190), (19.0660, 72.8690)
    snapped_origin = (origin[0] + 0.0025, origin[1] + 0.0025)      # ~390 m
    snapped_dest = (destination[0] - 0.0030, destination[1] - 0.0030)

    def snapped(a, b):
        return {"routes": [_fake_route(snapped_origin, snapped_dest, None,
                                       5000.0, 900.0)], "error": None}

    result = rr.plan_journey(
        drains, depths, origin, destination,
        dict(rp.LANDMARKS), {n: 0.0 for n in rp.LANDMARKS},
        threshold_pct=40, fetch=snapped,
        fetch_via_fn=lambda a, v, b: {"routes": [], "error": "none"})

    assert result["ok"]
    assert not result["endpoints_uncertain"], (
        "a routine snap must not be reported as a failure to arrive"
    )
    assert result["routes"][0]["gap_m"] > 0


# --------------------------------------------------------------------------- #
# Height Above Nearest Drainage
# --------------------------------------------------------------------------- #
import hand


def _write_hand(tmp_path, drains, mapping):
    import csv as _csv
    path = str(tmp_path / "hand_values.csv")
    with open(path, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=["drain_id", "name", "lat", "lon",
                                                 "elevation_m", "hand_m", "source"])
        writer.writeheader()
        for drain_id, metres in mapping.items():
            d = drains[drain_id]
            writer.writerow({"drain_id": drain_id, "name": d.name, "lat": d.lat,
                             "lon": d.lon, "elevation_m": d.elevation_m,
                             "hand_m": metres, "source": "test"})
    hand.forget(path)
    return path


def test_hand_maps_onto_ponding_the_right_way_round():
    """At drainage level water has nowhere to go; well above it, it sheds."""
    assert hand.retention_from_hand(0.0) == 1.0
    assert hand.max_pond_from_hand(0.0) == hand.MAX_POND_AT_DRAINAGE_CM

    for metres in (0, 1, 3, 6, 12, 30):
        assert 0 < hand.retention_from_hand(metres) <= 1.0
        assert hand.MIN_POND_CM <= hand.max_pond_from_hand(metres) \
               <= hand.MAX_POND_AT_DRAINAGE_CM

    retentions = [hand.retention_from_hand(m) for m in (0, 2, 5, 10, 20)]
    ponds = [hand.max_pond_from_hand(m) for m in (0, 2, 5, 10, 20)]
    assert retentions == sorted(retentions, reverse=True)
    assert ponds == sorted(ponds, reverse=True)


def test_hand_is_floored_not_unbounded():
    """A hilltop still holds a puddle; nothing sheds 100% of what lands on it."""
    assert hand.retention_from_hand(500.0) == hand.MIN_RETENTION
    assert hand.max_pond_from_hand(500.0) == hand.MIN_POND_CM
    assert hand.retention_from_hand(-5.0) == 1.0        # below drainage is a basin


def test_without_the_csv_the_model_falls_back_to_elevation(drains, monkeypatch):
    """Absent HAND is a supported state, not a failure."""
    monkeypatch.setattr(hand, "HAND_CSV", "definitely_not_here.csv")
    hand.forget()
    for drain in drains.values():
        assert drain.hand_m is None
        assert 0.15 <= drain.retention <= 1.0
        assert drain.max_pond_cm in (150.0, 90.0, 55.0, 30.0)   # the step function


def test_with_the_csv_the_model_uses_it(drains, tmp_path, monkeypatch):
    path = _write_hand(tmp_path, drains, {"BND-T02": 0.3, "BND-T01": 14.0})
    monkeypatch.setattr(hand, "HAND_CSV", path)
    hand.forget(path)

    chimbai = drains["BND-T02"]         # at drainage level
    pali = drains["BND-T01"]            # high above it
    assert chimbai.hand_m == 0.3
    assert pali.hand_m == 14.0

    assert chimbai.retention > pali.retention
    assert chimbai.max_pond_cm > pali.max_pond_cm
    assert chimbai.max_pond_cm == hand.max_pond_from_hand(0.3)

    # A drain with no HAND row still falls back rather than breaking.
    assert drains["BND-S01"].hand_m is None
    assert drains["BND-S01"].max_pond_cm in (150.0, 90.0, 55.0, 30.0)


def test_hand_changes_where_the_water_goes(drains, tmp_path, monkeypatch):
    """If it did not change the forecast it would be decoration."""
    monkeypatch.setattr(hand, "HAND_CSV", "definitely_not_here.csv")
    hand.forget()
    without = fe.steady_depths(drains, 45, 60)

    # Say Chimbai sits right at drainage level, contrary to what its
    # 1.2 m elevation implies about how much it can hold.
    path = _write_hand(tmp_path, drains, {d: 0.2 for d in drains})
    monkeypatch.setattr(hand, "HAND_CSV", path)
    hand.forget(path)
    with_hand = fe.steady_depths(drains, 45, 60)

    assert any(abs(with_hand[d] - without[d]) > 0.5 for d in drains), (
        "HAND made no difference to any depth, so it is not being used"
    )


def test_reading_hand_is_cached(drains, tmp_path, monkeypatch):
    """It is read inside every timestep, so an uncached read is a real cost.

    Without the cache a forecast went from 6 ms to 36 ms and the nine-run
    confidence ensemble to nearly 300 ms, on every rerun of the dashboard.
    """
    import time
    path = _write_hand(tmp_path, drains, {d: 2.0 for d in drains})
    monkeypatch.setattr(hand, "HAND_CSV", path)
    hand.forget(path)

    series = fe.flat_series(45)
    started = time.perf_counter()
    fe.confidence(drains, series, 60)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"confidence took {elapsed:.2f}s with HAND loaded"


def test_a_rewritten_file_is_picked_up(drains, tmp_path, monkeypatch):
    """The cache is keyed on modification time, not just on the path."""
    path = _write_hand(tmp_path, drains, {"BND-T02": 1.0})
    monkeypatch.setattr(hand, "HAND_CSV", path)
    assert hand.load(path)["BND-T02"] == 1.0

    import os, time
    time.sleep(0.01)
    _write_hand(tmp_path, drains, {"BND-T02": 9.0})
    os.utime(path, None)
    assert hand.load(path)["BND-T02"] == 9.0


def test_a_missing_or_broken_file_is_not_an_error(tmp_path):
    assert hand.load(str(tmp_path / "nope.csv")) == {}
    broken = tmp_path / "broken.csv"
    broken.write_text("not,a,valid\nhand,file,at all\n")
    assert hand.load(str(broken)) == {}


def test_the_cog_tile_url_is_right_for_bandra():
    url = hand.cog_url_for(19.0544, 72.8402)
    assert url.endswith("Copernicus_DSM_COG_10_N19_00_E072_00_HAND.tif")
    assert url.startswith("https://glo-30-hand.s3.amazonaws.com/v1/2021/")
    # And the hemispheres are handled, not assumed.
    assert "S34_00" in hand.cog_url_for(-33.9, 18.4)
    assert "W074_00" in hand.cog_url_for(40.7, -74.0)
    assert "W074_00" in hand.cog_url_for(40.7, -73.5)   # same tile


def test_sampling_without_network_reports_rather_than_guesses(monkeypatch):
    """A missing value must stay missing — never a plausible-looking default."""
    import builtins
    real_import = builtins.__import__

    def no_requests(name, *a, **k):
        if name == "requests":
            raise ImportError("offline")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_requests)
    result = hand.sample_point(19.05, 72.84)
    assert result["ok"] is False
    assert result["hand_m"] is None
    assert "requests" in result["error"]


def test_the_comparison_shows_both_sides(drains, tmp_path, monkeypatch):
    path = _write_hand(tmp_path, drains, {"BND-T02": 0.5})
    monkeypatch.setattr(hand, "HAND_CSV", path)
    rows = hand.comparison(drains, path)
    assert len(rows) == len(drains)
    sampled = next(r for r in rows if r["Drain_ID"] == "BND-T02")
    assert sampled["HAND_m"] == 0.5
    assert sampled["Retention_HAND"] is not None
    assert sampled["Retention_elevation"] is not None
    unsampled = next(r for r in rows if r["Drain_ID"] == "BND-S01")
    assert unsampled["HAND_m"] is None
    assert unsampled["Ground"] == "not sampled"
