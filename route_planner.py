"""Safe routing across Bandra, avoiding whatever is about to go under.

The flood engine says how deep each place will be. This turns that into a
decision: which way to send someone, and when to tell them not to go at all.

How the network is built
------------------------
The CSV gives ten drain locations. Those alone are not a road network, so a
handful of landmarks people actually name are added, and every node is joined
to its three nearest neighbours. That produces a connected graph over Bandra
which is enough to reason about detours. It is a simplification — the edges
are straight lines, not real streets — and a deployment would swap in OSM
road geometry. The routing logic does not change when that happens.

Risk as a percentage
--------------------
The engine works in centimetres, but people think in percentages, so depth is
mapped onto a 0-100 scale where 50 cm is 100%. That puts the 40% trigger at
20 cm, which is about where a small car starts to struggle — a sensible place
to send someone a different way.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from flood_engine import Drain, haversine_m

# 50 cm of water counts as 100% risk. 40% is therefore 20 cm, roughly the
# depth at which a hatchback stalls.
DEPTH_AT_FULL_RISK_CM = 50.0
DEFAULT_AVOID_ABOVE_PCT = 40.0

# Places people give as an origin or destination, so the picker reads like a
# map rather than a list of manholes.
LANDMARKS: Dict[str, Tuple[float, float]] = {
    "Bandra Station": (19.0544, 72.8406),
    "Bandra Reclamation": (19.0470, 72.8200),
    "Carter Road": (19.0640, 72.8200),
    "Bandra Kurla Complex": (19.0660, 72.8690),
    "Bandra Fort": (19.0430, 72.8190),
    "Kalanagar Junction": (19.0570, 72.8500),
    "Khar Subway": (19.0700, 72.8340),
    "Mahim Causeway": (19.0400, 72.8420),
    "Pali Naka": (19.0625, 72.8265),
    "Turner Road Junction": (19.0605, 72.8318),
}

# Four, not three. At three the graph develops chokepoints that no real road
# network has - the whole of west Bandra hung off a single junction, so one
# flooded drain made half the area unreachable. Four restores the redundancy
# that actual streets have.
NEIGHBOURS_PER_NODE = 4


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def risk_pct(depth_cm: float) -> float:
    """Depth in centimetres as a 0-100 risk percentage."""
    return round(min(depth_cm / DEPTH_AT_FULL_RISK_CM * 100.0, 100.0), 1)


def advice_for(pct: float) -> str:
    if pct >= 90:
        return "Impassable"
    if pct >= 60:
        return "Do not drive through"
    if pct >= 40:
        return "Cars will struggle"
    if pct >= 20:
        return "Two-wheelers should avoid"
    return "Passable"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class Network:
    def __init__(self, drains: Dict[str, Drain]):
        self.points: Dict[str, Tuple[float, float]] = {
            d.name: (d.lat, d.lon) for d in drains.values()
        }
        self.points.update(LANDMARKS)
        self.drain_names = {d.name for d in drains.values()}
        self.edges: Dict[str, List[str]] = {n: [] for n in self.points}
        self._connect()

    def _connect(self) -> None:
        for node, here in self.points.items():
            ranked = sorted(
                (other for other in self.points if other != node),
                key=lambda o: haversine_m(here, self.points[o]),
            )
            for other in ranked[:NEIGHBOURS_PER_NODE]:
                if other not in self.edges[node]:
                    self.edges[node].append(other)
                if node not in self.edges[other]:      # keep it two-way
                    self.edges[other].append(node)

    def distance(self, a: str, b: str) -> float:
        return haversine_m(self.points[a], self.points[b])

    def nearest(self, lat: float, lon: float) -> str:
        return min(self.points, key=lambda n: haversine_m((lat, lon), self.points[n]))


# --------------------------------------------------------------------------- #
# Directions
# --------------------------------------------------------------------------- #
COMPASS = ["north", "north-east", "east", "south-east",
           "south", "south-west", "west", "north-west"]


def bearing(a: Tuple[float, float], b: Tuple[float, float]) -> str:
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    degrees = (math.degrees(math.atan2(y, x)) + 360) % 360
    return COMPASS[int((degrees + 22.5) // 45) % 8]


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def _search(net: Network, start: str, goal: str, cost_of) -> Optional[List[str]]:
    """A* over the node graph. cost_of(a, b) returns metres, or inf to block."""
    queue = [(0.0, start)]
    came: Dict[str, str] = {}
    best = {start: 0.0}

    while queue:
        _, node = heapq.heappop(queue)
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came[path[-1]])
            return list(reversed(path))

        for neighbour in net.edges[node]:
            step = cost_of(node, neighbour)
            if step == float("inf"):
                continue
            tentative = best[node] + step
            if tentative < best.get(neighbour, float("inf")):
                best[neighbour] = tentative
                came[neighbour] = node
                heapq.heappush(queue, (tentative + net.distance(neighbour, goal), neighbour))
    return None


def plan(
    drains: Dict[str, Drain],
    depths: Dict[str, float],
    origin: str,
    destination: str,
    avoid_above_pct: float = DEFAULT_AVOID_ABOVE_PCT,
) -> dict:
    """Work out the safest way from origin to destination.

    `depths` maps drain NAME to predicted depth in cm. Landmarks carry no
    depth of their own; they take the risk of the nearest drain, because a
    junction fifty metres from a flooded gully is not dry.
    """
    net = Network(drains)

    node_risk: Dict[str, float] = {}
    for node, coords in net.points.items():
        if node in depths:
            node_risk[node] = risk_pct(depths[node])
        else:
            nearest_drain = min(
                net.drain_names,
                key=lambda d: haversine_m(coords, net.points[d]),
            )
            distance = haversine_m(coords, net.points[nearest_drain])
            # A landmark inherits the nearby drain's risk, fading with distance.
            fade = max(0.0, 1.0 - distance / 600.0)
            node_risk[node] = round(risk_pct(depths.get(nearest_drain, 0.0)) * fade, 1)

    def plain_cost(a: str, b: str) -> float:
        return net.distance(a, b)

    def safe_cost(a: str, b: str) -> float:
        # Never block where someone is or where they are going. If the
        # destination is under water they still need to know how to approach
        # it and how bad the last stretch is; refusing to draw a line is not
        # help. Only the route through the middle gets vetoed.
        movable = [n for n in (a, b) if n not in (origin, destination)]
        worst = max((node_risk[n] for n in movable), default=0.0)
        if worst >= avoid_above_pct:
            return float("inf")
        blended = max(node_risk[a], node_risk[b])
        # A road at 30% is not blocked, but it is worth going around.
        return net.distance(a, b) * (1.0 + (blended / 100.0) * 6.0)

    shortest = _search(net, origin, destination, plain_cost)
    shortest_m = (
        sum(net.distance(a, b) for a, b in zip(shortest, shortest[1:])) if shortest else 0.0
    )

    safe = _search(net, origin, destination, safe_cost)

    if safe is None:
        blocked = sorted(
            ((v, k) for k, v in node_risk.items() if v >= avoid_above_pct), reverse=True
        )
        return {
            "found": False,
            "message": (
                "Every route between these points passes somewhere above "
                f"{avoid_above_pct:.0f}% risk. Wait, or contact the control room."
            ),
            "blocking": [name for _, name in blocked[:5]],
            "shortest_path": shortest or [],
            "shortest_m": round(shortest_m),
            "path": [], "legs": [], "total_m": 0, "worst_pct": 100.0,
            "detour_m": 0, "node_risk": node_risk,
        }

    legs = []
    total = 0.0
    worst = 0.0
    for a, b in zip(safe, safe[1:]):
        metres = net.distance(a, b)
        total += metres
        leg_risk = max(node_risk[a], node_risk[b])
        worst = max(worst, leg_risk)
        legs.append({
            "from": a,
            "to": b,
            "metres": round(metres),
            "heading": bearing(net.points[a], net.points[b]),
            "risk_pct": leg_risk,
            "advice": advice_for(leg_risk),
        })

    detour = total - shortest_m
    ends_wet = max(node_risk[origin], node_risk[destination]) >= avoid_above_pct
    if ends_wet:
        which = destination if node_risk[destination] >= node_risk[origin] else origin
        message = (
            f"{which} is itself above {avoid_above_pct:.0f}% risk "
            f"({node_risk[which]:.0f}%). This is the safest way there, but the "
            "final stretch is flooded."
        )
    elif detour <= 20:
        message = "The most direct route is also the safest one right now."
    else:
        message = (
            f"Detour of {round(detour)} m to stay clear of water. "
            "The direct route runs through flooding."
        )

    return {
        "found": True,
        "message": message,
        "path": safe,
        "legs": legs,
        "total_m": round(total),
        "shortest_path": shortest or [],
        "shortest_m": round(shortest_m),
        "detour_m": round(detour),
        "worst_pct": round(worst, 1),
        "blocking": [],
        "node_risk": node_risk,
    }


def coords_for(drains: Dict[str, Drain], path: List[str]) -> List[Tuple[float, float]]:
    net = Network(drains)
    return [net.points[node] for node in path]


def place_names(drains: Dict[str, Drain]) -> List[str]:
    return sorted(Network(drains).points)
