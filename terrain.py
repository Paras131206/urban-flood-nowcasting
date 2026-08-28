"""What each catchment is made of, and why that decides how much runs off.

The rational method needs a runoff coefficient C — the fraction of rain that
becomes surface flow instead of soaking in. The original model used one number,
0.9, for all of Bandra. That is wrong in a way that matters: Pali Hill is
bungalow gardens and old trees, BKC is a planned commercial district with
almost no soft ground, and the two do not shed water alike.

So C is composed here from the actual surface mix. The per-surface values are
the standard rational-method figures used in Indian urban drainage design
(CPHEEO manual, and the same ranges appear in ASCE and IS practice):

    roof / terrace     0.90    water lands and leaves, nothing absorbs it
    asphalt road       0.88    sealed, and cambered to shed fast
    concrete / paved   0.90    forecourts, footpaths, parking
    bare / compacted   0.50    unpaved lanes, construction ground
    vegetation         0.20    gardens, tree cover, playgrounds
    open water         1.00    a pond is already full

A catchment that is 90% hard surface turns 0.86 of every millimetre into
runoff. One with a third under trees turns roughly 0.68. On a 100 mm/hr burst
across 20 hectares that difference is about 1 m³/s — comparable to the entire
rated capacity of a tertiary drain. Terrain is not decoration.

The mixes below are estimated from land use, not surveyed. A deployment would
take them from a land-cover raster; the numbers would change, the method would
not.
"""
from __future__ import annotations

from typing import Dict

# Runoff coefficient per surface type.
C_VALUES: Dict[str, float] = {
    "roof": 0.90,
    "asphalt": 0.88,
    "concrete": 0.90,
    "bare": 0.50,
    "vegetation": 0.20,
    "water": 1.00,
}

LABELS: Dict[str, str] = {
    "roof": "Rooftops",
    "asphalt": "Roads",
    "concrete": "Paved ground",
    "bare": "Bare / unpaved",
    "vegetation": "Trees & gardens",
    "water": "Open water",
}

# Fraction of each catchment by surface. Rows sum to 1.0.
MIX: Dict[str, Dict[str, float]] = {
    # Bandra East by the Mithi. Dense, largely informal, a lot of hard roof
    # and beaten earth, with mangrove and channel at the edge.
    "BND-P01": {"roof": 0.38, "asphalt": 0.18, "concrete": 0.14,
                "bare": 0.16, "vegetation": 0.08, "water": 0.06},
    # BKC. Planned commercial: towers, plazas, wide roads, almost nothing soft.
    "BND-P02": {"roof": 0.34, "asphalt": 0.26, "concrete": 0.28,
                "bare": 0.02, "vegetation": 0.10, "water": 0.00},
    # SV Road. Arterial with continuous frontage either side.
    "BND-S01": {"roof": 0.36, "asphalt": 0.30, "concrete": 0.22,
                "bare": 0.04, "vegetation": 0.08, "water": 0.00},
    # Hill Road. Shopping street, older buildings, some surviving tree line.
    "BND-S02": {"roof": 0.34, "asphalt": 0.24, "concrete": 0.20,
                "bare": 0.05, "vegetation": 0.17, "water": 0.00},
    # Linking Road. Retail spine, near-total hardstanding.
    "BND-S03": {"roof": 0.38, "asphalt": 0.30, "concrete": 0.24,
                "bare": 0.02, "vegetation": 0.06, "water": 0.00},
    # Pali Hill. Low-rise on large plots, mature canopy. The greenest catchment.
    "BND-T01": {"roof": 0.22, "asphalt": 0.14, "concrete": 0.12,
                "bare": 0.08, "vegetation": 0.44, "water": 0.00},
    # Chimbai. Fishing village: dense low roofs, narrow unpaved lanes, sea edge.
    "BND-T02": {"roof": 0.40, "asphalt": 0.10, "concrete": 0.14,
                "bare": 0.26, "vegetation": 0.06, "water": 0.04},
    # Turner Road. Commercial, fully sealed.
    "BND-T03": {"roof": 0.35, "asphalt": 0.28, "concrete": 0.25,
                "bare": 0.04, "vegetation": 0.08, "water": 0.00},
    # Mount Mary. Hillside, church grounds, steps and terraces under trees.
    "BND-T04": {"roof": 0.20, "asphalt": 0.12, "concrete": 0.16,
                "bare": 0.10, "vegetation": 0.42, "water": 0.00},
    # Bandra Talao. A pond, its promenade, and the roads that ring it.
    "BND-T05": {"roof": 0.24, "asphalt": 0.22, "concrete": 0.18,
                "bare": 0.06, "vegetation": 0.16, "water": 0.14},
}

# Used when a drain has no entry — the Bandra-wide average, so an unmapped
# catchment behaves like a typical one rather than silently becoming a park.
DEFAULT_MIX: Dict[str, float] = {
    "roof": 0.33, "asphalt": 0.22, "concrete": 0.19,
    "bare": 0.08, "vegetation": 0.16, "water": 0.02,
}


def mix_for(drain_id: str) -> Dict[str, float]:
    return MIX.get(drain_id, DEFAULT_MIX)


def runoff_coefficient(drain_id: str) -> float:
    """Area-weighted C for this catchment."""
    mix = mix_for(drain_id)
    total = sum(mix.values()) or 1.0
    return round(sum(C_VALUES[s] * f for s, f in mix.items()) / total, 3)


def sealed_fraction(drain_id: str) -> float:
    """How much of the catchment water cannot get into the ground through."""
    mix = mix_for(drain_id)
    return round(mix.get("roof", 0) + mix.get("asphalt", 0) + mix.get("concrete", 0), 3)


def describe(drain_id: str) -> str:
    """The mix as a sentence, biggest share first."""
    mix = mix_for(drain_id)
    parts = sorted(mix.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{round(f * 100)}% {LABELS[s].lower()}" for s, f in parts if f >= 0.01)


def table_rows(drain_id: str) -> list:
    """Rows for a Streamlit table: surface, share, its C, its contribution."""
    mix = mix_for(drain_id)
    rows = []
    for surface, fraction in sorted(mix.items(), key=lambda kv: -kv[1]):
        if fraction <= 0:
            continue
        rows.append({
            "Surface": LABELS[surface],
            "Share": f"{fraction * 100:.0f}%",
            "Runoff C": C_VALUES[surface],
            "Contributes": round(C_VALUES[surface] * fraction, 3),
        })
    return rows
