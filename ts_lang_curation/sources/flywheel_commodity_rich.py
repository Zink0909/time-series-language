"""Canonical Pair adapter for frozen, cache-only multi-event commodity examples."""
from flywheel.commodity_demo import COMMODITIES, build_commodity


def pairs():
    for commodity in COMMODITIES:
        examples, _ = build_commodity(
            commodity, refresh=False, offline=True, cache_only=True, style="multi")
        yield from examples
