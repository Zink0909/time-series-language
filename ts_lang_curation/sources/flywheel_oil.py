"""Canonical Pair adapter for the cache-first oil flywheel."""
from flywheel.oil_demo import pairs_and_trace


def pairs():
    examples, _ = pairs_and_trace(policy="prior_day", refresh=False, offline=True)
    yield from examples
