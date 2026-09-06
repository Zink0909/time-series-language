"""Canonical Pair adapter for the seeded point-in-time Wikipedia flywheel."""
from flywheel.wiki_demo import pairs_and_trace


def pairs():
    examples, _ = pairs_and_trace(refresh=False)
    yield from examples
