"""Canonical Pair adapter for the multi-peak Wikipedia flywheel."""
from flywheel.wiki_scale import build


def pairs():
    examples, _ = build(refresh=False, limit=0)
    yield from examples
