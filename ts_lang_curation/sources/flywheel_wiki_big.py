"""Canonical Pair adapter for the large Wikipedia flywheel."""
from flywheel.wiki_scale_big import build


def pairs():
    examples, _ = build(n_entities=600, refresh=False)
    yield from examples
