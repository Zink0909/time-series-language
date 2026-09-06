"""Preferred public entry point for the Treasury-yield + FOMC adapter."""
from sources.fred_fomc import pairs as _pairs


def pairs():
    """Yield canonical pairs from the migrated first-party implementation."""
    yield from _pairs()

__all__ = ["pairs"]
