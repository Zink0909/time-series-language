# companion/arms.py — the three experimental arms that separate "the TEXT CONTENT helps" from
# "just having a text channel / the alignment helps" (answers the MiGAS relabel critique):
#   A no_text        — series only (text removed)
#   B flywheel_text  — the manufactured world-knowledge text, correctly paired
#   C shuffled_text  — the same texts, but permuted onto the WRONG series (content decoupled)
# A real gain requires B to beat BOTH A and C. Beating A but not C => the gain is the channel, not
# the content. All three arms are identical series with only the conditioning text swapped.
import random
from collections import Counter


def partition_shuffleable(items, strata=("dataset",)):
    """Separate rows that can form a within-stratum derangement from singleton strata.

    Callers that intentionally tolerate coverage loss (for example the CPU plumbing smoke test)
    can report and drop the second return value. Training/export paths should keep using
    ``make_arms`` directly so an accidental singleton fails closed.
    """
    keys = [tuple(item.get(field) for field in strata) if strata else ("all",) for item in items]
    counts = Counter(keys)
    kept, dropped = [], []
    for item, key in zip(items, keys):
        (kept if counts[key] >= 2 else dropped).append(item)
    return kept, dropped


def make_arms(items, seed=0, strata=("dataset",)):
    """Build matched controls; shuffle within strata by default to preserve source/domain style."""
    if len(items) < 2:
        raise ValueError("at least two items are required to construct a shuffled control arm")
    B = [{**x, "arm": "B_flywheel_text"} for x in items]
    A = [{**x, "text": "", "arm": "A_no_text"} for x in items]
    texts = [x["text"] for x in items]
    perm = [None] * len(items)
    buckets = {}
    for i, item in enumerate(items):
        key = tuple(item.get(field) for field in strata) if strata else ("all",)
        buckets.setdefault(key, []).append(i)
    rng = random.Random(seed)
    for key, indices in buckets.items():
        if len(indices) < 2:
            raise ValueError(f"shuffle stratum {key!r} has only one item")
        shuffled = list(indices)
        # Sattolo's algorithm produces one cycle: a true permutation with no fixed points.
        for i in range(len(shuffled) - 1, 0, -1):
            j = rng.randrange(i)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        for source, target in zip(indices, shuffled):
            perm[source] = target
    assert sorted(perm) == list(range(len(items)))
    assert all(j != i for i, j in enumerate(perm))
    C = [{**items[i], "text": texts[perm[i]], "arm": "C_shuffled_text"} for i in range(len(items))]
    return {"A_no_text": A, "B_flywheel_text": B, "C_shuffled_text": C}
