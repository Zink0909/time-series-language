# core/detect.py — W1: source-agnostic salient-move detection, upgrading S2 from a hand-picked
# fixed threshold to a robust, MULTI-SCALE detector. Programmatic + deterministic (no training,
# no model dependency), per the reference papers' idea that events live at different time scales:
#   - Multi-scale changepoint detection (1905.06913): a single-day return misses slow drifts.
#   - Event vs changepoint detection (2408.12792): sustained moves ≠ instantaneous jumps.
#   - Deep changepoint detection (2211.03860): the direction we point toward (a trainable detector
#     is v2; this v1 is the robust-statistics rung that already fixes the threshold's blind spots).
#
# The old S2 = single-day |ret| >= RET_MIN. Its blind spot: a move spread over 3–5 days (each day
# under the threshold) is a real salient event but is never flagged. The multi-scale detector runs
# a robust MAD z-score over several return horizons and keeps a move salient if ANY scale fires —
# so slow drifts get caught without lowering the single-day bar into noise.
from statistics import median


def _k_returns(prices, k):
    return [(i, prices[i] / prices[i - k] - 1.0) for i in range(k, len(prices)) if prices[i - k]]


def _robust_z(vals):
    med = median(vals)
    mad = median([abs(v - med) for v in vals]) or 1e-9
    return med, mad


def detect_threshold(prices, ret_min=0.06, spacing=10, top_k=8):
    """Reproduces the old S2: single-day |return| >= ret_min, greedy by magnitude, >=spacing apart.
    Returns [(idx, scale=1, ret, None)] so it lines up with the multi-scale output shape."""
    cand = [(i, r) for i, r in _k_returns(prices, 1) if abs(r) >= ret_min]
    cand.sort(key=lambda c: -abs(c[1]))
    picked = []
    for i, r in cand:
        if (top_k is None or len(picked) < top_k) and all(abs(i - j) >= spacing for j, *_ in picked):
            picked.append((i, 1, r, None))
    picked.sort()
    return picked


def detect_robust_multiscale(prices, scales=(1, 3, 5), z_min=2.8, spacing=10, top_k=None):
    """Robust MAD z-score across several return horizons; a move is salient if any scale's |z| >=
    z_min. Greedy by |z|, deduped across scales by spacing (one event ~ one point, tagged with the
    scale that fired strongest). Returns [(idx, scale, ret, z)] sorted by index."""
    cand = []
    for k in scales:
        rets = _k_returns(prices, k)
        med, mad = _robust_z([r for _, r in rets])
        for i, r in rets:
            z = 0.6745 * (r - med) / mad
            if abs(z) >= z_min:
                cand.append((i, k, r, z))
    cand.sort(key=lambda c: -abs(c[3]))
    picked = []
    for i, k, r, z in cand:
        if (top_k is None or len(picked) < top_k) and all(abs(i - j) >= spacing for j, *_ in picked):
            picked.append((i, k, r, z))
    picked.sort()
    return picked
