# companion/arms.py — the three experimental arms that separate "the TEXT CONTENT helps" from
# "just having a text channel / the alignment helps" (answers the MiGAS relabel critique):
#   A no_text        — series only (text removed)
#   B flywheel_text  — the manufactured world-knowledge text, correctly paired
#   C shuffled_text  — the same texts, but permuted onto the WRONG series (content decoupled)
# A real gain requires B to beat BOTH A and C. Beating A but not C => the gain is the channel, not
# the content. All three arms are identical series with only the conditioning text swapped.
import random


def make_arms(items, seed=0):
    B = [{**x, "arm": "B_flywheel_text"} for x in items]
    A = [{**x, "text": "", "arm": "A_no_text"} for x in items]
    texts = [x["text"] for x in items]
    perm = list(range(len(items)))
    random.Random(seed).shuffle(perm)
    for i in range(len(perm)):                       # ensure no item keeps its own text
        if perm[i] == i:
            perm[i] = (i + 1) % len(perm)
    C = [{**items[i], "text": texts[perm[i]], "arm": "C_shuffled_text"} for i in range(len(items))]
    return {"A_no_text": A, "B_flywheel_text": B, "C_shuffled_text": C}
