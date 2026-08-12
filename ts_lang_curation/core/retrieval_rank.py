# core/retrieval_rank.py — W2: retrieval ranking + denoising for S4. GDELT returns a mixed bag
# (relevant event headlines + off-topic noise that happened to match the query window). Feeding
# noise to the LLM dilutes the cause. This ranks headlines by relevance to the domain/query terms
# and drops the off-topic ones BEFORE grounding — programmatic, deterministic, no model.
# Reference: Trade-the-Event (2105.12825) corporate-event extraction + From-News-to-Forecast
# (2409.17515) news filtering; the ranked, denoised set makes the cause traceable to on-topic events.
import os
import re

_TOK = re.compile(r"[a-z][a-z+]{2,}")


def _tokens(text):
    return _TOK.findall(text.lower())


def relevance(headline, terms):
    """# of domain terms that hit the headline (substring so 'opec' matches 'opec+', 'oil' matches
    'oilprice'). A cheap, transparent relevance score."""
    toks = _tokens(headline)
    joined = " ".join(toks)
    return sum(1 for t in terms if t in joined)


def rank_headlines(headlines, terms, min_score=1):
    """Return (kept, noise, scored). `kept` = headlines with >=min_score domain hits, ordered most-
    relevant first (stable within ties); `noise` = the off-topic remainder. `scored` = [(score,h)]."""
    scored = [(relevance(h, terms), h) for h in headlines]
    ordered = sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))   # stable, desc by score
    kept = [scored[i][1] for i in ordered if scored[i][0] >= min_score]
    noise = [scored[i][1] for i in ordered if scored[i][0] < min_score]
    return kept, noise, [scored[i] for i in ordered]


def semantic_rank(headlines, topic, cache_dir=None):
    """W2 v2: SEMANTIC denoising — an LLM judges each headline's topical relevance to `topic`,
    fixing bag-of-terms' word-ambiguity (keeps 'OPEC out of spare capacity', drops an unrelated
    'Naira forex reserves' even though both share the word 'reserves'). Returns (kept, dropped).
    Independent judge prompt + temperature 0 + cached => reproducible. FAIL-OPEN: on any API/parse
    failure it keeps ALL headlines (denoising must never silently delete a real event)."""
    import re as _re
    import hashlib, json
    key = hashlib.sha1((topic + "||" + "\n".join(headlines)).encode()).hexdigest()[:16]
    cache = os.path.join(cache_dir, f"sem_{key}.json") if cache_dir else None
    if cache and os.path.exists(cache):
        c = json.load(open(cache))
        return c["kept"], c["dropped"]
    numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
    sys_p = ("You judge whether each news headline is topically relevant to a subject. Relevant = "
             "it is about, or materially affects, the subject. Be strict about topical relevance, "
             "not just shared words.")
    user_p = (f"Subject: {topic}\n\nHeadlines:\n{numbered}\n\nFor each headline number answer "
              "relevant or not. Output only lines like '1: yes' or '2: no'.")
    try:
        from core.compress import complete
        out = complete(sys_p, user_p, max_tokens=8 * len(headlines) + 40, temperature=0.0)
    except Exception:
        return headlines, []
    keep = set()
    for m in _re.finditer(r"(\d+)\s*[:.\)]\s*(yes|no|relevant|irrelevant)", out.lower()):
        if m.group(2) in ("yes", "relevant"):
            keep.add(int(m.group(1)))
    kept = [h for i, h in enumerate(headlines, 1) if i in keep]
    dropped = [h for i, h in enumerate(headlines, 1) if i not in keep]
    if not kept:                                            # parse failed -> fail-open
        kept, dropped = headlines, []
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump({"kept": kept, "dropped": dropped}, open(cache, "w"), ensure_ascii=False)
    return kept, dropped
