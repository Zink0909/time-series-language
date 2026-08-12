#!/usr/bin/env python3
"""BigQuery GDELT pool builder — the SCALE retrieval path (Xinyue: pull broad to a local DB, not
the rate-limited DOC API). ONE SQL over the GKG `gkg_partitioned` table pulls every commodity-related
news doc in 2021-06..2024-12 with its real PAGE_TITLE (from the Extras field), deduped by URL, and
writes the SAME per-commodity monthly pool files `commodity_demo.py --offline` consumes
(raw/_pool_<key>.json). Cost is capped by partition pruning (_PARTITIONTIME) + minimal columns
(~250 GB scan, ~25% of the 1 TB/month free tier); the pull is one-time, everything after is local.

Auth: a read-only service-account JSON (path in GOOGLE_APPLICATION_CREDENTIALS or --key). The key is
never printed or committed. Run:
  micromamba run -n ts-language python flywheel/bq_pool.py [--dry] [--key /path/to/sa.json]
"""
import os, sys, re, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import commodity_demo as cd                                            # noqa: E402
from google.cloud import bigquery                                     # noqa: E402

DEFAULT_KEY = "/Users/mmmm/Downloads/ts-lang-4a3d1891d646.json"
START, END = cd.COSD, "2025-01-01"
PER_MONTH_CAP = 400                                                    # even-spread cap per month

# Route each on-topic doc to a commodity by TIGHT terms in its TITLE (not the URL) — this separates
# the four commodities cleanly and avoids the "oil article mentions 'dollar' -> eurusd" cross-routing
# that a broad noun list caused. A doc may match >1 (rare with these terms).
ROUTE = {
    "brent":    r"\b(crude|brent|opec|petroleum|wti|oil)\b",
    "natgas":   r"\b(natural[ -]?gas|natgas|lng|henry hub|liquefied natural)\b",
    "bitcoin":  r"\b(bitcoin|btc|crypto|cryptocurrency|blockchain)\b",
    "ethereum": r"\b(ethereum|ether|eth)\b",
    "eurusd":   r"\b(euro|ecb|eurozone|european central bank|euro[ -]?dollar)\b",
    "gbpusd":   r"\b(pound|sterling|gbp|bank of england)\b",
    "jpyusd":   r"\b(yen|jpy|bank of japan|boj)\b",
}


def _spread(arts, cap):
    """Dedupe by title, then keep an EVENLY-SPACED sample across the month (sorted by time) so every
    part of the month is represented — NOT the most-recent N, which starves early-month moves."""
    seen, uniq = set(), []
    for a in sorted(arts, key=lambda x: x["date"]):                   # chronological
        k = re.sub(r"\W+", "", a["title"].lower())
        if k and k not in seen:
            seen.add(k)
            uniq.append(a)
    if len(uniq) <= cap:
        return uniq
    step = len(uniq) / cap
    return [uniq[int(i * step)] for i in range(cap)]
# PRECISE URL keyword prefilter (one scan for all commodities; per-commodity routing is done
# locally). Deliberately NOT the broad words (oil/euro/dollar/gas) — those match Europe-wide and
# gasoline/Vegas noise and blow the result set into the millions. These slug-style terms keep the
# download to tens of thousands of on-topic docs; local routing below uses each commodity's noun list.
KW = (r"(opec|crude|brent|wti|petroleum|natural-gas|natgas|henry-hub|liquefied-natural"
      r"|bitcoin|cryptocurrency|ethereum|euro-dollar|eurusd|european-central-bank"
      r"|sterling|british-pound|pound-sterling|pound-dollar|gbp|bank-of-england"
      r"|japanese-yen|usd-jpy|yen-dollar|bank-of-japan)")

SQL = f"""
SELECT url, ANY_VALUE(domain) AS domain, ANY_VALUE(title) AS title, CAST(MIN(DATE) AS STRING) AS date
FROM (
  SELECT DocumentIdentifier AS url, SourceCommonName AS domain, DATE,
         REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('{START}') AND _PARTITIONTIME < TIMESTAMP('{END}')
    AND REGEXP_CONTAINS(Extras, r'<PAGE_TITLE>')
    AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'{KW}')
)
WHERE title IS NOT NULL AND title != ''
GROUP BY url
"""


def _client(key):
    return bigquery.Client.from_service_account_json(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="estimate scan cost only, do not run")
    ap.add_argument("--key", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", DEFAULT_KEY))
    a = ap.parse_args()
    c = _client(a.key)

    if a.dry:
        job = c.query(SQL, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
        print(f"dry-run: would scan {job.total_bytes_processed/1e9:.1f} GB "
              f"({job.total_bytes_processed/1e12*100:.1f}% of the 1 TB free tier)")
        return

    routers = {key: re.compile(rx, re.I) for key, rx in ROUTE.items()}
    pools = {key: {} for key in ROUTE}                                # key -> {month -> [articles]}

    job = c.query(SQL)
    kept = n_rows = 0
    for r in job.result():                                            # stream, don't load all in RAM
        n_rows += 1
        url, title, domain, date = r["url"], r["title"], r["domain"], r["date"]
        if not date or len(date) < 6 or not title:
            continue
        month = f"{date[:4]}-{date[4:6]}"
        for key, rx in routers.items():
            if rx.search(title):                                      # route on TITLE only
                pools[key].setdefault(month, []).append(
                    {"date": date, "title": title.strip(), "domain": domain or "", "url": url})
                kept += 1

    scanned = job.total_bytes_processed / 1e9
    print(f"scanned {scanned:.1f} GB{' (cached)' if scanned == 0 else ''} · {n_rows:,} deduped docs "
          f"-> routed {kept:,} into pools")
    for key in pools:
        for m, arts in pools[key].items():
            pools[key][m] = _spread(arts, PER_MONTH_CAP)              # even spread across the month
        path = os.path.join(cd.RAW, f"_pool_{key}.json")
        json.dump(pools[key], open(path, "w"), ensure_ascii=False, indent=1)
        n_art = sum(len(v) for v in pools[key].values())
        print(f"  {key:8s}: {len(pools[key])} months, {n_art} articles -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
