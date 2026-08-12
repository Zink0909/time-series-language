#!/usr/bin/env python3
"""Reproducible builder/scaler for raw/cyber/events.json  (fixes the 'result exists but no generator' P0).

Sources (public, no key):
  - CISA KEV catalog: known-exploited-vulnerabilities feed -> CVE text + dateAdded.
  - Daily EPSS snapshots (FIRST / empiricalsecurity mirror): one CSV per date, ~2.4MB gz, 343k rows
    `cve,epss,percentile`. The free EPSS *time-series* API only returns the last ~30 days, so historical
    spike windows are reconstructed from these daily snapshots.

Each event = {cve,name,vendor,product,desc,dateAdded, dates[], series[]}, series = daily EPSS PERCENT
over [dateAdded-before, dateAdded+after]. Same schema as the hand-built 80; merges + dedups by cve and
NEVER overwrites an existing (already-verified) event.

  micromamba run -n ts-language python scripts/scale_cyber.py --since 2025-06-01

SEC-bug guards (see project CLAUDE.md): daily snapshot cached by DATE (unique key) + score_date verified
== request + target-set hash guard against stale cross-scope reuse; backoff x3; old events.json backed up.
"""
import os, sys, json, gzip, time, hashlib, argparse, shutil, datetime as dt, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, "..", "raw", "cyber")
SNAP = os.path.join(RAW, "epss_daily")                 # cache: {date}.json = {"h":targets_hash,"s":{cve:pct}}
UA   = "ts-language research (majiaju89@gmail.com)"
KEV_URL  = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
EPSS_URL = "https://epss.cyentia.com/epss_scores-{d}.csv.gz"     # 302 -> empiricalsecurity mirror


def _get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if k == tries - 1:
                raise
            time.sleep((15 if e.code in (403, 429) else 2) * (k + 1))   # IP throttle needs a long pause
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def fetch_kev():
    cache = os.path.join(RAW, "kev_catalog.json")
    if not os.path.exists(cache):
        open(cache, "wb").write(_get(KEV_URL))
    return json.load(open(cache))["vulnerabilities"]


def snapshot(date, targets, thash, offline=False):
    """{cve: epss_pct} restricted to `targets` for one date. Cached by date; re-fetch if scope changed."""
    cpath = os.path.join(SNAP, f"{date}.json")
    if os.path.exists(cpath):
        c = json.load(open(cpath))
        if c.get("h") == thash:
            return c["s"]                                  # stale-scope guard (SEC lesson)
    if offline:
        return {}                                          # cache-only mode: skip snapshots not yet fetched
    txt = gzip.decompress(_get(EPSS_URL.format(d=date))).decode()
    lines = txt.splitlines()
    assert lines[0].startswith("#") and f"score_date:{date}" in lines[0], \
        f"snapshot date mismatch for {date}: {lines[0][:70]}"      # verify response matches request
    sub = {}
    for ln in lines[2:]:                                   # skip '#model...' header + 'cve,epss,percentile'
        cve, epss, _ = ln.split(",", 2)
        if cve in targets:
            sub[cve] = round(float(epss) * 100, 2)
    os.makedirs(SNAP, exist_ok=True)
    json.dump({"h": thash, "s": sub}, open(cpath, "w"))
    time.sleep(0.5)                                        # be polite to the snapshot mirror (avoid 403/429)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-06-01", help="only CVEs with dateAdded >= this")
    ap.add_argument("--before", type=int, default=21)
    ap.add_argument("--after", type=int, default=8)
    ap.add_argument("--min-days", type=int, default=10)
    ap.add_argument("--offline", action="store_true", help="build from already-cached snapshots only (no fetch)")
    a = ap.parse_args()

    kev = fetch_kev()
    targets = [x for x in kev if x["dateAdded"] >= a.since]
    tset = {x["cveID"] for x in targets}
    thash = hashlib.md5(",".join(sorted(tset)).encode()).hexdigest()[:8]
    today = dt.date.today().isoformat()
    print(f"KEV {len(kev)} total -> {len(targets)} targets since {a.since} (scope {thash})")

    windows, need = {}, set()
    for x in targets:
        d0 = dt.date.fromisoformat(x["dateAdded"])
        ds = [(d0 + dt.timedelta(days=k)).isoformat() for k in range(-a.before, a.after + 1)]
        windows[x["cveID"]] = ds
        need.update(ds)
    need = sorted(d for d in need if d <= today)
    print(f"need {len(need)} daily snapshots")

    series = {c: {} for c in tset}
    for i, date in enumerate(need):
        for cve, pct in snapshot(date, tset, thash, offline=a.offline).items():
            series[cve][date] = pct
        if i % 25 == 0 or i == len(need) - 1:
            print(f"  {i+1}/{len(need)} snapshots ({date})", flush=True)

    events = []
    for x in targets:
        cve, sd = x["cveID"], series[x["cveID"]]
        dates = [d for d in windows[cve] if d in sd]
        if len(dates) < a.min_days:
            continue
        events.append({"cve": cve, "name": x["vulnerabilityName"], "vendor": x["vendorProject"],
                       "product": x["product"], "desc": x["shortDescription"], "dateAdded": x["dateAdded"],
                       "dates": dates, "series": [sd[d] for d in dates]})
    print(f"built {len(events)}/{len(targets)} events (>= {a.min_days} days)")

    epath = os.path.join(RAW, "events.json")
    existing = json.load(open(epath)) if os.path.exists(epath) else []
    if os.path.exists(epath):
        shutil.copy(epath, epath + f".bak_{today.replace('-', '')}")
    have = {e["cve"] for e in existing}
    new = [e for e in events if e["cve"] not in have]
    merged = existing + new
    json.dump(merged, open(epath, "w"), ensure_ascii=False)
    print(f"events.json: {len(existing)} existing + {len(new)} new = {len(merged)}  (backup .bak_{today.replace('-','')})")


if __name__ == "__main__":
    main()
