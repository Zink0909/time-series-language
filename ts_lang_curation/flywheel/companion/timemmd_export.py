#!/usr/bin/env python3
"""Time-MMD → external GENERALIZATION test set for the MiGAS fine-tune experiment.

Xinyue's key methodology point: the test set should NOT be an i.i.d slice of our own flywheel data
(too easy); it should be Time-MMD-style data — same domains but DIFFERENT phrasing/expression — to
test whether TRAINING on our flywheel data improves forecasts on unseen, differently-worded data.

Time-MMD (AdityaLab/Time-MMD) is a perfect fit: its textual reports are already split into `fact`
and `preds`, exactly MiGAS's h_fact / h_pred inputs. This adapter aligns each numerical target (OT)
window with the latest report AVAILABLE BEFORE the forecast origin (leakage-clean, T-1), and emits
(history, future, fact, preds) in the same shape our probes use.

  micromamba run -n ts-language python flywheel/companion/timemmd_export.py \
      --timemmd-dir /path/to/Time-MMD --domains Energy Economy
"""
import os, csv, sys, json, argparse, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_timemmd_test.json")
H, F = 10, 5


def _num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _date(s):
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _read_numerical(path):
    """-> sorted [(date, OT_value)] using the start_date column + OT target."""
    rows = list(csv.reader(open(path)))
    hdr = rows[0]
    oti = hdr.index("OT") if "OT" in hdr else 1
    sdi = hdr.index("start_date") if "start_date" in hdr else None
    out = []
    for r in rows[1:]:
        if sdi is None or len(r) <= max(oti, sdi):
            continue
        d, v = _date(r[sdi]), _num(r[oti])
        if d and v is not None:
            out.append((d, v))
    out.sort(key=lambda x: x[0])
    return out


def _read_reports(path):
    """-> sorted [(end_date, fact, preds)] — text known as of end_date."""
    out = []
    for r in csv.DictReader(open(path)):
        ed = _date(r.get("end_date", ""))
        fact = (r.get("fact") or "").strip()
        preds = (r.get("preds") or "").strip()
        if ed and (fact or preds):
            out.append((ed, fact, preds))
    out.sort(key=lambda x: x[0])
    return out


def _latest_report_before(reports, origin):
    """The most recent report whose end_date <= origin (leakage-clean = text predates the forecast)."""
    best = None
    for ed, fact, preds in reports:
        if ed <= origin:
            best = (ed, fact, preds)
        else:
            break
    return best


def export_domain(base, domain):
    numf = os.path.join(base, "numerical", domain, f"{domain}.csv")
    repf = os.path.join(base, "textual", domain, f"{domain}_report.csv")
    if not (os.path.exists(numf) and os.path.exists(repf)):
        return []
    series = _read_numerical(numf)
    reports = _read_reports(repf)
    ev = []
    for i in range(H, len(series) - F + 1):
        origin = series[i][0]
        rep = _latest_report_before(reports, origin)
        if not rep:
            continue                                     # need pre-origin text for the generalization test
        ed, fact, preds = rep
        ev.append({"domain": domain, "series_id": f"{domain}_{origin.isoformat()}",
                   "dataset": "time_mmd", "origin": origin.isoformat(),
                   "knowledge_time": ed.isoformat(),
                   "history": [round(v, 4) for _, v in series[i - H:i]],
                   "future": [round(v, 4) for _, v in series[i:i + F]],
                   "fact": fact[:1200], "preds": preds[:1200],
                   "text": (fact + " " + preds).strip()[:1600]})
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timemmd-dir", default="/tmp/timemmd", help="path to a cloned AdityaLab/Time-MMD")
    ap.add_argument("--domains", nargs="+",
                    default=["Energy", "Economy", "Environment", "Climate", "Traffic"])
    a = ap.parse_args()
    ev, per = [], {}
    for d in a.domains:
        de = export_domain(a.timemmd_dir, d)
        # keep leakage sane: origin strictly after knowledge_time OR same period is fine (report is
        # dated to its window end, which is <= origin); assert no report dated after origin slipped in
        de = [e for e in de if e["knowledge_time"] <= e["origin"]]
        per[d] = len(de)
        ev += de
    json.dump({"hist_len": H, "pred_len": F, "space": "raw", "source": "Time-MMD (external)",
               "n": len(ev), "by_domain": per, "events": ev}, open(OUT, "w"), ensure_ascii=False)
    print(f"exported {len(ev)} Time-MMD generalization-test events -> {OUT} "
          f"({os.path.getsize(OUT)//1024} KB)")
    print("by domain:", per)
    if ev:
        e = ev[0]
        print(f"sample [{e['domain']}] origin={e['origin']} kt={e['knowledge_time']}")
        print("  hist:", e["history"], "-> fut:", e["future"])
        print("  fact:", e["fact"][:120])


if __name__ == "__main__":
    main()
