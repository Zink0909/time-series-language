"""Fetch and cache first-party U.S. government time-series sources.

This replaces FRED as a redistribution layer. Every cache is written atomically and records
the exact official endpoint in its header/manifest-facing adapter metadata.
"""
from __future__ import annotations

import csv
import concurrent.futures
import io
import os
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


UA = "ts-language research (majiaju89@gmail.com)"
TREASURY_XML = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    "?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".official-series-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _download(url):
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse_treasury_yield_xml(payload):
    """Return sorted ``(date, 2-year, 10-year)`` rows from a Treasury Atom feed."""
    root = ET.fromstring(payload)
    rows = []
    for properties in (node for node in root.iter() if node.tag.endswith("properties")):
        values = {child.tag.rsplit("}", 1)[-1]: (child.text or "").strip()
                  for child in properties}
        date = values.get("NEW_DATE", "")[:10]
        two, ten = values.get("BC_2YEAR"), values.get("BC_10YEAR")
        if date and two and ten:
            rows.append((date, float(two), float(ten)))
    return sorted(set(rows))


def fetch_treasury_yields(start_year, end_year, cache_path, refresh=False):
    """Load a canonical CSV cache, fetching official Treasury XML yearly when needed."""
    cache_path = Path(cache_path)
    if not cache_path.exists() or refresh:
        rows = []
        years = list(range(start_year, end_year + 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(years))) as pool:
            payloads = pool.map(lambda year: _download(TREASURY_XML.format(year=year)), years)
            for payload in payloads:
                rows.extend(parse_treasury_yield_xml(payload))
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["date", "treasury_2y", "treasury_10y"])
        writer.writerows(sorted(set(rows)))
        _atomic_text(cache_path, output.getvalue())
    with cache_path.open(newline="", encoding="utf-8") as handle:
        parsed = []
        for row in csv.DictReader(handle):
            if row.get("date") and row.get("treasury_2y") and row.get("treasury_10y"):
                parsed.append((row["date"], float(row["treasury_2y"]), float(row["treasury_10y"])))
    if not parsed:
        raise ValueError(f"official Treasury cache is empty: {cache_path}")
    return parsed
