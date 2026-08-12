# sources/fnspid.py — thin adapter (full-schema port of the AAPL earnings demo).
# NOTE: FNSPID is owned by Faisal; kept here only to prove the core handles >1 source.
import os
from core.schema import TextToTs

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")


def _aapl():
    rows = []
    for line in open(os.path.join(RAW, "AAPL_close.csv")).read().splitlines()[1:]:
        d, c = line.split(","); rows.append((d, float(c)))
    return rows


def pairs():
    s = _aapl(); dates = [d for d, _ in s]; closes = [c for _, c in s]
    i = dates.index("2020-07-30")
    hist = closes[i - 24:i + 1]            # 25 sessions through the earnings day
    fut = closes[i + 1:i + 6]              # 5 sessions after the report
    if len(hist) < 3 or len(fut) < 1:
        return
    news = ("Apple Q3 FY2020 (reported 2020-07-30 post-market): EPS $0.65 vs $0.51 est and revenue "
            "$59.69B vs $52.25B est, both beats, plus a 4-for-1 stock split announcement.")

    # text -> ts (ts_forecast): earnings news + pre-report price history -> post-report reaction
    yield TextToTs(
        user_text=(f"{news} Given this report and the prior price history, forecast AAPL's "
                   f"daily closing price."),
        history=hist, future=fut,
        series_name="AAPL daily closing price (USD)", unit="close_price_usd", freq="business_daily",
        meta={"dataset": "fnspid", "source": "FNSPID (news) + daily close",
              "series_id": "AAPL_2020-07-30_earnings", "ticker": "AAPL",
              "event_date": "2020-07-30", "direction": "text_to_ts"},
        text_source="financial_news_earnings", is_generated="real",
        knowledge_time="2020-07-30T20:30:00Z")
