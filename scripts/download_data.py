"""Download historical tick data into data/raw/<INSTRUMENT>/*.parquet.

Forex: Dukascopy public datafeed, free, no key. One .bi5 file per hour,
LZMA-compressed records of (ms_offset, ask, bid, ask_vol, bid_vol).

Futures: Databento, free usage credits, needs DATABENTO_API_KEY in .env.

Fallback: Yahoo 1-minute bars (secondary source per AGENTS.md), one tick
per minute close with the config base spread, when the Dukascopy CDN is
unreachable. Good enough for calibration (which fits minute returns);
replay realism prefers real Dukascopy ticks.

Usage:
  venv/bin/python scripts/download_data.py forex EUR_USD 2025-01-06 2025-01-10
  venv/bin/python scripts/download_data.py futures ES 2025-01-06 2025-01-10
  venv/bin/python scripts/download_data.py yahoo EUR_USD   # last ~7 days, 1m
"""

import lzma
import struct
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.config import load_contracts  # noqa: E402

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"

TICK_SCHEMA = pa.schema([
    ("ts", pa.timestamp("us", tz="UTC")),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("bid_vol", pa.float64()),
    ("ask_vol", pa.float64()),
])


def decode_bi5(raw, hour_start, price_divisor):
    """Dukascopy .bi5: LZMA stream of 20-byte big-endian records."""
    if not raw:
        return []
    data = lzma.decompress(raw)
    ticks = []
    for off in range(0, len(data), 20):
        ms, ask_i, bid_i, ask_v, bid_v = struct.unpack_from(">IIIff", data, off)
        ts = hour_start + timedelta(milliseconds=ms)
        ticks.append((ts, bid_i / price_divisor, ask_i / price_divisor,
                      float(bid_v), float(ask_v)))
    return ticks


def fetch_dukascopy_day(instrument, day, price_divisor):
    """All ticks for one UTC day. Dukascopy months are zero-indexed in URLs."""
    sym = instrument.replace("_", "")
    ticks = []
    for hour in range(24):
        url = DUKASCOPY_URL.format(sym=sym, y=day.year, m=day.month - 1,
                                   d=day.day, h=hour)
        hour_start = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                ticks.extend(decode_bi5(resp.read(), hour_start, price_divisor))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue  # market closed that hour
            raise
    return ticks


def write_day(instrument, day, ticks):
    if not ticks:
        return
    out_dir = DATA_RAW / instrument
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(zip(*ticks))
    table = pa.table(
        {name: list(col) for name, col in zip(("ts", "bid", "ask", "bid_vol", "ask_vol"), cols)},
        schema=TICK_SCHEMA)
    pq.write_table(table, out_dir / f"{day.isoformat()}.parquet")
    print(f"{instrument} {day}: {len(ticks)} ticks")


def download_forex(instrument, start, end):
    spec = load_contracts()[instrument]
    # Dukascopy stores prices as ints at one decimal finer than the pip.
    price_divisor = round(10 / spec["pip_size"])
    day = start
    while day <= end:
        write_day(instrument, day, fetch_dukascopy_day(instrument, day, price_divisor))
        day += timedelta(days=1)


def download_futures(instrument, start, end):
    import databento  # deferred: only futures downloads need the key + package

    import os
    key = os.environ.get("DATABENTO_API_KEY") or _read_env_key("DATABENTO_API_KEY")
    client = databento.Historical(key)
    df = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[f"{instrument}.c.0"],  # front month continuous
        stype_in="continuous",
        schema="trades",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
    ).to_df()
    # Trades carry one price; store it as both bid and ask, spread comes from config.
    for day, group in df.groupby(df.index.date):
        ticks = [(ts.to_pydatetime(), row["price"], row["price"], row["size"], row["size"])
                 for ts, row in group.iterrows()]
        write_day(instrument, day, ticks)


YAHOO_SYMBOLS = {"ES": "ES=F", "NQ": "NQ=F", "YM": "YM=F", "CL": "CL=F",
                 "GC": "GC=F", "ZN": "ZN=F", "6E": "6E=F"}


def download_yahoo(instrument):
    """Last ~7 days of 1-minute closes as one synthetic tick per minute.
    Spread comes from config; bid/ask are close -/+ half the base spread."""
    import json
    spec = load_contracts()[instrument]
    if spec["type"] == "forex":
        symbol = instrument.replace("_", "") + "=X"
        half = spec["spread_pips_peak"] * spec["pip_size"] / 2
    else:
        symbol = YAHOO_SYMBOLS[instrument]
        half = spec["spread_ticks"] * spec["tick_size"] / 2
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval=1m&range=7d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    result = data["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    by_day = {}
    for t, c in zip(stamps, closes):
        if c is None:
            continue
        ts = datetime.fromtimestamp(t, tz=timezone.utc)
        by_day.setdefault(ts.date(), []).append(
            (ts, c - half, c + half, 1.0, 1.0))
    for day, ticks in sorted(by_day.items()):
        write_day(instrument, day, ticks)


def _read_env_key(name):
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{name} not set in environment or .env")


if __name__ == "__main__":
    kind, instrument = sys.argv[1], sys.argv[2]
    if kind == "yahoo":
        download_yahoo(instrument)
    else:
        start, end = date.fromisoformat(sys.argv[3]), date.fromisoformat(sys.argv[4])
        if kind == "forex":
            download_forex(instrument, start, end)
        else:
            download_futures(instrument, start, end)
