"""Historical tick replay. Reads Parquet tick files through DuckDB, no
server, no setup step.

Tick file layout (written by scripts/download_data.py and by the synthetic
generator alike, so replay code never cares which source produced a file):

  data/raw/<INSTRUMENT>/*.parquet        historical ticks
  data/calibrated/<INSTRUMENT>/*.parquet synthetic ticks

Columns: ts TIMESTAMP, bid DOUBLE, ask DOUBLE, bid_vol DOUBLE, ask_vol DOUBLE.

Two clocks, per AGENTS.md: the agent acts once per simulated minute, every
tick inside that minute feeds the matching engine. iter_minutes yields
(minute_start, [ticks]) so the simulator can do exactly that.
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def tick_relation(instrument, source="raw", data_dir=None, start=None, end=None):
    """DuckDB relation over every tick file for one instrument, time ordered."""
    base = Path(data_dir or DATA_DIR) / source / instrument
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    q = f"SELECT ts, bid, ask FROM read_parquet('{base}/*.parquet')"
    clauses = []
    if start:
        clauses.append(f"ts >= TIMESTAMPTZ '{start}'")
    if end:
        clauses.append(f"ts < TIMESTAMPTZ '{end}'")
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY ts"
    return con, q


def iter_ticks(instrument, source="raw", data_dir=None, start=None, end=None):
    """Yield (ts, bid, ask) tuples in time order."""
    con, q = tick_relation(instrument, source, data_dir, start, end)
    cur = con.execute(q)
    while True:
        rows = cur.fetchmany(10_000)
        if not rows:
            break
        yield from rows
    con.close()


def iter_minutes(instrument, source="raw", data_dir=None, start=None, end=None):
    """Yield (minute_start, ticks) where ticks is every (ts, bid, ask) inside
    that simulated minute. Empty minutes are skipped, matching real feeds."""
    bucket_start, bucket = None, []
    for ts, bid, ask in iter_ticks(instrument, source, data_dir, start, end):
        minute = ts.replace(second=0, microsecond=0)
        if minute != bucket_start:
            if bucket:
                yield bucket_start, bucket
            bucket_start, bucket = minute, []
        bucket.append((ts, bid, ask))
    if bucket:
        yield bucket_start, bucket
