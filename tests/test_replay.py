"""Replay tests against a small generated parquet fixture, plus a decoder
test for the Dukascopy .bi5 format using a hand-built compressed blob."""

import lzma
import struct
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from engine import replay


@pytest.fixture
def tick_dir(tmp_path):
    """Three minutes of ticks, gap in minute two, across two files."""
    base = tmp_path / "raw" / "EUR_USD"
    base.mkdir(parents=True)
    t0 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

    def make(ticks, name):
        table = pa.table({
            "ts": [t for t, _, _ in ticks],
            "bid": [b for _, b, _ in ticks],
            "ask": [a for _, _, a in ticks],
            "bid_vol": [1.0] * len(ticks),
            "ask_vol": [1.0] * len(ticks),
        })
        pq.write_table(table, base / name)

    make([(t0 + timedelta(seconds=s), 1.1000 + s * 1e-5, 1.1001 + s * 1e-5)
          for s in (0, 15, 45)], "a.parquet")
    make([(t0 + timedelta(minutes=2, seconds=s), 1.1010, 1.1011)
          for s in (5, 30)], "b.parquet")
    return tmp_path


def test_iter_ticks_time_ordered(tick_dir):
    ticks = list(replay.iter_ticks("EUR_USD", data_dir=tick_dir))
    assert len(ticks) == 5
    assert all(ticks[i][0] <= ticks[i + 1][0] for i in range(4))
    assert ticks[0][1] == pytest.approx(1.1000)


def test_iter_minutes_groups_and_skips_empty(tick_dir):
    minutes = list(replay.iter_minutes("EUR_USD", data_dir=tick_dir))
    assert len(minutes) == 2  # minute 1 empty, skipped
    (m0, t0_ticks), (m2, m2_ticks) = minutes
    assert len(t0_ticks) == 3
    assert len(m2_ticks) == 2
    assert (m2 - m0).total_seconds() == 120


def test_time_window_filter(tick_dir):
    ticks = list(replay.iter_ticks("EUR_USD", data_dir=tick_dir,
                                   start="2026-01-05 09:01:00", end="2026-01-05 09:03:00"))
    assert len(ticks) == 2


def test_bi5_decoder_roundtrip():
    from scripts.download_data import decode_bi5
    hour = datetime(2026, 1, 5, 9, tzinfo=timezone.utc)
    records = struct.pack(">IIIff", 1500, 110015, 110005, 1.5, 2.5)
    records += struct.pack(">IIIff", 59_000, 110025, 110020, 1.0, 1.0)
    blob = lzma.compress(records)
    ticks = decode_bi5(blob, hour, price_divisor=100_000)
    assert len(ticks) == 2
    ts, bid, ask, bid_vol, ask_vol = ticks[0]
    assert ts == hour + timedelta(milliseconds=1500)
    assert bid == pytest.approx(1.10005)
    assert ask == pytest.approx(1.10015)
    assert bid_vol == 2.5  # dukascopy record order: ask vol before bid vol


def test_bi5_empty_hour():
    from scripts.download_data import decode_bi5
    assert decode_bi5(b"", datetime.now(timezone.utc), 100_000) == []
