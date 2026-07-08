"""Trailing stop order type: tightens as price moves favorably, never
loosens, exits when price reverses back through the trailed level."""

import pytest

from engine import ledger
from engine.simulator import Simulator
from tests.test_simulator import T0, START, write_ticks
from datetime import timedelta


def rally_then_reverse_ticks(start_price, up_minutes, down_minutes, step, start=T0):
    """Walk price up for up_minutes, then back down for down_minutes."""
    ticks = []
    price = start_price
    for m in range(up_minutes):
        price += step
        ticks.append((start + timedelta(minutes=m, seconds=0), price - 0.00005, price + 0.00005))
    for m in range(down_minutes):
        price -= step
        ticks.append((start + timedelta(minutes=up_minutes + m, seconds=0),
                      price - 0.00005, price + 0.00005))
    return ticks


@pytest.fixture
def env(tmp_path):
    conn = ledger.connect(":memory:")
    ledger.init_db(conn)
    agent = ledger.register_agent(conn, "t", "test")
    acct = ledger.open_account(conn, agent, "challenge", START, T0.isoformat())

    def make(instrument="EUR_USD"):
        return conn, acct, Simulator(conn, acct, [instrument], data_dir=tmp_path)
    return tmp_path, make


class TestTrailingStop:
    def test_stop_tightens_as_price_rallies(self, env):
        tmp, make = env
        write_ticks(tmp, "EUR_USD", rally_then_reverse_ticks(1.1000, 10, 0, 0.0005))
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1, trailing_stop=0.0020)
        sim.step()  # fills the order
        trade = conn.execute("SELECT stop_loss, entry_price FROM trades").fetchone()
        initial_stop = trade["stop_loss"]
        assert initial_stop == pytest.approx(trade["entry_price"] - 0.0020)
        for _ in range(5):
            sim.step()
        stop_after = conn.execute("SELECT stop_loss FROM trades").fetchone()["stop_loss"]
        assert stop_after > initial_stop

    def test_exits_on_reversal_through_trailed_level(self, env):
        tmp, make = env
        ticks = rally_then_reverse_ticks(1.1000, 10, 10, 0.0005)
        write_ticks(tmp, "EUR_USD", ticks)
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1, trailing_stop=0.0020)
        while sim.step() is not None:
            pass
        closed = conn.execute(
            "SELECT * FROM trades WHERE exit_time IS NOT NULL").fetchall()
        assert len(closed) == 1
        # exits with a gain: trailed stop locked in profit above entry
        assert closed[0]["realized_pnl"] > 0

    def test_never_loosens_on_pullback_before_reversal(self, env):
        tmp, make = env
        # rally 8 minutes, small 2-minute pullback that doesn't breach the trail
        ticks = rally_then_reverse_ticks(1.1000, 8, 0, 0.0005)
        ticks += rally_then_reverse_ticks(ticks[-1][2], 0, 2, 0.0002,
                                          start=T0 + timedelta(minutes=8))
        write_ticks(tmp, "EUR_USD", ticks)
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1, trailing_stop=0.0020)
        stops = []
        for _ in range(10):
            sim.step()
            row = conn.execute("SELECT stop_loss FROM trades").fetchone()
            if row:
                stops.append(row["stop_loss"])
        # monotonically non-decreasing for a long trailing stop
        assert all(b >= a for a, b in zip(stops, stops[1:]))

    def test_fixed_stop_unaffected_when_no_trailing(self, env):
        tmp, make = env
        write_ticks(tmp, "EUR_USD", rally_then_reverse_ticks(1.1000, 10, 0, 0.0005))
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1, stop_loss=1.0950)
        for _ in range(5):
            sim.step()
        trade = conn.execute("SELECT * FROM trades").fetchone()
        assert trade["stop_loss"] == 1.0950
