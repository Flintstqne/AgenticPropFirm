"""Simulator tests over generated parquet fixtures: order fills at the tick
where price crosses, tick-level rule breach mid-minute, SL/TP triggers,
daily reset, both agent wrappers driving the same core."""

from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agents.llm_tools import LLMToolbox
from agents.rl_env import PropFirmEnv, BUY, HOLD
from engine import ledger
from engine.simulator import Simulator

START = 100_000.0
T0 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)  # 09:00 ET


def write_ticks(dirpath, instrument, ticks):
    base = dirpath / "raw" / instrument
    base.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "ts": [t for t, _, _ in ticks],
        "bid": [b for _, b, _ in ticks],
        "ask": [a for _, _, a in ticks],
        "bid_vol": [1.0] * len(ticks),
        "ask_vol": [1.0] * len(ticks),
    })
    pq.write_table(table, base / "fixture.parquet")


def flat_ticks(mid, minutes, per_minute=4, spread=0.0001, start=T0):
    ticks = []
    for m in range(minutes):
        for s in range(per_minute):
            ts = start + timedelta(minutes=m, seconds=s * 15)
            ticks.append((ts, mid - spread / 2, mid + spread / 2))
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


class TestOrderFills:
    def test_market_fills_next_minute_first_tick(self, env):
        tmp, make = env
        write_ticks(tmp, "EUR_USD", flat_ticks(1.1000, 3))
        conn, acct, sim = make()
        sim.step()
        oid, err = sim.place_order("EUR_USD", "buy", 1)
        assert err is None
        sim.step()
        trades = ledger.get_open_trades(conn, acct)
        assert len(trades) == 1
        assert trades[0]["entry_price"] == pytest.approx(1.10005)  # ask

    def test_limit_fills_only_when_crossed(self, env):
        tmp, make = env
        # price walks down through 1.0995 in minute 2
        ticks = flat_ticks(1.1000, 1)
        ticks += [(T0 + timedelta(minutes=1, seconds=s * 15),
                   1.1000 - s * 0.0002, 1.1001 - s * 0.0002) for s in range(4)]
        write_ticks(tmp, "EUR_USD", ticks)
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1, "limit", limit_price=1.0996)
        sim.step()
        trades = ledger.get_open_trades(conn, acct)
        assert len(trades) == 1
        assert trades[0]["entry_price"] <= 1.0996

    def test_stop_loss_exits_at_crossing_tick(self, env):
        tmp, make = env
        ticks = flat_ticks(1.1000, 1)
        # minute 2: collapse through the stop
        ticks += [(T0 + timedelta(minutes=1, seconds=s * 15),
                   1.1000 - s * 0.0010, 1.1001 - s * 0.0010) for s in range(4)]
        write_ticks(tmp, "EUR_USD", ticks)
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1, stop_loss=1.0985)
        sim.step()
        closed = conn.execute(
            "SELECT * FROM trades WHERE exit_time IS NOT NULL").fetchall()
        assert len(closed) == 1
        assert closed[0]["exit_price"] <= 1.0985

    def test_position_cap_rejected(self, env):
        tmp, make = env
        write_ticks(tmp, "EUR_USD", flat_ticks(1.1000, 2))
        conn, acct, sim = make()
        sim.step()
        _, err = sim.place_order("EUR_USD", "buy", 1.5)
        assert err == "position size cap"


class TestRuleBreachOnTick:
    def test_daily_loss_fails_mid_minute(self, env):
        tmp, make = env
        # long 1 lot from ~1.1000; price gaps down 600 pips inside minute 3
        ticks = flat_ticks(1.1000, 2)
        ticks += [(T0 + timedelta(minutes=2, seconds=0), 1.09995, 1.10005),
                  (T0 + timedelta(minutes=2, seconds=15), 1.03995, 1.04005),
                  (T0 + timedelta(minutes=2, seconds=30), 1.03995, 1.04005)]
        write_ticks(tmp, "EUR_USD", ticks)
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1)
        sim.step()
        sim.step()
        assert sim.status == "failed"
        assert ledger.get_account(conn, acct)["status"] == "failed"
        v = conn.execute("SELECT rule_name FROM rule_violations").fetchone()
        assert v["rule_name"] in ("daily_loss", "max_drawdown")
        # positions force-closed
        assert ledger.get_open_trades(conn, acct) == []


class TestWrappers:
    def test_llm_toolbox_round_trip(self, env):
        tmp, make = env
        write_ticks(tmp, "EUR_USD", flat_ticks(1.1000, 5))
        conn, acct, sim = make()
        tools = LLMToolbox(sim)
        tools.advance_time()
        r = tools.place_order("EUR_USD", "buy", 0.5)
        assert r["ok"]
        obs = tools.advance_time()
        assert len(obs["open_trades"]) == 1
        pos = tools.get_open_positions()["positions"]
        tools.close_position(pos[0]["trade_id"])
        obs = tools.advance_time()
        assert obs["open_trades"] == []
        assert tools.get_account_state()["status"] == "active"

    def test_rl_env_step_and_reward_shape(self, env):
        tmp, make = env
        write_ticks(tmp, "EUR_USD", flat_ticks(1.1000, 10))
        gym_env = PropFirmEnv(lambda: make())
        obs, _ = gym_env.reset()
        assert obs.shape == (5,)
        obs, reward, terminated, truncated, _ = gym_env.step(BUY)
        assert -101 <= reward <= 101
        assert not truncated
        while not terminated:
            obs, reward, terminated, _, _ = gym_env.step(HOLD)
        # data exhausted, account still active: no -100 slap
        assert reward > -100
