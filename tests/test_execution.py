"""Phase five execution realism: spread widening, slippage, partial fills,
commission, swap, margin — unit level plus simulator integration."""

from datetime import datetime, timedelta, timezone

import pytest

from engine import ledger, matching
from engine.config import load_contracts
from engine.simulator import Simulator
from tests.test_simulator import flat_ticks, write_ticks, T0, START

CONTRACTS = load_contracts()
FX = CONTRACTS["EUR_USD"]
ES = CONTRACTS["ES"]


class TestSpreadMultiplier:
    def test_forex_tight_in_overlap(self):
        assert matching.spread_multiplier(FX, 2, 1.0) == 1.0

    def test_forex_wider_single_session(self):
        assert matching.spread_multiplier(FX, 1, 1.0) == 1.5

    def test_forex_widest_quiet_hours(self):
        assert matching.spread_multiplier(FX, 0, 1.0) == 3.0

    def test_volatility_widens(self):
        assert matching.spread_multiplier(FX, 2, 2.0) == 2.0

    def test_volatility_capped(self):
        assert matching.spread_multiplier(FX, 2, 50.0) == 3.0

    def test_futures_overnight(self):
        assert matching.spread_multiplier(ES, 0, 1.0) == 2.0


class TestSlippageAndPartials:
    def test_small_order_barely_slips(self):
        small = matching.slippage(FX, 0.1, 100)
        large = matching.slippage(FX, 1.0, 100)
        assert large > small > 0

    def test_split_fills_large_order(self):
        fills = matching.split_fills(1.0, 2.0)  # chunk 0.2 -> 5 fills
        assert len(fills) == 5
        assert sum(fills) == pytest.approx(1.0)

    def test_small_order_single_fill(self):
        assert matching.split_fills(0.1, 100) == [0.1]

    def test_realistic_fill_worse_than_naive(self):
        naive = matching.fill_market(FX, "buy", 1.1000)
        px, n = matching.fill_realistic(FX, "buy", 1.1000, 1.0,
                                        active_sessions=1, vol_ratio=1.5,
                                        liquidity=4)
        assert px > naive
        assert n >= 2  # split against thin liquidity

    def test_sell_side_symmetric(self):
        px, _ = matching.fill_realistic(FX, "sell", 1.1000, 1.0, 2, 1.0, 100)
        assert px < 1.1000


class TestCosts:
    def test_commission(self):
        assert matching.commission_usd(FX, 1.0) == 3.50
        assert matching.commission_usd(ES, 2) == 4.50

    def test_swap_long_negative_short_positive(self):
        assert matching.swap_usd(FX, "buy", 1.0) == -7.0
        assert matching.swap_usd(FX, "sell", 1.0) == 2.0

    def test_futures_no_swap(self):
        assert matching.swap_usd(ES, "buy", 2) == 0.0

    def test_margin_futures_per_contract(self):
        assert matching.margin_required(ES, "ES", 2, 5000.0) == 26_400

    def test_margin_forex_notional_over_leverage(self):
        m = matching.margin_required(FX, "EUR_USD", 1.0, 1.10)
        assert m == pytest.approx(110_000 / 30)


@pytest.fixture
def env(tmp_path):
    conn = ledger.connect(":memory:")
    ledger.init_db(conn)
    agent = ledger.register_agent(conn, "t", "test")
    acct = ledger.open_account(conn, agent, "challenge", START, T0.isoformat())

    def make(instrument="EUR_USD"):
        return conn, acct, Simulator(conn, acct, [instrument], data_dir=tmp_path)
    return tmp_path, make


class TestSimulatorIntegration:
    def test_commissions_hit_balance(self, env):
        tmp, make = env
        write_ticks(tmp, "EUR_USD", flat_ticks(1.1000, 4))
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1)
        sim.step()
        for t in ledger.get_open_trades(conn, acct):
            sim.close_position(t["trade_id"])
        sim.step()
        bal = ledger.get_account(conn, acct)["current_balance"]
        trade = conn.execute("SELECT * FROM trades").fetchone()
        assert trade["commission"] == 7.0  # both sides
        # flat price: loss = spread cost + commission
        assert bal < START - 7.0 + 1e-6

    def test_notional_cap_binds_before_margin(self, env):
        # with the default 5x notional cap, exposure rejects before margin:
        # futures margins sit near 5% of notional, far under the 20% the cap implies
        tmp, make = env
        write_ticks(tmp, "ES", flat_ticks(5000.0, 3, spread=0.25))
        conn, acct, sim = make("ES")
        conn.execute("UPDATE accounts SET current_balance = 20000, current_equity = 20000")
        conn.commit()
        sim.equity = 20_000
        sim.step()
        _, err = sim.place_order("ES", "buy", 2)
        assert err == "total notional cap"

    def test_margin_rejection_when_cap_loosened(self, env):
        from engine.config import load_phases
        tmp, make = env
        write_ticks(tmp, "ES", flat_ticks(5000.0, 3, spread=0.25))
        conn = ledger.connect(":memory:")
        ledger.init_db(conn)
        agent = ledger.register_agent(conn, "m", "test")
        acct = ledger.open_account(conn, agent, "challenge", 20_000, T0.isoformat())
        cfg = load_phases()
        cfg["position_limits"] = dict(cfg["position_limits"], max_total_notional_multiple=1000)
        sim = Simulator(conn, acct, ["ES"], data_dir=tmp, phases_cfg=cfg)
        sim.step()
        _, err = sim.place_order("ES", "buy", 2)  # 26.4k margin > 20k equity
        assert err == "insufficient margin"

    def test_swap_charged_at_rollover(self, env):
        tmp, make = env
        # ticks spanning the 17:00 ET rollover (22:00 UTC in January)
        day1 = datetime(2026, 1, 5, 21, 58, tzinfo=timezone.utc)
        ticks = flat_ticks(1.1000, 2, start=day1)
        ticks += flat_ticks(1.1000, 2, start=day1 + timedelta(minutes=4))  # past 22:00
        write_ticks(tmp, "EUR_USD", ticks)
        conn, acct, sim = make()
        sim.step()
        sim.place_order("EUR_USD", "buy", 1)
        while sim.step() is not None:
            pass
        trade = conn.execute("SELECT * FROM trades").fetchone()
        assert trade["swap"] == pytest.approx(7.0)  # long pays 7/night
