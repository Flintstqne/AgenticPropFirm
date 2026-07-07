"""Phase one gate, per AGENTS.md build order: one hand built account that
should fail actually fails, one that should pass actually passes, running
through ledger + matching + rules together."""

import pytest

from engine import ledger, matching, rules
from engine.config import load_contracts

DAILY_LOSS_PCT = 0.05
DRAWDOWN_PCT = 0.10
START = 100_000.0


@pytest.fixture
def conn():
    c = ledger.connect(":memory:")
    ledger.init_db(c)
    yield c
    c.close()


def run_day(conn, account_id, spec, instrument, side, size, entry_mid, exit_mid, day):
    """Open and close one trade, update equity, run rules. Returns worst grade."""
    entry = matching.fill_market(spec, side, entry_mid)
    exit_side = "sell" if side == "buy" else "buy"
    exit_px = matching.fill_market(spec, exit_side, exit_mid)
    tid = ledger.record_trade_open(conn, account_id, instrument, side, size, entry, f"{day}T09:30:00")
    pnl = matching.pnl_usd(spec, instrument, side, size, entry, exit_px)
    ledger.record_trade_close(conn, tid, exit_px, f"{day}T15:30:00", pnl)
    acct = ledger.get_account(conn, account_id)
    ledger.update_equity(conn, account_id, acct["current_balance"])
    return acct["current_balance"]


def grade_account(conn, account_id, start_of_day):
    acct = ledger.get_account(conn, account_id)
    eq = acct["current_equity"]
    daily = rules.daily_loss(eq, start_of_day, DAILY_LOSS_PCT)
    dd = rules.max_drawdown_static(eq, START, DRAWDOWN_PCT)
    return daily, dd


def test_reckless_account_fails_daily_loss(conn):
    """Loses 12 ES points x 10 contracts in one day: -6000 on 100k, past the 5% line."""
    agent = ledger.register_agent(conn, "reckless", "test")
    acct = ledger.open_account(conn, agent, "challenge", START, "2026-01-05T00:00:00")
    contracts = load_contracts()
    run_day(conn, acct, contracts["ES"], "ES", "buy", 10, 5000.0, 4988.0, "2026-01-05")
    daily, _ = grade_account(conn, acct, START)
    assert daily == "fail"
    ledger.record_violation(conn, acct, "daily_loss", "2026-01-05T15:30:00")
    ledger.set_account_status(conn, acct, "failed")
    assert ledger.get_account(conn, acct)["status"] == "failed"


def test_steady_account_passes(conn):
    """Five days, +30 EUR/USD pips per day at 1 lot: +300/day, never near a limit."""
    agent = ledger.register_agent(conn, "steady", "test")
    acct = ledger.open_account(conn, agent, "challenge", START, "2026-01-05T00:00:00")
    spec = load_contracts()["EUR_USD"]
    sod = START
    for i in range(5):
        day = f"2026-01-{5 + i:02d}"
        run_day(conn, acct, spec, "EUR_USD", "buy", 1, 1.1000, 1.1030, day)
        daily, dd = grade_account(conn, acct, sod)
        assert daily == "pass"
        assert dd == "pass"
        sod = ledger.get_account(conn, acct)["current_equity"]
    assert ledger.get_account(conn, acct)["current_balance"] > START
    assert ledger.get_account(conn, acct)["status"] == "active"


def test_slow_bleed_fails_drawdown_not_daily(conn):
    """Loses 3% a day for four days: never breaches the 5% daily line,
    breaches the 10% static drawdown line on day four."""
    agent = ledger.register_agent(conn, "bleeder", "test")
    acct = ledger.open_account(conn, agent, "challenge", START, "2026-01-05T00:00:00")
    spec = load_contracts()["ES"]
    sod = START
    failed_on = None
    for i in range(4):
        day = f"2026-01-{5 + i:02d}"
        # ~3% of current sod in ES points at 5 contracts: points = sod*0.03 / (50*5)
        pts = sod * 0.03 / (50 * 5)
        run_day(conn, acct, spec, "ES", "buy", 5, 5000.0, 5000.0 - pts, day)
        daily, dd = grade_account(conn, acct, sod)
        assert daily != "fail"
        if dd == "fail":
            failed_on = day
            break
        sod = ledger.get_account(conn, acct)["current_equity"]
    assert failed_on == "2026-01-08"
