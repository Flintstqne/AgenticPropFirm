import pytest

from engine import ledger

T0 = "2026-01-05T09:30:00-05:00"
T1 = "2026-01-05T10:30:00-05:00"


@pytest.fixture
def conn():
    c = ledger.connect(":memory:")
    ledger.init_db(c)
    yield c
    c.close()


@pytest.fixture
def account(conn):
    agent_id = ledger.register_agent(conn, "test-agent", "rl")
    return ledger.open_account(conn, agent_id, "challenge", 100_000.0, T0)


class TestOpenAccount:
    def test_starts_active_with_balance(self, conn, account):
        acct = ledger.get_account(conn, account)
        assert acct["status"] == "active"
        assert acct["current_balance"] == 100_000.0
        assert acct["current_equity"] == 100_000.0
        assert acct["phase"] == "challenge"


class TestTrades:
    def test_close_updates_balance_net_of_costs(self, conn, account):
        tid = ledger.record_trade_open(
            conn, account, "ES", "buy", 1, 5000.0, T0, commission=4.0, swap=1.0)
        ledger.record_trade_close(conn, tid, 5010.0, T1, realized_pnl=500.0)
        acct = ledger.get_account(conn, account)
        assert acct["current_balance"] == 100_000.0 + 500.0 - 4.0 - 1.0

    def test_losing_trade_reduces_balance(self, conn, account):
        tid = ledger.record_trade_open(conn, account, "EUR_USD", "sell", 1, 1.10, T0)
        ledger.record_trade_close(conn, tid, 1.11, T1, realized_pnl=-1000.0)
        assert ledger.get_account(conn, account)["current_balance"] == 99_000.0

    def test_open_trades_query(self, conn, account):
        t1 = ledger.record_trade_open(conn, account, "ES", "buy", 1, 5000.0, T0)
        ledger.record_trade_open(conn, account, "NQ", "buy", 1, 18000.0, T0)
        ledger.record_trade_close(conn, t1, 5001.0, T1, 50.0)
        open_trades = ledger.get_open_trades(conn, account)
        assert len(open_trades) == 1
        assert open_trades[0]["instrument"] == "NQ"


class TestEquityAndSnapshots:
    def test_update_and_snapshot(self, conn, account):
        ledger.update_equity(conn, account, 101_234.5)
        ledger.snapshot_equity(conn, account, T1)
        row = conn.execute("SELECT * FROM equity_snapshots").fetchone()
        assert row["equity"] == 101_234.5
        assert row["balance"] == 100_000.0
        assert row["timestamp"] == T1


class TestViolationsAndStatus:
    def test_violation_and_fail_status(self, conn, account):
        ledger.record_violation(conn, account, "daily_loss", T1, "equity 94000 below 95000")
        ledger.set_account_status(conn, account, "failed")
        acct = ledger.get_account(conn, account)
        assert acct["status"] == "failed"
        v = conn.execute("SELECT * FROM rule_violations").fetchone()
        assert v["rule_name"] == "daily_loss"

    def test_phase_progression(self, conn, account):
        ledger.set_account_phase(conn, account, "verification", T1)
        acct = ledger.get_account(conn, account)
        assert acct["phase"] == "verification"
        assert acct["phase_started_at"] == T1
