import pytest

from engine import ledger, phases
from engine.config import load_phases

PHASES = load_phases()
CH = PHASES["challenge"]
START = 100_000.0
T0 = "2026-01-05T00:00:00"


class TestEvaluatePhase:
    def test_below_target_continues(self):
        # challenge target 8%: 107k is not there yet
        assert phases.evaluate_phase(CH, START, 107_000, 10, [700] * 10) == "continue"

    def test_target_and_days_met_advances(self):
        assert phases.evaluate_phase(CH, START, 108_000, 5, [1600] * 5) == "advance"

    def test_target_met_but_days_short_continues(self):
        assert phases.evaluate_phase(CH, START, 108_000, 3, [2700, 2700, 2600]) == "continue"

    def test_target_met_but_inconsistent_fails(self):
        # one day carries 87% of profit, cap 30%
        assert phases.evaluate_phase(CH, START, 108_000, 5, [7000, 250, 250, 250, 250]) == "fail"

    def test_funded_never_advances(self):
        assert phases.evaluate_phase(PHASES["funded"], START, 150_000, 100, [500] * 100) == "continue"


class TestAdvanceAccount:
    @pytest.fixture
    def conn(self):
        c = ledger.connect(":memory:")
        ledger.init_db(c)
        yield c
        c.close()

    def test_advance_resets_balance(self, conn):
        agent = ledger.register_agent(conn, "a", "rl")
        acct = ledger.open_account(conn, agent, "challenge", START, T0)
        # simulate an 8% gain
        conn.execute("UPDATE accounts SET current_balance = 108000, current_equity = 108000")
        conn.commit()
        nxt = phases.advance_account(conn, acct, PHASES, "2026-01-12T17:00:00")
        assert nxt == "verification"
        row = ledger.get_account(conn, acct)
        assert row["phase"] == "verification"
        assert row["current_balance"] == START
        assert row["current_equity"] == START
        assert row["phase_started_at"] == "2026-01-12T17:00:00"

    def test_chain_to_funded_then_stops(self, conn):
        agent = ledger.register_agent(conn, "a", "rl")
        acct = ledger.open_account(conn, agent, "challenge", START, T0)
        assert phases.advance_account(conn, acct, PHASES, T0) == "verification"
        assert phases.advance_account(conn, acct, PHASES, T0) == "funded"
        assert phases.advance_account(conn, acct, PHASES, T0) is None
