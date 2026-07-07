"""Training harness: runs record their seed, same seed reproduces the same
outcome, different agents face identical market draws per seed."""

import pytest

from engine import ledger
from scripts.run_training import SEEDS, run_one


def test_seed_set_locked():
    assert len(SEEDS) == 20
    assert len(set(SEEDS)) == 20


@pytest.fixture
def conn(tmp_path):
    c = ledger.connect(":memory:")
    ledger.init_db(c)
    return c


def test_run_records_seed_and_outcome(conn, tmp_path):
    status = run_one(conn, "momentum", "EUR_USD", days=1,
                     seed=SEEDS[0], data_dir=tmp_path)
    row = conn.execute("SELECT * FROM runs").fetchone()
    assert row["seed"] == SEEDS[0]
    assert row["data_source"] == "synthetic"
    assert row["finished_at"] is not None
    assert row["final_status"] == status
    # snapshots exist for the leaderboard
    n = conn.execute("SELECT COUNT(*) AS n FROM equity_snapshots").fetchone()["n"]
    assert n > 0


def test_same_seed_same_market_data(conn, tmp_path):
    run_one(conn, "momentum", "EUR_USD", days=1, seed=SEEDS[1], data_dir=tmp_path)
    b1 = conn.execute(
        "SELECT current_balance FROM accounts WHERE account_id = 1").fetchone()[0]
    run_one(conn, "momentum", "EUR_USD", days=1, seed=SEEDS[1], data_dir=tmp_path)
    b2 = conn.execute(
        "SELECT current_balance FROM accounts WHERE account_id = 2").fetchone()[0]
    assert b1 == pytest.approx(b2)  # identical draw + deterministic policy
