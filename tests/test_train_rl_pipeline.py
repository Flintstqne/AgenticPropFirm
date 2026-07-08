"""Window rotation and training-progress logging in scripts/train_rl.py:
episodes see different seeded synthetic windows, not one repeated path, and
a short training run writes real rows to training_runs / training_metrics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import calibration, ledger  # noqa: E402
from scripts.train_rl import (TrainingProgressCallback, ensure_synthetic_windows,  # noqa: E402
                              make_env)

PARAMS = {
    "omega": 1e-9, "alpha": 0.08, "beta": 0.90,
    "p_stay_trend": 0.98, "p_stay_range": 0.98,
    "trend_mu": 3e-5, "revert_kappa": 0.05,
    "jump_prob": 0.001, "jump_scale": 6.0,
    "tick_density": [4] * 24,
}


@pytest.fixture
def calibrated(tmp_path):
    calibration.save_params(dict(PARAMS, instrument="EUR_USD", n_minutes_fit=0),
                            data_dir=tmp_path)
    return tmp_path


class TestWindowRotation:
    def test_generates_n_windows(self, calibrated):
        sources = ensure_synthetic_windows("EUR_USD", days=1, data_dir=calibrated,
                                           base_seed=1, n_windows=3)
        assert len(sources) == 3
        assert len(set(sources)) == 3  # every window is a distinct source path
        for s in sources:
            files = list((calibrated / s / "EUR_USD").glob("*.parquet"))
            assert len(files) == 1

    def test_episodes_round_robin_through_windows(self, calibrated):
        # every window starts at the same clock time but a different seed,
        # so the price path (not the timestamp) is what has to differ
        build = make_env("EUR_USD", days=1, seed=7)
        seen = set()
        for _ in range(6):
            conn, acct, sim = build()
            _, ticks = next(iter(sim.streams["EUR_USD"]))
            seen.add(round(ticks[0][1], 6))  # first tick's bid price
            conn.close()
        # with 6 windows and 6 episodes, every window should surface at least once
        assert len(seen) >= 2  # distinct opening prices prove distinct data, not one path


class TestProgressLogging:
    @pytest.fixture
    def db(self, tmp_path):
        conn = ledger.connect(str(tmp_path / "test.sqlite"))
        ledger.init_db(conn)
        return conn

    def test_run_and_metrics_rows_written(self, calibrated, db):
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from agents.rl_env import FlattenedActionEnv, PropFirmEnv

        # 1 day = 1440 minutes per episode; enough total steps to guarantee
        # at least one full episode completes, so ep_rew_mean is real, not None
        build = make_env("EUR_USD", days=1, seed=3)
        env = Monitor(FlattenedActionEnv(PropFirmEnv(build)))

        cur = db.execute(
            """INSERT INTO training_runs
               (instrument, algo, seed, total_timesteps, status, started_at)
               VALUES ('EUR_USD', 'PPO', 3, 3072, 'running', '2026-01-01T00:00:00')""")
        db.commit()
        run_id = cur.lastrowid

        model = PPO("MlpPolicy", env, n_steps=1536, batch_size=256, n_epochs=1,
                   verbose=0, device="cpu")
        model.learn(total_timesteps=3072, callback=TrainingProgressCallback(db, run_id))

        db.execute("UPDATE training_runs SET status = 'finished' WHERE training_run_id = ?",
                  (run_id,))
        db.commit()

        run = db.execute("SELECT * FROM training_runs WHERE training_run_id = ?",
                         (run_id,)).fetchone()
        assert run["status"] == "finished"
        metrics = db.execute(
            "SELECT * FROM training_metrics WHERE training_run_id = ?", (run_id,)).fetchall()
        assert len(metrics) >= 1
        assert metrics[0]["timesteps"] > 0
        assert metrics[-1]["ep_rew_mean"] is not None  # a full episode has completed by now
        assert metrics[-1]["fps"] > 0
