"""FlattenedActionEnv wrapper tests, plus a smoke test that PPO can train
a handful of steps against PropFirmEnv without crashing. Not a claim the
policy learns anything at this timestep count -- see scripts/train_rl.py
for the real-training scale."""

import numpy as np
import pytest

from agents.rl_env import BUY, CLOSE, HOLD, INACTION_COST, SELL, FlattenedActionEnv, PropFirmEnv
from engine import ledger
from engine.simulator import Simulator
from tests.test_simulator import START, T0, flat_ticks, write_ticks


@pytest.fixture
def make_env(tmp_path):
    write_ticks(tmp_path, "EUR_USD", flat_ticks(1.1000, 30))

    def build():
        conn = ledger.connect(":memory:")
        ledger.init_db(conn)
        agent = ledger.register_agent(conn, "t", "test")
        acct = ledger.open_account(conn, agent, "challenge", START, T0.isoformat())
        sim = Simulator(conn, acct, ["EUR_USD"], data_dir=tmp_path)
        return conn, acct, sim
    return build


class TestFlattenedActionEnv:
    def test_action_space_is_multidiscrete(self, make_env):
        env = FlattenedActionEnv(PropFirmEnv(make_env))
        assert list(env.action_space.nvec) == [4, 5]  # 1 instrument: pick(4) + size bin(5)

    @pytest.mark.parametrize("pick,expected", [(0, HOLD), (1, BUY), (2, SELL), (3, CLOSE)])
    def test_pick_decoding(self, make_env, pick, expected):
        env = FlattenedActionEnv(PropFirmEnv(make_env))
        decoded = env.action(np.array([pick, 0]))
        assert decoded["action"][0] == expected

    def test_size_bin_decoding(self, make_env):
        env = FlattenedActionEnv(PropFirmEnv(make_env))
        decoded = env.action(np.array([0, 0]))
        assert decoded["size"][0] == pytest.approx(0.2)  # bin 0 -> smallest nonzero size
        decoded = env.action(np.array([0, 4]))
        assert decoded["size"][0] == pytest.approx(1.0)  # bin 4 -> full cap

    def test_step_round_trip(self, make_env):
        env = FlattenedActionEnv(PropFirmEnv(make_env))
        obs, _ = env.reset()
        obs, reward, terminated, truncated, _ = env.step(np.array([BUY, 2]))  # BUY, mid size
        assert obs.shape == env.observation_space.shape
        trades = ledger.get_open_trades(env.unwrapped.conn, env.unwrapped.account_id)
        assert len(trades) == 1


class TestInactionCost:
    """Guards against the free-HOLD-equilibrium bug found in the first real
    500k-step training run: a policy that never trades produced flat
    equity, which the old reward returned as a bare 0.0 -- a safe, riskless
    reward that gave PPO no pressure to ever explore trading again."""

    def test_repeated_hold_costs_inaction_not_zero(self, tmp_path):
        write_ticks(tmp_path, "EUR_USD", flat_ticks(1.1000, 70))

        def build():
            conn = ledger.connect(":memory:")
            ledger.init_db(conn)
            agent = ledger.register_agent(conn, "t", "test")
            acct = ledger.open_account(conn, agent, "challenge", START, T0.isoformat())
            sim = Simulator(conn, acct, ["EUR_USD"], data_dir=tmp_path)
            return conn, acct, sim

        env = FlattenedActionEnv(PropFirmEnv(build))
        env.reset()
        hold = np.array([HOLD, 0])
        rewards = []
        for _ in range(65):
            _, reward, terminated, truncated, _ = env.step(hold)
            rewards.append(reward)
            if terminated or truncated:
                break
        # once the reward window fills (60 minutes) with zero-variance flat
        # equity, every further step should cost INACTION_COST, not 0.0
        assert rewards[-1] == pytest.approx(INACTION_COST)
        assert rewards[-1] < 0.0
        # nowhere near the terminal -100 penalty: this is a small nudge,
        # not a rule violation
        assert rewards[-1] > -1.0

    def test_holding_a_position_escapes_inaction_cost(self, tmp_path):
        """The bug this guards: an earlier version tied INACTION_COST to
        zero reward-window variance, which any open position (real or bad)
        happened to always avoid, making "hold something forever" a free
        way to dodge the penalty meant to discourage never trading at all.
        Opening a position must not, by itself, produce a reward of
        INACTION_COST -- it should be judged on its own risk-adjusted
        return instead."""
        write_ticks(tmp_path, "EUR_USD", flat_ticks(1.1000, 10))

        def build():
            conn = ledger.connect(":memory:")
            ledger.init_db(conn)
            agent = ledger.register_agent(conn, "t", "test")
            acct = ledger.open_account(conn, agent, "challenge", START, T0.isoformat())
            sim = Simulator(conn, acct, ["EUR_USD"], data_dir=tmp_path)
            return conn, acct, sim

        env = FlattenedActionEnv(PropFirmEnv(build))
        env.reset()
        env.step(np.array([BUY, 4]))  # open a position, full size
        # a position is open now; even on a flat-price tick, this must not
        # silently collapse to the same fixed inaction penalty
        _, reward, _, _, _ = env.step(np.array([HOLD, 0]))
        assert reward != pytest.approx(INACTION_COST)

    def test_closing_back_to_flat_costs_same_as_never_trading(self, tmp_path):
        """After closing, the account is flat again and should face exactly
        the same inaction cost as an account that never traded -- no reward
        advantage to having traded once and stopped versus never starting."""
        write_ticks(tmp_path, "EUR_USD", flat_ticks(1.1000, 70))

        def build():
            conn = ledger.connect(":memory:")
            ledger.init_db(conn)
            agent = ledger.register_agent(conn, "t", "test")
            acct = ledger.open_account(conn, agent, "challenge", START, T0.isoformat())
            sim = Simulator(conn, acct, ["EUR_USD"], data_dir=tmp_path)
            return conn, acct, sim

        env = FlattenedActionEnv(PropFirmEnv(build))
        env.reset()
        env.step(np.array([BUY, 4]))
        env.step(np.array([CLOSE, 0]))  # queued; fills next step
        rewards = []
        for _ in range(65):
            _, reward, terminated, truncated, _ = env.step(np.array([HOLD, 0]))
            rewards.append(reward)
            if terminated or truncated:
                break
        assert rewards[-1] == pytest.approx(INACTION_COST)


class TestPPOSmoke:
    def test_short_training_run_completes(self, make_env):
        from stable_baselines3 import PPO
        env = FlattenedActionEnv(PropFirmEnv(make_env))
        model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=1,
                   verbose=0, device="cpu")
        model.learn(total_timesteps=64)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
