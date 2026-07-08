"""Behavior-cloning warm start: collect_demonstrations produces real,
state-dependent (obs, action) pairs, and behavior_clone actually moves the
policy toward imitating them (loss drops, log-prob of demonstrated actions
rises). This is the next lever after three failed reward-tuning attempts
at the same "opens positions, never closes" collapse -- see AGENTS.md."""

import numpy as np
import pytest

from agents.rl_env import BUY, CLOSE, HOLD, SELL, FlattenedActionEnv, PropFirmEnv
from engine import ledger
from engine.simulator import Simulator
from scripts.train_rl import behavior_clone, collect_demonstrations
from tests.test_simulator import START, T0, write_ticks
from datetime import timedelta


def trending_ticks(start_price, minutes, step, start=T0):
    """A clean uptrend so the momentum demonstrator actually crosses its
    moving average and produces BUY/SELL/CLOSE, not just HOLD."""
    ticks = []
    price = start_price
    for m in range(minutes):
        price += step
        ticks.append((start + timedelta(minutes=m), price - 0.00005, price + 0.00005))
    return ticks


@pytest.fixture
def env(tmp_path):
    write_ticks(tmp_path, "EUR_USD", trending_ticks(1.1000, 200, 0.0002))

    def build():
        conn = ledger.connect(":memory:")
        ledger.init_db(conn)
        agent = ledger.register_agent(conn, "t", "test")
        acct = ledger.open_account(conn, agent, "challenge", START, T0.isoformat())
        sim = Simulator(conn, acct, ["EUR_USD"], data_dir=tmp_path)
        return conn, acct, sim
    return FlattenedActionEnv(PropFirmEnv(build))


class TestCollectDemonstrations:
    def test_produces_state_dependent_actions(self, env):
        obs, actions = collect_demonstrations(env, episodes=2)
        assert len(obs) == len(actions) > 0
        picks = actions[:, 0]
        # a clean uptrend must eventually cross the moving average and
        # produce something other than HOLD -- otherwise the demonstrator
        # itself is broken and BC has nothing real to learn from
        assert set(picks) != {HOLD}

    def test_observations_match_env_shape(self, env):
        obs, actions = collect_demonstrations(env, episodes=1)
        assert obs.shape[1] == env.observation_space.shape[0]
        assert actions.shape[1] == 2  # [pick, size_bin]


class TestBehaviorClone:
    def test_loss_decreases_and_fits_demonstrations(self, env):
        from stable_baselines3 import PPO
        import torch

        obs, actions = collect_demonstrations(env, episodes=3)
        model = PPO("MlpPolicy", env, verbose=0, device="cpu", seed=1)

        obs_t = torch.as_tensor(obs, device=model.device)
        act_t = torch.as_tensor(actions, device=model.device)
        _, log_prob_before, _ = model.policy.evaluate_actions(obs_t, act_t)
        loss_before = -log_prob_before.mean().item()

        behavior_clone(model, obs, actions, epochs=10)

        _, log_prob_after, _ = model.policy.evaluate_actions(obs_t, act_t)
        loss_after = -log_prob_after.mean().item()

        # the whole point: after cloning, the policy assigns meaningfully
        # higher probability to the demonstrated actions than before
        assert loss_after < loss_before * 0.7
