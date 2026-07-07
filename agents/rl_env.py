"""Gymnasium-style reinforcement learning wrapper around the shared
simulation core. Same ledger, matching, and rules as the LLM wrapper.

Version one scope: one instrument per environment, fixed order size
(1 lot forex / 1 contract futures), discrete actions.
# ponytail: single instrument + fixed size; multi-instrument action space when an agent needs it

Reward, per AGENTS.md: per-step risk-adjusted return on a rolling window
scaled to [-1, 1], plus +100 for passing a phase and -100 for a rule
violation that fails the account. The phase reward dominates.
"""

import statistics

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine import ledger
from engine.simulator import Simulator

HOLD, BUY, SELL, CLOSE = 0, 1, 2, 3
REWARD_WINDOW = 60  # minutes


class PropFirmEnv(gym.Env):
    """Observation: [mid, position(-1/0/1), equity/start, dist_daily, dist_dd]
    distances are fraction of the limit remaining, 0 = on the line."""

    metadata = {"render_modes": []}

    def __init__(self, make_sim, order_size=1.0):
        """make_sim: zero-arg callable returning (conn, account_id, Simulator).
        A callable so reset() can rebuild a fresh account and data stream."""
        super().__init__()
        self.make_sim = make_sim
        self.order_size = order_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float64)
        self.sim = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.conn, self.account_id, self.sim = self.make_sim()
        self.instrument = self.sim.instruments[0]
        self.equity_track = []
        obs = self.sim.step()  # prime: first minute of data, no action
        return self._vector(obs), {}

    def step(self, action):
        sim = self.sim
        if action == BUY:
            sim.place_order(self.instrument, "buy", self.order_size)
        elif action == SELL:
            sim.place_order(self.instrument, "sell", self.order_size)
        elif action == CLOSE:
            for t in ledger.get_open_trades(self.conn, self.account_id):
                sim.close_position(t["trade_id"])

        prev_status = sim.status
        obs = sim.step()
        terminated = sim.done
        reward = self._step_reward()
        if sim.status == "failed":
            reward -= 100.0
        elif sim.status == "passed" and prev_status != "passed":
            reward += 100.0
        return self._vector(obs), reward, terminated, False, {}

    def _step_reward(self):
        """Rolling risk-adjusted return, clipped to [-1, 1]."""
        self.equity_track.append(self.sim.equity)
        window = self.equity_track[-REWARD_WINDOW:]
        if len(window) < 2:
            return 0.0
        rets = [(b - a) / a for a, b in zip(window, window[1:])]
        mean = statistics.fmean(rets)
        sd = statistics.pstdev(rets)
        if sd == 0:
            return 0.0
        return float(np.clip(mean / sd, -1.0, 1.0))

    def _vector(self, obs):
        if obs is None:
            return np.zeros(5)
        sim = self.sim
        mid = obs["prices"].get(self.instrument, 0.0)
        open_trades = obs["open_trades"]
        pos = 0.0
        if open_trades:
            pos = 1.0 if open_trades[0]["side"] == "buy" else -1.0
        acct = ledger.get_account(self.conn, self.account_id)
        cfg = sim.phases_cfg[acct["phase"]]
        daily_limit = sim.sod_balance * cfg["daily_loss_pct"]
        dd_limit = sim.starting_balance * cfg["max_drawdown_pct"]
        dist_daily = 1 - (sim.sod_balance - sim.equity) / daily_limit
        dist_dd = 1 - (sim.starting_balance - sim.equity) / dd_limit
        return np.array([mid, pos, sim.equity / sim.starting_balance,
                         dist_daily, dist_dd], dtype=np.float64)
