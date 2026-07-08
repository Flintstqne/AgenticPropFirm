"""Gymnasium-style reinforcement learning wrapper around the shared
simulation core. Same ledger, matching, and rules as the LLM wrapper.

Multi-instrument, variable size: one action slot per instrument the
Simulator was built with. Each slot picks hold/buy/sell/close and a size
fraction (0, 1] of that instrument's position cap (1 standard lot forex,
2 contracts futures, from config/phases.yaml position_limits) — an agent
never has to know the raw cap numbers, only how hard to press.

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

# Flat equity (no open position, no realized PnL this window) has zero
# variance, so the raw risk-adjusted ratio is undefined and used to return
# a bare 0.0. That makes sitting at HOLD forever a free, zero-risk
# equilibrium: trading is noisy and briefly costs spread/commission the
# moment a position opens, so early in training a policy gradient pushes
# steadily toward HOLD and never back, converging to a no-trade policy
# that never earns the terminal reward and never risks the terminal
# penalty either. A small constant cost for producing no signal at all
# removes that free equilibrium without meaningfully perturbing the reward
# scale reward design calls for (still small relative to the +-100
# terminal reward, which stays dominant).
INACTION_COST = -0.3


class PropFirmEnv(gym.Env):
    """Observation is a flat vector: [mid, position(-1/0/1)] per instrument,
    in instrument order, followed by [equity/start, dist_daily, dist_dd].
    Action is a Dict: "action" MultiDiscrete([4] * n_instruments), "size"
    Box(0, 1, shape=(n_instruments,)) giving the fraction of that
    instrument's position cap to trade on a buy/sell slot."""

    metadata = {"render_modes": []}

    def __init__(self, make_sim):
        """make_sim: zero-arg callable returning (conn, account_id, Simulator).
        A callable so reset() can rebuild a fresh account and data stream.

        Builds one Simulator eagerly to learn the instrument list and
        declare action_space / observation_space in __init__, as Gymnasium
        wrappers and SB3's vec-env plumbing expect both spaces to exist
        before the first reset() call."""
        super().__init__()
        self.make_sim = make_sim
        self.conn, self.account_id, self.sim = make_sim()
        self._pending_sim = (self.conn, self.account_id, self.sim)
        self.instruments = self.sim.instruments
        n = len(self.instruments)
        self.action_space = spaces.Dict({
            "action": spaces.MultiDiscrete([4] * n),
            "size": spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float64),
        })
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * n + 3,),
                                            dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._pending_sim is not None:
            self.conn, self.account_id, self.sim = self._pending_sim
            self._pending_sim = None
        else:
            self.conn, self.account_id, self.sim = self.make_sim()
        self.equity_track = []
        obs = self.sim.step()  # prime: first minute of data, no action
        return self._vector(obs), {}

    def step(self, action):
        sim = self.sim
        acts, sizes = action["action"], action["size"]
        for i, inst in enumerate(self.instruments):
            a, frac = acts[i], float(np.clip(sizes[i], 0.0, 1.0))
            if a == BUY:
                sim.place_order(inst, "buy", self._sized(inst, frac))
            elif a == SELL:
                sim.place_order(inst, "sell", self._sized(inst, frac))
            elif a == CLOSE:
                for t in ledger.get_open_trades(self.conn, self.account_id):
                    if t["instrument"] == inst:
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

    def _sized(self, instrument, frac):
        """frac in (0, 1] of the instrument's position cap; a floor keeps a
        near-zero fraction from rounding to a zero-size, no-op order."""
        spec = self.sim.contracts[instrument]
        limits = self.sim.limits
        cap = (limits["max_forex_lots_per_instrument"] if spec["type"] == "forex"
              else limits["max_futures_contracts_per_instrument"])
        return max(cap * 0.05, cap * frac)

    def _step_reward(self):
        """Rolling risk-adjusted return, clipped to [-1, 1]. Holding no
        position at all costs INACTION_COST -- see the module-level
        comment.

        This checks open-position state directly rather than inferring
        "inactive" from zero variance in the reward window. An earlier
        version used sd == 0 as the inaction signal, which happened to be
        true only when flat (any open position's unrealized PnL moves
        every tick against real data, so variance is essentially never
        exactly zero once a position exists). That accidentally made
        holding *any* position, regardless of quality, a way to permanently
        dodge the inaction penalty -- confirmed empirically: a trained
        policy opened positions and then never closed either one. Tying
        the cost to the literal fact "no open position" removes that
        loophole: closing and going flat costs exactly what never having
        traded costs, no more, no less, so there is no reward advantage to
        hoarding a bad position purely to avoid re-entering the flat state.
        """
        self.equity_track.append(self.sim.equity)
        window = self.equity_track[-REWARD_WINDOW:]
        if len(window) < 2:
            return 0.0
        if not ledger.get_open_trades(self.conn, self.account_id):
            return INACTION_COST
        rets = [(b - a) / a for a, b in zip(window, window[1:])]
        mean = statistics.fmean(rets)
        sd = statistics.pstdev(rets)
        if sd == 0:
            return 0.0  # position open but this window happened to be flat
        return float(np.clip(mean / sd, -1.0, 1.0))

    def _vector(self, obs):
        return vector_observation(obs, self.sim, self.conn, self.account_id, self.instruments)


def vector_observation(obs, sim, conn, account_id, instruments):
    """[mid, position(-1/0/1)] per instrument, in instrument order, followed
    by [equity/start, dist_daily, dist_dd]. Standalone so a trained model can
    be driven outside PropFirmEnv too (see scripts/run_training.py's ppo
    policy), without duplicating this encoding."""
    n = len(instruments)
    if obs is None:
        return np.zeros(2 * n + 3)
    open_by_inst = {}
    for t in obs["open_trades"]:
        open_by_inst.setdefault(t["instrument"], t)
    per_inst = []
    for inst in instruments:
        mid = obs["prices"].get(inst, 0.0)
        t = open_by_inst.get(inst)
        pos = 0.0 if t is None else (1.0 if t["side"] == "buy" else -1.0)
        per_inst.extend([mid, pos])

    acct = ledger.get_account(conn, account_id)
    cfg = sim.phases_cfg[acct["phase"]]
    daily_limit = sim.sod_balance * cfg["daily_loss_pct"]
    dd_limit = sim.starting_balance * cfg["max_drawdown_pct"]
    dist_daily = 1 - (sim.sod_balance - sim.equity) / daily_limit
    dist_dd = 1 - (sim.starting_balance - sim.equity) / dd_limit
    per_inst.extend([sim.equity / sim.starting_balance, dist_daily, dist_dd])
    return np.array(per_inst, dtype=np.float64)


N_SIZE_BINS = 5  # size fractions: 0.2, 0.4, 0.6, 0.8, 1.0 of the position cap


class FlattenedActionEnv(gym.ActionWrapper):
    """Adapts PropFirmEnv's Dict action space to a single MultiDiscrete, since
    Stable-Baselines3 policies support Box, Discrete, MultiDiscrete, and
    MultiBinary action spaces but not Dict. Per instrument: one categorical
    slot over 4 values picks hold/buy/sell/close, a second categorical slot
    over N_SIZE_BINS values picks the size fraction.

    A first version of this wrapper used a continuous Box(-1, 1) with the
    pick decoded by quartile. That failed outright: SB3's default
    continuous-action PPO policy (use_sde=False, the standard setting) is a
    plain unbounded Gaussian, clipped into the box bounds by SB3 itself, not
    squashed through tanh (squash_output only applies when use_sde=True).
    Once the Gaussian's mean drifts past a bound, the clip has zero gradient
    out there, so the mean gets permanently stuck -- confirmed empirically
    across two full training runs where the deterministic policy sat frozen
    at exactly -1.0 (HOLD) even as the stochastic std kept climbing under a
    stronger entropy bonus. A categorical distribution over discrete actions
    has no such dead zone: softmax has gradient everywhere, so there is
    nothing here for the policy to get stuck against.
    """

    def __init__(self, env):
        super().__init__(env)
        n = len(env.unwrapped.instruments)
        self.n = n
        self.action_space = spaces.MultiDiscrete([4, N_SIZE_BINS] * n)

    def action(self, action):
        return decode_multidiscrete_action(action, self.n)


def decode_multidiscrete_action(action, n):
    """[pick_0, size_bin_0, pick_1, size_bin_1, ...] -> the Dict PropFirmEnv
    natively expects. Standalone for the same reason as vector_observation:
    a trained model driven outside FlattenedActionEnv (run_training.py's ppo
    policy) needs the identical decoding, not a second copy of it."""
    action = np.asarray(action, dtype=np.int64).reshape(n, 2)
    acts = action[:, 0]
    sizes = (action[:, 1] + 1) / N_SIZE_BINS  # bin 0 -> 0.2, ..., bin 4 -> 1.0
    return {"action": acts, "size": sizes}
