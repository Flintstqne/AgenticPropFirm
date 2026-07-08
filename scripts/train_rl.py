"""Train a PPO agent (stable-baselines3) against PropFirmEnv over calibrated
synthetic data. First real trained agent for the leaderboard, replacing the
random/momentum baselines in scripts/run_training.py.

PPO needs a flat action space, so this wraps PropFirmEnv in
FlattenedActionEnv (agents/rl_env.py): SB3 policies support Box, Discrete,
MultiDiscrete, MultiBinary action spaces, not the Dict PropFirmEnv exposes
natively for readability.

Episodes rotate through several distinct seeded synthetic windows rather
than replaying one fixed stretch of days every time: reusing a single
window across thousands of episodes teaches the policy that window's exact
price path, not a generalizable trading skill. Window selection is a
round-robin over a fixed seeded set, not true randomness, so a run is
reproducible from its --seed.

Progress writes to the shared SQLite file (training_runs / training_metrics,
db/migrations/003_add_training_tables.sql) so the dashboard's Training tab
can show a run live, the same loosely-joined pattern the rest of the system
uses between the Python and Node sides.

Timesteps default low (20k) so a first run finishes in a minute and proves
the pipeline end to end. Real training needs far more: the per-step reward
window alone is 60 minutes, so 20k steps is under two weeks of simulated
trading, nowhere near enough to shape a policy that reliably clears an 8%
target. Raise --timesteps for a real run (see AGENTS.md for the fps-based
time estimate).

Three full 500k-step runs converged to a policy that opened positions but
never closed them, tracing to a state-independent categorical collapse
(one action dominant everywhere, not learned per-state), not a reward bug
-- see AGENTS.md. Blind reward retuning already failed three times, so
--warm-start-episodes behavior-clones the policy against a simple
momentum rule before RL fine-tuning begins, so training starts from a
state-differentiated policy instead of a random init prone to collapsing
onto one action before it ever sees enough experience to differentiate.

Usage:
  venv/bin/python scripts/train_rl.py EUR_USD --timesteps 20000
  venv/bin/python scripts/train_rl.py EUR_USD --timesteps 500000 --evaluate
  venv/bin/python scripts/train_rl.py EUR_USD --timesteps 500000 --warm-start-episodes 30
"""

import argparse
import itertools
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO                       # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor     # noqa: E402
from stable_baselines3.common.utils import safe_mean     # noqa: E402

from agents.rl_env import BUY, CLOSE, HOLD, SELL, FlattenedActionEnv, PropFirmEnv  # noqa: E402
from engine import calibration, ledger, synthetic          # noqa: E402
from engine.config import load_contracts                   # noqa: E402
from engine.simulator import Simulator                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
N_WINDOWS = 6  # distinct seeded market draws an episode can land on


def ensure_synthetic_windows(instrument, days, data_dir, base_seed, n_windows=N_WINDOWS):
    """Generate n_windows distinct calibrated synthetic stretches, each its
    own seed, so training episodes see varied conditions instead of one
    memorizable path. Returns the source string each Simulator should use."""
    try:
        params = calibration.load_params(instrument, data_dir=data_dir)
    except FileNotFoundError:
        raise SystemExit(
            f"no calibration for {instrument}; run calibration.calibrate() "
            f"against downloaded history first (see scripts/download_data.py)")
    spec = load_contracts()[instrument]
    start_price = 1.10 if spec["type"] == "forex" else 5000.0
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)

    sources = []
    for i in range(n_windows):
        # days is part of the cache key: a leftover longer window from a
        # prior --days value must not get reused as a shorter one, since
        # that silently changes episode length out from under training
        tag = f"w{i}_d{days}"
        out = data_dir / "calibrated" / "train_rl" / tag / instrument
        source = f"calibrated/train_rl/{tag}"
        if not (out.exists() and len(list(out.glob("*.parquet"))) == days):
            synthetic.generate_day_files(params, spec, start_price, start, days,
                                         out, seed=base_seed + i * 1000)
        sources.append(source)
    return sources


def make_env(instrument, days, seed):
    """Returns a zero-arg callable PropFirmEnv wants: builds a fresh account
    and Simulator each episode, cycling round-robin through the seeded
    synthetic windows so no single path dominates training."""
    data_dir = ROOT / "data"
    sources = ensure_synthetic_windows(instrument, days, data_dir, seed)
    window_cycle = itertools.cycle(sources)

    def build():
        conn = ledger.connect(":memory:")
        ledger.init_db(conn)
        agent = ledger.register_agent(conn, "ppo", "rl")
        acct = ledger.open_account(conn, agent, "challenge", 100_000.0,
                                   datetime.now(timezone.utc).isoformat())
        sim = Simulator(conn, acct, [instrument], data_dir=data_dir,
                        source=next(window_cycle))
        return conn, acct, sim
    return build


class TrainingProgressCallback(BaseCallback):
    """Writes one training_metrics row per completed rollout to the shared
    SQLite file.

    Reads model.ep_info_buffer directly rather than model.logger.name_to_value:
    _on_rollout_end fires from inside collect_rollouts, before SB3's own
    _dump_logs call writes rollout/ep_rew_mean into the logger, so the
    logger dict is always one rollout stale at this point. ep_info_buffer
    (populated per-step as Monitor-wrapped episodes finish) already has
    whatever episodes have completed so far."""

    def __init__(self, db_conn, training_run_id):
        super().__init__()
        self.db_conn = db_conn
        self.training_run_id = training_run_id
        self._start_time = None

    def _on_training_start(self):
        self._start_time = time.time()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        ep_rew = safe_mean([e["r"] for e in buf]) if buf else None
        ep_len = safe_mean([e["l"] for e in buf]) if buf else None
        elapsed = time.time() - self._start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else None
        self.db_conn.execute(
            """INSERT INTO training_metrics
               (training_run_id, timesteps, ep_rew_mean, ep_len_mean, fps, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (self.training_run_id, self.num_timesteps, ep_rew, ep_len, fps,
             datetime.now(timezone.utc).isoformat()))
        self.db_conn.commit()


def collect_demonstrations(env, episodes, ma_window=20):
    """Run a simple rule-based trader through env, recording (observation,
    action) pairs in FlattenedActionEnv's own MultiDiscrete action space, for
    behavior-cloning pretraining. Operates directly on the vector
    observation (index 0 is the single instrument's mid price, index 1 is
    position) rather than through LLMToolbox, since BC needs actions in the
    exact encoding PPO will fine-tune, not tool calls.

    The rule itself is not the point -- it only has to be state-dependent
    (trades on real price/position signal) so the policy starts pretrained
    against *something* that varies by input, rather than random init that
    is free to collapse onto one output regardless of state.
    """
    observations, actions = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        window = []
        terminated = truncated = False
        while not (terminated or truncated):
            mid, pos = obs[0], obs[1]
            window.append(mid)
            window = window[-ma_window:]
            if len(window) < ma_window:
                pick, size_bin = HOLD, 0
            else:
                ma = sum(window) / len(window)
                if pos == 0 and mid > ma:
                    pick, size_bin = BUY, 2
                elif pos == 0 and mid < ma:
                    pick, size_bin = SELL, 2
                elif pos != 0 and (mid < ma if pos > 0 else mid > ma):
                    pick, size_bin = CLOSE, 0
                else:
                    pick, size_bin = HOLD, 0
            action = np.array([pick, size_bin])
            observations.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


def behavior_clone(model, observations, actions, epochs=5, batch_size=64):
    """Pretrain model.policy via supervised negative log-likelihood against
    demonstrated actions, using SB3's own evaluate_actions so the same
    MultiCategorical distribution PPO trains against is what gets fit here
    -- no separate classifier head to keep in sync."""
    import torch

    device = model.device
    obs_t = torch.as_tensor(observations, device=device)
    act_t = torch.as_tensor(actions, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n = len(obs_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            _, log_prob, _ = model.policy.evaluate_actions(obs_t[idx], act_t[idx])
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f"  BC epoch {epoch + 1}/{epochs}  loss={total_loss / n:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument")
    ap.add_argument("--timesteps", type=int, default=20_000)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evaluate", action="store_true",
                    help="run one episode with the trained policy and print the outcome")
    ap.add_argument("--warm-start-episodes", type=int, default=0,
                    help="behavior-clone against a momentum rule for this many "
                         "episodes before RL fine-tuning (0 disables)")
    args = ap.parse_args()

    env = Monitor(FlattenedActionEnv(PropFirmEnv(
        make_env(args.instrument, args.days, args.seed))))

    # the shared db may already hold accounts/trades from prior simulation
    # runs, so only the training tables (idempotent CREATE ... IF NOT EXISTS)
    # get applied here rather than the full schema
    db_conn = ledger.connect(str(ROOT / "db" / "propfirm.sqlite"))
    db_conn.executescript(
        (ROOT / "db" / "migrations" / "003_add_training_tables.sql").read_text())
    now = datetime.now(timezone.utc).isoformat()
    cur = db_conn.execute(
        """INSERT INTO training_runs
           (instrument, algo, seed, total_timesteps, status, started_at)
           VALUES (?, 'PPO', ?, ?, 'running', ?)""",
        (args.instrument, args.seed, args.timesteps, now))
    db_conn.commit()
    training_run_id = cur.lastrowid

    # ent_coef nonzero: SB3's PPO default is 0.0 (no entropy bonus), which
    # lets the policy collapse to a low-entropy near-deterministic policy
    # early and stop exploring. Combined with the free-HOLD-equilibrium fix
    # in agents/rl_env.py's _step_reward, this keeps exploration pressure
    # on long enough for the policy to actually sample enough trades to
    # learn from, instead of converging on "never trade." 0.01 alone still
    # wasn't enough (see AGENTS.md); 0.05 measurably shifted the *executed*
    # action distribution off HOLD in a short experiment, though the
    # deterministic policy took longer to follow -- still tuning.
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, device="cpu", ent_coef=0.05)

    if args.warm_start_episodes:
        print(f"collecting {args.warm_start_episodes} demonstration episodes...")
        demo_obs, demo_act = collect_demonstrations(env, args.warm_start_episodes)
        print(f"behavior-cloning against {len(demo_obs)} demonstrated steps...")
        behavior_clone(model, demo_obs, demo_act)

    model.learn(total_timesteps=args.timesteps,
               callback=TrainingProgressCallback(db_conn, training_run_id))

    db_conn.execute(
        "UPDATE training_runs SET status = 'finished', finished_at = ? WHERE training_run_id = ?",
        (datetime.now(timezone.utc).isoformat(), training_run_id))
    db_conn.commit()

    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"ppo_{args.instrument}.zip"
    model.save(path)
    print(f"saved {path}")

    if args.evaluate:
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
        sim = env.unwrapped.sim
        print(f"eval episode: status={sim.status} "
              f"final_equity={sim.equity:.2f} (started 100000.00)")


if __name__ == "__main__":
    main()
