# AGENTS.md

Instructions for any coding agent working on this repository. Read this whole file before writing code. Every design decision below comes from a locked planning phase. Follow the decisions as written. Do not substitute a different library, a different schema, or a different folder layout unless a section says the choice stays open.

## Project Overview

This repository holds a personal, self hosted prop firm simulation. The system runs AI trading agents, both reinforcement learning agents and LLM agents using tool calls, through a simulated evaluation program modeled on a real proprietary trading firm. Agents open a challenge account, trade forex and futures instruments against historical or synthetic price data, and get judged against profit targets, drawdown limits, and daily loss caps in real time. Agents that pass move through a verification phase and into a funded phase. A dashboard shows account state, equity curves, and rule status for every agent running in the system.

The owner of this project runs everything locally. No step in this build should introduce a paid service, a hosted database, or a cloud dependency. Every tool named in this document costs nothing to run.

## Ground Rules for Any Agent Working on This Repository

Build in the phase order given near the end of this file. Do not start phase three work before phase one and phase two pass their own tests.

Read every config value from `config/contracts.yaml` and `config/sessions.yaml`. Never hardcode a tick size, a margin figure, a session time, or a spread value inside engine code. A hardcoded number belongs in a config file, not in a Python module.

Write a test for every rule in the rules engine before moving to the next rule. A rule with no test does not count as done.

Keep the Python simulation side and the Node dashboard side loosely joined through the shared SQLite file described later in this document. Do not build a second API layer between them unless a future revision of this file says otherwise.

Prefer plain, direct code over clever abstraction. One person maintains this project. Optimize for a maintainer returning after a break and understanding a file in under a minute, not for extensibility nobody asked for yet.

## Version Control

Full branching rules and commit rules live in `docs/version-control.md`. Read that file before making a first commit on this repository. Two rules from that document matter enough to restate here directly: never push to main or staging directly, every change moves through a feature branch and a pull request, and never commit under an agent identity, every commit carries the user's own git identity and the user runs the actual commit and push from their own terminal.

## Environment and Secrets

Every Python dependency lives in `requirements.txt`, pinned to a minimum version. Add a new dependency to this file the moment you add an import, rather than letting the environment drift from what the file lists.

Keep every API key and secret in a local `.env` file, never committed. `.env.example` at the repository root lists every key name the project expects, with an empty value, as a template. Copy this file to `.env` and fill in real values locally.

`.gitignore` already excludes `venv/`, `node_modules/`, the compiled dashboard client, the SQLite database file, and the raw and calibrated data folders. Large data files and a live database do not belong in git history. The schema file, `db/schema.sql`, stays tracked, since the schema itself counts as source code, while the database it produces does not.

Run `scripts/setup.sh` once on a fresh clone to build the Python virtual environment, install every Node package on both sides of the dashboard, and load the database schema in one pass. Run `scripts/run_tests.sh` to run the full Python test suite through pytest.

## Tech Stack

Python 3 for the simulation engine: the matching engine, the rules engine, the account ledger, the synthetic data generator, and the calibration scripts.

SQLite for transactional data: accounts, trades, rule violations, equity snapshots, agents. One file, no server process.

DuckDB and Parquet for tick level historical and synthetic price data. DuckDB reads Parquet files directly with no server and no setup step.

Python packages: pandas, numpy, arch (GARCH volatility modeling), statsmodels (Markov regime switching), pyarrow (Parquet read and write), duckdb.

Gymnasium for the reinforcement learning agent interface, using its step and reset pattern.

Node.js and Express for the dashboard backend, reading the shared SQLite file through a Node SQLite driver.

React and Vite for the dashboard frontend.

A native WebSocket connection, or the `ws` package, for pushing account and price updates from the Express backend to the React frontend in real time.

pnpm as the Node package manager, chosen over npm for faster installs and lower disk use.

Historical data sources, every one free: Dukascopy for forex tick history, Databento free usage credits for futures history, Nasdaq Data Link free datasets for continuous futures contracts, yfinance as a secondary check against the primary futures source.

## Repository Structure

```
propfirm-sim/
  engine/
    config.py        # loads the yaml config files, nothing else
    matching.py
    rules.py
    ledger.py
    phases.py        # phase progression: challenge -> verification -> funded
    replay.py        # historical tick replay over Parquet through DuckDB
    simulator.py     # two-clock simulation loop shared by both agent wrappers
    synthetic.py
    calibration.py
  agents/
    rl_env.py
    llm_tools.py
  tests/
    test_ledger.py
    test_matching.py
    test_rules.py
    test_synthetic.py
  config/
    contracts.yaml
    sessions.yaml
    phases.yaml      # per-phase targets, loss limits, drawdown mode, caps
  data/
    raw/
    calibrated/
  db/
    propfirm.sqlite
    schema.sql
    migrations/      # numbered SQL migration scripts for live-table changes
  dashboard/
    server/
    client/
  scripts/
    setup.sh
    run_tests.sh
    download_data.py
    run_training.py
  docs/
    dashboard-visual-design.txt
    dashboard-mockup.html
    version-control.md
  .gitignore
  .env.example
  requirements.txt
  README.md
  AGENTS.md
```

Place the design reference documents from the planning phase, the dashboard visual design document and the dashboard mockup, inside `docs/`. Treat both as the source of truth for every frontend styling decision. Do not invent a different color palette, a different type system, or a different layout while building the React client. Match the reference files exactly.

## Instrument Scope

Build support for seven forex pairs and seven futures contracts in version one.

Forex majors: EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD.

Futures majors: E-mini S&P 500 (ES), E-mini Nasdaq 100 (NQ), E-mini Dow (YM), Crude Oil (CL), Gold (GC), 10 Year Treasury Note (ZN), Euro FX futures (6E).

## Contract Specification Table

Treat the figures below as representative starting values. Confirm current margin figures against the exchange contract specification pages before a final calibration run, since margin requirements shift with market volatility over time. Encode every value into `config/contracts.yaml`.

Futures:

| Contract | Tick size | Tick value | Point value | Typical margin | Typical spread |
|---|---|---|---|---|---|
| ES | 0.25 index points | 12.50 USD | 50 USD per point | approx 13,200 USD | 1 tick |
| NQ | 0.25 index points | 5.00 USD | 20 USD per point | approx 18,700 USD | 1 tick |
| YM | 1 point | 5.00 USD | 5 USD per point | approx 8,800 USD | 1 tick |
| CL | 0.01 USD per barrel | 10.00 USD | contract size 1,000 barrels | approx 6,500 USD | 1 to 2 ticks |
| GC | 0.10 USD per ounce | 10.00 USD | contract size 100 ounces | approx 9,900 USD | 1 tick |
| ZN | 1/64 of a point | 15.625 USD | contract size 100,000 USD face value | approx 1,900 USD | 1 tick |
| 6E | 0.00005 | 6.25 USD | contract size 125,000 euros | approx 2,400 USD | 1 tick |

Forex, standard lot equals 100,000 units, mini lot 10,000, micro lot 1,000:

| Pair | Pip size | Pip value at standard lot | Typical spread at peak liquidity |
|---|---|---|---|
| EUR/USD | 0.0001 | approx 10 USD | 0.1 to 1 pip |
| USD/JPY | 0.01 | recalculate against live rate | 0.1 to 1 pip |
| GBP/USD | 0.0001 | approx 10 USD | 0.5 to 1.5 pips |
| USD/CHF | 0.0001 | recalculate against live rate | 1 to 2 pips |
| AUD/USD | 0.0001 | approx 10 USD | 0.5 to 1.5 pips |
| USD/CAD | 0.0001 | recalculate against live rate | 1 to 2 pips |
| NZD/USD | 0.0001 | approx 10 USD | 1 to 2 pips |

Margin formula at 30 to 1 leverage: margin equals notional position size divided by 30. Recalculate margin per trade against the current price for any pair quoting USD as the counter currency, since notional value moves with the live rate.

## Session Calendar

Forex trades from Sunday 5pm Eastern Time through Friday 5pm Eastern Time. Four regional sessions drive liquidity: Sydney (5pm to 2am Eastern), Tokyo (7pm to 4am Eastern), London (3am to 12pm Eastern), New York (8am to 5pm Eastern). The London and New York overlap, 8am to 12pm Eastern, carries the deepest liquidity and the tightest spread of the day.

Futures trade close to twenty three hours a day on CME Globex, with a daily maintenance break near 5pm to 6pm Eastern. Weekly closure runs from Friday afternoon, generally near 5pm Eastern, through Sunday evening reopen, generally near 6pm Eastern.

Use 5pm Eastern as the daily reset time for the rules engine, across every instrument, forex and futures alike, in version one. A single reset time keeps the rules engine simple. A per instrument reset time is a deferred item, listed at the end of this file.

Encode every session window into `config/sessions.yaml`, one entry per instrument, following the format shown in the config section below.

## Simulation Clock and Time Step Design

Run two clocks at once inside the engine.

Give the agent one observation and one action per simulated minute, across every instrument. A one minute cadence keeps training runs fast and still catches a daily loss or drawdown breach with good precision.

Between two agent decisions, feed every tick inside that one minute window into the matching engine. Check every open stop loss order, take profit order, and pending limit order against each tick, so an order triggers at the tick where price actually crosses it, not only at the minute close. Historical replay supplies real ticks. Synthetic generation supplies generated ticks at a similar density, denser during active sessions, thinner during quiet hours.

Run the rules engine checks for daily loss and maximum drawdown on every tick, not only at the one minute boundary. An account that breaches a limit mid minute fails at the moment of the breach.

Store every tick in Parquet files for replay, audit, and calibration. Store one row per minute bar in the SQLite `equity_snapshots` table.

## Rules Engine

Build each rule as an independent function that reads account state and the current trade or tick, and returns pass, warn, or fail.

Daily loss limit: fail the account if equity drops below a set percentage of the start of day balance, reset daily at 5pm Eastern.

Maximum drawdown: support both a static version, measured against the starting balance, and a trailing version, measured against the highest equity the account ever reached. Make the choice between static and trailing a config value per phase, not a hardcoded constant.

Consistency rule: cap how much of total profit can come from a single trading day, in the 20 to 30 percent range, configurable.

Minimum trading days: count a day as active only if the agent placed at least one trade that day.

Position size and leverage limits: cap each forex position at one standard lot per instrument per account, cap each futures position at 2 contracts per instrument per account, and cap total open notional exposure across every open position at 5 times account equity. Apply these caps during phase one and phase two. Revisit the caps for funded accounts once a scaling plan exists, listed as deferred work below.

## Evaluation Phases

Phase one, challenge: profit target 8 to 10 percent of starting balance, maximum daily loss 4 to 5 percent, maximum overall drawdown 8 to 12 percent, minimum trading days 5 to 10. Make every one of these figures a config value, not a hardcoded constant, since the owner will want to test different phase parameters over time.

Phase two, verification: same daily loss and drawdown rules as phase one, profit target roughly half of phase one.

Phase three, funded: same daily loss and drawdown rules stay active. Track a profit split for scoring purposes only, a default of 80 percent to the agent and 20 percent held back, since no real money moves through this system.

## Order Types

Support four order types in version one: market order, limit order, stop order, stop-limit order.

Do not build trailing stop in version one. This order type is a deferred item, listed at the end of this file.

## Matching Engine and Execution Realism

Model spread as a base value per instrument that widens with current simulated volatility and with session timing, using the session calendar above. Forex spreads widen outside the London and New York overlap. Futures spreads widen near contract rollover dates and during low volume overnight sessions.

Model slippage as a function of order size relative to a liquidity estimate for that instrument at that moment. Base the liquidity estimate on recent tick volume from historical data during calibration.

Model partial fills for large orders relative to available liquidity. Split a large order into smaller fills across a short window of simulated time, each fill priced slightly worse than the last.

Track commission per trade or per lot, matched to typical forex and futures rates. Track overnight swap or financing charges for positions held past the daily rollover time. Track margin used against every open position, and reject an order that would exceed available margin.

## Synthetic Market Data Generator

Build the generator from four components, each modeling a real market property.

Volatility clustering: use a GARCH process, through the `arch` package, so simulated volatility rises and falls in realistic clusters.

Regime switching: use a Markov switching model, through `statsmodels`, so the generator alternates between trending regimes and range bound regimes.

Trend persistence and mean reversion: layer a trend component during trending regimes, and a mean reverting pull during range bound regimes.

Jump events: add a jump diffusion component, a small chance on any given bar of a sudden price move outside the normal volatility range, modeling news style shocks.

Calibrate every parameter against downloaded Dukascopy and Databento history, per instrument, so synthetic paths match the real volatility and regime statistics of that specific pair or contract. A calibration fit to one instrument does not transfer to another.

Train and evaluate agents against a blend of pure historical replay and generated synthetic paths, so an agent does not pass simply from memorizing one historical stretch.

## Agent Interfaces

Build two thin wrappers around one shared core, the same account ledger, market data, and rules engine underneath both.

Reinforcement learning wrapper: a Python class with a step function and a reset function, matching the Gymnasium pattern.

LLM wrapper: a set of plain Python functions an LLM agent calls as tools, for example get market state, get account state, place order, close position, returning readable text or structured JSON depending on the calling framework.

Build both wrappers together, in the same development pass, not one after the other. Since both sit on the same core, a fix to the rules engine or matching engine applies to both agent types without extra work.

## Reward Design

Combine two reward types for reinforcement learning agents.

Per step reward: a risk adjusted return calculated on a rolling window, scaled to a small range, negative one to positive one per step.

Phase completion reward: positive 100 for passing a phase, negative 100 for a rule violation that fails the account.

Weight the phase completion reward as the dominant signal. The per step reward shapes behavior between milestones, the phase completion reward defines the actual goal the agent optimizes toward.

A first real 500k-step training run against this design converged to a policy that never traded: flat equity has zero return variance, so the risk-adjusted ratio is undefined, and the original implementation returned a bare `0.0` for that case. A `0.0` reward with no risk is a free equilibrium next to a trade that costs spread and commission the instant it opens, and training episodes are short enough that the dominant terminal reward rarely fires early on to counteract it, so the policy gradient steadily walks toward "do nothing" and stays there. `agents/rl_env.py`'s `INACTION_COST` (a small constant) closes that gap: flat equity now costs a little instead of costing nothing, so sitting out forever is no longer strictly better than the noisy alternative. `scripts/train_rl.py` also sets PPO's `ent_coef` above SB3's own default of `0.0` (no entropy bonus), since without it the policy's action distribution can collapse to near-deterministic before it has sampled enough real trades to learn anything from them.

That reward fix alone did not fix the collapse, and diagnosing why surfaced a second, more fundamental bug: `agents/rl_env.py`'s `FlattenedActionEnv` originally used a continuous `Box(-1, 1)` action with the trade pick decoded by quartile (a common trick for fitting a discrete choice into SB3's continuous-action policies). That fails outright for PPO. SB3's default continuous-action policy (`use_sde=False`) is a plain unbounded Gaussian, clipped into the box bounds by SB3 itself, never squashed through tanh (`squash_output` only takes effect when `use_sde=True`). Once the Gaussian's mean drifts past a bound, the clip has zero gradient out there, so the mean gets permanently stuck. This was confirmed empirically across two full 500k-step training runs and a fast diagnostic script sampling the trained policy directly: the deterministic action sat frozen at exactly the HOLD-mapped boundary value even as the stochastic sampling std kept climbing under a stronger entropy bonus — a dead gradient, not a "needs more training" problem. The fix was to stop fighting the wrong tool: `FlattenedActionEnv` now uses a `MultiDiscrete([4, N_SIZE_BINS] * n_instruments)` action space instead, one categorical slot for the trade pick and one for the size bin, per instrument. A categorical (softmax) distribution has gradient everywhere, so there is no boundary for the policy to get stuck against. A follow-up diagnostic on a short run showed the qualitative difference immediately: the stochastic action distribution moved from ~98% HOLD (permanently collapsed) to a genuinely mixed distribution across all four actions that was still visibly evolving, not stuck.

With the action space fixed, a 500k-step run opened real positions but never closed either one. The `INACTION_COST` fix itself looked like the cause: it triggered on zero variance in the reward window, which in practice is true if and only if the account is flat, since any open position's unrealized PnL moves every tick against real data. That accidentally made holding *any* position, regardless of quality, a permanent way to dodge the penalty meant to discourage never trading at all — closing and going flat again risked re-incurring the fixed cost, while just sitting in a bad position never did. `_step_reward` now checks the literal fact "does the account hold an open position" via `ledger.get_open_trades` rather than inferring it from reward variance, so closing a position costs exactly what never having traded costs, no more. This is a real bug fix (covered by a unit test), but a follow-up 500k run with it applied produced the identical qualitative outcome, which means the variance-based loophole was not the actual driver of the no-close behavior.

Tracing the deterministic policy's action across a full episode (not just at reset) surfaced the real picture: it output SELL at 100% of 2000 sampled steps spanning the whole 20-day window, completely independent of price, position, or equity state. The two trades on record exist only because the position-size cap (1 lot forex) rejects every further SELL once the first two fills reach it; CLOSE never wins the argmax anywhere in the episode. Three consecutive full 500k runs, three different reward-design hypotheses (inaction cost magnitude, entropy coefficient, then the position-based inaction fix above), produced the same result. That pattern points away from a reward-shaping bug and toward the categorical policy head converging to a single dominant, state-independent action before it has learned to differentiate by input — a harder problem than a loophole, and not one more reward-constant guessing is likely to fix. Left open for a future session: smaller learning rate, a larger network, or a warm start (behavior cloning against the momentum baseline) rather than another blind hyperparameter pass.

## Reproducibility and Seeding

Assign a fixed random seed to every training run and every evaluation run. Store the seed value alongside the run record in the database.

Give each stochastic component its own seeded random generator instance: the synthetic price generator, the slippage model, and any exploration randomness inside a reinforcement learning policy, each on a separate seeded stream. A change to one component's seed should never shift another component's output.

Build a fixed set of twenty seeds representing twenty distinct market condition draws. Run every agent through the same twenty seeds for a fair, direct comparison on the leaderboard.

## Leaderboard Formula

Rank agents by the Sortino ratio, not the Sharpe ratio, since this evaluation cares about downside risk specifically.

Formula: take the mean of daily returns across the evaluation period, divide by the downside deviation, meaning the standard deviation calculated using only the days with a negative return, then scale by the square root of 252.

Break a tie between two agents with a similar Sortino ratio using maximum drawdown as the second sort key, favoring the smaller drawdown.

Lock this formula before training starts. A formula change after training begins means retraining every agent affected by the change.

## Database Schema

Create these tables in `db/schema.sql` and load them into `db/propfirm.sqlite` before writing any engine code that touches the database.

```sql
CREATE TABLE accounts (
  account_id INTEGER PRIMARY KEY,
  agent_id INTEGER NOT NULL,
  phase TEXT NOT NULL,
  starting_balance REAL NOT NULL,
  current_balance REAL NOT NULL,
  current_equity REAL NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  phase_started_at TEXT NOT NULL
);

CREATE TABLE trades (
  trade_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL,
  instrument TEXT NOT NULL,
  side TEXT NOT NULL,
  size REAL NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL,
  entry_time TEXT NOT NULL,
  exit_time TEXT,
  stop_loss REAL,
  take_profit REAL,
  commission REAL NOT NULL,
  swap REAL NOT NULL,
  realized_pnl REAL,
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE rule_violations (
  violation_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL,
  rule_name TEXT NOT NULL,
  triggered_at TEXT NOT NULL,
  details TEXT,
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE equity_snapshots (
  snapshot_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL,
  timestamp TEXT NOT NULL,
  equity REAL NOT NULL,
  balance REAL NOT NULL,
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE agents (
  agent_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  framework_type TEXT NOT NULL,
  notes TEXT
);
```

Never add a new column to a live table without a migration script. Never delete a column that trade history or equity snapshots depend on.

## Config File Formats

`config/contracts.yaml`, one entry per instrument, following this shape:

```yaml
ES:
  type: future
  tick_size: 0.25
  tick_value: 12.50
  point_value: 50
  margin: 13200
  spread_ticks: 1

EUR_USD:
  type: forex
  pip_size: 0.0001
  lot_size: 100000
  pip_value: 10
  spread_pips_peak: 0.5
  leverage: 30
```

`config/sessions.yaml`, one entry per instrument, following this shape:

```yaml
EUR_USD:
  sessions:
    - name: sydney
      start_et: "17:00"
      end_et: "02:00"
    - name: tokyo
      start_et: "19:00"
      end_et: "04:00"
    - name: london
      start_et: "03:00"
      end_et: "12:00"
    - name: new_york
      start_et: "08:00"
      end_et: "17:00"
  daily_reset_et: "17:00"
  weekend_close_et: "17:00"
  weekend_close_day: friday
  weekend_open_et: "17:00"
  weekend_open_day: sunday
```

Fill out both files for every instrument named in the instrument scope section, using the contract specification table and session calendar section above as the source values.

`config/phases.yaml` holds every evaluation phase parameter: profit target, daily loss percentage, maximum drawdown percentage and mode (static or trailing), minimum trading days, consistency cap, profit split, and a position limits block. The rules engine and phase progression read these values from config, never from constants.

## Build Decisions Locked During Implementation

Recorded here so no later change drifts silently.

Rule grading thresholds: a rule fails when equity drops strictly below the limit line; sitting exactly on the line grades warn, not fail. Warn triggers once 80 percent of the distance to the limit is consumed.

Phase advancement resets the account balance and equity to the starting balance, modeling a real firm issuing a fresh account per phase. Consistency and minimum trading days evaluate at phase completion (checked at the daily reset), not per tick, since the first profitable day always carries 100 percent of profit. A consistency breach at phase completion fails the account and records a violation.

Tick storage schema, shared by downloaded history and synthetic output so replay never cares which source produced a file: Parquet columns `ts` (UTC timestamp), `bid`, `ask`, `bid_vol`, `ask_vol`, one file per day, one directory per instrument under `data/raw/` or `data/calibrated/`. Databento futures trades store the trade price as both bid and ask; the spread comes from config.

The `databento` Python package handles futures downloads and imports lazily inside `scripts/download_data.py`, so the rest of the system runs without it installed.

Execution costs live in `config/contracts.yaml` per instrument: `commission_per_side` (USD per lot or contract, each side), and for forex `swap_long` / `swap_short` (USD per standard lot per night, charged at the 17:00 Eastern rollover). Futures carry no swap; financing sits in the futures price. Values are typical retail rates, tune to taste.

Execution realism model: market orders fill at the tick mid plus a modeled half spread widened by session count and a volatility ratio (recent over baseline sigma, capped at 3x), plus size-relative slippage against a liquidity estimate taken from the previous minute's tick count. Orders larger than a tenth of per-minute liquidity split into child fills, each worse than the last, recorded as one volume-weighted trade row. A market fill is never better than the tick's own bid or ask. Orders rejecting: position size cap, then total notional cap, then margin. With the default 5x notional cap and current margin figures, the notional cap always binds before margin; the margin check guards config changes.

A `runs` table (added by `db/migrations/001_add_runs_table.sql`) stores one row per training or evaluation run with its seed, per the reproducibility section. The locked twenty-seed set lives in `scripts/run_training.py` as `SEEDS`; never reorder or replace values once training begins.

The dashboard server reads SQLite through the built-in `node:sqlite` module (Node 22 or newer), no native driver build required.

`agents/rl_env.py` supports every instrument a `Simulator` was built with in one environment: the action space is a `Dict` of a `MultiDiscrete` (hold/buy/sell/close per instrument) and a `Box` size fraction per instrument, sized against that instrument's position cap from `config/phases.yaml`. The observation is `[mid, position]` per instrument in instrument order, followed by the three global fields (equity ratio, daily loss distance, drawdown distance).

`scripts/train_rl.py` trains a real PPO agent (stable-baselines3) against `PropFirmEnv`. SB3 policies only support Box, Discrete, MultiDiscrete, and MultiBinary action spaces, not the Dict `PropFirmEnv` exposes natively for readability, so `agents/rl_env.py` also carries `FlattenedActionEnv`, a Gymnasium `ActionWrapper` mapping a single `Box(-1, 1)` to that Dict (one dim per instrument picks hold/buy/sell/close by quartile, a second dim maps to the size fraction). Default `--timesteps 20000` finishes in about a minute and proves the pipeline end to end; it is not enough steps to shape a policy that reliably clears a real profit target — the reward window alone is 60 minutes, so 20k steps covers under two weeks of simulated trading. Raise `--timesteps` for a real training run before trusting the resulting model's leaderboard position.

Episodes rotate through several distinct seeded synthetic windows (`N_WINDOWS` = 6 by default) rather than replaying one fixed stretch every episode, since a policy trained against a single repeated path memorizes that path instead of learning a generalizable skill. The window cache key includes the requested day count, since a leftover longer window from an earlier `--days` value must not get silently reused as a shorter one — that changes episode length out from under training without warning.

`TrainingProgressCallback` reads `model.ep_info_buffer` directly rather than `model.logger.name_to_value`: SB3 calls a callback's `_on_rollout_end` from inside `collect_rollouts`, before its own `_dump_logs` call writes `rollout/ep_rew_mean` into the logger, so the logger dict is always one rollout stale at that point. `ep_info_buffer` is populated per-step as Monitor-wrapped episodes finish, so it already reflects whatever completed by the time the callback fires.

A trained checkpoint plugs into `scripts/run_training.py`'s seeded leaderboard harness as a `ppo` policy, decoding through the same `vector_observation` / `decode_multidiscrete_action` functions `FlattenedActionEnv` uses during training (both now live in `agents/rl_env.py` as standalone functions rather than private methods, so there is exactly one copy of that encoding). Wiring this in surfaced two more real bugs, not model problems: first, `run_one`'s synthetic data cache checked only whether a seed's directory existed, not whether it held the requested number of days, so a leftover shorter dataset from an earlier run at a different `--days` value silently fed the wrong window length into an evaluation (same bug class as the `train_rl.py` window cache, just not yet applied here — fixed the same way, by keying the cache path on `days`). Second, the per-seed console line printed `current_balance`, which only reflects closed trades; an account holding open positions at the end of a run (real trading activity, unrealized P&L) printed as an untouched flat starting balance. Both are fixed; the line now prints `current_equity`, which is also what the rules engine and the leaderboard's Sortino calculation actually judge an account on.

`engine/calibration.py` fits `revert_kappa` from history (deviation of log price from a slow EMA anchor, same anchor the generator itself tracks) rather than using a fixed default. GARCH persistence (`alpha + beta`) is capped just under 1 and `omega` is re-anchored to the sample variance, since real minute-level forex and futures data often fits near-integrated GARCH, which makes the closed-form unconditional variance explode. Regime transition probabilities floor at 0.6 each, since a thin or noisy fit can land near a 50/50 coin flip, which whipsaws the regime chain every other minute in the generator. `scripts/validate_synthetic.py` checks a calibrated instrument's synthetic path against its source data (volatility scale, clustering, tail weight) after every calibration run; all fourteen instruments are calibrated against real history, all pass except ZN, where the generic jump-event fit dominates the tail statistic on that instrument's thin, fine-tick data — a known limit of the generic pipeline worth a per-instrument look if ZN synthetic accounts start behaving unrealistically, not a bug to chase blind.

## Dashboard

Build the React client against the design tokens and layout described in `docs/dashboard-visual-design.txt`, and match the interaction patterns demonstrated in `docs/dashboard-mockup.html`. Key constraints, restated here so an agent editing only this file still gets them right:

No rounded corners anywhere. No box shadow anywhere. No purple anywhere. Elevation comes from a background tone step and a one pixel border, never a shadow.

Color roles: near black background (#0B0D10), panel background (#15181D), hairline border (#272C33), primary text (#ECEEF0), secondary text (#8A93A1), profit and pass green (#35B075), loss and fail red (#D64545), warning amber (#D4972B), interactive and brand blue (#3E7CFF). The blue color appears only on controls and navigation, never on a data value or a chart line.

Typography: Space Grotesk for headers and hero figures, IBM Plex Sans for labels and body text, IBM Plex Mono for every number, using tabular figures so digits align in a column.

The signature interface element is the session rail, a horizontal strip across the top of every screen mapping the trading day across Sydney, Tokyo, London, and New York, with a moving marker showing current simulated time and brightened segments where sessions overlap. Build this against the live session calendar config, not as a static image.

Build six screens: an overview screen showing every account as a grid of compact panels, an account detail screen with an equity curve and open positions table, a trade ledger screen, a leaderboard screen ranked by the Sortino formula above, an alert and rule violation log screen, and a training screen.

The training screen watches an RL training run live: a run picker across the runs in the `training_runs` table, a status tag (running or finished), a progress gauge (timesteps logged against the run's total), and two line charts (mean episode reward, mean episode length) built from `training_metrics` rows. `scripts/train_rl.py` writes both tables to the same shared SQLite file as the simulation side, through a `TrainingProgressCallback` that logs once per PPO rollout. `training_runs` mutates an existing row (status flips running to finished) rather than only appending, so the server broadcasts that whole small table on any change instead of using the append-only id-cursor poll the other tables use.

Read account and trade data from the shared SQLite file through a Node SQLite driver in the Express backend. Poll for new rows on a short interval, a quarter second to one second, and forward new rows to the browser over a WebSocket connection. Do not add a message queue or a second API layer for this handoff.

The shared SQLite file has two live writers now that training runs alongside the simulation side: `engine/ledger.py`'s `connect()` sets `PRAGMA journal_mode=WAL` and a five second `busy_timeout`, so the Node dashboard's read-only connection can poll while a training run or simulation commits without either side blocking the other (WAL mode is stored in the database file itself, so setting it once from the Python side covers every future connection, Python or Node). Confirmed under real concurrent load: a training run hammered with polling requests every 150ms produced no lock errors. The dashboard server's poll loop also wraps its query body in try/catch, since an uncaught exception inside a bare `setInterval` callback crashes the whole Node process — unlike a synchronous throw inside an Express route handler, which Express already catches. That gap is exactly what took the server down once during development: a `database is locked` error inside the unwrapped poll loop killed the process outright while training kept running unaffected in its own process.

## Build Order

Work through these phases in order. Do not begin a phase before the previous phase passes its own tests.

Phase one: account ledger functions (open account, record trade, update balance and equity), a minimal matching engine handling market orders only with fixed spread and instant fill, and a basic rule set covering daily loss and maximum drawdown. Confirm a hand built test account that should fail actually fails, and one that should pass actually passes, before moving on.

Phase two: the full rule catalog, phase progression across challenge, verification, and funded, and historical data replay using Dukascopy forex data and Databento futures data.

Phase three: both agent interfaces built together, the Gymnasium style wrapper and the LLM tool calling wrapper, plus the React and Express dashboard with live updates over WebSocket.

Phase four: the synthetic generator, GARCH volatility, Markov regime switching, trend and mean reversion layering, and jump events, calibrated against the historical data pulled in phase two.

Phase five: full execution realism, session and volatility aware spread widening, size relative slippage, partial fills, and commission, swap, and margin tracking, layered onto the matching engine built in phase one.

Phase six and beyond: train reinforcement learning agents against the combined reward, run LLM agents through the same challenges, and compare every agent on the leaderboard using the fixed seed set and the Sortino formula.

## Testing Requirements

Write every test with pytest, inside the `tests/` folder, mirroring the module it tests. Rules engine tests live in `tests/test_rules.py`, ledger tests in `tests/test_ledger.py`, matching engine tests in `tests/test_matching.py`, synthetic generator tests in `tests/test_synthetic.py`.

Write a unit test for every rule in isolation, using a synthetic account state built specifically to trigger that rule, including a case placed exactly at the threshold.

Build at least one hand crafted account that should fail on a known day, run it through the engine, and confirm the failure happens on that exact day.

Build at least one hand crafted account that should pass with a small margin, run it through the engine, and confirm it passes.

Run a batch of many simulated accounts through historical data and check the pass rate looks reasonable, not near zero and not near every account passing.

## Deferred Work

DONE: trailing stop order type. `place_order` takes a `trailing_stop` price distance; the stop follows the best price reached since entry, tightening only, never loosening back toward entry. Stored as a `trailing_stop` column on `trades` (`db/migrations/002_add_trailing_stop.sql`), since a live table never gets a new column without a migration script.

TODO: add per instrument daily reset times, if calibration against real account behavior shows a meaningful difference from the unified 5pm Eastern reset used in version one.

TODO: raise position size caps for funded accounts, once a scaling plan gets defined.
