# Prop Firm Simulation

A personal, self hosted simulation of a prop firm evaluation program. AI trading agents, both reinforcement learning agents and LLM agents using tool calls, trade forex and futures instruments against historical and synthetic price data, and get judged against profit targets, drawdown limits, and daily loss rules in real time.

## Setup

Run the setup script once to build both environments and load the database schema.

```
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Copy `.env.example` to `.env` and fill in your own Databento and Alpha Vantage API keys before pulling historical data.

## Running Tests

```
chmod +x scripts/run_tests.sh
./scripts/run_tests.sh
```

## Running the Dashboard

```
cd dashboard/server
pnpm start
```

Open a second terminal for the client.

```
cd dashboard/client
pnpm run dev
```

## Project Documentation

AGENTS.md and CLAUDE.md hold the full build directive for any coding agent working on this repository. Read one of these files before writing code.

docs/dashboard-visual-design.txt describes the dashboard color system, typography, layout, and signature elements.

docs/dashboard-mockup.html is a working mockup of the dashboard screens.

docs/version-control.md holds the branching strategy and commit rules for this repository.

## Instrument Scope

Seven forex pairs: EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD.

Seven futures contracts: ES, NQ, YM, CL, GC, ZN, 6E.

## License

Personal project. No public license granted.
