#!/bin/bash
set -e

echo "Setting up Python environment..."
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

echo "Setting up dashboard server..."
cd dashboard/server
pnpm install
cd ../..

echo "Setting up dashboard client..."
cd dashboard/client
pnpm install
cd ../..

echo "Loading database schema..."
./venv/bin/python -c "from engine import ledger; c = ledger.connect('db/propfirm.sqlite'); ledger.init_db(c)"

echo "Setup complete."
