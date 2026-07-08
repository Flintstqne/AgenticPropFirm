#!/bin/bash
# Runs the dashboard server and client together. Ctrl+C stops both.
set -e

cd "$(dirname "$0")/.."

if [ ! -f db/propfirm.sqlite ]; then
  echo "No db/propfirm.sqlite yet -- run scripts/setup.sh first."
  exit 1
fi

cleanup() {
  kill "$SERVER_PID" "$CLIENT_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

(cd dashboard/server && pnpm start) &
SERVER_PID=$!

(cd dashboard/client && pnpm run dev) &
CLIENT_PID=$!

wait
