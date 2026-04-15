#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/venv/bin/python"
APP="$ROOT_DIR/ReachyCheese.py"

usage() {
  cat <<EOF
Usage: $(basename "$0") [config.json]

Start ReachyCheese with --config.

Arguments:
  config.json   Optional config file path.
                Default: REACHYCHEESE_CONFIG env or config/reachycheese.example.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Default config path; override with first argument or REACHYCHEESE_CONFIG env.
CONFIG_PATH="${1:-${REACHYCHEESE_CONFIG:-$ROOT_DIR/config/reachycheese.example.json}}"

if [[ ! -x "$PY" ]]; then
  echo "❌ Python venv not found: $PY"
  echo "Run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
  exit 2
fi

if [[ ! -f "$APP" ]]; then
  echo "❌ ReachyCheese.py not found: $APP"
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "❌ Config file not found: $CONFIG_PATH"
  echo "Tip: copy and edit $ROOT_DIR/config/reachycheese.example.json"
  exit 2
fi

echo "🚀 Starting ReachyCheese"
echo "   project: $ROOT_DIR"
echo "   config : $CONFIG_PATH"

echo
exec "$PY" "$APP" --config "$CONFIG_PATH"
