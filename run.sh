#!/usr/bin/env bash
# Bootstrap + launch MLXr. Requires Apple Silicon + Python 3.10+.
# Loops on exit code 42 so the dashboard can self-restart after upgrades.
set -euo pipefail
cd "$(dirname "$0")"

MIN_PY="(3, 10)"

# Pick the newest Python >=3.10 available on PATH.
PICK=""
for candidate in python3.13 python3.12 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PICK="$candidate"
    break
  fi
done

# Fall back to python3 if it satisfies the floor — some systems only ship `python3`.
if [[ -z "$PICK" ]] && command -v python3 >/dev/null 2>&1; then
  if python3 -c "import sys; sys.exit(0 if sys.version_info >= ${MIN_PY} else 1)"; then
    PICK="python3"
  fi
fi

if [[ -z "$PICK" ]]; then
  cat >&2 <<EOF
Error: MLXr requires Python >=3.10 (mlx-lm ≥0.30 dropped 3.9).
Install one of: python3.10, python3.11, python3.12, python3.13.
On macOS:  brew install python@3.12
EOF
  exit 1
fi

# If an existing venv uses an outdated Python, stop — don't silently delete.
if [[ -d .venv ]]; then
  if ! .venv/bin/python -c "import sys; sys.exit(0 if sys.version_info >= ${MIN_PY} else 1)" 2>/dev/null; then
    current=$(.venv/bin/python --version 2>&1 || echo "unknown")
    cat >&2 <<EOF
Error: .venv uses $current, but mlx-lm requires Python >=3.10.
Recreate it (this will delete the existing .venv):
  rm -rf .venv
  $PICK -m venv .venv
  ./run.sh
EOF
    exit 1
  fi
fi

if [[ ! -d .venv ]]; then
  echo "Creating .venv with $PICK..."
  "$PICK" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

HOST="${MLXR_HOST:-127.0.0.1}"
PORT="${MLXR_PORT:-8000}"
export MLXR_MANAGED=1

while true; do
  echo "MLXr listening on http://${HOST}:${PORT}"
  set +e
  python -m uvicorn server:app --host "$HOST" --port "$PORT"
  code=$?
  set -e
  if [[ $code -ne 42 ]]; then
    exit $code
  fi
  echo "Restart requested — refreshing deps and relaunching..."
  pip install -r requirements.txt
done
