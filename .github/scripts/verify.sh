#!/usr/bin/env bash
# Checks for this repository on GitHub. There was no earlier test pipeline (the old workflow only deployed), so
# this runs the project's documented test suite (CLAUDE.md) in the Python 3.12 runtime its Dockerfile uses.
# Nothing deploys, nothing connects to a broker or exchange, and no real credentials are used.
set -euo pipefail
test "${GITHUB_SERVER_URL:?}" = 'https://github.com'
case "${GITHUB_REPOSITORY:?}" in
  HallyAus/FlashTrade|Agentic-Consciousness/FlashTrade) ;;
  *) echo "Unexpected repository: $GITHUB_REPOSITORY" >&2; exit 1 ;;
esac
test "$(git rev-parse HEAD)" = "${GITHUB_SHA:?}"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use the Python 3.12 runtime the Dockerfile uses"'
unset ANTHROPIC_API_KEY ALPACA_API_KEY ALPACA_SECRET_KEY SWYFTX_API_KEY DATABASE_URL REDIS_URL
export PYTHONPATH="$PWD" PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --no-cache-dir -r requirements.txt
python -m pytest -v
