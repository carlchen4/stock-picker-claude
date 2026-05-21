#!/bin/bash
# Monthly TSX picker run — intended to be invoked by launchd.
# Activates the project venv, runs `pick` (which emails the report when
# email_config.py is set), and appends stdout/stderr to logs/monthly.log.
#
# Uses the script's own directory so it works regardless of where launchd
# starts it from. Set up scheduling with:  see SCHEDULING.md
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
mkdir -p logs

{
    echo "=================================================="
    echo "=== TSX picker run: $(date) ==="
    if [ -d venv ]; then
        # shellcheck disable=SC1091
        source venv/bin/activate
    fi
    python picker.py pick
    echo "=== done: $(date) ==="
} >> logs/monthly.log 2>&1
