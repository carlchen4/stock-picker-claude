#!/bin/bash
# ============================================================================
#  run.sh — one entry point for the stock-picker toolkit
#  Usage:  ./run.sh <command>
#
#  MONTHLY (the real workflow — emails to hotmail + updates dashboard):
#    ./run.sh monthly      both CA + US picks   (= run_monthly.sh)
#    ./run.sh ca           Canadian picks only
#    ./run.sh us           US tech picks only
#
#  QUARTERLY (after bank earnings season ~ Feb/May/Aug/Dec):
#    ./run.sh banks        7-bank fundamental score & rank
#                          ** first hand-update CET1/PCL/segment in
#                             bank_score_ca.py DATA dict from the reports **
#    ./run.sh sectors      Energy/Industrials/Utilities/Gold fundamental
#                          score & rank (sector-specific metrics) — auto
#    ./run.sh bankdeep     bank yfinance trend (ROE/EPS/BVPS/op-lev) — auto
#    ./run.sh fund         full 30-name CA fundamental screen — auto
#    ./run.sh fund-us      US tech 21-name fundamental score — auto
#
#  ON DEMAND (validation):
#    ./run.sh backtest     CA walk-forward (Sharpe/IR)
#    ./run.sh backtest-us  US walk-forward
#    ./run.sh rigor        overfitting audit (DSR/CPCV/PBO)
#    ./run.sh monitor      daily holdings monitor
#    ./run.sh test         run unit tests (test_picker.py)
#
#  Reminder: execute picks AT MARKET on pick day. Entry timing was tested
#  (entry_timing.py / limit_buy_test.py) and is negative-expectation — don't
#  wait, don't bottom-pick with limit orders. Those scripts need no re-run.
# ============================================================================
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
[ -d venv ] && source venv/bin/activate

cmd="${1:-help}"
case "$cmd" in
  monthly)      python picker.py pick && python picker_us.py pick ;;
  ca)           python picker.py pick ;;
  us)           python picker_us.py pick ;;
  banks)        python bank_score_ca.py ;;
  bankdeep)     python bank_deep_ca.py ;;
  sectors)      python sector_score_ca.py ;;
  fund)         python fundamentals_ca.py ;;
  fund-us)      python fundamentals_us.py ;;
  backtest)     python picker.py backtest ;;
  backtest-us)  python picker_us.py backtest ;;
  rigor)        python picker.py rigor ;;
  monitor)      python picker.py monitor ;;
  test)         python test_picker.py ;;
  help|*)       sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
