"""Current TSX holdings to feed into apply_rebalancing_band.

When CURRENT_HOLDINGS is non-empty, predict_now passes it to
apply_rebalancing_band, which prefers keeping existing positions to
reduce turnover. The rank-buffer and score-tolerance logic in that
function only does meaningful work when it has real holdings to defend
against being kicked out by marginally higher-ranked newcomers.

Leave the list empty if you want every run to produce a fresh ranking
without any incumbency bias.

Use yfinance tickers (e.g. "RY.TO", "ENB.TO"). The benchmark XIU.TO
should NOT appear here.

Setup:
    cp portfolio_config.example.py portfolio_config.py
    # then edit portfolio_config.py with your real holdings

portfolio_config.py is gitignored so your holdings stay local.
"""

CURRENT_HOLDINGS = [
    # Add your current TSX holdings, one per line. Examples:
    # "RY.TO",
    # "ENB.TO",
    # "CNQ.TO",
]

# ── Legacy positions (sticky — sold only when the model is bearish) ───────
# Stocks you hold and prefer to keep, folded into the monthly portfolio's
# weights + reporting. Default = KEEP at the position's real market-value
# weight; the active picks split whatever capital remains. Legacy are excluded
# from the active candidate pool (no duplicate buys).
#
# "Sell only when the model thinks so": each picker emits a SELL advisory ONLY
# for legacy names ITS model actually scores — the CA picker (picker.py) judges
# TSX-universe legacy, the US picker (picker_us.py) judges its US-universe
# legacy. Names neither model covers (other US names, ETFs) are carry-only:
# held, weighted, never flagged to sell.
#
#   value    = market value of the position (in the currency below).
#   currency = "CAD" (default) or "USD". USD is converted to CAD via live
#              USDCAD so combined weights are apples-to-apples.
#   sector   = OPTIONAL. Auto from STOCK_PROFILE for names the model tracks;
#              REQUIRED for untracked names (US singles, ETFs) for sector mix.
#
# Leave empty for no legacy sleeve (behaves exactly as before).
LEGACY_HOLDINGS = {
    # --- TSX (CA picker scores these -> can advise SELL) ---
    # "CM.TO":  {"value": 43288},                  # sector auto (Financials)
    # "RY.TO":  {"value": 6645},
    # "AEM.TO": {"value": 2000},
    # --- US (US picker scores these -> can advise SELL) ---
    # "MSFT":   {"value": 5139, "currency": "USD", "sector": "Cloud"},
    # "NVDA":   {"value": 1281, "currency": "USD", "sector": "Semiconductors"},
    # --- carry-only (neither model scores -> held, never sold) ---
    # "KO":     {"value": 554,  "currency": "USD", "sector": "ConsumerStaples"},
    # "VFV.TO": {"value": 5039, "sector": "ETF"},  # S&P 500 ETF
}
