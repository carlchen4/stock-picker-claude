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
