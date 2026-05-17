"""Walk each per-ticker constraint check for the holdings and report which step rejects them."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import picker

TICKERS_TO_CHECK = picker.CURRENT_HOLDINGS

print(f"\n=== Per-ticker constraint walk for {TICKERS_TO_CHECK} ===\n")

all_tickers = picker.TSX_UNIVERSE + list(picker.MACRO_TICKERS.values())
price_df = picker.fetch_prices(all_tickers, years=7)
fund_df = picker.fetch_fundamentals(TICKERS_TO_CHECK)
C = picker.CONSTRAINTS

for ticker in TICKERS_TO_CHECK:
    print(f"--- {ticker} ---")
    prices, vol = picker.get_ohlcv(price_df, ticker)
    if prices is None:
        print("  REJECT: no price data")
        continue
    print(f"  price history: {len(prices)} days (min={C['min_listing_days']})")
    if len(prices) < C["min_listing_days"]:
        print("  REJECT: min_listing_days")
        continue
    last_price = float(prices.iloc[-1])
    print(f"  last_price: ${last_price:.2f}  [{C['min_price_cad']}, {C['max_price_cad']}]")
    if last_price < C["min_price_cad"] or last_price > C["max_price_cad"]:
        print("  REJECT: price band")
        continue
    if vol is not None:
        adv = float((vol.tail(20) * prices.tail(20)).mean())
        print(f"  ADV (20d): ${adv:,.0f}  (min={C['min_adv_cad']:,})")
        if adv < C["min_adv_cad"]:
            print("  REJECT: min_adv")
            continue
        vol_series = vol.tail(60)
        vmean, vstd = float(vol_series.mean()), float(vol_series.std())
        spike_days = int((vol_series > vmean + C["vol_spike_sigma"] * vstd).sum()) if vstd > 0 else 0
        print(f"  vol spike days: {spike_days}  (max={C['vol_spike_min_days']})")
        if spike_days >= C["vol_spike_min_days"]:
            print("  REJECT: vol_spike")
            continue
    if ticker in fund_df.index:
        row = fund_df.loc[ticker]
        pe = picker.safe_float(row.get("pe"))
        roe = picker.safe_float(row.get("roe"))
        mktcap = picker.safe_float(row.get("mktcap"))
        print(f"  PE: {pe}  [{C['min_pe']}, {C['max_pe']}]")
        if not np.isnan(pe) and (pe < C["min_pe"] or pe > C["max_pe"]):
            print("  REJECT: PE")
            continue
        print(f"  ROE: {roe}  [{C['min_roe']}, {C['max_roe']}]")
        if not np.isnan(roe) and (roe < C["min_roe"] or roe > C["max_roe"]):
            print("  REJECT: ROE")
            continue
        print(f"  MktCap: ${mktcap:,.0f}  (min={C['min_mktcap_cad']:,})")
        if not np.isnan(mktcap) and mktcap < C["min_mktcap_cad"]:
            print("  REJECT: mktcap")
            continue
    else:
        print("  NOTE: no fundamentals row")
    print("  PASSED all per-ticker checks")
    print()
