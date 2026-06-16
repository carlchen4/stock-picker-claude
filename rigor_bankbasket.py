"""在 '银行用 ZEB.TO' 变体下跑 rigor(DSR + CPCV),确认不是侥幸。"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import picker
from picker import (
    MACRO_TICKERS, FEATURE_COLS, USE_MOMENTUM_PCA, _RAW_MOMENTUM,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize,
    walk_forward, print_overfit_report, compute_cpcv, print_cpcv_report,
)

BANKS = ["RY.TO","TD.TO","BMO.TO","CM.TO","BNS.TO","NA.TO","EQB.TO"]
ETF = "ZEB.TO"
uni = [t for t in picker.TSX_UNIVERSE if t not in BANKS]
if ETF not in uni: uni.append(ETF)
picker.STOCK_PROFILE[ETF] = ("Financials","core","bank_etf")
picker.TSX_UNIVERSE = uni

price_df = fetch_prices(uni + list(MACRO_TICKERS.values()), years=7)
panel = build_panel(price_df, price_df, uni)
feats = [c for c in FEATURE_COLS if c in panel.columns]
panel = smart_impute(panel, feats)
panel = add_labels(panel)
if USE_MOMENTUM_PCA:
    panel = apply_momentum_pca(panel)
    feats = [c for c in FEATURE_COLS if c in panel.columns and c not in _RAW_MOMENTUM]
else:
    feats = [c for c in feats if c not in ("mom_pc1","mom_pc2")]
panel, mf = cross_sectional_normalize(panel, feats)

print("=== rigor @ 银行用 ZEB.TO ===")
results = walk_forward(panel, mf)
ws = None
if not results.empty and "port_ret" in results.columns:
    r = results["port_ret"].values; ws = r.mean()/r.std(ddof=1)*np.sqrt(12)
print_overfit_report(results)
print_cpcv_report(compute_cpcv(panel, mf), wf_sharpe=ws)
