"""
在 top_n=5(集中)下跑 rigor:DSR/过拟合审计 + CPCV。
确认"集中到5只多赚7个点"不是过拟合。复用 picker rigor 管线。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np

from picker import (
    TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS, CONSTRAINTS,
    USE_MOMENTUM_PCA, _RAW_MOMENTUM,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize,
    walk_forward, print_overfit_report, compute_cpcv, print_cpcv_report,
)

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
print(f"=== rigor @ top_n={TOP_N} ===")
print("建 panel ...")
price_df = fetch_prices(TSX_UNIVERSE + list(MACRO_TICKERS.values()), years=7)
panel = build_panel(price_df, price_df, TSX_UNIVERSE)
feats = [c for c in FEATURE_COLS if c in panel.columns]
panel = smart_impute(panel, feats)
panel = add_labels(panel)
if USE_MOMENTUM_PCA:
    panel = apply_momentum_pca(panel)
    feats = [c for c in FEATURE_COLS if c in panel.columns and c not in _RAW_MOMENTUM]
else:
    feats = [c for c in feats if c not in ("mom_pc1", "mom_pc2")]
panel, model_features = cross_sectional_normalize(panel, feats)

CONSTRAINTS["top_n"] = TOP_N

print("\nStep 1/2: walk-forward + DSR/过拟合审计 ...")
results = walk_forward(panel, model_features)
wf_sharpe = None
if not results.empty and "port_ret" in results.columns:
    r = results["port_ret"].values
    wf_sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
print_overfit_report(results)

print("\nStep 2/2: CPCV ...")
cpcv_paths = compute_cpcv(panel, model_features)
print_cpcv_report(cpcv_paths, wf_sharpe=wf_sharpe)
