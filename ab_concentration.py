"""
A/B: 集中度 / 权重法 对 收益 vs Sharpe vs 最大回撤 的影响。
复用 picker.py 的回测管线。只改 CONSTRAINTS["top_n"] 和 weight_method。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from picker import (
    TSX_UNIVERSE, MACRO_TICKERS, BENCHMARK_TICKER, FEATURE_COLS, CONSTRAINTS,
    USE_MOMENTUM_PCA, _RAW_MOMENTUM,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize, _wf_metrics,
)

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
print(f"panel: {len(panel)} 行, {panel['date'].nunique()} 月\n")

# (label, top_n, weight_method)
configs = [
    ("top10 等权 (现状)", 10, "equal"),
    ("top5  等权 (集中,1/行业)", 5, "equal"),
    ("top10 max_sharpe 加权", 10, "max_sharpe"),
    ("top5  max_sharpe 加权", 5, "max_sharpe"),
]

orig = CONSTRAINTS["top_n"]
rows = []
for label, tn, wm in configs:
    CONSTRAINTS["top_n"] = tn
    print(f"跑 {label} ...")
    sr, ir, ann, dd, hit = _wf_metrics(panel, model_features, weight_method=wm)
    rows.append((label, ann, sr, ir, dd, hit))
CONSTRAINTS["top_n"] = orig

print("\n" + "=" * 78)
print(f"{'配置':<26}{'年化收益':>9}{'Sharpe':>8}{'IR':>7}{'最大回撤':>9}{'命中率':>8}")
print("-" * 78)
for label, ann, sr, ir, dd, hit in rows:
    print(f"{label:<26}{ann*100:>8.1f}%{sr:>8.2f}{ir:>7.2f}{dd*100:>8.1f}%{hit*100:>7.0f}%")
print("=" * 78)
print("注:回撤越接近 0 越好(负值)。Sharpe=收益/风险;只追年化会牺牲回撤与 Sharpe。")
