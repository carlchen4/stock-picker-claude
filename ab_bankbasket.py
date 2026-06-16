"""
A/B: 选个股银行 vs 持银行 ETF。
基线 = 现状(金融板块含 7 只大银行,模型从中选)。
变体 = 把 7 只大银行换成 ZEB.TO(银行 ETF),保留 5 只差异化金融股
       (MFC/SLF/FFH/BAM/BN)。测"别再去选同质化银行"是否更优。
复用 picker 回测管线。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import picker
from picker import (
    MACRO_TICKERS, FEATURE_COLS, USE_MOMENTUM_PCA, _RAW_MOMENTUM,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize, _wf_metrics,
)

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "CM.TO", "BNS.TO", "NA.TO", "EQB.TO"]
ETF = "ZEB.TO"


def build(universe):
    price_df = fetch_prices(universe + list(MACRO_TICKERS.values()), years=7)
    panel = build_panel(price_df, price_df, universe)
    feats = [c for c in FEATURE_COLS if c in panel.columns]
    panel = smart_impute(panel, feats)
    panel = add_labels(panel)
    if USE_MOMENTUM_PCA:
        panel = apply_momentum_pca(panel)
        feats = [c for c in FEATURE_COLS if c in panel.columns and c not in _RAW_MOMENTUM]
    else:
        feats = [c for c in feats if c not in ("mom_pc1", "mom_pc2")]
    panel, mf = cross_sectional_normalize(panel, feats)
    return panel, mf


rows = []

# --- 基线:现状 ---
print("跑 基线(选个股银行) ...")
uni0 = list(picker.TSX_UNIVERSE)
p0, mf0 = build(uni0)
rows.append(("基线: 选 7 只银行 (现状)", *_wf_metrics(p0, mf0)))

# --- 变体:7 银行 -> ZEB.TO ---
print("跑 变体(银行 ETF) ...")
uni1 = [t for t in picker.TSX_UNIVERSE if t not in BANKS]
if ETF not in uni1:
    uni1.append(ETF)
picker.STOCK_PROFILE[ETF] = ("Financials", "core", "bank_etf")
picker.TSX_UNIVERSE = uni1            # 让内部用到全局的地方也一致
p1, mf1 = build(uni1)
rows.append(("变体: 银行用 ZEB.TO ETF", *_wf_metrics(p1, mf1)))
picker.TSX_UNIVERSE = uni0            # 还原

print("\n" + "=" * 76)
print(f"{'配置':<28}{'年化收益':>9}{'Sharpe':>8}{'IR':>7}{'最大回撤':>9}{'命中率':>8}")
print("-" * 76)
for label, sr, ir, ann, dd, hit in rows:
    print(f"{label:<28}{ann*100:>8.1f}%{sr:>8.2f}{ir:>7.2f}{dd*100:>8.1f}%{hit*100:>7.0f}%")
print("=" * 76)
