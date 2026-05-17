"""Diagnose why each holding in portfolio_config did or didn't make the picks.

Runs the pipeline once, then for each ticker in CURRENT_HOLDINGS:
  - Did it survive apply_constraints?
  - What was its raw score?
  - What's its rank in the constraint-passing candidate pool?
  - Would the rebalancing band's rank_buffer save it?
"""
import warnings
warnings.filterwarnings("ignore")

import picker

print("\n=== Diagnose holdings ===\n")
print(f"CURRENT_HOLDINGS: {picker.CURRENT_HOLDINGS}\n")

all_tickers = picker.TSX_UNIVERSE + list(picker.MACRO_TICKERS.values())
price_df = picker.fetch_prices(all_tickers, years=7)
panel = picker.build_panel(price_df, price_df, picker.TSX_UNIVERSE)
available = [c for c in picker.FEATURE_COLS if c in panel.columns]
panel = picker.smart_impute(panel, available)
panel = picker.add_labels(panel)

if picker.USE_MOMENTUM_PCA:
    panel = picker.apply_momentum_pca(panel)
    available = [c for c in picker.FEATURE_COLS
                 if c in panel.columns and c not in picker._RAW_MOMENTUM]

panel, model_features = picker.cross_sectional_normalize(panel, available)

# Fit on full history; score latest month
import pandas as pd
panel = panel.sort_values("date")
dates = sorted(panel["date"].unique())
train_df = panel[panel["date"] < dates[-1]].copy()
latest_df = panel[panel["date"] == dates[-1]].copy()

X_train = train_df[model_features].values
y_train = train_df["fwd_ret"].values
X_latest = latest_df[model_features].values
weights = picker.compute_time_decay_weights(len(X_train))
reg, clf = picker.fit_models(pd.DataFrame(X_train, columns=model_features),
                              pd.Series(y_train), sample_weights=weights)
scores = picker.ensemble_predict(reg, clf, X_latest)
latest_df = latest_df.copy()
latest_df["score"] = scores

# Constraint filter
candidates = latest_df.sort_values("score", ascending=False)["ticker"].tolist()
fund_df = picker.fetch_fundamentals(candidates[:30])
filtered = picker.apply_constraints(candidates, fund_df, price_df, mode="pick",
                                     current_holdings=picker.CURRENT_HOLDINGS)

C = picker.CONSTRAINTS
rank_buf = C["rank_buffer"]
hold_bonus = C["hold_bonus"]
top_n = C["top_n"]

print(f"Pool: {len(candidates)} total, {len(filtered)} pass constraints")
print(f"rank_buffer={rank_buf}, hold_bonus={hold_bonus}, top_n={top_n}\n")

# Score dict for filtered candidates (post-boost)
score_dict = dict(zip(filtered,
                      latest_df.set_index("ticker").loc[filtered, "score"].tolist()))
for h in picker.CURRENT_HOLDINGS:
    if h in score_dict:
        score_dict[h] += hold_bonus

ranked = sorted(score_dict.items(), key=lambda x: -x[1])
ranked_tickers = [t for t, s in ranked]

print(f"{'Ticker':<10} {'In universe':<12} {'Passed filters':<16} {'Raw score':<12} {'Boosted':<10} {'Rank':<8} {'Kept?':<8}")
print("-" * 80)

def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)

for h in picker.CURRENT_HOLDINGS:
    in_univ = h in picker.TSX_UNIVERSE
    passed = h in filtered
    raw_score = (float(latest_df[latest_df["ticker"] == h]["score"].iloc[0])
                 if h in latest_df["ticker"].values else None)
    boosted = score_dict.get(h)
    rank = ranked_tickers.index(h) + 1 if h in ranked_tickers else None
    kept = (rank is not None and rank <= rank_buf)
    print(f"{h:<10} {str(in_univ):<12} {str(passed):<16} "
          f"{fmt(raw_score, '.4f'):<12} {fmt(boosted, '.4f'):<10} "
          f"{(str(rank) if rank else 'n/a'):<8} "
          f"{'YES' if kept else 'no':<8}")

print(f"\nTop-20 by boosted score (rank_buffer cutoff = {rank_buf}):")
for i, (t, s) in enumerate(ranked[:20], 1):
    marker = "  <- holding" if t in picker.CURRENT_HOLDINGS else ""
    print(f"  {i:>2}. {t:<10} {s:.4f}{marker}")
