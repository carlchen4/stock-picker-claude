"""
80/20 配置 + 再平衡助手。
读两个 picker 的最新名单(picks_log.csv / picks_log_us.csv),按 CA 80% / US 20%
(加元口径)算出每只该买多少钱、多少股,并对比你当前持仓给出买卖清单。

用法:
  python allocate.py --ca 100000 --us 40000          # 当前 CA/US 加元值
  python allocate.py --ca 100000 --us 40000 --target-us 0.20 --band 0.05

注:看不到你券商持仓,所以当前 CA/US 金额要手输。价格用 yfinance 实时,
US 名单按实时 USD/CAD 折算股数。picks_log 由 picker 跑 pick 时生成。
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

CA_LOG = "picks_log.csv"
US_LOG = "picks_log_us.csv"
CA_BENCH, US_BENCH = "XIU.TO", "QQQ"


def latest_picks(path, bench):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["as_of"])
    df = df[df["as_of"] == df["as_of"].max()]
    df = df[df["ticker"] != bench]
    df = df[df["weight"] > 0]
    w = df.set_index("ticker")["weight"]
    return (w / w.sum()) if w.sum() > 0 else w   # 归一到 sleeve 内 1.0


def fx_usdcad():
    import yfinance as yf
    return float(yf.Ticker("CAD=X").history(period="5d")["Close"].iloc[-1])


def prices(tickers):
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            out[t] = float(yf.Ticker(t).history(period="5d")["Close"].iloc[-1])
        except Exception:
            out[t] = np.nan
    return out


def sleeve_table(weights, sleeve_cad, fx, is_us):
    """返回每只: 目标加元、(US 折美元)、价格、股数。"""
    rows = []
    px = prices(list(weights.index))
    for t, w in weights.items():
        cad = sleeve_cad * w
        p = px[t]
        if is_us:                       # US 价格是 USD,sleeve 是 CAD
            usd = cad / fx
            sh = usd / p if p and not np.isnan(p) else np.nan
            rows.append((t, cad, usd, p, sh))
        else:                           # CA 价格 CAD
            sh = cad / p if p and not np.isnan(p) else np.nan
            rows.append((t, cad, None, p, sh))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ca", type=float, required=True, help="当前 CA 加元值")
    ap.add_argument("--us", type=float, required=True, help="当前 US 加元值")
    ap.add_argument("--target-us", type=float, default=0.20, help="US 目标权重(默认0.20)")
    ap.add_argument("--band", type=float, default=0.05, help="再平衡带宽(默认±0.05)")
    a = ap.parse_args()

    ca_w = latest_picks(CA_LOG, CA_BENCH)
    us_w = latest_picks(US_LOG, US_BENCH)
    if ca_w is None or us_w is None:
        miss = [f for f, w in [(CA_LOG, ca_w), (US_LOG, us_w)] if w is None]
        print(f"⚠️ 找不到名单: {miss}。先跑 `./run.sh monthly`(或 ca / us)生成 picks_log。")
        sys.exit(1)

    total = a.ca + a.us
    cur_us_w = a.us / total
    tgt_us_cad = total * a.target_us
    tgt_ca_cad = total * (1 - a.target_us)
    fx = fx_usdcad()

    print(f"\n{'='*64}")
    print(f"总额 ${total:,.0f} CAD  |  当前 CA {a.ca/total*100:.0f}% / US {cur_us_w*100:.0f}%"
          f"  |  目标 CA {(1-a.target_us)*100:.0f}% / US {a.target_us*100:.0f}%  |  USDCAD {fx:.3f}")
    print('='*64)

    # 再平衡判断
    drift = cur_us_w - a.target_us
    move = tgt_us_cad - a.us          # >0 该加 US,<0 该减 US
    if abs(drift) <= a.band:
        print(f"✅ US 偏离 {drift*100:+.1f}pp,在 ±{a.band*100:.0f}pp 带宽内 —— 本次不用再平衡。")
    else:
        act = "买入" if move > 0 else "卖出"
        print(f"⚠️ US 偏离 {drift*100:+.1f}pp,超带宽 → {act} US 科技 "
              f"${abs(move):,.0f} CAD (≈${abs(move)/fx:,.0f} USD),转/来自 CA。")

    def show(title, rows, is_us):
        print(f"\n[{title}]  目标 ${ (tgt_us_cad if is_us else tgt_ca_cad):,.0f} CAD")
        if is_us:
            print(f"  {'票':<8}{'加元':>10}{'美元':>10}{'价(USD)':>10}{'股数':>8}")
            for t, cad, usd, p, sh in rows:
                ps = f"{p:.2f}" if p==p else "n/a"; ss=f"{sh:.0f}" if sh==sh else "n/a"
                print(f"  {t:<8}{cad:>10,.0f}{usd:>10,.0f}{ps:>10}{ss:>8}")
        else:
            print(f"  {'票':<10}{'加元':>10}{'价(CAD)':>10}{'股数':>8}")
            for t, cad, _, p, sh in rows:
                ps = f"{p:.2f}" if p==p else "n/a"; ss=f"{sh:.0f}" if sh==sh else "n/a"
                print(f"  {t:<10}{cad:>10,.0f}{ps:>10}{ss:>8}")

    show(f"CA 80% · {len(ca_w)} 只(等权)", sleeve_table(ca_w, tgt_ca_cad, fx, False), False)
    show(f"US 20% · {len(us_w)} 只(等权)", sleeve_table(us_w, tgt_us_cad, fx, True), True)
    print(f"\n{'='*64}\n描述性配置工具,非投资建议。价格实时,执行以你下单时为准。")


if __name__ == "__main__":
    main()
