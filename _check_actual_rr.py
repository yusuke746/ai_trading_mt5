import sqlite3

DB = r"C:\Users\user\openHands-test\ai_trading_mt5\trades.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# 実際のRR分布（entry_price, sl_price, tp_priceから計算）
print("=== 実際のRR分布（#付き全体）===")
cur.execute("""
    SELECT
        symbol, direction,
        entry_price, sl_price, tp_price,
        result_profit, smc_rr_pass
    FROM trades
    WHERE symbol LIKE '%#%' AND status='CLOSED'
      AND entry_price IS NOT NULL AND sl_price IS NOT NULL AND tp_price IS NOT NULL
""")
rows = cur.fetchall()

import statistics

rr_vals = []
for r in rows:
    sym, direction, ep, sl, tp, pnl, rr_pass = r
    sl_dist = abs(ep - sl)
    tp_dist = abs(tp - ep)
    if sl_dist > 0:
        rr = tp_dist / sl_dist
        rr_vals.append((sym, direction, round(rr, 2), pnl, rr_pass))

# RR分布
bands = {"<0.75": [], "0.75-1.0": [], "1.0-1.5": [], "1.5-2.0": [], "2.0+": []}
for sym, d, rr, pnl, rp in rr_vals:
    if rr < 0.75:
        bands["<0.75"].append(pnl)
    elif rr < 1.0:
        bands["0.75-1.0"].append(pnl)
    elif rr < 1.5:
        bands["1.0-1.5"].append(pnl)
    elif rr < 2.0:
        bands["1.5-2.0"].append(pnl)
    else:
        bands["2.0+"].append(pnl)

for band, pnls in bands.items():
    if pnls:
        wins = sum(1 for p in pnls if p > 0)
        wr = round(wins / len(pnls) * 100, 1)
        avg = round(sum(pnls) / len(pnls), 0)
        print(f"  RR {band}: cnt={len(pnls)}, WR={wr}%, avg_pnl={avg}")
    else:
        print(f"  RR {band}: cnt=0")

# 銘柄別 平均RR
print("\n=== 銘柄別 平均RR（実際）===")
from collections import defaultdict
sym_rr = defaultdict(list)
for sym, d, rr, pnl, rp in rr_vals:
    sym_rr[sym].append(rr)
for sym in sorted(sym_rr):
    vals = sym_rr[sym]
    print(f"  {sym}: n={len(vals)}, avg_RR={round(sum(vals)/len(vals),2)}, min={min(vals)}, max={max(vals)}")

# RR>=1.0 の件数（銘柄別）
print("\n=== RR>=1.0 件数（銘柄別）===")
from collections import Counter
rr1_cnt = Counter()
total_cnt = Counter()
for sym, d, rr, pnl, rp in rr_vals:
    total_cnt[sym] += 1
    if rr >= 1.0:
        rr1_cnt[sym] += 1
for sym in sorted(total_cnt):
    t = total_cnt[sym]
    r1 = rr1_cnt[sym]
    print(f"  {sym}: total={t}, RR>=1.0={r1} ({round(r1/t*100,1)}%)")

conn.close()
