"""銘柄×UTC時間帯別WR分析 (セッションフィルター設計用)"""
import sqlite3

import os
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
SELECT
    symbol as sym,
    CAST(strftime('%H', opened_at) AS INTEGER) as hour_utc,
    COUNT(*) as cnt,
    ROUND(100.0*SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END)/COUNT(*),1) as wr,
    ROUND(SUM(result_profit)) as total_pnl
FROM trades
WHERE symbol LIKE '%#%'
  AND status = 'CLOSED'
GROUP BY sym, hour_utc
ORDER BY sym, hour_utc
""")
rows = cur.fetchall()
conn.close()

current_sym = None
bad_hours: dict = {}
for sym, h, cnt, wr, pnl in rows:
    if sym != current_sym:
        print(f"\n=== {sym} ===")
        print(f"  {'UTC':<5} {'Cnt':>4} {'WR%':>6} {'PnL':>10}")
        current_sym = sym
        bad_hours[sym] = []
    flag = ""
    if wr is not None and cnt >= 3:
        if wr < 30:
            flag = " *** BLOCK候補"
            bad_hours[sym].append(h)
        elif wr < 40:
            flag = "  * 要注意"
    print(f"  UTC{h:02d} {cnt:>4}件  WR={wr:>5}%  PnL={pnl:>+10.0f}{flag}")

print("\n\n=== セッションフィルター候補 (WR<30% かつ 3件以上) ===")
for sym, hours in bad_hours.items():
    if hours:
        print(f"  {sym}: ブロック候補UTC時間 = {sorted(hours)}")
