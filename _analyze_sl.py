import sqlite3

db_path = r"C:\Users\user\openHands-test\ai_trading_mt5\trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 銘柄別集計
cur.execute("""
    SELECT symbol,
           COUNT(*) as cnt,
           SUM(CASE WHEN result_profit < 0 THEN 1 ELSE 0 END) as losses,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(AVG(result_profit),0) as avg_pnl,
           ROUND(AVG(CASE WHEN result_profit < 0 THEN result_profit END),0) as avg_loss,
           ROUND(AVG(CASE WHEN result_profit > 0 THEN result_profit END),0) as avg_win,
           ROUND(AVG(CASE WHEN exit_reason='SL_HIT' THEN result_profit END),0) as avg_sl_hit_loss,
           SUM(CASE WHEN exit_reason='SL_HIT' THEN 1 ELSE 0 END) as sl_hit_cnt
    FROM trades WHERE status='CLOSED' GROUP BY symbol ORDER BY symbol
""")
print("=== Symbol Stats ===")
for r in cur.fetchall():
    win_rate = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] > 0 else 0
    print(f"  {r['symbol']:12} cnt={r['cnt']:3} W={r['wins']:3} L={r['losses']:3} WR={win_rate}%"
          f" avgLoss={r['avg_loss']}円 avgWin={r['avg_win']}円"
          f" SL_HIT={r['sl_hit_cnt']}件 avg_SLloss={r['avg_sl_hit_loss']}円")

# 最近30件の詳細 (SL幅・RR・損益)
cur.execute("""
    SELECT symbol, direction, lot_size, entry_price, sl_price, tp_price,
           result_profit, exit_reason, entry_type, closed_at
    FROM trades WHERE status='CLOSED'
    ORDER BY closed_at DESC LIMIT 30
""")
print()
print("=== Last 30 Closed Trades (SL幅・RR) ===")
for r in cur.fetchall():
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    tp = r["tp_price"] or 0
    sl_d = abs(ep - sl) if sl and ep else 0
    tp_d = abs(tp - ep) if tp and ep else 0
    rr = round(tp_d / sl_d, 2) if sl_d > 0 else "-"
    print(f"  {r['symbol']:12} {r['direction']:4} lot={r['lot_size']:5.2f}"
          f" SL_dist={sl_d:7.3f} RR={str(rr):4}"
          f" pnl={r['result_profit']:9.0f}円"
          f"  {r['exit_reason']:30} type={r['entry_type'] or '-':20}"
          f" {(r['closed_at'] or '')[:16]}")

# GOLD専用: SL_HITのSL幅分布
print()
print("=== GOLD SL_HIT詳細 ===")
cur.execute("""
    SELECT entry_price, sl_price, lot_size, result_profit, entry_type, closed_at
    FROM trades WHERE status='CLOSED' AND exit_reason='SL_HIT'
    AND symbol LIKE 'GOLD%'
    ORDER BY closed_at DESC LIMIT 20
""")
for r in cur.fetchall():
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    sl_d = abs(ep - sl) if sl else 0
    print(f"  lot={r['lot_size']:5.2f} SL_dist={sl_d:6.3f}$ pnl={r['result_profit']:9.0f}円"
          f" type={r['entry_type'] or '-':20} {(r['closed_at'] or '')[:16]}")

print()
print("=== US100Cash SL_HIT詳細 ===")
cur.execute("""
    SELECT entry_price, sl_price, lot_size, result_profit, entry_type, closed_at
    FROM trades WHERE status='CLOSED' AND exit_reason='SL_HIT'
    AND symbol LIKE 'US100%'
    ORDER BY closed_at DESC LIMIT 20
""")
for r in cur.fetchall():
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    sl_d = abs(ep - sl) if sl else 0
    print(f"  lot={r['lot_size']:5.2f} SL_dist={sl_d:7.3f}pt pnl={r['result_profit']:9.0f}円"
          f" type={r['entry_type'] or '-':20} {(r['closed_at'] or '')[:16]}")

conn.close()
