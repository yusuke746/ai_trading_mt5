import sqlite3

conn = sqlite3.connect(r"C:\Users\user\openHands-test\ai_trading_mt5\trades.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== SL_HIT 詳細：SL幅・RR・conf ===")
cur.execute("""
    SELECT symbol, direction, entry_price, sl_price, tp_price, lot_size,
           result_profit, entry_type, closed_at, ai_confidence
    FROM trades
    WHERE status='CLOSED' AND exit_reason='SL_HIT'
    ORDER BY closed_at DESC
""")
rows = cur.fetchall()
for r in rows:
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    tp = r["tp_price"] or 0
    sl_d = abs(ep - sl) if sl else 0
    tp_d = abs(tp - ep) if tp else 0
    rr = tp_d / sl_d if sl_d > 0 else 0
    print(f'  {(r["closed_at"] or "")[:16]}  {r["symbol"]:12} {r["direction"]:4}'
          f'  SL_dist={sl_d:8.4f}  RR={rr:.2f}  lot={r["lot_size"]:.2f}'
          f'  conf={r["ai_confidence"]}  pnl={r["result_profit"]:8.0f}  type={r["entry_type"]}')

print()
print("=== SL_HIT vs CONTINUATION/REVERSAL 集計 ===")
cur.execute("""
    SELECT entry_type, COUNT(*) as cnt,
           ROUND(AVG(result_profit),0) as avg_pnl,
           SUM(result_profit) as total_pnl,
           ROUND(AVG(ABS(entry_price - sl_price)),4) as avg_sl_dist
    FROM trades
    WHERE status='CLOSED' AND exit_reason='SL_HIT'
    GROUP BY entry_type
""")
for r in cur.fetchall():
    print(f'  {str(r["entry_type"]):25} cnt={r["cnt"]:2}  avg_sl={r["avg_sl_dist"]}  avg_pnl={r["avg_pnl"]}  total={r["total_pnl"]}')

print()
print("=== 時系列SL幅推移 GOLD# CONTINUATION_BOS ===")
cur.execute("""
    SELECT opened_at, closed_at, entry_price, sl_price, result_profit, exit_reason, ai_confidence
    FROM trades
    WHERE status='CLOSED' AND symbol LIKE 'GOLD%' AND entry_type='CONTINUATION_BOS'
    ORDER BY opened_at
""")
for r in cur.fetchall():
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    sl_d = abs(ep - sl)
    mark = " <-- SL_HIT" if r["exit_reason"] == "SL_HIT" else ""
    print(f'  {(r["opened_at"] or "")[:16]}  SL_dist={sl_d:6.2f}  conf={r["ai_confidence"]}'
          f'  exit={str(r["exit_reason"]):25}  pnl={r["result_profit"]:8.0f}{mark}')

print()
print("=== 時系列SL幅推移 OILCash# (SL_HIT重点) ===")
cur.execute("""
    SELECT opened_at, closed_at, entry_price, sl_price, result_profit, exit_reason, ai_confidence, entry_type
    FROM trades
    WHERE status='CLOSED' AND symbol LIKE 'OILCash%'
    ORDER BY opened_at
""")
for r in cur.fetchall():
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    sl_d = abs(ep - sl)
    mark = " <-- SL_HIT" if r["exit_reason"] == "SL_HIT" else ""
    print(f'  {(r["opened_at"] or "")[:16]}  SL_dist={sl_d:6.4f}  conf={r["ai_confidence"]}'
          f'  exit={str(r["exit_reason"]):25}  pnl={r["result_profit"]:8.0f}  type={r["entry_type"]}{mark}')

print()
print("=== SL幅ヒストグラム（銘柄別・exit_reason別） ===")
cur.execute("""
    SELECT symbol, exit_reason,
           ROUND(AVG(ABS(entry_price - sl_price)), 4) as avg_sl,
           ROUND(MIN(ABS(entry_price - sl_price)), 4) as min_sl,
           ROUND(MAX(ABS(entry_price - sl_price)), 4) as max_sl,
           COUNT(*) as cnt
    FROM trades
    WHERE status='CLOSED' AND symbol LIKE '%#'
    GROUP BY symbol, exit_reason
    ORDER BY symbol, exit_reason
""")
print(f"  {'symbol':12} {'exit_reason':25} cnt  avg_sl    min_sl    max_sl")
for r in cur.fetchall():
    print(f'  {r["symbol"]:12} {str(r["exit_reason"]):25} {r["cnt"]:3}  {r["avg_sl"]:9.4f}  {r["min_sl"]:9.4f}  {r["max_sl"]:9.4f}')

print()
print("=== 直近10件 全シンボル詳細（SL幅確認） ===")
cur.execute("""
    SELECT symbol, direction, entry_price, sl_price, tp_price, lot_size,
           result_profit, entry_type, exit_reason, opened_at, closed_at, ai_confidence
    FROM trades
    WHERE status='CLOSED' AND symbol LIKE '%#'
    ORDER BY closed_at DESC LIMIT 10
""")
for r in cur.fetchall():
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    tp = r["tp_price"] or 0
    sl_d = abs(ep - sl) if sl else 0
    tp_d = abs(tp - ep) if tp else 0
    rr = tp_d / sl_d if sl_d > 0 else 0
    print(f'  {(r["closed_at"] or "")[:16]}  {r["symbol"]:12} {r["direction"]:4}'
          f'  SL={sl_d:8.4f}  RR={rr:.2f}  conf={r["ai_confidence"]}'
          f'  pnl={r["result_profit"]:8.0f}  {str(r["exit_reason"]):20}  {r["entry_type"]}')

conn.close()
