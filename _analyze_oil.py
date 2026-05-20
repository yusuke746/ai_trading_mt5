import sqlite3, json, sys

# Set stdout to utf-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

conn = sqlite3.connect('trades.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print('=== OILCash# ALL TRADES ===')
cur.execute("""SELECT mt5_ticket as ticket, opened_at as open_time, closed_at as close_time, 
               direction, lot_size as lot, entry_price, sl_price as sl, tp_price as tp,
               exit_price, exit_reason, result_profit as profit, entry_type as setup_type, market_regime as h1_trend
               FROM trades WHERE symbol='OILCash#' ORDER BY opened_at""")
rows = cur.fetchall()
for r in rows:
    d = dict(r)
    sl_dist = abs(d['entry_price'] - d['sl']) if d['sl'] else None
    tp_dist = abs(d['entry_price'] - d['tp']) if d['tp'] else None
    rr = round(tp_dist / sl_dist, 2) if sl_dist and tp_dist and sl_dist > 0 else None
    print(f"ticket={d['ticket']} {d['direction']} open={d['open_time']} close={d['close_time']}")
    print(f"  entry={d['entry_price']} sl={d['sl']} tp={d['tp']}")
    print(f"  SL_dist={round(sl_dist,4) if sl_dist else None} TP_dist={round(tp_dist,4) if tp_dist else None} RR={rr}")
    print(f"  exit_price={d['exit_price']} exit_reason={d['exit_reason']} profit={d['profit']}")
    print(f"  setup={d['setup_type']} h1_trend={d['h1_trend']}")
    print()

print(f"Total: {len(rows)} trades")
print(f"Wins: {sum(1 for r in rows if r['profit'] and r['profit'] > 0)}")
print(f"Losses: {sum(1 for r in rows if r['profit'] and r['profit'] <= 0)}")
print(f"Total PnL: {sum(r['profit'] for r in rows if r['profit'])}")

print()
print('=== OILCash# AI LOGS (entry decisions) ===')
cur.execute("""SELECT timestamp, action_type as action, decision, reasoning
               FROM ai_logs WHERE symbol='OILCash#'
               ORDER BY timestamp DESC LIMIT 20""")
for r in cur.fetchall():
    print(dict(r))

conn.close()
