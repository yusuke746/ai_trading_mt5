import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('trades.db')
conn.row_factory = sqlite3.Row

symbols = ['GOLD#', 'USDJPY#', 'EURUSD#', 'US100Cash#', 'OILCash#']

for sym in symbols:
    rows = conn.execute("SELECT * FROM trades WHERE symbol=? ORDER BY opened_at", (sym,)).fetchall()
    if not rows:
        continue

    bugs_dir = []
    bugs_small = []

    print(f"\n{'='*65}")
    print(f"=== {sym} ===")

    for r in rows:
        d = dict(r)
        entry = d.get('entry_price')
        sl = d.get('sl_price')
        tp = d.get('tp_price')
        profit = d.get('result_profit')
        direction = d.get('direction')
        ticket = d.get('mt5_ticket')
        opened = (d.get('opened_at') or '')[:16]
        exit_r = d.get('exit_reason')

        if entry is None or sl is None:
            continue

        sl_dist = abs(entry - sl)
        tp_dist = abs(entry - tp) if tp else None
        rr = round(tp_dist / sl_dist, 2) if tp_dist and sl_dist > 0 else None

        sl_dir_bug = (direction == 'BUY' and sl > entry) or (direction == 'SELL' and sl < entry)
        sl_pct = sl_dist / entry * 100 if entry > 0 else 0
        sl_tiny = sl_pct < 0.3

        flag = ""
        if sl_dir_bug:
            flag += " [SL方向バグ]"
        if sl_tiny:
            flag += f" [SL極小 {sl_pct:.3f}%]"

        marker = "  ⚠️" if flag else "    "
        print(f"{marker}ticket={ticket} {direction} {opened} entry={round(entry,5)} sl={round(sl,5)} sl_dist={round(sl_dist,5)} RR={rr} exit={exit_r} pnl={profit}{flag}")

        if sl_dir_bug:
            bugs_dir.append(ticket)
        if sl_tiny:
            bugs_small.append(ticket)

    wins = sum(1 for r in rows if (dict(r).get('result_profit') or 0) > 0)
    losses = sum(1 for r in rows if dict(r).get('result_profit') is not None and (dict(r).get('result_profit') or 0) <= 0)
    pnl = sum(dict(r).get('result_profit') or 0 for r in rows)
    print(f"  → W={wins} L={losses} PnL={pnl:,.0f}")
    if bugs_dir:
        print(f"  \U0001f6a8 SL方向バグ tickets: {bugs_dir}")
    if bugs_small:
        print(f"  \u26a0\ufe0f  SL極小 tickets: {bugs_small}")

conn.close()
