import sqlite3
conn = sqlite3.connect("trades.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# RR 1.5-2.0 の exit_reason 内訳
cur.execute("""
  SELECT exit_reason, symbol, entry_type, smc_bos_pass,
         COUNT(*) cnt, ROUND(SUM(result_profit),0) pnl,
         ROUND(AVG(result_profit),0) avg_pnl
  FROM trades WHERE status='CLOSED'
    AND ABS(tp_price - entry_price) / NULLIF(ABS(sl_price - entry_price), 0) BETWEEN 1.5 AND 2.0
  GROUP BY exit_reason, symbol, entry_type, smc_bos_pass
  ORDER BY pnl ASC
""")
print("=== RR 1.5-2.0 exit_reason x symbol x bos_pass ===")
for r in cur.fetchall():
    d = dict(r)
    print(f"  exit={d['exit_reason']:<20} symbol={d['symbol']:<15} "
          f"entry={d['entry_type']:<20} bos={d['smc_bos_pass']}  "
          f"cnt={d['cnt']:3d}  pnl={d['pnl']:>10,}  avg={d['avg_pnl']:>8,}")

print()
# CONTINUATION_BOS の conf 分布
cur.execute("""
  SELECT ai_confidence, COUNT(*) cnt,
         SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END) wins,
         ROUND(SUM(result_profit),0) pnl
  FROM trades WHERE status='CLOSED' AND entry_type='CONTINUATION_BOS'
  GROUP BY ai_confidence ORDER BY ai_confidence
""")
print("=== CONTINUATION_BOS conf分布（76以上が通過できている） ===")
for r in cur.fetchall():
    d = dict(r)
    wr = round(d["wins"] / d["cnt"] * 100, 1) if d["cnt"] else 0
    marker = " ← conf<78で通過" if (d["ai_confidence"] or 0) < 78 else ""
    print(f"  conf={d['ai_confidence']:3d}  cnt={d['cnt']:3d}  WR={wr:5.1f}%  pnl={d['pnl']:>10,}{marker}")

print()
# SIDEWAYS/REVERSAL_SWEEP の conf 分布
cur.execute("""
  SELECT ai_confidence, COUNT(*) cnt,
         SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END) wins,
         ROUND(SUM(result_profit),0) pnl
  FROM trades WHERE status='CLOSED'
    AND entry_type='REVERSAL_SWEEP' AND (market_regime='SIDEWAYS' OR market_regime IS NULL)
  GROUP BY ai_confidence ORDER BY ai_confidence
""")
print("=== SIDEWAYS/REVERSAL_SWEEP conf分布（bucketなし→global=76） ===")
for r in cur.fetchall():
    d = dict(r)
    wr = round(d["wins"] / d["cnt"] * 100, 1) if d["cnt"] else 0
    marker = " ← conf<78で通過" if (d["ai_confidence"] or 0) < 78 else ""
    print(f"  conf={d['ai_confidence']:3d}  cnt={d['cnt']:3d}  WR={wr:5.1f}%  pnl={d['pnl']:>10,}{marker}")

conn.close()
print("\n完了")
