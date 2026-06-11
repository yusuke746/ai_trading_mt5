import sqlite3
conn = sqlite3.connect(r"C:\Users\user\openHands-test\ai_trading_mt5\trades.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== SL_HIT: SL幅小さい件 (ATR×1.0未満推定) ===")
# GOLD M15 ATR≈12, OILCash≈1.0, USDJPY≈0.15, EURUSD≈0.0015, US100Cash≈30
atr_est = {"GOLD#": 12.0, "OILCash#": 1.0, "USDJPY#": 0.15, "EURUSD#": 0.0015, "US100Cash#": 30.0}
cur.execute("""
    SELECT symbol, entry_type, entry_price, sl_price, result_profit, closed_at
    FROM trades
    WHERE status='CLOSED' AND exit_reason='SL_HIT' AND symbol LIKE '%#'
    ORDER BY closed_at DESC
""")
for r in cur.fetchall():
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    sl_d = abs(ep - sl)
    atr = atr_est.get(r["symbol"], 1.0)
    atr_mult = sl_d / atr if atr > 0 else 0
    blocked = "→ 1.0倍設定でもSL_HIT" if atr_mult >= 1.0 else "→ ★ 1.0倍なら除外可能"
    print(f"  {r['symbol']:12} {str(r['entry_type']):25}  SL={sl_d:.4f}  ATR推定×{atr_mult:.2f}  {blocked}")

print()
print("=== CONTINUATION_BOS: RR帯×conf×exit分布 ===")
cur.execute("""
    SELECT
        CASE
            WHEN CAST(ABS(tp_price-entry_price) AS REAL)/CAST(ABS(entry_price-sl_price) AS REAL) < 1.2 THEN 'RR<1.2'
            WHEN CAST(ABS(tp_price-entry_price) AS REAL)/CAST(ABS(entry_price-sl_price) AS REAL) < 1.5 THEN 'RR1.2-1.5'
            ELSE 'RR>=1.5'
        END as rr_band,
        ai_confidence, exit_reason,
        COUNT(*) as cnt, SUM(result_profit) as total_pnl
    FROM trades
    WHERE status='CLOSED' AND entry_type='CONTINUATION_BOS' AND symbol LIKE '%#'
      AND sl_price IS NOT NULL AND tp_price IS NOT NULL AND entry_price IS NOT NULL
    GROUP BY rr_band, ai_confidence, exit_reason
    ORDER BY rr_band, ai_confidence
""")
print(f"  {'RR帯':<12} {'conf':<6} {'exit_reason':<25} cnt  total_pnl")
for r in cur.fetchall():
    print(f"  {r['rr_band']:<12} {str(r['ai_confidence']):<6} {str(r['exit_reason']):<25} {r['cnt']:3}  {r['total_pnl']:9.0f}")

# SMC_MECHANICAL_RR_RELAX_FACTOR の影響: RELAX=0.8 vs 1.0 でブロック件数
print()
print("=== RR緩和係数別: 除外されるトレード数 (CONTINUATION_BOS) ===")
print("  ※ 機械ゲートは構造的SLでのTP/SL比をチェックしている想定")
cur.execute("""
    SELECT
        CAST(ABS(tp_price-entry_price) AS REAL)/CAST(ABS(entry_price-sl_price) AS REAL) as rr,
        exit_reason, result_profit, symbol, ai_confidence
    FROM trades
    WHERE status='CLOSED' AND entry_type='CONTINUATION_BOS' AND symbol LIKE '%#'
      AND sl_price IS NOT NULL AND tp_price IS NOT NULL AND entry_price IS NOT NULL
    ORDER BY rr
""")
rows = cur.fetchall()
for thresh_label, thresh in [("RELAX=1.0 (RR>=1.0必須)", 1.0), ("RELAX=0.9 (RR>=0.9必須)", 0.9), ("RELAX=0.8 (現状)", 0.8)]:
    blocked = [r for r in rows if r["rr"] < thresh]
    passed  = [r for r in rows if r["rr"] >= thresh]
    bl_pnl  = sum(r["result_profit"] or 0 for r in blocked)
    pa_pnl  = sum(r["result_profit"] or 0 for r in passed)
    bl_sl   = sum(1 for r in blocked if r["exit_reason"] == "SL_HIT")
    pa_sl   = sum(1 for r in passed if r["exit_reason"] == "SL_HIT")
    print(f"  {thresh_label}: ブロック={len(blocked)}件(SL_HIT={bl_sl}, pnl={bl_pnl:+.0f})  通過={len(passed)}件(SL_HIT={pa_sl}, pnl={pa_pnl:+.0f})")

conn.close()
