"""regime×direction 整合性チェック (#シンボル)"""
import sqlite3, os

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db")
conn = sqlite3.connect(DB)
cur = conn.cursor()

print("=== market_regime × direction × entry_type 別WR (#シンボル) ===")
cur.execute("""
SELECT
    market_regime,
    direction,
    entry_type,
    COUNT(*) as cnt,
    ROUND(100.0*SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END)/COUNT(*),1) as wr,
    ROUND(SUM(result_profit)) as pnl,
    ROUND(AVG(result_profit)) as avg_pnl
FROM trades
WHERE symbol LIKE '%#%' AND status='CLOSED'
GROUP BY market_regime, direction, entry_type
ORDER BY market_regime, direction, cnt DESC
""")
rows = cur.fetchall()
for reg, dir_, et, cnt, wr, pnl, avg in rows:
    flag = ""
    # regimeとdirectionが逆行している場合
    if reg == "DOWN" and dir_ == "BUY":
        flag = " *** REGIME逆行(DOWN+BUY)"
    elif reg == "UP" and dir_ == "SELL":
        flag = " *** REGIME逆行(UP+SELL)"
    print(f"  {str(reg):<12} {str(dir_):<6} {str(et):<20} cnt={cnt:>3} WR={str(wr):>5}% PnL={pnl:>+10.0f} avg={avg:>+8.0f}{flag}")

print("\n=== regime逆行エントリー vs 整合エントリー 集計 ===")
cur.execute("""
SELECT
    CASE
        WHEN (market_regime='DOWN' AND direction='BUY') OR (market_regime='UP' AND direction='SELL')
        THEN 'AGAINST_REGIME'
        WHEN market_regime='SIDEWAYS'
        THEN 'SIDEWAYS'
        ELSE 'WITH_REGIME'
    END as regime_align,
    COUNT(*) as cnt,
    ROUND(100.0*SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END)/COUNT(*),1) as wr,
    ROUND(SUM(result_profit)) as pnl,
    ROUND(AVG(result_profit)) as avg_pnl
FROM trades
WHERE symbol LIKE '%#%' AND status='CLOSED'
  AND market_regime IS NOT NULL AND market_regime != ''
GROUP BY regime_align
ORDER BY regime_align
""")
for align, cnt, wr, pnl, avg in cur.fetchall():
    print(f"  {align:<20} cnt={cnt:>3} WR={str(wr):>5}% PnL={pnl:>+10.0f} avg={avg:>+8.0f}")

print("\n=== REVERSAL_SWEEPのregime整合性 (逆張りはregimeに逆らって当然) ===")
cur.execute("""
SELECT
    CASE
        WHEN (market_regime='DOWN' AND direction='BUY') OR (market_regime='UP' AND direction='SELL')
        THEN 'REVERSAL(regime逆行)'
        WHEN market_regime='SIDEWAYS'
        THEN 'REVERSAL(SIDEWAYS)'
        ELSE 'REVERSAL(regime同行)'
    END as type_,
    COUNT(*) as cnt,
    ROUND(100.0*SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END)/COUNT(*),1) as wr,
    ROUND(SUM(result_profit)) as pnl
FROM trades
WHERE symbol LIKE '%#%' AND status='CLOSED'
  AND entry_type='REVERSAL_SWEEP'
GROUP BY type_
""")
for t, cnt, wr, pnl in cur.fetchall():
    print(f"  {t:<30} cnt={cnt:>3} WR={str(wr):>5}% PnL={pnl:>+10.0f}")

conn.close()
