# period and rr analysis
import sqlite3

DB = r"C:\Users\user\openHands-test\ai_trading_mt5\trades.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

print("=== 銘柄別クローズトレード数 ===")
cur.execute("""
    SELECT symbol, COUNT(*) as cnt,
           SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN result_profit<=0 THEN 1 ELSE 0 END) as losses,
           ROUND(AVG(result_profit),0) as avg_pnl,
           ROUND(SUM(result_profit),0) as total_pnl
    FROM trades
    WHERE status='CLOSED'
    GROUP BY symbol ORDER BY cnt DESC
""")
for r in cur.fetchall():
    wr = round(r[2]/r[1]*100,1) if r[1] else 0
    print(f"  {r[0]}: total={r[1]}, W={r[2]}, L={r[3]}, WR={wr}%, avg={r[4]}, total_pnl={r[5]}")

cur.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'")
print(f"\n  TOTAL closed: {cur.fetchone()[0]}")
cur.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'")
print(f"  TOTAL open:   {cur.fetchone()[0]}")

print("\n=== 銘柄別 entry_type別 ===")
cur.execute("""
    SELECT symbol, entry_type, COUNT(*) as cnt,
           SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END) as wins
    FROM trades WHERE status='CLOSED'
    GROUP BY symbol, entry_type ORDER BY symbol, cnt DESC
""")
for r in cur.fetchall():
    wr = round(r[3]/r[2]*100,1) if r[2] else 0
    print(f"  {r[0]}, {r[1]}: cnt={r[2]}, W={r[3]}, WR={wr}%")

print("\n=== smc_rr_pass 分布 ===")
cur.execute("""
    SELECT smc_rr_pass, COUNT(*) FROM trades WHERE status='CLOSED' GROUP BY smc_rr_pass
""")
for r in cur.fetchall():
    print(f"  smc_rr_pass={r[0]}: {r[1]}")

print("\n=== AI Confidence分布 ===")
cur.execute("""
    SELECT 
        CASE WHEN ai_confidence < 60 THEN '<60'
             WHEN ai_confidence < 65 THEN '60-64'
             WHEN ai_confidence < 70 THEN '65-69'
             WHEN ai_confidence < 75 THEN '70-74'
             WHEN ai_confidence < 80 THEN '75-79'
             ELSE '80+' END as band,
        COUNT(*), SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END) as wins
    FROM trades WHERE status='CLOSED'
    GROUP BY band ORDER BY band
""")
for r in cur.fetchall():
    wr = round(r[2]/r[1]*100,1) if r[1] else 0
    print(f"  conf={r[0]}: cnt={r[1]}, WR={wr}%")

conn.close()
