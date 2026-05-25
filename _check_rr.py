import sqlite3

DB = r"C:\Users\user\openHands-test\ai_trading_mt5\trades.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("SELECT MIN(opened_at), MAX(opened_at) FROM trades")
r = cur.fetchone()
print("全体: first=", r[0], "last=", r[1])

cur.execute("SELECT MIN(opened_at), MAX(opened_at) FROM trades WHERE symbol LIKE '%#%'")
r2 = cur.fetchone()
print("#付き: first=", r2[0], "last=", r2[1])

print("\nsmc_rr_pass=0 内訳 (entry_type, bos_pass, cnt):")
cur.execute("SELECT entry_type, smc_bos_pass, COUNT(*) FROM trades WHERE smc_rr_pass=0 GROUP BY entry_type, smc_bos_pass")
for row in cur.fetchall():
    print(" ", row)

print("\nsmc_rr_pass=0 銘柄別:")
cur.execute("SELECT symbol, COUNT(*), SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END) FROM trades WHERE smc_rr_pass=0 GROUP BY symbol ORDER BY COUNT(*) DESC")
for row in cur.fetchall():
    print(" ", row)

conn.close()
