import sqlite3
conn = sqlite3.connect(r'C:\Users\user\openHands-test\ai_trading_mt5\trades.db')
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)
for t in tables:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f"  {t}: {cols}")
