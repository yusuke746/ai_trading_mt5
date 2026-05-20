import sqlite3

c = sqlite3.connect('trades.db')
print("=== trades columns ===")
for r in c.execute("PRAGMA table_info(trades)").fetchall():
    print(r)
print()
print("=== ai_logs columns ===")
for r in c.execute("PRAGMA table_info(ai_logs)").fetchall():
    print(r)
c.close()
