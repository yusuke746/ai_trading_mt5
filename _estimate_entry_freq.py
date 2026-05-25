"""Fix1〜5 適用後の銘柄別エントリー頻度推定"""
import sqlite3, os
from datetime import datetime

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db")
conn = sqlite3.connect(DB)
cur = conn.cursor()

# ── 観測期間の確認 ──────────────────────────────────────────────────────────────
cur.execute("""
SELECT
    MIN(opened_at) as first_trade,
    MAX(opened_at) as last_trade,
    COUNT(*) as total
FROM trades WHERE symbol LIKE '%#%' AND status='CLOSED'
""")
row = cur.fetchone()
first_dt = datetime.fromisoformat(row[0]) if row[0] else None
last_dt  = datetime.fromisoformat(row[1]) if row[1] else None
total    = row[2]
obs_days = max(1, (last_dt - first_dt).days + 1) if first_dt and last_dt else 1
obs_weeks = obs_days / 7

print(f"観測期間: {row[0][:10]} 〜 {row[1][:10]} ({obs_days}日間 = {obs_weeks:.1f}週)")
print(f"#シンボル クローズ済み: {total}件 → {total/obs_days:.2f}件/日 / {total/obs_weeks:.1f}件/週\n")

# ── 銘柄別 基礎頻度 ──────────────────────────────────────────────────────────────
print("=" * 65)
print("銘柄別 エントリー頻度 (修正前の実績)")
print("=" * 65)
cur.execute("""
SELECT symbol, COUNT(*) as cnt,
       ROUND(100.0*SUM(CASE WHEN result_profit>0 THEN 1 ELSE 0 END)/COUNT(*),1) as wr,
       ROUND(SUM(result_profit)) as pnl
FROM trades WHERE symbol LIKE '%#%' AND status='CLOSED'
GROUP BY symbol ORDER BY cnt DESC
""")
sym_base: dict[str, int] = {}
for sym, cnt, wr, pnl in cur.fetchall():
    sym_base[sym] = cnt
    print(f"  {sym:<14} {cnt:>3}件  {cnt/obs_days:>4.2f}件/日  {cnt/obs_weeks:>5.1f}件/週  WR={wr}%  PnL={pnl:>+10.0f}")

# ── Fix別 ブロック件数推定 ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Fix別 ブロック件数 (#シンボル)")
print("=" * 65)

# Fix1: bos_pass=False の REVERSAL_SWEEP
cur.execute("""
SELECT symbol, COUNT(*) as cnt FROM trades
WHERE symbol LIKE '%#%' AND status='CLOSED'
  AND entry_type='REVERSAL_SWEEP' AND smc_bos_pass=0
GROUP BY symbol
""")
fix1: dict[str, int] = {r[0]: r[1] for r in cur.fetchall()}
fix1_total = sum(fix1.values())
print(f"\n  Fix1 (bos=False REVERSAL_SWEEP ブロック):")
for sym, cnt in fix1.items():
    print(f"    {sym:<14} -{cnt:>2}件")
print(f"    計 -{fix1_total}件 ({100*fix1_total/total:.1f}%)")

# Fix2+3: conf < 78
cur.execute("""
SELECT symbol, COUNT(*) as cnt FROM trades
WHERE symbol LIKE '%#%' AND status='CLOSED'
  AND (ai_confidence IS NOT NULL AND ai_confidence < 78)
GROUP BY symbol
""")
fix23: dict[str, int] = {r[0]: r[1] for r in cur.fetchall()}
fix23_total = sum(fix23.values())
print(f"\n  Fix2+3 (conf<78 ブロック):")
for sym, cnt in fix23.items():
    print(f"    {sym:<14} -{cnt:>2}件")
print(f"    計 -{fix23_total}件 ({100*fix23_total/total:.1f}%)")

# Fix5: SIDEWAYS + REVERSAL_SWEEP
cur.execute("""
SELECT symbol, COUNT(*) as cnt FROM trades
WHERE symbol LIKE '%#%' AND status='CLOSED'
  AND entry_type='REVERSAL_SWEEP' AND market_regime='SIDEWAYS'
GROUP BY symbol
""")
fix5: dict[str, int] = {r[0]: r[1] for r in cur.fetchall()}
fix5_total = sum(fix5.values())
print(f"\n  Fix5 (SIDEWAYS+REVERSAL_SWEEP ブロック):")
for sym, cnt in fix5.items():
    print(f"    {sym:<14} -{cnt:>2}件")
print(f"    計 -{fix5_total}件 ({100*fix5_total/total:.1f}%)")

# Fix4: ADX<20 → DBに記録なし。閑散相場の頻度から推定
# 一般的なFX/CFDでADX<20は全足の約30-40%。但し既存のATR volatility filterが類似機能を持つ
# → 保守的に10-15%の追加ブロックを見込む
adx_block_rate = 0.12  # 12%推定 (ATR filterと重複分を差し引いた純増分)
fix4_estimated = round(total * adx_block_rate)
print(f"\n  Fix4 (ADX<20 ブロック・推定):")
print(f"    推定 ~{fix4_estimated}件 (全体の約{adx_block_rate*100:.0f}%  ※ATR filterとの重複後の純増推定)")

# ── 重複除去 (Fix1 + Fix5 はどちらも bos_pass=False か SIDEWAYS なので一部重複の可能性) ──
# Fix1/Fix5は互いに独立 (bos_pass vs regime)、Fix2+3はその後のconfフィルター
# 厳密な重複チェック
cur.execute("""
SELECT COUNT(*) FROM trades
WHERE symbol LIKE '%#%' AND status='CLOSED'
  AND (
    (entry_type='REVERSAL_SWEEP' AND smc_bos_pass=0)  -- Fix1
    OR (ai_confidence < 78)                             -- Fix2+3
    OR (entry_type='REVERSAL_SWEEP' AND market_regime='SIDEWAYS')  -- Fix5
  )
""")
total_blocked_any = cur.fetchone()[0]

print(f"\n  重複除去後 合計ブロック (Fix1+2+3+5 の和集合):")
print(f"    -{total_blocked_any}件 ({100*total_blocked_any/total:.1f}%)")
print(f"    (Fix4推定{fix4_estimated}件を加えると -{total_blocked_any+fix4_estimated}件, "
      f"{100*(total_blocked_any+fix4_estimated)/total:.1f}%)")

# ── 銘柄別 修正後の推定頻度 ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("銘柄別 推定エントリー頻度 (Fix1〜5 適用後)")
print("=" * 65)
print(f"  {'銘柄':<14} {'実績':>4} {'ブロック':>6} {'残存':>4} {'日頻度':>6} {'週頻度':>6}  {'備考'}")
print(f"  {'-'*63}")

for sym in sym_base:
    base  = sym_base[sym]
    # 銘柄別ブロック数 (Fix1+2+3+5 の和集合)
    cur.execute("""
    SELECT COUNT(*) FROM trades
    WHERE symbol=? AND status='CLOSED'
      AND (
        (entry_type='REVERSAL_SWEEP' AND smc_bos_pass=0)
        OR (ai_confidence < 78)
        OR (entry_type='REVERSAL_SWEEP' AND market_regime='SIDEWAYS')
      )
    """, (sym,))
    blocked = cur.fetchone()[0]
    # Fix4推定 (比例配分)
    fix4_sym = round(base * adx_block_rate)
    total_block = min(base, blocked + fix4_sym)
    remaining = max(0, base - total_block)
    per_day  = remaining / obs_days
    per_week = remaining / obs_weeks
    note = ""
    if per_week < 2:
        note = "← 週2件未満・要注意"
    elif per_week > 8:
        note = "← 頻度高め"
    print(f"  {sym:<14} {base:>4}件  -{total_block:>4}件  {remaining:>4}件  "
          f"{per_day:>5.2f}/日  {per_week:>5.1f}/週  {note}")

cur.execute("""
SELECT COUNT(*) FROM trades
WHERE symbol LIKE '%#%' AND status='CLOSED'
  AND NOT (
    (entry_type='REVERSAL_SWEEP' AND smc_bos_pass=0)
    OR (ai_confidence < 78)
    OR (entry_type='REVERSAL_SWEEP' AND market_regime='SIDEWAYS')
  )
""")
remaining_total = cur.fetchone()[0]
fix4_all = round(remaining_total * adx_block_rate)
final_total = max(0, remaining_total - fix4_all)
print(f"\n  合計  (実績){total}件 → (推定残存){final_total}件")
print(f"  推定頻度: {final_total/obs_days:.2f}件/日 / {final_total/obs_weeks:.1f}件/週")
print(f"  (修正前比: {100*final_total/total:.0f}%のエントリーが通過)")

conn.close()
