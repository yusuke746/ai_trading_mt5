"""
改善案の定量シミュレーション
「もし〇〇フィルターをかけていたら」の損益変化を計算する。
修正は一切しない。調査専用。
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

DB = r"C:\Users\user\openHands-test\ai_trading_mt5\trades.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

SEP = "=" * 65

def fetch_all():
    cur = conn.cursor()
    cur.execute("""
        SELECT id, opened_at, closed_at, symbol, direction,
               entry_price, exit_price, sl_price, tp_price,
               result_profit, exit_reason, ai_confidence,
               entry_type, market_regime,
               smc_sweep_pass, smc_bos_pass, smc_rr_pass
        FROM trades WHERE status='CLOSED'
        ORDER BY opened_at ASC
    """)
    return [dict(r) for r in cur.fetchall()]

trades = fetch_all()
base_pnl = sum(t["result_profit"] or 0 for t in trades)

def simulate(label: str, excluded: set):
    """excluded: 除外するトレードのid集合"""
    pnl = sum(t["result_profit"] or 0 for t in trades if t["id"] not in excluded)
    cnt_removed = len(excluded)
    cnt_remaining = len(trades) - cnt_removed
    diff = pnl - base_pnl
    pnl_removed = sum(t["result_profit"] or 0 for t in trades if t["id"] in excluded)
    wins_remaining = sum(1 for t in trades if t["id"] not in excluded and (t["result_profit"] or 0) > 0)
    wr = wins_remaining / cnt_remaining * 100 if cnt_remaining else 0
    print(f"\n{'─'*65}")
    print(f"[{label}]")
    print(f"  除外トレード数: {cnt_removed}件  残り: {cnt_remaining}件")
    print(f"  除外されたPnL合計: {pnl_removed:>+10,.0f} JPY")
    print(f"  残存PnL: {pnl:>+10,.0f} JPY  (変化: {diff:>+10,.0f} JPY)")
    print(f"  残存勝率: {wr:.1f}%")
    return pnl

print(SEP)
print(f"ベースライン: 全{len(trades)}件  PnL = {base_pnl:,.0f} JPY")
print(SEP)

# ─────────────────────────────────────────
# A. REVERSAL_SWEEP + bos_pass=False の除外
# ─────────────────────────────────────────
ex_A = {t["id"] for t in trades
        if t["entry_type"] == "REVERSAL_SWEEP"
        and t["smc_bos_pass"] == 0}
simulate("A. REVERSAL_SWEEP + bos_pass=False を除外", ex_A)

# ─────────────────────────────────────────
# B. Confidence < 78 の除外
# ─────────────────────────────────────────
ex_B = {t["id"] for t in trades
        if t["ai_confidence"] is not None and t["ai_confidence"] < 78}
simulate("B. AI confidence < 78 を除外", ex_B)

# ─────────────────────────────────────────
# C. EMERGENCY exit の除外（EMERGENCY起因のみ）
# ─────────────────────────────────────────
ex_C = {t["id"] for t in trades if t["exit_reason"] == "EMERGENCY"}
simulate("C. EMERGENCY exitトレードを除外（仮にSLヒットで代替）", ex_C)

# ─────────────────────────────────────────
# D. 時間帯フィルター: UTC 03, 04, 07, 18, 19 除外
# ─────────────────────────────────────────
bad_hours = {3, 4, 7, 18, 19, 20, 21}
def get_hour(t):
    oa = t.get("opened_at") or ""
    try:
        return datetime.fromisoformat(oa).hour
    except Exception:
        return -1

ex_D = {t["id"] for t in trades if get_hour(t) in bad_hours}
simulate(f"D. UTC {sorted(bad_hours)}時台のエントリーを除外", ex_D)

# ─────────────────────────────────────────
# E. DOWN regime でのエントリー除外
# ─────────────────────────────────────────
ex_E = {t["id"] for t in trades if t["market_regime"] == "DOWN"}
simulate("E. DOWN regime のエントリーを全除外", ex_E)

# ─────────────────────────────────────────
# F. DOWN regime のBUYエントリーのみ除外
# ─────────────────────────────────────────
ex_F = {t["id"] for t in trades
        if t["market_regime"] == "DOWN" and t["direction"] == "BUY"}
simulate("F. DOWN regime の BUYエントリーのみ除外", ex_F)

# ─────────────────────────────────────────
# G. 複合: A + B (bos_pass=False + conf<78)
# ─────────────────────────────────────────
ex_G = ex_A | ex_B
simulate("G. 複合: A+B (bos=False除外 + conf<78除外)", ex_G)

# ─────────────────────────────────────────
# H. 複合: A + B + D (bos=False + conf<78 + 悪時間)
# ─────────────────────────────────────────
ex_H = ex_A | ex_B | ex_D
simulate("H. 複合: A+B+D (bos=False + conf<78 + 悪時間)", ex_H)

# ─────────────────────────────────────────
# I. 複合: A + B + D + F (最適フィルター)
# ─────────────────────────────────────────
ex_I = ex_A | ex_B | ex_D | ex_F
simulate("I. 複合: A+B+D+F (bos=False + conf<78 + 悪時間 + DOWN-BUY除外)", ex_I)

# ─────────────────────────────────────────
# J. RR 1.5-2.0 除外（異常に0%WRのバンド）
# ─────────────────────────────────────────
def get_rr(t):
    ep = t.get("entry_price") or 0
    sl = t.get("sl_price") or 0
    tp = t.get("tp_price") or 0
    sl_dist = abs(ep - sl)
    tp_dist = abs(tp - ep)
    if sl_dist > 0:
        return tp_dist / sl_dist
    return None

ex_J = {t["id"] for t in trades
        if get_rr(t) is not None and 1.5 <= get_rr(t) < 2.0}
simulate("J. 設定RR 1.5-2.0 のエントリーを除外", ex_J)

# ─────────────────────────────────────────
# K. SYMBOL_MISMATCH_AUTO_CLOSE + UNKNOWN_AUTO_CLOSE 除外（システムバグ）
# ─────────────────────────────────────────
ex_K = {t["id"] for t in trades
        if t["exit_reason"] in ("SYMBOL_MISMATCH_AUTO_CLOSE", "UNKNOWN_AUTO_CLOSE")}
simulate("K. SYMBOL_MISMATCH / UNKNOWN 自動クローズを除外（システムバグ）", ex_K)

# ─────────────────────────────────────────
# L. 非#シンボル（旧銘柄）を除外
# ─────────────────────────────────────────
ex_L = {t["id"] for t in trades if not t["symbol"].endswith("#")}
simulate("L. 非#シンボル（EURUSD, USDJPY等の旧銘柄）を除外", ex_L)

# ─────────────────────────────────────────
# M. 最終複合: L + B + A + D（旧銘柄除外 + 強化フィルター）
# ─────────────────────────────────────────
ex_M = ex_L | ex_B | ex_A | ex_D
simulate("M. 複合: 旧銘柄除外 + conf<78除外 + bos=False除外 + 悪時間除外", ex_M)

# ─────────────────────────────────────────
# N. #シンボルのみ conf>=78 + bos_pass=True 限定
# ─────────────────────────────────────────
keep_N = {t["id"] for t in trades
          if t["symbol"].endswith("#")
          and (t["ai_confidence"] or 0) >= 78
          and t["smc_bos_pass"] == 1}
ex_N = {t["id"] for t in trades if t["id"] not in keep_N}
simulate("N. 【最厳格】#シンボル + conf>=78 + bos=True のみ残す", ex_N)

print(f"\n{SEP}")
print("【再確認用】BOS=Trueの REVERSAL_SWEEP 詳細（銘柄別 × confidence）")
print(SEP)
cur = conn.cursor()
cur.execute("""
    SELECT symbol,
           CASE WHEN ai_confidence >= 80 THEN '80+'
                WHEN ai_confidence >= 78 THEN '78-79'
                WHEN ai_confidence >= 75 THEN '75-77'
                ELSE '<75' END as conf_band,
           COUNT(*) as cnt,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(result_profit), 0) as pnl
    FROM trades
    WHERE status='CLOSED' AND entry_type='REVERSAL_SWEEP' AND smc_bos_pass=1
    GROUP BY symbol, conf_band
    ORDER BY symbol, conf_band DESC
""")
for r in cur.fetchall():
    r = dict(r)
    wr = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] else 0
    print(f"  {r['symbol']:<15} conf{r['conf_band']:<7}  cnt={r['cnt']:3d}  WR={wr:5.1f}%  pnl={r['pnl']:>10,.0f}")

print(f"\n{SEP}")
print("【再確認用】#シンボル + BOS=True + conf>=78 の詳細")
print(SEP)
cur.execute("""
    SELECT symbol, direction, market_regime,
           COUNT(*) as cnt,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(result_profit), 0) as pnl,
           ROUND(AVG(CASE WHEN result_profit > 0 THEN result_profit END), 0) as avg_win,
           ROUND(AVG(CASE WHEN result_profit <= 0 THEN result_profit END), 0) as avg_loss
    FROM trades
    WHERE status='CLOSED'
      AND symbol LIKE '%#%'
      AND ai_confidence >= 78
      AND smc_bos_pass = 1
    GROUP BY symbol, direction, market_regime
    ORDER BY symbol, direction, market_regime
""")
for r in cur.fetchall():
    r = dict(r)
    wr = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] else 0
    avg_win = r["avg_win"] or 0
    avg_loss = r["avg_loss"] or -1
    print(f"  {r['symbol']:<15} {r['direction']} {str(r['market_regime']):<10}  "
          f"cnt={r['cnt']:3d}  WR={wr:5.1f}%  pnl={r['pnl']:>10,.0f}  "
          f"avg_W={avg_win:>8,.0f}  avg_L={avg_loss:>8,.0f}")

print(f"\n{SEP}")
print("完了")
print(SEP)
conn.close()
