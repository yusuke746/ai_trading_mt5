"""
パフォーマンス総合調査スクリプト
利益改善のための詳細分析を行う。修正は一切しない。
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

DB = r"C:\Users\user\openHands-test\ai_trading_mt5\trades.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

SEP = "=" * 60

# ─────────────────────────────────────────
# 1. 全体サマリー
# ─────────────────────────────────────────
print(SEP)
print("1. 全体サマリー")
print(SEP)
cur.execute("""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN result_profit <= 0 THEN 1 ELSE 0 END) as losses,
           ROUND(SUM(result_profit), 0) as total_pnl,
           ROUND(AVG(result_profit), 0) as avg_pnl,
           ROUND(MAX(result_profit), 0) as max_win,
           ROUND(MIN(result_profit), 0) as max_loss,
           ROUND(AVG(CASE WHEN result_profit > 0 THEN result_profit END), 0) as avg_win,
           ROUND(AVG(CASE WHEN result_profit <= 0 THEN result_profit END), 0) as avg_loss
    FROM trades WHERE status='CLOSED'
""")
r = dict(cur.fetchone())
wr = round(r["wins"] / r["total"] * 100, 1) if r["total"] else 0
print(f"  総トレード数: {r['total']}")
print(f"  勝率: {wr}% (W={r['wins']} / L={r['losses']})")
print(f"  累積PnL: {r['total_pnl']:,.0f} JPY")
print(f"  平均PnL: {r['avg_pnl']:,.0f} JPY")
print(f"  平均勝ち: {r['avg_win']:,.0f} JPY / 平均負け: {r['avg_loss']:,.0f} JPY")
if r["avg_loss"] and r["avg_loss"] != 0:
    pf = abs(float(r["avg_win"] or 0)) * r["wins"] / (abs(float(r["avg_loss"] or 0)) * r["losses"]) if r["losses"] else 0
    print(f"  PF (Profit Factor): {pf:.2f}")
print(f"  最大1勝: {r['max_win']:,.0f} / 最大1敗: {r['max_loss']:,.0f} JPY")

# ─────────────────────────────────────────
# 2. 銘柄別成績（サフィックス正規化）
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("2. 銘柄別成績（# / 非# で分けて表示）")
print(SEP)
cur.execute("""
    SELECT symbol, COUNT(*) as cnt,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(result_profit), 0) as total_pnl,
           ROUND(AVG(result_profit), 0) as avg_pnl,
           ROUND(AVG(CASE WHEN result_profit > 0 THEN result_profit END), 0) as avg_win,
           ROUND(AVG(CASE WHEN result_profit <= 0 THEN result_profit END), 0) as avg_loss
    FROM trades WHERE status='CLOSED'
    GROUP BY symbol ORDER BY total_pnl DESC
""")
for r in cur.fetchall():
    r = dict(r)
    wr = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] else 0
    avg_win = r["avg_win"] or 0
    avg_loss = r["avg_loss"] or 1
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    print(f"  {r['symbol']:<15} cnt={r['cnt']:3d}  WR={wr:5.1f}%  "
          f"PnL={r['total_pnl']:>10,.0f}  avg={r['avg_pnl']:>8,.0f}  "
          f"avg_win={avg_win:>8,.0f}  avg_loss={avg_loss:>8,.0f}  "
          f"実RR比={rr_ratio:.2f}")

# ─────────────────────────────────────────
# 3. exit_reason 別分析
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("3. exit_reason 別分析")
print(SEP)
cur.execute("""
    SELECT exit_reason, COUNT(*) as cnt,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(result_profit), 0) as total_pnl,
           ROUND(AVG(result_profit), 0) as avg_pnl
    FROM trades WHERE status='CLOSED'
    GROUP BY exit_reason ORDER BY cnt DESC
""")
for r in cur.fetchall():
    r = dict(r)
    wr = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] else 0
    print(f"  {str(r['exit_reason']):<30}  cnt={r['cnt']:3d}  WR={wr:5.1f}%  "
          f"total_pnl={r['total_pnl']:>10,.0f}  avg={r['avg_pnl']:>8,.0f}")

# ─────────────────────────────────────────
# 4. AI confidence 閾値と勝率の関係
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("4. AI confidence 閾値の影響分析")
print(SEP)
cur.execute("""
    SELECT ai_confidence,
           result_profit,
           exit_reason,
           symbol
    FROM trades WHERE status='CLOSED' AND ai_confidence IS NOT NULL
    ORDER BY ai_confidence
""")
rows = [dict(r) for r in cur.fetchall()]

# 閾値を65,68,70,72,75,78,80に設定したときの成績シミュレーション
print("  閾値以上でエントリーした場合の統計（上から閾値を上げる）:")
print(f"  {'閾値':>4}  {'cnt':>5}  {'WR':>6}  {'総PnL':>12}  {'avgPnL':>10}  {'avgW':>10}  {'avgL':>10}  {'PF':>5}")
for threshold in [65, 68, 70, 72, 75, 78, 80, 82, 85]:
    filtered = [r for r in rows if (r["ai_confidence"] or 0) >= threshold]
    if not filtered:
        continue
    wins = [r for r in filtered if r["result_profit"] > 0]
    losses = [r for r in filtered if r["result_profit"] <= 0]
    total_pnl = sum(r["result_profit"] for r in filtered)
    avg_pnl = total_pnl / len(filtered) if filtered else 0
    avg_win = sum(r["result_profit"] for r in wins) / len(wins) if wins else 0
    avg_loss = sum(r["result_profit"] for r in losses) / len(losses) if losses else 0
    pf = (abs(avg_win) * len(wins)) / (abs(avg_loss) * len(losses)) if losses and avg_loss != 0 else 0
    wr = len(wins) / len(filtered) * 100 if filtered else 0
    print(f"  conf>={threshold:>3}  {len(filtered):>5}  {wr:>6.1f}%  {total_pnl:>12,.0f}  "
          f"{avg_pnl:>10,.0f}  {avg_win:>10,.0f}  {avg_loss:>10,.0f}  {pf:>5.2f}")

# ─────────────────────────────────────────
# 5. 時間帯別成績 (UTC)
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("5. 時間帯別成績 (opened_at UTC)")
print(SEP)
cur.execute("""
    SELECT CAST(strftime('%H', opened_at) AS INTEGER) as hour,
           COUNT(*) as cnt,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(result_profit), 0) as total_pnl
    FROM trades WHERE status='CLOSED' AND opened_at IS NOT NULL
    GROUP BY hour ORDER BY hour
""")
for r in cur.fetchall():
    r = dict(r)
    wr = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] else 0
    bar = "#" * r["cnt"]
    print(f"  {r['hour']:02d}:xx  cnt={r['cnt']:3d}  WR={wr:5.1f}%  pnl={r['total_pnl']:>10,.0f}  {bar}")

# ─────────────────────────────────────────
# 6. 実際のTP到達 vs SLヒット 詳細
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("6. 実際のTP到達 vs SLヒット")
print(SEP)
cur.execute("""
    SELECT direction, entry_price, sl_price, tp_price,
           exit_price, result_profit, symbol, exit_reason,
           ai_confidence
    FROM trades
    WHERE status='CLOSED' AND entry_price IS NOT NULL
      AND sl_price IS NOT NULL AND tp_price IS NOT NULL
      AND exit_price IS NOT NULL
""")
tp_hits = sl_hits = ai_exits = other_exits = 0
tp_pnl = sl_pnl = ai_pnl = other_pnl = 0.0

rr_actual_wins = []
rr_actual_losses = []

for r in cur.fetchall():
    r = dict(r)
    ep = r["entry_price"]
    sl = r["sl_price"]
    tp = r["tp_price"]
    xp = r["exit_price"]
    pnl = r["result_profit"] or 0
    direction = r["direction"]
    reason = r["exit_reason"] or ""

    sl_dist = abs(ep - sl)
    if sl_dist > 0:
        actual_move = (xp - ep) if direction == "BUY" else (ep - xp)
        actual_r = actual_move / sl_dist
        if pnl > 0:
            rr_actual_wins.append(actual_r)
        else:
            rr_actual_losses.append(actual_r)

    # TP/SL判定
    if direction == "BUY":
        if xp >= tp * 0.999:
            tp_hits += 1; tp_pnl += pnl
        elif xp <= sl * 1.001:
            sl_hits += 1; sl_pnl += pnl
        elif "AI" in reason or "EXIT" in reason or "PREMISE" in reason:
            ai_exits += 1; ai_pnl += pnl
        else:
            other_exits += 1; other_pnl += pnl
    else:
        if xp <= tp * 1.001:
            tp_hits += 1; tp_pnl += pnl
        elif xp >= sl * 0.999:
            sl_hits += 1; sl_pnl += pnl
        elif "AI" in reason or "EXIT" in reason or "PREMISE" in reason:
            ai_exits += 1; ai_pnl += pnl
        else:
            other_exits += 1; other_pnl += pnl

print(f"  TP到達(機械/AI)  : {tp_hits:3d}件  pnl={tp_pnl:>10,.0f}")
print(f"  SLヒット         : {sl_hits:3d}件  pnl={sl_pnl:>10,.0f}")
print(f"  AI/PREMISE EXIT  : {ai_exits:3d}件  pnl={ai_pnl:>10,.0f}")
print(f"  その他          : {other_exits:3d}件  pnl={other_pnl:>10,.0f}")
if rr_actual_wins:
    avg_win_r = sum(rr_actual_wins) / len(rr_actual_wins)
    print(f"\n  勝ちトレードの平均実際R: {avg_win_r:.2f}R  (n={len(rr_actual_wins)})")
if rr_actual_losses:
    avg_loss_r = sum(rr_actual_losses) / len(rr_actual_losses)
    print(f"  負けトレードの平均実際R: {avg_loss_r:.2f}R  (n={len(rr_actual_losses)})")

# ─────────────────────────────────────────
# 7. 設定TP/SLのR比率分布
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("7. 設定RR比率 (TP距離/SL距離) 分布")
print(SEP)
cur.execute("""
    SELECT entry_price, sl_price, tp_price, result_profit, symbol
    FROM trades WHERE status='CLOSED'
      AND entry_price IS NOT NULL AND sl_price IS NOT NULL AND tp_price IS NOT NULL
""")
rr_buckets = defaultdict(list)
for r in cur.fetchall():
    r = dict(r)
    sl_dist = abs(r["entry_price"] - r["sl_price"])
    tp_dist = abs(r["tp_price"] - r["entry_price"])
    if sl_dist > 0:
        rr = tp_dist / sl_dist
        if rr < 1.0:
            rr_buckets["<1.0"].append(r["result_profit"] or 0)
        elif rr < 1.5:
            rr_buckets["1.0-1.5"].append(r["result_profit"] or 0)
        elif rr < 2.0:
            rr_buckets["1.5-2.0"].append(r["result_profit"] or 0)
        elif rr < 3.0:
            rr_buckets["2.0-3.0"].append(r["result_profit"] or 0)
        elif rr < 5.0:
            rr_buckets["3.0-5.0"].append(r["result_profit"] or 0)
        else:
            rr_buckets["5.0+"].append(r["result_profit"] or 0)

for band in ["<1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0+"]:
    pnls = rr_buckets[band]
    if pnls:
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100
        total = sum(pnls)
        print(f"  RR {band:>8}: cnt={len(pnls):3d}  WR={wr:5.1f}%  total_pnl={total:>10,.0f}")

# ─────────────────────────────────────────
# 8. 直近20トレードの詳細
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("8. 直近20トレード詳細")
print(SEP)
cur.execute("""
    SELECT opened_at, closed_at, symbol, direction, entry_price,
           exit_price, sl_price, tp_price, result_profit,
           exit_reason, ai_confidence, entry_type, market_regime
    FROM trades WHERE status='CLOSED'
    ORDER BY closed_at DESC LIMIT 20
""")
for r in cur.fetchall():
    r = dict(r)
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    tp = r["tp_price"] or 0
    xp = r["exit_price"] or 0
    sl_dist = abs(ep - sl)
    tp_dist = abs(tp - ep)
    rr_set = tp_dist / sl_dist if sl_dist > 0 else 0
    actual_move = (xp - ep) if r["direction"] == "BUY" else (ep - xp)
    actual_r = actual_move / sl_dist if sl_dist > 0 else 0
    pnl_str = f"{r['result_profit']:>+9,.0f}" if r["result_profit"] is not None else "    N/A "
    print(f"  [{r['closed_at'][:16]}] {r['symbol']:<12} {r['direction']} "
          f"conf={r['ai_confidence']:>3} RR設定={rr_set:.1f} "
          f"R実際={actual_r:+.2f} PnL={pnl_str}  {r['exit_reason'] or 'N/A'}")

# ─────────────────────────────────────────
# 9. 機械ゲート通過 vs 非通過の成績
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("9. 機械ゲート (smc_rr_pass) 通過可否と成績")
print(SEP)
cur.execute("""
    SELECT smc_rr_pass, smc_sweep_pass, smc_bos_pass,
           COUNT(*) as cnt,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(result_profit), 0) as total_pnl
    FROM trades WHERE status='CLOSED'
    GROUP BY smc_rr_pass, smc_sweep_pass, smc_bos_pass
    ORDER BY smc_rr_pass DESC, smc_sweep_pass DESC
""")
for r in cur.fetchall():
    r = dict(r)
    wr = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] else 0
    print(f"  rr_pass={r['smc_rr_pass']} sweep={r['smc_sweep_pass']} bos={r['smc_bos_pass']}  "
          f"cnt={r['cnt']:3d}  WR={wr:5.1f}%  total_pnl={r['total_pnl']:>10,.0f}")

# ─────────────────────────────────────────
# 10. market_regime 別成績
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("10. market_regime 別成績")
print(SEP)
cur.execute("""
    SELECT market_regime, COUNT(*) as cnt,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(result_profit), 0) as total_pnl
    FROM trades WHERE status='CLOSED'
    GROUP BY market_regime ORDER BY cnt DESC
""")
for r in cur.fetchall():
    r = dict(r)
    wr = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] else 0
    print(f"  {str(r['market_regime']):<20}  cnt={r['cnt']:3d}  WR={wr:5.1f}%  "
          f"total_pnl={r['total_pnl']:>10,.0f}")

# ─────────────────────────────────────────
# 11. 保有時間別成績
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("11. 保有時間別成績")
print(SEP)
cur.execute("""
    SELECT opened_at, closed_at, result_profit
    FROM trades WHERE status='CLOSED'
      AND opened_at IS NOT NULL AND closed_at IS NOT NULL
""")
hold_buckets = defaultdict(list)
for r in cur.fetchall():
    r = dict(r)
    try:
        opened = datetime.fromisoformat(r["opened_at"])
        closed = datetime.fromisoformat(r["closed_at"])
        hold_min = (closed - opened).total_seconds() / 60
        pnl = r["result_profit"] or 0
        if hold_min < 15:
            hold_buckets["<15min"].append(pnl)
        elif hold_min < 60:
            hold_buckets["15-60min"].append(pnl)
        elif hold_min < 180:
            hold_buckets["1-3h"].append(pnl)
        elif hold_min < 480:
            hold_buckets["3-8h"].append(pnl)
        elif hold_min < 1440:
            hold_buckets["8-24h"].append(pnl)
        else:
            hold_buckets["24h+"].append(pnl)
    except Exception:
        pass

for band in ["<15min", "15-60min", "1-3h", "3-8h", "8-24h", "24h+"]:
    pnls = hold_buckets[band]
    if pnls:
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100
        total = sum(pnls)
        print(f"  {band:<10}: cnt={len(pnls):3d}  WR={wr:5.1f}%  total_pnl={total:>10,.0f}")

# ─────────────────────────────────────────
# 12. 連敗パターン（ドローダウン）分析
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("12. 連敗パターン分析（最大連続損失）")
print(SEP)
cur.execute("""
    SELECT result_profit FROM trades WHERE status='CLOSED'
    ORDER BY closed_at ASC
""")
results = [r[0] or 0 for r in cur.fetchall()]

max_streak = 0
cur_streak = 0
streaks = []
for p in results:
    if p <= 0:
        cur_streak += 1
        max_streak = max(max_streak, cur_streak)
    else:
        if cur_streak > 0:
            streaks.append(cur_streak)
        cur_streak = 0
if cur_streak > 0:
    streaks.append(cur_streak)

print(f"  最大連続損失: {max_streak}連敗")
print(f"  連敗分布: {dict(sorted(((s, streaks.count(s)) for s in set(streaks)), key=lambda x: x[0]))}")

# ─────────────────────────────────────────
# 13. 損益に大きく影響した外れ値トレード
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("13. 大損失トレード TOP10")
print(SEP)
cur.execute("""
    SELECT opened_at, closed_at, symbol, direction,
           entry_price, exit_price, sl_price, tp_price,
           result_profit, exit_reason, ai_confidence
    FROM trades WHERE status='CLOSED'
    ORDER BY result_profit ASC LIMIT 10
""")
for r in cur.fetchall():
    r = dict(r)
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    tp = r["tp_price"] or 0
    xp = r["exit_price"] or 0
    sl_dist = abs(ep - sl)
    actual_move = (xp - ep) if r["direction"] == "BUY" else (ep - xp)
    actual_r = actual_move / sl_dist if sl_dist > 0 else 0
    print(f"  [{r['closed_at'] or r['opened_at'][:16]}] {r['symbol']:<12} {r['direction']} "
          f"conf={r['ai_confidence']}  R={actual_r:+.2f}  "
          f"PnL={r['result_profit']:>+10,.0f}  reason={r['exit_reason']}")

print(f"\n{SEP}")
print("14. 大利益トレード TOP10")
print(SEP)
cur.execute("""
    SELECT opened_at, closed_at, symbol, direction,
           entry_price, exit_price, sl_price, tp_price,
           result_profit, exit_reason, ai_confidence
    FROM trades WHERE status='CLOSED'
    ORDER BY result_profit DESC LIMIT 10
""")
for r in cur.fetchall():
    r = dict(r)
    ep = r["entry_price"] or 0
    sl = r["sl_price"] or 0
    tp = r["tp_price"] or 0
    xp = r["exit_price"] or 0
    sl_dist = abs(ep - sl)
    actual_move = (xp - ep) if r["direction"] == "BUY" else (ep - xp)
    actual_r = actual_move / sl_dist if sl_dist > 0 else 0
    print(f"  [{r['closed_at'] or r['opened_at'][:16]}] {r['symbol']:<12} {r['direction']} "
          f"conf={r['ai_confidence']}  R={actual_r:+.2f}  "
          f"PnL={r['result_profit']:>+10,.0f}  reason={r['exit_reason']}")

# ─────────────────────────────────────────
# 15. entry_type × confidence × 銘柄 クロス分析
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("15. REVERSAL_SWEEP 銘柄別 × confidence帯 クロス分析")
print(SEP)
cur.execute("""
    SELECT symbol, 
           CASE WHEN ai_confidence >= 80 THEN 'conf80+'
                WHEN ai_confidence >= 75 THEN 'conf75-79'
                ELSE 'conf<75' END as conf_band,
           COUNT(*) as cnt,
           SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(result_profit), 0) as total_pnl
    FROM trades WHERE status='CLOSED' AND entry_type='REVERSAL_SWEEP'
    GROUP BY symbol, conf_band
    ORDER BY symbol, conf_band
""")
for r in cur.fetchall():
    r = dict(r)
    wr = round(r["wins"] / r["cnt"] * 100, 1) if r["cnt"] else 0
    print(f"  {r['symbol']:<15} {r['conf_band']:<12}  cnt={r['cnt']:3d}  "
          f"WR={wr:5.1f}%  total_pnl={r['total_pnl']:>10,.0f}")

# ─────────────────────────────────────────
# 16. SL距離分布（ATR比率）— SLの適切さ確認
# ─────────────────────────────────────────
print(f"\n{SEP}")
print("16. SL / TP 距離の統計（銘柄別）")
print(SEP)
cur.execute("""
    SELECT symbol, direction, entry_price, sl_price, tp_price, result_profit
    FROM trades WHERE status='CLOSED'
      AND entry_price IS NOT NULL AND sl_price IS NOT NULL AND tp_price IS NOT NULL
""")
sym_data = defaultdict(lambda: {"sl_dists": [], "tp_dists": [], "rrs": []})
for r in cur.fetchall():
    r = dict(r)
    sym = r["symbol"].rstrip("#.")
    sl_dist = abs(r["entry_price"] - r["sl_price"])
    tp_dist = abs(r["tp_price"] - r["entry_price"])
    if sl_dist > 0:
        rr = tp_dist / sl_dist
        sym_data[sym]["sl_dists"].append(sl_dist)
        sym_data[sym]["tp_dists"].append(tp_dist)
        sym_data[sym]["rrs"].append(rr)

for sym, d in sorted(sym_data.items()):
    if not d["rrs"]:
        continue
    avg_sl = sum(d["sl_dists"]) / len(d["sl_dists"])
    avg_tp = sum(d["tp_dists"]) / len(d["tp_dists"])
    avg_rr = sum(d["rrs"]) / len(d["rrs"])
    median_rr = sorted(d["rrs"])[len(d["rrs"]) // 2]
    print(f"  {sym:<12}  avg_SL={avg_sl:.4f}  avg_TP={avg_tp:.4f}  "
          f"avg_RR={avg_rr:.2f}  median_RR={median_rr:.2f}  n={len(d['rrs'])}")

conn.close()
print(f"\n{SEP}")
print("分析完了")
print(SEP)
