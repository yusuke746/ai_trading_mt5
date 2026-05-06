"""AI自動売買システム メインループ

15分足確定ごとに5銘柄を巡回:
  - 新規エントリー: H1+M15画像をAI分析 → 相関チェック → ロット計算 → 発注
  - 保有ポジション: M15画像をAI分析 → エグジット判断
  - 毎時Heartbeat通知
"""

import sys
import time
import logging
import base64
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import config
import mt5_connector
import chart_capture
import lot_calculator
import risk_manager
import ai_analyzer
import discord_notifier
import trade_logger
import adaptive_params
import news_monitor
import market_stress

# ── ログ設定 ────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"{config.LOG_DIR}/trading_{datetime.now():%Y%m%d}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# チケットごとの「前回AIチェック時価格」キャッシュ (Exit AI コスト削減用)
_exit_price_cache: dict[int, float] = {}


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC)
    return dt.replace(tzinfo=UTC)


def _calculate_hold_minutes(pos: dict, trade: dict | None) -> int:
    """保有時間はDB記録を優先し、異常値は0分に丸める。"""
    now_utc = datetime.now(UTC)

    opened_at = trade.get("opened_at") if trade else None
    if opened_at:
        try:
            opened_dt = _as_utc(datetime.fromisoformat(opened_at))
            if opened_dt is not None:
                hold_minutes = int((now_utc - opened_dt).total_seconds() / 60)
                if hold_minutes >= 0:
                    return hold_minutes
                logger.warning(
                    "[Exit] %s ticket=%s: DB保有時間が負値 (%d min) のため0に補正",
                    pos["symbol"], pos["ticket"], hold_minutes,
                )
                return 0
        except ValueError:
            logger.warning(
                "[Exit] %s ticket=%s: opened_at解析失敗 (%s)",
                pos["symbol"], pos["ticket"], opened_at,
            )

    pos_time = _as_utc(pos.get("time"))
    if pos_time is None:
        return 0

    hold_minutes = int((now_utc - pos_time).total_seconds() / 60)
    if hold_minutes < 0:
        logger.warning(
            "[Exit] %s ticket=%s: MT5保有時間が負値 (%d min) のため0に補正",
            pos["symbol"], pos["ticket"], hold_minutes,
        )
        return 0
    return hold_minutes


# ── メインループ ────────────────────────

def _startup_weekend_check():
    """起動時に週末クローズ中にポジションが存在する場合、CRITICALログ+Discord通知を出す。"""
    if mt5_connector.is_fx_market_open():
        return  # 市場オープン中は不要
    positions = mt5_connector.get_positions()
    if not positions:
        return
    symbols_with_holdover = [
        p["symbol"] for p in positions if _is_weekend_holdover(p)
    ]
    if not symbols_with_holdover:
        return
    msg = (
        f"[警告] 週末跨ぎポジション検出: {symbols_with_holdover} "
        f"| 市場再開後サイクルで即手仕舞いします"
    )
    logger.critical(msg)
    discord_notifier.send_skip("SYSTEM", msg, notify=True)


def main():
    logger.info("=" * 60)
    logger.info("AI Trading System 起動")
    logger.info("監視銘柄: %s", config.SYMBOLS)
    logger.info("=" * 60)

    # DB 初期化
    trade_logger.init_db()
    _run_db_maintenance(full_vacuum=False)

    # ニュース監視スレッド起動 (経済カレンダー + nanoニュースポーリング)
    news_monitor.start_background_monitor()

    # MT5 接続
    if not mt5_connector.initialize():
        logger.critical("MT5接続失敗 → 終了")
        return

    # 起動通知
    account = mt5_connector.get_account_info()
    if account:
        discord_notifier.send_heartbeat(
            account["balance"], account["equity"],
            len(mt5_connector.get_positions()),
        )

    # 起動時週末跨ぎポジションチェック
    _startup_weekend_check()

    last_heartbeat = datetime.now()
    last_db_maintenance = datetime.now()
    last_full_vacuum = datetime.now()
    last_cycle_minute = -1

    try:
        while True:
            now = datetime.now()

            # ── Heartbeat (1時間ごと) ──
            if (now - last_heartbeat).total_seconds() >= config.HEARTBEAT_INTERVAL_SEC:
                _send_heartbeat()
                last_heartbeat = now

            # ── SQLiteメンテナンス ──
            if (now - last_db_maintenance).total_seconds() >= config.DB_MAINTENANCE_INTERVAL_SEC:
                do_full_vacuum = (
                    now - last_full_vacuum
                ).total_seconds() >= config.DB_FULL_VACUUM_INTERVAL_SEC
                _run_db_maintenance(full_vacuum=do_full_vacuum)
                last_db_maintenance = now
                if do_full_vacuum:
                    last_full_vacuum = now

            # ── 市場クローズ中: ポジションなし時は5分スリープで空回り削減 ──
            # サーバー時刻(GMT+3)ベースで判定 (JSTベースにするとFX金曜夜を誤ってクローズ扱いする)
            if not mt5_connector.is_fx_market_open():
                has_positions = bool(mt5_connector.get_positions())
                if not has_positions:
                    time.sleep(300)
                    continue
                # 週末跨ぎポジションあり: 市場再開検知のため通常サイクル継続

            # ── 15分足確定タイミング (00, 15, 30, 45分) ──
            if now.minute % 15 == 0 and now.minute != last_cycle_minute:
                # 足確定を待つ
                time.sleep(config.CANDLE_WAIT_SEC)
                last_cycle_minute = now.minute

                if not mt5_connector.ensure_connected():
                    continue

                _trading_cycle()

            time.sleep(config.MAIN_LOOP_SLEEP_SEC)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt → シャットダウン")
    except Exception as e:
        logger.critical("予期せぬエラー: %s", e, exc_info=True)
        discord_notifier.send_error("致命的エラー", str(e))
    finally:
        mt5_connector.shutdown()
        logger.info("システム終了")


# ── トレーディングサイクル ──────────────

def _trading_cycle():
    """15分ごとの売買判断サイクル"""
    logger.info("─── Trading Cycle Start ───")

    # 0) MT5自動決済(TP/SL)でDBに孤立したトレードを照合・更新 (クールダウン機能の前提)
    try:
        _reconcile_orphaned_db_trades()
    except Exception as e:
        logger.error("孤立トレード照合例外: %s", e, exc_info=True)

    # 1) 保有ポジションのエグジットチェック
    _check_exits()

    # 2) 各銘柄の新規エントリーチェック
    for symbol in config.SYMBOLS:
        try:
            _check_entry(symbol)
        except Exception as e:
            logger.error("エントリーチェック例外: %s %s", symbol, e, exc_info=True)
            discord_notifier.send_error(f"エントリーチェック例外: {symbol}", str(e))

    logger.info("─── Trading Cycle End ───")


def _mechanical_smc_gate(
    df_h1,
    atr_h1: float,
    smc_data: dict,
    current_price: float,
) -> tuple[bool, bool, bool, str, str, float | None, float | None]:
    """H1データのみでSMC条件を数値判定する機械ゲート。AI呼び出し前のコスト削減フィルタ。

        Returns: (sweep_pass, bos_pass, rr_pass, sweep_type, entry_type, swept_level, structural_sl_dist)
      sweep_type: "HIGH" / "LOW" / "NONE"
            entry_type: "REVERSAL_SWEEP" / "CONTINUATION_BOS" / "NONE"
          swept_level: スイープされた価格レベル (REVERSAL_SWEEPのみ非None)
   structural_sl_dist: 構造的SL幅 (RR判定・発注・ロット計算で共通使用)
    """
    if not smc_data or atr_h1 <= 0:
                return False, False, False, "NONE", "NONE", None, None

    # キーレベル収集 (PDH/PDL/PWH/PWL + H1スウィング高安値)
    levels: list[float] = []
    for key in ("pdh", "pdl", "pwh", "pwl"):
        v = smc_data.get(key)
        if v:
            levels.append(float(v))
    for v in smc_data.get("swing_highs", []):
        levels.append(float(v))
    for v in smc_data.get("swing_lows", []):
        levels.append(float(v))

    if not levels:
        return False, False, False, "NONE", "NONE", None, None

    min_penetration = atr_h1 * config.SMC_SWEEP_ATR_MULT
    lookback = min(config.SMC_SWEEP_LOOKBACK_BARS, len(df_h1) - 1)
    recent = df_h1.iloc[-lookback:]

    sweep_pass = False
    sweep_type = "NONE"
    swept_level: float | None = None

    for level in levels:
        for _, bar in recent.iloc[::-1].iterrows():  # 最新バーから逆順に検索（最新スイープを優先）
            # 高値Sweep: レベル上にATR*mult以上侵食してレベル下でクローズ
            if bar["high"] > level + min_penetration and bar["close"] < level:
                sweep_pass = True
                sweep_type = "HIGH"
                swept_level = level
                break
            # 安値Sweep: レベル下にATR*mult以上侵食してレベル上でクローズ
            if bar["low"] < level - min_penetration and bar["close"] > level:
                sweep_pass = True
                sweep_type = "LOW"
                swept_level = level
                break
        if sweep_pass:
            break

    # ── Reversal: BOS + RR判定 ──────────────────────────────
    bos_pass = False
    ma = mt5_connector.calculate_ma(df_h1, config.MA_PERIOD)
    ma_clean = ma.dropna()
    rr_relax = max(0.5, min(1.0, config.SMC_MECHANICAL_RR_RELAX_FACTOR))
    rr_pass = False

    # 構造的SL距離: REVERSAL_SWEEPはswept_level外側、CONTINUATION_BOSはATR×1.5
    structural_sl_dist: float | None = None
    if sweep_pass and swept_level is not None:
        buffer = atr_h1 * 0.2
        if sweep_type == "HIGH":
            raw_dist = (swept_level + buffer) - current_price
        else:
            raw_dist = current_price - (swept_level - buffer)
        # 最低ATR×1.0を確保
        structural_sl_dist = max(raw_dist, atr_h1 * 1.0) if raw_dist > 0 else atr_h1 * 1.5

    sl_dist = structural_sl_dist if structural_sl_dist is not None else atr_h1 * 1.5
    min_tp_dist = sl_dist * config.ENTRY_MIN_TP_R * rr_relax

    if sweep_pass:
        if len(ma_clean) >= 2:
            ma_curr = float(ma_clean.iloc[-1])
            ma_prev = float(ma_clean.iloc[-2])
            if sweep_type == "HIGH":
                # 高値Sweep後は下トレンド期待: 現在価格 < swept_level かつ MA下向き
                bos_pass = current_price < (swept_level or current_price) and ma_curr < ma_prev
            else:
                # 安値Sweep後は上トレンド期待: 現在価格 > swept_level かつ MA上向き
                bos_pass = current_price > (swept_level or current_price) and ma_curr > ma_prev

        if sweep_type == "HIGH":
            targets = [v for v in levels if v < current_price]
            if targets:
                rr_pass = (current_price - max(targets)) >= min_tp_dist
        else:
            targets = [v for v in levels if v > current_price]
            if targets:
                rr_pass = (min(targets) - current_price) >= min_tp_dist

        if config.SMC_REVERSAL_ENABLED:
            return sweep_pass, bos_pass, rr_pass, sweep_type, "REVERSAL_SWEEP", swept_level, structural_sl_dist

    # ── Continuation BOS gate (順張り: BOS後の押し目/戻し) ─────────
    if config.SMC_CONTINUATION_ENABLED and len(ma_clean) >= config.SMC_CONTINUATION_BOS_LOOKBACK_BARS + 1:
        ma_curr = float(ma_clean.iloc[-1])
        ma_past = float(ma_clean.iloc[-(config.SMC_CONTINUATION_BOS_LOOKBACK_BARS + 1)])
        ma_slope = ma_curr - ma_past

        if abs(ma_slope) >= atr_h1 * config.SMC_CONTINUATION_MA_SLOPE_ATR_MULT:
            if ma_slope > 0 and current_price <= ma_curr + atr_h1 * 0.5:
                # 上昇トレンド + 価格がMA付近以下: BUY押し目セットアップ
                cont_sweep_type = "LOW"
                bos_pass = True
                cont_sl_dist = atr_h1 * 1.5
                cont_min_tp = cont_sl_dist * config.ENTRY_MIN_TP_R * rr_relax
                targets = [v for v in levels if v > current_price]
                rr_pass = bool(targets) and (min(targets) - current_price) >= cont_min_tp
                return False, bos_pass, rr_pass, cont_sweep_type, "CONTINUATION_BOS", None, cont_sl_dist
            elif ma_slope < 0 and current_price >= ma_curr - atr_h1 * 0.5:
                # 下降トレンド + 価格がMA付近以上: SELL戻しセットアップ
                cont_sweep_type = "HIGH"
                bos_pass = True
                cont_sl_dist = atr_h1 * 1.5
                cont_min_tp = cont_sl_dist * config.ENTRY_MIN_TP_R * rr_relax
                targets = [v for v in levels if v < current_price]
                rr_pass = bool(targets) and (current_price - max(targets)) >= cont_min_tp
                return False, bos_pass, rr_pass, cont_sweep_type, "CONTINUATION_BOS", None, cont_sl_dist

    return False, False, False, "NONE", "NONE", None, None


def _entry_direction_from_mech_gate(sweep_type: str, entry_type: str) -> str | None:
    """機械ゲート結果から想定エントリー方向を推定する。"""
    if entry_type == "NONE":
        return None
    sweep_type = str(sweep_type).upper()
    if sweep_type == "LOW":
        return "BUY"
    if sweep_type == "HIGH":
        return "SELL"
    return None


def _timeframe_to_minutes(timeframe: str) -> int:
    """M15/H1/D1 などを分に変換する。"""
    tf = str(timeframe).strip().upper()
    if not tf:
        return 15
    unit = tf[0]
    try:
        value = int(tf[1:])
    except ValueError:
        return 15

    if unit == "M":
        return value
    if unit == "H":
        return value * 60
    if unit == "D":
        return value * 1440
    return 15


def _is_weekend_holdover(pos: dict) -> bool:
    """市場クローズ中に保有されているポジションか判定する (サーバー時刻GMT+3ベース)。"""
    return not mt5_connector.is_fx_market_open()


def _should_flatten_before_market_close(now: datetime | None = None) -> str | None:
    """市場クローズ前の持ち越し回避ウィンドウなら理由を返す。
    週末クローズ: XMTradingサーバー時刻(GMT+3)で金曜23:59の lead_minutes 前から手仕舞い。
    日次クローズ: FXは24/5のため通常無効(FLAT_BEFORE_MARKET_CLOSE_ENABLED=false)。
    """
    # ── 週末クローズ前 (サーバー時刻ベース) ──
    if config.FLAT_BEFORE_WEEKEND_CLOSE_ENABLED:
        minutes_to_close = mt5_connector.get_minutes_to_weekend_close()
        if minutes_to_close is not None:  # 金曜のみ値が返る
            lead = config.FLAT_BEFORE_WEEKEND_CLOSE_LEAD_MINUTES
            if minutes_to_close <= lead:
                srv = mt5_connector.get_server_datetime()
                srv_str = srv.strftime('%H:%M') if srv else '??:??'
                return (
                    f"週末クローズ前の持ち越し回避: サーバー時刻{srv_str}(GMT+3) "
                    f"クローズまで{minutes_to_close:.0f}分 (lead={lead}分)"
                )

    # ── 日次クローズ前 (FXは通常無効) ──
    if not config.FLAT_BEFORE_MARKET_CLOSE_ENABLED:
        return None

    current = now or datetime.now()
    close_dt = current.replace(
        hour=config.FLAT_BEFORE_MARKET_CLOSE_HOUR,
        minute=config.FLAT_BEFORE_MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )
    if close_dt <= current:
        close_dt += timedelta(days=1)
    minutes_to_close = (close_dt - current).total_seconds() / 60
    lead_minutes = config.FLAT_BEFORE_MARKET_CLOSE_LEAD_MINUTES
    if 0 <= minutes_to_close <= lead_minutes:
        return (
            f"日次クローズ前の持ち越し回避: クローズまで{minutes_to_close:.0f}分 "
            f"(設定 {config.FLAT_BEFORE_MARKET_CLOSE_HOUR:02d}:{config.FLAT_BEFORE_MARKET_CLOSE_MINUTE:02d}, "
            f"lead={lead_minutes}分)"
        )
    return None


# ── エントリーチェック ──────────────────

def _is_weekend_entry_blocked() -> bool:
    """FX市場クローズ中はエントリー禁止 (サーバー時刻GMT+3ベース)。"""
    if not config.FLAT_BEFORE_WEEKEND_CLOSE_ENABLED:
        return False
    return not mt5_connector.is_fx_market_open()


def _check_entry(symbol: str):
    """1銘柄のエントリー判断"""

    # 既にポジション保有中ならスキップ
    positions = mt5_connector.get_positions(symbol)
    if positions:
        logger.info("[Entry] %s: ポジション保有中 → スキップ", symbol)
        return

    # 週末クローズ後はエントリー禁止 (ギャップリスク)
    if _is_weekend_entry_blocked():
        logger.info("[Entry] %s: 週末クローズ中 → エントリーブロック", symbol)
        return

    if not mt5_connector.is_symbol_market_active(symbol):
        logger.info("[Entry] %s: 市場クローズ/気配停止中 → AI判定スキップ", symbol)
        return

    # 相関リスクチェック
    can_open, reason = risk_manager.can_open_position(symbol)
    if not can_open:
        logger.info("[Entry] %s: 相関リスク超過 → スキップ (%s)", symbol, reason)
        discord_notifier.send_skip(symbol, reason, notify=True)
        return

    # 連敗銘柄の過剰エントリー抑制（クールダウン）
    loss_info = trade_logger.get_symbol_recent_loss_streak(symbol)
    streak = int(loss_info.get("loss_streak", 0))
    last_closed_at = loss_info.get("last_closed_at")
    if streak >= config.SYMBOL_LOSS_STREAK_PAUSE_TRIGGER and last_closed_at:
        try:
            last_dt = _as_utc(datetime.fromisoformat(last_closed_at))
            elapsed_min = (datetime.now(UTC) - last_dt).total_seconds() / 60
            if elapsed_min < config.SYMBOL_LOSS_STREAK_COOLDOWN_MINUTES:
                logger.warning(
                    "[Entry] %s: 直近%d連敗のためクールダウン中 (%.0f/%.0f min) → スキップ",
                    symbol,
                    streak,
                    elapsed_min,
                    config.SYMBOL_LOSS_STREAK_COOLDOWN_MINUTES,
                )
                return
        except ValueError:
            pass

    tf_minutes = _timeframe_to_minutes(config.EXECUTION_TF)

    # 同銘柄クールダウン（全Exit理由対象）
    if config.SYMBOL_REENTRY_COOLDOWN_ALL_EXITS_ENABLED:
        recent_closed = trade_logger.get_recent_closed_trade(symbol)
        closed_at = recent_closed.get("closed_at") if recent_closed else None
        if closed_at:
            try:
                last_closed_dt = _as_utc(datetime.fromisoformat(closed_at))
                elapsed_min = (datetime.now(UTC) - last_closed_dt).total_seconds() / 60
                block_minutes = tf_minutes * config.SYMBOL_REENTRY_COOLDOWN_ALL_EXITS_BARS
                if elapsed_min < block_minutes:
                    logger.warning(
                        "[Entry] %s: 同銘柄クールダウン中(全Exit) %.0f/%.0f min (%d bars, reason=%s) → スキップ",
                        symbol,
                        elapsed_min,
                        block_minutes,
                        config.SYMBOL_REENTRY_COOLDOWN_ALL_EXITS_BARS,
                        recent_closed.get("exit_reason", "UNKNOWN"),
                    )
                    return
            except ValueError:
                pass

    # 勝ちトレード後の再エントリー抑制
    if config.SYMBOL_REENTRY_COOLDOWN_AFTER_WIN_ENABLED:
        recent_win = trade_logger.get_recent_winning_closed_trade(symbol)
        win_closed_at = recent_win.get("closed_at") if recent_win else None
        if win_closed_at:
            try:
                last_win_dt = _as_utc(datetime.fromisoformat(win_closed_at))
                elapsed_min = (datetime.now(UTC) - last_win_dt).total_seconds() / 60
                block_minutes = tf_minutes * config.SYMBOL_REENTRY_COOLDOWN_AFTER_WIN_BARS
                if elapsed_min < block_minutes:
                    logger.warning(
                        "[Entry] %s: 勝ち後クールダウン中 %.0f/%.0f min (%d bars, profit=%.0f) → スキップ",
                        symbol,
                        elapsed_min,
                        block_minutes,
                        config.SYMBOL_REENTRY_COOLDOWN_AFTER_WIN_BARS,
                        float(recent_win.get("result_profit") or 0),
                    )
                    return
            except ValueError:
                pass

    # ニュース・経済カレンダーチェック (キャッシュ参照のみ、API呼び出しなし)
    news_blocked, news_reason = news_monitor.check_entry_news_block(symbol)
    if news_blocked:
        logger.info("[Entry] %s: ニュースブロック → スキップ (%s)", symbol, news_reason)
        discord_notifier.send_skip(symbol, news_reason, notify=False)
        return

    # 市場ストレス検知チェック (スプレッド/ATR急変 → 既にストレス状態)
    if market_stress.is_stressed(symbol):
        st = market_stress.get_stress_state(symbol)
        reason = f"[MarketStress] {st.risk_level}: {st.summary} (hold_until={st.hold_until.strftime('%H:%M UTC')})"
        logger.info("[Entry] %s: 市場ストレス状態 → エントリーブロック (%s)", symbol, reason)
        discord_notifier.send_skip(symbol, reason, notify=False)
        return

    # レートデータ取得
    df_h1 = mt5_connector.get_rates(symbol, config.TREND_TF, config.CHART_BARS + 30)
    df_m15 = mt5_connector.get_rates(symbol, config.EXECUTION_TF, config.CHART_BARS + 30)
    if df_h1 is None or df_m15 is None:
        logger.warning("[Entry] %s: レートデータ取得失敗", symbol)
        return

    # テクニカル指標
    atr_h1 = mt5_connector.calculate_atr(df_h1, config.ATR_PERIOD)
    atr_m15 = mt5_connector.calculate_atr(df_m15, config.ATR_PERIOD)
    ma20 = mt5_connector.calculate_ma(df_m15, config.MA_PERIOD)

    # 市場ストレス検知 (スプレッド/ATR急変) — エントリー前に計測・更新
    sym_info_stress = mt5_connector.get_symbol_info(symbol)
    if sym_info_stress:
        current_spread = float(sym_info_stress.get("spread", 0))
        # H1 ATR の過去20本平均をベースラインとして使用
        atr_series = mt5_connector.calculate_atr(df_h1, config.ATR_PERIOD)
        # pandas Series の場合は最後の N 要素平均を使う
        try:
            baseline_atr_val = float(atr_series.iloc[-20:].mean()) if hasattr(atr_series, "iloc") else None
        except Exception:
            baseline_atr_val = None
        stress = market_stress.check_and_update(
            symbol=symbol,
            current_spread=current_spread,
            current_atr=atr_h1,
            baseline_atr=baseline_atr_val,
        )
        if stress:
            reason = f"[MarketStress] {stress.risk_level}: {stress.summary} (hold_until={stress.hold_until.strftime('%H:%M UTC')})"
            logger.warning("[Entry] %s: 市場ストレス新規検知 → エントリーブロック (%s)", symbol, reason)
            discord_notifier.send_skip(symbol, reason, notify=True)
            return

    # 基本チェック用の指標を取得
    current_close = df_m15["close"].iloc[-1]
    ma20_val = ma20.iloc[-1]
    if ma20_val is None or atr_m15 <= 0:
        return

    # 現在価格・SMCレベルを先取得 (機械ゲートに必要)
    price_info = mt5_connector.get_current_price(symbol)
    if price_info is None:
        return
    current_price = price_info["bid"]

    sym_info_for_digits = mt5_connector.get_symbol_info(symbol)
    digits_for_smc = sym_info_for_digits["digits"] if sym_info_for_digits else 5
    smc_data = mt5_connector.get_price_levels(symbol, digits=digits_for_smc)
    logger.info(
        "[Entry] %s: SMC levels PDH=%.5f PDL=%.5f swings_h=%s swings_l=%s",
        symbol,
        smc_data.get("pdh") or 0,
        smc_data.get("pdl") or 0,
        smc_data.get("swing_highs", []),
        smc_data.get("swing_lows", []),
    )

    # 機械的SMCゲート (逆張り/順張り両対応)
    smc_sweep_pass, smc_bos_pass, smc_rr_pass, mech_sweep_type, mech_entry_type, mech_swept_level, mech_structural_sl_dist = _mechanical_smc_gate(
        df_h1, atr_h1, smc_data, current_price
    )
    logger.info(
        "[Entry] %s: MechGate sweep=%s bos=%s rr=%s type=%s entry_type=%s",
        symbol, smc_sweep_pass, smc_bos_pass, smc_rr_pass, mech_sweep_type, mech_entry_type,
    )

    # PREMISE_BREAK後の同方向再エントリーを一定本数ブロック
    inferred_direction = _entry_direction_from_mech_gate(mech_sweep_type, mech_entry_type)
    if config.PREMISE_BREAK_REENTRY_BLOCK_ENABLED and inferred_direction:
        recent_premise_break = trade_logger.get_recent_premise_break_exit(symbol, inferred_direction)
        closed_at = recent_premise_break.get("closed_at") if recent_premise_break else None
        if closed_at:
            try:
                last_break_dt = _as_utc(datetime.fromisoformat(closed_at))
                elapsed_min = (datetime.now(UTC) - last_break_dt).total_seconds() / 60
                block_minutes = _timeframe_to_minutes(config.EXECUTION_TF) * config.PREMISE_BREAK_REENTRY_BLOCK_BARS
                if elapsed_min < block_minutes:
                    logger.warning(
                        "[Entry] %s: PREMISE_BREAK後の同方向再エントリー禁止中 (%s %.0f/%.0f min, %d bars) → スキップ",
                        symbol,
                        inferred_direction,
                        elapsed_min,
                        block_minutes,
                        config.PREMISE_BREAK_REENTRY_BLOCK_BARS,
                    )
                    return
            except ValueError:
                pass

    if config.SMC_FILTER_ENABLED and config.SMC_MECHANICAL_GATE_ENABLED:
        if mech_entry_type == "NONE":
            logger.info(
                "[Entry] %s: 機械ゲート: セットアップ未検出 → AIコスト節約スキップ "
                "(sweep=%s bos=%s rr=%s type=%s)",
                symbol, smc_sweep_pass, smc_bos_pass, smc_rr_pass, mech_sweep_type,
            )
            return
        # RR不足は AI呼び出し前にスキップ (コスト節約)
        if not smc_rr_pass:
            logger.info(
                "[Entry] %s: 機械ゲート: RR不足 → AIコスト節約スキップ "
                "(rr_pass=False entry_type=%s)",
                symbol, mech_entry_type,
            )
            return
        # 順張りでbos_pass=Falseも事前スキップ
        if mech_entry_type == "CONTINUATION_BOS" and not smc_bos_pass:
            logger.info(
                "[Entry] %s: 機械ゲート: BOS未確認 → AIコスト節約スキップ "
                "(bos_pass=False entry_type=%s)",
                symbol, mech_entry_type,
            )
            return

    # MA近傍フィルタ: 順張りの押し目/戻しのみに適用 (逆張りはSweepレベルが遠い場合もある)
    if mech_entry_type != "REVERSAL_SWEEP":
        distance_from_ma = abs(current_close - ma20_val)
        if distance_from_ma > atr_m15 * 1.0:
            logger.info(
                "[Entry] %s: MA20から乖離 (dist=%.5f > ATR=%.5f) → スキップ",
                symbol, distance_from_ma, atr_m15,
            )
            return

    # SMCオーバーレイ付きチャート画像生成
    h1_b64, m15_b64 = chart_capture.generate_smc_chart_pair_base64(
        symbol=symbol,
        smc_features=smc_data,
    )
    if h1_b64 is None or m15_b64 is None:
        logger.warning("[Entry] %s: チャート画像生成失敗", symbol)
        return
    h1_img = base64.b64decode(h1_b64)
    m15_img = base64.b64decode(m15_b64)

    # AI分析
    account = mt5_connector.get_account_info()
    balance = account["balance"] if account else 0

    signal = ai_analyzer.analyze_entry(
        symbol=symbol,
        current_price=current_price,
        atr_h1=atr_h1,
        atr_m15=atr_m15,
        h1_image=h1_img,
        m15_image=m15_img,
        balance=balance,
        smc_data=smc_data,
        mech_gate={
            "sweep_pass": smc_sweep_pass,
            "bos_pass":   smc_bos_pass,
            "rr_pass":    smc_rr_pass,
            "sweep_type": mech_sweep_type,
            "entry_type": mech_entry_type,
            "swept_level": mech_swept_level,
            "structural_sl_dist": mech_structural_sl_dist,
        },
    )

    # AIログ保存
    trade_logger.insert_ai_log(
        symbol=symbol,
        action_type="ENTRY_CHECK",
        ai_response=signal.raw_response[:2000],
        decision=signal.decision,
        reasoning=signal.reasoning[:1000],
    )

    # 判定
    if signal.decision == "SKIP":
        logger.info(
            "[Entry] %s: AI判断 SKIP (conf=%d align=%s h1=%s smc_sweep=%s smc_dir=%s ob=%s fvg=%s reason=%s)",
            symbol,
            signal.confidence,
            signal.alignment,
            signal.h1_trend,
            signal.smc_liquidity_sweep,
            signal.smc_sweep_direction,
            signal.smc_ob_confirmed,
            signal.smc_fvg_present,
            (signal.reasoning or "")[:180].replace("\n", " "),
        )
        return

    if not signal.alignment:
        logger.info("[Entry] %s: H1/M15トレンド不一致 → スキップ", symbol)
        return

    conf_threshold = adaptive_params.get_confidence_threshold(signal.h1_trend, mech_entry_type)
    if signal.confidence < conf_threshold:
        logger.info(
            "[Entry] %s: 信頼度不足 %d < %d (adaptive, entry_type=%s, h1=%s) → スキップ",
            symbol, signal.confidence, conf_threshold, mech_entry_type, signal.h1_trend,
        )
        return

    # SMCフィルタ: エントリータイプに応じた必須条件チェック
    if config.SMC_FILTER_ENABLED:
        if mech_entry_type == "REVERSAL_SWEEP" and not signal.smc_liquidity_sweep:
            logger.info(
                "[Entry] %s: SMCフィルタ(逆張り) → Liquidity Sweep未確認 → スキップ (smc_sweep=%s)",
                symbol, signal.smc_sweep_direction,
            )
            return
        elif mech_entry_type == "CONTINUATION_BOS" and not signal.smc_ob_confirmed:
            logger.info(
                "[Entry] %s: SMCフィルタ(順張り) → OB/FVG未確認 → スキップ",
                symbol,
            )
            return

    logger.info(
        "[Entry] %s: SMC sweep=%s dir=%s OB=%s FVG=%s",
        symbol, signal.smc_liquidity_sweep, signal.smc_sweep_direction,
        signal.smc_ob_confirmed, signal.smc_fvg_present,
    )

    # ── 発注処理 ──
    direction = signal.decision  # "BUY" or "SELL"

    # エントリー価格・桁数取得
    entry_price = price_info["ask"] if direction == "BUY" else price_info["bid"]
    sym_info = mt5_connector.get_symbol_info(symbol)
    digits = sym_info["digits"] if sym_info else 5

    # ─ SL幅の決定プロセス (機械一元管理、AI値は使用しない) ─
    # REVERSAL_SWEEP: swept_level + spread/ATRバッファで機械的に確定
    # CONTINUATION_BOS: 機械ゲートのATR×1.5を使用

    if mech_entry_type == "REVERSAL_SWEEP" and mech_swept_level is not None:
        spread = price_info["ask"] - price_info["bid"]
        buffer = max(atr_m15 * 0.2, spread * 2)
        if direction == "SELL":
            structural_sl = round(mech_swept_level + buffer, digits)
            structural_sl_dist = structural_sl - entry_price
        else:
            structural_sl = round(mech_swept_level - buffer, digits)
            structural_sl_dist = entry_price - structural_sl

        if structural_sl_dist > 0:
            sl_distance = structural_sl_dist
            sl_price = structural_sl
            logger.info(
                "[Entry] %s: REVERSAL_SWEEP SL: swept_level=%.5f buffer=%.5f sl_dist=%.5f sl_price=%.5f",
                symbol, mech_swept_level, buffer, sl_distance, sl_price,
            )
        else:
            # swept_levelがエントリー価格より不利側にある異常はフォールバック
            sl_distance = mech_structural_sl_dist or lot_calculator.get_sl_distance(atr_m15)
            sl_price = lot_calculator.calculate_sl_price(
                symbol=symbol, direction=direction,
                entry_price=entry_price, atr=sl_distance, multiplier=1.0,
            )
    else:
        # CONTINUATION_BOS: ATR×1.5（機械ゲート一元）
        sl_distance = mech_structural_sl_dist or lot_calculator.get_sl_distance(atr_m15)
        sl_price = lot_calculator.calculate_sl_price(
            symbol=symbol, direction=direction,
            entry_price=entry_price, atr=sl_distance, multiplier=1.0,
        )

    # 極端に近いSLを防ぎ、低残高時の過剰ロット化を抑制
    min_sl_distance = atr_m15 * config.ENTRY_MIN_SL_ATR_MULT
    if sl_distance < min_sl_distance:
        old_sl_distance = sl_distance
        sl_distance = min_sl_distance
        if direction == "BUY":
            sl_price = round(entry_price - sl_distance, digits)
        else:
            sl_price = round(entry_price + sl_distance, digits)
        logger.warning(
            "[Entry] %s: SL下限適用 old=%.5f floor=%.5f (atr_m15=%.5f × %.2f)",
            symbol,
            old_sl_distance,
            min_sl_distance,
            atr_m15,
            config.ENTRY_MIN_SL_ATR_MULT,
        )

    logger.info(
        "[EntryMonitor] %s: entry_type=%s dir=%s balance=%.0f risk_per_trade=%.2f%% atr_m15=%.5f sl_dist=%.5f sl=%.5f",
        symbol,
        mech_entry_type,
        direction,
        balance,
        config.RISK_PER_TRADE * 100,
        atr_m15,
        sl_distance,
        sl_price,
    )

    # ロット計算 (リスク2%涸排が常に正確にsl_distanceで計算される)
    lot = lot_calculator.calculate_lot(symbol, sl_distance)
    if lot is None:
        logger.warning("[Entry] %s: ロット計算失敗 → スキップ", symbol)
        discord_notifier.send_error(f"ロット計算失敗: {symbol}", "calculate_lot returned None")
        return

    tp_distance = signal.tp_distance
    min_tp_distance = sl_distance * config.ENTRY_MIN_TP_R
    if tp_distance < min_tp_distance:
        tp_distance = min_tp_distance

    if direction == "BUY":
        tp_price = round(entry_price + tp_distance, digits)
    else:
        tp_price = round(entry_price - tp_distance, digits)

    logger.info(
        "[EntryMonitor] %s: tp_dist=%.5f min_tp=%.5f tp=%.5f",
        symbol,
        tp_distance,
        min_tp_distance,
        tp_price,
    )

    # 発注
    ticket = mt5_connector.place_order(symbol, direction, lot, sl_price, tp_price)
    if ticket is None:
        return

    # DB記録
    smc_summary = (
        f"[SMC] entry_type={mech_entry_type} sweep={signal.smc_liquidity_sweep} "
        f"dir={signal.smc_sweep_direction} OB={signal.smc_ob_confirmed} FVG={signal.smc_fvg_present}\n"
        f"[MechGate] sweep={smc_sweep_pass} bos={smc_bos_pass} rr={smc_rr_pass} type={mech_sweep_type}\n"
    )
    trade_logger.insert_trade(
        symbol=symbol, direction=direction, entry_price=entry_price,
        lot_size=lot, sl_price=sl_price, tp_price=tp_price,
        ai_reasoning=(smc_summary + signal.reasoning)[:1000],
        news_summary=signal.news_impact[:500],
        mt5_ticket=ticket,
        smc_sweep_pass=smc_sweep_pass,
        smc_bos_pass=smc_bos_pass,
        smc_rr_pass=smc_rr_pass,
        ai_confidence=signal.confidence,
        ai_smc_sweep=signal.smc_liquidity_sweep,
        ai_smc_ob=signal.smc_ob_confirmed,
        ai_smc_fvg=signal.smc_fvg_present,
        entry_type=mech_entry_type,
        market_regime=signal.h1_trend,
        invalidation_price=signal.invalidation_price,
    )

    # Discord通知
    discord_notifier.send_entry(
        symbol=symbol, direction=direction, lot=lot,
        entry_price=entry_price, sl=sl_price, tp=tp_price,
        reasoning=signal.reasoning,
    )

    logger.info(
        "[Entry] 発注完了: %s %s lot=%.2f entry=%.5f sl=%.5f tp=%.5f ticket=%s",
        symbol, direction, lot, entry_price, sl_price, tp_price, ticket,
    )


# ── 孤立トレード照合 ───────────────────

def _reconcile_orphaned_db_trades():
    """DBでOPENのままだがMT5に存在しないトレードをMT5履歴で照合してCLOSEDに更新する。
    MT5のTP/SL自動決済がボットのサイクル間に起きた場合でもクールダウンが正しく機能するようにする。
    """
    db_open_trades = trade_logger.get_open_trades()
    if not db_open_trades:
        return

    mt5_open_tickets = {p["ticket"] for p in mt5_connector.get_positions()}

    for trade in db_open_trades:
        ticket = trade.get("mt5_ticket")
        if ticket is None:
            continue
        if ticket in mt5_open_tickets:
            continue  # MT5にまだ存在 → 正常

        # MT5にないがDBはOPEN → 履歴から決済情報を取得
        deal_info = mt5_connector.get_closed_deal_by_ticket(ticket)
        if deal_info:
            # シンボル不一致検知: dealが別銘柄のものならデータ不正のため無視
            deal_symbol = deal_info.get("symbol", "")
            expected_symbol = trade.get("symbol", "")
            if deal_symbol and deal_symbol != expected_symbol:
                logger.error(
                    "[Reconcile] シンボル不一致: DB=%s deal_symbol=%s ticket=%s → 今回はDB更新せず次サイクル再試行",
                    expected_symbol, deal_symbol, ticket,
                )
                continue
            else:
                exit_price  = deal_info["exit_price"]
                exit_profit = deal_info["profit"]
                exit_reason = deal_info["exit_reason"]
                closed_at   = deal_info["closed_at"]
        else:
            # 履歴も取得できない場合は最低限クローズとして記録
            exit_price  = 0.0
            exit_profit = 0.0
            exit_reason = "UNKNOWN_AUTO_CLOSE"
            closed_at   = None  # close_trade内でutcnow()が使われる

        trade_logger.close_trade(
            trade_id=trade["id"],
            exit_price=exit_price,
            result_pips=0,
            result_profit=exit_profit,
            exit_reason=exit_reason,
        )
        # closed_at を履歴の実際の時刻で上書き
        if closed_at:
            trade_logger.update_trade_closed_at(trade["id"], closed_at)

        logger.warning(
            "[Reconcile] DB孤立トレードをCLOSEDに更新: symbol=%s ticket=%s exit_reason=%s profit=%.0f",
            trade.get("symbol"), ticket, exit_reason, exit_profit,
        )
        discord_notifier.send_exit(
            symbol=trade.get("symbol", ""),
            direction=trade.get("direction", ""),
            exit_price=exit_price,
            profit=exit_profit,
            reasoning=f"[Auto-Reconcile] {exit_reason}",
        )


# ── エグジットチェック ──────────────────

def _check_exits():
    """全保有ポジションのエグジット判断"""
    positions = mt5_connector.get_positions()
    if not positions:
        return

    for pos in positions:
        try:
            _check_single_exit(pos)
        except Exception as e:
            logger.error(
                "エグジットチェック例外: ticket=%s %s",
                pos["ticket"], e, exc_info=True,
            )


def _check_single_exit(pos: dict):
    """1ポジションのエグジット判断"""
    symbol = pos["symbol"]
    ticket = pos["ticket"]

    # 監視対象外のポジションはスキップ
    if symbol not in config.SYMBOLS:
        return

    trade = trade_logger.get_open_trade_by_ticket(ticket)

    _manage_profit_protection(pos)

    close_window_reason = _should_flatten_before_market_close()
    if close_window_reason:
        logger.warning(
            "[Exit SessionClose] %s ticket=%s: %s",
            symbol,
            ticket,
            close_window_reason,
        )
        _execute_exit(pos, close_window_reason, action_type="EXIT_SESSION_CLOSE")
        return

    # 週末跨ぎポジション: 市場が開いたら即クローズ (ギャップリスク回避)
    if config.FLAT_BEFORE_WEEKEND_CLOSE_ENABLED and _is_weekend_holdover(pos):
        if mt5_connector.is_symbol_market_active(symbol):
            reason = "週末跨ぎポジション: 市場再開後の即手仕舞い (ギャップリスク回避)"
            logger.warning("[Exit WeekendHoldover] %s ticket=%s: %s", symbol, ticket, reason)
            discord_notifier.send_skip(symbol, f"[緊急] {reason}", notify=True)
            _execute_exit(pos, reason, action_type="EXIT_WEEKEND_HOLDOVER")
        else:
            logger.warning(
                "[Exit] %s ticket=%s: 週末跨ぎポジション保有中 (市場クローズ中 - 再開待機)",
                symbol, ticket,
            )
        return

    # 市場ストレス (スプレッド/ATR急変) 検知中: クローズ閾値を超えた時のみ即クローズ
    stress = market_stress.get_stress_state(symbol)
    if stress and stress.should_close_positions:
        reason = f"[MarketStress-CLOSE] {stress.summary} (spread={stress.spread_at_trigger:.1f} ×{config.MARKET_STRESS_SPREAD_CLOSE_RATIO:.0f}以上)"
        logger.warning("[Exit MarketStress] %s ticket=%s: %s", symbol, ticket, reason)
        discord_notifier.send_skip(symbol, f"[緊急] {reason}", notify=True)
        _execute_exit(pos, reason, action_type="EXIT_MARKET_STRESS")
        return

    if not mt5_connector.is_symbol_market_active(symbol):
        logger.info("[Exit] %s ticket=%s: 市場クローズ/気配停止中 → AI判定スキップ", symbol, ticket)
        return

    # 保有中の監視足は設定で切り替え可能
    # invalidation_price が DB に記録されていれば、それを赤線としてチャートに描画
    invalidation_price: float | None = None
    if trade and trade.get("invalidation_price") is not None:
        try:
            invalidation_price = float(trade["invalidation_price"])
        except (TypeError, ValueError):
            pass

    # SMCオーバーレイ付きチャートを生成（invalidation_price があれば赤線入り）
    if invalidation_price is not None:
        exit_img_b64 = chart_capture.generate_smc_chart_base64(
            symbol, config.EXIT_MONITOR_TF, invalidation_price=invalidation_price
        )
        # base64文字列をbytesに戻す（analyze_exitのインターフェース互換のため）
        import base64 as _b64
        exit_img = _b64.b64decode(exit_img_b64) if exit_img_b64 else None
    else:
        exit_img = chart_capture.generate_chart(symbol, config.EXIT_MONITOR_TF)
    if exit_img is None:
        return

    # 保有時間計算
    hold_minutes = _calculate_hold_minutes(pos, trade)

    # ── 機械EXIT優先: AI前に確定条件を評価 ──
    should_exit_mech, mech_reason, mech_action = _evaluate_mechanical_exit(
        pos=pos,
        trade=trade,
        hold_minutes=hold_minutes,
        invalidation_price=invalidation_price,
    )
    if should_exit_mech:
        logger.warning("[Exit Mechanical] %s ticket=%s: %s", symbol, ticket, mech_reason)
        _execute_exit(pos, mech_reason, action_type=mech_action)
        return

    # ── Exit AI コスト制御: 微小変化時のみスキップ ──
    current_price = pos["price_current"]
    prev_price = _exit_price_cache.get(ticket)
    if prev_price is not None and prev_price > 0:
        price_move_pct = abs(current_price - prev_price) / prev_price * 100
        if price_move_pct < config.EXIT_AI_SKIP_MIN_MOVE_PCT:
            logger.info(
                "[Exit] %s ticket=%s: 価格変化微小 (%.4f%% < %.4f%%) → AIスキップ",
                symbol, ticket, price_move_pct, config.EXIT_AI_SKIP_MIN_MOVE_PCT,
            )
            return
    _exit_price_cache[ticket] = current_price

    # AI 分析
    signal = ai_analyzer.analyze_exit(
        symbol=symbol,
        direction=pos["type"],
        entry_price=pos["price_open"],
        current_price=pos["price_current"],
        unrealized_pnl=pos["profit"],
        hold_minutes=hold_minutes,
        m15_image=exit_img,
        entry_reasoning=(trade.get("ai_reasoning", "") if trade else ""),
        entry_news_impact=(trade.get("news_summary", "") if trade else ""),
        tp_price=(trade.get("tp_price") if trade else pos.get("tp")),
        current_sl=pos.get("sl"),
        invalidation_price=invalidation_price,
    )

    # AIログ保存
    trade_logger.insert_ai_log(
        symbol=symbol,
        action_type="EXIT_CHECK",
        ai_response=signal.raw_response[:2000],
        decision=signal.decision,
        reasoning=signal.reasoning[:1000],
    )

    logger.info(
        "[Exit] %s ticket=%s: AI判断=%s premise_valid=%s conf=%d 理由=%s",
        symbol, ticket, signal.decision, signal.entry_premise_valid, signal.confidence,
        signal.reasoning[:100],
    )

    if (
        config.FORCE_EXIT_ON_PREMISE_BREAK
        and invalidation_price is None
        and not signal.entry_premise_valid
    ):
        if hold_minutes < config.MIN_HOLD_MINUTES_BEFORE_FORCE_PREMISE_BREAK:
            logger.info(
                "[Exit] %s ticket=%s: 根拠崩壊だが保有時間不足 (%d < %d min) → 監視継続",
                symbol,
                ticket,
                hold_minutes,
                config.MIN_HOLD_MINUTES_BEFORE_FORCE_PREMISE_BREAK,
            )
            return
        logger.warning("[Exit] %s ticket=%s: エントリー根拠崩壊 → 強制EXIT", symbol, ticket)
        _execute_exit(pos, signal.reasoning, action_type="EXIT_PREMISE_BREAK")
        return

    # EXIT判定（AI中心）
    required_conf = config.EXIT_MIN_CONFIDENCE
    if hold_minutes < config.EXIT_EARLY_WINDOW_MINUTES:
        required_conf = max(required_conf, config.EXIT_MIN_CONFIDENCE_EARLY)

    if signal.decision == "EXIT" and signal.confidence >= required_conf:
        _execute_exit(pos, signal.reasoning, action_type="EXIT_CHECK")
        return

    if signal.decision == "EXIT" and signal.confidence < required_conf:
        logger.info(
            "[Exit] %s: 信頼度不足 %d < %d (hold=%d min) → HOLD継続",
            symbol,
            signal.confidence,
            required_conf,
            hold_minutes,
        )

    # AIが撤退しない場合のみ、機械的な緊急撤退をフォールバック適用
    emergency_exit, emergency_reason = _should_emergency_exit(pos)
    if emergency_exit:
        logger.warning(
            "[Exit Emergency:FALLBACK] %s ticket=%s: %s",
            symbol, ticket, emergency_reason,
        )
        _execute_exit(pos, emergency_reason, action_type="EXIT_EMERGENCY")


def _evaluate_mechanical_exit(
    pos: dict,
    trade: dict | None,
    hold_minutes: int,
    invalidation_price: float | None,
) -> tuple[bool, str, str]:
    """AI前に適用する機械的なEXIT判定。"""
    symbol = pos["symbol"]
    direction = str(pos["type"]).upper()
    current_price = float(pos.get("price_current") or 0)

    # 1) 構造崩壊: 無効化ラインを「確定足終値」でブレイク
    if invalidation_price is not None:
        close_price = _get_latest_confirmed_close(symbol, config.EXIT_MONITOR_TF)
        if close_price is not None:
            breached = (
                (direction == "BUY" and close_price < invalidation_price)
                or (direction == "SELL" and close_price > invalidation_price)
            )
            if breached:
                if hold_minutes < config.MIN_HOLD_MINUTES_BEFORE_FORCE_PREMISE_BREAK:
                    return (
                        False,
                        (
                            f"構造崩壊シグナルは検出(close={close_price:.5f}, inv={invalidation_price:.5f})"
                            f"だが保有時間不足({hold_minutes}<{config.MIN_HOLD_MINUTES_BEFORE_FORCE_PREMISE_BREAK}分)"
                        ),
                        "",
                    )
                return (
                    True,
                    (
                        f"無効化ライン終値ブレイクで構造崩壊: close={close_price:.5f}, "
                        f"invalidation={invalidation_price:.5f}"
                    ),
                    "EXIT_PREMISE_BREAK_MECH",
                )

    # 2) TP到達は即利確
    tp_price_raw = (trade.get("tp_price") if trade else None)
    if tp_price_raw is None:
        tp_price_raw = pos.get("tp")
    try:
        tp_price = float(tp_price_raw) if tp_price_raw is not None else None
    except (TypeError, ValueError):
        tp_price = None

    if tp_price is not None:
        tp_hit = (direction == "BUY" and current_price >= tp_price) or (
            direction == "SELL" and current_price <= tp_price
        )
        if tp_hit:
            return (
                True,
                f"TP到達による機械利確: current={current_price:.5f} tp={tp_price:.5f}",
                "EXIT_TP_HIT_MECH",
            )

    # 3) TP目前 + 反転シグナルで先行利確
    if config.EXIT_TP_NEAR_ENABLED and tp_price is not None:
        entry_price = float(pos.get("price_open") or 0)
        df = mt5_connector.get_rates(symbol, config.EXIT_MONITOR_TF, max(config.MA_PERIOD + 5, 40))
        if df is not None and len(df) >= config.MA_PERIOD + 3:
            atr = float(mt5_connector.calculate_atr(df, config.ATR_PERIOD))
            r_dist = abs(tp_price - entry_price)
            near_dist = max(atr * config.EXIT_TP_NEAR_ATR_MULT, r_dist * config.EXIT_TP_NEAR_R_MULT)
            tp_near = (
                (direction == "BUY" and (tp_price - current_price) <= near_dist)
                or (direction == "SELL" and (current_price - tp_price) <= near_dist)
            )

            o = df["open"]
            c = df["close"]
            ma20 = mt5_connector.calculate_ma(df, config.MA_PERIOD)
            last_open = float(o.iloc[-2])
            last_close = float(c.iloc[-2])
            last_ma = float(ma20.iloc[-2])

            reversal = (
                (direction == "BUY" and last_close < last_open and last_close < last_ma)
                or (direction == "SELL" and last_close > last_open and last_close > last_ma)
            )
            if tp_near and reversal:
                return (
                    True,
                    (
                        f"TP目前で反転確認により機械利確: current={current_price:.5f}, "
                        f"tp={tp_price:.5f}, near_dist={near_dist:.5f}"
                    ),
                    "EXIT_TP_NEAR_REVERSAL_MECH",
                )

    return False, "", ""


def _get_latest_confirmed_close(symbol: str, timeframe: str) -> float | None:
    """最新の確定足終値を返す。取得失敗時はNone。"""
    df = mt5_connector.get_rates(symbol, timeframe, 5)
    if df is None or len(df) < 3:
        return None
    try:
        # 末尾[-1]は進行中バーの可能性があるため[-2]を採用
        return float(df["close"].iloc[-2])
    except Exception:
        return None


def _manage_profit_protection(pos: dict):
    """期待値を守るため、利益が乗ったポジションのSLを段階的に引き上げる。"""
    if not config.PROFIT_PROTECTION_ENABLED:
        return

    symbol = pos["symbol"]
    ticket = pos["ticket"]
    trade = trade_logger.get_open_trade_by_ticket(ticket)
    if trade is None:
        return

    initial_sl = trade.get("sl_price")
    entry_price = pos["price_open"]
    current_price = pos["price_current"]
    current_sl = pos.get("sl") or initial_sl
    if initial_sl is None or current_sl is None:
        return

    initial_risk = abs(entry_price - initial_sl)
    if initial_risk <= 0:
        return

    if pos["type"] == "BUY":
        profit_distance = current_price - entry_price
    else:
        profit_distance = entry_price - current_price
    if profit_distance <= 0:
        return

    r_multiple = profit_distance / initial_risk
    candidate_sl = None
    reason = ""

    if r_multiple >= config.LOCK_PROFIT_2_TRIGGER_R:
        candidate_sl = _sl_from_r(pos["type"], entry_price, initial_risk, config.LOCK_PROFIT_2_R)
        reason = f"lock profit {config.LOCK_PROFIT_2_R:.2f}R"
    elif r_multiple >= config.LOCK_PROFIT_1_TRIGGER_R:
        candidate_sl = _sl_from_r(pos["type"], entry_price, initial_risk, config.LOCK_PROFIT_1_R)
        reason = f"lock profit {config.LOCK_PROFIT_1_R:.2f}R"
    elif r_multiple >= config.BREAKEVEN_R:
        candidate_sl = _sl_from_r(pos["type"], entry_price, initial_risk, config.BREAKEVEN_BUFFER_R)
        reason = f"breakeven+{config.BREAKEVEN_BUFFER_R:.2f}R"

    if candidate_sl is None:
        return

    digits = mt5_connector.get_symbol_info(symbol)["digits"]
    candidate_sl = round(candidate_sl, digits)

    if not _is_better_sl(pos["type"], current_sl, candidate_sl):
        return

    if mt5_connector.modify_position_sl(ticket, candidate_sl):
        trade_logger.update_trade_sl_by_ticket(ticket, candidate_sl)
        trade_logger.insert_ai_log(
            symbol=symbol,
            action_type="PROFIT_PROTECTION",
            ai_response=reason,
            decision="SL_UPDATE",
            reasoning=f"ticket={ticket} r_multiple={r_multiple:.2f} new_sl={candidate_sl}",
        )
        logger.info(
            "[Profit Protection] %s ticket=%s r=%.2f new_sl=%.5f reason=%s",
            symbol, ticket, r_multiple, candidate_sl, reason,
        )


def _should_emergency_exit(pos: dict) -> tuple[bool, str]:
    """AI判断とは別に、構造崩れ時だけ機械的に撤退する。"""
    if not config.EMERGENCY_EXIT_ENABLED:
        return False, ""

    symbol = pos["symbol"]
    df = mt5_connector.get_rates(symbol, config.EXIT_MONITOR_TF, config.CHART_BARS + 40)
    if df is None or len(df) < max(config.MA_PERIOD, config.ATR_PERIOD) + 5:
        return False, ""

    ma = mt5_connector.calculate_ma(df, config.MA_PERIOD)
    current_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    current_ma = ma.iloc[-1]
    prev_ma = ma.iloc[-2]
    if current_ma is None or prev_ma is None:
        return False, ""

    atr_series = _calculate_atr_series(df, config.ATR_PERIOD)
    current_atr = float(atr_series.iloc[-1])
    baseline_atr = float(atr_series.iloc[-(config.EMERGENCY_EXIT_ATR_SPIKE_LOOKBACK + 1):-1].median())
    if current_atr <= 0 or baseline_atr <= 0:
        return False, ""

    if pos["type"] == "BUY":
        adverse_move = float(pos["price_open"] - current_close)
        structure_break = current_close < current_ma and prev_close < prev_ma
    else:
        adverse_move = float(current_close - pos["price_open"])
        structure_break = current_close > current_ma and prev_close > prev_ma

    adverse_atr_threshold = config.EMERGENCY_EXIT_ADVERSE_ATR_BY_SYMBOL.get(
        symbol,
        config.EMERGENCY_EXIT_ADVERSE_ATR,
    )
    atr_spike = current_atr >= baseline_atr * config.EMERGENCY_EXIT_ATR_SPIKE_MULTIPLIER
    adverse_break = adverse_move >= current_atr * adverse_atr_threshold

    if structure_break and adverse_break:
        return (
            True,
            f"M{config.EXIT_MONITOR_TF}構造崩れ + 逆行{adverse_move:.5f} >= ATR{current_atr:.5f} x {adverse_atr_threshold}",
        )

    if structure_break and atr_spike and adverse_move >= current_atr * config.EMERGENCY_EXIT_ATR_SPIKE_MIN_ADVERSE_ATR:
        return (
            True,
            f"M{config.EXIT_MONITOR_TF}構造崩れ + ATR急拡大 ({current_atr:.5f} >= {baseline_atr:.5f} x {config.EMERGENCY_EXIT_ATR_SPIKE_MULTIPLIER}) + 逆行{adverse_move:.5f} >= ATR{current_atr:.5f} x {config.EMERGENCY_EXIT_ATR_SPIKE_MIN_ADVERSE_ATR}",
        )

    return False, ""


def _calculate_atr_series(df, period: int):
    prev_close = df["close"].shift(1)
    tr = (df["high"] - df["low"]).to_frame("hl")
    tr["hc"] = (df["high"] - prev_close).abs()
    tr["lc"] = (df["low"] - prev_close).abs()
    true_range = tr.max(axis=1)
    return true_range.rolling(window=period).mean().bfill()


def _sl_from_r(direction: str, entry_price: float, initial_risk: float, target_r: float) -> float:
    if direction == "BUY":
        return entry_price + initial_risk * target_r
    return entry_price - initial_risk * target_r


def _is_better_sl(direction: str, current_sl: float, candidate_sl: float) -> bool:
    if direction == "BUY":
        return candidate_sl > current_sl
    return candidate_sl < current_sl


def _execute_exit(pos: dict, reasoning: str, action_type: str):
    """決済共通処理。"""
    _ACTION_TO_EXIT_REASON = {
        "EXIT_CHECK":        "AI_EXIT",
        "EXIT_EMERGENCY":    "EMERGENCY",
        "EXIT_PREMISE_BREAK": "PREMISE_BREAK",
        "EXIT_SESSION_CLOSE": "SESSION_CLOSE",
    }
    exit_reason = _ACTION_TO_EXIT_REASON.get(action_type, action_type)
    symbol = pos["symbol"]
    ticket = pos["ticket"]

    trade_logger.insert_ai_log(
        symbol=symbol,
        action_type=action_type,
        ai_response=reasoning[:2000],
        decision="EXIT",
        reasoning=reasoning[:1000],
    )

    # 決済実行
    success = mt5_connector.close_position(ticket)
    if not success:
        return

    # DB更新
    db_trade = trade_logger.get_open_trade_by_ticket(ticket)
    if db_trade:
        trade_logger.close_trade(
            trade_id=db_trade["id"],
            exit_price=pos["price_current"],
            result_pips=0,  # 後で正確に計算可能
            result_profit=pos["profit"],
            exit_reason=exit_reason,
        )

    # Discord通知
    discord_notifier.send_exit(
        symbol=symbol,
        direction=pos["type"],
        exit_price=pos["price_current"],
        profit=pos["profit"],
        reasoning=reasoning,
    )


# ── Heartbeat ──────────────────────────

def _send_heartbeat():
    """1時間ごとのHeartbeat通知"""
    account = mt5_connector.get_account_info()
    if account is None:
        discord_notifier.send_error("Heartbeat失敗", "アカウント情報取得不可")
        return

    positions = mt5_connector.get_positions()
    discord_notifier.send_heartbeat(
        balance=account["balance"],
        equity=account["equity"],
        open_positions=len(positions),
    )
    trade_logger.insert_heartbeat(
        "OK",
        f"balance={account['balance']:.0f} equity={account['equity']:.0f} "
        f"positions={len(positions)}",
    )
    logger.info(
        "[Heartbeat] balance=%.0f equity=%.0f positions=%d",
        account["balance"], account["equity"], len(positions),
    )


def _run_db_maintenance(full_vacuum: bool):
    """SQLiteの定期メンテナンスを実行する。"""
    try:
        stats = trade_logger.run_maintenance(full_vacuum=full_vacuum)
        logger.info(
            "[DB] maintenance done: ai=%d hb=%d closed=%d trim_ai=%d trim_hb=%d size=%.2fMB->%.2fMB vacuum=%s",
            stats["deleted_ai_logs"],
            stats["deleted_heartbeats"],
            stats["deleted_closed_trades"],
            stats["trimmed_ai_logs"],
            stats["trimmed_heartbeats"],
            stats["db_size_mb_before"],
            stats["db_size_mb_after"],
            stats["vacuum_executed"],
        )
        _refresh_regime_dashboard()
        _run_adaptive_eval()
    except Exception as e:
        logger.error("[DB] maintenance failed: %s", e, exc_info=True)
        discord_notifier.send_error("SQLiteメンテ失敗", str(e))


def _refresh_regime_dashboard():
    """レジーム別期待値ダッシュボードを定期生成する。"""
    if not config.DASHBOARD_ENABLED:
        return
    try:
        dashboard = trade_logger.build_regime_dashboard()
        logger.info(
            "[Dashboard] generated: lookback=%d days output=%s trades=%s",
            dashboard["lookback_days"],
            dashboard["output_path"],
            dashboard.get("overview", {}).get("trades"),
        )
    except Exception as e:
        logger.error("[Dashboard] generation failed: %s", e, exc_info=True)


def _run_adaptive_eval():
    """アダプティブ評価: 直近トレードから閾値を自動更新する。"""
    if not config.ADAPTIVE_ENABLED:
        return
    try:
        result = adaptive_params.evaluate_and_adapt()
        if result.get("skipped"):
            logger.info("[Adaptive] skipped: %s", result.get("reason"))
        else:
            logger.info(
                "[Adaptive] eval done: trades=%d buckets=%d updated=%d",
                result["total_trades"],
                result["buckets_evaluated"],
                result["buckets_updated"],
            )
    except Exception as e:
        logger.error("[Adaptive] eval failed: %s", e, exc_info=True)


# ── エントリポイント ────────────────────

if __name__ == "__main__":
    main()
