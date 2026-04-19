"""AI自動売買システム メインループ

15分足確定ごとに5銘柄を巡回:
  - 新規エントリー: H1+M15画像をAI分析 → 相関チェック → ロット計算 → 発注
  - 保有ポジション: M15画像をAI分析 → エグジット判断
  - 毎時Heartbeat通知
"""

import sys
import time
import logging
from datetime import datetime

import config
import mt5_connector
import chart_capture
import lot_calculator
import risk_manager
import ai_analyzer
import discord_notifier
import trade_logger

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


# ── メインループ ────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("AI Trading System 起動")
    logger.info("監視銘柄: %s", config.SYMBOLS)
    logger.info("=" * 60)

    # DB 初期化
    trade_logger.init_db()
    _run_db_maintenance(full_vacuum=False)

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
) -> tuple[bool, bool, bool, str, str]:
    """H1データのみでSMC条件を数値判定する機械ゲート。AI呼び出し前のコスト削減フィルタ。

        Returns: (sweep_pass, bos_pass, rr_pass, sweep_type, entry_type)
      sweep_type: "HIGH" / "LOW" / "NONE"
            entry_type: "REVERSAL_SWEEP" / "CONTINUATION_BOS" / "NONE"
    """
    if not smc_data or atr_h1 <= 0:
                return False, False, False, "NONE", "NONE"

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
        return False, False, False, "NONE", "NONE"

    min_penetration = atr_h1 * config.SMC_SWEEP_ATR_MULT
    lookback = min(config.SMC_SWEEP_LOOKBACK_BARS, len(df_h1) - 1)
    recent = df_h1.iloc[-lookback:]

    sweep_pass = False
    sweep_type = "NONE"
    swept_level: float | None = None

    for level in levels:
        for _, bar in recent.iterrows():
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
    sl_dist = atr_h1 * 1.5
    min_tp_dist = sl_dist * config.ENTRY_MIN_TP_R
    rr_pass = False

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

        return sweep_pass, bos_pass, rr_pass, sweep_type, "REVERSAL_SWEEP"

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
                targets = [v for v in levels if v > current_price]
                rr_pass = bool(targets) and (min(targets) - current_price) >= min_tp_dist
                return False, bos_pass, rr_pass, cont_sweep_type, "CONTINUATION_BOS"
            elif ma_slope < 0 and current_price >= ma_curr - atr_h1 * 0.5:
                # 下降トレンド + 価格がMA付近以上: SELL戻しセットアップ
                cont_sweep_type = "HIGH"
                bos_pass = True
                targets = [v for v in levels if v < current_price]
                rr_pass = bool(targets) and (current_price - max(targets)) >= min_tp_dist
                return False, bos_pass, rr_pass, cont_sweep_type, "CONTINUATION_BOS"

    return False, False, False, "NONE", "NONE"


# ── エントリーチェック ──────────────────

def _check_entry(symbol: str):
    """1銘柄のエントリー判断"""

    # 既にポジション保有中ならスキップ
    positions = mt5_connector.get_positions(symbol)
    if positions:
        logger.info("[Entry] %s: ポジション保有中 → スキップ", symbol)
        return

    if not mt5_connector.is_symbol_market_active(symbol):
        logger.info("[Entry] %s: 市場クローズ/気配停止中 → AI判定スキップ", symbol)
        return

    # 相関リスクチェック
    can_open, reason = risk_manager.can_open_position(symbol)
    if not can_open:
        logger.info("[Entry] %s: 相関リスク超過 → スキップ (%s)", symbol, reason)
        discord_notifier.send_skip(symbol, reason)
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
    smc_sweep_pass, smc_bos_pass, smc_rr_pass, mech_sweep_type, mech_entry_type = _mechanical_smc_gate(
        df_h1, atr_h1, smc_data, current_price
    )
    logger.info(
        "[Entry] %s: MechGate sweep=%s bos=%s rr=%s type=%s entry_type=%s",
        symbol, smc_sweep_pass, smc_bos_pass, smc_rr_pass, mech_sweep_type, mech_entry_type,
    )

    if config.SMC_FILTER_ENABLED and config.SMC_MECHANICAL_GATE_ENABLED:
        if mech_entry_type == "NONE":
            logger.info("[Entry] %s: 機械ゲート: セットアップ未検出 → AIコスト節約スキップ", symbol)
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

    # チャート画像生成
    h1_img, m15_img = chart_capture.generate_chart_pair(symbol)
    if h1_img is None or m15_img is None:
        logger.warning("[Entry] %s: チャート画像生成失敗", symbol)
        return

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
        logger.info("[Entry] %s: AI判断 SKIP (conf=%d)", symbol, signal.confidence)
        return

    if not signal.alignment:
        logger.info("[Entry] %s: H1/M15トレンド不一致 → スキップ", symbol)
        return

    if signal.confidence < 60:
        logger.info("[Entry] %s: 信頼度不足 %d < 60 → スキップ", symbol, signal.confidence)
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

    # SL距離: AIの推奨値 or ATR × 1.5
    sl_distance = signal.sl_distance
    if sl_distance <= 0:
        sl_distance = lot_calculator.get_sl_distance(atr_m15)

    # ロット計算
    lot = lot_calculator.calculate_lot(symbol, sl_distance)
    if lot is None:
        logger.warning("[Entry] %s: ロット計算失敗 → スキップ", symbol)
        discord_notifier.send_error(f"ロット計算失敗: {symbol}", "calculate_lot returned None")
        return

    # SL価格算出
    entry_price = price_info["ask"] if direction == "BUY" else price_info["bid"]
    sym_info = mt5_connector.get_symbol_info(symbol)
    digits = sym_info["digits"] if sym_info else 5

    if direction == "BUY":
        sl_price = round(entry_price - sl_distance, digits)
    else:
        sl_price = round(entry_price + sl_distance, digits)

    tp_distance = signal.tp_distance
    min_tp_distance = sl_distance * config.ENTRY_MIN_TP_R
    if tp_distance < min_tp_distance:
        tp_distance = min_tp_distance

    if direction == "BUY":
        tp_price = round(entry_price + tp_distance, digits)
    else:
        tp_price = round(entry_price - tp_distance, digits)

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

    emergency_exit, emergency_reason = _should_emergency_exit(pos)
    if emergency_exit:
        logger.warning(
            "[Exit Emergency] %s ticket=%s: %s",
            symbol, ticket, emergency_reason,
        )
        _execute_exit(pos, emergency_reason, action_type="EXIT_EMERGENCY")
        return

    if not mt5_connector.is_symbol_market_active(symbol):
        logger.info("[Exit] %s ticket=%s: 市場クローズ/気配停止中 → AI判定スキップ", symbol, ticket)
        return

    # 保有中の監視足は設定で切り替え可能
    exit_img = chart_capture.generate_chart(symbol, config.EXIT_MONITOR_TF)
    if exit_img is None:
        return

    # 保有時間計算
    hold_minutes = int((datetime.now() - pos["time"]).total_seconds() / 60)

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

    if config.FORCE_EXIT_ON_PREMISE_BREAK and not signal.entry_premise_valid:
        logger.warning("[Exit] %s ticket=%s: エントリー根拠崩壊 → 強制EXIT", symbol, ticket)
        _execute_exit(pos, signal.reasoning, action_type="EXIT_PREMISE_BREAK")
        return

    # EXIT判定
    if signal.decision != "EXIT":
        return

    if signal.confidence < config.EXIT_MIN_CONFIDENCE:
        logger.info(
            "[Exit] %s: 信頼度不足 %d < %d → HOLD継続",
            symbol,
            signal.confidence,
            config.EXIT_MIN_CONFIDENCE,
        )
        return

    _execute_exit(pos, signal.reasoning, action_type="EXIT_CHECK")


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

    if structure_break and atr_spike and adverse_move > 0:
        return (
            True,
            f"M{config.EXIT_MONITOR_TF}構造崩れ + ATR急拡大 ({current_atr:.5f} >= {baseline_atr:.5f} x {config.EMERGENCY_EXIT_ATR_SPIKE_MULTIPLIER})",
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


# ── エントリポイント ────────────────────

if __name__ == "__main__":
    main()
