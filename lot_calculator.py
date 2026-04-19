"""資金管理 & ロット計算モジュール

【重要】ゴールド等での過剰ロットを防ぐため、以下を厳密に実装:
  1. 許容損失額 = 口座残高 × RISK_PER_TRADE (2%)
  2. 銘柄の profit_currency を MT5 symbol_info から動的取得
  3. 口座通貨 (JPY) → profit_currency への換算
  4. contract_size を MT5 から動的取得 (GOLD=100oz 等)
  5. ロット数 = 許容損失額(profit通貨) / (SL幅 × contract_size)
  6. min/max/step でクランプ + 余剰証拠金チェック
"""

import math
import logging

import config
import mt5_connector

logger = logging.getLogger(__name__)


def calculate_lot(symbol: str, sl_distance: float) -> float | None:
    """
    ATRベースのSL幅から、リスク2%に収まるロット数を算出する。

    Args:
        symbol: MT5シンボル名 (例: "GOLD", "USDJPY")
        sl_distance: SLまでの価格幅 (正の値、price単位)

    Returns:
        ロット数 (float)。計算不能時は None。
    """
    if sl_distance <= 0:
        logger.error("SL幅が0以下: %s sl_distance=%s", symbol, sl_distance)
        return None

    # ── 1. アカウント情報取得 ──
    account = mt5_connector.get_account_info()
    if account is None:
        logger.error("アカウント情報取得失敗")
        return None

    balance = account["balance"]
    margin_free = account["margin_free"]
    account_currency = account["currency"]  # 通常 "JPY"

    if balance <= 0:
        logger.error("残高が0以下: %.0f", balance)
        return None

    # ── 2. シンボル情報取得 ──
    sym_info = mt5_connector.get_symbol_info(symbol)
    if sym_info is None:
        logger.error("シンボル情報取得失敗: %s", symbol)
        return None

    contract_size = sym_info["trade_contract_size"]
    profit_currency = sym_info["currency_profit"]
    vol_min = sym_info["volume_min"]
    vol_max = sym_info["volume_max"]
    vol_step = sym_info["volume_step"]

    logger.info(
        "[LotCalc] %s: contract_size=%.2f, profit_currency=%s, "
        "vol_min=%.2f, vol_max=%.2f, vol_step=%.2f",
        symbol, contract_size, profit_currency,
        vol_min, vol_max, vol_step,
    )

    # ── 3. 許容損失額 (口座通貨) ──
    max_loss_account = balance * config.RISK_PER_TRADE
    logger.info(
        "[LotCalc] 許容損失額: %.0f %s (残高 %.0f × %.1f%%)",
        max_loss_account, account_currency, balance, config.RISK_PER_TRADE * 100,
    )

    # ── 4. 許容損失額を profit_currency に変換 ──
    if account_currency == profit_currency:
        max_loss_profit_ccy = max_loss_account
    else:
        # 口座通貨 → profit_currency の変換レート
        # 例: JPY口座, profit_currency=USD → USDJPY で割る
        rate = _get_conversion_rate(account_currency, profit_currency)
        if rate is None or rate <= 0:
            logger.error(
                "通貨換算レート取得失敗: %s → %s",
                account_currency, profit_currency,
            )
            return None
        max_loss_profit_ccy = max_loss_account / rate
        logger.info(
            "[LotCalc] 換算: %.0f %s → %.2f %s (rate=%.4f)",
            max_loss_account, account_currency,
            max_loss_profit_ccy, profit_currency, rate,
        )

    # ── 5. ロット数計算 ──
    # 1ロットの損失 = SL幅 × contract_size (profit_currency 単位)
    loss_per_lot = sl_distance * contract_size
    if loss_per_lot <= 0:
        logger.error("loss_per_lot が0以下: %s", loss_per_lot)
        return None

    raw_lot = max_loss_profit_ccy / loss_per_lot
    logger.info(
        "[LotCalc] raw_lot = %.2f %s / (%.5f × %.0f) = %.6f",
        max_loss_profit_ccy, profit_currency,
        sl_distance, contract_size, raw_lot,
    )

    # ── 6. ロット数をMT5の制約に合わせる ──
    lot = _round_lot(raw_lot, vol_step)
    lot = max(lot, vol_min)
    lot = min(lot, vol_max)
    lot = min(lot, config.MAX_LOT)

    # ── 7. 余剰証拠金チェック ──
    lot = _check_margin(symbol, lot, margin_free, sym_info)

    if lot < vol_min:
        logger.warning(
            "[LotCalc] 最終ロット %.4f が最小ロット %.2f 未満 → 発注不可",
            lot, vol_min,
        )
        return None

    logger.info("[LotCalc] 最終ロット: %s → %.2f lot", symbol, lot)
    return lot


def calculate_sl_price(symbol: str, direction: str, entry_price: float,
                       atr: float, multiplier: float = 1.5) -> float:
    """ATRベースでSL価格を算出する"""
    sl_distance = atr * multiplier
    if direction == "BUY":
        return round(entry_price - sl_distance, _get_digits(symbol))
    else:
        return round(entry_price + sl_distance, _get_digits(symbol))


def get_sl_distance(atr: float, multiplier: float = 1.5) -> float:
    """ATRからSL幅 (price単位) を返す"""
    return atr * multiplier


# ── 内部ヘルパー ────────────────────────

def _get_conversion_rate(from_ccy: str, to_ccy: str) -> float | None:
    """from_ccy → to_ccy の変換レートを取得

    例: JPY → USD → USDJPY レートを返す (JPY を USD に変えるには USDJPY で割る)
    """
    if from_ccy == to_ccy:
        return 1.0

    # JPY → USD: USDJPY を返す (max_loss_jpy / usdjpy = max_loss_usd)
    if from_ccy == "JPY" and to_ccy == "USD":
        return mt5_connector.get_usdjpy_rate()

    # JPY → EUR: EURJPY を返す
    if from_ccy == "JPY":
        pair = to_ccy + "JPY"
        tick = _get_mid_price(pair)
        if tick is not None:
            return tick
        return None

    # USD → JPY: 1 / USDJPY
    if from_ccy == "USD" and to_ccy == "JPY":
        rate = mt5_connector.get_usdjpy_rate()
        return 1.0 / rate if rate > 0 else None

    # その他: from → JPY → to で間接変換
    from_to_jpy = _get_conversion_rate(from_ccy, "JPY")
    to_to_jpy = _get_conversion_rate(to_ccy, "JPY")
    if from_to_jpy and to_to_jpy and to_to_jpy > 0:
        return from_to_jpy / to_to_jpy

    return None


def _get_mid_price(symbol: str) -> float | None:
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return (tick.bid + tick.ask) / 2.0


def _round_lot(lot: float, step: float) -> float:
    """ロット数を volume_step の倍数に切り捨て"""
    if step <= 0:
        return lot
    return math.floor(lot / step) * step


def _get_digits(symbol: str) -> int:
    sym_info = mt5_connector.get_symbol_info(symbol)
    if sym_info:
        return sym_info["digits"]
    return 5


def _check_margin(symbol: str, lot: float, margin_free: float,
                  sym_info: dict) -> float:
    """余剰証拠金に対してロット数が過大でないかチェック

    余剰証拠金の80%以上を使うロットは段階的に縮小する。
    """
    import MetaTrader5 as mt5

    # 安全策: 余剰証拠金の50%を上限とする
    if margin_free <= 0:
        logger.warning("余剰証拠金が0以下 → 最小ロット")
        return sym_info["volume_min"]

    # 段階的縮小: 大きすぎるロットは半分にして再チェック
    max_iterations = 10
    for _ in range(max_iterations):
        ask = sym_info.get("ask") or 0.0
        bid = sym_info.get("bid") or 0.0
        price = ask if ask > 0 else bid
        if price <= 0:
            logger.warning("%s: 価格取得失敗のため最小ロットへフォールバック", symbol)
            return sym_info["volume_min"]
        else:
            margin_buy = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot, price)
            margin_sell = mt5.order_calc_margin(mt5.ORDER_TYPE_SELL, symbol, lot, price)
            # BUY/SELLのうち大きい方を採用して保守的に判定
            candidates = [m for m in (margin_buy, margin_sell) if m is not None and m > 0]
            if candidates:
                estimated_margin = max(candidates)
            else:
                contract_size = sym_info["trade_contract_size"]
                estimated_margin = price * contract_size * lot * 0.01

        if estimated_margin < margin_free * 0.5:
            break
        lot = _round_lot(lot * 0.5, sym_info["volume_step"])
        if lot < sym_info["volume_min"]:
            return sym_info["volume_min"]

    return lot
