"""MT5 接続・データ取得・注文管理モジュール"""

import time
import logging
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import config
import discord_notifier

logger = logging.getLogger(__name__)

# タイムフレーム変換テーブル
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


# ── 接続管理 ────────────────────────────

def initialize() -> bool:
    if not mt5.initialize(path=config.MT5_PATH):
        err = mt5.last_error()
        logger.error("MT5初期化失敗: %s", err)
        discord_notifier.send_error("MT5初期化失敗", str(err))
        return False

    if config.MT5_LOGIN:
        ok = mt5.login(
            login=config.MT5_LOGIN,
            password=config.MT5_PASSWORD,
            server=config.MT5_SERVER,
        )
        if not ok:
            err = mt5.last_error()
            logger.error("MT5ログイン失敗: %s", err)
            discord_notifier.send_error("MT5ログイン失敗", str(err))
            return False

    info = mt5.account_info()
    if info is None:
        logger.error("アカウント情報取得失敗")
        return False

    logger.info(
        "MT5接続成功: server=%s login=%s balance=%.0f %s",
        info.server, info.login, info.balance, info.currency,
    )
    return True


def shutdown():
    mt5.shutdown()
    logger.info("MT5シャットダウン完了")


def ensure_connected() -> bool:
    info = mt5.account_info()
    if info is not None:
        return True
    logger.warning("MT5接続断検知 → 再接続試行")
    for attempt in range(3):
        if initialize():
            return True
        time.sleep(2 ** attempt)
    discord_notifier.send_error("MT5接続断", "3回の再接続試行すべて失敗")
    return False


# ── アカウント情報 ──────────────────────

def get_account_info() -> dict | None:
    info = mt5.account_info()
    if info is None:
        return None
    return {
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "margin_free": info.margin_free,
        "currency": info.currency,
        "login": info.login,
    }


# ── シンボル情報 ────────────────────────

def get_symbol_info(symbol: str) -> dict | None:
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error("シンボル情報取得失敗: %s", symbol)
        return None
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
    return {
        "name": info.name,
        "bid": info.bid,
        "ask": info.ask,
        "spread": info.spread,
        "digits": info.digits,
        "trade_contract_size": info.trade_contract_size,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "currency_base": info.currency_base,
        "currency_profit": info.currency_profit,
        "currency_margin": info.currency_margin,
        "trade_tick_size": info.trade_tick_size,
        "trade_tick_value": info.trade_tick_value,
    }


# ── レート取得 ──────────────────────────

def get_rates(symbol: str, timeframe: str, count: int = 200) -> pd.DataFrame | None:
    tf = TF_MAP.get(timeframe)
    if tf is None:
        logger.error("不明なタイムフレーム: %s", timeframe)
        return None

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        logger.error("レート取得失敗: %s %s", symbol, timeframe)
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def get_current_price(symbol: str) -> dict | None:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}


# ── テクニカル指標 ──────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0
    return float(np.mean(tr[-period:]))


def calculate_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["close"].rolling(window=period).mean()


# ── ポジション取得 ──────────────────────

def get_positions(symbol: str | None = None) -> list[dict]:
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return []

    result = []
    for p in positions:
        result.append({
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume": p.volume,
            "price_open": p.price_open,
            "price_current": p.price_current,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit,
            "time": datetime.fromtimestamp(p.time),
        })
    return result


def get_all_open_symbols() -> list[str]:
    positions = get_positions()
    return list({p["symbol"] for p in positions})


# ── 注文執行 ────────────────────────────

def place_order(symbol: str, direction: str, lot: float,
                sl: float, tp: float | None = None) -> int | None:
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        logger.error("シンボル情報なし: %s", symbol)
        return None

    price = sym_info.ask if direction == "BUY" else sym_info.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "deviation": 20,
        "magic": 202604,
        "comment": "AI_Trader",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if tp is not None:
        request["tp"] = tp

    result = mt5.order_send(request)
    if result is None:
        logger.error("注文送信失敗: result=None")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            "注文失敗: %s retcode=%s comment=%s",
            symbol, result.retcode, result.comment,
        )
        discord_notifier.send_error(
            f"注文失敗: {symbol}",
            f"retcode={result.retcode} comment={result.comment}",
        )
        return None

    logger.info(
        "注文成功: %s %s lot=%.2f price=%.5f ticket=%s",
        symbol, direction, lot, result.price, result.order,
    )
    return result.order


def close_position(ticket: int) -> bool:
    position = mt5.positions_get(ticket=ticket)
    if not position:
        logger.warning("ポジション未検出: ticket=%s", ticket)
        return False

    pos = position[0]
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": 202604,
        "comment": "AI_Trader_Exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.comment if result else "None"
        logger.error("決済失敗: ticket=%s err=%s", ticket, err)
        discord_notifier.send_error("決済失敗", f"ticket={ticket} err={err}")
        return False

    logger.info("決済成功: ticket=%s", ticket)
    return True


def modify_position_sl(ticket: int, new_sl: float) -> bool:
    position = mt5.positions_get(ticket=ticket)
    if not position:
        logger.warning("SL更新対象ポジション未検出: ticket=%s", ticket)
        return False

    pos = position[0]
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": pos.symbol,
        "sl": new_sl,
        "tp": pos.tp,
        "magic": 202604,
        "comment": "AI_Trader_SL_Update",
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.comment if result else "None"
        logger.error("SL更新失敗: ticket=%s new_sl=%.5f err=%s", ticket, new_sl, err)
        return False

    logger.info("SL更新成功: ticket=%s new_sl=%.5f", ticket, new_sl)
    return True


# ── USDJPY レート取得 (通貨換算用) ──────

def get_usdjpy_rate() -> float:
    tick = mt5.symbol_info_tick("USDJPY")
    if tick is None:
        logger.warning("USDJPYレート取得失敗、デフォルト150.0を使用")
        return 150.0
    return (tick.bid + tick.ask) / 2.0


def get_conversion_rate_to_jpy(currency: str) -> float:
    if currency == "JPY":
        return 1.0
    if currency == "USD":
        return get_usdjpy_rate()

    # EUR, GBP 等 → xxxJPY で変換
    pair = currency + "JPY"
    tick = mt5.symbol_info_tick(pair)
    if tick is not None:
        return (tick.bid + tick.ask) / 2.0

    # xxxJPY が無い場合: xxxUSD → USDJPY で間接変換
    pair_usd = currency + "USD"
    tick_usd = mt5.symbol_info_tick(pair_usd)
    if tick_usd is not None:
        return ((tick_usd.bid + tick_usd.ask) / 2.0) * get_usdjpy_rate()

    logger.warning("%s→JPY変換レート取得失敗、1.0で継続", currency)
    return 1.0
