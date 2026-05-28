"""MT5 接続・データ取得・注文管理モジュール"""

import time
import logging
from datetime import datetime, timezone

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


def _symbol_candidates(base_symbol: str) -> list[str]:
    """ブローカー依存のサフィックス差分(#, .)を吸収した候補を返す。"""
    base = (base_symbol or "").strip()
    if not base:
        return []

    candidates: list[str] = [base]
    stripped = base.rstrip("#.")
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    for suffix in ("#", "."):
        sym = stripped + suffix
        if stripped and sym not in candidates:
            candidates.append(sym)

    # 監視銘柄にも同一ベース名がいれば候補に含める
    for watched in config.SYMBOLS:
        if watched.rstrip("#.") == stripped and watched not in candidates:
            candidates.append(watched)

    return candidates


def _get_tick_with_fallback(base_symbol: str):
    """シンボル候補を順に試し、最初に取得できたtickを返す。"""
    for candidate in _symbol_candidates(base_symbol):
        mt5.symbol_select(candidate, True)
        tick = mt5.symbol_info_tick(candidate)
        if tick is not None:
            return tick, candidate
    return None, None


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
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df


def get_current_price(symbol: str) -> dict | None:
    tick, resolved = _get_tick_with_fallback(symbol)
    if tick is None:
        return None
    if resolved != symbol:
        logger.warning("価格取得でシンボル補正: %s -> %s", symbol, resolved)
    return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}


def is_symbol_market_active(symbol: str, stale_seconds: int | None = None) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning("市場状態確認失敗: symbol_infoなし %s", symbol)
        return False

    if not info.visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.info("[Market] %s: tick取得不可 → クローズ扱い", symbol)
        return False

    max_age = stale_seconds if stale_seconds is not None else config.MARKET_DATA_STALE_SEC
    tick_age = time.time() - tick.time
    if tick_age > max_age:
        logger.info(
            "[Market] %s: tickが古いためクローズ扱い age=%.0fs threshold=%ss",
            symbol, tick_age, max_age,
        )
        return False

    if tick.bid <= 0 and tick.ask <= 0:
        logger.info("[Market] %s: bid/ask不正のためクローズ扱い", symbol)
        return False

    return True


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


def calculate_atr_sma(df: pd.DataFrame, atr_period: int = 14, sma_period: int = 50) -> float | None:
    """ATR(atr_period)のSMA(sma_period)を返す。データ不足時はNone。"""
    needed = atr_period + sma_period
    if len(df) < needed:
        return None
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    # 各インデックスのATR (単純移動平均)
    n = len(tr)
    atr_arr = np.full(n, np.nan)
    for i in range(atr_period - 1, n):
        atr_arr[i] = np.mean(tr[i - atr_period + 1: i + 1])
    valid = atr_arr[~np.isnan(atr_arr)]
    if len(valid) < sma_period:
        return None
    return float(np.mean(valid[-sma_period:]))


def calculate_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["close"].rolling(window=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float | None:
    """Wilder's ADX (Average Directional Index) を計算して最新値を返す。
    データ不足時は None。
    """
    needed = period * 3 + 5
    if len(df) < needed:
        return None

    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    n = len(high)

    tr  = np.zeros(n)
    pdm = np.zeros(n)
    ndm = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i]  - close[i - 1])
        tr[i]  = max(hl, hc, lc)
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        ndm[i] = dn if (dn > up and dn > 0) else 0.0

    # Wilder's 初期スムージング値 (最初の period 本の合計)
    atr_s = float(np.sum(tr[1: period + 1]))
    pdm_s = float(np.sum(pdm[1: period + 1]))
    ndm_s = float(np.sum(ndm[1: period + 1]))

    dx_vals: list[float] = []
    for i in range(period + 1, n):
        atr_s = atr_s - atr_s / period + tr[i]
        pdm_s = pdm_s - pdm_s / period + pdm[i]
        ndm_s = ndm_s - ndm_s / period + ndm[i]
        if atr_s <= 0:
            dx_vals.append(0.0)
            continue
        pdi = 100.0 * pdm_s / atr_s
        ndi = 100.0 * ndm_s / atr_s
        denom = pdi + ndi
        dx_vals.append(100.0 * abs(pdi - ndi) / denom if denom > 0 else 0.0)

    if len(dx_vals) < period:
        return None

    # ADX初期値: 最初の period 本の DX の平均
    adx_val = float(np.mean(dx_vals[:period]))
    for dxi in dx_vals[period:]:
        adx_val = (adx_val * (period - 1) + dxi) / period

    return float(adx_val) if adx_val > 0 else None


# ── SMC 価格レベル計算 ──────────────────

def _detect_swings(df: pd.DataFrame, window: int = 3) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """ピボット高値/安値を返す。戻り値は (index, price)。"""
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    if df is None or len(df) < window * 2 + 3:
        return highs, lows

    end = len(df) - window - 1  # 最後の1本は未確定足として除外
    for i in range(window, end):
        h = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        if h == float(df["high"].iloc[i - window: i + window + 1].max()):
            highs.append((i, h))
        if lo == float(df["low"].iloc[i - window: i + window + 1].min()):
            lows.append((i, lo))
    return highs, lows


def _detect_equal_levels(
    swings: list[tuple[int, float]],
    digits: int,
    tolerance: float,
    min_count: int = 2,
) -> list[float]:
    """Equal High / Equal Low レベルを検出する。

    tolerance 以内に min_count 個以上のスウィングが集まった価格帯を返す。
    単発のヒゲ先ではなく、複数回試された籠密性の高い流動性プールのみを検出する。
    """
    if len(swings) < min_count:
        return []
    prices = [v for _, v in swings]
    eq_levels: list[float] = []
    used = [False] * len(prices)
    for i in range(len(prices)):
        if used[i]:
            continue
        cluster = [prices[i]]
        for j in range(i + 1, len(prices)):
            if not used[j] and abs(prices[j] - prices[i]) <= tolerance:
                cluster.append(prices[j])
                used[j] = True
        if len(cluster) >= min_count:
            level = round(sum(cluster) / len(cluster), digits)
            eq_levels.append(level)
            used[i] = True
    return eq_levels


def _detect_fvg_zones(df: pd.DataFrame, digits: int, lookback: int = 180, max_zones: int = 3) -> list[dict]:
    """3本足でFVGを検出して返す（軽量版）。"""
    zones: list[dict] = []
    if df is None or len(df) < 6:
        return zones

    subset = df.tail(lookback)
    end = len(subset) - 1  # 最後の足は未確定扱い
    for i in range(2, end):
        c0_hi = float(subset["high"].iloc[i - 2])
        c0_lo = float(subset["low"].iloc[i - 2])
        c2_hi = float(subset["high"].iloc[i])
        c2_lo = float(subset["low"].iloc[i])
        bar_offset = len(subset) - 1 - i  # 末尾（最新足）からの距離（バー数）

        if c0_hi < c2_lo:  # Bullish FVG
            zones.append({
                "low": round(c0_hi, digits),
                "high": round(c2_lo, digits),
                "type": "bull",
                "bar_offset": bar_offset,
            })
        if c0_lo > c2_hi:  # Bearish FVG
            zones.append({
                "low": round(c2_hi, digits),
                "high": round(c0_lo, digits),
                "type": "bear",
                "bar_offset": bar_offset,
            })

    # bull/bear 別に最新 max_zones 件ずつ返す（片側トレンドでも両側が表示されるよう）
    bull_fvg = [z for z in zones if z["type"] == "bull"]
    bear_fvg = [z for z in zones if z["type"] == "bear"]
    return bull_fvg[-max_zones:] + bear_fvg[-max_zones:]


def _detect_ob_zones(df: pd.DataFrame, digits: int, atr: float, lookback: int = 180, max_zones: int = 3) -> list[dict]:
    """TradingView準拠のOB検出。

    BOSを引き起こした変位直前の逆色ローソク足をOrder Blockとして特定する。
    - Bull OB : スウィング高値をブレイクしたBOSバーの直前にある最後の陰線 (close < open)
    - Bear OB : スウィング安値をブレイクしたBOSバーの直前にある最後の陽線 (close > open)
    - Mitigated: Bull OBはlowを下抜け、Bear OBはhighを上抜けした場合に除外
      (ゾーン内タッチはエントリー機会のため除外しない)
    """
    zones: list[dict] = []
    if df is None or len(df) < 10:
        return zones

    subset = df.tail(lookback).reset_index(drop=True)
    n = len(subset)
    window = 3  # スウィング検出ウィンドウ

    # スウィング高値/安値を検出
    swing_highs: list[tuple[int, float]] = []
    swing_lows:  list[tuple[int, float]] = []
    for i in range(window, n - window - 1):  # 末尾 window+1 本は未確定足として除外
        h = float(subset["high"].iloc[i])
        l = float(subset["low"].iloc[i])
        if h == float(subset["high"].iloc[i - window: i + window + 1].max()):
            swing_highs.append((i, h))
        if l == float(subset["low"].iloc[i - window: i + window + 1].min()):
            swing_lows.append((i, l))

    seen_ob_bars: set[int] = set()  # 同一バーの重複登録を防止

    # ── Bull OB: スウィング高値BOS → 直前の最後の陰線 ──
    for sw_idx, sw_price in swing_highs:
        for j in range(sw_idx + 1, n - 1):  # n-1: 最終足は未確定
            if float(subset["close"].iloc[j]) > sw_price:
                # j がBOSバー。sw_idx〜j-1 の間で最後の陰線を逆順探索
                ob_bar = None
                for k in range(j - 1, sw_idx - 1, -1):
                    if float(subset["close"].iloc[k]) < float(subset["open"].iloc[k]):
                        ob_bar = k
                        break
                if ob_bar is None or ob_bar in seen_ob_bars:
                    break
                hi = float(subset["high"].iloc[ob_bar])
                lo = float(subset["low"].iloc[ob_bar])
                # Mitigated: OB形成後に low を下抜けたか (タッチは除外しない)
                mitigated = any(
                    float(subset["low"].iloc[m]) < lo
                    for m in range(ob_bar + 1, n)
                )
                if not mitigated:
                    seen_ob_bars.add(ob_bar)
                    zones.append({
                        "low": round(lo, digits),
                        "high": round(hi, digits),
                        "type": "bull",
                        "bar_offset": n - 1 - ob_bar,
                    })
                break  # このスウィング高値に対するBOSは1件だけ処理

    # ── Bear OB: スウィング安値BOS → 直前の最後の陽線 ──
    for sw_idx, sw_price in swing_lows:
        for j in range(sw_idx + 1, n - 1):
            if float(subset["close"].iloc[j]) < sw_price:
                # j がBOSバー。直前の最後の陽線を逆順探索
                ob_bar = None
                for k in range(j - 1, sw_idx - 1, -1):
                    if float(subset["close"].iloc[k]) > float(subset["open"].iloc[k]):
                        ob_bar = k
                        break
                if ob_bar is None or ob_bar in seen_ob_bars:
                    break
                hi = float(subset["high"].iloc[ob_bar])
                lo = float(subset["low"].iloc[ob_bar])
                # Mitigated: OB形成後に high を上抜けたか (タッチは除外しない)
                mitigated = any(
                    float(subset["high"].iloc[m]) > hi
                    for m in range(ob_bar + 1, n)
                )
                if not mitigated:
                    seen_ob_bars.add(ob_bar)
                    zones.append({
                        "low": round(lo, digits),
                        "high": round(hi, digits),
                        "type": "bear",
                        "bar_offset": n - 1 - ob_bar,
                    })
                break  # このスウィング安値に対するBOSは1件だけ処理

    # bull/bear 別に最新 max_zones 件ずつ返す (片側トレンド時に一方が全滅しないよう)
    bull_zones = [z for z in zones if z["type"] == "bull"]
    bear_zones = [z for z in zones if z["type"] == "bear"]
    return bull_zones[-max_zones:] + bear_zones[-max_zones:]

def get_price_levels(symbol: str, digits: int = 5) -> dict:
    """SMC分析用の価格レベルを計算して返す。

    Returns:
        dict with:
          pdh/pdl  : 前日高値/安値 (D1の1本前確定足)
          pwh/pwl  : 前週高値/安値 (直近5日足の高安値)
          swing_highs : H1直近スウィング高値リスト (最大5個)
          swing_lows  : H1直近スウィング安値リスト (最大5個)
                    buy_liquidity / sell_liquidity : 流動性プール候補
                    bos_levels / choch_levels : 構造ブレイク候補レベル
                    ob_zones / fvg_zones : M15ベースのゾーン候補
    """
    result: dict = {
        "pdh": None, "pdl": None,
        "pwh": None, "pwl": None,
        "swing_highs": [],
        "swing_lows": [],
                "buy_liquidity": [],
                "sell_liquidity": [],
                "bos_levels": [],
                "choch_levels": [],
                "ob_zones": [],
                "fvg_zones": [],
                "eq_highs": [],
                "eq_lows": [],
                "h1_ob_zones": [],
                "m15_choch_levels": [],
    }

    # PDH / PDL : D1の1本前確定足
    df_d1 = get_rates(symbol, "D1", 10)
    if df_d1 is not None and len(df_d1) >= 2:
        prev = df_d1.iloc[-2]
        result["pdh"] = round(float(prev["high"]), digits)
        result["pdl"] = round(float(prev["low"]), digits)

    # PWH / PWL : 直近5本のD1足 (当日除く)
    if df_d1 is not None and len(df_d1) >= 6:
        week_slice = df_d1.iloc[-6:-1]
        result["pwh"] = round(float(week_slice["high"].max()), digits)
        result["pwl"] = round(float(week_slice["low"].min()), digits)

    # スウィング高値/安値 : H1で±3本窓のピボット検出 (確定足のみ)
    df_h1 = get_rates(symbol, "H1", 180)
    if df_h1 is not None and len(df_h1) >= 10:
        swings_h, swings_l = _detect_swings(df_h1, window=3)
        highs = [round(v, digits) for _, v in swings_h]
        lows = [round(v, digits) for _, v in swings_l]
        result["swing_highs"] = highs[-5:]
        result["swing_lows"] = lows[-5:]
        result["buy_liquidity"] = highs[-4:]
        result["sell_liquidity"] = lows[-4:]

        # EQH / EQL: H1 ATR ベースの許容誤差内に複数スウィングが密集
        atr_h1 = calculate_atr(df_h1, config.ATR_PERIOD)
        eq_tol = atr_h1 * 0.2  # ATRの20%以内を「等値」とみなす
        result["eq_highs"] = _detect_equal_levels(swings_h, digits, eq_tol)
        result["eq_lows"]  = _detect_equal_levels(swings_l, digits, eq_tol)

        # BOS / CHoCH の簡易判定（H1確定足ベース）
        if len(swings_h) >= 2 and len(swings_l) >= 2 and len(df_h1) >= 3:
            prev_high = swings_h[-2][1]
            last_high = swings_h[-1][1]
            prev_low = swings_l[-2][1]
            last_low = swings_l[-1][1]
            confirmed_close = float(df_h1["close"].iloc[-2])

            trend = "RANGE"
            if last_high > prev_high and last_low > prev_low:
                trend = "UP"
            elif last_high < prev_high and last_low < prev_low:
                trend = "DOWN"

            if trend == "UP":
                if confirmed_close > last_high:
                    result["bos_levels"] = [round(last_high, digits)]
                if confirmed_close < last_low:
                    result["choch_levels"] = [round(last_low, digits)]
            elif trend == "DOWN":
                if confirmed_close < last_low:
                    result["bos_levels"] = [round(last_low, digits)]
                if confirmed_close > last_high:
                    result["choch_levels"] = [round(last_high, digits)]

        # H1ベースのOBゾーン（コンフルエンス検出用）
        result["h1_ob_zones"] = _detect_ob_zones(df_h1, digits=digits, atr=atr_h1, lookback=100, max_zones=3)

    # M15ベースのOB/FVGゾーン
    df_m15 = get_rates(symbol, "M15", 260)
    if df_m15 is not None and len(df_m15) >= 20:
        atr_m15 = calculate_atr(df_m15, config.ATR_PERIOD)
        result["ob_zones"] = _detect_ob_zones(df_m15, digits=digits, atr=atr_m15)
        result["fvg_zones"] = _detect_fvg_zones(df_m15, digits=digits)

        # M15ベースのCHoCH判定 (ウィンドウ=2本、小さい局面構造の変化を捕捉)
        m15_swings_h, m15_swings_l = _detect_swings(df_m15, window=2)
        if len(m15_swings_h) >= 2 and len(m15_swings_l) >= 2:
            m15_prev_high = m15_swings_h[-2][1]
            m15_last_high = m15_swings_h[-1][1]
            m15_prev_low  = m15_swings_l[-2][1]
            m15_last_low  = m15_swings_l[-1][1]
            m15_confirmed = float(df_m15["close"].iloc[-2])

            m15_trend = "RANGE"
            if m15_last_high > m15_prev_high and m15_last_low > m15_prev_low:
                m15_trend = "UP"
            elif m15_last_high < m15_prev_high and m15_last_low < m15_prev_low:
                m15_trend = "DOWN"

            if m15_trend == "UP" and m15_confirmed < m15_last_low:
                result["m15_choch_levels"] = [round(m15_last_low, digits)]
            elif m15_trend == "DOWN" and m15_confirmed > m15_last_high:
                result["m15_choch_levels"] = [round(m15_last_high, digits)]

    return result


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
            "time": datetime.utcfromtimestamp(p.time),
        })
    return result


def get_all_open_symbols() -> list[str]:
    positions = get_positions()
    return list({p["symbol"] for p in positions})


def get_closed_deal_by_ticket(ticket: int) -> dict | None:
    """MT5の約定履歴からポジション(ticket)の決済deal情報を返す。
    TP/SL自動決済を含む全パターンに対応。
    Returns:
        {exit_price, profit, closed_at, exit_reason} or None
    """
    from datetime import timezone, timedelta
    # 過去180日分のdealを検索
    date_from = datetime.utcnow() - timedelta(days=180)
    date_to   = datetime.utcnow() + timedelta(days=1)
    deals = mt5.history_deals_get(
        date_from.replace(tzinfo=timezone.utc),
        date_to.replace(tzinfo=timezone.utc),
        position=ticket,
    )
    if not deals:
        return None

    # entry=DEAL_ENTRY_OUT (1) or DEAL_ENTRY_INOUT (2) が決済deal。
    # 取得結果に別ポジションのdealが混ざるケースがあるため、position_idで厳密に絞る。
    close_candidates = [
        d for d in deals
        if getattr(d, "position_id", None) == ticket
        and d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)
    ]

    if close_candidates:
        # 同一ポジション内で最も新しい決済dealを採用する。
        close_deal = max(
            close_candidates,
            key=lambda d: (getattr(d, "time_msc", 0), getattr(d, "time", 0)),
        )
    else:
        close_deal = None

    if close_deal is None:
        return None

    # 決済理由のマッピング
    _REASON_MAP = {
        mt5.DEAL_REASON_TP:     "TP_HIT",
        mt5.DEAL_REASON_SL:     "SL_HIT",
        mt5.DEAL_REASON_CLIENT: "MANUAL",
        mt5.DEAL_REASON_EXPERT: "EA",
        mt5.DEAL_REASON_MOBILE: "MOBILE",
        mt5.DEAL_REASON_WEB:    "WEB",
        mt5.DEAL_REASON_SO:     "STOP_OUT",
    }
    exit_reason = _REASON_MAP.get(close_deal.reason, f"MT5_REASON_{close_deal.reason}")

    # time_msc はミリ秒 Unix タイムスタンプ（UTC epoch）。
    # utcfromtimestamp はナイーブUTCを返すが、念のためtzinfo付きで統一する。
    _ts_sec = getattr(close_deal, "time_msc", None)
    if _ts_sec:
        closed_at = datetime.fromtimestamp(_ts_sec / 1000, tz=timezone.utc).replace(tzinfo=None).isoformat()
    else:
        closed_at = datetime.utcfromtimestamp(close_deal.time).isoformat()
    return {
        "exit_price": close_deal.price,
        "profit":     close_deal.profit,
        "closed_at":  closed_at,
        "exit_reason": exit_reason,
        "symbol":     close_deal.symbol,
    }


def _send_order_with_filling_fallback(request: dict, symbol: str, context: str):
    """IOC→FOK→RETURNの順でfillingモードをフォールバックする。"""
    _FILLING_MODES = [
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_RETURN,
    ]
    _MODE_NAMES = {mt5.ORDER_FILLING_IOC: "IOC", mt5.ORDER_FILLING_FOK: "FOK", mt5.ORDER_FILLING_RETURN: "RETURN"}
    last_result = None
    for mode in _FILLING_MODES:
        attempt = dict(request)
        attempt["type_filling"] = mode
        result = mt5.order_send(attempt)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            mode_name = _MODE_NAMES.get(mode, str(mode))
            if mode != mt5.ORDER_FILLING_IOC:
                logger.info("%s: %sで注文成功: %s", context, mode_name, symbol)
            return result
        retcode = result.retcode if result is not None else "None"
        logger.debug("%s: %s filling=%s -> retcode=%s", context, symbol, _MODE_NAMES.get(mode, str(mode)), retcode)
        last_result = result if result is not None else last_result
    return last_result


# ── 注文執行 ────────────────────────────

def place_order(symbol: str, direction: str, lot: float,
                sl: float, tp: float | None = None) -> int | None:
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        logger.error("シンボル情報なし: %s", symbol)
        return None

    # trade_mode: 0=disabled 1=long_only 2=short_only 3=close_only 4=full
    _TRADE_MODE_NAMES = {0: "DISABLED", 1: "LONG_ONLY", 2: "SHORT_ONLY", 3: "CLOSE_ONLY", 4: "FULL"}
    trade_mode = getattr(sym_info, "trade_mode", 4)
    if trade_mode != 4:
        mode_name = _TRADE_MODE_NAMES.get(trade_mode, str(trade_mode))
        logger.warning("注文スキップ: %s trade_mode=%s(%s) — 市場が新規注文を受け付けていません", symbol, trade_mode, mode_name)
        discord_notifier.send_error(
            f"注文スキップ: {symbol}",
            f"trade_mode={trade_mode}({mode_name}) — 市場クローズ中または新規注文不可",
        )
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

    result = _send_order_with_filling_fallback(request, symbol, "新規注文")
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
        "注文成功: %s %s lot=%.2f price=%.5f order=%s deal=%s",
        symbol, direction, lot, result.price, result.order, result.deal,
    )
    # ヘッジ口座ではポジションticketは通常 order ticket 側。
    # deal ticket を保存すると positions_get(ticket=...) と一致せず、
    # 後続の照合で誤判定しやすいため order を優先する。
    position_ticket = result.order if result.order else result.deal
    return position_ticket


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

    result = _send_order_with_filling_fallback(request, pos.symbol, "決済注文")
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


# ── サーバー時刻・市場開閉判定 ───────────

def get_server_datetime() -> datetime | None:
    """XMTradingサーバー時刻を返す (GMT+3固定、夏冬不変)。
    tick.time は UTC UNIX秒なので +3h でサーバー時刻に変換する。
    """
    from datetime import timezone, timedelta
    server_tz = timezone(timedelta(hours=3))
    symbols_to_probe = list(config.SYMBOLS) + ["EURUSD", "USDJPY", "XAUUSD", "GOLD"]
    for sym in symbols_to_probe:
        tick, _ = _get_tick_with_fallback(sym)
        if tick is not None:
            utc_dt = datetime.utcfromtimestamp(tick.time).replace(tzinfo=timezone.utc)
            return utc_dt.astimezone(server_tz)
    return None


def is_fx_market_open() -> bool:
    """FX市場が開場中かを XMTradingサーバー時刻(GMT+3)で判定する。
    クローズ期間:
      - 土曜 00:00〜日曜 22:59 (サーバー時刻)
      - 金曜 23:59 のクローズ後も含む
    オープン: 日曜 23:00〜金曜 23:59
    """
    srv = get_server_datetime()
    if srv is None:
        logger.warning("[MarketHours] サーバー時刻取得失敗 → 保守的にクローズ扱い")
        return False
    wd = srv.weekday()
    if wd == 5:  # 土曜: 終日クローズ
        return False
    if wd == 6:  # 日曜: 23:00以降にオープン
        return srv.hour >= 23
    if wd == 4:  # 金曜: 23:59でクローズ
        return not (srv.hour == 23 and srv.minute >= 59)
    return True  # 月〜木: 終日オープン


def get_minutes_to_weekend_close() -> float | None:
    """金曜のみ: サーバー時刻で週末クローズまでの残り分を返す。
    金曜以外は None を返す。
    """
    srv = get_server_datetime()
    if srv is None or srv.weekday() != 4:
        return None
    close_time = srv.replace(hour=23, minute=59, second=0, microsecond=0)
    remaining = (close_time - srv).total_seconds() / 60
    return max(0.0, remaining)


# ── USDJPY レート取得 (通貨換算用) ──────

def get_usdjpy_rate() -> float:
    tick, resolved = _get_tick_with_fallback("USDJPY")
    if tick is None:
        # 最終フォールバック: symbols_get からUSDJPY派生を探索
        symbols = mt5.symbols_get("USDJPY*") or []
        for s in symbols:
            mt5.symbol_select(s.name, True)
            tick = mt5.symbol_info_tick(s.name)
            if tick is not None:
                resolved = s.name
                break

    if tick is None:
        logger.warning("USDJPYレート取得失敗(候補全滅)、デフォルト150.0を使用")
        return 150.0

    if resolved and resolved != "USDJPY":
        logger.info("USDJPY換算シンボル補正: USDJPY -> %s", resolved)
    return (tick.bid + tick.ask) / 2.0


def get_conversion_rate_to_jpy(currency: str) -> float:
    if currency == "JPY":
        return 1.0
    if currency == "USD":
        return get_usdjpy_rate()

    # EUR, GBP 等 → xxxJPY で変換
    pair = currency + "JPY"
    tick, _ = _get_tick_with_fallback(pair)
    if tick is not None:
        return (tick.bid + tick.ask) / 2.0

    # xxxJPY が無い場合: xxxUSD → USDJPY で間接変換
    pair_usd = currency + "USD"
    tick_usd, _ = _get_tick_with_fallback(pair_usd)
    if tick_usd is not None:
        return ((tick_usd.bid + tick_usd.ask) / 2.0) * get_usdjpy_rate()

    logger.warning("%s→JPY変換レート取得失敗、1.0で継続", currency)
    return 1.0
