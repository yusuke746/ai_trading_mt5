"""チャート画像生成 & MT5ウィンドウキャプチャ

2つの方法を提供:
  1. generate_chart()   - MT5データからmplfinanceで描画 (AI送信用・メイン)
  2. capture_mt5_window() - MT5ウィンドウ直接キャプチャ (補助・デバッグ)
"""

import io
import base64
import ctypes
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mplfinance as mpf
from PIL import Image

import config
import mt5_connector

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. MT5データからチャート画像を生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_chart(symbol: str, timeframe: str,
                   bars: int = config.CHART_BARS) -> bytes | None:
    """MT5からOHLCデータを取得し、ローソク足+MA20チャート画像をPNGバイト列で返す"""

    df = mt5_connector.get_rates(symbol, timeframe, bars + config.MA_PERIOD)
    if df is None or len(df) < config.MA_PERIOD + 10:
        logger.error("チャート生成失敗: データ不足 %s %s", symbol, timeframe)
        return None

    # mplfinance 用にカラム名を変換
    ohlc = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]]

    # 末尾 bars 本のみ描画
    ohlc = ohlc.tail(bars)

    # MA20 を追加プロットとして定義
    ma = mpf.make_addplot(
        ohlc["Close"].rolling(config.MA_PERIOD).mean(),
        color="blue", width=1.2,
    )

    # ATR をサブプロットに表示
    atr_val = mt5_connector.calculate_atr(df, config.ATR_PERIOD)

    # スタイル設定
    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={"font.size": 8},
    )

    fig_size = (config.CHART_WIDTH / 100, config.CHART_HEIGHT / 100)

    buf = io.BytesIO()
    mpf.plot(
        ohlc,
        type="candle",
        style=style,
        addplot=ma,
        volume=True,
        title=f"{symbol}  {timeframe}   ATR({config.ATR_PERIOD})={atr_val:.5f}",
        figsize=fig_size,
        savefig=dict(fname=buf, dpi=100, bbox_inches="tight"),
    )
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()
    return png_bytes


def generate_chart_pair(symbol: str) -> tuple[bytes | None, bytes | None]:
    """H1 と M15 の2枚のチャート画像を生成して返す"""
    h1_img = generate_chart(symbol, config.TREND_TF)
    m15_img = generate_chart(symbol, config.EXECUTION_TF)
    return h1_img, m15_img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SMCオーバーレイ付きチャート生成 (メモリ上)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_smc_chart_base64(
    symbol: str,
    timeframe: str,
    smc_features: dict | None = None,
    invalidation_price: float | None = None,
    bars: int = config.CHART_BARS,
) -> str | None:
    """SMC特徴量をオーバーレイしたローソク足チャートを生成し、base64文字列で返す。

    Args:
        symbol: MT5銘柄名
        timeframe: タイムフレーム ("H1", "M15" など)
        smc_features: SMC特徴量の辞書。以下のキーをサポート:
            bos_levels      : list[float]  BOS価格レベルリスト
            choch_levels    : list[float]  CHoCH価格レベルリスト
            ob_zones        : list[dict]   OBゾーン {"high": float, "low": float, "type": "bull"|"bear"}
            fvg_zones       : list[dict]   FVGゾーン {"high": float, "low": float}
            buy_liquidity   : list[float]  Buy-side Liquidity価格リスト
            sell_liquidity  : list[float]  Sell-side Liquidity価格リスト
            swing_highs     : list[float]  スウィング高値リスト (get_price_levels互換)
            swing_lows      : list[float]  スウィング安値リスト
            pdh             : float        前日高値
            pdl             : float        前日安値
            pwh             : float        前週高値
            pwl             : float        前週安値
        invalidation_price: エグジット監視用の無効化ライン価格 (赤い太線で描画)
        bars: 表示するバー数

    Returns:
        base64エンコードされたPNG文字列、失敗時はNone
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df = mt5_connector.get_rates(symbol, timeframe, bars + config.MA_PERIOD)
    if df is None or len(df) < config.MA_PERIOD + 10:
        logger.error("SMCチャート生成失敗: データ不足 %s %s", symbol, timeframe)
        return None

    ohlc = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]]
    ohlc = ohlc.tail(bars)

    smc = smc_features or {}

    # ── addplot リスト構築 ──────────────────────
    add_plots = []

    # MA20
    ma_series = ohlc["Close"].rolling(config.MA_PERIOD).mean()
    add_plots.append(mpf.make_addplot(ma_series, color="royalblue", width=1.2, label="MA20"))

    # BOS水平線: 各レベルをパネルに重ねる (mplfinanceのhlines引数で描画)
    bos_levels: list[float] = smc.get("bos_levels", [])
    choch_levels: list[float] = smc.get("choch_levels", [])

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={"font.size": 8},
    )

    fig_size = (config.CHART_WIDTH / 100, config.CHART_HEIGHT / 100)

    # ── hlines (水平線) 引数構築 ──
    hlines_prices: list[float] = []
    hlines_colors: list[str] = []
    hlines_styles: list[str] = []
    hlines_widths: list[float] = []

    # BOS: 青の実線
    for lvl in bos_levels:
        hlines_prices.append(float(lvl))
        hlines_colors.append("dodgerblue")
        hlines_styles.append("solid")
        hlines_widths.append(1.2)

    # CHoCH: オレンジの破線
    for lvl in choch_levels:
        hlines_prices.append(float(lvl))
        hlines_colors.append("darkorange")
        hlines_styles.append("dashed")
        hlines_widths.append(1.2)

    # Buy-side / Sell-side Liquidity: 点線
    for lvl in smc.get("buy_liquidity", []):
        hlines_prices.append(float(lvl))
        hlines_colors.append("limegreen")
        hlines_styles.append("dotted")
        hlines_widths.append(1.0)

    for lvl in smc.get("sell_liquidity", []):
        hlines_prices.append(float(lvl))
        hlines_colors.append("red")
        hlines_styles.append("dotted")
        hlines_widths.append(1.0)

    # PDH/PDL/PWH/PWL (get_price_levels互換)
    for key, color in [("pdh", "gold"), ("pdl", "gold"), ("pwh", "orchid"), ("pwl", "orchid")]:
        val = smc.get(key)
        if val is not None:
            hlines_prices.append(float(val))
            hlines_colors.append(color)
            hlines_styles.append("dashed")
            hlines_widths.append(1.0)

    # スウィング高値/安値: 紫の点線
    for lvl in smc.get("swing_highs", []):
        hlines_prices.append(float(lvl))
        hlines_colors.append("mediumpurple")
        hlines_styles.append("dotted")
        hlines_widths.append(0.9)

    for lvl in smc.get("swing_lows", []):
        hlines_prices.append(float(lvl))
        hlines_colors.append("mediumpurple")
        hlines_styles.append("dotted")
        hlines_widths.append(0.9)

    # 無効化ライン (エグジット用): 太い赤実線
    if invalidation_price is not None:
        hlines_prices.append(float(invalidation_price))
        hlines_colors.append("crimson")
        hlines_styles.append("solid")
        hlines_widths.append(2.5)

    hlines_cfg = (
        {
            "hlines": hlines_prices,
            "colors": hlines_colors,
            "linestyle": hlines_styles,
            "linewidths": hlines_widths,
        }
        if hlines_prices
        else None
    )

    # ── ATR 表示 ──
    atr_val = mt5_connector.calculate_atr(df, config.ATR_PERIOD)
    title_suffix = f"  【無効化ライン: {invalidation_price}】" if invalidation_price is not None else ""
    title = f"{symbol}  {timeframe}   ATR({config.ATR_PERIOD})={atr_val:.5f}{title_suffix}"

    buf = io.BytesIO()
    plot_kwargs = dict(
        type="candle",
        style=style,
        addplot=add_plots,
        volume=True,
        title=title,
        figsize=fig_size,
        returnfig=True,
    )
    if hlines_cfg is not None:
        plot_kwargs["hlines"] = hlines_cfg

    fig, axes = mpf.plot(
        ohlc,
        **plot_kwargs,
    )

    # ── OBゾーン・FVGゾーンをfill_betweenで描画 ──
    ax_main = axes[0]
    x_indices = np.arange(len(ohlc))

    ob_zones: list[dict] = smc.get("ob_zones", [])
    fvg_zones: list[dict] = smc.get("fvg_zones", [])

    for zone in ob_zones:
        try:
            hi = float(zone["high"])
            lo = float(zone["low"])
            zone_type = str(zone.get("type", "bull")).lower()
            color = "rgba(0,200,100,0.15)" if zone_type == "bull" else "rgba(220,50,50,0.15)"
            # matplotlibはrgba文字列不可 → (r,g,b,a) tupleに変換
            if zone_type == "bull":
                fc = (0.0, 0.78, 0.39, 0.15)
                ec = (0.0, 0.78, 0.39, 0.6)
            else:
                fc = (0.86, 0.20, 0.20, 0.15)
                ec = (0.86, 0.20, 0.20, 0.6)
            ax_main.fill_between(x_indices, lo, hi, alpha=0.15,
                                 facecolor=fc[:3], edgecolor=ec[:3], linewidth=0.5)
            ax_main.axhline(y=hi, color=ec[:3], linewidth=0.6, linestyle="--", alpha=0.7)
            ax_main.axhline(y=lo, color=ec[:3], linewidth=0.6, linestyle="--", alpha=0.7)
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("OBゾーン描画スキップ: %s", e)

    for zone in fvg_zones:
        try:
            hi = float(zone["high"])
            lo = float(zone["low"])
            # FVG: シアン系で塗りつぶし
            ax_main.fill_between(x_indices, lo, hi, alpha=0.12,
                                 facecolor=(0.0, 0.75, 0.85), edgecolor=(0.0, 0.6, 0.8),
                                 linewidth=0.5)
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("FVGゾーン描画スキップ: %s", e)

    fig.savefig(buf, dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()

    return base64.standard_b64encode(png_bytes).decode("utf-8")


def generate_smc_chart_pair_base64(
    symbol: str,
    smc_features: dict | None = None,
    invalidation_price: float | None = None,
) -> tuple[str | None, str | None]:
    """H1 と M15 のSMCオーバーレイ付きチャートをbase64で返す。

    エントリー時: invalidation_price=None で両足生成
    エグジット監視時: invalidation_price を指定してM15のみ生成してもよい
    """
    h1_b64 = generate_smc_chart_base64(symbol, config.TREND_TF, smc_features, None)
    m15_b64 = generate_smc_chart_base64(symbol, config.EXECUTION_TF, smc_features, invalidation_price)
    return h1_b64, m15_b64


def chart_to_base64(png_bytes: bytes) -> str:
    return base64.standard_b64encode(png_bytes).decode("utf-8")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MT5ウィンドウ直接キャプチャ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def capture_mt5_window() -> Image.Image | None:
    """MT5ウィンドウをキャプチャしてPIL Imageで返す (Chromeを使わない)"""
    try:
        import win32gui
        import win32ui
        import win32con
    except ImportError:
        logger.error("pywin32が未インストールです")
        return None

    # MT5ウィンドウを検索
    hwnd = _find_mt5_window()
    if not hwnd:
        logger.error("MT5ウィンドウが見つかりません")
        return None

    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            logger.error("MT5ウィンドウサイズが不正: %dx%d", width, height)
            return None

        # デバイスコンテキスト取得
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        # PrintWindow API (PW_RENDERFULLCONTENT = 2)
        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)

        bmp_info = bitmap.GetInfo()
        bmp_bits = bitmap.GetBitmapBits(True)

        img = Image.frombuffer(
            "RGB",
            (bmp_info["bmWidth"], bmp_info["bmHeight"]),
            bmp_bits, "raw", "BGRX", 0, 1,
        )

        # リソース解放
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        return img

    except Exception as e:
        logger.error("MT5ウィンドウキャプチャ失敗: %s", e)
        return None


def save_mt5_screenshot(filename: str | None = None) -> str | None:
    """MT5ウィンドウのスクリーンショットを保存し、ファイルパスを返す"""
    img = capture_mt5_window()
    if img is None:
        return None

    if filename is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mt5_{ts}.png"

    path = Path(config.SCREENSHOT_DIR) / filename
    img.save(str(path))
    logger.info("MT5スクリーンショット保存: %s", path)
    return str(path)


def _find_mt5_window() -> int:
    """MT5のウィンドウハンドルを検索"""
    try:
        import win32gui
    except ImportError:
        return 0

    result = []

    def _enum_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            # XMTrading MT5 or MetaTrader 5 のウィンドウを検索
            if "metatrader" in title.lower() or "mt5" in title.lower():
                result.append(hwnd)

    win32gui.EnumWindows(_enum_callback, None)
    return result[0] if result else 0
