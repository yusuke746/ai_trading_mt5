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
