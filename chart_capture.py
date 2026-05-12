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

# ── SMCゾーン描画カラー定数 (TradingView準拠) ──────────────────────
_SMC_BULL_COLOR = (0.149, 0.651, 0.604)   # #26a69a Emerald Green (Bullish)
_SMC_BEAR_COLOR = (0.937, 0.325, 0.314)   # #ef5350 Soft Red       (Bearish)
_SMC_FVG_COLOR  = (0.380, 0.337, 0.784)   # #6157c8 Purple         (Fair Value Gap)


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
    from matplotlib.lines import Line2D

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
    current_close = float(ohlc["Close"].iloc[-1])

    # Y軸描画範囲: OHLCローソク足実体 ± 30% (遠距離hlineによるY軸圧縮を防ぐ)
    _ohlc_hi = float(ohlc["High"].max())
    _ohlc_lo = float(ohlc["Low"].min())
    _price_span = max(_ohlc_hi - _ohlc_lo, 1e-10)
    _draw_hi = _ohlc_hi + _price_span * 0.30
    _draw_lo = _ohlc_lo - _price_span * 0.30

    def _pick_near_levels(levels: list, max_count: int) -> list[float]:
        """現在価格に近いレベルを優先して上位N本だけ返す。"""
        cleaned: list[float] = []
        seen: set[float] = set()
        for lv in levels or []:
            try:
                f = float(lv)
            except (TypeError, ValueError):
                continue
            key = round(f, 6)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(f)
        cleaned.sort(key=lambda x: abs(x - current_close))
        return cleaned[:max_count]

    # ── addplot リスト構築 ──────────────────────
    add_plots = []

    # MA20
    ma_series = ohlc["Close"].rolling(config.MA_PERIOD).mean()
    add_plots.append(mpf.make_addplot(ma_series, color="black", width=1.2, label="MA20"))

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
    # 主要ラインの右端ラベル用 (price, label, color)
    line_labels: list[tuple[float, str, str]] = []

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
    for lvl in _pick_near_levels(
        smc.get("buy_liquidity", []),
        config.SMC_DRAW_MAX_LIQUIDITY_PER_SIDE,
    ):
        hlines_prices.append(float(lvl))
        hlines_colors.append("deepskyblue")
        hlines_styles.append("dotted")
        hlines_widths.append(1.0)

    for lvl in _pick_near_levels(
        smc.get("sell_liquidity", []),
        config.SMC_DRAW_MAX_LIQUIDITY_PER_SIDE,
    ):
        hlines_prices.append(float(lvl))
        hlines_colors.append("firebrick")
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
            line_labels.append((float(val), key.upper(), color))

    # スウィング高値/安値: 紫の点線
    for lvl in _pick_near_levels(
        smc.get("swing_highs", []),
        config.SMC_DRAW_MAX_SWING_PER_SIDE,
    ):
        hlines_prices.append(float(lvl))
        hlines_colors.append("mediumpurple")
        hlines_styles.append("dotted")
        hlines_widths.append(0.9)

    for lvl in _pick_near_levels(
        smc.get("swing_lows", []),
        config.SMC_DRAW_MAX_SWING_PER_SIDE,
    ):
        hlines_prices.append(float(lvl))
        hlines_colors.append("mediumpurple")
        hlines_styles.append("dotted")
        hlines_widths.append(0.9)

    # Equal Highs / Equal Lows: 複数スウィングが集まった高密度流動性クラスター
    for lvl in smc.get("eq_highs", []):
        try:
            hlines_prices.append(float(lvl))
            hlines_colors.append("#FFD700")   # Gold
            hlines_styles.append("dashed")
            hlines_widths.append(1.8)
            line_labels.append((float(lvl), "EQH", "#FFD700"))
        except (TypeError, ValueError):
            pass

    for lvl in smc.get("eq_lows", []):
        try:
            hlines_prices.append(float(lvl))
            hlines_colors.append("#00CED1")   # Dark Turquoise
            hlines_styles.append("dashed")
            hlines_widths.append(1.8)
            line_labels.append((float(lvl), "EQL", "#00CED1"))
        except (TypeError, ValueError):
            pass

    # 無効化ライン (エグジット用): 太い赤実線
    if invalidation_price is not None:
        hlines_prices.append(float(invalidation_price))
        hlines_colors.append("crimson")
        hlines_styles.append("solid")
        hlines_widths.append(2.5)
        line_labels.append((float(invalidation_price), "INV", "crimson"))

    # ── 描画範囲外の hline を除去: 遠距離水平線によるY軸圧縮を防ぐ ──
    if hlines_prices:
        _kept = [
            (p, c, s, w)
            for p, c, s, w in zip(hlines_prices, hlines_colors, hlines_styles, hlines_widths)
            if _draw_lo <= p <= _draw_hi
        ]
        if _kept:
            hlines_prices[:], hlines_colors[:], hlines_styles[:], hlines_widths[:] = (
                list(col) for col in zip(*_kept)
            )
        else:
            hlines_prices.clear(); hlines_colors.clear()
            hlines_styles.clear(); hlines_widths.clear()
    line_labels = [(p, lbl, c) for p, lbl, c in line_labels if _draw_lo <= p <= _draw_hi]

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
    title_suffix = f"  INV={invalidation_price}" if invalidation_price is not None else ""
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

    ax_main = axes[0]

    # グリッド: 最小限 (ローソク足を主役に)
    for ax in axes:
        ax.grid(True, alpha=0.1, linewidth=0.4, linestyle="-", color="gray")

    # ラベル右余白: PDH/PDL等テキストが y軸と重ならないよう確保
    _label_margin = 5  # バー5本分
    ax_main.set_xlim(right=len(ohlc) - 1 + _label_margin)
    # Y軸を明示固定 (遠距離hline描画後でもローソク足が画面に収まるよう)
    ax_main.set_ylim(_draw_lo, _draw_hi)

    # ── OBゾーン・FVGゾーンをRectangle Boxで描画 ──
    x_indices = np.arange(len(ohlc))  # 互換性のため保持

    ob_zones: list[dict] = smc.get("ob_zones", [])
    fvg_zones: list[dict] = smc.get("fvg_zones", [])

    box_width = len(ohlc) - 1

    for zone in ob_zones:
        try:
            hi = float(zone["high"])
            lo = float(zone["low"])
            if not (_draw_lo <= lo <= _draw_hi or _draw_lo <= hi <= _draw_hi):
                continue  # 描画範囲外のゾーンはスキップ
            zone_type = str(zone.get("type", "bull")).lower()
            color = _SMC_BULL_COLOR if zone_type == "bull" else _SMC_BEAR_COLOR
            # Mitigation: 終値が完全に貫通したゾーンは透明度を落として「使用済み」表示
            mitigated = (current_close < lo) if zone_type == "bull" else (current_close > hi)
            face_alpha = 0.05 if mitigated else 0.18
            edge_alpha = 0.25 if mitigated else 0.75
            rect = mpatches.Rectangle(
                xy=(0, lo),
                width=box_width,
                height=hi - lo,
                linewidth=0.8,
                edgecolor=(*color, edge_alpha),
                facecolor=(*color, face_alpha),
                zorder=2,
            )
            ax_main.add_patch(rect)
            # AIが種別を認識できるようにゾーン右端にラベルを添付
            label_text = ("Bull OB" if zone_type == "bull" else "Bear OB") + (" [M]" if mitigated else "")
            ax_main.text(
                box_width * 0.98,
                (hi + lo) / 2,
                label_text,
                color=(*color, 0.9),
                fontsize=6,
                va="center",
                ha="right",
                fontweight="bold",
                zorder=11,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("OBゾーン描画スキップ: %s", e)

    for zone in fvg_zones:
        try:
            hi = float(zone["high"])
            lo = float(zone["low"])
            if not (_draw_lo <= lo <= _draw_hi or _draw_lo <= hi <= _draw_hi):
                continue  # 描画範囲外のゾーンはスキップ
            # FVG Mitigation: 価格がゾーン内に入っている = 充填中
            fvg_mitigated = lo <= current_close <= hi
            face_alpha = 0.05 if fvg_mitigated else 0.11
            edge_alpha = 0.30 if fvg_mitigated else 0.55
            rect = mpatches.Rectangle(
                xy=(0, lo),
                width=box_width,
                height=hi - lo,
                linewidth=0.6,
                edgecolor=(*_SMC_FVG_COLOR, edge_alpha),
                facecolor=(*_SMC_FVG_COLOR, face_alpha),
                zorder=2,
            )
            ax_main.add_patch(rect)
            # AIが種別を認識できるようにゾーン右端にラベルを添付
            fvg_label = "FVG [M]" if fvg_mitigated else "FVG"
            ax_main.text(
                box_width * 0.98,
                (hi + lo) / 2,
                fvg_label,
                color=(*_SMC_FVG_COLOR, 0.9),
                fontsize=6,
                va="center",
                ha="right",
                fontweight="bold",
                zorder=11,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("FVGゾーン描画スキップ: %s", e)

    # ── 主要ライン右端ラベル (PDH/PDL/PWH/PWL/INV) ──
    x_right = len(ohlc) - 1
    for price, label, color in line_labels:
        ax_main.text(
            x_right + 0.6,
            price,
            label,
            color=color,
            fontsize=7,
            va="center",
            ha="left",
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.55, boxstyle="round,pad=0.15"),
            zorder=10,
        )

    # ── 凡例: 枠なし・半透明背景 ──
    legend_handles = [
        Line2D([0], [0], color="dodgerblue", lw=1.2, ls="solid", label="BOS"),
        Line2D([0], [0], color="black", lw=1.2, ls="solid", label="MA20"),
        Line2D([0], [0], color="darkorange", lw=1.2, ls="dashed", label="CHoCH"),
        Line2D([0], [0], color="deepskyblue", lw=1.0, ls="dotted", label="Liquidity (Buy-side)"),
        Line2D([0], [0], color="firebrick", lw=1.0, ls="dotted", label="Liquidity (Sell-side)"),
        mpatches.Patch(facecolor=_SMC_BULL_COLOR, alpha=0.20, label="OB (Bullish)"),
        mpatches.Patch(facecolor=_SMC_BEAR_COLOR, alpha=0.20, label="OB (Bearish)"),
        mpatches.Patch(facecolor=_SMC_FVG_COLOR, alpha=0.18, label="FVG"),
        Line2D([0], [0], color="gold", lw=1.0, ls="dashed", label="PDH/PDL"),
        Line2D([0], [0], color="orchid", lw=1.0, ls="dashed", label="PWH/PWL"),
        Line2D([0], [0], color="#FFD700", lw=1.8, ls="dashed", label="Equal Highs (EQH)"),
        Line2D([0], [0], color="#00CED1", lw=1.8, ls="dashed", label="Equal Lows (EQL)"),
    ]
    if invalidation_price is not None:
        legend_handles.append(
            Line2D([0], [0], color="crimson", lw=2.5, ls="solid", label="Invalidation")
        )
    ax_main.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=6,
        ncol=2,
        framealpha=0.55,
        facecolor="white",
        edgecolor="none",
        borderpad=0.5,
    )

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
