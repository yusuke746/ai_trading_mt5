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
_SMC_BULL_COLOR        = (0.149, 0.651, 0.604)   # #26a69a Emerald Green (Bullish)
_SMC_BEAR_COLOR        = (0.937, 0.325, 0.314)   # #ef5350 Soft Red       (Bearish)
_SMC_FVG_COLOR         = (0.380, 0.337, 0.784)   # #6157c8 Purple         (Fair Value Gap)
_SMC_CONF_BULL_COLOR   = (1.000, 0.843, 0.000)   # #FFD700 Gold           (H1+M15 Confluence Bull)
_SMC_CONF_BEAR_COLOR   = (1.000, 0.500, 0.000)   # #FF8000 Orange         (H1+M15 Confluence Bear)


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
    swept_level: float | None = None,
    swept_type: str | None = None,
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

    # H1チャートにはH1ベースのOBゾーンを使用、M15チャートにはM15ベースを使用
    if timeframe == config.TREND_TF:  # H1
        ob_zones_raw: list[dict] = smc.get("h1_ob_zones", []) or smc.get("ob_zones", [])
    else:
        ob_zones_raw: list[dict] = smc.get("ob_zones", [])

    # Y軸描画範囲: OHLCローソク足実体 ± 10% (遠距離hlineによるY軸圧縮を防ぐ)
    _ohlc_hi = float(ohlc["High"].max())
    _ohlc_lo = float(ohlc["Low"].min())
    _price_span = max(_ohlc_hi - _ohlc_lo, 1e-10)
    _draw_hi = _ohlc_hi + _price_span * 0.10
    _draw_lo = _ohlc_lo - _price_span * 0.10

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
    add_plots.append(mpf.make_addplot(ma_series, color="#b2b5be", width=1.2, label="MA20"))

    # BOS水平線: 各レベルをパネルに重ねる (mplfinanceのhlines引数で描画)
    bos_levels: list[float] = smc.get("bos_levels", [])
    choch_levels: list[float] = smc.get("choch_levels", [])

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
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

    # CHoCH (H1): オレンジの破線
    for lvl in choch_levels:
        hlines_prices.append(float(lvl))
        hlines_colors.append("darkorange")
        hlines_styles.append("dashed")
        hlines_widths.append(1.2)

    # BOS/CHoCH 右端ラベル: TradingView準拠で線上に直接表示
    for lvl in bos_levels:
        try:
            line_labels.append((float(lvl), "BOS", "dodgerblue"))
        except (TypeError, ValueError):
            pass
    for lvl in choch_levels:
        try:
            line_labels.append((float(lvl), "CHoCH", "darkorange"))
        except (TypeError, ValueError):
            pass

    # CHoCH (M15): mplfinanceのhlines validatorはタプル形式linestyleを受け付けないため
    # post-plotのax.axhline()で別途描画する（_m15_choch_draw_levels に収集）
    _m15_choch_draw_levels: list[float] = []
    for lvl in smc.get("m15_choch_levels", []):
        try:
            _m15_choch_draw_levels.append(float(lvl))
        except (TypeError, ValueError):
            pass

    # Buy-side Liquidity: 現在価格より「上」にあるスウィング高値のみ
    # (価格に上抜かれた高値は流動性が消費済みなので描画しない)
    for lvl in _pick_near_levels(
        [l for l in smc.get("buy_liquidity", []) if float(l) > current_close],
        config.SMC_DRAW_MAX_LIQUIDITY_PER_SIDE,
    ):
        hlines_prices.append(float(lvl))
        hlines_colors.append("deepskyblue")
        hlines_styles.append("dotted")
        hlines_widths.append(1.0)

    # Sell-side Liquidity: 現在価格より「下」にあるスウィング安値のみ
    # (価格に下抜かれた安値は流動性が消費済みなので描画しない)
    for lvl in _pick_near_levels(
        [l for l in smc.get("sell_liquidity", []) if float(l) < current_close],
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

    # Equal Highs: 現在価格より「上」にあるクラスターのみ（Sweep前の有効な流動性プール）
    for lvl in smc.get("eq_highs", []):
        try:
            f = float(lvl)
            if f <= current_close:  # Sweep済み（上抜かれた）EQHは無効
                continue
            hlines_prices.append(f)
            hlines_colors.append("#FFD700")   # Gold
            hlines_styles.append("dashed")
            hlines_widths.append(1.8)
            line_labels.append((f, "EQH", "#FFD700"))
        except (TypeError, ValueError):
            pass

    # Equal Lows: 現在価格より「下」にあるクラスターのみ（Sweep前の有効な流動性プール）
    for lvl in smc.get("eq_lows", []):
        try:
            f = float(lvl)
            if f >= current_close:  # Sweep済み（下抜かれた）EQLは無効
                continue
            hlines_prices.append(f)
            hlines_colors.append("#00CED1")   # Dark Turquoise
            hlines_styles.append("dashed")
            hlines_widths.append(1.8)
            line_labels.append((f, "EQL", "#00CED1"))
        except (TypeError, ValueError):
            pass

    # スイープされたレベル: 太い点線 (HIGH→赤橙, LOW→青緑)
    if swept_level is not None:
        _sw_color = "#FF6B35" if str(swept_type).upper() == "HIGH" else "#00E676"
        hlines_prices.append(float(swept_level))
        hlines_colors.append(_sw_color)
        hlines_styles.append("dashed")
        hlines_widths.append(2.0)
        _sw_label = f"SWEEP({'↑HIGH' if str(swept_type).upper() == 'HIGH' else '↓LOW'})"
        line_labels.append((float(swept_level), _sw_label, _sw_color))

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
        volume=False,
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
    _label_margin = 8  # バー8本分（ラベル密集回避のため余白を広めに）
    ax_main.set_xlim(right=len(ohlc) - 1 + _label_margin)
    # Y軸を明示固定 (遠距離hline描画後でもローソク足が画面に収まるよう)
    ax_main.set_ylim(_draw_lo, _draw_hi)

    # M15 CHoCH: コーラルの一点鎖線 (post-plot描画 — タプルlinestyleはhlines非対応)
    # M15チャートのみに描画 (H1チャートにM15微細構造は不要)
    if timeframe != config.TREND_TF:
        for _lvl in _m15_choch_draw_levels:
            if _draw_lo <= _lvl <= _draw_hi:
                ax_main.axhline(_lvl, color="coral", linewidth=1.0, linestyle=(0, (3, 1, 1, 1)), zorder=6, alpha=0.85)
                ax_main.text(
                    len(ohlc) - 1 + 0.6, _lvl, "CHoCH",
                    color="coral", fontsize=6, va="center", ha="left", fontweight="bold",
                    bbox=dict(facecolor="#1e222d", edgecolor="coral", alpha=0.85, boxstyle="round,pad=0.15"),
                    zorder=10,
                )

    # 現在価格ライン: 黄色の点線 + 右端に価格ラベル
    _digits = len(str(current_close).rstrip('0').split('.')[-1]) if '.' in str(current_close) else 2
    _price_fmt = f"{current_close:.{min(_digits, 5)}f}"
    ax_main.axhline(current_close, color="#FFD700", linewidth=1.2, linestyle=(0, (4, 2)), zorder=9, alpha=0.9)
    ax_main.text(
        len(ohlc) - 1 + 0.5,
        current_close,
        f" ▶{_price_fmt}",
        color="#FFD700",
        fontsize=7,
        va="center",
        ha="left",
        fontweight="bold",
        bbox=dict(facecolor="#1a1a1a", edgecolor="#FFD700", alpha=0.8, boxstyle="round,pad=0.2"),
        zorder=12,
    )

    # ── OBゾーン・FVGゾーンをRectangle Boxで描画 ──
    ob_zones: list[dict] = ob_zones_raw
    fvg_zones: list[dict] = smc.get("fvg_zones", [])
    h1_ob_zones: list[dict] = smc.get("h1_ob_zones", [])

    # ── OBゾーン絞り込み: Mitigated除外・価格に近いBull/Bear各最大2件 ──
    _MAX_OB_PER_SIDE = 2
    _MAX_FVG = 3
    _bull_obs_tmp: list[tuple] = []
    _bear_obs_tmp: list[tuple] = []
    for _z in ob_zones:
        try:
            _hi = float(_z["high"]); _lo = float(_z["low"])
            _zt = str(_z.get("type", "bull")).lower()
            _mit = (current_close < _lo) if _zt == "bull" else (current_close > _hi)
            if _mit or not (_draw_lo <= _lo <= _draw_hi or _draw_lo <= _hi <= _draw_hi):
                continue
            _mid = (_hi + _lo) / 2
            (_bull_obs_tmp if _zt == "bull" else _bear_obs_tmp).append((abs(_mid - current_close), _z))
        except (KeyError, TypeError, ValueError):
            pass
    _bull_obs_tmp.sort(key=lambda x: x[0])
    _bear_obs_tmp.sort(key=lambda x: x[0])
    ob_zones_draw = [z for _, z in _bull_obs_tmp[:_MAX_OB_PER_SIDE]] + [z for _, z in _bear_obs_tmp[:_MAX_OB_PER_SIDE]]

    # ── FVGゾーン絞り込み: Mitigated除外・近接上位3件 ──
    # Bull FVG: 価格が下端(lo)割れ = 上昇ギャップを完全通過 = Mitigated
    # Bear FVG: 価格が上端(hi)超え = 下降ギャップを完全通過 = Mitigated
    # ゾーン内(lo <= price <= hi)はエントリーチャンスなので描画する
    _fvg_tmp: list[tuple] = []
    for _z in fvg_zones:
        try:
            _hi = float(_z["high"]); _lo = float(_z["low"])
            if not (_draw_lo <= _lo <= _draw_hi or _draw_lo <= _hi <= _draw_hi):
                continue
            _fvg_type = str(_z.get("type", "bull")).lower()
            _fvg_mit = (current_close < _lo) if _fvg_type == "bull" else (current_close > _hi)
            if _fvg_mit:
                continue
            _fvg_tmp.append((abs((_hi + _lo) / 2 - current_close), _z))
        except (TypeError, ValueError):
            pass
    _fvg_tmp.sort(key=lambda x: x[0])
    fvg_zones_draw = [z for _, z in _fvg_tmp[:_MAX_FVG]]

    # OBゾーンは形成バー(bar_offset)から右端まで描画。
    # bar_offsetがない場合は右3/4のみ（フォールバック）
    _ob_start_fallback = len(ohlc) // 4

    for zone in ob_zones_draw:
        try:
            hi = float(zone["high"])
            lo = float(zone["low"])
            zone_type = str(zone.get("type", "bull")).lower()

            # OB形成バーからの描画開始位置を計算
            _bar_off = zone.get("bar_offset")
            if _bar_off is not None:
                _ob_start_x = max(0, len(ohlc) - 1 - int(_bar_off))
            else:
                _ob_start_x = _ob_start_fallback
            box_width = max(1, len(ohlc) - 1 - _ob_start_x)

            # H1 OBとのコンフルエンス判定: M15チャートのみ (H1チャートでは自己参照になるため無効)
            # M15 OBがH1 OBと重複 → High Probability OBとして強調
            if timeframe == config.TREND_TF:
                is_confluence = False  # H1チャートはコンフルエンス表示なし
            else:
                is_confluence = any(
                    str(h1z.get("type", "")).lower() == zone_type
                    and float(h1z["low"]) < hi
                    and float(h1z["high"]) > lo
                    for h1z in h1_ob_zones
                )

            if is_confluence:
                color = _SMC_CONF_BULL_COLOR if zone_type == "bull" else _SMC_CONF_BEAR_COLOR
            else:
                color = _SMC_BULL_COLOR if zone_type == "bull" else _SMC_BEAR_COLOR

            face_alpha = 0.28 if is_confluence else 0.18
            edge_alpha = 1.0  if is_confluence else 0.75
            linewidth  = 1.8  if is_confluence else 0.8
            rect = mpatches.Rectangle(
                xy=(_ob_start_x, lo),
                width=box_width,
                height=hi - lo,
                linewidth=linewidth,
                edgecolor=(*color, edge_alpha),
                facecolor=(*color, face_alpha),
                zorder=2,
            )
            ax_main.add_patch(rect)
            # OB 50%ミッドライン (TradingView準拠)
            _ob_mid = (hi + lo) / 2
            ax_main.plot(
                [_ob_start_x, _ob_start_x + box_width],
                [_ob_mid, _ob_mid],
                color=(*color, 0.55),
                linewidth=0.7,
                linestyle="dashed",
                zorder=3,
            )
            label_text = ("★ H.Prob Bull OB" if zone_type == "bull" else "★ H.Prob Bear OB") if is_confluence else ("Bull OB" if zone_type == "bull" else "Bear OB")
            ax_main.text(
                (_ob_start_x + box_width) * 0.98,
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

    for zone in fvg_zones_draw:
        try:
            hi = float(zone["high"])
            lo = float(zone["low"])

            # FVG形成バーからの描画開始位置を計算
            _bar_off = zone.get("bar_offset")
            if _bar_off is not None:
                _fvg_start_x = max(0, len(ohlc) - 1 - int(_bar_off))
            else:
                _fvg_start_x = _ob_start_fallback
            _fvg_width = max(1, len(ohlc) - 1 - _fvg_start_x)

            rect = mpatches.Rectangle(
                xy=(_fvg_start_x, lo),
                width=_fvg_width,
                height=hi - lo,
                linewidth=0.6,
                edgecolor=(*_SMC_FVG_COLOR, 0.55),
                facecolor=(*_SMC_FVG_COLOR, 0.11),
                zorder=2,
            )
            ax_main.add_patch(rect)
            ax_main.text(
                (_fvg_start_x + _fvg_width) * 0.98,
                (hi + lo) / 2,
                "FVG",
                color=(*_SMC_FVG_COLOR, 0.9),
                fontsize=6,
                va="center",
                ha="right",
                fontweight="bold",
                zorder=11,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("FVGゾーン描画スキップ: %s", e)

    # ── 主要ライン右端ラベル (PDH/PDL/PWH/PWL/INV/SWEEP) ──
    # 密集回避: 近接ラベルをY方向にずらす
    x_right = len(ohlc) - 1
    _min_label_gap = _price_span * 0.035  # 表示範囲の3.5%以内は「密集」とみなす
    _used_y: list[float] = []  # 使用済みY位置トラッキング
    for price, label, color in sorted(line_labels, key=lambda x: x[0]):
        # 密集チェック: 既存ラベルとの最小距離を確保
        disp_y = price
        attempts = 0
        while any(abs(disp_y - used) < _min_label_gap for used in _used_y) and attempts < 10:
            disp_y += _min_label_gap * (1 if attempts % 2 == 0 else -1) * ((attempts // 2) + 1)
            attempts += 1
        _used_y.append(disp_y)
        ax_main.text(
            x_right + 0.6,
            disp_y,
            label,
            color=color,
            fontsize=7,
            va="center",
            ha="left",
            bbox=dict(facecolor="#1e222d", edgecolor=color, alpha=0.85, boxstyle="round,pad=0.15"),
            zorder=10,
        )

    # ── 凡例: タイムフレーム別に必要なエントリのみ ──
    legend_handles = [
        Line2D([0], [0], color="dodgerblue", lw=1.2, ls="solid",  label="BOS (H1)"),
        Line2D([0], [0], color="#b2b5be",    lw=1.2, ls="solid",  label="MA20"),
        Line2D([0], [0], color="darkorange", lw=1.2, ls="dashed", label="CHoCH (H1)"),
    ]
    # CHoCH (M15) はM15チャートのみ表示 (H1チャートでは描画されない)
    if timeframe != config.TREND_TF:
        legend_handles.append(
            Line2D([0], [0], color="coral", lw=1.0, ls=(0,(3,1,1,1)), label="CHoCH (M15)")
        )
    legend_handles += [
        Line2D([0], [0], color="deepskyblue", lw=1.0, ls="dotted", label="Liquidity (Buy-side ↑)"),
        Line2D([0], [0], color="firebrick", lw=1.0, ls="dotted", label="Liquidity (Sell-side ↓)"),
        mpatches.Patch(facecolor=_SMC_BULL_COLOR, alpha=0.20, label="OB (Bullish)"),
        mpatches.Patch(facecolor=_SMC_BEAR_COLOR, alpha=0.20, label="OB (Bearish)"),
        mpatches.Patch(facecolor=_SMC_CONF_BULL_COLOR, alpha=0.35, label="★ H.Prob OB (H1+M15)"),
        mpatches.Patch(facecolor=_SMC_FVG_COLOR, alpha=0.18, label="FVG"),
        Line2D([0], [0], color="gold", lw=1.0, ls="dashed", label="PDH/PDL"),
        Line2D([0], [0], color="orchid", lw=1.0, ls="dashed", label="PWH/PWL"),
        Line2D([0], [0], color="#FFD700", lw=1.8, ls="dashed", label="Equal Highs (EQH ↑)"),
        Line2D([0], [0], color="#00CED1", lw=1.8, ls="dashed", label="Equal Lows (EQL ↓)"),
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
        framealpha=0.80,
        facecolor="#1e222d",
        edgecolor="#434651",
        labelcolor="white",
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
    swept_level: float | None = None,
    swept_type: str | None = None,
) -> tuple[str | None, str | None]:
    """H1 と M15 のSMCオーバーレイ付きチャートをbase64で返す。

    エントリー時: invalidation_price=None で両足生成
    エグジット監視時: invalidation_price を指定してM15のみ生成してもよい
    """
    h1_b64 = generate_smc_chart_base64(symbol, config.TREND_TF, smc_features, None,
                                        swept_level=swept_level, swept_type=swept_type)
    m15_b64 = generate_smc_chart_base64(symbol, config.EXECUTION_TF, smc_features, invalidation_price,
                                         swept_level=swept_level, swept_type=swept_type)
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
