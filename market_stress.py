"""市場ストレス検知モジュール

「高騰・暴落の事前察知」ではなく
「異常を検知したら素早く退避する」設計。

フロー:
  1. サイクルごとにスプレッド/ATRを計測
  2. 平常時の N 倍以上 → ストレス検知 → gpt-5-nano で状況判断
  3. GPTが返した hold_minutes の間エントリーをブロック
  4. 解除条件: hold_until 超過 AND スプレッド正常化
     ただし min_hold_until (最低保持) は必ず守る
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone

_JST = timezone(timedelta(hours=9))

import config

logger = logging.getLogger(__name__)

# ── データ構造 ──────────────────────────────────────────

@dataclass
class MarketStressState:
    symbol: str
    risk_level: str            # "HIGH" | "MEDIUM"
    summary: str
    triggered_at: datetime
    hold_until: datetime       # GPT提案 or フォールバックTTL (これを超えたら解除候補)
    min_hold_until: datetime   # 最低保持期限 (早期解除防止)
    source: str                # "spread_spike" | "atr_spike" | "both"
    spread_at_trigger: float   # 検知時のスプレッド値 (デバッグ用)
    should_close_positions: bool = False  # 既存ポジションもクローズすべきか


@dataclass
class PostRecoveryState:
    """ストレス解除直後の慎重モード状態。残りトレード数が0になるまでロット縮小を適用する。"""
    symbol: str
    trades_remaining: int   # あと何回ロット縮小するか
    lot_multiplier: float   # ロット倍率 (例: 0.5 = 50%)
    cleared_at: datetime    # ストレス解除時刻


# symbol → MarketStressState (アクティブなストレス状態)
_stress_states: dict[str, MarketStressState] = {}
# symbol → PostRecoveryState (復帰後慎重モード状態)
_post_recovery_states: dict[str, PostRecoveryState] = {}
_lock = threading.Lock()

# 銘柄ごとのスプレッドベースライン (過去N件のスプレッドを保持)
_spread_baseline: dict[str, deque[float]] = {}


# ── ベースライン管理 ──────────────────────────────────

def update_spread_baseline(symbol: str, spread: float) -> None:
    """スプレッドのサンプルを追加してベースラインを更新する。"""
    if symbol not in _spread_baseline:
        _spread_baseline[symbol] = deque(maxlen=config.MARKET_STRESS_SPREAD_BASELINE_N)
    _spread_baseline[symbol].append(spread)


def get_baseline_spread(symbol: str) -> float | None:
    """ベースライン平均スプレッドを返す。サンプル不足は None。"""
    samples = _spread_baseline.get(symbol)
    if not samples or len(samples) < 5:  # 最低5サンプル必要
        return None
    return sum(samples) / len(samples)


# ── ストレス検知 ─────────────────────────────────────

def check_and_update(
    symbol: str,
    current_spread: float,
    current_atr: float,
    baseline_atr: float | None,
) -> MarketStressState | None:
    """
    スプレッド/ATR が閾値を超えていたらストレス検知。
    アクティブなストレス状態があれば解除チェックも行う。
    Returns: 現在有効な MarketStressState (なければ None)
    """
    now = datetime.now(UTC)

    # ベースラインにスプレッドを記録
    update_spread_baseline(symbol, current_spread)
    baseline_spread = get_baseline_spread(symbol)

    # ── 既存ストレス状態の解除チェック ──
    with _lock:
        state = _stress_states.get(symbol)

    if state is not None:
        cleared = _check_clear(state, symbol, current_spread, baseline_spread, now, current_atr, baseline_atr)
        if cleared:
            with _lock:
                del _stress_states[symbol]
                # 復帰後慎重モードを開始: 最初 N 回はロット縮小
                _post_recovery_states[symbol] = PostRecoveryState(
                    symbol=symbol,
                    trades_remaining=config.POST_RECOVERY_TRADE_COUNT,
                    lot_multiplier=config.POST_RECOVERY_LOT_MULTIPLIER,
                    cleared_at=now,
                )
            logger.info(
                "[MarketStress] %s: ストレス状態解除 (保持 %.0f 分, spread=%.1f) "
                "→ 復帰後慎重モード開始 (x%.1f × 最初%d回)",
                symbol,
                (now - state.triggered_at).total_seconds() / 60,
                current_spread,
                config.POST_RECOVERY_LOT_MULTIPLIER,
                config.POST_RECOVERY_TRADE_COUNT,
            )
            state = None

    if state is not None:
        return state  # まだアクティブ

    # ── 新規ストレス検知 ──
    triggered_sources: list[str] = []

    # スプレッド急拡大チェック
    if baseline_spread and baseline_spread > 0:
        spread_ratio = current_spread / baseline_spread
        if spread_ratio >= config.MARKET_STRESS_SPREAD_RATIO:
            triggered_sources.append("spread_spike")
            logger.warning(
                "[MarketStress] %s: スプレッド急拡大 %.1f → %.1f (x%.1f)",
                symbol, baseline_spread, current_spread, spread_ratio,
            )
    # ATR急拡大チェック
    if baseline_atr and baseline_atr > 0:
        atr_ratio = current_atr / baseline_atr
        if atr_ratio >= config.MARKET_STRESS_ATR_RATIO:
            triggered_sources.append("atr_spike")
            logger.warning(
                "[MarketStress] %s: ATR急拡大 baseline=%.5f current=%.5f (x%.1f)",
                symbol, baseline_atr, current_atr, atr_ratio,
            )

    if not triggered_sources:
        return None

    # ストレス検知 → GPT判断 or フォールバックTTL
    source_str = "_".join(triggered_sources) if len(triggered_sources) == 1 else "both"

    # クローズ閘値判定: スプレッドが CLOSE_RATIO 以上 かつ GPTがHIGH の時のみ
    spread_close_triggered = (
        baseline_spread is not None
        and baseline_spread > 0
        and (current_spread / baseline_spread) >= config.MARKET_STRESS_SPREAD_CLOSE_RATIO
    )

    new_state = _create_stress_state(
        symbol=symbol,
        source=source_str,
        spread_at_trigger=current_spread,
        baseline_spread=baseline_spread or current_spread,
        now=now,
        spread_close_triggered=spread_close_triggered,
    )

    with _lock:
        _stress_states[symbol] = new_state

    logger.warning(
        "[MarketStress] %s: ストレス状態追加 risk=%s hold_until=%s source=%s summary=%s",
        symbol,
        new_state.risk_level,
        new_state.hold_until.astimezone(_JST).strftime("%m/%d %H:%M JST"),
        new_state.source,
        new_state.summary,
    )
    return new_state


def _check_clear(
    state: MarketStressState,
    symbol: str,
    current_spread: float,
    baseline_spread: float | None,
    now: datetime,
    current_atr: float | None = None,
    baseline_atr: float | None = None,
) -> bool:
    """ストレス状態を解除してよいか判定する。"""
    # 最低保持期限内は絶対に解除しない
    if now < state.min_hold_until:
        return False

    # hold_until を超えていない場合はまだ解除しない
    if now < state.hold_until:
        return False

    # 強制解除期限: hold_until + GRACE 分を過ぎたらスプレッド/ATR問わず強制解除
    force_clear_at = state.hold_until + timedelta(minutes=config.MARKET_STRESS_FORCE_CLEAR_GRACE_MIN)
    if now >= force_clear_at:
        logger.warning(
            "[MarketStress] %s: 強制解除 (hold_until+%d分経過, spread=%.1f) → エントリー再開",
            symbol, config.MARKET_STRESS_FORCE_CLEAR_GRACE_MIN, current_spread,
        )
        return True

    # 条件1: スプレッドが正常範囲に戻っているか
    if baseline_spread and baseline_spread > 0:
        spread_ratio = current_spread / baseline_spread
        if spread_ratio >= config.MARKET_STRESS_SPREAD_CLEAR_RATIO:
            # スプレッドがまだ広い → 解除しない
            return False

    # 条件2: ATRが正常範囲に戻っているか
    if current_atr and baseline_atr and baseline_atr > 0:
        atr_ratio = current_atr / baseline_atr
        if atr_ratio >= config.MARKET_STRESS_ATR_CLEAR_RATIO:
            logger.debug(
                "[MarketStress] %s: ATRまだ高い (x%.2f >= %.2f) → 解除保留",
                symbol, atr_ratio, config.MARKET_STRESS_ATR_CLEAR_RATIO,
            )
            return False

    # 条件3 (廃止): ニュースブロックはエントリー時に main.py が個別チェックするため
    # ストレス解除の条件としては使わない。スプレッド/ATRが正常化すれば解除を許可する。

    return True


def _create_stress_state(
    symbol: str,
    source: str,
    spread_at_trigger: float,
    baseline_spread: float,
    now: datetime,
    spread_close_triggered: bool = False,
) -> MarketStressState:
    """GPT判断(有効時)またはフォールバックTTLでストレス状態を作成する。"""
    # フォールバック (GPT無効 or 失敗時)
    fallback_hold_minutes = 60 if "spread_spike" in source else 90
    risk_level = "HIGH"
    summary = f"スプレッド/ATR急変検知 ({source})"

    if config.MARKET_STRESS_AI_ENABLED and config.OPENAI_API_KEY:
        ai_result = _ask_gpt_for_stress(symbol, source, spread_at_trigger, baseline_spread)
        if ai_result:
            risk_level = ai_result.get("risk_level", "HIGH")
            summary = ai_result.get("summary", summary)
            hold_min = int(ai_result.get("hold_minutes", fallback_hold_minutes))
            hold_min = max(
                config.MARKET_STRESS_HOLD_MIN_MIN,
                min(config.MARKET_STRESS_HOLD_MAX_MIN, hold_min),
            )
            fallback_hold_minutes = hold_min

    return MarketStressState(
        symbol=symbol,
        risk_level=risk_level,
        summary=summary,
        triggered_at=now,
        hold_until=now + timedelta(minutes=fallback_hold_minutes),
        min_hold_until=now + timedelta(minutes=config.MARKET_STRESS_HOLD_MIN_MIN),
        source=source,
        spread_at_trigger=spread_at_trigger,
        # クローズは: スプレッドが CLOSE_RATIO 以上 かつ GPTがHIGH の時のみ (AND条件)
        should_close_positions=(spread_close_triggered and risk_level == "HIGH"),
    )


def _ask_gpt_for_stress(
    symbol: str,
    source: str,
    spread_now: float,
    baseline_spread: float,
) -> dict | None:
    """gpt-5-nano にスプレッド急拡大の状況を渡して保持時間・リスクを判断させる。"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)

        ratio = spread_now / baseline_spread if baseline_spread > 0 else 0
        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

        prompt = f"""FX自動売買システムのリスク管理AIです。
現在 {now_str} に {symbol} でスプレッド/ATR急拡大を検知しました。

【検知情報】
- 検知種別: {source}
- 現在スプレッド: {spread_now:.1f} pips (平常時の {ratio:.1f} 倍)
- 銘柄: {symbol}

web検索で現在の {symbol} のスプレッド拡大・市場異常の原因を調べてから判断してください。

JSONのみで回答してください:
{{
  "risk_level": "HIGH" | "MEDIUM",
  "hold_minutes": <エントリー禁止する推奨時間(整数・分)>,
  "summary": "web検索で判明した原因と状況の1文要約"
}}

判断基準:
- 経済指標発表直後の一時的なスプレッド拡大 → hold_minutes: 15〜30
- 重要指標・要人発言による高ボラティリティ → hold_minutes: 30〜60
- 地政学リスク・市場クラッシュ・フラッシュクラッシュ → hold_minutes: 120〜480
- 週明け早朝など流動性が薄い時間帯のスプレッド拡大 → hold_minutes: 15〜30"""

        response = client.responses.create(
            model=config.NEWS_MONITOR_MODEL,
            tools=[{"type": "web_search_preview"}],
            input=[{"role": "user", "content": prompt}],
        )
        raw = response.output_text
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            rl = str(data.get("risk_level", "HIGH")).upper()
            if rl not in {"HIGH", "MEDIUM"}:
                rl = "HIGH"
            data["risk_level"] = rl
            return data
    except Exception as e:
        logger.warning("[MarketStress] GPT判断失敗 %s: %s → フォールバックTTL使用", symbol, e)
    return None


# ── 復帰後慎重モード API ────────────────────────────────

def get_lot_multiplier(symbol: str) -> float:
    """ストレス復帰後慎重モード中であればロット倍率を返す。通常時は 1.0。"""
    with _lock:
        pr = _post_recovery_states.get(symbol)
    if pr is None or pr.trades_remaining <= 0:
        return 1.0
    return pr.lot_multiplier


def consume_post_recovery_trade(symbol: str) -> None:
    """発注成立後に呼び出してカウンタを 1 減らす。0 になったら慎重モード終了。"""
    with _lock:
        pr = _post_recovery_states.get(symbol)
        if pr is None:
            return
        pr.trades_remaining -= 1
        if pr.trades_remaining <= 0:
            del _post_recovery_states[symbol]
            logger.info(
                "[MarketStress] %s: 復帰後慎重モード終了 (通常ロットに復帰)",
                symbol,
            )
        else:
            logger.info(
                "[MarketStress] %s: 復帰後慎重モード残り %d 回 (x%.1f)",
                symbol, pr.trades_remaining, pr.lot_multiplier,
            )


# ── 外部公開API ──────────────────────────────────────

def get_stress_state(symbol: str) -> MarketStressState | None:
    """現在のストレス状態を返す (なければ None)。"""
    with _lock:
        return _stress_states.get(symbol)


def is_stressed(symbol: str) -> bool:
    """エントリーをブロックすべきストレス状態か。

    NOTE: main.py では is_stressed() が True の場合に早期 return するため
    check_and_update() が呼ばれない。そのため hold_until の期限切れを
    ここで自律的にチェックして強制解除する。
    """
    with _lock:
        state = _stress_states.get(symbol)
    if state is None:
        return False

    now = datetime.now(UTC)

    # --- 強制解除チェック (check_and_update が呼ばれない場合の安全弁) ---
    # hold_until + GRACE_MIN を超えたら無条件に解除
    force_clear_at = state.hold_until + timedelta(minutes=config.MARKET_STRESS_FORCE_CLEAR_GRACE_MIN)
    if now >= force_clear_at:
        with _lock:
            _stress_states.pop(symbol, None)
            # 復帰後慎重モードを開始
            _post_recovery_states[symbol] = PostRecoveryState(
                symbol=symbol,
                trades_remaining=config.POST_RECOVERY_TRADE_COUNT,
                lot_multiplier=config.POST_RECOVERY_LOT_MULTIPLIER,
                cleared_at=now,
            )
        logger.warning(
            "[MarketStress] %s: 強制解除 (hold_until+%d分経過, is_stressed経由) → エントリー再開",
            symbol, config.MARKET_STRESS_FORCE_CLEAR_GRACE_MIN,
        )
        return False

    return True


def clear_all() -> None:
    """全ストレス状態をクリア (テスト・デバッグ用)。"""
    with _lock:
        _stress_states.clear()
