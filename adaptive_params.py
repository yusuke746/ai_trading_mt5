"""アダプティブパラメータの読み書き・更新ロジック

ローリング窓（直近 ADAPTIVE_LOOKBACK_DAYS 日）のトレード結果を評価し、
バケット（market_regime × entry_type）ごとに confidence 閾値を自動更新する。

更新ルール:
  - 勝率 < 40%   → 閾値 +STEP (絞り込み)
  - 勝率 > 65% かつ 期待値 > 0 → 閾値 -STEP (緩和)
  - 変動上限: ±ADAPTIVE_CONF_MAX_WEEKLY_DELTA / cycle
  - 閾値範囲: ADAPTIVE_CONF_MIN 〜 ADAPTIVE_CONF_MAX
  - サンプル数 < ADAPTIVE_MIN_SAMPLES のバケットは更新しない
"""

import json
import logging
import os
from datetime import datetime

import config
import trade_logger

logger = logging.getLogger(__name__)

_PARAMS_PATH = os.path.join(config.ANALYTICS_DIR, "adaptive_params.json")


def _default() -> dict:
    return {
        "global_confidence_threshold": config.ADAPTIVE_CONF_MIN,
        "updated_at": None,
        "buckets": {},
    }


def load() -> dict:
    """adaptive_params.json を読み込む。ファイルがなければデフォルト値を返す。"""
    if not os.path.exists(_PARAMS_PATH):
        return _default()
    try:
        with open(_PARAMS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # キー移行: 旧フォーマット対応
        if "confidence_threshold" in data and "global_confidence_threshold" not in data:
            data["global_confidence_threshold"] = data.pop("confidence_threshold")
        return data
    except Exception as e:
        logger.warning("[Adaptive] params load failed: %s — using defaults", e)
        return _default()


def _save(params: dict) -> None:
    params["updated_at"] = datetime.utcnow().isoformat()
    with open(_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def get_confidence_threshold(regime: str | None = None, entry_type: str | None = None) -> int:
    """バケット別の confidence 閾値を返す。該当バケットがなければグローバル閾値。"""
    params = load()
    global_thr = int(params.get("global_confidence_threshold", config.ADAPTIVE_CONF_MIN))
    if regime and entry_type:
        key = f"{regime}/{entry_type}"
        bucket = params.get("buckets", {}).get(key)
        if bucket:
            return int(bucket.get("confidence_threshold", global_thr))
    return global_thr


def evaluate_and_adapt() -> dict:
    """直近 ADAPTIVE_LOOKBACK_DAYS 日のトレードを評価し、閾値を更新する。

    Returns:
        更新サマリ dict
    """
    if not config.ADAPTIVE_ENABLED:
        return {"skipped": True, "reason": "ADAPTIVE_ENABLED=false"}

    rows = trade_logger.fetch_recent_closed(config.ADAPTIVE_LOOKBACK_DAYS)
    if not rows:
        return {
            "skipped": True,
            "reason": "no_closed_trades",
            "lookback_days": config.ADAPTIVE_LOOKBACK_DAYS,
        }

    params = load()

    # ─── バケット別集計 ───────────────────────
    buckets: dict[str, dict] = {}
    for row in rows:
        regime = row.get("market_regime") or "UNKNOWN"
        entry_type = row.get("entry_type") or "UNKNOWN"
        profit = row.get("result_profit") or 0.0
        key = f"{regime}/{entry_type}"
        if key not in buckets:
            buckets[key] = {"wins": 0, "losses": 0, "total_profit": 0.0}
        if profit > 0:
            buckets[key]["wins"] += 1
        else:
            buckets[key]["losses"] += 1
        buckets[key]["total_profit"] += profit

    # ─── 閾値更新 ────────────────────────────
    changes = []
    global_thr = int(params.get("global_confidence_threshold", config.ADAPTIVE_CONF_MIN))

    for key, bucket in buckets.items():
        total = bucket["wins"] + bucket["losses"]
        if total < config.ADAPTIVE_MIN_SAMPLES:
            # サンプル不足 → スキップ
            continue

        win_rate = bucket["wins"] / total
        expectancy = bucket["total_profit"] / total
        existing = params.get("buckets", {}).get(key, {})
        current_thr = int(existing.get("confidence_threshold", global_thr))

        if win_rate < 0.40:
            raw_delta = config.ADAPTIVE_CONF_STEP       # 絞り込み
        elif win_rate > 0.65 and expectancy > 0:
            raw_delta = -config.ADAPTIVE_CONF_STEP      # 緩和
        else:
            raw_delta = 0

        if raw_delta == 0:
            continue

        # 週次変動上限キャップ
        capped_delta = max(
            -config.ADAPTIVE_CONF_MAX_WEEKLY_DELTA,
            min(config.ADAPTIVE_CONF_MAX_WEEKLY_DELTA, raw_delta),
        )
        new_thr = max(
            config.ADAPTIVE_CONF_MIN,
            min(config.ADAPTIVE_CONF_MAX, current_thr + capped_delta),
        )
        actual_delta = new_thr - current_thr

        if not isinstance(params.get("buckets"), dict):
            params["buckets"] = {}

        params["buckets"][key] = {
            "confidence_threshold": new_thr,
            "win_rate": round(win_rate, 4),
            "expectancy": round(expectancy, 2),
            "total_trades": total,
            "lookback_days": config.ADAPTIVE_LOOKBACK_DAYS,
            "last_updated": datetime.utcnow().isoformat(),
        }
        changes.append({
            "bucket": key,
            "old_threshold": current_thr,
            "new_threshold": new_thr,
            "delta": actual_delta,
            "win_rate": round(win_rate, 4),
            "expectancy": round(expectancy, 2),
            "total_trades": total,
        })

    if changes:
        _save(params)
        logger.info(
            "[Adaptive] %d bucket(s) updated (lookback=%dd): %s",
            len(changes),
            config.ADAPTIVE_LOOKBACK_DAYS,
            ", ".join(f"{c['bucket']} {c['old_threshold']}→{c['new_threshold']}" for c in changes),
        )
    else:
        logger.info(
            "[Adaptive] no threshold changes (buckets=%d, lookback=%dd, min_samples=%d)",
            len(buckets),
            config.ADAPTIVE_LOOKBACK_DAYS,
            config.ADAPTIVE_MIN_SAMPLES,
        )

    return {
        "skipped": False,
        "lookback_days": config.ADAPTIVE_LOOKBACK_DAYS,
        "total_trades": len(rows),
        "buckets_evaluated": len(buckets),
        "buckets_updated": len(changes),
        "changes": changes,
    }
