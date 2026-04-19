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

from openai import OpenAI

import config
import trade_logger

logger = logging.getLogger(__name__)

_PARAMS_PATH = os.path.join(config.ANALYTICS_DIR, "adaptive_params.json")
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _default() -> dict:
    return {
        "global_confidence_threshold": config.ADAPTIVE_CONF_MIN,
        "updated_at": None,
        "last_llm_analysis_at": None,
        "last_llm_model": "",
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


def _extract_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _is_llm_due(last_iso: str | None) -> bool:
    if not config.ADAPTIVE_LLM_ENABLED:
        return False
    if not last_iso:
        return True
    try:
        last_dt = datetime.fromisoformat(last_iso)
    except ValueError:
        return True
    elapsed = (datetime.utcnow() - last_dt).total_seconds()
    return elapsed >= config.ADAPTIVE_LLM_INTERVAL_SEC


def _build_bucket_stats(rows: list[dict]) -> dict[str, dict]:
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
    return buckets


def _apply_delta(current_thr: int, delta: int) -> int:
    capped_delta = max(
        -config.ADAPTIVE_CONF_MAX_WEEKLY_DELTA,
        min(config.ADAPTIVE_CONF_MAX_WEEKLY_DELTA, delta),
    )
    return max(
        config.ADAPTIVE_CONF_MIN,
        min(config.ADAPTIVE_CONF_MAX, current_thr + capped_delta),
    )


def _rule_based_update(params: dict, buckets: dict[str, dict]) -> list[dict]:
    changes = []
    global_thr = int(params.get("global_confidence_threshold", config.ADAPTIVE_CONF_MIN))
    changed = False

    for key, bucket in buckets.items():
        total = bucket["wins"] + bucket["losses"]
        if total < config.ADAPTIVE_MIN_SAMPLES:
            continue

        win_rate = bucket["wins"] / total
        expectancy = bucket["total_profit"] / total
        existing = params.get("buckets", {}).get(key, {})
        current_thr = int(existing.get("confidence_threshold", global_thr))

        if win_rate < 0.40:
            raw_delta = config.ADAPTIVE_CONF_STEP
        elif win_rate > 0.65 and expectancy > 0:
            raw_delta = -config.ADAPTIVE_CONF_STEP
        else:
            raw_delta = 0

        if raw_delta == 0:
            continue

        new_thr = _apply_delta(current_thr, raw_delta)
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
        changed = changed or (new_thr != current_thr)
        changes.append({
            "bucket": key,
            "old_threshold": current_thr,
            "new_threshold": new_thr,
            "delta": new_thr - current_thr,
            "win_rate": round(win_rate, 4),
            "expectancy": round(expectancy, 2),
            "total_trades": total,
            "source": "rule",
        })

    total_wins = sum(b["wins"] for b in buckets.values())
    total_losses = sum(b["losses"] for b in buckets.values())
    total_trades = total_wins + total_losses
    if total_trades >= config.ADAPTIVE_MIN_SAMPLES:
        global_win_rate = total_wins / total_trades
        total_profit = sum(b["total_profit"] for b in buckets.values())
        global_expectancy = total_profit / total_trades

        if global_win_rate < 0.40:
            global_raw_delta = config.ADAPTIVE_CONF_STEP
        elif global_win_rate > 0.65 and global_expectancy > 0:
            global_raw_delta = -config.ADAPTIVE_CONF_STEP
        else:
            global_raw_delta = 0

        if global_raw_delta != 0:
            new_global_thr = _apply_delta(global_thr, global_raw_delta)
            if new_global_thr != global_thr:
                params["global_confidence_threshold"] = new_global_thr
                changed = True
                changes.append({
                    "bucket": "GLOBAL",
                    "old_threshold": global_thr,
                    "new_threshold": new_global_thr,
                    "delta": new_global_thr - global_thr,
                    "win_rate": round(global_win_rate, 4),
                    "expectancy": round(global_expectancy, 2),
                    "total_trades": total_trades,
                    "source": "rule",
                })

    if changed:
        _save(params)
    return changes


def _llm_suggest_changes(params: dict, buckets: dict[str, dict]) -> tuple[list[dict], str | None]:
    global_thr = int(params.get("global_confidence_threshold", config.ADAPTIVE_CONF_MIN))

    compact = []
    for key, b in buckets.items():
        total = b["wins"] + b["losses"]
        compact.append({
            "bucket": key,
            "wins": b["wins"],
            "losses": b["losses"],
            "total_trades": total,
            "win_rate": round(b["wins"] / total, 4) if total > 0 else 0.0,
            "expectancy": round(b["total_profit"] / total, 2) if total > 0 else 0.0,
            "current_threshold": int(params.get("buckets", {}).get(key, {}).get("confidence_threshold", global_thr)),
        })

    prompt = {
        "task": "Suggest weekly confidence-threshold adjustments for trading buckets.",
        "rules": {
            "only_json": True,
            "global_threshold_range": [config.ADAPTIVE_CONF_MIN, config.ADAPTIVE_CONF_MAX],
            "max_abs_delta_per_cycle": config.ADAPTIVE_CONF_MAX_WEEKLY_DELTA,
            "min_samples_for_update": config.ADAPTIVE_MIN_SAMPLES,
            "lookback_days": config.ADAPTIVE_LOOKBACK_DAYS,
            "no_update_when_sample_small": True,
        },
        "current": {
            "global_confidence_threshold": global_thr,
            "buckets": compact,
        },
        "output_schema": {
            "global_delta": "int",
            "bucket_deltas": [{"bucket": "str", "delta": "int"}],
            "rationale": "short str",
        },
    }

    client = _get_client()
    response = client.responses.create(
        model=config.ADAPTIVE_LLM_MODEL,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)}],
        }],
        reasoning={"effort": "medium"},
    )

    text = response.output_text or ""
    payload = _extract_json(text)
    if payload is None:
        raise ValueError("LLM response JSON parse failed")

    changes = []
    global_delta = int(payload.get("global_delta", 0))
    if global_delta != 0:
        old = global_thr
        new = _apply_delta(old, global_delta)
        if new != old:
            params["global_confidence_threshold"] = new
            changes.append({
                "bucket": "GLOBAL",
                "old_threshold": old,
                "new_threshold": new,
                "delta": new - old,
                "source": "llm",
            })

    bucket_deltas = payload.get("bucket_deltas") or []
    for item in bucket_deltas:
        if not isinstance(item, dict):
            continue
        key = str(item.get("bucket", "")).strip()
        if key not in buckets:
            continue
        total = buckets[key]["wins"] + buckets[key]["losses"]
        if total < config.ADAPTIVE_MIN_SAMPLES:
            continue
        delta = int(item.get("delta", 0))
        if delta == 0:
            continue

        if not isinstance(params.get("buckets"), dict):
            params["buckets"] = {}

        current = int(params["buckets"].get(key, {}).get("confidence_threshold", params["global_confidence_threshold"]))
        new = _apply_delta(current, delta)
        params["buckets"][key] = {
            "confidence_threshold": new,
            "win_rate": round(buckets[key]["wins"] / total, 4),
            "expectancy": round(buckets[key]["total_profit"] / total, 2),
            "total_trades": total,
            "lookback_days": config.ADAPTIVE_LOOKBACK_DAYS,
            "last_updated": datetime.utcnow().isoformat(),
        }
        if new != current:
            changes.append({
                "bucket": key,
                "old_threshold": current,
                "new_threshold": new,
                "delta": new - current,
                "source": "llm",
            })

    params["last_llm_analysis_at"] = datetime.utcnow().isoformat()
    params["last_llm_model"] = config.ADAPTIVE_LLM_MODEL
    if changes:
        _save(params)

    rationale = payload.get("rationale")
    return changes, rationale if isinstance(rationale, str) else None


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
    buckets = _build_bucket_stats(rows)

    if config.ADAPTIVE_LLM_ENABLED and not _is_llm_due(params.get("last_llm_analysis_at")):
        logger.info("[Adaptive] LLM weekly analysis not due yet")
        return {
            "skipped": True,
            "reason": "llm_not_due",
            "lookback_days": config.ADAPTIVE_LOOKBACK_DAYS,
            "total_trades": len(rows),
            "buckets_evaluated": len(buckets),
            "buckets_updated": 0,
            "changes": [],
        }

    llm_rationale = None
    if config.ADAPTIVE_LLM_ENABLED:
        try:
            changes, llm_rationale = _llm_suggest_changes(params, buckets)
            logger.info(
                "[Adaptive] weekly LLM analysis done: model=%s updated=%d",
                config.ADAPTIVE_LLM_MODEL,
                len(changes),
            )
        except Exception as e:
            logger.warning("[Adaptive] LLM analysis failed, fallback to rules: %s", e)
            changes = _rule_based_update(params, buckets)
    else:
        changes = _rule_based_update(params, buckets)

    if changes:
        logger.info(
            "[Adaptive] %d update(s) applied (lookback=%dd): %s",
            len(changes),
            config.ADAPTIVE_LOOKBACK_DAYS,
            ", ".join(f"{c['bucket']} {c['old_threshold']}→{c['new_threshold']}[{c.get('source', 'rule')}]" for c in changes),
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
        "llm_model": config.ADAPTIVE_LLM_MODEL if config.ADAPTIVE_LLM_ENABLED else None,
        "llm_rationale": llm_rationale,
    }
