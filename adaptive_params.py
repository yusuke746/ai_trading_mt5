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

    now = datetime.utcnow()

    # 土曜(5) または 日曜(6) のみ実行を許可
    # weekday(): 0=月, 1=火, 2=水, 3=木, 4=金, 5=土, 6=日
    if now.weekday() not in (5, 6):
        return False

    # 初回 (DBにレコードなし) は即実行
    if not last_iso:
        return True
    try:
        last_dt = datetime.fromisoformat(last_iso)
    except ValueError:
        return True

    # 前回実行から最低6日経過していること (同じ週末に2回実行されないための制限)
    elapsed = (now - last_dt).total_seconds()
    return elapsed >= 6 * 24 * 3600


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
    else:
        # ── 緊急脱出: サンプル不足 + 上限張り付き = エントリー枯渇デッドロック
        # 閾値が上限に達しているためエントリーが来ず、サンプルも増えない状態。
        # 最小ステップだけ強制緩和して次週のサンプル収集を促す。
        if global_thr >= config.ADAPTIVE_CONF_MAX:
            new_global_thr = _apply_delta(global_thr, -config.ADAPTIVE_CONF_STEP)
            if new_global_thr != global_thr:
                params["global_confidence_threshold"] = new_global_thr
                changed = True
                changes.append({
                    "bucket": "GLOBAL",
                    "old_threshold": global_thr,
                    "new_threshold": new_global_thr,
                    "delta": new_global_thr - global_thr,
                    "win_rate": 0.0,
                    "expectancy": 0.0,
                    "total_trades": total_trades,
                    "source": "rule_starvation_escape",
                })
                logger.info(
                    "[Adaptive] starvation escape: global threshold %d→%d "
                    "(samples=%d < min=%d, was at max)",
                    global_thr, new_global_thr, total_trades, config.ADAPTIVE_MIN_SAMPLES,
                )

    if changed:
        _save(params)
    return changes


def _fetch_macro_context() -> dict | None:
    """Web検索でVIX・DXY・主要リスクイベントを取得する。

    gpt-5-mini + web_search_preview を使い、最新のマクロ環境を
    JSON形式で返してもらう。失敗時は None を返す。

    取得する情報:
        - VIX: 現在値と先週比トレンド
        - DXY: ドル指数のトレンド
        - risk_events: 今週の主要イベント（FOMC・CPI・NFP等）
        - market_sentiment: "risk-on" / "risk-off" / "neutral"
    """
    if not config.OPENAI_API_KEY:
        return None

    macro_prompt = (
        "Search for the current market macro environment and return ONLY a JSON object "
        "(no markdown, no explanation) with this exact schema:\n"
        "{\n"
        '  "vix": {"value": <float or null>, "trend": "rising"|"falling"|"stable"},\n'
        '  "dxy": {"trend": "rising"|"falling"|"stable", "note": "<1 sentence>"},\n'
        '  "risk_events_this_week": [{"event": "<name>", "currency": "<USD/JPY/etc>", "days_until": <int>}],\n'
        '  "market_sentiment": "risk-on"|"risk-off"|"neutral",\n'
        '  "summary": "<2 sentences max>"\n'
        "}\n"
        "Use today's date as reference. Include only HIGH impact events."
    )

    try:
        client = _get_client()
        response = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": macro_prompt}],
            }],
        )
        text = response.output_text or ""
        payload = _extract_json(text)
        if payload and isinstance(payload, dict):
            logger.info("[Adaptive] macro context fetched via web search")
            return payload
        logger.debug("[Adaptive] macro context parse failed: %s", text[:200])
    except Exception as e:
        logger.debug("[Adaptive] macro context web search failed: %s", e)
    return None


def _build_counterfactual(lookback_days: int) -> dict | None:
    """「もし閾値がXだったら何件・勝率・期待値だったか」を計算する。

    DBの ai_confidence 付きトレード履歴をもとに、
    候補閾値それぞれでシミュレーションした結果を返す。
    サンプルが 5 件未満の場合は None を返す。

    例:
        {
          "sample_size": 27,
          "lookback_days": 30,
          "scenarios": {
            "65": {"entries": 27, "win_rate": 0.33, "avg_profit": -45.2},
            "70": {"entries": 18, "win_rate": 0.39, "avg_profit": -30.1},
            "75": {"entries":  8, "win_rate": 0.50, "avg_profit": 120.3},
            "78": {"entries":  2, "win_rate": 0.50, "avg_profit":  80.0},
          },
          "recommended_range": "70-75 offers best balance of frequency and expectancy"
        }
    """
    rows = trade_logger.fetch_counterfactual_data(lookback_days)
    if len(rows) < 5:
        return None

    # 候補閾値: config範囲を3刻みでカバー + 境界値
    candidates: list[int] = sorted({
        config.ADAPTIVE_CONF_MIN,
        config.ADAPTIVE_CONF_MIN + 3,
        config.ADAPTIVE_CONF_MIN + 6,
        config.ADAPTIVE_CONF_MIN + 9,
        config.ADAPTIVE_CONF_MAX - 3,
        config.ADAPTIVE_CONF_MAX,
    })

    scenarios: dict[str, dict] = {}
    for thr in candidates:
        filtered = [r for r in rows if (r["ai_confidence"] or 0) >= thr]
        if not filtered:
            scenarios[str(thr)] = {"entries": 0, "win_rate": 0.0, "avg_profit": 0.0}
            continue
        wins = sum(1 for r in filtered if (r["result_profit"] or 0) > 0)
        total_profit = sum((r["result_profit"] or 0.0) for r in filtered)
        scenarios[str(thr)] = {
            "entries": len(filtered),
            "win_rate": round(wins / len(filtered), 2),
            "avg_profit": round(total_profit / len(filtered), 1),
        }

    # 期待値プラスかつ最もエントリー数が多い閾値を推奨
    best_thr: int | None = None
    best_score = float("-inf")
    for thr in candidates:
        s = scenarios.get(str(thr), {})
        if s.get("entries", 0) == 0:
            continue
        # スコア = avg_profit * sqrt(entries)  ← 件数も考慮
        score = s["avg_profit"] * (s["entries"] ** 0.5)
        if score > best_score:
            best_score = score
            best_thr = thr

    return {
        "sample_size": len(rows),
        "lookback_days": lookback_days,
        "scenarios": scenarios,
        "best_threshold_by_score": best_thr,
        "note": (
            "scenarios show: if threshold=X, how many trades/win_rate/avg_profit would result. "
            "Use this to find the threshold with best balance of frequency and profitability."
        ),
    }


def _build_adaptive_context(params: dict, rows: list[dict]) -> dict:
    """LLMに渡す追加コンテキストを収集する。

    ・閾値の張り付き状態（何日上限に居るか）
    ・エントリー頻度（週次件数・枯渇警告）
    ・confidence帯別の勝率分布
    ・市場ボラティリティ（MT5 D1 ATR比）

    MT5未接続時や例外はスキップしてフォールバックする。
    """
    context: dict = {}
    global_thr = int(params.get("global_confidence_threshold", config.ADAPTIVE_CONF_MIN))

    # ── 1. 閾値の張り付き状態 ─────────────────────
    stagnant: dict = {}
    for key, bkt in params.get("buckets", {}).items():
        thr = int(bkt.get("confidence_threshold", global_thr))
        days_ago: int | None = None
        last_upd = bkt.get("last_updated")
        if last_upd:
            try:
                dt = datetime.fromisoformat(last_upd)
                days_ago = (datetime.utcnow() - dt).days
            except ValueError:
                pass
        stagnant[key] = {
            "threshold": thr,
            "at_max": thr >= config.ADAPTIVE_CONF_MAX,
            "days_since_last_change": days_ago,
        }

    context["threshold_status"] = {
        "global": global_thr,
        "global_at_max": global_thr >= config.ADAPTIVE_CONF_MAX,
        "max_allowed": config.ADAPTIVE_CONF_MAX,
        "min_allowed": config.ADAPTIVE_CONF_MIN,
        "buckets": stagnant,
    }

    # ── 2. エントリー頻度 ──────────────────────────
    total_entries = len(rows)
    context["entry_frequency"] = {
        "closed_trades_in_lookback": total_entries,
        "lookback_days": config.ADAPTIVE_LOOKBACK_DAYS,
        "avg_per_day": round(total_entries / max(config.ADAPTIVE_LOOKBACK_DAYS, 1), 2),
        "starvation_warning": (
            total_entries < 3 and global_thr >= config.ADAPTIVE_CONF_MAX
        ),
    }

    # ── 3. Confidence帯別の勝率分布 ───────────────
    try:
        detailed = trade_logger.fetch_recent_closed_detailed(config.ADAPTIVE_LOOKBACK_DAYS)
        if detailed:
            bands: dict[str, list[float]] = {
                "65-69": [], "70-74": [], "75-78": [], "79+": [],
            }
            for r in detailed:
                c = r.get("ai_confidence") or 0
                p = float(r.get("result_profit") or 0.0)
                if c < 70:
                    bands["65-69"].append(p)
                elif c < 75:
                    bands["70-74"].append(p)
                elif c < 79:
                    bands["75-78"].append(p)
                else:
                    bands["79+"].append(p)

            conf_dist: dict = {}
            for band, profits in bands.items():
                if profits:
                    wins = sum(1 for p in profits if p > 0)
                    conf_dist[band] = {
                        "count": len(profits),
                        "win_rate": round(wins / len(profits), 2),
                        "avg_profit": round(sum(profits) / len(profits), 1),
                    }
            if conf_dist:
                context["confidence_distribution"] = conf_dist
    except Exception as e:
        logger.debug("[Adaptive] conf_dist build failed: %s", e)

    # ── 4. 市場ボラティリティ (MT5 D1 ATR比) ──────
    # MT5未接続・分析スクリプト実行時はスキップ
    try:
        import mt5_connector  # noqa: PLC0415
        atr_data: dict = {}
        for symbol in config.SYMBOLS[:3]:  # 主要3銘柄のみ
            try:
                df = mt5_connector.get_rates(symbol, "D1", count=60)
                if df is not None and len(df) >= 37:
                    current_atr = mt5_connector.calculate_atr(df.tail(14), 7)
                    avg_atr = mt5_connector.calculate_atr_sma(df, 7, 30)
                    if avg_atr and avg_atr > 0:
                        atr_data[symbol] = round(current_atr / avg_atr, 2)
            except Exception:
                continue
        if atr_data:
            context["market_volatility"] = {
                "atr_ratio_by_symbol": atr_data,
                "note": "ratio > 1.2 = elevated volatility; losses may be volatility-driven",
            }
    except ImportError:
        pass

    # ── 5. 反実仮想分析（閾値別シミュレーション）──────
    # 直近 lookback の 2 倍の期間を使ってサンプルを増やす
    try:
        cf_lookback = max(config.ADAPTIVE_LOOKBACK_DAYS * 2, 30)
        cf = _build_counterfactual(cf_lookback)
        if cf:
            context["counterfactual_analysis"] = cf
    except Exception as e:
        logger.debug("[Adaptive] counterfactual build failed: %s", e)

    # ── 6. マクロ環境（Web検索: VIX / DXY / リスクイベント）──
    try:
        macro = _fetch_macro_context()
        if macro:
            context["macro_environment"] = macro
    except Exception as e:
        logger.debug("[Adaptive] macro context failed: %s", e)

    return context


def _llm_suggest_changes(
    params: dict,
    buckets: dict[str, dict],
    context: dict | None = None,
) -> tuple[list[dict], str | None]:
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
        "role": "quantitative_risk_manager",
        "task": (
            "Analyze the past week's trading performance and recommend "
            "confidence threshold adjustments. Primary goals: "
            "(1) maintain positive expectancy, "
            "(2) prevent entry starvation (target >=3 entries/week), "
            "(3) distinguish volatility-driven losses from strategy failures."
        ),
        "decision_rules": {
            "entry_starvation": (
                f"If starvation_warning=true (threshold at max AND <3 trades in lookback), "
                f"recommend global_delta of at least -{config.ADAPTIVE_CONF_STEP} to loosen."
            ),
            "high_volatility_losses": (
                "If any atr_ratio > 1.2, losses are likely volatility-driven. "
                "Do NOT tighten. Prefer delta=0 or small negative delta."
            ),
            "strategy_failure": (
                f"If win_rate < 30% AND all atr_ratios < 1.1 (normal market), "
                f"strategy is underperforming — tighten: delta=+{config.ADAPTIVE_CONF_STEP}."
            ),
            "healthy_recovery": (
                f"If win_rate > 50% AND expectancy > 0, "
                f"loosen cautiously: delta=-{config.ADAPTIVE_CONF_STEP}."
            ),
            "use_counterfactual": (
                "IMPORTANT: Use counterfactual_analysis.scenarios to find the threshold "
                "that maximizes avg_profit while keeping entries >= 3. "
                "Set global_delta to move current threshold toward best_threshold_by_score. "
                "Never ignore counterfactual data if it is available."
            ),
            "use_macro_environment": (
                "If macro_environment is available: "
                "(1) If market_sentiment=risk-off OR vix.trend=rising: do NOT loosen threshold — "
                "high volatility regime, protect capital. "
                "(2) If market_sentiment=risk-on AND vix.trend=falling: losses may recover — "
                "consider loosening if counterfactual supports it. "
                "(3) If risk_events_this_week has events with days_until <= 2: "
                "prefer delta=0 regardless of other signals (uncertainty too high)."
            ),
            "threshold_range": [config.ADAPTIVE_CONF_MIN, config.ADAPTIVE_CONF_MAX],
            "max_delta_per_cycle": config.ADAPTIVE_CONF_MAX_WEEKLY_DELTA,
            "skip_if_samples_below": config.ADAPTIVE_MIN_SAMPLES,
        },
        "performance_data": {
            "current_global_threshold": global_thr,
            "buckets": compact,
        },
        "market_context": context or {},
        "output_format": "JSON only. No markdown fences.",
        "output_schema": {
            "diagnosis": "str: 1-2 sentences explaining WHY you make these changes",
            "global_delta": "int (positive=tighten, negative=loosen, 0=no change)",
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
            thr = int(bucket.get("confidence_threshold", global_thr))
            # config.ADAPTIVE_CONF_MAX を上限として常にキャップ
            return min(thr, config.ADAPTIVE_CONF_MAX)
    # config範囲内に収める
    return min(global_thr, config.ADAPTIVE_CONF_MAX)


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
            adaptive_context = _build_adaptive_context(params, rows)
            changes, llm_rationale = _llm_suggest_changes(params, buckets, adaptive_context)
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
