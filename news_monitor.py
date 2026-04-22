"""ニュースモニター

2種類のリスクをバックグラウンドで管理する:

① 経済カレンダー (ForexFactory RSS)
   - 30分おきに高インパクト指標を取得・キャッシュ
   - 指標発表の前後 EVENT_BLOCK_MINUTES 分以内はエントリー禁止

② 突発地政学ニュース (gpt-5-nano + web_search_preview)
   - 30分おきに主要銘柄に影響するニュースを確認
   - risk_level が HIGH の銘柄はエントリー禁止

利用側は check_entry_news_block(symbol) を呼ぶだけ。
スレッドは start_background_monitor() で起動する。
"""

import json
import logging
import threading
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.request import urlopen, Request

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# ── キャッシュ (スレッド間共有) ────────────────────────

_lock = threading.Lock()

# ForexFactory RSSから取得した今後のイベントリスト
# 各要素: {"title": str, "currency": str, "impact": str, "date": datetime}
_calendar_events: list[dict] = []
_calendar_last_updated: datetime | None = None

# nanoニュース判断キャッシュ
# 各要素: {"symbol": str, "risk_level": "LOW"|"MEDIUM"|"HIGH", "summary": str, "expires_at": datetime}
_news_cache: list[dict] = []
_news_last_updated: datetime | None = None

# ── 設定マッピング ─────────────────────────────────────

# 各銘柄に関連する通貨コード (ForexFactory RSSのcurrencyと照合)
_SYMBOL_CURRENCIES: dict[str, list[str]] = {
    "GOLD":      ["USD", "XAU"],
    "USDJPY":    ["USD", "JPY"],
    "EURUSD":    ["USD", "EUR"],
    "US100Cash": ["USD"],
    "OILCash":   ["USD", "OIL"],
}

# nanoに渡す銘柄別のニュース確認キーワード
_SYMBOL_TOPICS: dict[str, str] = {
    "GOLD":      "gold, XAU, USD strength, geopolitical risk",
    "USDJPY":    "USD/JPY, Fed, BOJ, US-Japan trade",
    "EURUSD":    "EUR/USD, ECB, Fed, European politics",
    "US100Cash": "NASDAQ, US tech stocks, Fed rate",
    "OILCash":   "crude oil, WTI, OPEC, Middle East, geopolitical",
}


# ── ForexFactory カレンダー ──────────────────────────────

_FF_RSS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
_FF_TIMEOUT_SEC = 15


def _fetch_calendar_events() -> list[dict]:
    """ForexFactory RSS を取得してHigh-impactイベントを返す。
    取得失敗時は空リストを返す（例外は上位でキャッチ）。
    """
    req = Request(_FF_RSS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=_FF_TIMEOUT_SEC) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    events: list[dict] = []

    for item in root.iter("event"):
        impact = (item.findtext("impact") or "").strip()
        if impact.lower() != "high":
            continue

        title = (item.findtext("title") or "").strip()
        currency = (item.findtext("country") or "").strip().upper()
        date_str = (item.findtext("date") or "").strip()
        time_str = (item.findtext("time") or "").strip()

        if not date_str:
            continue

        # 日付パース: "04-22-2026" と "2:30pm" を結合
        try:
            event_dt = datetime.strptime(
                f"{date_str} {time_str}", "%m-%d-%Y %I:%M%p"
            ).replace(tzinfo=UTC)
        except ValueError:
            try:
                event_dt = datetime.strptime(date_str, "%m-%d-%Y").replace(tzinfo=UTC)
            except ValueError:
                continue

        events.append({
            "title": title,
            "currency": currency,
            "impact": impact,
            "date": event_dt,
        })

    return events


def _update_calendar() -> None:
    global _calendar_events, _calendar_last_updated
    try:
        events = _fetch_calendar_events()
        with _lock:
            _calendar_events = events
            _calendar_last_updated = datetime.now(UTC)
        logger.info("[NewsMonitor] カレンダー更新: %d件のHighインパクトイベント", len(events))
    except Exception as e:
        logger.warning("[NewsMonitor] カレンダー取得失敗 (無視して継続): %s", e)


def _is_calendar_blocked(symbol: str) -> tuple[bool, str]:
    """経済カレンダーで直近±EVENT_BLOCK_MINUTESに高インパクト指標があればブロック。"""
    block_min = config.NEWS_EVENT_BLOCK_MINUTES
    currencies = _SYMBOL_CURRENCIES.get(symbol, [])
    now_utc = datetime.now(UTC)

    with _lock:
        events = list(_calendar_events)

    for ev in events:
        if ev["currency"] not in currencies:
            continue
        diff_min = (ev["date"] - now_utc).total_seconds() / 60
        if -block_min <= diff_min <= block_min:
            return True, (
                f"経済指標ブロック: [{ev['currency']}] {ev['title']} "
                f"({diff_min:+.0f}分, impact={ev['impact']})"
            )
    return False, ""


# ── nano ニュースポーリング ─────────────────────────────

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _fetch_news_for_symbols() -> list[dict]:
    """gpt-5-nano + web_search_preview で主要銘柄のニュースリスクを確認。"""
    symbols_to_check = list(config.SYMBOLS)
    topics_list = "\n".join(
        f"- {sym}: {_SYMBOL_TOPICS.get(sym, sym)}"
        for sym in symbols_to_check
    )

    prompt = f"""あなたはFXトレーダーのリスク管理AIです。
現時点（{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}）における
以下の各銘柄のリスクをweb検索で確認し、JSONで返してください。

【確認対象銘柄とキーワード】
{topics_list}

【判定基準】
- HIGH: 価格に大きな影響を与える可能性のある重大ニュース（地政学紛争・戦争勃発/終結・制裁・OPEC緊急会合・中央銀行緊急声明など）
- MEDIUM: 注意が必要だが直接的影響は限定的なニュース
- LOW: 特筆すべきニュースなし

【回答フォーマット（JSONのみ）】
{{
  "results": [
    {{
      "symbol": "GOLD",
      "risk_level": "LOW" | "MEDIUM" | "HIGH",
      "summary": "確認したニュースの要約（1〜2文）"
    }},
    ...
  ]
}}

必ずJSONのみで返答してください。"""

    client = _get_openai_client()
    response = client.responses.create(
        model=config.NEWS_MONITOR_MODEL,
        input=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search_preview"}],
    )

    raw = response.output_text
    return _parse_news_response(raw, symbols_to_check)


def _parse_news_response(raw: str, symbols: list[str]) -> list[dict]:
    """APIレスポンスを安全にパースする。失敗時は全銘柄LOWを返す。"""
    import re

    data: Any = None
    # 直接パース
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # ```json ... ``` を試行
    if data is None:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # { ... } を抽出
    if data is None:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

    results: list[dict] = []
    expires_at = datetime.now(UTC) + timedelta(minutes=config.NEWS_CACHE_EXPIRE_MINUTES)

    if isinstance(data, dict):
        for item in data.get("results", []):
            sym = str(item.get("symbol", "")).strip()
            risk = str(item.get("risk_level", "LOW")).upper()
            if risk not in {"LOW", "MEDIUM", "HIGH"}:
                risk = "LOW"
            results.append({
                "symbol": sym,
                "risk_level": risk,
                "summary": str(item.get("summary", "")),
                "expires_at": expires_at,
            })

    # パース失敗や未登場のsymbolはLOWで補完
    found_symbols = {r["symbol"] for r in results}
    for sym in symbols:
        if sym not in found_symbols:
            results.append({
                "symbol": sym,
                "risk_level": "LOW",
                "summary": "取得失敗 or 該当なし",
                "expires_at": expires_at,
            })

    return results


def _update_news_cache() -> None:
    global _news_cache, _news_last_updated
    if not config.NEWS_MONITOR_ENABLED:
        return
    if not config.OPENAI_API_KEY:
        logger.warning("[NewsMonitor] OPENAI_API_KEY未設定 → ニュース取得スキップ")
        return
    try:
        results = _fetch_news_for_symbols()
        with _lock:
            _news_cache = results
            _news_last_updated = datetime.now(UTC)
        high_syms = [r["symbol"] for r in results if r["risk_level"] == "HIGH"]
        logger.info(
            "[NewsMonitor] ニュースキャッシュ更新: %d銘柄 (HIGH=%s)",
            len(results),
            high_syms or "なし",
        )
    except Exception as e:
        logger.warning("[NewsMonitor] ニュース取得失敗 (無視して継続): %s", e)


def _is_news_blocked(symbol: str) -> tuple[bool, str]:
    """nanoキャッシュで対象銘柄のリスクがHIGH/MEDIUM(設定による)ならブロック。"""
    now_utc = datetime.now(UTC)
    with _lock:
        cache = list(_news_cache)

    for item in cache:
        if item["symbol"] != symbol:
            continue
        if item["expires_at"] < now_utc:
            # 期限切れ → 非ブロック（次回更新を待つ）
            return False, ""
        risk = item["risk_level"]
        if risk == "HIGH":
            return True, f"重大ニュースリスク({risk}): {item['summary']}"
        if risk == "MEDIUM" and config.NEWS_BLOCK_ON_MEDIUM:
            return True, f"中程度ニュースリスク({risk}): {item['summary']}"
    return False, ""


# ── 公開API ────────────────────────────────────────────

def check_entry_news_block(symbol: str) -> tuple[bool, str]:
    """エントリー前チェック。ブロックすべきなら (True, 理由) を返す。

    呼び出し側はAPI呼び出しなし・純粋なキャッシュ参照のみ。
    """
    # ① 経済カレンダーチェック
    blocked, reason = _is_calendar_blocked(symbol)
    if blocked:
        return True, reason

    # ② ニュースリスクチェック
    blocked, reason = _is_news_blocked(symbol)
    if blocked:
        return True, reason

    return False, ""


def get_news_summary(symbol: str) -> str:
    """最新のニュースサマリーを文字列で返す（ログ・通知用）。"""
    with _lock:
        cache = list(_news_cache)
    for item in cache:
        if item["symbol"] == symbol:
            return f"[{item['risk_level']}] {item['summary']}"
    return "[UNKNOWN] キャッシュなし"


def start_background_monitor() -> None:
    """バックグラウンドスレッドを起動する。main()から1回だけ呼ぶ。"""
    if not config.NEWS_MONITOR_ENABLED and not config.NEWS_CALENDAR_ENABLED:
        logger.info("[NewsMonitor] カレンダー・ニュース監視ともに無効 → スレッド起動なし")
        return

    # 起動直後に1回即時取得
    if config.NEWS_CALENDAR_ENABLED:
        _update_calendar()
    if config.NEWS_MONITOR_ENABLED:
        _update_news_cache()

    def _loop():
        calendar_interval = config.NEWS_CALENDAR_INTERVAL_MINUTES * 60
        news_interval = config.NEWS_MONITOR_INTERVAL_MINUTES * 60
        last_calendar = time.monotonic()
        last_news = time.monotonic()

        while True:
            time.sleep(30)  # 30秒おきに時間チェック
            now = time.monotonic()

            if config.NEWS_CALENDAR_ENABLED and now - last_calendar >= calendar_interval:
                _update_calendar()
                last_calendar = now

            if config.NEWS_MONITOR_ENABLED and now - last_news >= news_interval:
                _update_news_cache()
                last_news = now

    t = threading.Thread(target=_loop, name="NewsMonitor", daemon=True)
    t.start()
    logger.info("[NewsMonitor] バックグラウンドスレッド起動")
