"""ニュースモニター

2種類のリスクをバックグラウンドで管理する:

① 経済カレンダー (ForexFactory RSS)
   - 定期的に高インパクト指標を取得・キャッシュ
   - 指標発表の前後 EVENT_BLOCK_MINUTES 分以内はエントリー禁止

② 突発地政学ニュース (Finnhub REST + ローカルキーワードフィルタ + gpt-5-nano)
   コスト最適化フロー:
     Step1: Finnhub から直近ニュースヘッドラインを取得 (無料API)
     Step2: ローカルキーワードスコアリングで危険度を事前判定
     Step3: スコアが閾値超えた場合のみ gpt-5-nano でリスク確認
   → 通常時は AI 呼び出しゼロ、突発リスク時のみ AI 判断

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

# OpenAI クライアント (nano AI判断用)
_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


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


# ── Finnhub ニュース + ローカルキーワードフィルタ ──────────

# 銘柄 → Finnhub の category/symbol マッピング
_FINNHUB_SYMBOL_MAP: dict[str, str] = {
    "GOLD":      "OANDA:XAUUSD",
    "USDJPY":    "OANDA:USD_JPY",
    "EURUSD":    "OANDA:EUR_USD",
    "US100Cash": "NASDAQ:QQQ",
    "OILCash":   "OANDA:USOIL",
}

# HIGH判定キーワード (スコア+3)
_HIGH_KEYWORDS = [
    "war", "invasion", "attack", "military", "strike", "nuclear",
    "sanction", "embargo", "default", "collapse", "crash", "emergency",
    "rate hike", "rate cut", "surprise", "unexpected", "shock",
    "ceasefire", "escalat", "threat", "missil", "explosion",
    "opec cut", "opec+ cut", "oil supply", "strait of hormuz",
    "bank run", "contagion", "crisis",
]

# MEDIUM判定キーワード (スコア+1)
_MEDIUM_KEYWORDS = [
    "inflation", "recession", "slowdown", "tariff", "trade war",
    "fed", "fomc", "boj", "ecb", "central bank",
    "gdp", "cpi", "ppi", "nfp", "unemployment",
    "geopolit", "tension", "conflict", "protest", "election",
    "oil", "crude", "gold", "yen", "dollar", "euro",
]

_FINNHUB_TIMEOUT_SEC = 10


def _fetch_finnhub_headlines(symbol: str, hours_back: int = 6) -> list[str]:
    """Finnhub REST API から直近ニュースヘッドラインを取得する。"""
    import json as _json
    api_key = config.FINNHUB_API_KEY
    if not api_key:
        return []

    finnhub_sym = _FINNHUB_SYMBOL_MAP.get(symbol)
    if not finnhub_sym:
        return []

    from_dt = datetime.now(UTC) - timedelta(hours=hours_back)
    to_dt = datetime.now(UTC)
    from_str = from_dt.strftime("%Y-%m-%d")
    to_str = to_dt.strftime("%Y-%m-%d")

    # Forex/commodity は general news category でフォールバック
    urls_to_try: list[str] = []
    if ":" in finnhub_sym:
        # symbol-specific news
        encoded = finnhub_sym.replace(":", "%3A")
        urls_to_try.append(
            f"https://finnhub.io/api/v1/company-news?symbol={encoded}"
            f"&from={from_str}&to={to_str}&token={api_key}"
        )
    # general forex news
    urls_to_try.append(
        f"https://finnhub.io/api/v1/news?category=forex&token={api_key}"
    )

    for url in urls_to_try:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=_FINNHUB_TIMEOUT_SEC) as resp:
                data = _json.loads(resp.read())
            if isinstance(data, list) and data:
                headlines = [item.get("headline", "") for item in data[:20] if item.get("headline")]
                if headlines:
                    return headlines
        except Exception:
            continue

    return []


def _score_headlines(headlines: list[str]) -> tuple[int, list[str]]:
    """ヘッドラインをキーワードスコアリングしてスコアとヒットキーワードを返す。"""
    score = 0
    hits: list[str] = []
    text = " ".join(headlines).lower()

    for kw in _HIGH_KEYWORDS:
        if kw in text:
            score += 3
            hits.append(kw)

    for kw in _MEDIUM_KEYWORDS:
        if kw in text:
            score += 1
            hits.append(kw)

    return score, hits


def _fetch_news_for_symbols() -> list[dict]:
    """Finnhub + ローカルフィルタ + (必要時のみ) gpt-5-nano でニュースリスクを判断。"""
    results: list[dict] = []
    expires_at = datetime.now(UTC) + timedelta(minutes=config.NEWS_CACHE_EXPIRE_MINUTES)
    ai_threshold = config.NEWS_AI_ESCALATION_SCORE  # このスコア以上のみAI判断

    for sym in config.SYMBOLS:
        headlines = _fetch_finnhub_headlines(sym)
        score, hits = _score_headlines(headlines)

        if score == 0 or not headlines:
            results.append({
                "symbol": sym,
                "risk_level": "LOW",
                "summary": "Finnhub: 注目ニュースなし",
                "expires_at": expires_at,
            })
            continue

        # スコアが閾値未満はローカル判定のみ (AI呼び出しなし)
        if score < ai_threshold:
            risk = "MEDIUM" if score >= 3 else "LOW"
            results.append({
                "symbol": sym,
                "risk_level": risk,
                "summary": f"Finnhub: キーワード検出 [{', '.join(hits[:3])}] score={score}",
                "expires_at": expires_at,
            })
            continue

        # スコアが閾値以上 → gpt-5-nano で詳細判断
        logger.info(
            "[NewsMonitor] %s: Finnhubスコア高(%d) → AI詳細判断 hits=%s",
            sym, score, hits[:5],
        )
        ai_result = _ask_nano_for_risk(sym, headlines[:10])
        results.append({**ai_result, "expires_at": expires_at})

    return results


def _ask_nano_for_risk(symbol: str, headlines: list[str]) -> dict:
    """gpt-5-nano にヘッドラインを渡してリスク判定させる (web検索なし・安価)。"""
    if not config.OPENAI_API_KEY:
        return {
            "symbol": symbol,
            "risk_level": "MEDIUM",
            "summary": "AI判定スキップ (APIキーなし)",
        }

    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""以下のニュースヘッドラインを読み、{symbol}へのリスクを判定してください。

【ヘッドライン】
{headlines_text}

リスク判定:
- HIGH: 価格に大きな影響を与える重大ニュース (戦争・制裁・中銀緊急声明・市場クラッシュなど)
- MEDIUM: 注意が必要だが直接的影響は限定的
- LOW: 特筆すべきリスクなし

JSONのみで回答:
{{"symbol": "{symbol}", "risk_level": "HIGH"|"MEDIUM"|"LOW", "summary": "1文で要約"}}"""

    try:
        client = _get_openai_client()
        # web_search_preview なし → トークンコスト最小
        response = client.responses.create(
            model=config.NEWS_MONITOR_MODEL,
            input=[{"role": "user", "content": prompt}],
        )
        raw = response.output_text
        import re as _re, json as _json
        match = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if match:
            data = _json.loads(match.group())
            risk = str(data.get("risk_level", "MEDIUM")).upper()
            if risk not in {"LOW", "MEDIUM", "HIGH"}:
                risk = "MEDIUM"
            return {
                "symbol": symbol,
                "risk_level": risk,
                "summary": str(data.get("summary", raw[:200])),
            }
    except Exception as e:
        logger.warning("[NewsMonitor] AI詳細判断失敗 %s: %s", symbol, e)

    return {
        "symbol": symbol,
        "risk_level": "MEDIUM",
        "summary": "AI判定失敗 → 安全側でMEDIUM",
    }


def _update_news_cache() -> None:
    global _news_cache, _news_last_updated
    if not config.NEWS_MONITOR_ENABLED:
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

            # 土曜・日曜はAPIポーリングをスキップ (コスト削減)
            if datetime.now().weekday() in (5, 6):
                continue

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
