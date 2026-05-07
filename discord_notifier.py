import logging
from datetime import datetime

import requests

import config

logger = logging.getLogger(__name__)


def _send(content: str, embeds: list[dict] | None = None):
    if not config.DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL が未設定です")
        return

    payload: dict = {}
    if content:
        payload["content"] = content[:2000]
    if embeds:
        payload["embeds"] = embeds[:10]

    try:
        resp = requests.post(
            config.DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=10,
        )
        if resp.status_code not in (200, 204):
            logger.error("Discord送信失敗: %s %s", resp.status_code, resp.text[:200])
    except requests.RequestException as e:
        logger.error("Discord送信エラー: %s", e)


# ── 通知種別 ────────────────────────────

def send_heartbeat(balance: float, equity: float, open_positions: int):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    embed = {
        "title": "💓 Heartbeat",
        "color": 0x00FF00,
        "fields": [
            {"name": "時刻", "value": now, "inline": True},
            {"name": "残高", "value": f"¥{balance:,.0f}", "inline": True},
            {"name": "有効証拠金", "value": f"¥{equity:,.0f}", "inline": True},
            {"name": "保有ポジション", "value": str(open_positions), "inline": True},
        ],
    }
    _send("", embeds=[embed])


def send_entry(symbol: str, direction: str, lot: float,
               entry_price: float, sl: float, tp: float, reasoning: str):
    color = 0x2ECC71 if direction == "BUY" else 0xE74C3C
    embed = {
        "title": f"📈 Entry: {symbol} {direction}",
        "color": color,
        "fields": [
            {"name": "ロット", "value": f"{lot:.2f}", "inline": True},
            {"name": "エントリー価格", "value": str(entry_price), "inline": True},
            {"name": "SL", "value": str(sl), "inline": True},
            {"name": "TP", "value": str(tp), "inline": True},
            {"name": "AI判断 (抜粋)", "value": reasoning[:500]},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    _send("", embeds=[embed])


def send_exit(symbol: str, direction: str, exit_price: float,
              profit: float, reasoning: str, source: str = "AI判断"):
    color = 0x2ECC71 if profit >= 0 else 0xE74C3C
    embed = {
        "title": f"📉 Exit: {symbol} {direction}",
        "color": color,
        "fields": [
            {"name": "決済価格", "value": str(exit_price), "inline": True},
            {"name": "損益", "value": f"¥{profit:,.0f}", "inline": True},
            {"name": f"{source} (抜粋)", "value": reasoning[:500]},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    _send("", embeds=[embed])


def send_error(title: str, detail: str):
    embed = {
        "title": f"🚨 Error: {title}",
        "color": 0xFF0000,
        "description": detail[:2000],
        "timestamp": datetime.utcnow().isoformat(),
    }
    _send("", embeds=[embed])


def send_skip(symbol: str, reason: str, notify: bool = False):
    logger.info("SKIP %s: %s", symbol, reason)
    if not notify:
        return

    embed = {
        "title": f"⏭️ Skip: {symbol}",
        "color": 0xF1C40F,
        "description": reason[:1000],
        "timestamp": datetime.utcnow().isoformat(),
    }
    _send("", embeds=[embed])
