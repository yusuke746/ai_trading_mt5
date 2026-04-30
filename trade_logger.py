import json
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from html import escape

import config

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _migrate_db(conn: sqlite3.Connection):
    """既存DBに不足カラムを追加するマイグレーション"""
    cur = conn.execute("PRAGMA table_info(trades)")
    existing = {row[1] for row in cur.fetchall()}
    new_cols = [
        ("exit_reason",    "TEXT"),
        ("smc_sweep_pass", "INTEGER"),
        ("smc_bos_pass",   "INTEGER"),
        ("smc_rr_pass",    "INTEGER"),
        ("ai_confidence",  "INTEGER"),
        ("ai_smc_sweep",   "INTEGER"),
        ("ai_smc_ob",      "INTEGER"),
        ("ai_smc_fvg",     "INTEGER"),
        ("entry_type",     "TEXT"),
        ("market_regime",  "TEXT"),
        ("invalidation_price", "REAL"),
    ]
    for col, typ in new_cols:
        if col not in existing:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                opened_at       TEXT    NOT NULL,
                closed_at       TEXT,
                symbol          TEXT    NOT NULL,
                direction       TEXT    NOT NULL,
                entry_price     REAL,
                exit_price      REAL,
                lot_size        REAL,
                sl_price        REAL,
                tp_price        REAL,
                ai_reasoning    TEXT,
                news_summary    TEXT,
                result_pips     REAL,
                result_profit   REAL,
                mt5_ticket      INTEGER,
                status          TEXT    DEFAULT 'OPEN',
                exit_reason     TEXT,
                smc_sweep_pass  INTEGER,
                smc_bos_pass    INTEGER,
                smc_rr_pass     INTEGER,
                ai_confidence   INTEGER,
                ai_smc_sweep    INTEGER,
                ai_smc_ob       INTEGER,
                ai_smc_fvg      INTEGER,
                entry_type      TEXT,
                market_regime   TEXT,
                invalidation_price REAL
            );

            CREATE TABLE IF NOT EXISTS ai_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                action_type     TEXT    NOT NULL,
                ai_response     TEXT,
                decision        TEXT,
                reasoning       TEXT
            );

            CREATE TABLE IF NOT EXISTS heartbeats (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                status          TEXT    NOT NULL,
                details         TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at);
            CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trades(mt5_ticket);
            CREATE INDEX IF NOT EXISTS idx_ai_logs_timestamp ON ai_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_heartbeats_timestamp ON heartbeats(timestamp);
        """)

        # WALサイズの無制限増加を防ぐため、定期的に自動チェックポイントを設定
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        # 既存DBへのマイグレーション
        _migrate_db(conn)
    logger.info("Database initialized: %s", config.DB_PATH)


# ── trades ──────────────────────────────

def insert_trade(symbol: str, direction: str, entry_price: float,
                 lot_size: float, sl_price: float, tp_price: float | None,
                 ai_reasoning: str, news_summary: str,
                 mt5_ticket: int,
                 smc_sweep_pass: bool | None = None,
                 smc_bos_pass: bool | None = None,
                 smc_rr_pass: bool | None = None,
                 ai_confidence: int | None = None,
                 ai_smc_sweep: bool | None = None,
                 ai_smc_ob: bool | None = None,
                 ai_smc_fvg: bool | None = None,
                 entry_type: str | None = None,
                 market_regime: str | None = None,
                 invalidation_price: float | None = None) -> int:
    def _b(v): return int(v) if v is not None else None
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (opened_at, symbol, direction, entry_price, lot_size,
                sl_price, tp_price, ai_reasoning, news_summary, mt5_ticket, status,
                smc_sweep_pass, smc_bos_pass, smc_rr_pass,
                     ai_confidence, ai_smc_sweep, ai_smc_ob, ai_smc_fvg, entry_type, market_regime, invalidation_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), symbol, direction,
             entry_price, lot_size, sl_price, tp_price,
             ai_reasoning, news_summary, mt5_ticket,
             _b(smc_sweep_pass), _b(smc_bos_pass), _b(smc_rr_pass),
                 ai_confidence, _b(ai_smc_sweep), _b(ai_smc_ob), _b(ai_smc_fvg), entry_type, market_regime,
                 invalidation_price),
        )
        return cur.lastrowid


def close_trade(trade_id: int, exit_price: float,
                result_pips: float, result_profit: float,
                exit_reason: str | None = None):
    with _get_conn() as conn:
        conn.execute(
            """UPDATE trades
               SET closed_at = ?, exit_price = ?, result_pips = ?,
                   result_profit = ?, status = 'CLOSED', exit_reason = ?
               WHERE id = ?""",
            (datetime.utcnow().isoformat(), exit_price,
             result_pips, result_profit, exit_reason, trade_id),
        )


def update_trade_closed_at(trade_id: int, closed_at: str):
    """close_tradeで設定したclosed_atをMT5履歴の実際の時刻で上書きする。"""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE trades SET closed_at = ? WHERE id = ?",
            (closed_at, trade_id),
        )


def get_open_trades() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'OPEN'"
        ).fetchall()
        return [dict(r) for r in rows]



def get_open_trade_by_ticket(mt5_ticket: int) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM trades WHERE mt5_ticket = ? AND status = 'OPEN'",
            (mt5_ticket,),
        ).fetchone()
        return dict(row) if row else None


def get_symbol_recent_loss_streak(symbol: str, lookback: int = 10) -> dict:
    """銘柄ごとの直近連敗数と最終クローズ時刻を返す。"""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT closed_at, result_profit
            FROM trades
            WHERE symbol = ?
              AND status = 'CLOSED'
              AND closed_at IS NOT NULL
              AND result_profit IS NOT NULL
            ORDER BY datetime(closed_at) DESC
            LIMIT ?
            """,
            (symbol, lookback),
        ).fetchall()

    streak = 0
    last_closed_at = None
    for idx, r in enumerate(rows):
        if idx == 0:
            last_closed_at = r["closed_at"]
        if (r["result_profit"] or 0) < 0:
            streak += 1
        else:
            break

    return {
        "loss_streak": streak,
        "last_closed_at": last_closed_at,
    }


def get_recent_premise_break_exit(symbol: str, direction: str) -> dict | None:
    """同一銘柄・同方向の直近PREMISE_BREAK決済を返す。"""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT closed_at, exit_reason, direction
            FROM trades
            WHERE symbol = ?
              AND direction = ?
              AND status = 'CLOSED'
              AND exit_reason = 'PREMISE_BREAK'
              AND closed_at IS NOT NULL
            ORDER BY datetime(closed_at) DESC
            LIMIT 1
            """,
            (symbol, direction),
        ).fetchone()
    return dict(row) if row else None


def get_recent_closed_trade(symbol: str) -> dict | None:
    """同一銘柄の直近クローズ済みトレードを返す。"""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT closed_at, exit_reason, result_profit, direction
            FROM trades
            WHERE symbol = ?
              AND status = 'CLOSED'
              AND closed_at IS NOT NULL
            ORDER BY datetime(closed_at) DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
    return dict(row) if row else None


def get_recent_winning_closed_trade(symbol: str) -> dict | None:
    """同一銘柄の直近勝ちトレード (result_profit>0) を返す。"""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT closed_at, exit_reason, result_profit, direction
            FROM trades
            WHERE symbol = ?
              AND status = 'CLOSED'
              AND closed_at IS NOT NULL
              AND result_profit > 0
            ORDER BY datetime(closed_at) DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
    return dict(row) if row else None


def update_trade_sl_by_ticket(mt5_ticket: int, sl_price: float):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE trades SET sl_price = ? WHERE mt5_ticket = ? AND status = 'OPEN'",
            (sl_price, mt5_ticket),
        )


def fetch_recent_closed(lookback_days: int) -> list[dict]:
    """直近 lookback_days 日の CLOSED トレードを返す (アダプティブ学習用)。"""
    since = _iso_days_ago(lookback_days)
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT market_regime, entry_type, result_profit
            FROM trades
            WHERE status = 'CLOSED'
              AND closed_at IS NOT NULL
              AND result_profit IS NOT NULL
              AND datetime(closed_at) >= datetime(?)
            ORDER BY closed_at
            """,
            (since,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── AI logs ─────────────────────────────

def insert_ai_log(symbol: str, action_type: str, ai_response: str,
                  decision: str, reasoning: str):
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO ai_logs
               (timestamp, symbol, action_type, ai_response, decision, reasoning)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), symbol, action_type,
             ai_response, decision, reasoning),
        )


# ── Heartbeat ───────────────────────────

def insert_heartbeat(status: str, details: str = ""):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO heartbeats (timestamp, status, details) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(), status, details),
        )


# ── DB Maintenance ─────────────────────

def run_maintenance(full_vacuum: bool = False) -> dict:
    """SQLiteメンテナンスを実行して、削除件数などを返す。"""
    stats = {
        "deleted_ai_logs": 0,
        "deleted_heartbeats": 0,
        "deleted_closed_trades": 0,
        "trimmed_ai_logs": 0,
        "trimmed_heartbeats": 0,
        "db_size_mb_before": _db_size_mb(),
        "db_size_mb_after": 0.0,
        "vacuum_executed": full_vacuum,
    }

    ai_cutoff = _iso_days_ago(config.DB_RETENTION_DAYS_AI_LOGS)
    hb_cutoff = _iso_days_ago(config.DB_RETENTION_DAYS_HEARTBEATS)
    trade_cutoff = _iso_days_ago(config.DB_RETENTION_DAYS_CLOSED_TRADES)

    try:
        with _get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM ai_logs WHERE timestamp < ?",
                (ai_cutoff,),
            )
            stats["deleted_ai_logs"] = cur.rowcount

            cur = conn.execute(
                "DELETE FROM heartbeats WHERE timestamp < ?",
                (hb_cutoff,),
            )
            stats["deleted_heartbeats"] = cur.rowcount

            cur = conn.execute(
                """DELETE FROM trades
                   WHERE status = 'CLOSED' AND closed_at IS NOT NULL AND closed_at < ?""",
                (trade_cutoff,),
            )
            stats["deleted_closed_trades"] = cur.rowcount

            stats["trimmed_ai_logs"] = _trim_table_to_max_rows(
                conn, "ai_logs", "id", config.DB_MAX_AI_LOG_ROWS,
            )
            stats["trimmed_heartbeats"] = _trim_table_to_max_rows(
                conn, "heartbeats", "id", config.DB_MAX_HEARTBEAT_ROWS,
            )
    except sqlite3.OperationalError as e:
        # 稼働中のロック競合は想定内。次回メンテ周期で再試行する。
        logger.warning("DB maintenance skipped due to lock: %s", e)
        stats["vacuum_executed"] = False

    try:
        with _get_conn() as conn:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.execute("PRAGMA optimize")
        if full_vacuum or stats["deleted_ai_logs"] or stats["deleted_heartbeats"] or stats["deleted_closed_trades"]:
            with _get_conn() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if full_vacuum:
            with _get_conn() as conn:
                conn.execute("VACUUM")
    except sqlite3.OperationalError as e:
        logger.warning("DB post-maintenance optimize/checkpoint skipped due to lock: %s", e)
        stats["vacuum_executed"] = False

    stats["db_size_mb_after"] = _db_size_mb()
    logger.info(
        "DB maintenance: ai_logs=%d, heartbeats=%d, closed_trades=%d, trimmed_ai=%d, trimmed_hb=%d, "
        "size=%.2fMB -> %.2fMB, vacuum=%s",
        stats["deleted_ai_logs"],
        stats["deleted_heartbeats"],
        stats["deleted_closed_trades"],
        stats["trimmed_ai_logs"],
        stats["trimmed_heartbeats"],
        stats["db_size_mb_before"],
        stats["db_size_mb_after"],
        full_vacuum,
    )
    return stats


def build_regime_dashboard(lookback_days: int | None = None) -> dict:
    """レジーム別期待値ダッシュボード(JSON)を生成する。"""
    lookback_days = lookback_days or config.DASHBOARD_LOOKBACK_DAYS
    generated_at = datetime.utcnow().isoformat()
    dashboard = {
        "generated_at": generated_at,
        "lookback_days": lookback_days,
        "db_path": config.DB_PATH,
    }
    since = _iso_days_ago(lookback_days)

    with _get_conn() as conn:
        dashboard["overview"] = dict(conn.execute(
            """
            SELECT
                COUNT(*) AS trades,
                ROUND(AVG(result_profit), 2) AS expectancy,
                ROUND(100.0 * AVG(CASE WHEN result_profit > 0 THEN 1.0 ELSE 0.0 END), 2) AS win_rate_pct,
                ROUND(SUM(result_profit), 2) AS total_profit,
                ROUND(AVG(ai_confidence), 2) AS avg_ai_confidence
            FROM trades
            WHERE status = 'CLOSED'
              AND closed_at IS NOT NULL
              AND result_profit IS NOT NULL
              AND datetime(closed_at) >= datetime(?)
            """,
            (since,),
        ).fetchone())

        dashboard["by_market_regime"] = _fetch_dashboard_rows(conn, since, "COALESCE(market_regime, 'UNKNOWN')")
        dashboard["by_entry_type"] = _fetch_dashboard_rows(conn, since, "COALESCE(entry_type, 'UNKNOWN')")
        dashboard["by_exit_reason"] = _fetch_dashboard_rows(conn, since, "COALESCE(exit_reason, 'UNKNOWN')")
        dashboard["by_symbol"] = _fetch_dashboard_rows(conn, since, "symbol")
        dashboard["by_regime_entry_type"] = _fetch_dashboard_rows(
            conn,
            since,
            "COALESCE(market_regime, 'UNKNOWN') || ' / ' || COALESCE(entry_type, 'UNKNOWN')",
        )

    output_path = os.path.join(config.ANALYTICS_DIR, "regime_expectancy_dashboard.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2)

    html_output_path = os.path.join(config.ANALYTICS_DIR, "regime_expectancy_dashboard.html")
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(_render_dashboard_html(dashboard))

    logger.info("Regime dashboard generated: %s", output_path)
    dashboard["output_path"] = output_path
    dashboard["html_output_path"] = html_output_path
    return dashboard


def _fetch_dashboard_rows(conn: sqlite3.Connection, since: str, group_expr: str) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            {group_expr} AS bucket,
            COUNT(*) AS trades,
            ROUND(AVG(result_profit), 2) AS expectancy,
            ROUND(100.0 * AVG(CASE WHEN result_profit > 0 THEN 1.0 ELSE 0.0 END), 2) AS win_rate_pct,
            ROUND(AVG(CASE WHEN result_profit > 0 THEN result_profit END), 2) AS avg_win,
            ROUND(AVG(CASE WHEN result_profit < 0 THEN result_profit END), 2) AS avg_loss,
            ROUND(SUM(result_profit), 2) AS total_profit,
            ROUND(AVG(ai_confidence), 2) AS avg_ai_confidence,
            SUM(CASE WHEN COALESCE(ai_smc_sweep, 0) = 1 THEN 1 ELSE 0 END) AS sweep_trades,
            SUM(CASE WHEN COALESCE(ai_smc_ob, 0) = 1 THEN 1 ELSE 0 END) AS ob_trades,
            SUM(CASE WHEN COALESCE(ai_smc_fvg, 0) = 1 THEN 1 ELSE 0 END) AS fvg_trades,
            SUM(CASE
                WHEN tp_price IS NOT NULL AND direction='BUY' AND exit_price >= tp_price THEN 1
                WHEN tp_price IS NOT NULL AND direction='SELL' AND exit_price <= tp_price THEN 1
                ELSE 0
            END) AS tp_hits,
            SUM(CASE
                WHEN sl_price IS NOT NULL AND direction='BUY' AND exit_price <= sl_price THEN 1
                WHEN sl_price IS NOT NULL AND direction='SELL' AND exit_price >= sl_price THEN 1
                ELSE 0
            END) AS sl_hits
        FROM trades
        WHERE status = 'CLOSED'
          AND closed_at IS NOT NULL
          AND result_profit IS NOT NULL
          AND datetime(closed_at) >= datetime(?)
        GROUP BY bucket
        HAVING COUNT(*) > 0
        ORDER BY trades DESC, bucket ASC
        """,
        (since,),
    ).fetchall()
    return [dict(row) for row in rows]


def _trim_table_to_max_rows(
    conn: sqlite3.Connection,
    table_name: str,
    order_column: str,
    max_rows: int,
) -> int:
    if max_rows <= 0:
        return 0
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    overflow = count - max_rows
    if overflow <= 0:
        return 0
    conn.execute(
        f"DELETE FROM {table_name} WHERE {order_column} IN (SELECT {order_column} FROM {table_name} ORDER BY {order_column} ASC LIMIT ?)",
        (overflow,),
    )
    return overflow


def _render_dashboard_html(dashboard: dict) -> str:
        overview = dashboard.get("overview", {})
        sections = [
                ("Market Regime", dashboard.get("by_market_regime", [])),
                ("Entry Type", dashboard.get("by_entry_type", [])),
                ("Regime x Entry Type", dashboard.get("by_regime_entry_type", [])),
                ("Exit Reason", dashboard.get("by_exit_reason", [])),
                ("Symbol", dashboard.get("by_symbol", [])),
        ]

        def metric_card(label: str, value) -> str:
                return (
                        "<div class=\"metric-card\">"
                        f"<div class=\"metric-label\">{escape(label)}</div>"
                        f"<div class=\"metric-value\">{escape(_fmt_metric(value))}</div>"
                        "</div>"
                )

        cards_html = "".join([
                metric_card("Closed Trades", overview.get("trades")),
                metric_card("Expectancy", overview.get("expectancy")),
                metric_card("Win Rate", _fmt_pct(overview.get("win_rate_pct"))),
                metric_card("Total Profit", overview.get("total_profit")),
                metric_card("Avg AI Confidence", _fmt_pct(overview.get("avg_ai_confidence"))),
        ])

        sections_html = "".join(
                _render_dashboard_section(title, rows)
                for title, rows in sections
        )

        generated_at = escape(str(dashboard.get("generated_at", "")))
        lookback_days = escape(str(dashboard.get("lookback_days", "")))

        return f"""<!DOCTYPE html>
<html lang=\"ja\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Regime Expectancy Dashboard</title>
    <style>
        :root {{
            --bg: #f4efe6;
            --panel: #fffaf2;
            --panel-strong: #f8e6c8;
            --ink: #1f1a14;
            --muted: #6f6254;
            --line: #ddc9ad;
            --accent: #a64b2a;
            --accent-soft: #d97757;
            --good: #17633d;
            --bad: #8c2d1f;
            --shadow: 0 18px 40px rgba(85, 47, 24, 0.12);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Georgia, "Yu Mincho", "Hiragino Mincho ProN", serif;
            background:
                radial-gradient(circle at top left, rgba(217, 119, 87, 0.22), transparent 28%),
                radial-gradient(circle at top right, rgba(166, 75, 42, 0.18), transparent 24%),
                linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
            color: var(--ink);
        }}
        .shell {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 40px 24px 56px;
        }}
        .hero {{
            background: linear-gradient(135deg, rgba(255,250,242,0.92), rgba(248,230,200,0.86));
            border: 1px solid rgba(166, 75, 42, 0.18);
            border-radius: 28px;
            padding: 28px;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }}
        .hero::after {{
            content: "";
            position: absolute;
            inset: auto -80px -120px auto;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(166, 75, 42, 0.14), transparent 70%);
            transform: rotate(18deg);
        }}
        h1 {{
            margin: 0 0 8px;
            font-size: clamp(30px, 5vw, 54px);
            line-height: 1;
            letter-spacing: -0.03em;
        }}
        .sub {{
            margin: 0;
            color: var(--muted);
            font-size: 15px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 24px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 16px 18px;
            backdrop-filter: blur(10px);
        }}
        .metric-label {{
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .metric-value {{
            margin-top: 6px;
            font-size: 30px;
            font-weight: 700;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            margin-top: 24px;
        }}
        .panel {{
            background: rgba(255, 250, 242, 0.85);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }}
        .panel-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 18px 20px;
            background: linear-gradient(180deg, rgba(248,230,200,0.8), rgba(255,250,242,0.65));
            border-bottom: 1px solid var(--line);
        }}
        .panel-title {{
            margin: 0;
            font-size: 20px;
        }}
        .panel-body {{ padding: 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{
            padding: 12px 14px;
            text-align: right;
            border-bottom: 1px solid rgba(221, 201, 173, 0.7);
            font-size: 14px;
            white-space: nowrap;
        }}
        th:first-child, td:first-child {{ text-align: left; }}
        th {{
            position: sticky;
            top: 0;
            background: rgba(248,230,200,0.9);
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 11px;
        }}
        tr:hover td {{ background: rgba(248,230,200,0.22); }}
        .positive {{ color: var(--good); font-weight: 700; }}
        .negative {{ color: var(--bad); font-weight: 700; }}
        .empty {{ padding: 24px 20px; color: var(--muted); }}
        .footer {{ margin-top: 18px; color: var(--muted); font-size: 13px; }}
        @media (max-width: 900px) {{
            .shell {{ padding: 20px 14px 40px; }}
            .hero, .panel {{ border-radius: 20px; }}
            .panel-body {{ overflow-x: auto; }}
            th, td {{ font-size: 13px; }}
        }}
    </style>
</head>
<body>
    <div class=\"shell\">
        <section class=\"hero\">
            <h1>Regime Expectancy Dashboard</h1>
            <p class=\"sub\">Generated at {generated_at} UTC / Lookback {lookback_days} days</p>
            <div class=\"metrics\">{cards_html}</div>
        </section>
        <section class=\"grid\">{sections_html}</section>
        <div class=\"footer\">This HTML is self-contained. It does not fetch external JSON at runtime.</div>
    </div>
</body>
</html>
"""


def _render_dashboard_section(title: str, rows: list[dict]) -> str:
        if not rows:
                return (
                        "<section class=\"panel\">"
                        f"<div class=\"panel-head\"><h2 class=\"panel-title\">{escape(title)}</h2></div>"
                        "<div class=\"empty\">No closed trades in this bucket.</div>"
                        "</section>"
                )

        headers = [
                ("bucket", "Bucket"),
                ("trades", "Trades"),
                ("expectancy", "Expectancy"),
                ("win_rate_pct", "Win Rate"),
                ("avg_win", "Avg Win"),
                ("avg_loss", "Avg Loss"),
                ("total_profit", "Total Profit"),
                ("avg_ai_confidence", "AI Conf"),
                ("tp_hits", "TP Hits"),
                ("sl_hits", "SL Hits"),
                ("sweep_trades", "Sweep"),
                ("ob_trades", "OB"),
                ("fvg_trades", "FVG"),
        ]
        head_html = "".join(f"<th>{escape(label)}</th>" for _, label in headers)
        rows_html = "".join(_render_dashboard_row(row, headers) for row in rows)
        return (
                "<section class=\"panel\">"
                f"<div class=\"panel-head\"><h2 class=\"panel-title\">{escape(title)}</h2><div class=\"sub\">{len(rows)} buckets</div></div>"
                "<div class=\"panel-body\">"
                f"<table><thead><tr>{head_html}</tr></thead><tbody>{rows_html}</tbody></table>"
                "</div></section>"
        )


def _render_dashboard_row(row: dict, headers: list[tuple[str, str]]) -> str:
        cells = []
        for key, _ in headers:
                value = row.get(key)
                text = _fmt_cell(key, value)
                css_class = ""
                if key in {"expectancy", "avg_win", "avg_loss", "total_profit"} and isinstance(value, (int, float)):
                        if value > 0:
                                css_class = ' class=\"positive\"'
                        elif value < 0:
                                css_class = ' class=\"negative\"'
                cells.append(f"<td{css_class}>{escape(text)}</td>")
        return f"<tr>{''.join(cells)}</tr>"


def _fmt_cell(key: str, value) -> str:
        if key in {"win_rate_pct", "avg_ai_confidence"}:
                return _fmt_pct(value)
        return _fmt_metric(value)


def _fmt_pct(value) -> str:
        if value is None:
                return "-"
        return f"{float(value):.2f}%"


def _fmt_metric(value) -> str:
        if value is None:
                return "-"
        if isinstance(value, float):
                return f"{value:,.2f}"
        if isinstance(value, int):
                return f"{value:,}"
        return str(value)


def _iso_days_ago(days: int) -> str:
    return (datetime.utcnow() - timedelta(days=days)).isoformat()


def _db_size_mb() -> float:
    total = 0
    db_files = [
        config.DB_PATH,
        config.DB_PATH + "-wal",
        config.DB_PATH + "-shm",
    ]
    for path in db_files:
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total / (1024 * 1024)
