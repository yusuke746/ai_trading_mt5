import sqlite3
import logging
import os
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
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
                entry_type      TEXT
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
                 entry_type: str | None = None) -> int:
    def _b(v): return int(v) if v is not None else None
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (opened_at, symbol, direction, entry_price, lot_size,
                sl_price, tp_price, ai_reasoning, news_summary, mt5_ticket, status,
                smc_sweep_pass, smc_bos_pass, smc_rr_pass,
                ai_confidence, ai_smc_sweep, ai_smc_ob, ai_smc_fvg, entry_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), symbol, direction,
             entry_price, lot_size, sl_price, tp_price,
             ai_reasoning, news_summary, mt5_ticket,
             _b(smc_sweep_pass), _b(smc_bos_pass), _b(smc_rr_pass),
             ai_confidence, _b(ai_smc_sweep), _b(ai_smc_ob), _b(ai_smc_fvg), entry_type),
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


def update_trade_sl_by_ticket(mt5_ticket: int, sl_price: float):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE trades SET sl_price = ? WHERE mt5_ticket = ? AND status = 'OPEN'",
            (sl_price, mt5_ticket),
        )


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

            # WALファイル肥大化対策
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.execute("PRAGMA optimize")

            if full_vacuum:
                conn.execute("VACUUM")
    except sqlite3.OperationalError as e:
        # 稼働中のロック競合は想定内。次回メンテ周期で再試行する。
        logger.warning("DB maintenance skipped due to lock: %s", e)
        stats["vacuum_executed"] = False

    stats["db_size_mb_after"] = _db_size_mb()
    logger.info(
        "DB maintenance: ai_logs=%d, heartbeats=%d, closed_trades=%d, "
        "size=%.2fMB -> %.2fMB, vacuum=%s",
        stats["deleted_ai_logs"],
        stats["deleted_heartbeats"],
        stats["deleted_closed_trades"],
        stats["db_size_mb_before"],
        stats["db_size_mb_after"],
        full_vacuum,
    )
    return stats


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
