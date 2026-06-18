"""
bossbot_db.py – Eigene SQLite-Datenbank des BossBots.

Bewusst GETRENNT von network.db: der BossBot schreibt seine eigenen Trades
NIE in die 50-Bot-Datenbank, damit
  (1) das Bot-Ranking (get_all_bot_rankings) nicht sich selbst abguckt und
  (2) Modell B nicht auf BossBot-Trades mittrainiert.
Er liest network.db nur lesend (für das Ranking).

Datei: data/bossbot_trades.db (WAL). Eine Zeile pro BossBot-Trade.
Beim Öffnen wird die Zeile angelegt (exit_price NULL), beim Schließen ergänzt.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("bossbot.db")

_DB_PATH = Path("data/bossbot_trades.db")


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bossbot_trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source_bot_id INTEGER,
                symbol        TEXT,
                side          TEXT,
                entry         REAL,
                exit_price    REAL,
                qty           REAL,
                leverage      REAL,
                margin        REAL,
                sl_price      REAL,
                tp_price      REAL,
                pnl           REAL,
                pnl_pct       REAL,
                fees          REAL,
                exit_reason   TEXT,
                mode          TEXT,
                opened_at     TEXT,
                closed_at     TEXT
            )
        """)
        conn.commit()
        logger.info(f"BossBot-DB initialisiert: {_DB_PATH}")
    finally:
        conn.close()


def insert_open(*, source_bot_id: int, symbol: str, side: str, entry: float,
                qty: float, leverage: float, margin: float,
                sl_price: float, tp_price: float, mode: str, opened_at: str) -> int:
    """Legt die Trade-Zeile beim Öffnen an (exit_price NULL). Gibt die id zurück."""
    conn = _conn()
    try:
        cur = conn.execute("""
            INSERT INTO bossbot_trades
                (source_bot_id, symbol, side, entry, qty, leverage, margin,
                 sl_price, tp_price, mode, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (source_bot_id, symbol, side, entry, qty, leverage, margin,
              sl_price, tp_price, mode, opened_at))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def update_close(trade_id: int, *, exit_price: float, pnl: float, pnl_pct: float,
                 fees: float, exit_reason: str, closed_at: str):
    """Ergänzt die Zeile beim Schließen."""
    conn = _conn()
    try:
        conn.execute("""
            UPDATE bossbot_trades
            SET exit_price=?, pnl=?, pnl_pct=?, fees=?, exit_reason=?, closed_at=?
            WHERE id=?
        """, (exit_price, pnl, pnl_pct, fees, exit_reason, closed_at, trade_id))
        conn.commit()
    finally:
        conn.close()


def get_summary() -> dict:
    """Aggregierte Statistik über alle geschlossenen BossBot-Trades."""
    conn = _conn()
    try:
        row = conn.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(pnl) AS total_pnl
            FROM bossbot_trades
            WHERE exit_price IS NOT NULL
        """).fetchone()
        total = row["total"] or 0
        wins  = row["wins"] or 0
        return {
            "total":     total,
            "wins":      wins,
            "win_rate":  (wins / total) if total else 0.0,
            "total_pnl": row["total_pnl"] or 0.0,
        }
    finally:
        conn.close()
