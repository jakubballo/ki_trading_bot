"""
logger_db.py – SQLite-Datenbank für Trade-Logging.
Speichert alle Trades, Signale, Fehler und Funding-Charges.
"""

import sqlite3
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

DB_PATH = Path(config.paths["db_file"])


def get_connection() -> sqlite3.Connection:
    """Erstellt eine neue Datenbankverbindung."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # WAL-Modus für bessere Concurrent-Performance
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Erstellt alle Tabellen falls noch nicht vorhanden."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Trades-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                qty REAL,
                sl REAL,
                tp REAL,
                pnl_usdt REAL,
                pnl_pct REAL,
                fees_usdt REAL,
                funding_paid_usdt REAL DEFAULT 0.0,
                regime TEXT,
                score INTEGER,
                hold_duration_hours REAL,
                exit_reason TEXT,
                opened_at TEXT,
                closed_at TEXT
            )
        """)

        # Signale-Tabelle (für Layer 3 Scoring-Ergebnisse)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                symbol TEXT NOT NULL,
                score INTEGER,
                direction TEXT,
                regime TEXT,
                macro_direction TEXT,
                atr REAL,
                atr_ratio REAL,
                funding_rate REAL,
                fg_index REAL,
                action TEXT,
                reject_reason TEXT
            )
        """)

        # Fehler-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                module TEXT,
                error_type TEXT,
                message TEXT,
                traceback TEXT
            )
        """)

        # Funding-Charges-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS funding_charges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                symbol TEXT NOT NULL,
                funding_rate REAL,
                position_size REAL,
                charge_usdt REAL
            )
        """)

        conn.commit()
        logger.info(f"Datenbank initialisiert: {DB_PATH}")

    except Exception as e:
        logger.error(f"Fehler beim Initialisieren der Datenbank: {e}")
        raise
    finally:
        conn.close()


def log_trade_opened(symbol: str, side: str, entry_price: float, qty: float,
                     sl: float, tp: float, regime: str, score: int,
                     opened_at: str = None) -> int:
    """Erstellt einen neuen Trade-Eintrag beim Öffnen."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (symbol, side, entry_price, qty, sl, tp, regime, score, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, side, entry_price, qty, sl, tp, regime, score,
            opened_at or datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
        trade_id = cursor.lastrowid
        logger.debug(f"Trade geloggt (ID: {trade_id}): {side} {symbol} @ {entry_price}")
        return trade_id
    except Exception as e:
        logger.error(f"Fehler beim Loggen des Trades: {e}")
        return -1
    finally:
        conn.close()


def log_trade_closed(trade_id: int, exit_price: float, pnl_usdt: float,
                     pnl_pct: float, fees_usdt: float, funding_paid_usdt: float,
                     hold_duration_hours: float, exit_reason: str,
                     closed_at: str = None):
    """Aktualisiert einen Trade-Eintrag beim Schließen."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE trades SET
                exit_price = ?,
                pnl_usdt = ?,
                pnl_pct = ?,
                fees_usdt = ?,
                funding_paid_usdt = ?,
                hold_duration_hours = ?,
                exit_reason = ?,
                closed_at = ?
            WHERE id = ?
        """, (
            exit_price, pnl_usdt, pnl_pct, fees_usdt, funding_paid_usdt,
            hold_duration_hours, exit_reason,
            closed_at or datetime.now(timezone.utc).isoformat(),
            trade_id
        ))
        conn.commit()
        logger.debug(f"Trade geschlossen (ID: {trade_id}): PnL {pnl_usdt:.2f} USDT ({exit_reason})")
    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren des Trades: {e}")
    finally:
        conn.close()


def log_signal(symbol: str, score: int, direction: str, regime: str,
               macro_direction: str, atr: float, atr_ratio: float,
               funding_rate: float, fg_index: float,
               action: str, reject_reason: str = None):
    """Loggt ein Scoring-Signal."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO signals (timestamp_utc, symbol, score, direction, regime,
                               macro_direction, atr, atr_ratio, funding_rate,
                               fg_index, action, reject_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            symbol, score, direction, regime, macro_direction,
            atr, atr_ratio, funding_rate, fg_index, action, reject_reason
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"Fehler beim Loggen des Signals: {e}")
    finally:
        conn.close()


def log_error(module: str, error_type: str, message: str, traceback_str: str = None):
    """Loggt einen Fehler in die Datenbank."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO errors (timestamp_utc, module, error_type, message, traceback)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            module, error_type, message, traceback_str
        ))
        conn.commit()
    except Exception as e:
        # Fehler beim Fehler-Loggen – nur in Logger schreiben
        logger.error(f"Fehler beim Loggen des Fehlers in DB: {e}")
    finally:
        conn.close()


def log_funding_charge(symbol: str, funding_rate: float,
                       position_size: float, charge_usdt: float):
    """Loggt eine Funding-Charge."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO funding_charges (timestamp_utc, symbol, funding_rate, position_size, charge_usdt)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            symbol, funding_rate, position_size, charge_usdt
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"Fehler beim Loggen der Funding-Charge: {e}")
    finally:
        conn.close()


def get_recent_trades(limit: int = 10) -> list:
    """Gibt die letzten N Trades zurück."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Trades: {e}")
        return []
    finally:
        conn.close()
