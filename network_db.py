"""
network_db.py – Zentrale SQLite Outcome-Datenbank für das Bot-Netzwerk.
WAL-Modus: mehrere Bots können gleichzeitig schreiben.
Tabellen: trades_network, pbt_history, reflection_rules
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _get_db_path() -> Path:
    from config import config
    path = Path(config.paths.get("network_db", "data/network.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_DB_INITIALIZED = False


def get_connection() -> sqlite3.Connection:
    global _DB_INITIALIZED
    conn = sqlite3.connect(str(_get_db_path()), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    if not _DB_INITIALIZED:
        _DB_INITIALIZED = True
        _bootstrap(conn)
    return conn


def _bootstrap(conn: sqlite3.Connection):
    """Erstellt Tabellen (falls nötig) und führt Schema-Migrationen aus. Einmalig pro Prozess."""
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades_network (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id           INTEGER NOT NULL,
                symbol           TEXT NOT NULL,
                side             TEXT NOT NULL,
                entry            REAL NOT NULL,
                exit_price       REAL,
                pnl              REAL,
                exit_reason      TEXT,
                score            INTEGER,
                regime           TEXT,
                funding_rate     REAL DEFAULT 0,
                rsi              REAL DEFAULT 50,
                atr              REAL DEFAULT 0,
                fg_index         REAL DEFAULT 50,
                strategy         TEXT DEFAULT 'momentum',
                is_shadow        INTEGER DEFAULT 0,
                is_synthetic     INTEGER DEFAULT 0,
                block_reason     TEXT,
                is_veto          INTEGER DEFAULT 0,
                config_snapshot  TEXT,
                opened_at        TEXT,
                closed_at        TEXT,
                weight           REAL DEFAULT 1.0
            );
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades_network(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_bot    ON trades_network(bot_id);
            CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades_network(closed_at);
            CREATE TABLE IF NOT EXISTS pbt_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                worst_bot_id INTEGER,
                best_bot_id  INTEGER,
                mutation     TEXT,
                new_config   TEXT
            );
            CREATE TABLE IF NOT EXISTS reflection_rules (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   TEXT NOT NULL,
                rule_text    TEXT NOT NULL,
                basis_trades INTEGER DEFAULT 0,
                active       INTEGER DEFAULT 1
            );
        """)
        # Neue Win-Modell-Spalten (non-destruktiv, ADD COLUMN ignoriert bestehende)
        new_cols = [
            ("macd_diff",        "REAL DEFAULT 0"),
            ("macd_signal_val",  "REAL DEFAULT 0"),
            ("ema_ratio_9_21",   "REAL DEFAULT 0"),
            ("ema_ratio_21_50",  "REAL DEFAULT 0"),
            ("price_vs_ema50",   "REAL DEFAULT 0"),
            ("bb_pct",           "REAL DEFAULT 0.5"),
            ("bb_width",         "REAL DEFAULT 0"),
            ("vol_ratio",        "REAL DEFAULT 1.0"),
            ("rsi_slope",        "REAL DEFAULT 0"),
            ("ret_1",            "REAL DEFAULT 0"),
            ("ret_4",            "REAL DEFAULT 0"),
            ("ret_8",            "REAL DEFAULT 0"),
            ("ret_16",           "REAL DEFAULT 0"),
        ]
        existing = {row[1] for row in conn.execute("PRAGMA table_info(trades_network)").fetchall()}
        added = []
        for col, defn in new_cols:
            if col not in existing:
                conn.execute(f"ALTER TABLE trades_network ADD COLUMN {col} {defn}")
                added.append(col)
        conn.commit()
        if added:
            logger.info(f"DB-Migration: neue Spalten hinzugefügt: {added}")
    except Exception as e:
        logger.error(f"DB-Bootstrap Fehler: {e}")


def init_network_db():
    """Erstellt alle Netzwerk-Tabellen (wird von _bootstrap übernommen)."""
    conn = get_connection()
    conn.close()
    logger.info(f"Network-DB initialisiert: {_get_db_path()}")


def log_network_trade(
    bot_id: int,
    symbol: str,
    side: str,
    entry: float,
    exit_price: Optional[float],
    pnl: Optional[float],
    exit_reason: Optional[str],
    score: int = 0,
    regime: str = "ranging",
    funding_rate: float = 0.0,
    rsi: float = 50.0,
    atr: float = 0.0,
    fg_index: float = 50.0,
    strategy: str = "momentum",
    is_shadow: bool = False,
    is_synthetic: bool = False,
    block_reason: str = None,
    is_veto: bool = False,
    config_snapshot: dict = None,
    # Neue Marktstruktur-Features (Problem 2)
    macd_diff: float = 0.0,
    macd_signal_val: float = 0.0,
    ema_ratio_9_21: float = 0.0,
    ema_ratio_21_50: float = 0.0,
    price_vs_ema50: float = 0.0,
    bb_pct: float = 0.5,
    bb_width: float = 0.0,
    vol_ratio: float = 1.0,
    rsi_slope: float = 0.0,
    ret_1: float = 0.0,
    ret_4: float = 0.0,
    ret_8: float = 0.0,
    ret_16: float = 0.0,
    opened_at: Optional[str] = None,
) -> int:
    """Schreibt einen Trade (real/shadow/synthetic) in die zentrale DB."""
    from config import config as cfg
    weight = (
        cfg.ml.get("weight_real", 1.0)       if not is_shadow and not is_synthetic
        else cfg.ml.get("weight_shadow", 0.5) if is_shadow
        else cfg.ml.get("weight_synthetic", 0.2)
    )

    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO trades_network (
                bot_id, symbol, side, entry, exit_price, pnl, exit_reason,
                score, regime, funding_rate, rsi, atr, fg_index, strategy,
                is_shadow, is_synthetic, block_reason, is_veto,
                config_snapshot, opened_at, closed_at, weight,
                macd_diff, macd_signal_val, ema_ratio_9_21, ema_ratio_21_50, price_vs_ema50,
                bb_pct, bb_width, vol_ratio, rsi_slope, ret_1, ret_4, ret_8, ret_16
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            bot_id, symbol, side, entry, exit_price, pnl, exit_reason,
            score, regime, funding_rate, rsi, atr, fg_index, strategy,
            int(is_shadow), int(is_synthetic), block_reason, int(is_veto),
            json.dumps(config_snapshot) if config_snapshot else None,
            # S6-1: echte Eröffnungszeit verwenden, falls übergeben; sonst jetzt
            # (Shadows/Synthetik geben keine → behalten bisheriges Verhalten).
            opened_at or datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat() if exit_price is not None else None,
            weight,
            macd_diff, macd_signal_val, ema_ratio_9_21, ema_ratio_21_50, price_vs_ema50,
            bb_pct, bb_width, vol_ratio, rsi_slope, ret_1, ret_4, ret_8, ret_16,
        ))
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Network-Trade-Log Fehler: {e}")
        return -1
    finally:
        conn.close()


def update_network_trade_outcome(trade_id: int, exit_price: float, pnl: float,
                                  exit_reason: str):
    """Aktualisiert Outcome eines offenen Shadow/Veto-Trades."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE trades_network
            SET exit_price=?, pnl=?, exit_reason=?, closed_at=?
            WHERE id=?
        """, (exit_price, pnl, exit_reason,
              datetime.now(timezone.utc).isoformat(), trade_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Trade-Update Fehler: {e}")
    finally:
        conn.close()


def get_bot_stats(bot_id: int, min_trades: int = 0, since: str = None) -> dict:
    """
    Gibt Statistiken für einen Bot zurück (nur echte Trades).
    since: optionaler ISO-Zeitstempel – nur Trades ab diesem Zeitpunkt (H5-Fix für PBT).
    """
    conn = get_connection()
    try:
        where_since = ""
        params: list = [bot_id]
        if since:
            where_since = " AND closed_at >= ?"
            params.append(since)

        row = conn.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MIN(pnl) as min_pnl,
                MAX(pnl) as max_pnl
            FROM trades_network
            WHERE bot_id=? AND is_shadow=0 AND is_synthetic=0
              AND exit_price IS NOT NULL AND pnl IS NOT NULL {where_since}
        """, params).fetchone()

        if not row or row["total"] < min_trades:
            return {}

        total = row["total"] or 1
        wins  = row["wins"] or 0
        return {
            "bot_id":     bot_id,
            "total":      total,
            "win_rate":   wins / total,
            "total_pnl":  row["total_pnl"] or 0,
            "avg_pnl":    row["avg_pnl"] or 0,
            "sharpe":     _calc_sharpe(bot_id, conn, since=since),
        }
    finally:
        conn.close()


def _calc_sharpe(bot_id: int, conn: sqlite3.Connection, since: str = None) -> float:
    """Berechnet Sharpe-Ratio aus den PnL-Werten (optional ab `since`)."""
    where_since = ""
    params: list = [bot_id]
    if since:
        where_since = " AND closed_at >= ?"
        params.append(since)
    rows = conn.execute(f"""
        SELECT pnl FROM trades_network
        WHERE bot_id=? AND is_shadow=0 AND is_synthetic=0
          AND pnl IS NOT NULL {where_since}
        ORDER BY closed_at
    """, params).fetchall()
    pnls = [r["pnl"] for r in rows]
    if len(pnls) < 5:
        return 0.0
    avg = sum(pnls) / len(pnls)
    std = (sum((p - avg) ** 2 for p in pnls) / len(pnls)) ** 0.5
    return (avg / std) if std > 0 else 0.0


def get_all_bot_rankings(min_trades: int = 20, since: str = None) -> List[dict]:
    """
    Gibt alle Bots nach Sharpe-Ratio sortiert zurück.
    since: optionaler ISO-Zeitstempel – nur Trades ab diesem Zeitpunkt (H5-Fix für PBT).
    """
    conn = get_connection()
    try:
        bot_ids = [r["bot_id"] for r in
                   conn.execute("SELECT DISTINCT bot_id FROM trades_network").fetchall()]
    finally:
        conn.close()

    rankings = []
    for bid in bot_ids:
        stats = get_bot_stats(bid, min_trades, since=since)
        if stats:
            rankings.append(stats)
    return sorted(rankings, key=lambda x: x["sharpe"], reverse=True)


def get_network_summary(since: str = None) -> dict:
    """
    Aggregierte Netzwerk-Statistik für den Shutdown-Report.
    since: optionaler ISO-Zeitstempel – nur Trades ab diesem Zeitpunkt zählen.
    Gibt zurück: echte Trades, Gewinne, Verluste, PnL, blockierte Signale (Shadow/Veto).
    """
    conn = get_connection()
    try:
        where_since = ""
        params: list = []
        # S6-3: realisierte (geschlossene) Trades nach closed_at filtern, nicht
        # opened_at. Ein in diesem Lauf geschlossener Trade muss zählen, auch wenn
        # er in einem früheren Lauf eröffnet wurde (carry-over via state.json). Mit
        # opened_at-Filter fiel z.B. ein +13.46-TP-Win aus dem Bericht (PnL 0.00).
        where_closed = ""
        params_closed: list = []
        if since:
            where_since = " AND opened_at >= ?"
            params = [since]
            where_closed = " AND closed_at >= ?"
            params_closed = [since]

        real = conn.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl
            FROM trades_network
            WHERE is_shadow=0 AND is_synthetic=0
              AND exit_price IS NOT NULL AND pnl IS NOT NULL {where_closed}
        """, params_closed).fetchone()

        # Geschlossene echte Trades pro Bot (Lauf-Fenster)
        closed_by_bot = {
            r["bot_id"]: r["cnt"] for r in conn.execute(f"""
                SELECT bot_id, COUNT(*) as cnt FROM trades_network
                WHERE is_shadow=0 AND is_synthetic=0
                  AND exit_price IS NOT NULL AND pnl IS NOT NULL {where_closed}
                GROUP BY bot_id ORDER BY cnt DESC
            """, params_closed).fetchall()
        }

        # S7-Fix: Offene echte Trades sind ein AKTUELLER Snapshot – NICHT nach
        # opened_at filtern (sonst verschwinden Carry-over-Positionen aus früheren
        # Läufen, wie im 12:45-Mikro-Lauf → "offen: 0" trotz 4 offener Positionen).
        open_real = conn.execute("""
            SELECT COUNT(*) as cnt FROM trades_network
            WHERE is_shadow=0 AND is_synthetic=0 AND exit_price IS NULL
        """).fetchone()
        open_by_bot = {
            r["bot_id"]: r["cnt"] for r in conn.execute("""
                SELECT bot_id, COUNT(*) as cnt FROM trades_network
                WHERE is_shadow=0 AND is_synthetic=0 AND exit_price IS NULL
                GROUP BY bot_id ORDER BY cnt DESC
            """).fetchall()
        }

        blocked = conn.execute(f"""
            SELECT COUNT(*) as cnt FROM trades_network
            WHERE (is_shadow=1 OR is_veto=1) {where_since}
        """, params).fetchone()

        # Shadow-Trades: in diesem Lauf erzeugt + aktuell noch offen (Snapshot)
        shadow_run = conn.execute(f"""
            SELECT COUNT(*) as cnt FROM trades_network
            WHERE is_shadow=1 {where_since}
        """, params).fetchone()
        shadow_open = conn.execute("""
            SELECT COUNT(*) as cnt FROM trades_network
            WHERE is_shadow=1 AND exit_price IS NULL
        """).fetchone()

        return {
            "real_trades":   real["total"] or 0,
            "wins":          real["wins"] or 0,
            "losses":        real["losses"] or 0,
            "total_pnl":     real["total_pnl"] or 0.0,
            "open":          open_real["cnt"] or 0,
            "closed_by_bot": closed_by_bot,
            "open_by_bot":   open_by_bot,
            "blocked":       blocked["cnt"] or 0,
            "shadow_run":    shadow_run["cnt"] or 0,
            "shadow_open":   shadow_open["cnt"] or 0,
        }
    except Exception as e:
        logger.error(f"Network-Summary Fehler: {e}")
        return {}
    finally:
        conn.close()


def get_open_counts() -> dict:
    """Aktuell offene Trades (Snapshot) für den Start-Report: echte + Shadow."""
    conn = get_connection()
    try:
        real_open = conn.execute(
            "SELECT COUNT(*) FROM trades_network "
            "WHERE is_shadow=0 AND is_synthetic=0 AND exit_price IS NULL"
        ).fetchone()[0]
        shadow_open = conn.execute(
            "SELECT COUNT(*) FROM trades_network "
            "WHERE is_shadow=1 AND exit_price IS NULL"
        ).fetchone()[0]
        return {"real_open": real_open or 0, "shadow_open": shadow_open or 0}
    except Exception as e:
        logger.error(f"get_open_counts Fehler: {e}")
        return {"real_open": 0, "shadow_open": 0}
    finally:
        conn.close()


def get_training_data(symbol: str = None, limit: int = 10_000) -> List[dict]:
    """Holt Trainingsdaten für ML (alle Quellen, gewichtet)."""
    conn = get_connection()
    try:
        query = """
            SELECT * FROM trades_network
            WHERE exit_price IS NOT NULL AND pnl IS NOT NULL
        """
        params: list = []
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
        query += " ORDER BY closed_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_training_data_balanced(per_symbol: int = 8000) -> List[dict]:
    """
    B2 — Symbol-balanciertes Sampling: zieht je Symbol bis zu `per_symbol` neueste
    Outcomes (closed_at DESC) und kombiniert sie. Verhindert die Symbol-Schieflage
    des globalen LIMIT (sonst dominierte ein Symbol die Trainingsmenge → andere
    Symbol-Modelle fielen mangels Zeilen auf das Base-Modell zurück).
    """
    conn = get_connection()
    try:
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM trades_network "
            "WHERE exit_price IS NOT NULL AND pnl IS NOT NULL"
        ).fetchall()]
        rows: List[dict] = []
        for sym in symbols:
            sym_rows = conn.execute(
                "SELECT * FROM trades_network "
                "WHERE exit_price IS NOT NULL AND pnl IS NOT NULL AND symbol=? "
                "ORDER BY closed_at DESC LIMIT ?",
                (sym, per_symbol),
            ).fetchall()
            rows.extend(dict(r) for r in sym_rows)
        return rows
    finally:
        conn.close()


def log_pbt_event(worst_bot_id: int, best_bot_id: int,
                  mutation: dict, new_config: dict):
    """Loggt ein PBT-Selektions-Event."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO pbt_history (timestamp, worst_bot_id, best_bot_id, mutation, new_config)
            VALUES (?,?,?,?,?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            worst_bot_id, best_bot_id,
            json.dumps(mutation), json.dumps(new_config),
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"PBT-Log Fehler: {e}")
    finally:
        conn.close()


def log_reflection_rule(rule_text: str, basis_trades: int):
    """Speichert eine LLM-generierte Regel."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO reflection_rules (created_at, rule_text, basis_trades)
            VALUES (?,?,?)
        """, (datetime.now(timezone.utc).isoformat(), rule_text, basis_trades))
        conn.commit()
    except Exception as e:
        logger.error(f"Reflection-Rule Fehler: {e}")
    finally:
        conn.close()


def get_active_trades_summary() -> dict:
    """Gibt offene echte Trades und Shadow-Trades zurück (für Dashboard-Telegram-Report)."""
    conn = get_connection()
    try:
        real_open = conn.execute("""
            SELECT bot_id, symbol, side, entry, opened_at, score
            FROM trades_network
            WHERE is_shadow=0 AND is_synthetic=0 AND exit_price IS NULL
            ORDER BY opened_at
        """).fetchall()

        shadow_open = conn.execute("""
            SELECT bot_id, symbol, side, opened_at
            FROM trades_network
            WHERE is_shadow=1 AND exit_price IS NULL
            ORDER BY bot_id
        """).fetchall()

        today_start = (
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )
        today_stats = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as pnl
            FROM trades_network
            WHERE is_shadow=0 AND is_synthetic=0
              AND exit_price IS NOT NULL AND closed_at >= ?
        """, (today_start,)).fetchone()

        return {
            "real_open":   [dict(r) for r in real_open],
            "shadow_open": [dict(r) for r in shadow_open],
            "today_total": today_stats["total"] or 0,
            "today_wins":  today_stats["wins"]  or 0,
            "today_pnl":   today_stats["pnl"]   or 0.0,
        }
    except Exception as e:
        logger.error(f"Active-Trades-Summary Fehler: {e}")
        return {"real_open": [], "shadow_open": [], "today_total": 0, "today_wins": 0, "today_pnl": 0.0}
    finally:
        conn.close()


def count_new_outcomes_since(last_count: int) -> int:
    """Gibt die Anzahl neuer geschlossener Trades seit letztem Retrain zurück."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT COUNT(*) as cnt FROM trades_network
            WHERE exit_price IS NOT NULL AND id > ?
        """, (last_count,)).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def get_max_trade_id() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(id) as mid FROM trades_network").fetchone()
        return row["mid"] or 0
    finally:
        conn.close()
