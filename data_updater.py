"""
data_updater.py – Holt frische 15m-Kerzen von Kraken Charts API.
Hängt neue Kerzen an bestehende CSVs an (inkrementell, idempotent).

Quelle: https://futures.kraken.com/api/charts/v1/mark/{symbol}/15m
Kraken liefert ~2000 Kerzen pro Abruf → paginieren für längere Historien.

Aufruf:
  python data_updater.py                    # alle Symbole aktualisieren
  python data_updater.py --symbol PF_XBTUSD # nur BTC
  python data_updater.py --full             # vollständige Historie (paginiert)
"""

import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("data_updater")

HISTORY_DIR  = Path("data/history")
KRAKEN_CHART = "https://futures.kraken.com/api/charts/v1/mark/{symbol}/15m"
RATE_LIMIT   = 1.5  # Sekunden zwischen Requests

SYMBOLS = [
    "PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD",
]


def fetch_klines(symbol: str, from_ts: Optional[int] = None) -> List[dict]:
    """
    Holt Kerzen von Kraken Charts API.
    from_ts: Unix-Timestamp in Millisekunden (optional, für inkrementelles Update).
    """
    url    = KRAKEN_CHART.format(symbol=symbol)
    params = {"resolution": "15m"}
    if from_ts:
        params["from"] = from_ts // 1000  # Kraken erwartet Sekunden

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        candles = data.get("candles", [])
        logger.debug(f"  {symbol}: {len(candles)} Kerzen erhalten")
        return candles
    except Exception as e:
        logger.error(f"Fetch Fehler [{symbol}]: {e}")
        return []


def parse_candles(candles: list) -> List[tuple]:
    """Konvertiert Kraken-Kerzen in CSV-Zeilen (timestamp_ms, o, h, l, c, v)."""
    rows = []
    for c in candles:
        try:
            ts = int(c["time"])  # Kraken: Zeit in ms
            row = (ts,
                   float(c["open"]),
                   float(c["high"]),
                   float(c["low"]),
                   float(c["close"]),
                   float(c.get("volume", 0)))
            rows.append(row)
        except Exception:
            pass
    return sorted(rows, key=lambda x: x[0])  # chronologisch sortieren


def load_csv_last_ts(path: Path) -> Optional[int]:
    """Liest letzten Timestamp aus bestehender CSV (für inkrementelles Update)."""
    if not path.exists():
        return None
    try:
        last_ts = None
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Header
            for row in reader:
                if row and row[0].replace(".", "").isdigit():
                    last_ts = int(float(row[0]))
        return last_ts
    except Exception:
        return None


def write_csv(path: Path, new_rows: List[tuple], append: bool = True):
    """Schreibt/hängt neue Zeilen an CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode   = "a" if (append and path.exists()) else "w"
    header = not (append and path.exists())

    with open(path, mode, newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(new_rows)


def _paginate_history(symbol: str, start_ms: int) -> List[tuple]:
    """
    Paginiert die Kraken Charts API ab start_ms bis zur Gegenwart.
    Kraken liefert max ~2000 Kerzen pro Abruf → mehrere Seiten nötig.
    Gibt chronologisch sortierte, dedupte Zeilen zurück.
    """
    all_rows: List[tuple] = []
    seen = set()
    cursor = start_ms
    page = 0
    MAX_PAGES = 400  # Sicherheitsobergrenze (~800k Kerzen)

    while page < MAX_PAGES:
        candles = fetch_klines(symbol, from_ts=cursor)
        if not candles:
            break
        rows = parse_candles(candles)
        new = [r for r in rows if r[0] not in seen]
        for r in new:
            seen.add(r[0])
            all_rows.append(r)
        page += 1

        if not new:
            break

        last_ts  = max(r[0] for r in new)
        first_dt = datetime.fromtimestamp(min(r[0] for r in new) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        last_dt  = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        logger.info(f"  {symbol}: Seite {page} +{len(new)} Kerzen ({first_dt} bis {last_dt})")

        if len(candles) < 2000:      # letzte (unvollständige) Seite → Live-Rand erreicht
            break
        cursor = last_ts + 1000      # 1s weiter (Kraken rückt eine Sekunde vor; Dedup fängt Überlappung)
        time.sleep(RATE_LIMIT)

    all_rows.sort(key=lambda x: x[0])
    return all_rows


def update_symbol(symbol: str, full: bool = False) -> int:
    """Aktualisiert CSV für ein Symbol (paginiert). Gibt Anzahl neuer Kerzen zurück."""
    path    = HISTORY_DIR / f"{symbol}_15m.csv"
    last_ts = load_csv_last_ts(path) if not full else None

    if last_ts and not full:
        logger.info(f"{symbol}: inkrementell ab {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
        start_ms = last_ts + 1000
        append   = True
    else:
        logger.info(f"{symbol}: vollständige Historie (paginiert, so weit Kraken zurückreicht)")
        # 2021-01-01 → Kraken clamped automatisch auf das früheste verfügbare Datum
        start_ms = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        append   = False

    rows = _paginate_history(symbol, start_ms)

    # Bei inkrementell nur echt neue Kerzen behalten
    if last_ts and not full:
        rows = [r for r in rows if r[0] > last_ts]

    if not rows:
        logger.info(f"  {symbol}: keine neuen Kerzen")
        return 0

    write_csv(path, rows, append=append)
    logger.info(f"  {symbol}: +{len(rows)} Kerzen → {path}")
    return len(rows)


def run_update(symbols: List[str] = None, full: bool = False):
    """Aktualisiert alle angegebenen Symbole."""
    targets = symbols or SYMBOLS
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for sym in targets:
        n = update_symbol(sym, full=full)
        total += n
        time.sleep(RATE_LIMIT)  # Rate-Limit beachten

    logger.info(f"Daten-Update abgeschlossen: {total} neue Kerzen für {len(targets)} Symbole")

    # Nach Update: ML Candle-Modelle neu trainieren wenn viele neue Daten
    if total > 500:
        logger.info("Viele neue Kerzen → starte Candle-Modell-Training...")
        try:
            from ml_network import ml_network
            for sym in targets:
                csv_path = str(HISTORY_DIR / f"{sym}_15m.csv")
                ml_network.train_from_csv(sym, csv_path)
        except Exception as e:
            logger.warning(f"ML-Training nach Update Fehler: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="Nur dieses Symbol")
    parser.add_argument("--full",   action="store_true",
                        help="Vollständige Historie (ignoriert letzten Timestamp)")
    args = parser.parse_args()

    syms = [args.symbol] if args.symbol else None
    run_update(symbols=syms, full=args.full)
