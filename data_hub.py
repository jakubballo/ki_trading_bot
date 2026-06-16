"""
data_hub.py – Zentraler WebSocket-Proxy für das Bot-Netzwerk.
Öffnet EINE Kraken-Verbindung für alle Symbole und verteilt
Kerzen + Ticker-Events via lokalem WebSocket (ws://127.0.0.1:8770).

Damit vermeiden 50 parallele Bots Rate-Limit-Probleme bei Kraken.

Starten: python data_hub.py
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Set

import websockets
# S5-6-Fix: WebSocketServerProtocol existiert nur in websockets <15 (Legacy-API).
# In v15+ heißt der Verbindungstyp ServerConnection. Import resilient halten, damit
# der Hub nicht beim Import crasht → sonst fallen alle 50 Bots auf Direktverbindungen
# zu Kraken zurück (503/Rate-Limit). Der Typ wird ausschließlich als Annotation genutzt.
try:
    from websockets.server import WebSocketServerProtocol
except ImportError:  # pragma: no cover
    try:
        from websockets.asyncio.server import ServerConnection as WebSocketServerProtocol
    except ImportError:
        WebSocketServerProtocol = object  # type: ignore

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HUB] %(levelname)s %(message)s",
)
logger = logging.getLogger("data_hub")

# Alle Symbole die der Hub abonnieren soll
HUB_SYMBOLS = [
    "PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD",
]

# S6-4: Marktdaten-WS IMMER vom Live-Public-Endpoint (öffentlich, keine Auth) —
# auch im paper-Modus. Der Demo-WS war instabil (HTTP 502 / "no close frame" /
# häufige Disconnects) und lieferte leicht abweichende Preise (~0.015 %) sowie ein
# Funding-Artefakt. Fills/Kontostand bleiben lokal simuliert (paper_simulate_fill),
# Scoring-Kerzen kommen ohnehin von der Live-Charts-REST-API → reine Marktdaten-
# Quelle, keine Logik-/Feature-Änderung. Ersetzt M14 (modusabhängige URL).
# Hinweis: Live nutzt den Candle-Feed `candles_trade_15m` (nicht `candles_15`).
_TRADING_MODE    = os.environ.get("TRADING_MODE", "paper").lower()
KRAKEN_WS_URL    = "wss://futures.kraken.com/ws/v1"
LOCAL_HUB_PORT   = int(os.environ.get("HUB_PORT", 8770))
RECONNECT_DELAY  = 5  # Sekunden zwischen Reconnect-Versuchen

# Alle verbundenen Bot-Clients
_clients: Set[WebSocketServerProtocol] = set()
_clients_lock = asyncio.Lock()

# Letzter Snapshot per Symbol (für neue Clients)
_last_snapshot: dict[str, dict] = {}


async def _broadcast(message: str):
    """Sendet eine Nachricht an alle verbundenen Bot-Clients."""
    async with _clients_lock:
        if not _clients:
            return
        # H4-Fix: parallel senden. Vorher blockierte ein langsamer Bot (await ws.send)
        # die Weitergabe an alle anderen 49 → Ticker-Stau für das ganze Netz.
        targets = list(_clients)
        results = await asyncio.gather(
            *[ws.send(message) for ws in targets],
            return_exceptions=True,
        )
        dead = {ws for ws, res in zip(targets, results) if isinstance(res, Exception)}
        _clients.difference_update(dead)


async def _client_handler(ws: WebSocketServerProtocol):
    """Verwaltet eine eingehende Bot-Verbindung."""
    addr = ws.remote_address
    logger.info(f"Bot verbunden: {addr}")
    async with _clients_lock:
        _clients.add(ws)

    # Snapshots der letzten Ticks sofort senden
    for snap in _last_snapshot.values():
        try:
            await ws.send(json.dumps(snap))
        except Exception:
            pass

    try:
        async for _ in ws:
            pass  # Hub empfängt nichts von Bots
    except Exception:
        pass
    finally:
        async with _clients_lock:
            _clients.discard(ws)
        logger.info(f"Bot getrennt: {addr}")


async def _kraken_listener():
    """
    Verbindet zum öffentlichen Kraken-WebSocket, abonniert alle Symbole
    und leitet Events an _broadcast weiter.
    """
    subscribe_msg = json.dumps({
        "event":    "subscribe",
        "feed":     "ticker",
        "product_ids": HUB_SYMBOLS,
    })
    subscribe_candles_15 = json.dumps({
        "event":    "subscribe",
        "feed":     "candles_trade_15m",   # S6-4: Live-Feed-Name (Demo war candles_15)
        "product_ids": HUB_SYMBOLS,
    })

    # Exponentieller Reconnect-Backoff: bei aufeinanderfolgenden Fehlversuchen
    # (z.B. Kraken-503/1013-Stürmen) wächst die Wartezeit 5→10→20→40→60s, statt
    # fix alle 5s zu hämmern (was 503-Rate-Limits verlängern kann). Sobald wieder
    # echte Daten fließen, springt sie auf den Basiswert zurück. Reines
    # Reconnect-Timing – an der Datenverarbeitung/Broadcast ändert sich nichts.
    backoff = RECONNECT_DELAY
    MAX_BACKOFF = 60

    while True:
        got_data = False
        try:
            logger.info(f"Verbinde zu Kraken WS: {KRAKEN_WS_URL}")
            async with websockets.connect(
                KRAKEN_WS_URL,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
            ) as ws:
                await ws.send(subscribe_msg)
                await ws.send(subscribe_candles_15)
                logger.info(f"Abonniert: {len(HUB_SYMBOLS)} Symbole")

                async for raw in ws:
                    got_data = True
                    try:
                        msg = json.loads(raw)
                        feed = msg.get("feed", "")

                        # Heartbeat ignorieren
                        if feed == "heartbeat" or msg.get("event") in ("heartbeat", "subscribed"):
                            continue

                        # Snapshot für neue Clients cachen
                        if feed in ("ticker", "ticker_lite"):
                            symbol = msg.get("product_id")
                            if symbol:
                                _last_snapshot[symbol] = msg

                        # Weiterleiten an alle Bots
                        await _broadcast(raw)

                    except Exception as e:
                        logger.warning(f"Fehler beim Verarbeiten: {e}")

        except Exception as e:
            logger.error(f"Kraken WS Verbindung verloren: {e}")

        # Backoff: nach erfolgreichem Datenempfang zurücksetzen, sonst verdoppeln (Deckel 60s)
        if got_data:
            backoff = RECONNECT_DELAY
        else:
            backoff = min(backoff * 2, MAX_BACKOFF)
        logger.info(f"Reconnect in {backoff}s...")
        await asyncio.sleep(backoff)


async def _hub_stats_loop():
    """Gibt alle 60s Statistiken aus."""
    while True:
        await asyncio.sleep(60)
        async with _clients_lock:
            n = len(_clients)
        logger.info(f"Hub-Status: {n} Bot(s) verbunden, {len(_last_snapshot)} Symbole gecacht")


async def _main():
    """Startet Hub-Server und Kraken-Listener."""
    logger.info(f"Data Hub startet auf ws://127.0.0.1:{LOCAL_HUB_PORT}")

    server = await websockets.serve(
        _client_handler,
        "127.0.0.1",
        LOCAL_HUB_PORT,
        ping_interval=30,
        ping_timeout=60,
    )

    listener_task = asyncio.create_task(_kraken_listener())
    stats_task    = asyncio.create_task(_hub_stats_loop())

    logger.info("Data Hub bereit.")
    try:
        await asyncio.gather(listener_task, stats_task)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        logger.info("Data Hub wird gestoppt...")
        listener_task.cancel()
        stats_task.cancel()
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(_main())
