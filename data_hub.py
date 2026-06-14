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
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HUB] %(levelname)s %(message)s",
)
logger = logging.getLogger("data_hub")

# Alle Symbole die der Hub abonnieren soll
HUB_SYMBOLS = [
    "PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD",
]

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
        dead = set()
        for ws in _clients:
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
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
        "feed":     "candles_15",
        "product_ids": HUB_SYMBOLS,
    })

    while True:
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

        logger.info(f"Reconnect in {RECONNECT_DELAY}s...")
        await asyncio.sleep(RECONNECT_DELAY)


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
