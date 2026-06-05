"""
websocket_manager.py – Verwaltet WebSocket-Verbindungen zu Binance Futures.
Market-Stream (Klines, Mark-Price) und User-Data-Stream (Fills, Balance-Updates).
Auto-Reconnect mit Exponential Backoff.
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import websockets

from config import config
from state import state
from exchange import exchange
from notifier import notifier

logger = logging.getLogger(__name__)

# WebSocket URLs
FUTURES_WS_URL = "wss://fstream.binance.com/stream"
USER_DATA_WS_BASE = "wss://fstream.binance.com/ws"

# Reconnect-Delays: [1, 2, 4, 8, 16, 32, 60] Sekunden, danach immer 60s
RECONNECT_DELAYS = [1, 2, 4, 8, 16, 32, 60]

# Listen-Key erneuern alle 30 Minuten
LISTEN_KEY_RENEW_INTERVAL = 30 * 60

# Health-Check alle 30 Sekunden
HEALTH_CHECK_INTERVAL = 30

# Callbacks (werden von main.py gesetzt)
_on_kline_15m_closed: Optional[Callable] = None
_on_kline_4h_closed: Optional[Callable] = None

# Fill-Waiter Dict: order_id → (Event, data_dict)
_fill_waiters: Dict[str, tuple] = {}

# Laufende Tasks
_market_task: Optional[asyncio.Task] = None
_user_data_task: Optional[asyncio.Task] = None
_listen_key_task: Optional[asyncio.Task] = None


def register_fill_waiter(order_id: str, event: asyncio.Event, data: dict):
    """Registriert einen Waiter für einen Order-Fill."""
    _fill_waiters[order_id] = (event, data)


def unregister_fill_waiter(order_id: str):
    """Entfernt einen Waiter."""
    _fill_waiters.pop(order_id, None)


def signal_fill_waiter(order_id: str, fill_data: dict):
    """Signalisiert einen Fill an den Waiter."""
    if order_id in _fill_waiters:
        event, data = _fill_waiters[order_id]
        data.update(fill_data)
        event.set()
        logger.debug(f"Fill-Waiter signalisiert für Order {order_id}")


def set_callbacks(on_kline_15m: Callable = None, on_kline_4h: Callable = None):
    """Setzt die Callbacks für Kline-Events."""
    global _on_kline_15m_closed, _on_kline_4h_closed
    _on_kline_15m_closed = on_kline_15m
    _on_kline_4h_closed = on_kline_4h


async def start():
    """Startet alle WebSocket-Verbindungen."""
    global _market_task, _user_data_task, _listen_key_task

    logger.info("WebSocket-Manager wird gestartet...")

    # Market-Stream starten
    _market_task = asyncio.create_task(_run_market_stream())

    # User-Data-Stream starten
    _user_data_task = asyncio.create_task(_run_user_data_stream())

    logger.info("WebSocket-Verbindungen gestartet")


async def stop():
    """Stoppt alle WebSocket-Verbindungen."""
    for task in [_market_task, _user_data_task, _listen_key_task]:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    logger.info("WebSocket-Manager gestoppt")


async def _run_market_stream():
    """
    Verwaltet den Market-Stream mit Auto-Reconnect.
    Abonniert: kline_15m, kline_4h, markPrice für alle konfigurierten Symbole.
    """
    delay_index = 0

    while True:
        try:
            # Stream-Namen für alle Symbole zusammenstellen
            streams = []
            for symbol in config.symbols:
                sym_lower = symbol.lower()
                streams.append(f"{sym_lower}@kline_15m")
                streams.append(f"{sym_lower}@kline_4h")
                streams.append(f"{sym_lower}@markPrice@1s")

            stream_url = f"{FUTURES_WS_URL}?streams={'/'.join(streams)}"
            logger.info(f"Market-Stream Verbindung: {stream_url[:80]}...")

            async with websockets.connect(
                stream_url,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
            ) as ws:
                # Erfolgreich verbunden – Delay zurücksetzen
                delay_index = 0
                logger.info("Market-Stream verbunden")

                # Health-Check Task
                health_task = asyncio.create_task(_health_check(ws, "Market-Stream"))

                try:
                    async for message in ws:
                        try:
                            await _process_market_message(json.loads(message))
                        except Exception as e:
                            logger.error(f"Fehler beim Verarbeiten der Market-Message: {e}")
                finally:
                    health_task.cancel()

        except websockets.ConnectionClosed as e:
            logger.warning(f"Market-Stream getrennt: {e}")
        except asyncio.CancelledError:
            logger.info("Market-Stream Task abgebrochen")
            break
        except Exception as e:
            logger.error(f"Market-Stream Fehler: {e}\n{traceback.format_exc()}")

        # Exponential Backoff
        delay = RECONNECT_DELAYS[min(delay_index, len(RECONNECT_DELAYS) - 1)]
        delay_index = min(delay_index + 1, len(RECONNECT_DELAYS))
        logger.info(f"Market-Stream Reconnect in {delay}s...")

        # Bei Reconnect: State reconcilen
        asyncio.create_task(_reconcile_on_reconnect())

        await asyncio.sleep(delay)


async def _run_user_data_stream():
    """
    Verwaltet den User-Data-Stream mit Listen-Key-Erneuerung.
    Empfängt Order-Fills und Balance-Updates.
    """
    global _listen_key_task
    delay_index = 0
    listen_key = None

    while True:
        try:
            # Listen-Key holen
            listen_key = await exchange.create_listen_key()
            if not listen_key:
                logger.error("Konnte keinen Listen-Key erstellen – Retry in 30s")
                await asyncio.sleep(30)
                continue

            ws_url = f"{USER_DATA_WS_BASE}/{listen_key}"
            logger.info("User-Data-Stream Verbindung aufgebaut")

            # Listen-Key-Erneuerungs-Task starten
            if _listen_key_task:
                _listen_key_task.cancel()
            _listen_key_task = asyncio.create_task(
                _renew_listen_key_loop(listen_key)
            )

            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
            ) as ws:
                delay_index = 0
                logger.info("User-Data-Stream verbunden")

                health_task = asyncio.create_task(_health_check(ws, "User-Data-Stream"))

                try:
                    async for message in ws:
                        try:
                            await _process_user_data_message(json.loads(message))
                        except Exception as e:
                            logger.error(f"Fehler beim Verarbeiten der User-Data-Message: {e}")
                finally:
                    health_task.cancel()

        except websockets.ConnectionClosed as e:
            logger.warning(f"User-Data-Stream getrennt: {e}")
        except asyncio.CancelledError:
            logger.info("User-Data-Stream Task abgebrochen")
            break
        except Exception as e:
            logger.error(f"User-Data-Stream Fehler: {e}")

        # Listen-Key-Task stoppen
        if _listen_key_task and not _listen_key_task.done():
            _listen_key_task.cancel()

        # Exponential Backoff
        delay = RECONNECT_DELAYS[min(delay_index, len(RECONNECT_DELAYS) - 1)]
        delay_index = min(delay_index + 1, len(RECONNECT_DELAYS))
        logger.info(f"User-Data-Stream Reconnect in {delay}s...")

        asyncio.create_task(_reconcile_on_reconnect())
        await asyncio.sleep(delay)


async def _renew_listen_key_loop(listen_key: str):
    """Erneuert den Listen-Key alle 30 Minuten."""
    while True:
        try:
            await asyncio.sleep(LISTEN_KEY_RENEW_INTERVAL)
            success = await exchange.renew_listen_key(listen_key)
            if success:
                logger.debug("Listen-Key erneuert")
            else:
                logger.warning("Listen-Key-Erneuerung fehlgeschlagen")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Fehler beim Erneuern des Listen-Keys: {e}")


async def _health_check(ws, stream_name: str):
    """
    Sendet alle 30 Sekunden einen Ping und wartet auf Pong.
    Bei Timeout: Verbindung wird als tot markiert (schließt den ws).
    """
    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            pong = await ws.ping()
            await asyncio.wait_for(pong, timeout=10)
            logger.debug(f"Health-Check OK: {stream_name}")
        except asyncio.TimeoutError:
            logger.warning(f"Health-Check Timeout: {stream_name} – Reconnect")
            await ws.close()
            break
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health-Check Fehler ({stream_name}): {e}")
            break


async def _process_market_message(data: dict):
    """Verarbeitet eingehende Market-Stream-Nachrichten."""
    stream = data.get("stream", "")
    payload = data.get("data", data)
    event_type = payload.get("e", "")

    # Mark-Price Update
    if event_type == "markPriceUpdate":
        price = float(payload.get("p", 0))
        if price > 0:
            from position_monitor import update_mark_price
            update_mark_price(price)

    # Kline-Event
    elif event_type == "kline":
        kline = payload.get("k", {})
        interval = kline.get("i")
        is_closed = kline.get("x", False)

        if is_closed:
            if interval == "15m" and _on_kline_15m_closed:
                symbol = kline.get("s")
                logger.debug(f"15m Kerze geschlossen: {symbol}")
                asyncio.create_task(_on_kline_15m_closed(symbol, kline))

            elif interval == "4h" and _on_kline_4h_closed:
                symbol = kline.get("s")
                logger.debug(f"4h Kerze geschlossen: {symbol}")
                asyncio.create_task(_on_kline_4h_closed(symbol, kline))


async def _process_user_data_message(data: dict):
    """Verarbeitet eingehende User-Data-Stream-Nachrichten."""
    event_type = data.get("e", "")

    if event_type == "ORDER_TRADE_UPDATE":
        # Order-Fill oder Status-Update
        from order_manager import on_fill_event
        await on_fill_event(data)

    elif event_type == "ACCOUNT_UPDATE":
        # Balance-Update
        state.update_balance(data)
        logger.debug("Balance via WebSocket aktualisiert")


async def _reconcile_on_reconnect():
    """
    Führt nach einem WebSocket-Reconnect eine State-Reconciliation durch.
    Stellt sicher dass SL/TP noch aktiv sind.
    """
    try:
        logger.info("Reconciliation nach Reconnect...")
        pos = state.open_position

        if not pos.is_open:
            return

        # Aktuelle Positionen vom Exchange holen
        exchange_positions = await exchange.get_open_positions()
        exchange_pos_map = {p.get("symbol"): p for p in exchange_positions}

        if pos.symbol not in exchange_pos_map:
            # Position im State, aber nicht mehr am Exchange – State korrigieren
            logger.warning(f"Position {pos.symbol} nicht mehr am Exchange – State wird korrigiert")
            state.close_position()
            return

        # Offene Orders prüfen
        open_orders = await exchange.get_open_orders(pos.symbol)
        order_ids = {str(o.get("orderId")) for o in open_orders}

        # SL fehlt?
        if pos.sl_order_id and str(pos.sl_order_id) not in order_ids:
            logger.warning(f"SL fehlt nach Reconnect – Neu setzen")
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            from order_manager import _set_sl_with_retry
            await _set_sl_with_retry(pos.symbol, close_side, pos.sl_price or 0, pos.qty or 0)

        # TP fehlt?
        if pos.tp_order_id and str(pos.tp_order_id) not in order_ids:
            logger.warning(f"TP fehlt nach Reconnect – Neu setzen")
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            from order_manager import _set_tp_with_retry
            await _set_tp_with_retry(pos.symbol, close_side, pos.tp_price or 0)

    except Exception as e:
        logger.error(f"Fehler bei Reconciliation: {e}")
