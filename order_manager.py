"""
order_manager.py – Verwaltet den kompletten Order-Lifecycle.
Entry-Orders, SL/TP setzen, Fill-Bestätigung, Retry-Logik.
WICHTIG: KEINE Market-Orders als Fallback – nur Emergency-Close darf Market verwenden.
"""

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional

from config import config
from state import state
from exchange import exchange
from notifier import notifier
from logger_db import log_trade_opened, log_error

logger = logging.getLogger(__name__)

# Timeout für Entry-Orders (30 Minuten)
ENTRY_ORDER_TIMEOUT_SECONDS = 30 * 60

# Retry-Delays für SL/TP (in Sekunden)
SL_RETRY_DELAYS = [1, 2, 4, 8, 16]
TP_RETRY_DELAYS = [1, 2, 4, 8, 16]


def round_qty(raw_qty: float, step_size: float, min_qty: float,
              min_notional: float, price: float) -> Optional[float]:
    """
    Rundet die Menge auf step_size, prüft Mindestmengen.
    Gibt None zurück wenn Trade nicht möglich.
    """
    if step_size <= 0:
        return None

    # Auf step_size runden
    qty = round(round(raw_qty / step_size) * step_size, 8)

    # Mindestmenge prüfen
    if qty < min_qty:
        logger.warning(f"Qty {qty} unter Minimum {min_qty}")
        return None

    # Mindest-Notional prüfen
    if qty * price < min_notional:
        # Auf Mindest-Notional hochrunden
        min_qty_for_notional = min_notional / price
        qty = round(round(min_qty_for_notional / step_size + 0.5) * step_size, 8)
        logger.debug(f"Qty auf Mindest-Notional angepasst: {qty}")

    if qty < min_qty:
        logger.warning(f"Qty {qty} auch nach Anpassung unter Minimum {min_qty}")
        return None

    return qty


def calculate_entry_price(mark_price: float, side: str) -> float:
    """
    Berechnet den Entry-Preis mit 0.1% Slippage-Puffer.
    LONG: leicht unter Mark-Price, SHORT: leicht über Mark-Price.
    """
    if side.upper() == "BUY":
        return mark_price * (1 - 0.001)
    else:
        return mark_price * (1 + 0.001)


async def place_entry_order(symbol: str, side: str, qty: float,
                             entry_price: float, sl_price: float,
                             tp_price: float, score: int,
                             atr: float, regime: str) -> Optional[dict]:
    """
    Platziert eine Limit-Entry-Order und wartet auf Fill (max 30 Minuten).
    
    Returns:
        Fill-Event-Daten oder None bei Timeout/Fehler
    """
    logger.info(f"Entry-Order: {side} {qty} {symbol} @ {entry_price:.4f} "
                f"(SL={sl_price:.4f}, TP={tp_price:.4f})")

    # Order platzieren
    order = await exchange.place_limit_order(symbol, side, qty, entry_price)
    if not order:
        logger.error("Entry-Order konnte nicht platziert werden")
        return None

    # Kraken gibt string-UUID zurück, Paper-Orders: "paper_TIMESTAMP"
    order_id = order.get("orderId") or order.get("order_id") or order.get("uid")
    state.open_position.entry_order_id = str(order_id)
    state.write_on_event("order_placed")

    notifier.send_info(
        f"Entry-Order platziert: {side} {qty} {symbol} @ {entry_price:.4f}\n"
        f"Order-ID: {order_id}"
    )

    # Warten auf Fill via WebSocket + REST-Fallback
    fill_event = await _wait_for_fill(symbol, order_id)

    if fill_event is None:
        # Timeout – Order canceln
        logger.warning(f"Entry-Order Timeout nach {ENTRY_ORDER_TIMEOUT_SECONDS}s – Canceln")
        await exchange.cancel_order(symbol, order_id)
        state.open_position.entry_order_id = None
        state.write_on_event("order_cancelled")
        notifier.send_warning(f"Entry-Order Timeout – Order gecancelt: {order_id}")
        return None

    # Fill erhalten – SL und TP setzen
    await on_fill_event_internal(fill_event, sl_price, tp_price, score, atr, regime)
    return fill_event


async def _wait_for_fill(symbol: str, order_id, timeout: int = ENTRY_ORDER_TIMEOUT_SECONDS) -> Optional[dict]:
    """
    Wartet auf Order-Fill: primär über shared Event, Fallback REST-Poll alle 5s.
    """
    # Shared Event für WebSocket-Fill-Benachrichtigung
    fill_event = asyncio.Event()
    fill_data = {}

    # Event in globalem Dict registrieren (WebSocket-Manager signalisiert hier)
    from websocket_manager import register_fill_waiter, unregister_fill_waiter
    register_fill_waiter(str(order_id), fill_event, fill_data)

    try:
        deadline = asyncio.get_event_loop().time() + timeout
        poll_interval = 5  # REST-Fallback alle 5 Sekunden

        while asyncio.get_event_loop().time() < deadline:
            # Warte auf WebSocket-Event (max 5s)
            wait_time = min(poll_interval, deadline - asyncio.get_event_loop().time())
            if wait_time <= 0:
                break

            try:
                await asyncio.wait_for(asyncio.shield(fill_event.wait()), timeout=wait_time)
                if fill_event.is_set():
                    logger.info(f"Order {order_id} via WebSocket gefüllt")
                    return fill_data.copy()
            except asyncio.TimeoutError:
                pass

            # REST-Fallback: Order-Status prüfen
            order_status = await exchange.get_order_status(symbol, order_id)
            # Kraken status: "filled" | Binance: "FILLED" | Paper: "FILLED"
            status = (order_status.get("status") or "").upper()
            if order_status and status in ("FILLED", "FULLY_EXECUTED"):
                logger.info(f"Order {order_id} via REST-Poll gefüllt")
                qty   = float(order_status.get("executedQty") or
                              order_status.get("filledSize") or 0)
                price = float(order_status.get("avgPrice") or
                              order_status.get("last_price") or
                              order_status.get("price") or 0)
                return {
                    "symbol": symbol,
                    "side": order_status.get("side"),
                    "qty": qty,
                    "price": price,
                    "order_id": order_id,
                }

        return None  # Timeout

    finally:
        unregister_fill_waiter(str(order_id))


async def on_fill_event(event: dict):
    """
    Callback vom WebSocket-Manager / Paper-Fill-Simulator.
    Unterstützt beide Formate: Kraken WS und Paper-Order-Dicts.
    """
    try:
        # Kraken Paper-Order / normalisiertes Format
        order_id    = str(event.get("order_id") or event.get("orderId") or "")
        status      = (event.get("status") or "").upper()
        symbol      = event.get("symbol")
        side        = event.get("side")
        qty         = float(event.get("qty") or event.get("filledSize") or 0)
        price       = float(event.get("price") or event.get("avg_price") or 0)

        from websocket_manager import signal_fill_waiter
        if status in ("FILLED", "FULLY_EXECUTED") and order_id:
            fill_data = {"symbol": symbol, "side": side, "qty": qty,
                         "price": price, "order_id": order_id}
            signal_fill_waiter(order_id, fill_data)

    except Exception as e:
        logger.error(f"Fehler in on_fill_event: {e}")
        log_error("order_manager", type(e).__name__, str(e), traceback.format_exc())


async def on_fill_event_internal(fill_data: dict, sl_price: float, tp_price: float,
                                  score: int, atr: float, regime: str):
    """
    Verarbeitet einen bestätigten Fill: State aktualisieren, SL/TP setzen.
    """
    symbol = fill_data.get("symbol", fill_data.get("s"))
    side = fill_data.get("side", fill_data.get("S"))
    qty = float(fill_data.get("qty", fill_data.get("z", 0)))
    entry_price = float(fill_data.get("price", fill_data.get("ap", 0)))
    order_id = str(fill_data.get("order_id", fill_data.get("i", "")))

    logger.info(f"Fill bestätigt: {side} {qty} {symbol} @ {entry_price:.4f}")

    # State aktualisieren
    state.set_position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        qty=qty,
        sl_price=sl_price,
        tp_price=tp_price,
        entry_order_id=order_id,
        atr_at_entry=atr,
        regime_at_entry=regime,
    )

    # Trade in DB loggen
    log_trade_opened(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        qty=qty,
        sl=sl_price,
        tp=tp_price,
        regime=regime,
        score=score,
        opened_at=datetime.now(timezone.utc).isoformat(),
    )

    # Gegenseite für SL/TP
    close_side = "SELL" if side.upper() == "BUY" else "BUY"

    # SL und TP parallel setzen
    sl_task = asyncio.create_task(_set_sl_with_retry(symbol, close_side, sl_price, qty))
    tp_task = asyncio.create_task(_set_tp_with_retry(symbol, close_side, tp_price))
    await asyncio.gather(sl_task, tp_task, return_exceptions=True)

    notifier.send_trade_opened(
        symbol=symbol, side=side, entry_price=entry_price, qty=qty,
        sl=sl_price, tp=tp_price, score=score, regime=regime,
        balance=state.account_balance_usdt,
    )


async def _set_sl_with_retry(symbol: str, side: str, sl_price: float, qty: float):
    """
    Setzt SL mit bis zu 5 Versuchen.
    Bei totalem Fehlschlag: Emergency-Close auslösen.
    """
    for attempt, delay in enumerate(SL_RETRY_DELAYS):
        try:
            order = await exchange.place_stop_market(symbol, side, sl_price)
            if order:
                state.open_position.sl_order_id = str(
                    order.get("orderId") or order.get("order_id") or "")
                state.write_on_event("sl_set")
                logger.info(f"SL gesetzt: {symbol} @ {sl_price:.4f} (Versuch {attempt + 1})")
                return
        except Exception as e:
            logger.warning(f"SL setzen fehlgeschlagen (Versuch {attempt + 1}): {e}")

        if attempt < len(SL_RETRY_DELAYS) - 1:
            await asyncio.sleep(delay)

    # Alle Versuche fehlgeschlagen – Emergency-Close
    logger.critical(f"SL konnte nicht gesetzt werden für {symbol} – Emergency-Close!")
    await notifier.send_critical(
        f"🚨 SL FEHLGESCHLAGEN für {symbol}!\n"
        f"Emergency-Close wird ausgeführt..."
    )

    pos = state.open_position
    if pos.symbol and pos.qty:
        await exchange.emergency_close(pos.symbol, pos.qty, pos.side or "BUY")

    state.close_position()


async def _set_tp_with_retry(symbol: str, side: str, tp_price: float):
    """
    Setzt TP mit bis zu 5 Versuchen.
    Bei Fehler: Warnung senden, manuelles Eingreifen nötig.
    """
    for attempt, delay in enumerate(TP_RETRY_DELAYS):
        try:
            order = await exchange.place_take_profit_market(symbol, side, tp_price)
            if order:
                state.open_position.tp_order_id = str(
                    order.get("orderId") or order.get("order_id") or "")
                state.write_on_event("tp_set")
                logger.info(f"TP gesetzt: {symbol} @ {tp_price:.4f} (Versuch {attempt + 1})")
                return
        except Exception as e:
            logger.warning(f"TP setzen fehlgeschlagen (Versuch {attempt + 1}): {e}")

        if attempt < len(TP_RETRY_DELAYS) - 1:
            await asyncio.sleep(delay)

    # Alle Versuche fehlgeschlagen – Warnung senden
    logger.error(f"TP konnte nicht gesetzt werden für {symbol} – Manuelles Eingreifen nötig!")
    notifier.send_warning(
        f"⚠️ TP FEHLGESCHLAGEN für {symbol}!\n"
        f"Position läuft OHNE Take-Profit!\n"
        f"Manuelles Eingreifen erforderlich!"
    )
