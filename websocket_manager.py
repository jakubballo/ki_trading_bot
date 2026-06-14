"""
websocket_manager.py – Kraken Futures WebSocket Manager.
Öffentlicher Feed: candles_15, candles_240, ticker für alle Symbole.
APScheduler-Fallback: 4h-Regime-Update (Demo liefert keine candles_240-Events).
Paper-Trading: überwacht Mark-Preis und simuliert SL/TP-Fills lokal.
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

# Kraken Futures WebSocket URLs
WS_URL_LIVE = "wss://futures.kraken.com/ws/v1"
WS_URL_DEMO = "wss://demo-futures.kraken.com/ws/v1"

# Lokaler Data Hub (bevorzugt – verhindert 50 direkte Kraken-Verbindungen)
HUB_URL = "ws://127.0.0.1:8770"

RECONNECT_DELAYS = [1, 2, 4, 8, 16, 32, 60]
HEALTH_CHECK_INTERVAL = 30

# Callbacks (gesetzt von main.py)
_on_kline_15m_closed: Optional[Callable] = None
_on_kline_4h_closed: Optional[Callable] = None

# Fill-Waiter für Entry-Orders
_fill_waiters: Dict[str, tuple] = {}

# Letzter bekannter Mark-Preis pro Symbol
_mark_prices: Dict[str, float] = {}

_market_task: Optional[asyncio.Task] = None
_paper_monitor_task: Optional[asyncio.Task] = None


def register_fill_waiter(order_id: str, event: asyncio.Event, data: dict):
    _fill_waiters[order_id] = (event, data)


def unregister_fill_waiter(order_id: str):
    _fill_waiters.pop(order_id, None)


def signal_fill_waiter(order_id: str, fill_data: dict):
    if order_id in _fill_waiters:
        event, data = _fill_waiters[order_id]
        data.update(fill_data)
        event.set()


def set_callbacks(on_kline_15m: Callable = None, on_kline_4h: Callable = None):
    global _on_kline_15m_closed, _on_kline_4h_closed
    _on_kline_15m_closed = on_kline_15m
    _on_kline_4h_closed = on_kline_4h


async def start():
    """Startet den WebSocket-Manager und (im Paper-Modus) den Fill-Simulator."""
    global _market_task, _paper_monitor_task

    logger.info("WebSocket-Manager wird gestartet...")
    _market_task = asyncio.create_task(_run_market_stream())

    if config.is_paper:
        _paper_monitor_task = asyncio.create_task(_run_paper_fill_monitor())
        logger.info("Paper-Fill-Simulator gestartet")

    logger.info("WebSocket-Verbindungen gestartet")


async def stop():
    for task in [_market_task, _paper_monitor_task]:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    logger.info("WebSocket-Manager gestoppt")


# ─── Markt-Stream ─────────────────────────────────────────────────────────────

async def _run_market_stream():
    """
    Verbindet bevorzugt mit dem lokalen Data Hub (ws://127.0.0.1:8770).
    Fallback: direkte Kraken WS Verbindung wenn Hub nicht erreichbar.
    Der Hub leitet rohe Kraken-Nachrichten 1:1 weiter → kein Abonnement nötig.
    """
    delay_index = 0
    kraken_url = WS_URL_DEMO if config.is_paper else WS_URL_LIVE
    use_hub = True  # Starte mit Hub-Verbindung

    while True:
        connect_url = HUB_URL if use_hub else kraken_url
        try:
            logger.info(f"Verbinde WS: {connect_url}")
            async with websockets.connect(
                connect_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024,
            ) as ws:
                delay_index = 0
                if use_hub:
                    logger.info("Lokaler Data Hub verbunden")
                else:
                    logger.info("Kraken WS verbunden (direkt)")
                    await _subscribe(ws)

                health_task = asyncio.create_task(_health_check(ws))
                try:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            await _process_message(msg)
                        except Exception as e:
                            logger.error(f"Fehler beim Verarbeiten der WS-Nachricht: {e}")
                finally:
                    health_task.cancel()

        except websockets.ConnectionClosed as e:
            logger.warning(f"WS getrennt: {e}")
        except asyncio.CancelledError:
            logger.info("WS-Task abgebrochen")
            break
        except Exception as e:
            if use_hub:
                logger.warning(f"Hub nicht erreichbar – Fallback zu Kraken Direct: {e}")
                use_hub = False
                continue
            logger.error(f"WS-Fehler: {e}\n{traceback.format_exc()}")

        delay = RECONNECT_DELAYS[min(delay_index, len(RECONNECT_DELAYS) - 1)]
        delay_index = min(delay_index + 1, len(RECONNECT_DELAYS))
        logger.info(f"WS Reconnect in {delay}s...")
        asyncio.create_task(_reconcile_on_reconnect())
        await asyncio.sleep(delay)
        use_hub = True  # Nach jedem Reconnect wieder Hub versuchen


async def _subscribe(ws):
    """Sendet Abonnements für alle konfigurierten Symbole."""
    product_ids = config.symbols

    # Ticker (für Mark-Preis Echtzeit)
    await ws.send(json.dumps({
        "event": "subscribe",
        "feed": "ticker",
        "product_ids": product_ids,
    }))

    # 15-Minuten-Kerzen
    await ws.send(json.dumps({
        "event": "subscribe",
        "feed": "candles_15",
        "product_ids": product_ids,
    }))

    # 240-Minuten-Kerzen (4h) – Demo liefert diese oft nicht
    await ws.send(json.dumps({
        "event": "subscribe",
        "feed": "candles_240",
        "product_ids": product_ids,
    }))

    logger.info(f"WS abonniert für: {product_ids}")


async def _process_message(msg: dict):
    """Verarbeitet eingehende WS-Nachrichten."""
    if not isinstance(msg, dict):
        return

    feed = msg.get("feed", "")
    event = msg.get("event", "")

    # Heartbeat / System-Nachrichten ignorieren
    if event in ("heartbeat", "subscribed", "unsubscribed", "info", "alert"):
        return

    # Ticker-Update (Echtzeit-Preis)
    if feed == "ticker":
        symbol = msg.get("product_id")
        price = float(msg.get("markPrice", msg.get("last", 0)) or 0)
        if symbol and price > 0:
            _mark_prices[symbol] = price
            if symbol in config.symbols:
                from position_monitor import update_mark_price
                update_mark_price(price)
                # Shadow-Trades (blockierte Signale) auflösen → Lernen aus Nicht-Trades.
                # Ohne diesen Aufruf bekämen Shadows nie ein Outcome (exit_price=NULL).
                try:
                    from shadow_tracker import shadow_tracker
                    shadow_tracker.update_prices(symbol, price)
                except Exception:
                    pass

    # 15-Minuten-Kerze geschlossen
    elif feed.startswith("candles_15") and "candle" in msg:
        candle = msg["candle"]
        symbol = msg.get("product_id")
        is_closed = candle.get("close") is not None
        if is_closed and symbol and _on_kline_15m_closed:
            logger.info(f"15m Kerze geschlossen: {symbol}")
            asyncio.create_task(_on_kline_15m_closed(symbol, candle))

    # 4h-Kerze geschlossen
    elif feed.startswith("candles_240") and "candle" in msg:
        candle = msg["candle"]
        symbol = msg.get("product_id")
        if symbol and _on_kline_4h_closed:
            logger.info(f"4h Kerze geschlossen: {symbol}")
            asyncio.create_task(_on_kline_4h_closed(symbol, candle))


async def _health_check(ws):
    """Sendet alle 30s einen Ping."""
    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            pong = await ws.ping()
            await asyncio.wait_for(pong, timeout=10)
            logger.debug("WS Health-Check OK")
        except asyncio.TimeoutError:
            logger.warning("WS Health-Check Timeout – Reconnect")
            await ws.close()
            break
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health-Check Fehler: {e}")
            break


# ─── Paper-Fill-Simulator ──────────────────────────────────────────────────────

async def _run_paper_fill_monitor():
    """
    Paper-Trading-Simulator:
    Überwacht Mark-Preise und simuliert Fills wenn Preis die Order-Grenze kreuzt.
    Prüft alle 5 Sekunden.
    """
    logger.info("Paper-Fill-Monitor aktiv")
    while True:
        try:
            await asyncio.sleep(5)
            pos = state.open_position

            for symbol in config.symbols:
                mark = _mark_prices.get(symbol, 0)
                if mark <= 0:
                    continue

                # Entry-Order prüfen
                if not pos.is_open:
                    await _check_entry_fills(symbol, mark)
                else:
                    # SL/TP prüfen wenn Position offen
                    await _check_sl_tp_fills(pos, symbol, mark)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Paper-Fill-Monitor Fehler: {e}")


async def _check_entry_fills(symbol: str, mark_price: float):
    """Prüft ob eine Limit-Entry-Order gefüllt werden sollte."""
    orders_to_fill = []
    for order_id, order in list(exchange._paper_orders.items()):
        if order.get("symbol") != symbol or order.get("type") != "LIMIT":
            continue
        if order.get("status") != "NEW":
            continue

        limit_price = float(order.get("price", 0))
        side = order.get("side", "")

        # LONG: fill wenn Mark ≤ Limit (günstigerer Preis erreicht)
        # SHORT: fill wenn Mark ≥ Limit
        should_fill = (
            (side == "BUY" and mark_price <= limit_price * 1.002) or
            (side == "SELL" and mark_price >= limit_price * 0.998)
        )

        if should_fill:
            orders_to_fill.append((order_id, mark_price))

    for order_id, fill_price in orders_to_fill:
        fill_data = exchange.paper_simulate_fill(order_id, fill_price)
        if fill_data:
            # Fill-Waiter signalisieren (order_manager wartet darauf)
            signal_fill_waiter(str(order_id), fill_data)
            # Paper-Position setzen
            order = exchange._paper_orders.get(order_id, {})
            qty = float(order.get("qty", 0))
            sign = 1 if order.get("side") == "BUY" else -1
            exchange._paper_position = {
                "symbol": symbol,
                "positionAmt": qty * sign,
                "entryPrice": fill_price,
                "side": order.get("side"),
            }


async def _check_sl_tp_fills(pos, symbol: str, mark_price: float):
    """Prüft ob SL oder TP für eine offene Position erreicht wurde."""
    if pos.symbol != symbol:
        return

    sl_price = pos.sl_price or 0
    tp_price = pos.tp_price or 0
    is_long = pos.side == "BUY"

    sl_hit = (is_long and mark_price <= sl_price) or (not is_long and mark_price >= sl_price)
    tp_hit = (is_long and mark_price >= tp_price) or (not is_long and mark_price <= tp_price)

    if sl_hit or tp_hit:
        exit_price = sl_price if sl_hit else tp_price
        exit_reason = "sl" if sl_hit else "tp"
        fee_rate = config.risk.get("fee_taker", 0.0005) + config.risk.get("fee_slippage", 0.0002)

        qty = pos.qty or 0
        entry = pos.entry_price or exit_price
        if is_long:
            pnl_raw = qty * (exit_price - entry)
        else:
            pnl_raw = qty * (entry - exit_price)
        fees = qty * exit_price * fee_rate * 2  # Ein- und Ausstieg
        pnl_net = pnl_raw - fees

        logger.info(f"[PAPER] {exit_reason.upper()} getroffen: {symbol} @ {exit_price:.4f} "
                    f"| PnL: {pnl_net:+.2f} USD")

        # PnL zum Paper-Kontostand addieren
        exchange._paper_balance += pnl_net
        exchange._paper_pnl += pnl_net

        # State aktualisieren
        pnl_pct = pnl_net / (qty * entry) if qty * entry > 0 else 0
        state.update_daily_pnl(pnl_net)
        state.update_weekly_pnl(pnl_net)

        from logger_db import log_trade_closed
        trade_id = getattr(state.open_position, "_db_trade_id", -1)
        if trade_id > 0:
            from datetime import datetime
            entry_time = datetime.fromisoformat(pos.entry_time_utc.replace("Z", "+00:00"))
            hold_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            log_trade_closed(
                trade_id=trade_id,
                exit_price=exit_price,
                pnl_usdt=pnl_net,
                pnl_pct=pnl_pct * 100,
                fees_usdt=fees,
                funding_paid_usdt=0.0,
                hold_duration_hours=hold_hours,
                exit_reason=exit_reason,
            )

        # Trade in network.db schreiben
        try:
            from network_db import log_network_trade
            log_network_trade(
                bot_id=config.bot_id,
                symbol=symbol,
                side=pos.side,
                entry=entry,
                exit_price=exit_price,
                pnl=pnl_net,
                exit_reason=exit_reason,
                score=getattr(state.open_position, "_score", 0),
                regime=pos.regime_at_entry or "unknown",
                is_shadow=False,
            )
        except Exception:
            pass

        notifier.send_trade_closed(
            symbol=symbol,
            side=pos.side or "",
            entry_price=entry,
            exit_price=exit_price,
            pnl_usdt=pnl_net,
            pnl_pct=pnl_pct * 100,
            hold_hours=0,
            exit_reason=exit_reason,
            balance=exchange._paper_balance,
        )

        exchange._paper_close_position(symbol)
        state.close_position()


# ─── Reconciliation ────────────────────────────────────────────────────────────

async def _reconcile_on_reconnect():
    """Reconciliert State nach WS-Reconnect."""
    try:
        logger.info("Reconciliation nach Reconnect...")
        pos = state.open_position
        if not pos.is_open:
            return

        if config.is_paper:
            return  # Paper: lokaler State ist immer korrekt

        exchange_positions = await exchange.get_open_positions()
        exchange_pos_map = {p.get("symbol"): p for p in exchange_positions}

        if pos.symbol not in exchange_pos_map:
            logger.warning(f"Position {pos.symbol} nicht am Exchange – State korrigieren")
            state.close_position()
            return

        open_orders = await exchange.get_open_orders(pos.symbol)
        order_ids = {str(o.get("orderId")) for o in open_orders}

        if pos.sl_order_id and str(pos.sl_order_id) not in order_ids:
            logger.warning("SL fehlt nach Reconnect – Neu setzen")
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            from order_manager import _set_sl_with_retry
            await _set_sl_with_retry(pos.symbol, close_side, pos.sl_price or 0, pos.qty or 0)

        if pos.tp_order_id and str(pos.tp_order_id) not in order_ids:
            logger.warning("TP fehlt nach Reconnect – Neu setzen")
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            from order_manager import _set_tp_with_retry
            await _set_tp_with_retry(pos.symbol, close_side, pos.tp_price or 0)

    except Exception as e:
        logger.error(f"Fehler bei Reconciliation: {e}")


def get_mark_price(symbol: str) -> float:
    """Gibt den zuletzt bekannten Mark-Preis für ein Symbol zurück."""
    return _mark_prices.get(symbol, 0.0)
