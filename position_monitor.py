"""
position_monitor.py – Laufende Überwachung offener Positionen.
Prüft Liquidations-Risiko, Haltedauer, fehlende SL/TP und Funding-Charges.
"""

import asyncio
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import config
from state import state
from exchange import exchange
from notifier import notifier
from logger_db import log_error, log_funding_charge

logger = logging.getLogger(__name__)

# Intervall für Haupt-Monitoring-Loop (Sekunden)
MONITOR_INTERVAL = 30

# Intervall für Order-Check (Minuten)
ORDER_CHECK_INTERVAL_MINUTES = 5

# Funding-Rate-Zeiten (UTC Stunden: 00, 08, 16)
FUNDING_HOURS = {0, 8, 16}

# Aktueller Mark-Preis (wird via WebSocket aktualisiert)
_mark_price: float = 0.0
_last_order_check: Optional[datetime] = None
_last_funding_hour: Optional[int] = None


def update_mark_price(price: float):
    """Wird vom WebSocket-Manager aufgerufen mit dem aktuellen Mark-Preis."""
    global _mark_price
    _mark_price = price


async def run_position_monitor():
    """
    Haupt-Monitoring-Task. Läuft alle 30 Sekunden und prüft:
    - Liquidations-Abstand
    - Max Haltedauer (48h)
    - Fehlende SL/TP Orders
    - Funding-Charges
    """
    global _last_order_check, _last_funding_hour
    logger.info("Position-Monitor gestartet")

    while True:
        try:
            await asyncio.sleep(MONITOR_INTERVAL)

            # Nur prüfen wenn eine Position offen ist
            pos = state.open_position
            if not pos.is_open:
                continue

            # 1. Liquidations-Risiko prüfen
            await _check_liquidation_risk(pos)

            # 2. Max Haltedauer prüfen (48h)
            await _check_max_hold_duration(pos)

            # 3. Alle 5 Minuten: SL/TP Orders prüfen
            now = datetime.now(timezone.utc)
            if (_last_order_check is None or
                    (now - _last_order_check).total_seconds() >= ORDER_CHECK_INTERVAL_MINUTES * 60):
                await _check_sl_tp_orders(pos)
                _last_order_check = now

            # 4. Funding-Charges protokollieren (00:00, 08:00, 16:00 UTC)
            current_hour = now.hour
            if current_hour in FUNDING_HOURS and current_hour != _last_funding_hour:
                await _log_funding_charge(pos)
                _last_funding_hour = current_hour

        except asyncio.CancelledError:
            logger.info("Position-Monitor gestoppt")
            break
        except Exception as e:
            logger.error(f"Fehler im Position-Monitor: {e}")
            log_error("position_monitor", type(e).__name__, str(e), traceback.format_exc())


async def _check_liquidation_risk(pos):
    """
    Prüft den Abstand zur Liquidation.
    Warnung bei < 10%, Emergency-Close bei < 5%.
    """
    try:
        mark_price = _mark_price
        if mark_price <= 0:
            # Fallback: Mark-Preis via REST holen
            mark_price = await exchange.get_mark_price(pos.symbol)

        if mark_price <= 0:
            return

        # Liquidation-Preis aus Position-Risk holen
        pos_risk = await exchange.get_position_risk(pos.symbol)
        if not pos_risk:
            return

        liq_price = float(pos_risk.get("liquidationPrice", 0))
        if liq_price <= 0:
            return

        # Abstand berechnen
        distance = abs(liq_price - mark_price) / mark_price

        # State aktualisieren
        state.open_position.liquidation_price = liq_price

        logger.debug(f"Liquidations-Abstand: {distance * 100:.1f}% "
                     f"(Mark={mark_price:.4f}, Liq={liq_price:.4f})")

        if distance < 0.05:
            # Kritisch: Emergency-Close
            msg = (f"🚨 LIQUIDATION GEFAHR!\n"
                   f"Abstand: {distance * 100:.1f}% < 5%\n"
                   f"Mark: {mark_price:.4f}, Liq: {liq_price:.4f}\n"
                   f"Emergency-Close wird ausgeführt!")
            logger.critical(msg)
            await notifier.send_critical(msg)

            close_side = "SELL" if pos.side == "BUY" else "BUY"
            await exchange.emergency_close(pos.symbol, pos.qty or 0, close_side)
            state.close_position()

        elif distance < 0.10:
            # Warnung
            notifier.send_warning(
                f"⚠️ Liquidations-Warnung!\n"
                f"Abstand: {distance * 100:.1f}% < 10%\n"
                f"Mark: {mark_price:.4f}, Liq: {liq_price:.4f}"
            )

    except Exception as e:
        logger.error(f"Fehler beim Prüfen des Liquidations-Risikos: {e}")


async def _check_max_hold_duration(pos):
    """
    Prüft ob die Position die maximale Haltedauer überschritten hat (48h).
    Wenn ja: Position schließen.
    """
    if not pos.entry_time_utc:
        return

    try:
        entry_time = datetime.fromisoformat(pos.entry_time_utc.replace("Z", "+00:00"))
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        max_hours = config.risk.get("max_hold_hours", 48)
        max_duration = timedelta(hours=max_hours)
        hold_duration = datetime.now(timezone.utc) - entry_time

        if hold_duration >= max_duration:
            msg = (f"⏰ Max Haltedauer erreicht ({max_hours}h)\n"
                   f"Position: {pos.symbol} ({pos.side})\n"
                   f"Gehalten: {hold_duration.total_seconds() / 3600:.1f}h\n"
                   f"Schließe Position...")
            logger.warning(msg)
            notifier.send_warning(msg)

            # Alle Orders canceln und Position schließen (via Market-Order)
            await exchange.cancel_all_orders(pos.symbol)
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            await exchange.place_market_order(pos.symbol, close_side, pos.qty or 0)
            state.close_position()

    except Exception as e:
        logger.error(f"Fehler beim Prüfen der Haltedauer: {e}")


async def _check_sl_tp_orders(pos):
    """
    Prüft ob SL und TP Orders noch aktiv sind.
    Falls nicht: neu setzen.
    """
    try:
        open_orders = await exchange.get_open_orders(pos.symbol)
        order_ids = {str(o.get("orderId")) for o in open_orders}

        # SL prüfen
        if pos.sl_order_id and str(pos.sl_order_id) not in order_ids:
            logger.warning(f"SL-Order fehlt für {pos.symbol} – Neu setzen")
            notifier.send_warning(f"⚠️ SL-Order fehlt für {pos.symbol} – Wird neu gesetzt")

            close_side = "SELL" if pos.side == "BUY" else "BUY"
            from order_manager import _set_sl_with_retry
            await _set_sl_with_retry(pos.symbol, close_side, pos.sl_price or 0, pos.qty or 0)

        # TP prüfen
        if pos.tp_order_id and str(pos.tp_order_id) not in order_ids:
            logger.warning(f"TP-Order fehlt für {pos.symbol} – Neu setzen")
            notifier.send_warning(f"⚠️ TP-Order fehlt für {pos.symbol} – Wird neu gesetzt")

            close_side = "SELL" if pos.side == "BUY" else "BUY"
            from order_manager import _set_tp_with_retry
            await _set_tp_with_retry(pos.symbol, close_side, pos.tp_price or 0)

    except Exception as e:
        logger.error(f"Fehler beim Prüfen der SL/TP-Orders: {e}")


async def _log_funding_charge(pos):
    """
    Protokolliert die Funding-Charge zum aktuellen Funding-Zeitpunkt.
    """
    try:
        funding_rate = await exchange.get_funding_rate(pos.symbol)
        mark_price = _mark_price or await exchange.get_mark_price(pos.symbol)
        position_size = (pos.qty or 0) * mark_price

        # Funding-Charge = Positionsgröße × Funding-Rate
        charge_usdt = position_size * funding_rate

        log_funding_charge(
            symbol=pos.symbol,
            funding_rate=funding_rate,
            position_size=position_size,
            charge_usdt=charge_usdt,
        )

        if abs(charge_usdt) > 1.0:  # Nur relevante Charges loggen
            logger.info(f"Funding-Charge: {charge_usdt:.4f} USDT "
                        f"(Rate: {funding_rate:.4%}, Größe: {position_size:.2f})")

    except Exception as e:
        logger.error(f"Fehler beim Protokollieren der Funding-Charge: {e}")
