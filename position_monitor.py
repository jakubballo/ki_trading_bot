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


def _log_closed_position(pos, exit_price: float, exit_reason: str):
    """
    K-C-Fix: Schreibt das Outcome eines Nicht-SL/TP-Exits (Timeout, Liquidation)
    in network.db (ML-Training) UND logger_db (Pro-Bot-Statistik) – VOR close_position().
    Ohne diesen Aufruf verschwanden Timeout-/Emergency-Closes komplett aus dem Lernsystem
    und blieben als "offen" in der DB stehen.
    """
    try:
        qty = pos.qty or 0
        entry = pos.entry_price or exit_price
        is_long = pos.side == "BUY"
        if is_long:
            pnl_raw = qty * (exit_price - entry)
        else:
            pnl_raw = qty * (entry - exit_price)
        fee_rate = (config.risk.get("fee_taker", 0.0005)
                    + config.risk.get("fee_slippage", 0.0002))
        fees = qty * exit_price * fee_rate * 2
        pnl_net = pnl_raw - fees
        pnl_pct = pnl_net / (qty * entry) if qty * entry > 0 else 0.0

        # Tages-/Wochen-PnL aktualisieren (für Tagesverlust-Limit)
        state.update_daily_pnl(pnl_net)
        state.update_weekly_pnl(pnl_net)

        # Pro-Bot-DB
        try:
            from logger_db import log_trade_closed
            trade_id = getattr(state.open_position, "_db_trade_id", -1)
            if trade_id > 0:
                entry_time = datetime.fromisoformat(
                    pos.entry_time_utc.replace("Z", "+00:00")) if pos.entry_time_utc else None
                hold_hours = ((datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                              if entry_time else 0.0)
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
        except Exception as e:
            logger.error(f"log_trade_closed fehlgeschlagen: {e}")

        # Netzwerk-DB (ML-Training)
        # S5-5-Fix: Marktstruktur-Features aus dem bei Entry gemerkten Scoring mitgeben.
        try:
            from network_db import log_network_trade
            f = getattr(state.open_position, "_features", {}) or {}
            log_network_trade(
                bot_id=config.bot_id,
                symbol=pos.symbol,
                side=pos.side,
                entry=entry,
                exit_price=exit_price,
                pnl=pnl_net,
                exit_reason=exit_reason,
                score=getattr(state.open_position, "_score", 0),
                regime=getattr(pos, "regime_at_entry", None) or "unknown",
                rsi=f.get("_rsi", 50.0),
                atr=getattr(pos, "atr_at_entry", 0.0) or 0.0,
                fg_index=f.get("_fg_index", 50.0),
                strategy=config.strategy,
                is_shadow=False,
                macd_diff=f.get("_macd_diff", 0.0),
                macd_signal_val=f.get("_macd_signal", 0.0),
                ema_ratio_9_21=f.get("_ema_ratio_9_21", 0.0),
                ema_ratio_21_50=f.get("_ema_ratio_21_50", 0.0),
                price_vs_ema50=f.get("_price_vs_ema50", 0.0),
                bb_pct=f.get("_bb_pct", 0.5),
                bb_width=f.get("_bb_width", 0.0),
                vol_ratio=f.get("_vol_ratio", 1.0),
                rsi_slope=f.get("_rsi_slope", 0.0),
                ret_1=f.get("_ret_1", 0.0),
                ret_4=f.get("_ret_4", 0.0),
                ret_8=f.get("_ret_8", 0.0),
                ret_16=f.get("_ret_16", 0.0),
                opened_at=getattr(pos, "entry_time_utc", None),  # S6-1: echte Open-Zeit
            )
        except Exception as e:
            logger.error(f"log_network_trade fehlgeschlagen: {e}")

    except Exception as e:
        logger.error(f"_log_closed_position Fehler: {e}")


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
        # Kraken: kein explizites liquidationPrice – aus Hebel und Entry schätzen
        pos_risk = await exchange.get_position_risk(pos.symbol)
        liq_price = 0.0
        if pos_risk:
            liq_price = float(pos_risk.get("liquidationPrice") or
                              pos_risk.get("liq_price") or 0)

        # Schätzung wenn nicht verfügbar (konservativ: 80% der Bewegung bis Liq)
        if liq_price <= 0 and pos.entry_price and pos.side:
            leverage = config.risk.get("leverage", 5)
            safety   = 1.0 / leverage
            if pos.side == "BUY":
                liq_price = pos.entry_price * (1 - safety)
            else:
                liq_price = pos.entry_price * (1 + safety)

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
            # K-C-Fix: Outcome loggen BEVOR die Position aus dem State entfernt wird
            _log_closed_position(pos, exit_price=mark_price, exit_reason="liquidation")
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
            # K-C-Fix: Outcome loggen BEVOR die Position aus dem State entfernt wird
            exit_mark = _mark_price or await exchange.get_mark_price(pos.symbol)
            _log_closed_position(pos, exit_price=exit_mark or (pos.entry_price or 0),
                                 exit_reason="timeout")
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
        # Kraken: "order_id" | Binance/Paper: "orderId"
        order_ids = {
            str(o.get("orderId") or o.get("order_id") or "")
            for o in open_orders
        }

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
