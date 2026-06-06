"""
main.py – Einstiegspunkt des KI-Trading-Bots.
Verwaltet Startup-Sequenz, Scheduler, Hauptloop und Kill-Switch.
"""

import asyncio
import logging
import os
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# .env laden (vor allen anderen Imports)
load_dotenv()

# Logging konfigurieren
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "bot.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# Module importieren
from config import config
from state import state
from exchange import exchange
from notifier import notifier
from logger_db import init_db, log_error
from watchdog import run_watchdog
from position_monitor import run_position_monitor
import websocket_manager
from order_manager import (
    place_entry_order, calculate_entry_price, round_qty
)
from risk_gate import check_all, MarketData, SymbolFilters
from layers.layer1_macro import calculate_layer1_macro, get_cached_direction
from layers.layer2_regime import calculate_layer2_regime, get_cached_regime
from layers.layer3_scoring import (
    calculate_layer3_score, fetch_fear_greed_index
)
from logger_db import log_signal

# Globales Kill-Switch-Flag
_kill_switch_active = False


# ─── STARTUP-SEQUENZ ────────────────────────────────────────────────────────

async def startup() -> bool:
    """
    Führt die vollständige Startup-Sequenz durch.
    Gibt False zurück wenn ein kritischer Fehler auftritt.
    """
    logger.info("=" * 60)
    logger.info("KI-Trading-Bot wird gestartet...")
    logger.info(f"Modus: {config.trading_mode.upper()}")
    logger.info(f"Symbole: {config.symbols}")
    logger.info("=" * 60)

    # Notifier starten
    await notifier.start()
    exchange._notifier = notifier

    # Datenbank initialisieren
    init_db()

    # Schritt 1: State laden
    logger.info("[1/10] State laden...")
    try:
        state.load()
    except Exception as e:
        logger.critical(f"State laden fehlgeschlagen: {e}")
        return False

    # Schritt 2: Offene Positionen vom Exchange holen
    logger.info("[2/10] Positionen vom Exchange abrufen...")
    try:
        exchange_positions = await exchange.get_open_positions()
        logger.info(f"Offene Positionen am Exchange: {len(exchange_positions)}")
    except Exception as e:
        logger.critical(f"Fehler beim Abrufen der Exchange-Positionen: {e}")
        return False

    # Schritt 3: State vs Exchange reconcilen
    logger.info("[3/10] State mit Exchange reconcilen...")
    try:
        await _reconcile_state_with_exchange(exchange_positions)
    except Exception as e:
        logger.error(f"Reconciliation fehlgeschlagen: {e}")
        # Kein kritischer Fehler – weitermachen

    # Schritt 4: Kontostand synchronisieren
    logger.info("[4/10] Kontostand synchronisieren...")
    try:
        balance = await exchange.sync_account_balance(state)
        logger.info(f"Kontostand: {balance:.2f} USDT")
    except Exception as e:
        logger.critical(f"Kontostand-Synchronisation fehlgeschlagen: {e}")
        return False

    # Schritt 5: Margin-Typ setzen (ISOLATED, Fehler -4046 ignorieren)
    logger.info("[5/10] Margin-Typ auf ISOLATED setzen...")
    for symbol in config.symbols:
        try:
            await exchange.set_margin_type(symbol, "ISOLATED")
        except Exception as e:
            logger.warning(f"Margin-Typ für {symbol}: {e}")

    # Schritt 6: Hebel setzen
    logger.info("[6/10] Hebel setzen...")
    for symbol in config.symbols:
        try:
            await exchange.set_leverage(symbol, config.leverage)
        except Exception as e:
            logger.warning(f"Hebel für {symbol}: {e}")

    # Schritt 7: Symbol-Filter laden und cachen
    logger.info("[7/10] Symbol-Filter laden...")
    try:
        await exchange.load_symbol_filters()
    except Exception as e:
        logger.critical(f"Symbol-Filter laden fehlgeschlagen: {e}")
        return False

    # Schritt 8: Verwaiste Orders canceln
    logger.info("[8/10] Verwaiste Orders prüfen...")
    try:
        await exchange.check_orphan_orders()
    except Exception as e:
        logger.warning(f"Orphan-Order-Check fehlgeschlagen: {e}")

    # Schritt 9: WebSocket-Verbindungen starten
    logger.info("[9/10] WebSocket-Verbindungen starten...")
    try:
        websocket_manager.set_callbacks(
            on_kline_15m=trigger_scoring_cycle,
            on_kline_4h=trigger_regime_update,
        )
        await websocket_manager.start()
    except Exception as e:
        logger.critical(f"WebSocket-Start fehlgeschlagen: {e}")
        return False

    # Schritt 10: Startup-Benachrichtigung
    logger.info("[10/10] Startup-Benachrichtigung senden...")
    pos_info = state.open_position.symbol or "keine"
    notifier.send(
        f"🚀 Bot gestartet\n"
        f"Balance: {state.account_balance_usdt:.2f} USDT\n"
        f"Offene Position: {pos_info}\n"
        f"Modus: {config.trading_mode.upper()}"
    )

    # Initiale Makro-Analyse (im Hintergrund)
    asyncio.create_task(_initial_macro_update())

    logger.info("Startup erfolgreich abgeschlossen ✓")
    return True


async def _reconcile_state_with_exchange(exchange_positions: list):
    """
    Vergleicht State mit Exchange-Positionen.
    Exchange-State gewinnt bei Abweichungen.
    """
    exchange_pos_map = {p.get("symbol"): p for p in exchange_positions}
    pos = state.open_position

    # Fall 1: State hat Position, Exchange nicht
    if pos.is_open and pos.symbol not in exchange_pos_map:
        logger.warning(f"Position {pos.symbol} im State, aber nicht am Exchange – State korrigieren")
        state.close_position()

    # Fall 2: Exchange hat Position, State nicht
    elif not pos.is_open and exchange_pos_map:
        for symbol, ex_pos in exchange_pos_map.items():
            logger.warning(f"Unbekannte Position am Exchange: {symbol} – State aktualisieren")
            side = "BUY" if float(ex_pos.get("positionAmt", 0)) > 0 else "SELL"
            entry_price = float(ex_pos.get("entryPrice", 0))
            qty = abs(float(ex_pos.get("positionAmt", 0)))

            state.open_position.symbol = symbol
            state.open_position.side = side
            state.open_position.entry_price = entry_price
            state.open_position.qty = qty
            state.write_on_event("order_filled")

            # Fehlende SL/TP neu setzen
            logger.warning("SL/TP für unbekannte Position fehlen – müssen manuell gesetzt werden!")
            await notifier.send_critical(
                f"⚠️ Unbekannte Position gefunden: {symbol} {side}\n"
                f"Einstieg: {entry_price}, Menge: {qty}\n"
                f"SL/TP fehlen – bitte prüfen!"
            )

    # Fall 3: State und Exchange stimmen überein – SL/TP prüfen
    elif pos.is_open and pos.symbol in exchange_pos_map:
        logger.info(f"Position {pos.symbol} korrekt – Prüfe SL/TP...")
        open_orders = await exchange.get_open_orders(pos.symbol)
        order_ids = {str(o.get("orderId")) for o in open_orders}

        if pos.sl_order_id and str(pos.sl_order_id) not in order_ids:
            logger.warning("SL fehlt – wird neu gesetzt")
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            from order_manager import _set_sl_with_retry
            await _set_sl_with_retry(pos.symbol, close_side, pos.sl_price or 0, pos.qty or 0)

        if pos.tp_order_id and str(pos.tp_order_id) not in order_ids:
            logger.warning("TP fehlt – wird neu gesetzt")
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            from order_manager import _set_tp_with_retry
            await _set_tp_with_retry(pos.symbol, close_side, pos.tp_price or 0)


async def _initial_macro_update():
    """Führt beim Start eine initiale Makro-Analyse durch."""
    try:
        await asyncio.sleep(5)  # Kurz warten bis alles initialisiert ist
        logger.info("Initiale Makro-Analyse...")
        direction, confidence = await calculate_layer1_macro()
        logger.info(f"Makro-Richtung: {direction} (Confidence: {confidence:.2f})")
    except Exception as e:
        logger.error(f"Initiale Makro-Analyse fehlgeschlagen: {e}")


# ─── SCORING-ZYKLUS (alle 15 Minuten) ───────────────────────────────────────

async def trigger_scoring_cycle(symbol: str, kline_data: dict = None):
    """
    Wird vom WebSocket-Manager aufgerufen wenn eine 15m-Kerze schließt.
    Führt den kompletten Analyse- und Entry-Workflow durch.
    """
    global _kill_switch_active

    if _kill_switch_active:
        return

    try:
        logger.info(f"Scoring-Zyklus gestartet: {symbol}")

        # Klines laden
        klines_15m = await exchange.get_klines(symbol, "15m", 200)
        if len(klines_15m) < 50:
            logger.warning(f"Zu wenig Klines für {symbol}")
            return

        # Funding-Rate und Fear & Greed abrufen
        funding_rate = await exchange.get_funding_rate(symbol)
        fg_index = await fetch_fear_greed_index()

        # Layer 3 Scoring (Layer 2 wird NICHT hier aufgerufen!)
        result = calculate_layer3_score(
            symbol=symbol,
            klines_15m=klines_15m,
            funding_rate=funding_rate,
            fg_index=fg_index,
        )

        # Makro-Richtung (gecacht, nicht neu berechnet)
        macro_direction = get_cached_direction()

        # Mark-Preis für Qty-Berechnung
        mark_price = float(klines_15m[-1][4])  # Close-Preis der letzten Kerze

        # Symbol-Filter holen
        sym_filters = exchange.get_symbol_filters(symbol)
        if not sym_filters:
            logger.error(f"Symbol-Filter für {symbol} nicht gefunden")
            return

        # Qty berechnen (Risikobasiert: max 10% des Kapitals)
        risk_pct = config.risk.get("max_position_size_pct", 0.10)
        raw_qty = (state.account_balance_usdt * risk_pct * config.leverage) / mark_price
        calculated_qty = round_qty(
            raw_qty,
            sym_filters["step_size"],
            sym_filters["min_qty"],
            sym_filters["min_notional"],
            mark_price,
        )

        if calculated_qty is None:
            calculated_qty = 0.0

        # Order-Seite bestimmen
        order_side = "BUY" if result.direction == "long" else "SELL"

        # Risk-Gate prüfen
        market = MarketData(
            atr=result.atr,
            atr_ratio=result.atr_ratio,
            funding_rate=funding_rate,
            mark_price=mark_price,
        )
        filters = SymbolFilters(
            min_qty=sym_filters["min_qty"],
            min_notional=sym_filters["min_notional"],
            step_size=sym_filters["step_size"],
        )

        gate_passed, reject_reason = check_all(
            state_ref=state,
            market=market,
            symbol_filters=filters,
            calculated_qty=calculated_qty,
            price=mark_price,
            order_side=order_side,
            macro_direction=macro_direction,
        )

        # Signal loggen
        log_signal(
            symbol=symbol,
            score=result.score,
            direction=result.direction or "none",
            regime=result.regime,
            macro_direction=macro_direction,
            atr=result.atr,
            atr_ratio=result.atr_ratio,
            funding_rate=funding_rate,
            fg_index=fg_index,
            action="entry" if (result.signal and gate_passed) else "skip",
            reject_reason=reject_reason,
        )

        if not result.signal:
            msg = (
                f"⏸ Kein Trade | {symbol}\n"
                f"Score: {result.score} (zu niedrig für Signal)\n"
                f"Richtung: {result.direction or 'keine'} | Regime: {result.regime}\n"
                f"Makro: {macro_direction} | FundingRate: {funding_rate:.4%}\n"
                f"F&G Index: {fg_index:.0f} | ATR-Ratio: {result.atr_ratio:.2f}"
            )
            logger.info(f"Kein Signal für {symbol} (Score: {result.score}, Richtung: {result.direction}, Regime: {result.regime})")
            notifier.send_info(msg)
            return

        if not gate_passed:
            msg = (
                f"🚫 Trade blockiert | {symbol}\n"
                f"Grund: {reject_reason}\n"
                f"Score: {result.score} | Richtung: {result.direction}\n"
                f"Makro: {macro_direction} | Regime: {result.regime}"
            )
            logger.info(f"Risk-Gate blockiert Entry: {reject_reason}")
            notifier.send_warning(msg)
            return

        # ─── Trade eröffnen ─────────────────────────────────────────────────

        # SL/TP basierend auf ATR berechnen
        sl_multiplier = config.risk.get("sl_atr_multiplier", 2.0)
        tp_multiplier = config.risk.get("tp_atr_multiplier", 3.0)

        entry_price = calculate_entry_price(mark_price, order_side)

        if order_side == "BUY":
            sl_price = entry_price - result.atr * sl_multiplier
            tp_price = entry_price + result.atr * tp_multiplier
        else:
            sl_price = entry_price + result.atr * sl_multiplier
            tp_price = entry_price - result.atr * tp_multiplier

        # Preise auf tick_size runden
        tick_size = sym_filters["tick_size"]
        if tick_size > 0:
            entry_price = round(round(entry_price / tick_size) * tick_size, 8)
            sl_price = round(round(sl_price / tick_size) * tick_size, 8)
            tp_price = round(round(tp_price / tick_size) * tick_size, 8)

        logger.info(
            f"Trade-Signal: {order_side} {symbol}\n"
            f"  Entry: {entry_price:.4f}, SL: {sl_price:.4f}, TP: {tp_price:.4f}\n"
            f"  Qty: {calculated_qty}, Score: {result.score}, Regime: {result.regime}"
        )

        # Order platzieren
        await place_entry_order(
            symbol=symbol,
            side=order_side,
            qty=calculated_qty,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            score=result.score,
            atr=result.atr,
            regime=result.regime,
        )

    except Exception as e:
        logger.error(f"Fehler im Scoring-Zyklus für {symbol}: {e}")
        log_error("main", type(e).__name__, str(e), traceback.format_exc())


# ─── REGIME-UPDATE (alle 4h) ─────────────────────────────────────────────────

async def trigger_regime_update(symbol: str, kline_data: dict = None):
    """
    Wird vom WebSocket-Manager aufgerufen wenn eine 4h-Kerze schließt.
    Führt Layer 2 (ADX-Regime-Erkennung) durch.
    """
    try:
        logger.info(f"4h-Regime-Update: {symbol}")
        klines_4h = await exchange.get_klines(symbol, "4h", 100)
        if len(klines_4h) >= 20:
            calculate_layer2_regime(symbol, klines_4h)
    except Exception as e:
        logger.error(f"Fehler beim Regime-Update: {e}")


# ─── KILL-SWITCH ─────────────────────────────────────────────────────────────

async def execute_kill_switch():
    """
    Führt den Emergency-Kill-Switch aus:
    1. Alle offenen Orders canceln
    2. Alle Positionen mit Market-Order schließen
    3. State speichern
    4. Prozess beenden
    """
    global _kill_switch_active
    _kill_switch_active = True

    logger.critical("KILL-SWITCH AKTIVIERT!")
    await notifier.send_critical("🛑 KILL-SWITCH aktiviert! Bot wird gestoppt...")

    try:
        # Alle Orders canceln
        for symbol in config.symbols:
            await exchange.cancel_all_orders(symbol)
            logger.info(f"Alle Orders gecancelt: {symbol}")

        # Offene Positionen schließen
        pos = state.open_position
        if pos.is_open:
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            await exchange.place_market_order(pos.symbol, close_side, pos.qty or 0)
            logger.info(f"Position geschlossen: {pos.symbol}")
            state.close_position()

        # State speichern
        state.save()

        await notifier.send_critical("✅ Kill-Switch ausgeführt. Bot gestoppt.")

    except Exception as e:
        logger.critical(f"Fehler beim Kill-Switch: {e}")
    finally:
        # Prozess beenden
        await asyncio.sleep(2)
        sys.exit(0)


async def _run_telegram_kill_switch_listener():
    """
    Hört auf den /killswitch Telegram-Befehl via Polling.
    """
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        logger.info("Telegram-Kill-Switch deaktiviert (kein Token)")
        return

    try:
        import aiohttp
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        last_update_id = None

        while not _kill_switch_active:
            try:
                url = f"https://api.telegram.org/bot{token}/getUpdates"
                params = {"timeout": 30, "allowed_updates": ["message"]}
                if last_update_id:
                    params["offset"] = last_update_id + 1

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params,
                                           timeout=aiohttp.ClientTimeout(total=40)) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(5)
                            continue

                        data = await resp.json()
                        updates = data.get("result", [])

                        for update in updates:
                            last_update_id = update.get("update_id")
                            message = update.get("message", {})
                            text = message.get("text", "")
                            chat_id = str(message.get("chat", {}).get("id", ""))

                            # Nur vom konfigurierten Chat-ID akzeptieren
                            if (text == "/killswitch" and
                                    chat_id == os.environ.get("TELEGRAM_CHAT_ID")):
                                logger.critical("Kill-Switch via Telegram empfangen!")
                                await execute_kill_switch()
                                return

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Telegram-Polling Fehler: {e}")
                await asyncio.sleep(10)

    except Exception as e:
        logger.error(f"Kill-Switch-Listener Fehler: {e}")


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

async def main():
    """Hauptfunktion: Startup + alle Background-Tasks starten."""

    # Startup-Sequenz
    success = await startup()
    if not success:
        logger.critical("Startup fehlgeschlagen – Bot wird beendet")
        await notifier.send_critical("💥 Bot-Startup fehlgeschlagen! Bitte Logs prüfen.")
        sys.exit(1)

    # Background-Tasks starten
    tasks = [
        asyncio.create_task(run_watchdog(state), name="watchdog"),
        asyncio.create_task(run_position_monitor(), name="position_monitor"),
        asyncio.create_task(_run_telegram_kill_switch_listener(), name="kill_switch"),
        asyncio.create_task(_macro_update_scheduler(), name="macro_scheduler"),
    ]

    logger.info("Alle Background-Tasks gestartet – Bot läuft")

    try:
        # Auf Beendigung warten
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Bot wird gestoppt...")
    except Exception as e:
        logger.critical(f"Unbehandelter Fehler im Main-Loop: {e}")
        log_error("main", type(e).__name__, str(e), traceback.format_exc())
        await notifier.send_critical(f"💥 Kritischer Fehler: {e}")
    finally:
        # Cleanup
        await websocket_manager.stop()
        await exchange.close()
        await notifier.stop()
        state.save()
        logger.info("Bot gestoppt")


async def _macro_update_scheduler():
    """Aktualisiert die Makro-Analyse alle 12 Stunden."""
    while not _kill_switch_active:
        try:
            await asyncio.sleep(12 * 3600)  # 12 Stunden warten
            logger.info("Geplante Makro-Aktualisierung...")
            await calculate_layer1_macro()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Fehler bei Makro-Aktualisierung: {e}")


def _handle_shutdown(signum, frame):
    """Signal-Handler für graceful shutdown."""
    logger.info(f"Signal {signum} empfangen – Bot wird gestoppt")
    for task in asyncio.all_tasks():
        task.cancel()


if __name__ == "__main__":
    # Signal-Handler registrieren
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # Bot starten
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot durch Benutzer gestoppt")
    except Exception as e:
        logger.critical(f"Fataler Fehler: {e}")
        sys.exit(1)
