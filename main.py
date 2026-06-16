"""
main.py – Einstiegspunkt des KI-Trading-Bots (Kraken Futures).
Unterstützt --config <pfad> für Multi-Bot-Betrieb.
APScheduler-Fallback für 4h-Regime (Kraken Demo liefert keine candles_240).
"""

import argparse
import os
import sys

# ─── --config MUSS vor allen Modul-Imports gesetzt werden ─────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", default=None)
_parser.add_argument("--bot-id", default=None)
_known, _ = _parser.parse_known_args()
if _known.config:
    os.environ["BOT_CONFIG"] = _known.config
if _known.bot_id:
    os.environ["BOT_ID"] = _known.bot_id
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import logging
import signal
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
from config import config  # noqa: E402  (nach env-Setup)

log_dir = Path(config.paths.get("log_dir", "logs"))
log_dir.mkdir(parents=True, exist_ok=True)

# Jede Zeile mit [Bot NN]-Präfix → in der gemeinsamen Konsole pro Bot lesbar/filterbar.
_LOG_FORMAT = f"%(asctime)s [Bot {config.bot_id:>2}] [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

import collections
_LOG_BUFFER = collections.deque(maxlen=5000)   # gesamter Session-Log dieses Bots
_SCORING_SNAPSHOT_FILE = log_dir / "scoring_snapshot.txt"


class _BufferHandler(logging.Handler):
    """Sammelt alle Log-Zeilen für den Scoring-Snapshot (überschreibbare Textdatei)."""
    def emit(self, record):
        try:
            _LOG_BUFFER.append(self.format(record))
        except Exception:
            pass


def write_scoring_snapshot():
    """Schreibt den gesamten bisherigen Session-Log in eine Textdatei (überschreibt)."""
    try:
        _SCORING_SNAPSHOT_FILE.write_text("\n".join(_LOG_BUFFER), encoding="utf-8")
    except Exception:
        pass


_formatter      = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
_stream_handler = logging.StreamHandler(sys.stdout)
_file_handler   = logging.FileHandler(log_dir / "bot.log", encoding="utf-8")
_buf_handler    = _BufferHandler()
for _h in (_stream_handler, _file_handler, _buf_handler):
    _h.setFormatter(_formatter)

logging.basicConfig(level=logging.INFO,
                    handlers=[_stream_handler, _file_handler, _buf_handler])
logger = logging.getLogger(__name__)

# ─── Restliche Imports ────────────────────────────────────────────────────────
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
from layers.layer3_scoring import calculate_layer3_score, fetch_fear_greed_index
from scoring_core import passes_regime_gate
from logger_db import log_signal

_kill_switch_active = False

# H1-Fix: Doppel-Scoring verhindern. Sowohl der WS-Kerzenschluss-Callback als auch
# der _scoring_timer-Fallback können trigger_scoring_cycle für dasselbe Symbol fast
# gleichzeitig aufrufen → doppelte Trades/Shadows. Guard + 60s-Debounce pro Symbol.
# (Single-Prozess + asyncio single-threaded → set/dict ohne Lock sicher, da zwischen
#  Prüfung und Eintrag kein await liegt.)
_scoring_in_progress: set = set()
_last_scoring_ts: dict = {}
_SCORING_DEBOUNCE_SECS = 60


# ─── STARTUP ──────────────────────────────────────────────────────────────────

async def startup() -> bool:
    logger.info("=" * 60)
    logger.info(f"KI-Trading-Bot wird gestartet (Bot-ID: {config.bot_id})")
    logger.info(f"Modus: {config.trading_mode.upper()} | Symbole: {config.symbols}")
    logger.info(f"Strategie: {config.strategy} | Macro-Modus: {config.macro_mode}")
    logger.info("=" * 60)

    await notifier.start()
    exchange._notifier = notifier
    init_db()

    logger.info("[1/7] State laden...")
    try:
        state.load()
    except Exception as e:
        logger.critical(f"State laden fehlgeschlagen: {e}")
        return False

    logger.info("[2/7] Kontostand synchronisieren...")
    try:
        balance = await exchange.sync_account_balance(state)
        logger.info(f"Kontostand: {balance:.2f} USD")
    except Exception as e:
        logger.critical(f"Kontostand fehlgeschlagen: {e}")
        return False

    logger.info("[3/7] Symbol-Filter laden...")
    try:
        await exchange.load_symbol_filters()
    except Exception as e:
        logger.critical(f"Symbol-Filter fehlgeschlagen: {e}")
        return False

    logger.info("[4/7] Verwaiste Orders prüfen...")
    try:
        await exchange.check_orphan_orders()
    except Exception as e:
        logger.warning(f"Orphan-Check fehlgeschlagen: {e}")

    logger.info("[5/7] WebSocket starten...")
    try:
        websocket_manager.set_callbacks(
            on_kline_15m=trigger_scoring_cycle,
            on_kline_4h=trigger_regime_update,
        )
        await websocket_manager.start()
    except Exception as e:
        logger.critical(f"WebSocket-Start fehlgeschlagen: {e}")
        return False

    logger.info("[6/7] Initiale Makro-Analyse + Regime...")
    asyncio.create_task(_initial_updates())

    logger.info("[7/7] Timer-Fallbacks starten...")
    asyncio.create_task(_scoring_timer())
    asyncio.create_task(_regime_fallback_timer())  # Fallback für Kraken Demo

    notifier.send(
        f"Bot {config.bot_id} gestartet\n"
        f"Balance: {state.account_balance_usdt:.2f} USD\n"
        f"Modus: {config.trading_mode.upper()} | Strategie: {config.strategy}"
    )
    logger.info("Startup abgeschlossen")
    return True


async def _initial_updates():
    """Führt beim Start sofort Makro-Analyse und Regime-Update durch."""
    await asyncio.sleep(5)
    try:
        await calculate_layer1_macro()
        for symbol in config.symbols:
            klines_4h = await exchange.get_klines(symbol, "4h", 100)
            if len(klines_4h) >= 20:
                calculate_layer2_regime(symbol, klines_4h)
    except Exception as e:
        logger.error(f"Initiale Updates fehlgeschlagen: {e}")


# ─── SCORING-ZYKLUS (alle 15 Minuten) ─────────────────────────────────────────

async def trigger_scoring_cycle(symbol: str, kline_data: dict = None):
    """Hauptzyklus: Analyse → Risk-Gate → Trade."""
    global _kill_switch_active
    if _kill_switch_active:
        return

    # H1-Fix: Doppel-Scoring (Timer + WS-Event gleichzeitig) verhindern.
    import time as _time_module
    now_ts = _time_module.time()
    if symbol in _scoring_in_progress:
        logger.debug(f"Scoring für {symbol} läuft bereits – übersprungen")
        return
    if now_ts - _last_scoring_ts.get(symbol, 0.0) < _SCORING_DEBOUNCE_SECS:
        logger.debug(f"Scoring für {symbol} kürzlich gelaufen – Debounce")
        return
    _scoring_in_progress.add(symbol)
    _last_scoring_ts[symbol] = now_ts

    try:
        logger.info(f"Scoring-Zyklus: {symbol}")

        klines_15m = await exchange.get_klines(symbol, "15m", 200)
        if len(klines_15m) < 50:
            logger.warning(f"Zu wenig Klines: {symbol} ({len(klines_15m)})")
            return

        ticker        = await exchange.get_ticker(symbol)
        mark_price    = float(ticker.get("markPrice", ticker.get("last", 0)) or 0)
        # S6-2: Funding IMMER vom Live-Endpoint (Demo-Ticker liefert ein Artefakt,
        # z.B. ETH/SOL/XRP dauerhaft ~−0.25 %). get_funding_rate() zieht den echten
        # Live-Wert und normalisiert mit Live-markPrice. OI/vwap/Preise bleiben Demo.
        funding_rate  = await exchange.get_funding_rate(symbol)
        open_interest = float(ticker.get("openInterest", 0) or 0)
        vwap24h       = float(ticker.get("vwap24h", 0) or 0)
        high24h       = float(ticker.get("high24h", mark_price) or mark_price)
        low24h        = float(ticker.get("low24h", mark_price) or mark_price)
        fg_index      = await fetch_fear_greed_index()

        result = calculate_layer3_score(
            symbol=symbol,
            klines_15m=klines_15m,
            funding_rate=funding_rate,
            fg_index=fg_index,
            open_interest=open_interest,
            vwap24h=vwap24h,
            high24h=high24h,
            low24h=low24h,
            strategy=config.strategy,
        )

        # Fix 4 (optional, Schalter require_4h_regime_confirmation): Entry nur wenn
        # das 4h-Regime die Richtung stützt. Als Veto → wird unten als Veto-Shadow
        # getrackt (gelernt), Trade wird nicht eröffnet.
        if (result.signal and config.require_4h_regime_confirmation
                and not passes_regime_gate(result.direction, result.regime, config.strategy)):
            logger.info(f"4h-Regime-Gate: {symbol} {result.direction} blockiert "
                        f"(Regime={result.regime})")
            result.signal = False
            result.veto_reason = "regime_gate_4h"

        macro_direction = _get_macro_direction()
        if mark_price <= 0:
            mark_price = float(klines_15m[-1][4])

        sym_filters = exchange.get_symbol_filters(symbol)
        if not sym_filters:
            logger.error(f"Symbol-Filter fehlt: {symbol}")
            return

        # Risk-per-Trade-Sizing: Position so groß, dass ein SL-Treffer genau
        # risk_per_trade des Kapitals kostet – unabhängig vom Symbol-Preis.
        # qty = (Kapital × risk_per_trade) / SL-Abstand(in Preis-Einheiten)
        risk_per_trade = config.risk.get("risk_per_trade", 0.01)
        sl_mult        = config.risk.get("sl_atr_multiplier", 1.5)
        sl_distance    = (result.atr or 0) * sl_mult

        if sl_distance > 0 and state.account_balance_usdt > 0:
            raw_qty = (state.account_balance_usdt * risk_per_trade) / sl_distance
        else:
            raw_qty = 0.0

        # Sicherheits-Cap: Notional nie über max_position_size_pct × Hebel
        max_notional = (state.account_balance_usdt
                        * config.risk.get("max_position_size_pct", 0.10)
                        * config.leverage)
        if max_notional > 0 and raw_qty * mark_price > max_notional:
            raw_qty = max_notional / mark_price

        calculated_qty = round_qty(
            raw_qty, sym_filters["step_size"], sym_filters["min_qty"],
            sym_filters["min_notional"], mark_price,
        )
        if calculated_qty is None:
            calculated_qty = 0.0

        order_side = "BUY" if result.direction == "long" else "SELL"

        market = MarketData(
            atr=result.atr, atr_ratio=result.atr_ratio,
            funding_rate=funding_rate, mark_price=mark_price,
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
            macro_mode=config.macro_mode,
        )

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

        # Shadow-Trade registrieren wenn geblockt
        if result.signal and not gate_passed:
            _register_shadow_trade(
                symbol=symbol, side=order_side, mark_price=mark_price,
                result=result, reject_reason=reject_reason,
                funding_rate=funding_rate, fg_index=fg_index,
            )

        # Veto-Shadow wenn kein Signal (Chop oder ML-Veto)
        if not result.signal and result.direction:
            _register_veto_shadow(
                symbol=symbol, side=order_side, mark_price=mark_price,
                result=result, veto_reason=result.veto_reason or "no_signal",
                funding_rate=funding_rate, fg_index=fg_index,
            )

        if not result.signal:
            logger.info(f"Kein Signal: {symbol} (Score={result.score}, Regime={result.regime})")
            notifier.send_info(
                f"Kein Trade | {symbol}\n"
                f"Score: {result.score} | Regime: {result.regime}\n"
                f"Funding: {funding_rate:.4%} | F&G: {fg_index:.0f}"
            )
            return

        if not gate_passed:
            logger.info(f"Risk-Gate blockiert: {reject_reason}")
            notifier.send_warning(
                f"Trade blockiert | {symbol}\nGrund: {reject_reason}\n"
                f"Score: {result.score} | Richtung: {result.direction}"
            )
            return

        # ─── Trade eröffnen ───────────────────────────────────────────────────
        # K-B-Fix: ATR==0 → sl_price == tp_price == entry_price → sofortiger Fill/Verlust.
        # Ungültige ATR niemals zu einem Trade führen lassen.
        if not result.atr or result.atr <= 0:
            logger.warning(f"ATR ungültig ({result.atr}) für {symbol} – Trade abgebrochen")
            return

        sl_mult = config.risk.get("sl_atr_multiplier", 1.5)
        tp_mult = config.risk.get("tp_atr_multiplier", 3.0)
        entry_price = calculate_entry_price(mark_price, order_side)

        if order_side == "BUY":
            sl_price = entry_price - result.atr * sl_mult
            tp_price = entry_price + result.atr * tp_mult
        else:
            sl_price = entry_price + result.atr * sl_mult
            tp_price = entry_price - result.atr * tp_mult

        tick = sym_filters.get("tick_size", 0)
        if tick > 0:
            entry_price = round(round(entry_price / tick) * tick, 8)
            sl_price    = round(round(sl_price / tick) * tick, 8)
            tp_price    = round(round(tp_price / tick) * tick, 8)

        logger.info(
            f"Trade-Signal: {order_side} {symbol}\n"
            f"  Entry: {entry_price:.4f} | SL: {sl_price:.4f} | TP: {tp_price:.4f}\n"
            f"  Qty: {calculated_qty} | Score: {result.score} | Regime: {result.regime}"
        )

        await place_entry_order(
            symbol=symbol, side=order_side, qty=calculated_qty,
            entry_price=entry_price, sl_price=sl_price, tp_price=tp_price,
            score=result.score, atr=result.atr, regime=result.regime,
            details=result.details,
        )

    except Exception as e:
        logger.error(f"Fehler im Scoring-Zyklus {symbol}: {e}")
        log_error("main", type(e).__name__, str(e), traceback.format_exc())
    finally:
        # H1-Fix: Scoring-Guard für dieses Symbol freigeben
        _scoring_in_progress.discard(symbol)
        # Gesamten Session-Log als Textdatei sichern (bei jedem Scoring überschrieben)
        write_scoring_snapshot()


def _get_macro_direction() -> str:
    """Gibt die Makro-Richtung zurück, angepasst an den Macro-Modus."""
    base = get_cached_direction()
    if config.macro_mode == "both":
        return "both"
    if config.macro_mode == "invert":
        return {"long": "short", "short": "long", "both": "both"}.get(base, "both")
    return base  # "filter"


def _register_shadow_trade(symbol, side, mark_price, result, reject_reason,
                           funding_rate, fg_index):
    """Registriert einen blockierten Trade als Shadow-Trade."""
    try:
        from shadow_tracker import shadow_tracker
        shadow_tracker.register(
            bot_id=config.bot_id,
            symbol=symbol,
            side=side,
            entry_price=mark_price,
            sl_price=mark_price - result.atr * config.risk.get("sl_atr_multiplier", 1.5)
                     if side == "BUY"
                     else mark_price + result.atr * config.risk.get("sl_atr_multiplier", 1.5),
            tp_price=mark_price + result.atr * config.risk.get("tp_atr_multiplier", 3.0)
                     if side == "BUY"
                     else mark_price - result.atr * config.risk.get("tp_atr_multiplier", 3.0),
            score=result.score,
            regime=result.regime,
            block_reason=reject_reason,
            is_veto=False,
            funding_rate=funding_rate,
            fg_index=fg_index,
            rsi=result.details.get("_rsi", 50.0),
            details=result.details,
        )
    except Exception as e:
        logger.debug(f"Shadow-Tracker nicht verfügbar: {e}")


def _register_veto_shadow(symbol, side, mark_price, result, veto_reason,
                          funding_rate, fg_index):
    """Registriert ein vetiertes Signal (Chop/ML) als Shadow-Trade."""
    try:
        from shadow_tracker import shadow_tracker
        shadow_tracker.register(
            bot_id=config.bot_id,
            symbol=symbol,
            side=side,
            entry_price=mark_price,
            sl_price=mark_price - result.atr * config.risk.get("sl_atr_multiplier", 1.5)
                     if side == "BUY"
                     else mark_price + result.atr * config.risk.get("sl_atr_multiplier", 1.5),
            tp_price=mark_price + result.atr * config.risk.get("tp_atr_multiplier", 3.0)
                     if side == "BUY"
                     else mark_price - result.atr * config.risk.get("tp_atr_multiplier", 3.0),
            score=result.score,
            regime=result.regime,
            block_reason=veto_reason,
            is_veto=not veto_reason.startswith("regime_gate"),
            funding_rate=funding_rate,
            fg_index=fg_index,
            rsi=result.details.get("_rsi", 50.0),
            details=result.details,
        )
    except Exception as e:
        logger.debug(f"Veto-Shadow-Tracker nicht verfügbar: {e}")


# ─── REGIME-UPDATE (alle 4h) ──────────────────────────────────────────────────

async def trigger_regime_update(symbol: str, kline_data: dict = None):
    """4h-Regime-Update via ADX."""
    try:
        logger.info(f"4h-Regime-Update: {symbol}")
        klines_4h = await exchange.get_klines(symbol, "4h", 100)
        if len(klines_4h) >= 20:
            calculate_layer2_regime(symbol, klines_4h)
    except Exception as e:
        logger.error(f"Regime-Update Fehler: {e}")


# ─── TIMER-FALLBACKS ──────────────────────────────────────────────────────────

async def _scoring_timer():
    """Fallback: Scoring alle 15 Minuten unabhängig vom WebSocket."""
    import time as time_module
    now = time_module.time()
    wait = (15 * 60) - (now % (15 * 60))
    logger.info(f"Scoring-Timer: erster Tick in {wait:.0f}s")
    await asyncio.sleep(wait)

    while not _kill_switch_active:
        try:
            for symbol in config.symbols:
                asyncio.create_task(trigger_scoring_cycle(symbol))
        except Exception as e:
            logger.error(f"Scoring-Timer Fehler: {e}")
        await asyncio.sleep(15 * 60)


async def _regime_fallback_timer():
    """
    Fallback für 4h-Regime (Kraken Demo sendet keine candles_240-Events).
    Läuft alle 4 Stunden auf den vollen Stunden 0/4/8/12/16/20 UTC.
    """
    import time as time_module
    hour_secs = 4 * 3600
    now = time_module.time()
    wait = hour_secs - (now % hour_secs)
    logger.info(f"Regime-Fallback-Timer: erster Tick in {wait:.0f}s")
    await asyncio.sleep(wait)

    while not _kill_switch_active:
        try:
            for symbol in config.symbols:
                asyncio.create_task(trigger_regime_update(symbol))
        except Exception as e:
            logger.error(f"Regime-Fallback-Timer Fehler: {e}")
        await asyncio.sleep(hour_secs)


async def _macro_scheduler():
    """Makro-Update alle 12 Stunden."""
    while not _kill_switch_active:
        try:
            await asyncio.sleep(12 * 3600)
            await calculate_layer1_macro()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Makro-Update Fehler: {e}")


# ─── KILL-SWITCH ──────────────────────────────────────────────────────────────

async def execute_kill_switch():
    global _kill_switch_active
    _kill_switch_active = True
    logger.critical("KILL-SWITCH AKTIVIERT!")
    await notifier.send_critical("Bot wird gestoppt (Kill-Switch)...")

    try:
        for symbol in config.symbols:
            await exchange.cancel_all_orders(symbol)
        pos = state.open_position
        if pos.is_open:
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            await exchange.place_market_order(pos.symbol, close_side, pos.qty or 0)
            state.close_position()
        state.save()
        await notifier.send_critical("Kill-Switch ausgeführt.")
    except Exception as e:
        logger.critical(f"Kill-Switch Fehler: {e}")
    finally:
        await asyncio.sleep(2)
        sys.exit(0)


async def _telegram_kill_switch_listener():
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        return
    try:
        import aiohttp
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        last_update_id = None

        while not _kill_switch_active:
            try:
                params = {"timeout": 30, "allowed_updates": ["message"]}
                if last_update_id:
                    params["offset"] = last_update_id + 1

                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        f"https://api.telegram.org/bot{token}/getUpdates",
                        params=params, timeout=aiohttp.ClientTimeout(total=40)
                    ) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(5)
                            continue
                        data = await resp.json()
                        for update in data.get("result", []):
                            last_update_id = update.get("update_id")
                            msg = update.get("message", {})
                            text = msg.get("text", "")
                            chat_id = str(msg.get("chat", {}).get("id", ""))
                            if (text == "/killswitch" and
                                    chat_id == os.environ.get("TELEGRAM_CHAT_ID")):
                                await execute_kill_switch()
                                return
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Telegram-Polling Fehler: {e}")
                await asyncio.sleep(10)
    except Exception as e:
        logger.error(f"Kill-Switch-Listener Fehler: {e}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def main():
    success = await startup()
    if not success:
        logger.critical("Startup fehlgeschlagen – Bot beendet")
        await notifier.send_critical(f"Bot {config.bot_id} Startup fehlgeschlagen!")
        sys.exit(1)

    tasks = [
        asyncio.create_task(run_watchdog(state), name="watchdog"),
        asyncio.create_task(run_position_monitor(), name="position_monitor"),
        asyncio.create_task(_telegram_kill_switch_listener(), name="kill_switch"),
        asyncio.create_task(_macro_scheduler(), name="macro_scheduler"),
    ]

    logger.info(f"Bot {config.bot_id} läuft")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Bot wird gestoppt...")
    except Exception as e:
        logger.critical(f"Unbehandelter Fehler: {e}")
        log_error("main", type(e).__name__, str(e), traceback.format_exc())
        await notifier.send_critical(f"Kritischer Fehler Bot {config.bot_id}: {e}")
    finally:
        await websocket_manager.stop()
        await exchange.close()
        await notifier.stop()
        state.save()
        logger.info(f"Bot {config.bot_id} gestoppt")


def _handle_shutdown(signum, frame):
    logger.info(f"Signal {signum} – Bot wird gestoppt")
    for task in asyncio.all_tasks():
        task.cancel()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot durch Benutzer gestoppt")
    except Exception as e:
        logger.critical(f"Fataler Fehler: {e}")
        sys.exit(1)
