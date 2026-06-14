"""
notifier.py – Telegram-Benachrichtigungen für den Trading Bot.
Unterstützt verschiedene Prioritätsstufen mit Rate-Limiting (max 1 Nachricht / 5 Sekunden).
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Telegram-Zugangsdaten aus Umgebungsvariablen
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Rate-Limit: nicht öfter als 1 Nachricht pro 5 Sekunden
RATE_LIMIT_SECONDS = 5

# Quiet-Mode: pro Bot NUR Trade-Eröffnung/-Schließung + kritische Alerts senden.
# Routine-Meldungen (Info/Warnung/Startup je Bot) werden unterdrückt.
# Netzwerk-Start + Shutdown-Report kommen zentral aus network_manager.py.
QUIET_MODE = True


def send_telegram_sync(text: str) -> bool:
    """
    Synchroner Telegram-Sender (ohne Queue/Worker) – für network_manager.py.
    Genau eine Nachricht, mit Retry bei 429. Nutzt requests.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not (token and chat):
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat, "text": text,
                   "parse_mode": "HTML", "disable_web_page_preview": True}
        for attempt in range(4):
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            if resp.status_code == 429:
                try:
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                except Exception:
                    retry_after = 5
                time.sleep(retry_after)
            else:
                return False
        return False
    except Exception:
        return False

# Emoji für verschiedene Nachrichtentypen
EMOJI = {
    "info": "ℹ️",
    "warning": "⚠️",
    "critical": "🚨",
    "trade_opened": "📈",
    "trade_closed": "📊",
    "heartbeat": "💚",
}


class TelegramNotifier:
    """
    Versendet Telegram-Nachrichten mit Rate-Limiting.
    Verwendet aiohttp für asynchrone Requests.
    """

    def __init__(self):
        self._last_send_time: float = 0.0
        self._queue: asyncio.Queue = None
        self._worker_task: Optional[asyncio.Task] = None
        self._enabled: bool = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

        if not self._enabled:
            logger.warning("Telegram nicht konfiguriert – Benachrichtigungen deaktiviert")

    async def start(self):
        """Startet den Nachrichten-Worker."""
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._message_worker())
        logger.info("Telegram-Notifier gestartet")

    async def stop(self):
        """Stoppt den Nachrichten-Worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _message_worker(self):
        """Worker-Task: verarbeitet die Nachrichten-Queue mit Rate-Limiting."""
        while True:
            try:
                message = await self._queue.get()

                # Rate-Limiting: min. 5 Sekunden zwischen Nachrichten
                elapsed = time.monotonic() - self._last_send_time
                if elapsed < RATE_LIMIT_SECONDS:
                    await asyncio.sleep(RATE_LIMIT_SECONDS - elapsed)

                await self._send_raw(message)
                self._last_send_time = time.monotonic()
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Telegram-Worker: {e}")

    async def _send_raw(self, text: str):
        """Sendet eine Nachricht direkt an Telegram API. Retry bei 429."""
        if not self._enabled:
            logger.debug(f"[TELEGRAM DEAKTIVIERT] {text}")
            return

        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }

            for attempt in range(4):
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            logger.debug("Telegram-Nachricht gesendet")
                            return
                        elif resp.status == 429:
                            try:
                                data = await resp.json()
                                retry_after = data.get("parameters", {}).get("retry_after", 5)
                            except Exception:
                                retry_after = 5
                            logger.warning(f"Telegram Rate-Limit – warte {retry_after}s (Versuch {attempt + 1}/4)")
                            await asyncio.sleep(retry_after)
                        else:
                            resp_text = await resp.text()
                            logger.error(f"Telegram API Fehler {resp.status}: {resp_text}")
                            return

        except Exception as e:
            logger.error(f"Fehler beim Senden der Telegram-Nachricht: {e}")

    def _enqueue(self, text: str):
        """Fügt eine Nachricht zur Queue hinzu (thread-safe)."""
        if self._queue is not None:
            try:
                self._queue.put_nowait(text)
            except asyncio.QueueFull:
                logger.warning("Telegram-Queue voll – Nachricht verworfen")
        else:
            # Fallback: direkt loggen wenn Queue noch nicht initialisiert
            logger.info(f"[TELEGRAM PENDING] {text}")

    async def _enqueue_and_wait(self, text: str):
        """Enqueued Nachricht und wartet auf Verarbeitung (für kritische Alerts)."""
        if self._queue is not None:
            await self._queue.put(text)
        else:
            await self._send_raw(text)

    # ─── Öffentliche API ────────────────────────────────────────────────────

    def send_info(self, message: str):
        """Info-Nachricht. Im Quiet-Mode unterdrückt (nur Log)."""
        if QUIET_MODE:
            logger.debug(f"[TELEGRAM QUIET/info] {message}")
            return
        text = f"{EMOJI['info']} <b>INFO</b>\n{message}\n<i>{_timestamp()}</i>"
        self._enqueue(text)

    def send_warning(self, message: str):
        """Warnung. Im Quiet-Mode unterdrückt (nur Log)."""
        if QUIET_MODE:
            logger.debug(f"[TELEGRAM QUIET/warn] {message}")
            return
        text = f"{EMOJI['warning']} <b>WARNUNG</b>\n{message}\n<i>{_timestamp()}</i>"
        self._enqueue(text)

    async def send_critical(self, message: str):
        """Kritischer Alert (Kill-Switch/Liquidation) – wird IMMER gesendet."""
        text = f"{EMOJI['critical']} <b>KRITISCH</b>\n{message}\n<i>{_timestamp()}</i>"
        await self._enqueue_and_wait(text)

    def send(self, message: str):
        """Allgemeine Nachricht. Im Quiet-Mode unterdrückt (nur Log)."""
        if QUIET_MODE:
            logger.debug(f"[TELEGRAM QUIET/send] {message}")
            return
        self._enqueue(f"🤖 {message}\n<i>{_timestamp()}</i>")

    def send_trade_opened(self, symbol: str, side: str, entry_price: float,
                          qty: float, sl: float, tp: float, score: int,
                          regime: str, balance: float):
        """Sendet eine Benachrichtigung wenn ein Trade geöffnet wird."""
        direction_emoji = "🟢" if side == "BUY" else "🔴"
        sl_pct = abs(entry_price - sl) / entry_price * 100
        tp_pct = abs(tp - entry_price) / entry_price * 100

        text = (
            f"{EMOJI['trade_opened']} <b>TRADE GEÖFFNET</b> {direction_emoji}\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Richtung: <b>{side}</b>\n"
            f"Einstieg: <b>{entry_price:.4f}</b> USDT\n"
            f"Menge: {qty}\n"
            f"SL: {sl:.4f} (-{sl_pct:.1f}%)\n"
            f"TP: {tp:.4f} (+{tp_pct:.1f}%)\n"
            f"Score: {score} | Regime: {regime}\n"
            f"Balance: {balance:.2f} USDT\n"
            f"<i>{_timestamp()}</i>"
        )
        self._enqueue(text)

    def send_trade_closed(self, symbol: str, side: str, entry_price: float,
                          exit_price: float, pnl_usdt: float, pnl_pct: float,
                          hold_hours: float, exit_reason: str, balance: float):
        """Sendet eine Benachrichtigung wenn ein Trade geschlossen wird."""
        pnl_emoji = "✅" if pnl_usdt >= 0 else "❌"
        pnl_sign = "+" if pnl_usdt >= 0 else ""

        text = (
            f"{EMOJI['trade_closed']} <b>TRADE GESCHLOSSEN</b> {pnl_emoji}\n"
            f"Symbol: <b>{symbol}</b> ({side})\n"
            f"Einstieg: {entry_price:.4f} → Ausstieg: {exit_price:.4f}\n"
            f"PnL: <b>{pnl_sign}{pnl_usdt:.2f} USDT ({pnl_sign}{pnl_pct:.2f}%)</b>\n"
            f"Haltezeit: {hold_hours:.1f}h\n"
            f"Grund: {exit_reason}\n"
            f"Balance: {balance:.2f} USDT\n"
            f"<i>{_timestamp()}</i>"
        )
        self._enqueue(text)


def _timestamp() -> str:
    """Gibt den aktuellen UTC-Zeitstempel als String zurück."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# Globale Notifier-Instanz
notifier = TelegramNotifier()
