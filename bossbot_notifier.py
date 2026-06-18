"""
bossbot_notifier.py – Eigener Telegram-Sender für den BossBot.

Bewusst getrennt von notifier.py: der BossBot nutzt ein EIGENES Bot-Token
(BOSSBOT_TELEGRAM_TOKEN / BOSSBOT_TELEGRAM_CHAT_ID), damit sich seine
Echtgeld-/Paper-Meldungen nicht mit dem 50-Bot-Netzwerk vermischen.

Synchron via requests – der BossBot ruft das aus seinem eigenen Loop auf,
eine Nachricht alle paar Sekunden, kein Rate-Limit-Problem.
"""

import logging
import os
import time
from datetime import datetime, timezone

logger = logging.getLogger("bossbot.telegram")

_TOKEN = os.environ.get("BOSSBOT_TELEGRAM_TOKEN", "")
_CHAT  = os.environ.get("BOSSBOT_TELEGRAM_CHAT_ID", "")


def is_configured() -> bool:
    return bool(_TOKEN and _CHAT)


def send(text: str) -> bool:
    """Sendet genau eine Telegram-Nachricht (HTML), mit Retry bei 429."""
    if not is_configured():
        logger.debug(f"[BOSSBOT-TG DEAKTIVIERT] {text}")
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
        payload = {
            "chat_id": _CHAT,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        for _ in range(4):
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
                logger.warning(f"Telegram-Fehler {resp.status_code}: {resp.text[:200]}")
                return False
        return False
    except Exception as e:
        logger.error(f"Telegram-Sendefehler: {e}")
        return False


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
