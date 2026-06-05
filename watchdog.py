"""
watchdog.py – Schreibt alle 60 Sekunden eine Heartbeat-Datei.
Ermöglicht externes Monitoring des Bot-Status (z.B. via cron oder Uptime-Tool).
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

HEARTBEAT_FILE = Path(config.paths["heartbeat_file"])
HEARTBEAT_INTERVAL = 60  # Sekunden


async def run_watchdog(state_ref):
    """
    Asyncio-Task: schreibt alle 60 Sekunden die Heartbeat-Datei.
    
    Args:
        state_ref: Referenz auf die globale BotState-Instanz
    """
    logger.info("Watchdog gestartet")

    while True:
        try:
            await _write_heartbeat(state_ref)
        except asyncio.CancelledError:
            logger.info("Watchdog gestoppt")
            break
        except Exception as e:
            logger.error(f"Fehler im Watchdog: {e}")

        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _write_heartbeat(state_ref):
    """Schreibt die Heartbeat-JSON-Datei."""
    # Verzeichnis anlegen falls nötig
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)

    heartbeat_data = {
        "alive": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "open_position": state_ref.open_position.symbol,
        "balance_usdt": round(state_ref.account_balance_usdt, 2),
        "daily_pnl": round(state_ref.daily.realized_pnl_usdt, 2),
        "daily_loss_pct": round(state_ref.daily.loss_pct_of_capital * 100, 3),
        "last_regime": state_ref.last_regime,
        "consecutive_negative_weeks": state_ref.weekly.consecutive_negative_weeks,
    }

    # Temporäre Datei für atomares Schreiben
    tmp_file = HEARTBEAT_FILE.with_suffix(".tmp")

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(heartbeat_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_file, HEARTBEAT_FILE)
    logger.debug(f"Heartbeat geschrieben: Position={heartbeat_data['open_position']}")
