"""
network_manager.py – Orchestriert alle Bot-Prozesse im Netzwerk.

Startet und überwacht:
  1. Data Hub (data_hub.py) – WebSocket-Proxy
  2. Brain Bot (brain.py)   – PBT + ML + Reports
  3. 50 Bot-Instanzen       – je eigene Config

Watchdog: startet abgestürzte Prozesse nach 10s neu.
Health-Check: alle 30s prüfen ob Prozesse noch laufen.

Starten: python network_manager.py
Stoppen: Ctrl+C (sendet SIGTERM an alle Kinder)
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MANAGER] %(levelname)s %(message)s",
)
logger = logging.getLogger("network_manager")

BOTS_DIR   = Path("bots")
PYTHON     = sys.executable
SCRIPT_DIR = Path(__file__).parent

RESTART_DELAY      = 10   # Sekunden warten nach Crash bevor Neustart
HEALTH_CHECK_SECS  = 30   # Wie oft Prozesse geprüft werden
MAX_RESTARTS       = 20   # Maximale Neustarts pro Prozess (dann aufgeben)
BOT_START_DELAY    = 0.5  # Sekunden zwischen Bot-Starts (verhindert Thundering Herd)


class ManagedProcess:
    """Verwaltet einen Subprocess mit Restart-Logik."""

    def __init__(self, name: str, cmd: list, env: dict = None):
        self.name      = name
        self.cmd       = cmd
        self.env       = env or {}
        self.proc:     Optional[subprocess.Popen] = None
        self.restarts: int = 0
        self.start_time: float = 0.0
        self.stopped:  bool = False

    def start(self):
        if self.stopped:
            return
        env = {**os.environ, **self.env}
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(SCRIPT_DIR),
            env=env,
        )
        self.start_time = time.time()
        logger.info(f"Gestartet: {self.name} (PID={self.proc.pid})")

    def is_running(self) -> bool:
        if self.proc is None:
            return False
        return self.proc.poll() is None

    def stop(self):
        self.stopped = True
        if self.proc and self.proc.poll() is None:
            logger.info(f"Stoppe: {self.name} (PID={self.proc.pid})")
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    async def restart(self):
        # H3-Fix: async statt sync. Vorher blockierte time.sleep(10) den gesamten
        # asyncio-Event-Loop des Managers (10s × Anzahl abgestürzter Prozesse).
        if self.restarts >= MAX_RESTARTS:
            logger.error(f"{self.name}: Max Neustarts ({MAX_RESTARTS}) erreicht – aufgegeben")
            self.stopped = True
            return
        self.restarts += 1
        uptime = time.time() - self.start_time
        logger.warning(
            f"Neustart #{self.restarts}: {self.name} "
            f"(war {uptime:.0f}s aktiv, Exit={self.proc.returncode if self.proc else '?'})"
        )
        await asyncio.sleep(RESTART_DELAY)
        self.start()


class NetworkManager:
    """Startet und überwacht alle Prozesse des Bot-Netzwerks."""

    def __init__(self, num_bots: int = None):
        self._processes: Dict[str, ManagedProcess] = {}
        self._num_bots = num_bots or _count_bot_configs()
        self._running  = False

    def setup(self):
        """Konfiguriert alle zu verwaltenden Prozesse."""
        # 1. Data Hub
        self._add("data_hub", [PYTHON, "data_hub.py"])

        # 2. Brain Bot
        self._add("brain", [PYTHON, "brain.py"])

        # 3. Bot-Instanzen
        for i in range(1, self._num_bots + 1):
            cfg = BOTS_DIR / f"bot{i}.json"
            if not cfg.exists():
                logger.warning(f"Config nicht gefunden: {cfg} – Bot {i} übersprungen")
                continue
            self._add(
                name=f"bot{i}",
                cmd=[PYTHON, "main.py", "--config", str(cfg.absolute())],
                env={"BOT_ID": str(i)},
            )

        logger.info(f"Network-Manager bereit: {len(self._processes)} Prozesse konfiguriert")

    def _add(self, name: str, cmd: list, env: dict = None):
        self._processes[name] = ManagedProcess(name, cmd, env)

    async def start_all(self):
        """Startet alle Prozesse mit kleinem Versatz."""
        logger.info("Starte alle Prozesse...")
        self._running = True

        # Hub und Brain zuerst
        for name in ("data_hub", "brain"):
            if name in self._processes:
                self._processes[name].start()
                await asyncio.sleep(2)  # Hub muss bereit sein bevor Bots verbinden

        # Bots mit Versatz
        for name, proc in self._processes.items():
            if name.startswith("bot"):
                proc.start()
                await asyncio.sleep(BOT_START_DELAY)

        logger.info("Alle Prozesse gestartet")

    async def monitor_loop(self):
        """Überwacht alle Prozesse und startet abgestürzte neu."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_SECS)
            dead = []
            for name, proc in self._processes.items():
                if not proc.stopped and not proc.is_running():
                    dead.append(name)

            for name in dead:
                logger.warning(f"Prozess abgestürzt: {name}")
                await self._processes[name].restart()

            active = sum(1 for p in self._processes.values() if p.is_running())
            total  = len(self._processes)
            logger.info(f"Health-Check: {active}/{total} Prozesse aktiv")

    def stop_all(self):
        """Stoppt alle Prozesse geordnet (Bots zuerst, dann Hub+Brain)."""
        self._running = False
        logger.info("Stoppe alle Bot-Prozesse...")
        for name, proc in self._processes.items():
            if name.startswith("bot"):
                proc.stop()

        for name in ("brain", "data_hub"):
            if name in self._processes:
                self._processes[name].stop()

        logger.info("Alle Prozesse gestoppt")

    def status(self) -> dict:
        return {
            name: {
                "running":   proc.is_running(),
                "restarts":  proc.restarts,
                "pid":       proc.proc.pid if proc.proc else None,
            }
            for name, proc in self._processes.items()
        }


def _count_bot_configs() -> int:
    """Zählt vorhandene Bot-Config-Dateien."""
    return len(list(BOTS_DIR.glob("bot*.json")))


def _port_open(host: str = "127.0.0.1", port: int = 8770, timeout: float = 2.0) -> bool:
    """Prüft, ob der Data-Hub-Port erreichbar ist (Verbindungs-Check)."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _fmt_bot_counts(d: dict, limit: int = 15) -> str:
    """Formatiert {bot_id: count} → 'Bot 8×1, Bot 10×1, …'."""
    if not d:
        return "–"
    items = list(d.items())[:limit]
    s = ", ".join(f"Bot {b}×{c}" for b, c in items)
    if len(d) > limit:
        s += f" +{len(d) - limit} weitere"
    return s


def _send_start_telegram(manager: "NetworkManager"):
    """Start-Status: Bots aktiv, Verbindungen (Hub/Brain), offene Trades."""
    try:
        from notifier import send_telegram_sync
        from network_db import get_open_counts

        procs = manager._processes
        total_bots  = manager._num_bots
        active_bots = sum(1 for n, p in procs.items()
                          if n.startswith("bot") and p.is_running())
        hub_proc   = procs.get("data_hub")
        brain_proc = procs.get("brain")
        hub_ok   = bool(hub_proc and hub_proc.is_running()) and _port_open()
        brain_ok = bool(brain_proc and brain_proc.is_running())

        oc = get_open_counts()
        bots_emoji  = "✅" if active_bots == total_bots else "⚠️"
        hub_emoji   = "✅" if hub_ok else "❌"
        brain_emoji = "✅" if brain_ok else "❌"

        send_telegram_sync(
            f"🟢 <b>Netzwerk gestartet – Status</b>\n"
            f"Bots aktiv: {bots_emoji} <b>{active_bots}/{total_bots}</b>\n"
            f"Data Hub: {hub_emoji} (Port 8770) | Brain: {brain_emoji}\n"
            f"Offene Trades: <b>{oc['real_open']}</b> echt | {oc['shadow_open']} Shadow\n"
            f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
    except Exception as e:
        logger.warning(f"Start-Telegram fehlgeschlagen: {e}")


def _send_shutdown_report(start_iso: str):
    """Telegram-Bericht über den gesamten Lauf beim Abschalten."""
    try:
        from notifier import send_telegram_sync
        from network_db import get_network_summary
        s = get_network_summary(since=start_iso)
        wins   = s.get("wins", 0)
        total  = s.get("real_trades", 0)
        win_rate = (wins / total * 100) if total else 0.0
        pnl = s.get("total_pnl", 0.0)
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        closed_bots = _fmt_bot_counts(s.get("closed_by_bot", {}))
        open_bots   = _fmt_bot_counts(s.get("open_by_bot", {}))
        send_telegram_sync(
            f"🔴 <b>Netzwerk gestoppt – Lauf-Bericht</b>\n"
            f"Echte Trades (geschlossen): <b>{total}</b> | WR {win_rate:.0f}% | "
            f"{pnl_emoji} <b>{pnl:+.2f} USD</b>\n"
            f"  ↳ {closed_bots}\n"
            f"Offene echte Trades: <b>{s.get('open', 0)}</b>\n"
            f"  ↳ {open_bots}\n"
            f"Shadow-Trades: {s.get('shadow_run', 0)} (offen gesamt: {s.get('shadow_open', 0)})\n"
            f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
    except Exception as e:
        logger.warning(f"Shutdown-Report fehlgeschlagen: {e}")


async def main(num_bots: int = None):
    start_iso = datetime.now(timezone.utc).isoformat()
    manager = NetworkManager(num_bots=num_bots)
    manager.setup()

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _handle_signal(*_):
        logger.info("Shutdown-Signal erhalten")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)  # Unix only
        except NotImplementedError:
            signal.signal(sig, lambda s, f: _handle_signal())  # Windows fallback

    await manager.start_all()
    monitor_task = asyncio.create_task(manager.monitor_loop())

    active = sum(1 for p in manager._processes.values()
                 if p.name.startswith("bot") and p.is_running())
    logger.info(f"Network-Manager läuft – {active}/{manager._num_bots} Bots + Hub + Brain")
    _send_start_telegram(manager)

    await stop_event.wait()

    monitor_task.cancel()
    manager.stop_all()
    _send_shutdown_report(start_iso)
    logger.info("Network-Manager beendet")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", type=int, default=None,
                        help="Anzahl Bots (Standard: alle Configs in bots/)")
    args = parser.parse_args()
    asyncio.run(main(num_bots=args.bots))
