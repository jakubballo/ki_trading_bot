"""
brain.py – Brain Bot: Koordiniert das gesamte Bot-Netzwerk.

Aufgaben:
  - PBT-Selektion (Population-Based Training): täglich 02:00 UTC
    Schlechtester Bot erbt Parameter des besten + Mutation
    Referenz-Bots 1+2 sind geschützt
  - ML-Retrain-Trigger: stündlich prüfen
  - Nightly Learning Factory: 03:00 UTC
  - Täglicher Telegram-Report: 08:00 UTC
  - LLM-Reflexion: wöchentlich Sonntag 04:00 UTC

Starten: python brain.py
"""

import asyncio
import json
import logging
import os
import random
import time
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRAIN] %(levelname)s %(message)s",
)
logger = logging.getLogger("brain")

BOTS_DIR          = Path("bots")
PROTECTED_BOT_IDS = {1, 2}  # Referenz-Bots niemals überschreiben
# H6-Fix: 20 war unerreichbar (bei ~3 Trades/Tag im Gesamtnetz bräuchte jeder Bot
# 333 Tage) → PBT de facto deaktiviert. 5 macht PBT in der Anfangsphase wirksam.
MIN_TRADES_PBT    = 5        # Mindest-Trades bevor PBT greift
# H5-Fix: PBT bewertet nur die letzten N Tage (aktuelle Performance), nicht die
# gesamte Laufzeit – sonst bleibt ein früh guter, jetzt schlechter Bot top-gerankt.
PBT_LOOKBACK_DAYS = 7

# Parameter die PBT mutieren darf
PBT_MUTABLE_KEYS = [
    "risk_per_trade",
    "min_score_long",
    "min_score_short",
    "atr_sl_multiplier",
    "atr_tp_multiplier",
    "adx_chop_threshold",
    "funding_rate_limit",
]

MUTATION_RATES = {
    "risk_per_trade":       0.001,
    "min_score_long":       0.5,
    "min_score_short":      0.5,
    "atr_sl_multiplier":    0.05,
    "atr_tp_multiplier":    0.05,
    "adx_chop_threshold":   1.0,
    "funding_rate_limit":   0.00005,
}

# K-F: Strategie-spezifische Score-Schwellen dürfen nicht cross-strategy kopiert werden
# (Breakout braucht 6, Momentum 3). Diese Keys nur übernehmen wenn gleiche Strategie.
STRATEGY_SPECIFIC_KEYS = {"min_score_long", "min_score_short"}


def _atomic_write_json(path: Path, data: dict):
    """
    K-G-Fix: Atomares Schreiben (.tmp → os.replace) verhindert korrupte JSON-Dateien
    bei Absturz mitten im Schreiben. NTFS garantiert atomares Replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ──────────────────────────── PBT ──────────────────────────── #

async def run_pbt():
    """Population-Based Training: schlechtester Bot erbt besten."""
    logger.info("PBT-Selektion gestartet")
    try:
        from network_db import get_all_bot_rankings, log_pbt_event
        # H5-Fix: nur die letzten PBT_LOOKBACK_DAYS bewerten (aktuelle Performance)
        since_iso = (datetime.now(timezone.utc)
                     - timedelta(days=PBT_LOOKBACK_DAYS)).isoformat()
        rankings = get_all_bot_rankings(min_trades=MIN_TRADES_PBT, since=since_iso)
        if len(rankings) < 2:
            logger.info(f"PBT: zu wenig bewertete Bots ({len(rankings)}), übersprungen")
            return

        # Bots die nicht geschützt sind
        eligible = [r for r in rankings if r["bot_id"] not in PROTECTED_BOT_IDS]
        if not eligible:
            return

        best_bot  = rankings[0]   # Bester (höchste Sharpe)
        worst_bot = eligible[-1]  # Schlechtester nicht-geschützter

        if best_bot["bot_id"] == worst_bot["bot_id"]:
            return

        logger.info(
            f"PBT: Bot {worst_bot['bot_id']} (Sharpe={worst_bot['sharpe']:.3f}) "
            f"→ erbt von Bot {best_bot['bot_id']} (Sharpe={best_bot['sharpe']:.3f})"
        )

        mutation = _mutate_config(best_bot["bot_id"], worst_bot["bot_id"])
        if mutation:
            log_pbt_event(
                worst_bot_id=worst_bot["bot_id"],
                best_bot_id=best_bot["bot_id"],
                mutation=mutation["delta"],
                new_config=mutation["new_config"],
            )
            logger.info(f"PBT abgeschlossen: {mutation['delta']}")

    except Exception as e:
        logger.error(f"PBT Fehler: {e}", exc_info=True)


def _mutate_config(best_id: int, worst_id: int) -> dict | None:
    """Liest beste Config, mutiert PBT_MUTABLE_KEYS, schreibt zu schlechtestem Bot."""
    best_path  = BOTS_DIR / f"bot{best_id}.json"
    worst_path = BOTS_DIR / f"bot{worst_id}.json"

    if not best_path.exists() or not worst_path.exists():
        logger.warning(f"Config nicht gefunden: {best_path} oder {worst_path}")
        return None

    with open(best_path)  as f: best_cfg  = json.load(f)
    with open(worst_path) as f: worst_cfg = json.load(f)

    new_cfg = deepcopy(worst_cfg)
    delta   = {}

    # K-F-Fix: Score-Schwellen nur kopieren wenn Donor + Empfänger dieselbe Strategie
    # haben. Sonst bekäme z.B. ein Breakout-Bot (braucht Schwelle 6) die Momentum-
    # Schwelle 3 → massenhaft Fehlsignale.
    same_strategy = best_cfg.get("strategy") == worst_cfg.get("strategy")

    for key in PBT_MUTABLE_KEYS:
        if key not in best_cfg:
            continue
        if key in STRATEGY_SPECIFIC_KEYS and not same_strategy:
            continue
        best_val  = best_cfg[key]
        noise     = MUTATION_RATES.get(key, 0)
        new_val   = best_val + random.uniform(-noise, noise)

        # Vernünftige Grenzen einhalten
        if key == "risk_per_trade":
            new_val = max(0.005, min(0.03, new_val))
        elif key in ("min_score_long", "min_score_short"):
            new_val = max(2.0, min(12.0, new_val))
        elif key in ("atr_sl_multiplier", "atr_tp_multiplier"):
            new_val = max(0.5, min(4.0, new_val))
        elif key == "adx_chop_threshold":
            new_val = max(12.0, min(25.0, new_val))
        elif key == "funding_rate_limit":
            new_val = max(0.0001, min(0.001, new_val))

        old_val = worst_cfg.get(key)
        if old_val != new_val:
            delta[key] = {"old": old_val, "new": round(new_val, 6)}
            new_cfg[key] = round(new_val, 6)

    # pbt_mutable Felder auch in pbt_mutable Block schreiben
    if "pbt_mutable" in new_cfg:
        for k in PBT_MUTABLE_KEYS:
            if k in new_cfg:
                new_cfg["pbt_mutable"][k] = new_cfg[k]

    new_cfg["_pbt_inherited_from"] = best_id
    new_cfg["_pbt_timestamp"]      = datetime.now(timezone.utc).isoformat()

    _atomic_write_json(worst_path, new_cfg)  # K-G-Fix: atomar

    return {"delta": delta, "new_config": new_cfg}


# ──────────────────────────── ML ──────────────────────────── #

async def run_ml_check():
    """Prüft ob ML-Retrain nötig, triggert wenn ja (non-blocking via Executor)."""
    try:
        from ml_network import ml_network
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ml_network.maybe_retrain)
    except Exception as e:
        logger.error(f"ML-Check Fehler: {e}")


async def run_ml_full_train():
    """Vollständiges ML-Training (nightly, non-blocking via Executor)."""
    logger.info("Nightly ML-Full-Train gestartet")
    try:
        from ml_network import ml_network
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ml_network.train_all)
    except Exception as e:
        logger.error(f"ML-Full-Train Fehler: {e}")


# ──────────────────────────── Learning Factory ──────────────────────────── #

async def run_learning_factory():
    """Startet die Learning Factory in separatem Prozess (blockiert nicht)."""
    logger.info("Nightly Learning Factory wird gestartet...")
    try:
        import subprocess
        proc = subprocess.Popen(
            ["python", "learning_factory.py", "--quick"],
            cwd=str(Path(__file__).parent),
        )
        logger.info(f"Learning Factory PID: {proc.pid}")
    except Exception as e:
        logger.error(f"Learning Factory Start Fehler: {e}")


# ──────────────────────────── Tagesreport ──────────────────────────── #

async def send_daily_report():
    """Sendet täglichen Netzwerk-Report via Telegram."""
    try:
        from network_db import get_all_bot_rankings
        from notifier import notifier

        rankings = get_all_bot_rankings(min_trades=5)
        if not rankings:
            notifier.send_info("Tagesreport: Noch keine bewerteten Bots.")
            return

        total_pnl = sum(r.get("total_pnl", 0) for r in rankings)
        best      = rankings[0]
        worst_r   = [r for r in rankings if r["bot_id"] not in PROTECTED_BOT_IDS]
        worst     = worst_r[-1] if worst_r else rankings[-1]

        lines = [
            f"Netzwerk-Report {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            f"Aktive Bots: {len(rankings)}",
            f"Gesamt-PnL: {total_pnl:+.4f}",
            f"Bester Bot: #{best['bot_id']} (Sharpe={best['sharpe']:.3f})",
            f"Schlechtester: #{worst['bot_id']} (Sharpe={worst['sharpe']:.3f})",
            "",
            "Top 5:",
        ]
        for i, r in enumerate(rankings[:5]):
            lines.append(
                f"  {i+1}. Bot#{r['bot_id']} "
                f"WR={r['win_rate']*100:.0f}% "
                f"PnL={r['total_pnl']:+.4f} "
                f"Sharpe={r['sharpe']:.3f}"
            )

        notifier.send_info("\n".join(lines))
        logger.info("Tagesreport gesendet")

    except Exception as e:
        logger.error(f"Tagesreport Fehler: {e}")


# ──────────────────────────── LLM-Reflexion ──────────────────────────── #

def _send_dashboard_telegram_report():
    """Stündlicher Status-Bericht nach jedem Dashboard-Refresh via Telegram."""
    try:
        from network_db import get_active_trades_summary
        from notifier import send_telegram_sync

        data    = get_active_trades_summary()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines   = [f"📊 <b>Dashboard-Report</b> | {now_str}"]

        # Aktive echte Trades
        real_open = data["real_open"]
        if real_open:
            lines.append(f"\n<b>Aktive Trades ({len(real_open)}):</b>")
            for t in real_open:
                try:
                    opened  = datetime.fromisoformat(t["opened_at"])
                    hold_h  = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
                    hold_str = f"{hold_h:.1f}h"
                except Exception:
                    hold_str = "?"
                lines.append(
                    f"• Bot #{t['bot_id']} {t['symbol']} {t['side']} "
                    f"@ {t['entry']:.2f} ({hold_str})"
                )
        else:
            lines.append("\n<b>Aktive Trades:</b> keine")

        # Shadow-Trades
        shadow_open = data["shadow_open"]
        if shadow_open:
            bot_ids = sorted({t["bot_id"] for t in shadow_open})
            ids_str = ", ".join(f"#{b}" for b in bot_ids[:12])
            if len(bot_ids) > 12:
                ids_str += f" +{len(bot_ids) - 12} weitere"
            lines.append(
                f"\n<b>Shadow-Trades offen ({len(shadow_open)}):</b>\n{ids_str}"
            )
        else:
            lines.append("\n<b>Shadow-Trades offen:</b> keine")

        # Tagesstatistik
        total = data["today_total"]
        wins  = data["today_wins"]
        pnl   = data["today_pnl"]
        wr    = f"{wins / total * 100:.0f}%" if total else "–"
        pnl_e = "✅" if pnl >= 0 else "❌"
        lines.append(
            f"\n<b>Heute:</b> {total} Trades | WR {wr} | {pnl_e} {pnl:+.2f} USD"
        )

        send_telegram_sync("\n".join(lines))
        logger.info("Dashboard-Telegram-Report gesendet")
    except Exception as e:
        logger.warning(f"Dashboard-Telegram-Report Fehler: {e}")


async def _run_dashboard():
    """Dashboard regenerieren + Telegram-Report senden."""
    try:
        from dashboard import generate_dashboard
        generate_dashboard()
    except Exception as e:
        logger.warning(f"Dashboard Fehler: {e}")
    _send_dashboard_telegram_report()


async def _run_data_update():
    """Daten-Update für alle Symbole."""
    logger.info("Nightly Daten-Update gestartet")
    try:
        from data_updater import run_update
        await asyncio.get_running_loop().run_in_executor(None, run_update)
    except Exception as e:
        logger.error(f"Daten-Update Fehler: {e}")


async def run_llm_reflection():
    """Wöchentliche LLM-Reflexion über Netzwerk-Performance."""
    logger.info("LLM-Reflexion gestartet")
    try:
        from network_db import get_all_bot_rankings, log_reflection_rule
        rankings = get_all_bot_rankings(min_trades=20)
        if not rankings:
            return

        summary = _build_reflection_summary(rankings)

        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Schnell + günstig
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "Du analysierst Crypto-Trading-Bot-Ergebnisse. "
                    "Gib 3 konkrete, umsetzbare Regeln als Bullet-Points aus. "
                    "Fokus: Wann war der Einstieg gut/schlecht? "
                    "Antworte auf Englisch, max 3 Sätze.\n\n" + summary
                ),
            }],
        )
        rule_text = msg.content[0].text
        log_reflection_rule(rule_text=rule_text, basis_trades=sum(r["total"] for r in rankings))
        logger.info(f"LLM-Reflexion: {rule_text[:80]}...")

    except Exception as e:
        logger.warning(f"LLM-Reflexion Fehler (nicht kritisch): {e}")


def _build_reflection_summary(rankings: list) -> str:
    """Erstellt Text-Summary der Bot-Rankings für LLM."""
    lines = ["=== Bot-Netzwerk Wöchentliche Performance ==="]
    for r in rankings[:10]:
        lines.append(
            f"Bot#{r['bot_id']}: WR={r['win_rate']*100:.0f}%, "
            f"AvgPnL={r['avg_pnl']:+.4f}, Sharpe={r['sharpe']:.3f}"
        )
    return "\n".join(lines)


# ──────────────────────────── Startup-Check / State ──────────────────────────── #

BRAIN_STATE_FILE = Path("data/brain_state.json")

# Wie oft soll jede Aufgabe maximal ausgeführt werden
TASK_INTERVALS = {
    "pbt":              timedelta(hours=24),
    "data_update":      timedelta(hours=6),
    "ml_full_train":    timedelta(hours=24),
    "learning_factory": timedelta(hours=24),
    "daily_report":     timedelta(hours=24),
    "llm_reflection":   timedelta(days=7),
}


def _load_brain_state() -> dict:
    if BRAIN_STATE_FILE.exists():
        try:
            return json.loads(BRAIN_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_task_run(task_name: str):
    state = _load_brain_state()
    state[task_name] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(BRAIN_STATE_FILE, state)  # K-G-Fix: atomar


def _is_overdue(task_name: str) -> bool:
    state = _load_brain_state()
    last_run_str = state.get(task_name)
    if not last_run_str:
        return True
    try:
        last_run = datetime.fromisoformat(last_run_str)
        return datetime.now(timezone.utc) - last_run > TASK_INTERVALS[task_name]
    except Exception:
        return True


async def _run_if_overdue(task_name: str, func):
    if _is_overdue(task_name):
        logger.info(f"Startup-Check: '{task_name}' überfällig → starte sofort")
        await func()
        _save_task_run(task_name)


# Wrapper die nach Ausführung den Timestamp speichern
async def _tracked(task_name: str, func):
    await func()
    _save_task_run(task_name)


# ──────────────────────────── Scheduler ──────────────────────────── #

async def main():
    scheduler = AsyncIOScheduler(timezone="UTC")

    # Interval-basiert (läuft unabhängig von Uhrzeit)
    scheduler.add_job(lambda: asyncio.ensure_future(_tracked("pbt",              run_pbt)),              "interval", hours=24)
    scheduler.add_job(lambda: asyncio.ensure_future(_tracked("data_update",      _run_data_update)),     "interval", hours=6)
    scheduler.add_job(lambda: asyncio.ensure_future(_tracked("ml_full_train",    run_ml_full_train)),    "interval", hours=24)
    scheduler.add_job(lambda: asyncio.ensure_future(_tracked("learning_factory", run_learning_factory)), "interval", hours=24)
    scheduler.add_job(lambda: asyncio.ensure_future(_tracked("daily_report",     send_daily_report)),    "interval", hours=24)
    scheduler.add_job(lambda: asyncio.ensure_future(_tracked("llm_reflection",   run_llm_reflection)),   "interval", days=7)

    # ML-Check und Dashboard bleiben stündlich (kurz, immer sinnvoll)
    scheduler.add_job(run_ml_check,   "interval", hours=1)
    scheduler.add_job(_run_dashboard, "interval", hours=1)

    scheduler.start()
    logger.info("Brain Bot gestartet — Startup-Checks laufen...")

    # Startup: überfällige Aufgaben sofort nachholen
    await _run_if_overdue("data_update",      _run_data_update)
    await _run_if_overdue("pbt",              run_pbt)
    await _run_if_overdue("ml_full_train",    run_ml_full_train)
    await _run_if_overdue("learning_factory", run_learning_factory)
    await _run_if_overdue("daily_report",     send_daily_report)
    await _run_if_overdue("llm_reflection",   run_llm_reflection)

    # Dashboard immer sofort bauen
    asyncio.ensure_future(_run_dashboard())

    logger.info("Brain Bot bereit (interval-basiert, keine festen Uhrzeiten)")

    try:
        while True:
            await asyncio.sleep(60)
    except (asyncio.CancelledError, KeyboardInterrupt):
        scheduler.shutdown()
        logger.info("Brain Bot gestoppt")


if __name__ == "__main__":
    asyncio.run(main())
