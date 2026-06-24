"""
bossbot_daily_report.py – Eigenständiger Tagesbericht für den BossBot.

Liest NUR lesend die bossbot_trades.db + bossbot_state.json und schickt EINE
Telegram-Nachricht mit dem heute realisierten Gewinn/Verlust. Unabhängig vom
laufenden BossBot-Prozess (kein Neustart nötig) – wird per Cron um 20:00
(Europe/Berlin) aufgerufen:

    CRON_TZ=Europe/Berlin
    0 20 * * * cd /root/ki_trading_bot && .venv/bin/python bossbot_daily_report.py >> logs/bossbot_daily.log 2>&1

"today" = Kalendertag in Europe/Berlin (DST-sicher), abgeglichen gegen die in
UTC gespeicherten closed_at-Zeitstempel.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# .env früh laden, damit BOSSBOT_TELEGRAM_TOKEN/_CHAT_ID beim Import von
# bossbot_notifier gesetzt sind (das Modul liest sie beim Import).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import bossbot_notifier as tg

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Berlin")
except Exception:
    TZ = timezone.utc

BASE = Path(__file__).resolve().parent
DB = BASE / "data" / "bossbot_trades.db"
STATE = BASE / "data" / "bossbot_state.json"


def _berlin_date(iso_utc: str):
    """closed_at (UTC-ISO) → Kalendertag in Europe/Berlin."""
    try:
        dt = datetime.fromisoformat(iso_utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(TZ).date()
    except Exception:
        return None


def build_report() -> str:
    today = datetime.now(TZ).date()
    datum = today.strftime("%d.%m.%Y")

    if not DB.exists():
        return f"📅 <b>BossBot Tagesbericht</b> – {datum}\nKeine Trade-DB gefunden.\n<i>{tg.ts()}</i>"

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    closed = conn.execute(
        "SELECT symbol, side, pnl, exit_reason, closed_at "
        "FROM bossbot_trades WHERE closed_at IS NOT NULL"
    ).fetchall()
    open_n = conn.execute(
        "SELECT COUNT(*) FROM bossbot_trades WHERE closed_at IS NULL"
    ).fetchone()[0]
    conn.close()

    todays = [r for r in closed if _berlin_date(r["closed_at"]) == today]

    n = len(todays)
    wins = [r for r in todays if (r["pnl"] or 0) > 0]
    losses = [r for r in todays if (r["pnl"] or 0) <= 0]
    net = sum(r["pnl"] or 0 for r in todays)
    win_sum = sum(r["pnl"] or 0 for r in wins)
    loss_sum = sum(r["pnl"] or 0 for r in losses)
    wr = (len(wins) / n * 100) if n else 0.0

    # Pro-Symbol-Aufschlüsselung (nur heute)
    by_sym: dict[str, float] = {}
    for r in todays:
        by_sym[r["symbol"]] = by_sym.get(r["symbol"], 0.0) + (r["pnl"] or 0)
    sym_lines = "\n".join(
        f"  • {s.replace('PF_', '').replace('USD', '')}: {p:+.2f}"
        for s, p in sorted(by_sym.items(), key=lambda kv: kv[1])
    )

    # Kapital-Kontext
    cap_line = ""
    try:
        d = json.loads(STATE.read_text(encoding="utf-8"))
        cap = d.get("capital", 0.0)
        start = d.get("start_capital", 0.0)
        total = d.get("realized_pnl", 0.0)
        cap_line = (f"\nKapital: <b>{cap:.2f}</b> (Start {start:.0f}, "
                    f"gesamt {total:+.2f})")
    except Exception:
        pass

    emoji = "✅" if net >= 0 else "❌"
    if n == 0:
        body = "Heute keine Trades geschlossen."
    else:
        body = (
            f"Heute realisiert: {emoji} <b>{net:+.2f} USD</b>\n"
            f"Trades: {n} | WR {wr:.0f}% ({len(wins)}✅ / {len(losses)}❌)\n"
            f"Gewinne +{win_sum:.2f} | Verluste {loss_sum:.2f}\n"
            f"{sym_lines}"
        )

    return (
        f"📅 <b>BossBot Tagesbericht</b> – {datum}\n"
        f"{body}\n"
        f"Offene Positionen: {open_n}"
        f"{cap_line}\n"
        f"<i>{tg.ts()}</i>"
    )


def main():
    msg = build_report()
    ok = tg.send(msg)
    print(("OK gesendet" if ok else "Telegram nicht konfiguriert/Fehler") + ":\n" + msg)


if __name__ == "__main__":
    main()
