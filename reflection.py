"""
reflection.py – LLM-Reflexion über Verlustmuster.

Destilliert Klartextregeln aus Verlustmustern → reflection_rules-Tabelle + Telegram.
Unterstützt zwei Backends:
  1. LM Studio (localhost:1234) – lokal, kostenlos (Qwen2.5-14B empfohlen)
  2. Anthropic API (claude-haiku-4-5) – Fallback wenn LM Studio nicht erreichbar

Aufruf:
  python reflection.py               # analysiert letzte 50 Verlust-Trades
  python reflection.py --days 7      # letzte 7 Tage
  python reflection.py --min 30      # mindestens 30 Verluste
"""

import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("reflection")

LM_STUDIO_URL  = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "local-model"  # LM Studio nutzt "local-model" als ID
REFLECTION_PROMPT = """Du bist ein erfahrener Algo-Trading-Analyst.
Analysiere diese Liste von Verlust-Trades eines Crypto-Futures-Bots und
destilliere GENAU 3 konkrete, umsetzbare Regeln zur Verbesserung.

Format (immer genau so):
• Regel 1: [konkrete Bedingung] → [konkrete Maßnahme]
• Regel 2: [konkrete Bedingung] → [konkrete Maßnahme]
• Regel 3: [konkrete Bedingung] → [konkrete Maßnahme]

Fokus auf: Wann war Einstieg schlecht? Welche Marktbedingungen führten zum Verlust?
Antworte auf Deutsch, max 5 Sätze pro Regel.

Verlust-Trades:\n"""


def build_loss_summary(rows: list) -> str:
    """Erstellt kompaktes Text-Summary der Verlust-Trades für LLM."""
    lines = []
    for r in rows[:50]:  # Max 50 Trades an LLM
        lines.append(
            f"Symbol={r.get('symbol')}, Seite={r.get('side')}, "
            f"Score={r.get('score')}, Regime={r.get('regime')}, "
            f"Strategy={r.get('strategy')}, "
            f"Funding={float(r.get('funding_rate') or 0):.4f}, "
            f"RSI={float(r.get('rsi') or 50):.0f}, "
            f"FG={float(r.get('fg_index') or 50):.0f}, "
            f"Grund={r.get('exit_reason')}, "
            f"PnL={float(r.get('pnl') or 0):+.4f}"
        )
    return "\n".join(lines)


def call_lm_studio(prompt: str) -> Optional[str]:
    """Ruft LM Studio lokales Modell auf."""
    try:
        resp = requests.post(
            LM_STUDIO_URL,
            json={
                "model":    LM_STUDIO_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.3,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        logger.warning(f"LM Studio Status {resp.status_code}: {resp.text[:200]}")
    except requests.exceptions.ConnectionError:
        logger.info("LM Studio nicht erreichbar (Port 1234)")
    except Exception as e:
        logger.warning(f"LM Studio Fehler: {e}")
    return None


def call_anthropic(prompt: str) -> Optional[str]:
    """Fallback: Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Anthropic API Fehler: {e}")
    return None


def run_reflection(days: Optional[int] = None, min_losses: int = 20) -> Optional[str]:
    """Hauptfunktion: analysiert Verluste, generiert Regeln, speichert in DB."""
    from network_db import get_connection, log_reflection_rule

    conn = get_connection()
    query = """
        SELECT symbol, side, score, regime, strategy, funding_rate, rsi, fg_index,
               exit_reason, pnl, opened_at
        FROM trades_network
        WHERE pnl < 0 AND exit_price IS NOT NULL
          AND is_synthetic = 0
    """
    params = []
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        query += " AND closed_at >= ?"
        params.append(cutoff)
    query += " ORDER BY closed_at DESC LIMIT 100"

    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()

    if len(rows) < min_losses:
        logger.info(f"Nur {len(rows)} Verlust-Trades (min={min_losses}) – übersprungen")
        return None

    logger.info(f"Reflektiere über {len(rows)} Verlust-Trades...")
    summary = build_loss_summary(rows)
    prompt  = REFLECTION_PROMPT + summary

    # LM Studio zuerst versuchen
    text = call_lm_studio(prompt)
    if text is None:
        logger.info("Fallback zu Anthropic API...")
        text = call_anthropic(prompt)

    if text is None:
        logger.error("Keine LLM-Antwort erhalten")
        return None

    # In DB speichern
    log_reflection_rule(rule_text=text, basis_trades=len(rows))
    logger.info(f"Reflexionsregel gespeichert ({len(rows)} Trades):\n{text[:200]}...")

    # Telegram senden
    try:
        from notifier import notifier
        notifier.send_info(
            f"LLM-Reflexion ({len(rows)} Verluste, {days or 'alle'} Tage):\n\n{text}"
        )
    except Exception:
        pass

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="Letzte N Tage")
    parser.add_argument("--min",  type=int, default=20,   help="Mindest-Verluste")
    args = parser.parse_args()
    run_reflection(days=args.days, min_losses=args.min)
