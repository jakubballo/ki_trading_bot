"""
bossbot.py – Der BossBot: ein eigenständiger Prozess, der von den besten der
50 Paper-Bots "abguckt" und die Trades mit eigenem Budget ausführt.

KERN-PRINZIPIEN
  - Eigenständiger Prozess (eigener systemd-Service `bossbot`), unabhängig von
    den 50 Bots (`tradebot`). Liest network.db NUR lesend (Ranking) und die
    state.json der Follow-Bots (offene Positionen). Fasst die 50 Bots NICHT an.
  - Eigene DB (bossbot_trades.db), eigenes persistentes Budget-Hauptbuch
    (data/bossbot_state.json), eigener Telegram-Token.
  - Eigener Modus (BOSSBOT_MODE=paper|live), UNABHÄNGIG vom globalen
    TRADING_MODE der 50 Bots. So können die 50 weiter Paper laufen, während der
    BossBot später live geht – ohne TRADING_MODE zu kippen.

ABGUCK-LOGIK (umgebaut 2026-06-19 — „mach das, was Gewinn macht")
  - Ranking (alle RANK_REFRESH_SEC, Default 10 Min): wählt die besten
    NUM_STRATEGIES (Default 25) der 50 Bots nach **realisiertem Netto-PnL** über
    ein rollendes Fenster (RANK_WINDOW_HOURS, Default 48 h). NICHT mehr nach
    Win-Rate — der Gewinn kommt aus dem R:R, nicht aus der WR (eine 100%-WR aus
    1 Trade ist Rauschen). Gates: mind. MIN_TRADES geschlossene Trades im Fenster
    (Rausch-Filter) UND (bei ONLY_PROFITABLE) Netto-PnL > 0 → es werden NUR
    profitable Strategien kopiert. Tiebreak: mehr Trades. Dynamisch neu berechnet.
  - Mirroring (alle paar Sekunden): öffnet einer der Top-N Bots frisch
    (<FRESHNESS_SEC) eine Position UND Kapital reicht → spiegeln, mit eigenem
    SL/TP (Abstände vom Vorbild auf unseren Mark-Einstieg übertragen).
    **Mehrere Positionen auf denselben Coin erlaubt** (z.B. alle LINK-Gewinner
    parallel) — kein Pro-Bot-/Pro-Symbol-Slot-Block mehr. Begrenzt nur durch
    MAX_POSITIONS (global), freies Kapital und optional MAX_PER_SYMBOL.
  - Einstieg zum **aktuellen Mark** (live-realistisch, kein Vorbild-Preis-
    Teleport) — bewusst gewählt, damit das Paper-Ergebnis den späteren Live-
    Betrieb ehrlich abbildet (Slippage inklusive).

LIVE-TODO (vor Echtgeld nachschärfen):
    (1) MIN_TRADES auf z.B. 15 anheben (Paper-Default 5) → verlässlichere Stichproben.
    (2) Zusätzlich WR- oder Expectancy-Schwelle, falls gewünscht.
    (3) Erfüllt zu wenige Bots die Gates → weniger Slots / WARTEN statt das am
        wenigsten schlechte kopieren (kapitalerhaltend, da kein belegter Edge).

BUDGET
  - Startkapital BOSSBOT_START_CAPITAL (default 2500), persistent, wird NIE
    zurückgesetzt. Jeder geschlossene Trade rechnet PnL auf/ab.
  - Kapital wird GLEICH auf die NUM_STRATEGIES (25) Slots aufgeteilt:
    PER_STRATEGY_BUDGET = START_CAPITAL / NUM_STRATEGIES (z.B. 2500/25 = 100 €).
    Das ist die Margin/der Einsatz je Position; Notional = Margin × Hebel.
  - Reicht das freie Kapital für einen Slot nicht → Signal verworfen, warten bis
    eine Position schließt. Max. NUM_STRATEGIES Positionen gleichzeitig.

EXIT
  - Mirror-Close (BOSSBOT_MIRROR_CLOSE=1, default an): schließt der Vorbild-Bot
    DIESEN Trade (erkannt an geänderter/fehlender Vorbild-Einstiegszeit), schließt
    der BossBot mit (zum aktuellen Mark, Grund "source_closed"). Verhindert, dass
    der BossBot länger hält als der Vorbild-Bot.
  - Zusätzlich eigenes SL/TP (Abstände vom Vorbild-Bot) + Max-Hold-Timeout als
    Sicherheitsnetz — was zuerst kommt.
  - Live: SL/TP liegen als echte Orders auf Kraken; bei source_closed werden die
    Rest-Orders gecancelt.
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# UTF-8 erzwingen (Box-/Emoji-Zeichen auf Windows-cp1252 sonst Crash; No-op auf VPS)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# .env früh laden, damit BOSSBOT_*- und TRADING_MODE-Variablen schon beim
# Modul-Import verfügbar sind (wie data_hub.py).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from config import config
from exchange import ExchangeClient, BASE_URL_LIVE
from network_db import get_connection
import bossbot_db
import bossbot_notifier as tg

# ─── Konfiguration (aus Umgebung, mit Defaults) ──────────────────────────────

def _envf(key, default):
    try:
        return float(os.environ.get(key, default))
    except Exception:
        return float(default)

def _envi(key, default):
    try:
        return int(float(os.environ.get(key, default)))
    except Exception:
        return int(default)

MODE              = os.environ.get("BOSSBOT_MODE", "paper").lower()   # paper | live
START_CAPITAL     = _envf("BOSSBOT_START_CAPITAL", 2500.0)
NUM_STRATEGIES    = _envi("BOSSBOT_NUM_STRATEGIES", 25)      # Top-N Bots + Kapital-Divisor
LEVERAGE          = _envf("BOSSBOT_LEVERAGE", 3.0)
FEE_RATE          = _envf("BOSSBOT_FEE_RATE", 0.0005)        # 0,05 % je Seite
MAX_HOLD_HOURS    = _envf("BOSSBOT_MAX_HOLD_HOURS", 48.0)
FRESHNESS_SEC     = _envf("BOSSBOT_FRESHNESS_SEC", 90.0)     # nur frische Opens spiegeln
MIRROR_CLOSE      = os.environ.get("BOSSBOT_MIRROR_CLOSE", "1").lower() not in ("0", "false", "no")
MAX_DRAWDOWN_PCT  = _envf("BOSSBOT_MAX_DRAWDOWN_PCT", 0.30)  # Kill-Switch bei -30 % vom Start
RANK_REFRESH_SEC  = _envf("BOSSBOT_RANK_REFRESH_SEC", 600.0)   # alle 10 Min neu wählen (adaptiert schneller)
LOOP_SEC          = _envf("BOSSBOT_LOOP_SEC", 10.0)
N_BOTS            = _envi("BOSSBOT_N_BOTS", 50)

# Auswahl-Politik (umgebaut 2026-06-19): besten Bots nach Netto-PnL statt WR.
RANK_WINDOW_HOURS = _envf("BOSSBOT_RANK_WINDOW_HOURS", 48.0)   # rollendes Bewertungsfenster
ONLY_PROFITABLE   = os.environ.get("BOSSBOT_ONLY_PROFITABLE", "1").lower() not in ("0", "false", "no")
MAX_PER_SYMBOL    = _envi("BOSSBOT_MAX_PER_SYMBOL", 0)         # 0 = unbegrenzt (mehrere Pos. je Coin)

# Modell-B-Filter (2026-06-24): BossBot ist die EXPLOIT-Schicht. Die 50 Bots handeln
# bei veto_threshold=0.42 (explore, sammeln Labels über die ganze P-Verteilung); der
# BossBot spiegelt NUR Trades, deren Modell-B P(win) >= dieser Schwelle liegt.
# OOS-Holdout 2026-06-24 (AUC 0,62): @0.50 dreht die Testwoche −551→+333 USD.
# 0 = Filter aus (jeden Open spiegeln wie bisher). Fehlt p_win am Vorbild-Open
# (alter Bot-State / A-Exploration ohne B-Score) → wird NICHT gespiegelt (fail-closed).
B_THRESHOLD       = _envf("BOSSBOT_B_THRESHOLD", 0.50)

# Einsatz/Margin je Strategie = Startkapital gleich auf NUM_STRATEGIES aufgeteilt
# (z.B. 2500 / 25 = 100 € Margin je Position; mit LEVERAGE → Notional 100×Hebel).
PER_STRATEGY_BUDGET = START_CAPITAL / max(NUM_STRATEGIES, 1)
# Ein Slot je ausgewähltem Bot → max. gleichzeitig offene Positionen = NUM_STRATEGIES.
MAX_POSITIONS       = NUM_STRATEGIES

# Mindest-Trade-Zahl im Bewertungsfenster (Rausch-Filter). Paper 5; Live ≥15.
MIN_TRADES        = _envi("BOSSBOT_MIN_TRADES", 5)

STATE_FILE = Path("data/bossbot_state.json")
BOTS_DIR   = Path("bots")
STATE_DIR  = Path("data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOSS] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bossbot")


# ─── Budget-Hauptbuch (persistent) ───────────────────────────────────────────

class Ledger:
    """Persistentes Budget + offene Positionen. Überlebt Neustart, kein Reset."""

    def __init__(self):
        self.capital = START_CAPITAL
        self.start_capital = START_CAPITAL
        self.realized_pnl = 0.0
        self.closed_count = 0
        self.halted = False
        self.positions: list[dict] = []
        self._load()

    def _load(self):
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                self.capital = d.get("capital", START_CAPITAL)
                self.start_capital = d.get("start_capital", START_CAPITAL)
                self.realized_pnl = d.get("realized_pnl", 0.0)
                self.closed_count = d.get("closed_count", 0)
                self.halted = d.get("halted", False)
                self.positions = d.get("positions", [])
                logger.info(f"Hauptbuch geladen: Kapital {self.capital:.2f}, "
                            f"{len(self.positions)} offen, {self.closed_count} geschlossen")
            except Exception as e:
                logger.error(f"Hauptbuch-Laden fehlgeschlagen: {e} – starte mit Defaults")

    def save(self):
        data = {
            "capital": self.capital,
            "start_capital": self.start_capital,
            "realized_pnl": self.realized_pnl,
            "closed_count": self.closed_count,
            "halted": self.halted,
            "positions": self.positions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"Hauptbuch-Speichern fehlgeschlagen: {e}")

    def free_capital(self) -> float:
        """Kapital minus reservierter Margin der offenen Positionen."""
        reserved = sum(p.get("margin", 0.0) for p in self.positions)
        return self.capital - reserved

    def has_symbol(self, symbol: str) -> bool:
        return any(p["symbol"] == symbol for p in self.positions)

    def count_symbol(self, symbol: str) -> int:
        """Anzahl offener BossBot-Positionen auf diesem Coin (für MAX_PER_SYMBOL)."""
        return sum(1 for p in self.positions if p["symbol"] == symbol)

    def has_bot(self, bot_id: int) -> bool:
        """Hat dieser Vorbild-Bot bereits eine offene BossBot-Position? (nicht mehr
        als Block genutzt — Dedup läuft über _mirrored_keys; nur noch informativ.)"""
        return any(p.get("source_bot_id") == bot_id for p in self.positions)

    def add_position(self, pos: dict):
        self.positions.append(pos)
        self.save()

    def close_position(self, pos: dict, net_pnl: float):
        self.capital += net_pnl
        self.realized_pnl += net_pnl
        self.closed_count += 1
        self.positions = [p for p in self.positions if p is not pos]
        # Kill-Switch
        if self.capital <= self.start_capital * (1.0 - MAX_DRAWDOWN_PCT):
            self.halted = True
            logger.critical(f"KILL-SWITCH: Kapital {self.capital:.2f} unter "
                            f"{(1-MAX_DRAWDOWN_PCT)*100:.0f} % vom Start – keine neuen Trades.")
            tg.send(f"🛑 <b>BossBot KILL-SWITCH</b>\nKapital {self.capital:.2f} "
                    f"(-{MAX_DRAWDOWN_PCT*100:.0f} % vom Start). Keine neuen Trades mehr.\n"
                    f"<i>{tg.ts()}</i>")
        self.save()


# ─── Bot-Symbol-Mapping ──────────────────────────────────────────────────────

def load_bot_meta() -> dict[int, tuple[str, str]]:
    """bot_id → (Symbol, Strategie) aus bots/bot{id}.json."""
    mapping = {}
    for bid in range(1, N_BOTS + 1):
        f = BOTS_DIR / f"bot{bid}.json"
        if not f.exists():
            continue
        try:
            cfg = json.loads(f.read_text(encoding="utf-8"))
            sym = cfg.get("symbol")
            strat = cfg.get("strategy", "unknown")
            if sym:
                mapping[bid] = (sym, strat)
        except Exception as e:
            logger.debug(f"Bot-Config {bid} nicht lesbar: {e}")
    return mapping


def select_top_bots(bot_meta: dict[int, tuple[str, str]]) -> dict[int, dict]:
    """
    Wählt die besten NUM_STRATEGIES (Default 25) der 50 Bots nach **realisiertem
    Netto-PnL** über das rollende Fenster RANK_WINDOW_HOURS (Default 48 h).

    Gates:
      - mind. MIN_TRADES geschlossene Trades im Fenster (Rausch-Filter),
      - bei ONLY_PROFITABLE: Netto-PnL > 0 (kopiert NUR profitable Strategien).
    Sortierung: Netto-PnL desc, Tiebreak mehr Trades. `dedup_replaced`/`orphaned`
    ausgeschlossen (keine echten Outcomes). Gibt
    {bot_id: {symbol, strategy, win_rate, total, net_pnl, exp_pnl}} zurück.
    """
    cutoff = (datetime.now(timezone.utc)
              - timedelta(hours=RANK_WINDOW_HOURS)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT bot_id,
                   COUNT(*) AS total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(pnl) AS net_pnl,
                   AVG(pnl) AS exp_pnl
            FROM trades_network
            WHERE is_shadow=0 AND is_synthetic=0
              AND exit_price IS NOT NULL AND pnl IS NOT NULL
              AND exit_reason NOT IN ('dedup_replaced','orphaned')
              AND closed_at >= ?
            GROUP BY bot_id
        """, (cutoff,)).fetchall()
    finally:
        conn.close()

    cands = []
    for r in rows:
        bid = r["bot_id"]
        if bid not in bot_meta:
            continue  # nur die 50 echten Bots (kein synthetischer bot_id=0)
        total = r["total"] or 0
        if total < MIN_TRADES:
            continue  # zu kleine Stichprobe → Rauschen
        net_pnl = r["net_pnl"] or 0.0
        if ONLY_PROFITABLE and net_pnl <= 0:
            continue  # nur Bots, die im Fenster Geld gemacht haben
        win_rate = (r["wins"] or 0) / total
        sym, strat = bot_meta[bid]
        cands.append({"bot_id": bid, "symbol": sym, "strategy": strat,
                      "win_rate": win_rate, "total": total,
                      "net_pnl": net_pnl, "exp_pnl": r["exp_pnl"] or 0.0})

    # beste zuerst: höchster Netto-PnL, bei Gleichstand mehr Trades
    cands.sort(key=lambda c: (c["net_pnl"], c["total"]), reverse=True)
    top = cands[:NUM_STRATEGIES]
    return {c["bot_id"]: c for c in top}


# ─── Vorbild-Positionen aus state.json lesen ─────────────────────────────────

def read_open_position(bot_id: int) -> dict | None:
    """
    Liest die offene Position eines Bots aus data/bot{id}/bot_state.json.
    Gibt None zurück, wenn keine offen / Datei fehlt / unleserlich.
    """
    f = STATE_DIR / f"bot{bot_id}" / "bot_state.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        op = data.get("open_position", {})
        if not op or not op.get("symbol") or not op.get("entry_price"):
            return None
        return op
    except Exception:
        return None


def _age_seconds(iso_ts: str | None) -> float:
    if not iso_ts:
        return 1e9
    try:
        t = datetime.fromisoformat(iso_ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds()
    except Exception:
        return 1e9


# ─── Live-Marktpreise (immer vom Live-Public-Endpoint) ───────────────────────

class MarketData:
    def __init__(self, client: ExchangeClient):
        self._client = client
        self._marks: dict[str, float] = {}

    async def refresh(self):
        """Holt alle Mark-Preise in EINEM Request (immer live, kein Auth)."""
        try:
            data = await self._client._request(
                "GET", "/derivatives/api/v3/tickers", base_url=BASE_URL_LIVE)
            for t in data.get("tickers", []):
                mp = t.get("markPrice", t.get("last"))
                if mp:
                    self._marks[t["symbol"]] = float(mp)
        except Exception as e:
            logger.warning(f"Mark-Preis-Refresh fehlgeschlagen: {e}")

    def mark(self, symbol: str) -> float:
        return self._marks.get(symbol, 0.0)


# ─── BossBot ─────────────────────────────────────────────────────────────────

class BossBot:
    def __init__(self):
        self.ledger = Ledger()
        self.client = ExchangeClient()
        # Eigener Modus, unabhängig vom globalen TRADING_MODE der 50 Bots:
        self.client.is_paper = (MODE != "live")
        self.market = MarketData(self.client)
        self.bot_meta = load_bot_meta()
        # bot_id → {symbol, strategy, win_rate, total}  (die Top-N Bots)
        self.follow: dict[int, dict] = {}
        self._mirrored_keys: set[str] = set()  # (bot_id|entry_time) bereits gespiegelt
        self._last_rank_ts = 0.0

    # ── Ranking ──────────────────────────────────────────────────────────────

    def refresh_ranking(self):
        self.follow = select_top_bots(self.bot_meta)
        if self.follow:
            top = sorted(self.follow.values(),
                         key=lambda d: (d["net_pnl"], d["total"]), reverse=True)
            head = " | ".join(
                f"#{d['bot_id']} {d['strategy'][:4]}/{d['symbol'].replace('PF_','').replace('USD','')} "
                f"{d['net_pnl']:+.0f}USD/{d['total']}t/{d['win_rate']*100:.0f}%" for d in top[:8])
            logger.info(f"Top-{len(self.follow)} profitable Bots gewählt (Fenster "
                        f"{RANK_WINDOW_HOURS:.0f}h, Einsatz {PER_STRATEGY_BUDGET:.0f} €/Strategie). "
                        f"Beste: {head}")
        else:
            logger.info(f"Follow-Set leer – kein Bot erfüllt Gates "
                        f"(min {MIN_TRADES} Trades/{RANK_WINDOW_HOURS:.0f}h"
                        f"{', Netto-PnL>0' if ONLY_PROFITABLE else ''}).")

    # ── Sizing ─────────────────────────────────────────────────────────────────

    def compute_qty(self, symbol: str, entry: float) -> float:
        """
        Fester Slot je Strategie: Margin = PER_STRATEGY_BUDGET (z.B. 100 €),
        Notional = Margin × Hebel. Auf Step abgerundet, Mindestgröße als Untergrenze.
        Gibt 0.0 zurück, wenn nicht finanzierbar (freies Kapital reicht nicht).
        """
        filt = self.client.get_symbol_filters(symbol) or {}
        step = float(filt.get("step_size", 0.01))
        min_qty = float(filt.get("min_qty", step))
        if entry <= 0:
            return 0.0

        target_notional = PER_STRATEGY_BUDGET * LEVERAGE
        qty = math.floor((target_notional / entry) / step) * step
        if qty < min_qty:
            qty = min_qty  # Mindestgröße als Untergrenze

        margin = qty * entry / LEVERAGE
        if margin > self.ledger.free_capital():
            return 0.0  # Kapital reicht gerade nicht → warten bis ein Slot frei wird
        return round(qty, 8)

    # ── Position öffnen ────────────────────────────────────────────────────────

    async def open_position(self, source_bot_id: int, op: dict):
        symbol = op["symbol"]
        side = op["side"]                       # "BUY" / "SELL"
        ref_entry = float(op["entry_price"])
        ref_sl = float(op.get("sl_price") or 0)
        ref_tp = float(op.get("tp_price") or 0)
        if ref_sl <= 0 or ref_tp <= 0:
            logger.info(f"{symbol}: Vorbild ohne SL/TP – übersprungen.")
            return

        mark = self.market.mark(symbol)
        if mark <= 0:
            logger.info(f"{symbol}: kein Live-Mark-Preis – übersprungen.")
            return

        # SL/TP-Abstände des Vorbilds auf UNSEREN Einstieg (aktueller Mark) übertragen
        sl_dist = abs(ref_entry - ref_sl)
        tp_dist = abs(ref_tp - ref_entry)
        if side == "BUY":
            sl_price = mark - sl_dist
            tp_price = mark + tp_dist
        else:
            sl_price = mark + sl_dist
            tp_price = mark - tp_dist

        qty = self.compute_qty(symbol, mark)
        if qty <= 0:
            logger.info(f"{symbol}: nicht finanzierbar (frei {self.ledger.free_capital():.2f}) "
                        f"– warte bis ein Slot frei wird.")
            return

        strat = self.bot_meta.get(source_bot_id, (symbol, "?"))[1]

        notional = qty * mark
        margin = notional / LEVERAGE
        entry_fee = notional * FEE_RATE

        sl_oid = tp_oid = None
        if MODE == "live":
            # Echte Orders: Market-Entry + SL + TP auf Kraken
            res = await self.client.place_market_order(symbol, side, qty)
            if not res:
                logger.error(f"{symbol}: Live-Entry fehlgeschlagen – kein Trade.")
                return
            close_side = "SELL" if side == "BUY" else "BUY"
            sl_res = await self.client.place_stop_market(symbol, close_side, sl_price, close_position=True)
            tp_res = await self.client.place_take_profit_market(symbol, close_side, tp_price, close_position=True)
            sl_oid = (sl_res or {}).get("orderId")
            tp_oid = (tp_res or {}).get("orderId")

        opened_at = datetime.now(timezone.utc).isoformat()
        db_id = bossbot_db.insert_open(
            source_bot_id=source_bot_id, symbol=symbol, side=side, entry=mark,
            qty=qty, leverage=LEVERAGE, margin=margin, sl_price=sl_price,
            tp_price=tp_price, mode=MODE, opened_at=opened_at)

        pos = {
            "db_id": db_id, "source_bot_id": source_bot_id, "symbol": symbol,
            "side": side, "entry": mark, "qty": qty, "leverage": LEVERAGE,
            "margin": margin, "sl_price": sl_price, "tp_price": tp_price,
            "entry_fee": entry_fee, "opened_at": opened_at,
            "sl_order_id": sl_oid, "tp_order_id": tp_oid,
            # Einstiegszeit des Vorbilds → erkennt, wenn der Vorbild-Bot DIESEN
            # Trade schließt (Mirror-Close, sonst hielte der BossBot zu lange).
            "source_entry_time": op.get("entry_time_utc"),
        }
        self.ledger.add_position(pos)

        logger.info(f"GEÖFFNET {symbol} {side} qty={qty} @ {mark:.4f} "
                    f"SL {sl_price:.4f} TP {tp_price:.4f} (Vorbild Bot #{source_bot_id} "
                    f"{strat}, Margin {margin:.2f}, frei {self.ledger.free_capital():.2f})")
        emoji = "🟢" if side == "BUY" else "🔴"
        tg.send(
            f"📈 <b>BossBot ÖFFNET</b> {emoji} [{MODE.upper()}]\n"
            f"{symbol} <b>{side}</b> @ {mark:.4f}\n"
            f"Menge: {qty} | Hebel: {LEVERAGE:.0f}x | Margin: {margin:.2f}\n"
            f"SL: {sl_price:.4f} | TP: {tp_price:.4f}\n"
            f"Abgeguckt von Bot #{source_bot_id} ({strat})\n"
            f"Freies Kapital: {self.ledger.free_capital():.2f} / {self.ledger.capital:.2f} "
            f"({len(self.ledger.positions)}/{MAX_POSITIONS} Slots)\n"
            f"<i>{tg.ts()}</i>")

    # ── Position schließen ─────────────────────────────────────────────────────

    async def close_position(self, pos: dict, exit_price: float, reason: str):
        qty = pos["qty"]
        entry = pos["entry"]
        side = pos["side"]
        direction = 1 if side == "BUY" else -1
        gross = qty * (exit_price - entry) * direction
        exit_fee = qty * exit_price * FEE_RATE
        fees = pos.get("entry_fee", 0.0) + exit_fee
        net = gross - fees
        margin = pos.get("margin", 0.0)
        pnl_pct = (net / margin * 100.0) if margin > 0 else 0.0

        if MODE == "live":
            # Real liegt der Close schon (getriggerte SL/TP-Order); Restorder canceln.
            try:
                await self.client.cancel_all_orders(pos["symbol"])
            except Exception:
                pass

        closed_at = datetime.now(timezone.utc).isoformat()
        bossbot_db.update_close(pos["db_id"], exit_price=exit_price, pnl=net,
                                pnl_pct=pnl_pct, fees=fees, exit_reason=reason,
                                closed_at=closed_at)
        self.ledger.close_position(pos, net)

        logger.info(f"GESCHLOSSEN {pos['symbol']} {side} @ {exit_price:.4f} "
                    f"({reason}) PnL {net:+.2f} ({pnl_pct:+.1f} %) → Kapital {self.ledger.capital:.2f}")
        emoji = "✅" if net >= 0 else "❌"
        tg.send(
            f"📊 <b>BossBot SCHLIESST</b> {emoji} [{MODE.upper()}]\n"
            f"{pos['symbol']} ({side}) @ {exit_price:.4f}\n"
            f"Grund: {reason}\n"
            f"PnL: <b>{net:+.2f}</b> ({pnl_pct:+.1f} %) | Gebühren: {fees:.2f}\n"
            f"Kapital: <b>{self.ledger.capital:.2f}</b> "
            f"(Start {self.ledger.start_capital:.0f}, gesamt {self.ledger.realized_pnl:+.2f})\n"
            f"<i>{tg.ts()}</i>")

    # ── Überwachung offener Positionen ─────────────────────────────────────────

    async def monitor_positions(self):
        for pos in list(self.ledger.positions):
            mark = self.market.mark(pos["symbol"])
            if mark <= 0:
                continue

            # Mirror-Close: hat der Vorbild-Bot DIESEN Trade geschlossen (oder schon
            # einen neuen eröffnet)? Dann schließt der BossBot mit, statt ewig auf
            # sein eigenes SL/TP zu warten.
            if MIRROR_CLOSE:
                src = read_open_position(pos["source_bot_id"])
                src_et = (src.get("entry_time_utc")
                          if src and src.get("symbol") == pos["symbol"] else None)
                set_et = pos.get("source_entry_time")
                if set_et is None:
                    # Alt-Position (vor dem Feature): gleichen Trade nachtragen,
                    # sonst (Vorbild flach oder neu eröffnet) mitschließen.
                    if (src_et is not None and
                            abs(_age_seconds(src_et) - _age_seconds(pos["opened_at"])) < 300):
                        pos["source_entry_time"] = src_et
                        self.ledger.save()
                    else:
                        await self.close_position(pos, mark, "source_closed")
                        continue
                elif src_et != set_et:
                    await self.close_position(pos, mark, "source_closed")
                    continue

            side = pos["side"]
            sl = pos["sl_price"]
            tp = pos["tp_price"]
            hit = None
            if side == "BUY":
                if mark <= sl:
                    hit = (sl, "sl")
                elif mark >= tp:
                    hit = (tp, "tp")
            else:  # SELL
                if mark >= sl:
                    hit = (sl, "sl")
                elif mark <= tp:
                    hit = (tp, "tp")

            if hit is None and _age_seconds(pos["opened_at"]) > MAX_HOLD_HOURS * 3600:
                hit = (mark, "timeout")

            if hit:
                await self.close_position(pos, hit[0], hit[1])

    # ── Mirroring ──────────────────────────────────────────────────────────────

    async def mirror_new_opens(self):
        if self.ledger.halted:
            return
        # Öffnet einer der Top-N profitablen Bots frisch eine Position und Kapital
        # reicht → spiegeln. Mehrere Positionen auf denselben Coin ausdrücklich
        # erlaubt (verschiedene Vorbild-Bots auf demselben Symbol laufen parallel);
        # kein Pro-Bot-Block mehr. Dedup verhindert nur, denselben Vorbild-Trade
        # (bot_id|entry_time) doppelt zu öffnen.
        for bot_id in self.follow:
            if len(self.ledger.positions) >= MAX_POSITIONS:
                break
            op = read_open_position(bot_id)
            if not op:
                continue
            key = f"{bot_id}|{op.get('entry_time_utc')}"
            if key in self._mirrored_keys:
                continue
            if _age_seconds(op.get("entry_time_utc")) > FRESHNESS_SEC:
                continue  # zu alt – verpasst / Carry-over
            # Modell-B-Filter (exploit): nur spiegeln, wenn das Vorbild-Signal eine
            # ausreichend hohe Win-Wahrscheinlichkeit hatte. Fehlt p_win → fail-closed.
            if B_THRESHOLD > 0:
                p_win = op.get("p_win")
                if p_win is None or float(p_win) < B_THRESHOLD:
                    self._mirrored_keys.add(key)  # nicht erneut prüfen, gilt als erledigt
                    logger.info(f"Bot {bot_id} {op['symbol']} übersprungen "
                                f"(P(win)={p_win} < {B_THRESHOLD})")
                    continue
            if MAX_PER_SYMBOL > 0 and self.ledger.count_symbol(op["symbol"]) >= MAX_PER_SYMBOL:
                continue  # optionales Pro-Coin-Limit erreicht
            self._mirrored_keys.add(key)
            await self.open_position(bot_id, op)

    # ── Hauptloop ──────────────────────────────────────────────────────────────

    async def run(self):
        bossbot_db.init_db()
        await self.client.load_symbol_filters()
        self.refresh_ranking()
        self._last_rank_ts = time.monotonic()

        self._send_start_telegram()

        while True:
            try:
                now = time.monotonic()
                if now - self._last_rank_ts >= RANK_REFRESH_SEC:
                    self.refresh_ranking()
                    self._last_rank_ts = now

                await self.market.refresh()
                await self.monitor_positions()   # erst schließen → Kapital frei
                await self.mirror_new_opens()    # dann ggf. neu öffnen
            except Exception as e:
                logger.error(f"Loop-Fehler: {e}", exc_info=True)
            await asyncio.sleep(LOOP_SEC)

    def _send_start_telegram(self):
        tg.send(
            f"🤖 <b>BossBot gestartet</b> [{MODE.upper()}]\n"
            f"Kapital: {self.ledger.capital:.2f} (Start {self.ledger.start_capital:.0f})\n"
            f"Top-{NUM_STRATEGIES} Bots gewählt | Einsatz {PER_STRATEGY_BUDGET:.0f} €/Strategie | "
            f"Hebel: {LEVERAGE:.0f}x\n"
            f"Aktuell gewählt: {len(self.follow)} Bots | offen: {len(self.ledger.positions)}\n"
            f"Refresh alle {RANK_REFRESH_SEC/60:.0f} Min\n"
            f"<i>{tg.ts()}</i>")


async def main():
    if MODE == "live":
        logger.warning("=== BOSSBOT LÄUFT IM LIVE-MODUS – ECHTES GELD ===")
    else:
        logger.info("BossBot läuft im PAPER-Modus (kein echtes Geld).")
    boss = BossBot()
    try:
        await boss.run()
    finally:
        await boss.client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("BossBot gestoppt (Ctrl+C).")
