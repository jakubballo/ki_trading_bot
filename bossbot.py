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

ABGUCK-LOGIK
  - Ranking (stündlich): pro Coin die beste STRATEGIE – gepoolt über beide
    Varianten (Standard + Aggressiv).
      * PAPER (jetzt): KEIN Mindest-Trade-Gate. Folge jeder Strategie mit
        gepoolter Winrate > BOSSBOT_MIN_WINRATE (70 %). Mehrere über 70 % →
        die mit den MEISTEN Trades. Bewusst rausch-tolerant, um zu handeln und
        Daten zu sammeln.
      * LIVE-TODO (Echtgeld): zusätzlich Robustheits-Gate min. BOSSBOT_MIN_TRADES
        (z.B. 15) gepoolte Trades UND WR-Schwelle (z.B. ≥ 50 %). Qualifiziert
        nichts beides → BossBot WARTET (lieber nicht handeln als mittelmäßig
        kopieren). Siehe Abschnitt LIVE-TODO unten.
  - Mirroring (alle paar Sekunden): öffnet einer der beiden Varianten-Bots der
    Gewinner-Strategie frisch (<Frische-Fenster) eine Position, spiegelt der
    BossBot sie – mit eigenem budget-basiertem Sizing und eigenem SL/TP
    (Abstände vom Vorbild kopiert), max. 1 Position pro Coin.

LIVE-TODO (vor Echtgeld umsetzen) — Robustes Ranking statt rausch-tolerant:
  Im Paper-Modus folgt der BossBot bewusst jeder Strategie mit WR > 70 % OHNE
  Mindest-Trade-Zahl, damit überhaupt gehandelt wird und Daten reinkommen. Das
  ist für Echtgeld zu rauschanfällig: WR 6/7 = 86 % ist fast sicher Glück.
  Beim Live-Umstieg daher umstellen auf:
    (1) Mindest-Trade-Gate aktivieren (z.B. MIN_TRADES = 15 gepoolt) → nur
        verlässliche Stichproben.
    (2) WR-Schwelle (z.B. ≥ 50 %, deutlich über Break-even ~33 % bei R:R 1:2).
    (3) Erfüllt KEINE Strategie×Coin beides → WARTEN statt das am wenigsten
        schlechte kopieren. Nicht-handeln ist bei einem System ohne belegten
        Edge die korrekte, kapitalerhaltende Wahl.
  Optional Stufe 2: innerhalb der Gewinner-Strategie zwischen Standard/Aggressiv
  wählen — erst sinnvoll, wenn JEDE Variante ~30–50 Trades hat (sonst Rauschen).

BUDGET
  - Startkapital BOSSBOT_START_CAPITAL (default 100), persistent, wird NIE
    zurückgesetzt. Jeder geschlossene Trade rechnet PnL auf/ab.
  - Position-Sizing: Risk-per-Trade, gedeckelt durch freies Kapital und
    Börsen-Mindestgröße. Reicht das Budget nicht → Signal verworfen, warten bis
    eine Position schließt.
  - Max. BOSSBOT_MAX_POSITIONS gleichzeitig offen.

EXIT (Phase 1)
  - Eigenes SL/TP (Abstände vom Vorbild-Bot) + Max-Hold-Timeout.
  - Live: SL/TP liegen als echte Orders auf Kraken. (Exaktes Mit-Schließen,
    wenn der Vorbild-Bot früher aussteigt → Phase 2.)
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
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
START_CAPITAL     = _envf("BOSSBOT_START_CAPITAL", 100.0)
LEVERAGE          = _envf("BOSSBOT_LEVERAGE", 3.0)
MAX_POSITIONS     = _envi("BOSSBOT_MAX_POSITIONS", 2)
RISK_PER_TRADE    = _envf("BOSSBOT_RISK_PER_TRADE", 0.02)   # 2 % des Kapitals je Trade
FEE_RATE          = _envf("BOSSBOT_FEE_RATE", 0.0005)        # 0,05 % je Seite
MIN_TRADES        = _envi("BOSSBOT_MIN_TRADES", 15)          # (LIVE-TODO) Robustheits-Gate, im Paper-Modus AUS
MIN_WINRATE       = _envf("BOSSBOT_MIN_WINRATE", 0.70)        # Paper: folge nur Strategien mit WR > 70 %
MAX_HOLD_HOURS    = _envf("BOSSBOT_MAX_HOLD_HOURS", 48.0)
FRESHNESS_SEC     = _envf("BOSSBOT_FRESHNESS_SEC", 90.0)     # nur frische Opens spiegeln
MAX_SINGLE_RISK   = _envf("BOSSBOT_MAX_SINGLE_RISK", 0.10)   # ein Trade max. 10 % Kapital-Risiko
MAX_DRAWDOWN_PCT  = _envf("BOSSBOT_MAX_DRAWDOWN_PCT", 0.30)  # Kill-Switch bei -30 % vom Start
RANK_REFRESH_SEC  = _envf("BOSSBOT_RANK_REFRESH_SEC", 3600.0)
LOOP_SEC          = _envf("BOSSBOT_LOOP_SEC", 10.0)
N_BOTS            = _envi("BOSSBOT_N_BOTS", 50)

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


def best_strategy_per_coin(bot_meta: dict[int, tuple[str, str]]) -> dict[str, dict]:
    """
    Pro Coin die Strategie, der gefolgt wird – GEPOOLT über die beiden Varianten
    (Standard + Aggressiv).

    PAPER-MODUS (jetzt): KEIN Mindest-Trade-Gate. Eine Strategie×Coin qualifiziert,
    wenn ihre gepoolte Winrate > MIN_WINRATE (Default 70 %) ist. Gibt es pro Coin
    mehrere über 70 %, gewinnt die mit den MEISTEN Trades (Tiebreak: höhere WR).
    Bewusst rausch-tolerant, um im Paper-Modus überhaupt zu handeln und Daten zu
    sammeln. Das robustere Gate (min. Trades + WR-Schwelle) ist als LIVE-TODO
    dokumentiert und greift erst beim Echtgeld-Umstieg.

    Gibt {symbol: {strategy, bot_ids, win_rate, total}} zurück.
    """
    # bot_ids je (symbol, strategy) gruppieren
    groups: dict[tuple[str, str], list[int]] = {}
    for bid, (sym, strat) in bot_meta.items():
        groups.setdefault((sym, strat), []).append(bid)

    by_coin: dict[str, dict] = {}
    conn = get_connection()
    try:
        for (sym, strat), bids in groups.items():
            placeholders = ",".join("?" * len(bids))
            row = conn.execute(f"""
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
                FROM trades_network
                WHERE is_shadow=0 AND is_synthetic=0
                  AND exit_price IS NOT NULL AND pnl IS NOT NULL
                  AND bot_id IN ({placeholders})
            """, bids).fetchone()
            total = (row["total"] if row else 0) or 0
            if total <= 0:
                continue
            win_rate = (row["wins"] or 0) / total
            if win_rate <= MIN_WINRATE:
                continue  # Paper: nur Strategien mit WR > 70 %
            cand = {"strategy": strat, "bot_ids": sorted(bids),
                    "win_rate": win_rate, "total": total}
            cur = by_coin.get(sym)
            # bei mehreren >70 %: die mit MEHR Trades (Tiebreak: höhere WR)
            if cur is None or (total, win_rate) > (cur["total"], cur["win_rate"]):
                by_coin[sym] = cand
    finally:
        conn.close()
    return by_coin


# ─── Vorbild-Positionen aus state.json lesen ─────────────────────────────────

def read_open_position(bot_id: int) -> dict | None:
    """
    Liest die offene Position eines Bots aus data/bot{id}/state.json.
    Gibt None zurück, wenn keine offen / Datei fehlt / unleserlich.
    """
    f = STATE_DIR / f"bot{bot_id}" / "state.json"
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
        # symbol → {strategy, bot_ids, win_rate, total}
        self.follow: dict[str, dict] = {}
        self._mirrored_keys: set[str] = set()  # (bot_id|entry_time) bereits gespiegelt
        self._last_rank_ts = 0.0

    # ── Ranking ──────────────────────────────────────────────────────────────

    def refresh_ranking(self):
        self.follow = best_strategy_per_coin(self.bot_meta)
        if self.follow:
            lines = [f"{s}: {d['strategy']} {d['bot_ids']} "
                     f"(WR {d['win_rate']*100:.0f} %, {d['total']} Trades gepoolt)"
                     for s, d in sorted(self.follow.items())]
            logger.info("Follow-Set aktualisiert | " + " | ".join(lines))
        else:
            logger.info(f"Follow-Set leer – noch keine Strategie×Coin mit "
                        f"WR > {MIN_WINRATE*100:.0f} %.")

    # ── Sizing ─────────────────────────────────────────────────────────────────

    def compute_qty(self, symbol: str, entry: float, sl_price: float) -> float:
        """
        Risk-per-Trade-Sizing, gedeckelt durch freies Kapital + Mindestgröße.
        Gibt 0.0 zurück, wenn nicht (sicher) finanzierbar → Signal überspringen.
        """
        filt = self.client.get_symbol_filters(symbol) or {}
        step = float(filt.get("step_size", 0.01))
        min_qty = float(filt.get("min_qty", step))

        sl_dist = abs(entry - sl_price)
        if sl_dist <= 0 or entry <= 0:
            return 0.0

        free = self.ledger.free_capital()
        if free <= 0:
            return 0.0

        risk_amount = self.ledger.capital * RISK_PER_TRADE
        raw_qty = risk_amount / sl_dist

        # Budget-Deckel: Margin (= Notional/Hebel) darf freies Kapital nicht übersteigen
        max_qty_budget = (free * LEVERAGE) / entry
        qty = min(raw_qty, max_qty_budget)

        # auf Step abrunden
        qty = math.floor(qty / step) * step
        if qty < min_qty:
            qty = min_qty  # Mindestgröße versuchen

        margin = qty * entry / LEVERAGE
        if margin > free:
            return 0.0  # nicht finanzierbar → warten
        if qty * sl_dist > self.ledger.capital * MAX_SINGLE_RISK:
            return 0.0  # zu groß fürs Budget (z.B. 1 XBT-Kontrakt bei 100 €) → überspringen

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

        qty = self.compute_qty(symbol, mark, sl_price)
        if qty <= 0:
            logger.info(f"{symbol}: nicht finanzierbar (frei {self.ledger.free_capital():.2f}) "
                        f"– warte bis Kapital frei wird.")
            return

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
        }
        self.ledger.add_position(pos)

        logger.info(f"GEÖFFNET {symbol} {side} qty={qty} @ {mark:.4f} "
                    f"SL {sl_price:.4f} TP {tp_price:.4f} (Vorbild Bot #{source_bot_id}, "
                    f"Margin {margin:.2f}, frei {self.ledger.free_capital():.2f})")
        emoji = "🟢" if side == "BUY" else "🔴"
        tg.send(
            f"📈 <b>BossBot ÖFFNET</b> {emoji} [{MODE.upper()}]\n"
            f"{symbol} <b>{side}</b> @ {mark:.4f}\n"
            f"Menge: {qty} | Hebel: {LEVERAGE:.0f}x | Margin: {margin:.2f}\n"
            f"SL: {sl_price:.4f} | TP: {tp_price:.4f}\n"
            f"Abgeguckt von Bot #{source_bot_id}\n"
            f"Freies Kapital: {self.ledger.free_capital():.2f} / {self.ledger.capital:.2f}\n"
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
        if len(self.ledger.positions) >= MAX_POSITIONS:
            return
        for symbol, info in self.follow.items():
            if len(self.ledger.positions) >= MAX_POSITIONS:
                break
            if self.ledger.has_symbol(symbol):
                continue  # nicht doppelt auf demselben Coin
            # Gewinner-Strategie pro Coin: BEIDE Varianten-Bots beobachten,
            # spiegeln wir den, der gerade frisch öffnet (max. 1 pro Coin).
            for bot_id in info["bot_ids"]:
                op = read_open_position(bot_id)
                if not op or op.get("symbol") != symbol:
                    continue
                key = f"{bot_id}|{op.get('entry_time_utc')}"
                if key in self._mirrored_keys:
                    continue
                if _age_seconds(op.get("entry_time_utc")) > FRESHNESS_SEC:
                    continue  # zu alt – verpasst / Carry-over
                self._mirrored_keys.add(key)
                await self.open_position(bot_id, op)
                break  # nur eine Position pro Coin

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
        follow_txt = ", ".join(f"{s.replace('PF_','').replace('USD','')}→{d['strategy']}"
                               for s, d in sorted(self.follow.items())) or "noch leer"
        tg.send(
            f"🤖 <b>BossBot gestartet</b> [{MODE.upper()}]\n"
            f"Kapital: {self.ledger.capital:.2f} (Start {self.ledger.start_capital:.0f})\n"
            f"Offen: {len(self.ledger.positions)} | Max: {MAX_POSITIONS} | "
            f"Hebel: {LEVERAGE:.0f}x | Risk/Trade: {RISK_PER_TRADE*100:.0f} %\n"
            f"Follow-Set: {follow_txt}\n"
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
