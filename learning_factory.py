"""
learning_factory.py – Learning Factory: generiert synthetische Outcomes.

Prozess:
  1. Lade historische CSV-Kerzen (data/history/{symbol}_{interval}.csv)
  2. Sweept alle Bot-Parameter-Kombinationen
  3. Berechnet Scoring via scoring_core.score_candles()
  4. Schreibt Outcomes als is_synthetic=True in network.db
  5. Parallelisiert über alle 32 CPU-Threads (Ryzen 9950X3D)

CSV-Format: timestamp,open,high,low,close,volume
(z.B. heruntergeladen via Kraken Charts API)

Aufruf:
  python learning_factory.py            # Vollständig
  python learning_factory.py --quick    # Nur letzte 3 Monate
"""

import argparse
import csv
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger("learning_factory")
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FACTORY] %(message)s",
    handlers=[
        logging.StreamHandler(),                                          # Konsole
        logging.FileHandler("logs/learning_factory.log", encoding="utf-8"),  # persistenter Log
    ],
)

HISTORY_DIR = Path("data/history")
BOTS_DIR    = Path("bots")
MAX_WORKERS = int(os.cpu_count() or 32)  # Alle Kerne nutzen

SYMBOLS = [
    "PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD",
]

STRATEGY_VARIANTS = ["momentum", "mean_reversion", "breakout", "contrarian", "scalper"]

# Schwelle je Strategie – MUSS mit den Live-Bot-Configs übereinstimmen, damit die
# synthetischen Trainingsdaten genau das abbilden, was die Bots live produzieren.
# (Sweep 2026-06-13: einheitliche Schwellen ließen momentum/scalper/contrarian fast
#  leer und über-repräsentierten breakout.)
STRATEGY_THRESHOLD = {
    "momentum": 3, "contrarian": 3, "scalper": 3,
    "mean_reversion": 5, "breakout": 6,
}

# Parameter-Sweep-Grid (nur noch SL/TP – die Schwelle kommt per Strategie oben).
PARAM_GRID = {
    "atr_sl_multiplier": [1.0, 1.5, 2.0],
    "atr_tp_multiplier": [2.0, 3.0, 4.0],
}


def load_csv_klines(path: Path, limit: int = 0) -> List[list]:
    """
    Lädt CSV-Kerzen in Binance-kompatibles 12-Element-Format.
    CSV-Spalten: timestamp,open,high,low,close,volume
    """
    klines = []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                ts    = int(float(row[0])) if row[0].replace(".", "").isdigit() else 0
                o, h, l, c, v = [float(x) for x in row[1:6]]
                # Binance-Format: [ts, o, h, l, c, v, close_ts, qav, nt, tbbav, tbqav, _]
                klines.append([ts, o, h, l, c, v, ts + 900_000, v, 0, 0, 0, 0])
    except Exception as e:
        logger.warning(f"CSV-Ladefehler {path}: {e}")
        return []

    if limit > 0:
        klines = klines[-limit:]
    return klines


def _score_window(klines_window: list, strategy: str, params: dict) -> Optional[dict]:
    """
    Berechnet Score für ein Kerzen-Fenster.
    Läuft im Worker-Prozess.
    """
    from scoring_core import score_candles
    try:
        result = score_candles(
            symbol="SYNTHETIC",
            klines=klines_window,
            funding_rate=0.0,
            fg_index=50.0,
            strategy=strategy,
            min_score_long=params.get("min_score_long", 5),
            min_score_short=-params.get("min_score_long", 5),
            cached_regime="ranging",
            adx_chop_threshold=params.get("adx_chop_threshold", 18),
        )
        d = result.details or {}
        return {
            "score":          result.score,
            "signal":         result.signal,
            "side":           result.direction,
            "regime":         result.regime,
            "rsi":            d.get("_rsi", 50.0),
            "atr":            getattr(result, "atr", 0.0),
            "veto":           result.veto_reason,
            # Neue Marktstruktur-Features
            "macd_diff":      d.get("_macd_diff",       0.0),
            "macd_signal_val": d.get("_macd_signal",    0.0),
            "ema_ratio_9_21": d.get("_ema_ratio_9_21",  0.0),
            "ema_ratio_21_50": d.get("_ema_ratio_21_50", 0.0),
            "price_vs_ema50": d.get("_price_vs_ema50",  0.0),
            "bb_pct":         d.get("_bb_pct",          0.5),
            "bb_width":       d.get("_bb_width",        0.0),
            "vol_ratio":      d.get("_vol_ratio",       1.0),
            "rsi_slope":      d.get("_rsi_slope",       0.0),
            "ret_1":          d.get("_ret_1",           0.0),
            "ret_4":          d.get("_ret_4",           0.0),
            "ret_8":          d.get("_ret_8",           0.0),
            "ret_16":         d.get("_ret_16",          0.0),
        }
    except Exception:
        return None


def _simulate_outcome(
    entry_price: float,
    side: str,
    future_klines: list,
    atr: float,
    sl_mult: float,
    tp_mult: float,
) -> dict:
    """
    Simuliert SL/TP-Hit in den nächsten Kerzen.
    Gibt exit_price, pnl, exit_reason zurück.
    """
    is_long = side == "BUY"
    sl = entry_price - atr * sl_mult if is_long else entry_price + atr * sl_mult
    tp = entry_price + atr * tp_mult if is_long else entry_price - atr * tp_mult

    for k in future_klines:
        high = float(k[2])
        low  = float(k[3])

        sl_hit = low  <= sl if is_long else high >= sl
        tp_hit = high >= tp if is_long else low  <= tp

        if sl_hit and tp_hit:
            # Beide in gleicher Kerze: SL als konservativer
            exit_price = sl
            exit_reason = "sl"
        elif sl_hit:
            exit_price = sl
            exit_reason = "sl"
        elif tp_hit:
            exit_price = tp
            exit_reason = "tp"
        else:
            continue

        pnl_raw = (exit_price - entry_price) / entry_price
        if not is_long:
            pnl_raw = -pnl_raw
        pnl_net = pnl_raw - 0.0007 * 2  # Taker + Slippage
        return {"exit_price": exit_price, "pnl": pnl_net, "exit_reason": exit_reason}

    # Kein Hit: letzte Kerze als Outcome
    last_close = float(future_klines[-1][4]) if future_klines else entry_price
    pnl_raw = (last_close - entry_price) / entry_price
    if not is_long:
        pnl_raw = -pnl_raw
    return {"exit_price": last_close, "pnl": pnl_raw - 0.0014, "exit_reason": "timeout"}


def _process_symbol_strategy(
    symbol: str,
    strategy: str,
    klines: list,
    params: dict,
    quick: bool,
) -> int:
    """Worker-Funktion: verarbeitet ein Symbol + Strategie."""
    from network_db import log_network_trade

    WINDOW     = 100   # Anzahl Kerzen für Scoring
    FUTURE     = 20    # Kerzen in die Zukunft für Outcome-Simulation
    sl_mult    = params.get("atr_sl_multiplier", 1.5)
    tp_mult    = params.get("atr_tp_multiplier", 3.0)

    written = 0
    step    = 2  # Alle 2 Kerzen einen Signal-Versuch

    for i in range(WINDOW, len(klines) - FUTURE, step):
        window  = klines[i - WINDOW:i]
        future  = klines[i:i + FUTURE]
        score_r = _score_window(window, strategy, params)

        if score_r is None or not score_r["signal"]:
            continue

        entry_price = float(klines[i][1])  # Open der nächsten Kerze
        atr         = score_r["atr"] or (entry_price * 0.005)
        # FIX: score_candles liefert "long"/"short" – _simulate_outcome + DB erwarten "BUY"/"SELL"
        # (vorher: side blieb "long" → is_long==False → ALLE Synthetik-Outcomes als Short simuliert)
        side        = "BUY" if score_r["side"] == "long" else "SELL"

        outcome = _simulate_outcome(entry_price, side, future, atr, sl_mult, tp_mult)

        try:
            log_network_trade(
                bot_id=0,
                symbol=symbol,
                side=side,
                entry=entry_price,
                exit_price=outcome["exit_price"],
                pnl=outcome["pnl"],
                exit_reason=outcome["exit_reason"],
                score=score_r["score"],
                regime=score_r["regime"],
                rsi=score_r["rsi"],
                atr=atr,
                strategy=strategy,
                is_synthetic=True,
                config_snapshot=params,
                macd_diff=score_r["macd_diff"],
                macd_signal_val=score_r["macd_signal_val"],
                ema_ratio_9_21=score_r["ema_ratio_9_21"],
                ema_ratio_21_50=score_r["ema_ratio_21_50"],
                price_vs_ema50=score_r["price_vs_ema50"],
                bb_pct=score_r["bb_pct"],
                bb_width=score_r["bb_width"],
                vol_ratio=score_r["vol_ratio"],
                rsi_slope=score_r["rsi_slope"],
                ret_1=score_r["ret_1"],
                ret_4=score_r["ret_4"],
                ret_8=score_r["ret_8"],
                ret_16=score_r["ret_16"],
            )
            written += 1
        except Exception:
            pass

    return written


def run_factory(quick: bool = False):
    """Hauptprozess der Learning Factory."""
    t0 = time.time()
    total_written = 0

    logger.info(f"Learning Factory startet (quick={quick}, workers={MAX_WORKERS})")

    tasks = []
    for symbol in SYMBOLS:
        # Kerzen laden
        csv_path = HISTORY_DIR / f"{symbol}_15m.csv"
        if not csv_path.exists():
            logger.warning(f"Keine CSV für {symbol}: {csv_path}")
            continue

        limit  = 3000 if quick else 0  # quick: ~31 Tage bei 15m
        klines = load_csv_klines(csv_path, limit=limit)
        if len(klines) < 200:
            logger.warning(f"Zu wenig Kerzen für {symbol}: {len(klines)}")
            continue

        # Alle Kombinationen aus Strategien × SL/TP-Params; Schwelle je Strategie.
        keys   = list(PARAM_GRID.keys())
        values = list(PARAM_GRID.values())
        for strategy in STRATEGY_VARIANTS:
            for combo in product(*values):
                params = dict(zip(keys, combo))
                params["min_score_long"] = STRATEGY_THRESHOLD[strategy]  # = Live-Schwelle
                tasks.append((symbol, strategy, klines, params, quick))

    logger.info(f"{len(tasks)} Worker-Tasks erstellt | Schwellen je Strategie: {STRATEGY_THRESHOLD}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_process_symbol_strategy, *args): args[:2]
            for args in tasks
        }
        done  = 0
        total = len(tasks)
        for future in as_completed(futures):
            sym, strat = futures[future]
            try:
                n = future.result()
                total_written += n
            except Exception as e:
                logger.warning(f"Task {sym}/{strat} Fehler: {e}")
            done += 1
            # Live-Fortschritt alle 5 Tasks: Prozent, Outcomes, verstrichen, ETA
            if done % 5 == 0 or done == total:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (total - done) / rate if rate > 0 else 0
                logger.info(
                    f"Fortschritt: {done}/{total} ({done/total*100:.0f}%) | "
                    f"{total_written} Outcomes | "
                    f"verstrichen {elapsed/60:.1f}min | ETA ~{eta/60:.1f}min"
                )

    dt = time.time() - t0
    logger.info(
        f"Learning Factory abgeschlossen: {total_written} synthetische Outcomes "
        f"in {dt:.0f}s ({dt/60:.1f} min)"
    )
    return total_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Nur letzte 3 Monate (schneller)")
    args = parser.parse_args()
    run_factory(quick=args.quick)
