"""
threshold_sweep.py – Diagnose: optimale Score-Schwelle je Strategie.

Für jede Strategie wird über die historischen 15m-CSVs für jede Schwelle 3..8
gezählt: Anzahl Signale, Win-Rate, PnL%. Zeigt, wo eine Strategie "tot" ist
(0 Signale) und wo ihr PnL-Optimum liegt.

WICHTIG:
  - Ändert NICHTS am Live-Netzwerk / keine DB-Schreibzugriffe.
  - Kein Look-Ahead (Outcome nur aus zukünftigen Kerzen).
  - 4h-Regime-Gate ist hier AUS → reine Schwellen-Kalibrierung (der Score
    selbst hängt nicht von der Schwelle ab; pro Fenster wird einmal gescort
    und gegen alle Schwellen verglichen).

Aufruf:
  python threshold_sweep.py            # alle Symbole + Strategien
  python threshold_sweep.py --quick    # nur letzte ~3000 Kerzen
"""

import argparse
import logging
import sys
from pathlib import Path

from scoring_core import score_candles
from learning_factory import load_csv_klines

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("threshold_sweep")

HISTORY_DIR = Path("data/history")
OUTPUT_FILE = Path("data/threshold_sweep.txt")

SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]
STRATEGIES = ["momentum", "mean_reversion", "breakout", "contrarian", "scalper"]
THRESHOLDS = [3, 4, 5, 6, 7, 8]

WINDOW = 100
FUTURE = 20
STEP = 2
FEE = 0.0007 * 2
SL_MULT = 1.5
TP_MULT = 3.0


def _simulate(entry, direction, future, atr, sl_mult, tp_mult):
    is_long = direction == "long"
    sl = entry - atr * sl_mult if is_long else entry + atr * sl_mult
    tp = entry + atr * tp_mult if is_long else entry - atr * tp_mult
    for k in future:
        hi = float(k[2]); lo = float(k[3])
        sl_hit = lo <= sl if is_long else hi >= sl
        tp_hit = hi >= tp if is_long else lo <= tp
        if sl_hit:
            ex = sl
        elif tp_hit:
            ex = tp
        else:
            continue
        pnl = (ex - entry) / entry
        return (pnl if is_long else -pnl) - FEE
    last = float(future[-1][4]) if future else entry
    pnl = (last - entry) / entry
    return (pnl if is_long else -pnl) - FEE


def sweep_strategy(strategy: str, all_klines: dict) -> dict:
    """
    Liefert pro Schwelle: {T: {"n":, "wins":, "pnl":}}.
    Pro Fenster wird EINMAL gescort; das Ergebnis zählt für jede Schwelle,
    die der |Score| erreicht.
    """
    acc = {t: {"n": 0, "wins": 0, "pnl": 0.0} for t in THRESHOLDS}

    for symbol, klines in all_klines.items():
        for i in range(WINDOW, len(klines) - FUTURE, STEP):
            window = klines[i - WINDOW:i]
            r = score_candles(
                symbol=symbol, klines=window,
                funding_rate=0.0, fg_index=50.0,
                strategy=strategy,
                min_score_long=1, min_score_short=-1,  # permissiv – wir lesen nur r.score
                cached_regime="ranging", adx_chop_threshold=18.0,
            )
            score = r.score
            if score == 0:
                continue
            direction = "long" if score > 0 else "short"
            entry = float(klines[i][1])
            atr = r.atr or (entry * 0.005)
            mag = abs(score)

            # Outcome nur einmal simulieren
            pnl = None
            for t in THRESHOLDS:
                if mag >= t:
                    if pnl is None:
                        pnl = _simulate(entry, direction, klines[i:i + FUTURE], atr, SL_MULT, TP_MULT)
                    a = acc[t]
                    a["n"] += 1
                    a["pnl"] += pnl
                    if pnl > 0:
                        a["wins"] += 1
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    limit = 3000 if args.quick else 0

    all_klines = {}
    for sym in SYMBOLS:
        p = HISTORY_DIR / f"{sym}_15m.csv"
        if not p.exists():
            logger.warning(f"Keine CSV: {p}")
            continue
        kl = load_csv_klines(p, limit=limit)
        if len(kl) >= WINDOW + FUTURE:
            all_klines[sym] = kl

    lines = []
    lines.append("=" * 70)
    lines.append("THRESHOLD-SWEEP – optimale Score-Schwelle je Strategie")
    lines.append(f"Symbole: {list(all_klines.keys())} | SL/TP={SL_MULT}/{TP_MULT}xATR | Gate AUS")
    lines.append("=" * 70)

    summary = []
    for strategy in STRATEGIES:
        logger.info(f"Sweep {strategy}...")
        acc = sweep_strategy(strategy, all_klines)
        lines.append(f"\nStrategie: {strategy}")
        lines.append(f"  {'Schwelle':>8} {'Signale':>8} {'WinRate':>8} {'PnL%':>9}")
        lines.append("  " + "-" * 36)
        best_t, best_pnl = None, None
        for t in THRESHOLDS:
            a = acc[t]
            wr = (a["wins"] / a["n"] * 100) if a["n"] else 0.0
            pnl = a["pnl"] * 100
            mark = ""
            if a["n"] >= 20 and (best_pnl is None or pnl > best_pnl):
                best_pnl, best_t = pnl, t
            lines.append(f"  {t:>8} {a['n']:>8} {wr:>7.0f}% {pnl:>+9.2f}")
        if best_t is not None:
            lines.append(f"  → bestes PnL bei Schwelle {best_t} ({best_pnl:+.2f}%, >=20 Signale)")
            summary.append((strategy, best_t, best_pnl))
        else:
            lines.append(f"  → zu wenige Signale für eine Empfehlung")
            summary.append((strategy, None, None))

    lines.append("\n" + "=" * 70)
    lines.append("EMPFEHLUNG je Strategie (bestes PnL mit >=20 Signalen):")
    for strat, t, pnl in summary:
        if t is not None:
            lines.append(f"  {strat:<16} Schwelle {t}  (PnL {pnl:+.2f}%)")
        else:
            lines.append(f"  {strat:<16} – zu dünn, evtl. Schwelle senken")
    lines.append("Aktuell: Standard-Bots=6, Aggressiv-Bots=5 (einheitlich über alle Strategien).")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print("\n" + report)
    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
        logger.info(f"Bericht gespeichert: {OUTPUT_FILE}")
    except Exception as e:
        logger.warning(f"Speichern fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()
