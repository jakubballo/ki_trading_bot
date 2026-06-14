"""
test_regime_gate.py – Vergleichstest für Fix 4 (4h-Regime-Gate).

Läuft über die historischen 15m-CSV-Daten (data/history/*.csv) und simuliert
für jede Strategie zwei Varianten: OHNE und MIT 4h-Regime-Gate.
Vergleicht Anzahl Trades, Win-Rate und PnL.

WICHTIG:
  - Ändert NICHTS am Live-Netzwerk. Liest nur CSVs.
  - Kein Look-Ahead: das 4h-Regime nutzt nur VERGANGENE 4h-Kerzen,
    das Outcome nur ZUKÜNFTIGE 15m-Kerzen.
  - Identische Trades in beiden Varianten – der EINZIGE Unterschied ist,
    ob das Gate ein Signal herausfiltert. So ist der Gate-Effekt sauber isoliert.

Aufruf:
  python test_regime_gate.py                 # alle Symbole + Strategien (dauert)
  python test_regime_gate.py --quick         # nur letzte ~3000 Kerzen (schnell)
  python test_regime_gate.py --symbol PF_XBTUSD
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows-Konsole: Δ etc. darstellbar
except Exception:
    pass

from scoring_core import score_candles, passes_regime_gate
from learning_factory import load_csv_klines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("regime_gate_test")

HISTORY_DIR = Path("data/history")
OUTPUT_FILE = Path("data/regime_gate_test.txt")

SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]
STRATEGIES = ["momentum", "mean_reversion", "breakout", "contrarian", "scalper"]

WINDOW = 100          # Kerzen für das Scoring
FUTURE = 20           # Kerzen in die Zukunft für die Outcome-Simulation
STEP = 2              # alle 2 Kerzen ein Signal-Versuch
ADX_TREND_THRESHOLD = 25
CANDLES_PER_4H = 16   # 16 × 15m = 4h
FEE = 0.0007 * 2      # Taker + Slippage hin + zurück

# Repräsentative Standard-Parameter (wie ein Standard-Bot)
MIN_SCORE_LONG = 5
SL_MULT = 1.5
TP_MULT = 3.0


def build_4h_regimes(klines_15m: list) -> List[str]:
    """
    Baut nicht-überlappende 4h-Kerzen aus 15m, berechnet je 4h-Kerze das
    ADX-Regime und mappt es auf jeden 15m-Index (nur zuletzt ABGESCHLOSSENE
    4h-Kerze → kein Look-Ahead).
    """
    from layers.layer2_regime import _klines_to_dataframe, _calculate_adx

    h4 = []
    h4_end_idx = []   # 15m-Index, bei dem die jeweilige 4h-Kerze abgeschlossen ist
    for start in range(0, len(klines_15m) - CANDLES_PER_4H + 1, CANDLES_PER_4H):
        chunk = klines_15m[start:start + CANDLES_PER_4H]
        ts  = chunk[0][0]
        o   = chunk[0][1]
        hi  = max(c[2] for c in chunk)
        lo  = min(c[3] for c in chunk)
        cl  = chunk[-1][4]
        vol = sum(c[5] for c in chunk)
        h4.append([ts, o, hi, lo, cl, vol, ts, 0, 0, 0, 0, 0])
        h4_end_idx.append(start + CANDLES_PER_4H - 1)

    if len(h4) < 20:
        return ["ranging"] * len(klines_15m)

    df = _klines_to_dataframe(h4)
    adx, plus_di, minus_di = _calculate_adx(df, 14)

    regime_4h = []
    for j in range(len(h4)):
        a = float(adx.iloc[j])      if j < len(adx)      and not np.isnan(adx.iloc[j])      else 0.0
        p = float(plus_di.iloc[j])  if j < len(plus_di)  and not np.isnan(plus_di.iloc[j])  else 0.0
        m = float(minus_di.iloc[j]) if j < len(minus_di) and not np.isnan(minus_di.iloc[j]) else 0.0
        if a > ADX_TREND_THRESHOLD:
            regime_4h.append("trending_up" if p > m else "trending_down")
        else:
            regime_4h.append("ranging")

    regime_for_i = ["ranging"] * len(klines_15m)
    cur = "ranging"
    j = 0
    for i in range(len(klines_15m)):
        while j < len(h4_end_idx) and h4_end_idx[j] <= i:
            cur = regime_4h[j]
            j += 1
        regime_for_i[i] = cur
    return regime_for_i


def _simulate(entry: float, direction: str, future: list,
              atr: float, sl_mult: float, tp_mult: float):
    """Simuliert SL/TP-Hit. Gibt (pnl_fraction, reason) zurück. Korrektes long/short."""
    is_long = direction == "long"
    sl = entry - atr * sl_mult if is_long else entry + atr * sl_mult
    tp = entry + atr * tp_mult if is_long else entry - atr * tp_mult

    for k in future:
        hi = float(k[2]); lo = float(k[3])
        sl_hit = lo <= sl if is_long else hi >= sl
        tp_hit = hi >= tp if is_long else lo <= tp
        if sl_hit:                      # konservativ: SL zuerst, wenn beide in einer Kerze
            ex, reason = sl, "sl"
        elif tp_hit:
            ex, reason = tp, "tp"
        else:
            continue
        pnl = (ex - entry) / entry
        if not is_long:
            pnl = -pnl
        return pnl - FEE, reason

    last = float(future[-1][4]) if future else entry
    pnl = (last - entry) / entry
    if not is_long:
        pnl = -pnl
    return pnl - FEE, "timeout"


def _tally(acc: dict, pnl: float):
    acc["n"] += 1
    acc["pnl"] += pnl
    if pnl > 0:
        acc["wins"] += 1


def run_symbol(symbol: str, klines: list) -> dict:
    """Backtestet alle Strategien für ein Symbol, OHNE vs MIT Gate."""
    regimes = build_4h_regimes(klines)
    off = {"n": 0, "wins": 0, "pnl": 0.0}
    on  = {"n": 0, "wins": 0, "pnl": 0.0}

    for strategy in STRATEGIES:
        for i in range(WINDOW, len(klines) - FUTURE, STEP):
            window = klines[i - WINDOW:i]
            regime = regimes[i]
            r = score_candles(
                symbol=symbol, klines=window,
                funding_rate=0.0, fg_index=50.0,
                strategy=strategy,
                min_score_long=MIN_SCORE_LONG, min_score_short=-MIN_SCORE_LONG,
                cached_regime=regime, adx_chop_threshold=18.0,
            )
            if not r.signal:
                continue
            entry = float(klines[i][1])
            atr = r.atr or (entry * 0.005)
            pnl, _ = _simulate(entry, r.direction, klines[i:i + FUTURE], atr, SL_MULT, TP_MULT)

            _tally(off, pnl)                                           # OHNE Gate: alle Signale
            if passes_regime_gate(r.direction, regime, strategy):     # MIT Gate: gefiltert
                _tally(on, pnl)

    return {"symbol": symbol, "off": off, "on": on}


def _fmt_row(res: dict) -> str:
    off, on = res["off"], res["on"]
    wr_off = (off["wins"] / off["n"] * 100) if off["n"] else 0.0
    wr_on  = (on["wins"]  / on["n"]  * 100) if on["n"]  else 0.0
    delta = on["pnl"] - off["pnl"]
    return (f"{res['symbol']:<12} "
            f"{off['n']:>6} {wr_off:>6.0f}% {off['pnl']*100:>+9.2f} | "
            f"{on['n']:>6} {wr_on:>6.0f}% {on['pnl']*100:>+9.2f} | "
            f"{delta*100:>+9.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Nur letzte ~3000 Kerzen")
    parser.add_argument("--symbol", default=None, help="Nur dieses Symbol")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS
    limit = 3000 if args.quick else 0

    results = []
    for sym in symbols:
        csv_path = HISTORY_DIR / f"{sym}_15m.csv"
        if not csv_path.exists():
            logger.warning(f"Keine CSV: {csv_path} – übersprungen")
            continue
        klines = load_csv_klines(csv_path, limit=limit)
        if len(klines) < WINDOW + FUTURE + CANDLES_PER_4H * 20:
            logger.warning(f"{sym}: zu wenig Kerzen ({len(klines)}) – übersprungen")
            continue
        logger.info(f"Teste {sym} ({len(klines)} Kerzen)...")
        results.append(run_symbol(sym, klines))

    # Ausgabe-Tabelle bauen
    lines = []
    lines.append("=" * 78)
    lines.append("FIX 4 – 4h-REGIME-GATE: Vergleichstest (historisch, out-of-sample)")
    lines.append(f"Parameter: min_score_long={MIN_SCORE_LONG}, SL={SL_MULT}xATR, TP={TP_MULT}xATR")
    lines.append("PnL in % (Summe über alle Strategien; mean_reversion ist Gate-ausgenommen)")
    lines.append("-" * 78)
    lines.append(f"{'Symbol':<12} {'OHNE Gate':>24} | {'MIT Gate':>24} | {'Δ PnL':>9}")
    lines.append(f"{'':<12} {'Trades':>6} {'WR':>7} {'PnL%':>9} | "
                 f"{'Trades':>6} {'WR':>7} {'PnL%':>9} | {'%':>9}")
    lines.append("-" * 78)

    total_off = total_on = 0.0
    helped = 0
    for res in results:
        lines.append(_fmt_row(res))
        total_off += res["off"]["pnl"]
        total_on  += res["on"]["pnl"]
        if res["on"]["pnl"] > res["off"]["pnl"]:
            helped += 1

    lines.append("-" * 78)
    lines.append(f"GESAMT PnL%:  OHNE {total_off*100:+.2f}   MIT {total_on*100:+.2f}   "
                 f"Δ {(total_on-total_off)*100:+.2f}")
    lines.append(f"Gate hilft bei {helped}/{len(results)} Symbolen.")
    lines.append("")
    lines.append("ENTSCHEIDUNG:")
    lines.append("  Δ PnL deutlich positiv bei den meisten Symbolen → Gate AKTIVIEREN")
    lines.append("    (require_4h_regime_confirmation: true in den Bot-Configs / config.py)")
    lines.append("  Δ PnL negativ / kaum Änderung → Gate AUS lassen (Standard)")
    lines.append("=" * 78)

    report = "\n".join(lines)
    print("\n" + report)
    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
        logger.info(f"Bericht gespeichert: {OUTPUT_FILE}")
    except Exception as e:
        logger.warning(f"Konnte Bericht nicht speichern: {e}")


if __name__ == "__main__":
    main()
