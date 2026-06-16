"""
walkforward.py – KORREKTER Out-of-Sample-Test des ML-Vetos (real, kein Orakel).

Frühere Version war kaputt: Basislinie nutzte strict labels (per Definition nur
Gewinner) → Orakel, das ML konnte es nie schlagen. Diese Version macht es richtig:

  1. Zeit-Split pro Symbol: erste TRAIN_FRAC = Training, Rest = ungesehener Test.
  2. Training NUR auf Trainingsperiode:
       - Modell A (Candle, 3-Klassen) aus strict labels + Features
       - Modell B (Win, P(win)) aus ECHTEN Signal-Outcomes der Trainingsperiode
  3. Test auf ungesehener Periode:
       - echte Signale via score_candles (alle 5 Strategien, Live-Schwellen)
       - ECHTES Outcome via SL/TP-Simulation (Gewinner UND Verlierer)
       - Basislinie = PnL ALLER Signale
       - Mit Veto  = PnL der Signale, die A (Konfidenz/Richtung) UND B (P(win)>=0.42)
         durchlassen — exakt wie live (layers/layer3_scoring._apply_ml_veto)
  4. Vergleich Basislinie vs. Veto. KEIN Look-Ahead (Modelle kennen den Test nicht).

Aufruf:
  python walkforward.py                 # alle Symbole (parallel, dauert ein paar Min)
  python walkforward.py --quick         # nur BTC, kleinerer Ausschnitt (Schnelltest)
  python walkforward.py --symbol PF_XBTUSD
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("walkforward")

HISTORY_DIR = Path("data/history")
OUTPUT_FILE = Path("data/walkforward.txt")

SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]
STRATEGIES = ["momentum", "mean_reversion", "breakout", "contrarian", "scalper"]
STRATEGY_THRESHOLD = {
    "momentum": 3, "contrarian": 3, "scalper": 3, "mean_reversion": 5, "breakout": 6,
}

TRAIN_FRAC = 0.65        # erste 65 % Training, letzte 35 % Out-of-Sample-Test
WINDOW   = 100           # Kerzen-Historie fürs Scoring
FUTURE   = 20            # Kerzen für SL/TP-Outcome
STEP     = 2             # jede 2. Kerze ein Signal-Versuch
SL_MULT  = 1.5
TP_MULT  = 3.0
FEE      = 0.0007 * 2
CONF_THRESHOLD = 0.55    # Modell A
VETO_THRESHOLD = 0.42    # Modell B


def _simulate(entry, direction, future, atr):
    """Echtes Outcome: SL oder TP zuerst getroffen? Gibt PnL-Bruchteil zurück."""
    is_long = direction == "long"
    sl = entry - atr * SL_MULT if is_long else entry + atr * SL_MULT
    tp = entry + atr * TP_MULT if is_long else entry - atr * TP_MULT
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


def _build_4h_regimes(klines):
    """4h-Regime je 15m-Index (nur vergangene 4h-Kerzen → kein Look-Ahead)."""
    from layers.layer2_regime import _klines_to_dataframe, _calculate_adx
    CP4H = 16
    h4, end_idx = [], []
    for s in range(0, len(klines) - CP4H + 1, CP4H):
        c = klines[s:s + CP4H]
        h4.append([c[0][0], c[0][1], max(x[2] for x in c), min(x[3] for x in c),
                   c[-1][4], sum(x[5] for x in c), 0, 0, 0, 0, 0, 0])
        end_idx.append(s + CP4H - 1)
    if len(h4) < 20:
        return ["ranging"] * len(klines)
    df = _klines_to_dataframe(h4)
    adx, pdi, mdi = _calculate_adx(df, 14)
    reg4 = []
    for j in range(len(h4)):
        a = float(adx.iloc[j]) if j < len(adx) and not np.isnan(adx.iloc[j]) else 0.0
        p = float(pdi.iloc[j]) if j < len(pdi) and not np.isnan(pdi.iloc[j]) else 0.0
        m = float(mdi.iloc[j]) if j < len(mdi) and not np.isnan(mdi.iloc[j]) else 0.0
        reg4.append(("trending_up" if p > m else "trending_down") if a > 25 else "ranging")
    out = ["ranging"] * len(klines)
    cur, j = "ranging", 0
    for i in range(len(klines)):
        while j < len(end_idx) and end_idx[j] <= i:
            cur = reg4[j]; j += 1
        out[i] = cur
    return out


def _gen_signals(symbol, klines, regimes, lo, hi):
    """Echte Signale + echtes Outcome im Bereich [lo, hi). Liste (result, regime, strategy, pnl)."""
    from scoring_core import score_candles
    out = []
    start = max(lo, WINDOW)
    end = min(hi, len(klines) - FUTURE)
    for strat in STRATEGIES:
        thr = STRATEGY_THRESHOLD[strat]
        for i in range(start, end, STEP):
            r = score_candles(
                symbol=symbol, klines=klines[i - WINDOW:i],
                funding_rate=0.0, fg_index=50.0, strategy=strat,
                min_score_long=thr, min_score_short=-thr,
                cached_regime=regimes[i], adx_chop_threshold=18.0,
            )
            if not r.signal:
                continue
            entry = float(klines[i][1])
            atr = r.atr or (entry * 0.005)
            pnl = _simulate(entry, r.direction, klines[i:i + FUTURE], atr)
            out.append((r, regimes[i], strat, pnl))
    return out


def _is_vetoed(r, regime, strategy, model_a, model_b):
    """Repliziert das Live-Veto (layer3): Modell A (Richtung/Konfidenz) → Modell B (P(win))."""
    from ml_network import _candle_features_from_result, REGIME_MAP, STRATEGY_MAP, CLASS_NAMES
    # Stufe A
    if model_a is not None:
        feat = _candle_features_from_result(r)
        if feat is not None:
            try:
                proba = model_a.predict_proba(feat)[0]
                cls = int(np.argmax(proba))
                conf = float(proba[cls])
                if conf < CONF_THRESHOLD:
                    return True
                if CLASS_NAMES.get(cls, "neutral") != r.direction:
                    return True
            except Exception:
                pass
    # Stufe B – S5-9-Fix: 21-Feature-Vektor identisch zu ml_network.predict_win_prob.
    # Vorher nur 9 Features → Shape-Mismatch gegen das 21-dim Modell → vom except
    # verschluckt → Modell B vetoed im Test NIE → Walk-Forward zeigte nur Modell A.
    if model_b is not None:
        d = r.details or {}
        try:
            feat_b = np.array([[
                float(r.score),
                float(getattr(r, "funding_rate", 0.0)),
                float(d.get("_rsi_14", d.get("_rsi", 50))),
                float(r.atr),
                0.0, 0.0,  # is_shadow, is_synthetic
                float(REGIME_MAP.get(regime, 2)),
                float(STRATEGY_MAP.get(strategy, 0)),
                float(d.get("_macd_diff",       0)),
                float(d.get("_macd_signal",     0)),
                float(d.get("_ema_ratio_9_21",  0)),
                float(d.get("_ema_ratio_21_50", 0)),
                float(d.get("_price_vs_ema50",  0)),
                float(d.get("_bb_pct",          0.5)),
                float(d.get("_bb_width",        0)),
                float(d.get("_vol_ratio",       1.0)),
                float(d.get("_rsi_slope",       0)),
                float(d.get("_ret_1",           0)),
                float(d.get("_ret_4",           0)),
                float(d.get("_ret_8",           0)),
                float(d.get("_ret_16",          0)),
            ]], dtype=np.float32)
            p = float(model_b.predict_proba(feat_b)[0][1])
            if p < VETO_THRESHOLD:
                return True
        except Exception:
            pass
    return False


def process_symbol(symbol, klines):
    """Worker: trainiert A+B auf Trainingsperiode, testet Veto out-of-sample."""
    from ml_network import (ml_network, _compute_candle_features_batch,
                            generate_strict_labels)

    regimes = _build_4h_regimes(klines)
    split = int(len(klines) * TRAIN_FRAC)

    # --- Modell A auf Trainingsperiode ---
    train_kl = klines[:split]
    model_a = None
    try:
        feats = _compute_candle_features_batch(train_kl)
        labels = generate_strict_labels(train_kl, TP_MULT, SL_MULT)
        valid = ~np.isnan(feats).any(axis=1)
        X, y = feats[valid], labels[valid]
        if len(X) >= 200:
            model_a = ml_network._train_candle_model(X, y)
    except Exception as e:
        logger.warning(f"{symbol}: Modell A Training fehlgeschlagen: {e}")

    # --- Modell B aus ECHTEN Trainings-Signal-Outcomes ---
    train_sig = _gen_signals(symbol, klines, regimes, WINDOW, split)
    rows = [{
        "score": r.score, "funding_rate": float(getattr(r, "funding_rate", 0.0)),
        "rsi": r.details.get("_rsi", 50.0), "atr": r.atr, "fg_index": 50.0,
        "is_shadow": 0, "is_synthetic": 0, "regime": reg, "strategy": strat,
        "pnl": pnl, "weight": 1.0,
        # S5-9-Fix: 13 Marktstruktur-Features mittrainieren, damit Modell B hier
        # exakt dieselben 21 Features lernt wie live (sonst inerte Default-Features).
        "macd_diff":       r.details.get("_macd_diff",       0.0),
        "macd_signal_val": r.details.get("_macd_signal",     0.0),
        "ema_ratio_9_21":  r.details.get("_ema_ratio_9_21",  0.0),
        "ema_ratio_21_50": r.details.get("_ema_ratio_21_50", 0.0),
        "price_vs_ema50":  r.details.get("_price_vs_ema50",  0.0),
        "bb_pct":          r.details.get("_bb_pct",          0.5),
        "bb_width":        r.details.get("_bb_width",        0.0),
        "vol_ratio":       r.details.get("_vol_ratio",       1.0),
        "rsi_slope":       r.details.get("_rsi_slope",       0.0),
        "ret_1":           r.details.get("_ret_1",           0.0),
        "ret_4":           r.details.get("_ret_4",           0.0),
        "ret_8":           r.details.get("_ret_8",           0.0),
        "ret_16":          r.details.get("_ret_16",          0.0),
    } for (r, reg, strat, pnl) in train_sig]
    model_b = None
    try:
        if len(rows) >= 50:
            model_b = ml_network._train_win_model(rows)
    except Exception as e:
        logger.warning(f"{symbol}: Modell B Training fehlgeschlagen: {e}")

    # --- Test out-of-sample ---
    test_sig = _gen_signals(symbol, klines, regimes, split, len(klines))
    base_pnl = base_n = base_w = 0.0
    veto_pnl = veto_n = veto_w = 0.0
    for (r, reg, strat, pnl) in test_sig:
        base_pnl += pnl; base_n += 1; base_w += (1 if pnl > 0 else 0)
        if not _is_vetoed(r, reg, strat, model_a, model_b):
            veto_pnl += pnl; veto_n += 1; veto_w += (1 if pnl > 0 else 0)

    return {
        "symbol": symbol,
        "train_signals": len(train_sig),
        "base_n": int(base_n), "base_pnl": base_pnl, "base_w": int(base_w),
        "veto_n": int(veto_n), "veto_pnl": veto_pnl, "veto_w": int(veto_w),
        "model_a": model_a is not None, "model_b": model_b is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Nur BTC, kleinerer Ausschnitt")
    parser.add_argument("--symbol", default=None)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else (["PF_XBTUSD"] if args.quick else SYMBOLS)
    limit = 20000 if args.quick else 0

    from learning_factory import load_csv_klines
    data = {}
    for s in symbols:
        p = HISTORY_DIR / f"{s}_15m.csv"
        if not p.exists():
            logger.warning(f"Keine CSV: {p}"); continue
        kl = load_csv_klines(p, limit=limit)
        if len(kl) >= 5000:
            data[s] = kl
        else:
            logger.warning(f"{s}: zu wenig Kerzen ({len(kl)})")

    logger.info(f"Walk-Forward (korrekt, real) für {list(data.keys())} – trainiere + teste...")

    results = []
    if len(data) <= 1:
        for s, kl in data.items():
            results.append(process_symbol(s, kl))
    else:
        with ProcessPoolExecutor(max_workers=min(5, len(data))) as ex:
            futs = {ex.submit(process_symbol, s, kl): s for s, kl in data.items()}
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                    logger.info(f"  {futs[f]} fertig")
                except Exception as e:
                    logger.warning(f"  {futs[f]} Fehler: {e}")

    results.sort(key=lambda r: SYMBOLS.index(r["symbol"]) if r["symbol"] in SYMBOLS else 99)

    lines = []
    lines.append("=" * 86)
    lines.append("WALK-FORWARD (korrekt) – ML-Veto out-of-sample, ECHTE Outcomes")
    lines.append(f"Zeit-Split {int(TRAIN_FRAC*100)}% Train / {100-int(TRAIN_FRAC*100)}% Test | "
                 f"SL/TP {SL_MULT}/{TP_MULT}xATR | Veto: A(conf<{CONF_THRESHOLD}) + B(P(win)<{VETO_THRESHOLD})")
    lines.append("-" * 86)
    lines.append(f"{'Symbol':<11} {'OHNE Veto (alle Signale)':>30} | {'MIT Veto':>26} | {'Δ PnL%':>8}")
    lines.append(f"{'':<11} {'Trades':>8} {'WR':>6} {'PnL%':>12} | {'Trades':>7} {'WR':>6} {'PnL%':>10} | {'':>8}")
    lines.append("-" * 86)

    tot_base = tot_veto = 0.0
    helped = 0
    for r in results:
        bwr = (r["base_w"] / r["base_n"] * 100) if r["base_n"] else 0
        vwr = (r["veto_w"] / r["veto_n"] * 100) if r["veto_n"] else 0
        delta = (r["veto_pnl"] - r["base_pnl"]) * 100
        tot_base += r["base_pnl"]; tot_veto += r["veto_pnl"]
        if r["veto_pnl"] > r["base_pnl"]:
            helped += 1
        lines.append(f"{r['symbol']:<11} {r['base_n']:>8} {bwr:>5.0f}% {r['base_pnl']*100:>+12.2f} | "
                     f"{r['veto_n']:>7} {vwr:>5.0f}% {r['veto_pnl']*100:>+10.2f} | {delta:>+8.2f}")

    lines.append("-" * 86)
    lines.append(f"GESAMT PnL%:  OHNE {tot_base*100:+.2f}   MIT {tot_veto*100:+.2f}   "
                 f"Δ {(tot_veto-tot_base)*100:+.2f}")
    lines.append(f"ML-Veto hilft bei {helped}/{len(results)} Symbolen.")
    lines.append("")
    lines.append("Lesart: ECHTE Outcomes (Gewinner UND Verlierer). Basislinie = jedes Signal handeln.")
    lines.append("  Δ positiv = Veto filtert Verlierer raus = ML lernt echte Muster (kein Overfitting).")
    lines.append("  Δ negativ = Veto kostet netto = ML bringt out-of-sample (noch) keinen Mehrwert.")
    lines.append("=" * 86)

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
