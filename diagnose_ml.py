"""
diagnose_ml.py – Analysiert Confidence-Verteilung der trainierten ML-Modelle.

Beantwortet: Warum liefert Modell A fast nur neutrale Vorhersagen?
  - Zeigt Confidence-Histogramm (textbasiert)
  - Zeigt Klassenverteilung der Vorhersagen
  - Zeigt Wahrscheinlichkeitsverteilung über alle Klassen
  - Gibt Empfehlung ob Schwelle 0.55 sinnvoll ist

Aufruf:
  python diagnose_ml.py                        # alle Symbole, Candle-Modell
  python diagnose_ml.py --symbol PF_XBTUSD     # nur BTC
  python diagnose_ml.py --win                  # Win-Modell B statt Candle-Modell A
  python diagnose_ml.py --samples 5000         # Anzahl Kerzen für Analyse
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("diagnose_ml")

SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]
HISTORY_DIR = Path("data/history")


def _ascii_hist(values, bins=10, width=40, label="") -> str:
    """Gibt ein einfaches ASCII-Histogramm zurück."""
    if not values:
        return "(keine Daten)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"(alle Werte = {lo:.3f})"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar_lo = lo + i * step
        bar_hi = bar_lo + step
        bar = "█" * int(c / max_count * width)
        lines.append(f"  {bar_lo:6.3f}–{bar_hi:6.3f} | {bar:<{width}} {c}")
    return "\n".join(lines)


def diagnose_candle(symbol: str, n_samples: int):
    """Analysiert Modell A für ein Symbol."""
    from ml_network import ml_network, _load_csv_klines, _compute_candle_features_batch, generate_strict_labels

    model = ml_network._candle_models.get(symbol) or ml_network._candle_base
    if model is None:
        print(f"[{symbol}] Kein Candle-Modell geladen. Zuerst train_from_csv() ausführen.")
        return

    csv_path = HISTORY_DIR / f"{symbol}_15m.csv"
    if not csv_path.exists():
        print(f"[{symbol}] CSV nicht gefunden: {csv_path}")
        return

    klines = _load_csv_klines(str(csv_path))
    if n_samples and len(klines) > n_samples:
        klines = klines[-n_samples:]

    feats  = _compute_candle_features_batch(klines)
    labels = generate_strict_labels(klines)
    valid  = ~np.isnan(feats).any(axis=1)
    X      = feats[valid]
    y_true = labels[valid]

    if len(X) == 0:
        print(f"[{symbol}] Keine gültigen Feature-Vektoren.")
        return

    probas     = model.predict_proba(X)          # shape: (n, 3)
    max_probas = probas.max(axis=1)              # Konfidenz je Sample
    preds      = probas.argmax(axis=1)           # Klasse je Sample

    n_neutral = (preds == 0).sum()
    n_long    = (preds == 1).sum()
    n_short   = (preds == 2).sum()
    n_total   = len(preds)

    above_55  = (max_probas >= 0.55).sum()
    above_50  = (max_probas >= 0.50).sum()
    above_45  = (max_probas >= 0.45).sum()

    print(f"\n{'='*60}")
    print(f"Candle-Modell A — {symbol}  ({n_total} Samples)")
    print(f"{'='*60}")
    print(f"Trainings-Labels: Long={( y_true==1).sum()}  Short={(y_true==2).sum()}  Neutral={(y_true==0).sum()}")
    print(f"Vorhersagen:      Long={n_long}  Short={n_short}  Neutral={n_neutral}")
    print()
    print(f"Confidence-Schwellen (aktuell: 0.55):")
    print(f"  ≥ 0.55  →  {above_55:>6} Trades würden passieren  ({above_55/n_total*100:.1f}%)")
    print(f"  ≥ 0.50  →  {above_50:>6} Trades würden passieren  ({above_50/n_total*100:.1f}%)")
    print(f"  ≥ 0.45  →  {above_45:>6} Trades würden passieren  ({above_45/n_total*100:.1f}%)")
    print()
    print(f"Confidence-Verteilung (max_proba je Sample):")
    print(_ascii_hist(max_probas.tolist(), bins=12))
    print()

    # Getrennte Confidence für jede Klasse
    for cls_idx, cls_name in [(0, "neutral"), (1, "long"), (2, "short")]:
        cls_probas = probas[:, cls_idx]
        print(f"P({cls_name})-Verteilung (alle Samples):")
        print(_ascii_hist(cls_probas.tolist(), bins=10))
        print()

    # Empfehlung
    median_conf  = float(np.median(max_probas))
    pct_above_55 = above_55 / n_total * 100
    print(f"Diagnose:")
    print(f"  Median Confidence:  {median_conf:.3f}")
    if median_conf < 0.45:
        print(f"  → Modell ist schlecht kalibriert oder hat zu wenig diskriminative Power.")
        print(f"  → Ursachen prüfen: F&G-Platzhalter (50.0 fix im Training), Feature-Shift?")
    elif pct_above_55 < 5.0:
        print(f"  → Nur {pct_above_55:.1f}% der Signale überschreiten 0.55.")
        print(f"  → Schwelle auf 0.50 senken würde {above_50/n_total*100:.1f}% durchlassen.")
        print(f"  → Empfehlung: erst mehr echte/shadow Trades sammeln, dann neu evaluieren.")
    else:
        print(f"  → {pct_above_55:.1f}% der Signale über 0.55 — Schwelle erscheint sinnvoll.")


def diagnose_win(n_samples: int):
    """Analysiert Win-Modell B auf network.db."""
    from ml_network import ml_network, _win_features_from_row
    from network_db import get_training_data

    model = ml_network._win_base
    if model is None:
        print("Kein Win-Basis-Modell geladen.")
        return

    rows = get_training_data(limit=n_samples)
    rows = [r for r in rows if r.get("pnl") is not None]
    if not rows:
        print("Keine Trainingsdaten in network.db.")
        return

    X = np.array([_win_features_from_row(r) for r in rows], dtype=np.float32)
    y = np.array([1 if float(r["pnl"]) > 0 else 0 for r in rows], dtype=np.int32)

    probas   = model.predict_proba(X)[:, 1]  # P(win)
    preds    = (probas >= 0.42).astype(int)

    n_total  = len(y)
    n_pos    = y.sum()
    n_neg    = n_total - n_pos
    above_42 = (probas >= 0.42).sum()
    above_50 = (probas >= 0.50).sum()

    print(f"\n{'='*60}")
    print(f"Win-Modell B — Basis  ({n_total} Samples)")
    print(f"{'='*60}")
    print(f"Labels: Gewinner={n_pos}  Verlierer={n_neg}  "
          f"(Base-WR={n_pos/n_total*100:.1f}%)")
    print()
    print(f"P(win)-Schwellen (aktuell: 0.42):")
    print(f"  ≥ 0.42  →  {above_42:>6} Trades würden passieren  ({above_42/n_total*100:.1f}%)")
    print(f"  ≥ 0.50  →  {above_50:>6} Trades würden passieren  ({above_50/n_total*100:.1f}%)")
    print()
    print(f"P(win)-Verteilung:")
    print(_ascii_hist(probas.tolist(), bins=12))
    print()

    # Genauigkeit bei aktueller Schwelle
    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    tn = ((preds == 0) & (y == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"Bei Schwelle 0.42:")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default=None, help="Nur dieses Symbol")
    parser.add_argument("--win",     action="store_true", help="Win-Modell B analysieren")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Anzahl Kerzen/Zeilen für Analyse (Standard: 10000)")
    args = parser.parse_args()

    if args.win:
        diagnose_win(args.samples)
    else:
        targets = [args.symbol] if args.symbol else SYMBOLS
        for sym in targets:
            diagnose_candle(sym, args.samples)
