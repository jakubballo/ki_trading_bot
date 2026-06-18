"""
signal_scan.py — Signal-Ideen-Scan (2026-06-18), Phase-0-Fortsetzung

Sucht nach echtem Alpha jenseits von Long-Beta. Schwerpunkt:
CROSS-SECTIONAL LONG/SHORT über die 5 Symbole — marktneutral, daher
beta-frei (im Gegensatz zu time-series Daily-Momentum, das sich als
reines Long-Beta entlarvt hat, siehe daily_backtest.py).

Ein Lookback-Gitter deckt beide Richtungen ab:
  - kleiner L + "long Verlierer"  = Short-Term-Reversal
  - großer  L + "long Gewinner"   = Cross-Sectional-Momentum

Schritt 1: In-Sample-Grid (schnell, t-stat). Clustert nichts positiv -> fertig.
Schritt 2 (--wf): Walk-Forward-OOS + Bootstrap auf dem besten Setup.

Reines Offline-Backtesting auf den 15m-CSVs (auf Daily resampled).
Rührt network.db / Exchange / Live-System NICHT an.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HIST = Path(__file__).parent / "data" / "history"
SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]
FEE = 0.0014        # round-trip pro Position
K = 2               # long top-K, short bottom-K (von 5)
RNG = np.random.default_rng(42)

L_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30]
H_GRID = [1, 2, 3, 5, 7, 10]


def load_close_matrix():
    """Tages-Closes aller Symbole, auf gemeinsame Tage ausgerichtet."""
    series = {}
    for s in SYMBOLS:
        df = pd.read_csv(HIST / f"{s}_15m.csv")
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        d = df.set_index("dt").sort_index()["close"].resample("1D").last()
        series[s] = d
    mat = pd.concat(series, axis=1).dropna()
    return mat


def ls_returns(mat, L, H, direction, fee=FEE, k=K, lo=None, hi=None):
    """
    Cross-Sectional Long/Short, nicht-überlappend (Rebalance alle H Tage).
    direction: 'mom' = long Gewinner/short Verlierer; 'rev' = invers.
    Rückgabe: array der Portfolio-Renditen pro Rebalance (auf Brutto-Kapital,
    Gross-Exposure 2 = 1x long + 1x short, Netto ~0 -> marktneutral).
    """
    c = mat.to_numpy()
    dates = mat.index
    n = len(c)
    rets = []
    t = L
    while t + H < n:
        if lo is not None and not (lo <= dates[t] < hi):
            t += H
            continue
        past = c[t] / c[t - L] - 1.0
        fwd = c[t + H] / c[t] - 1.0
        order = np.argsort(past)              # aufsteigend: [Verlierer ... Gewinner]
        if direction == "mom":
            longs, shorts = order[-k:], order[:k]
        else:                                  # reversal
            longs, shorts = order[:k], order[-k:]
        long_pnl = np.mean(fwd[longs] - fee)
        short_pnl = np.mean(-fwd[shorts] - fee)
        rets.append((long_pnl + short_pnl) / 2.0)
        t += H
    return np.array(rets, dtype=float)


def tstat(r):
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    return r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else np.nan


def boot_ci(r, n_boot=3000):
    n = len(r)
    b = np.array([RNG.choice(r, size=n, replace=True).mean() for _ in range(n_boot)])
    return np.percentile(b, [2.5, 97.5])


def grid_scan(mat):
    print("=" * 84)
    print("CROSS-SECTIONAL LONG/SHORT  In-Sample-Grid  (marktneutral, top/bottom-%d)" % K)
    print("Per-Rebalance-Rendite auf Brutto-Kapital nach Fee %.2f%% | t-stat | n" % (FEE * 100))
    print("=" * 84)
    print(f"{'dir':<5}{'L':>4}{'H':>4}{'Exp/Reb%':>11}{'t':>8}{'n':>6}{'Σ%':>9}")
    hits = []
    for direction in ("mom", "rev"):
        for L in L_GRID:
            for H in H_GRID:
                r = ls_returns(mat, L, H, direction)
                if len(r) < 10:
                    continue
                t = tstat(r)
                e = r.mean() * 100
                flag = "  <<< +" if (t and t > 2) else ("  < -" if (t and t < -2) else "")
                if abs(t or 0) > 1.8:
                    print(f"{direction:<5}{L:>4}{H:>4}{e:>11.3f}{t:>8.2f}{len(r):>6}{r.sum()*100:>9.1f}{flag}")
                if t and t > 2:
                    hits.append((direction, L, H, e, t, len(r)))
    print("-" * 84)
    if hits:
        print(f"{len(hits)} Setup(s) mit t>2 in-sample (NICHT OOS — Hindsight!). "
              f"Bestes: {max(hits, key=lambda x: x[4])}")
    else:
        print("KEIN Setup mit t>2 in-sample. Cross-Sectional L/S zeigt hier kein Alpha.")
    print("(Nur |t|>1.8 gelistet. Schritt 2: --wf validiert das beste Setup OOS.)")
    return hits


def walk_forward_ls(mat, fee=FEE):
    """OOS-Walk-Forward: (dir,L,H) in-sample wählen, OOS testen, rollen."""
    dates = mat.index
    TRAIN, TEST, EMB = 365, 90, max(H_GRID)
    oos = []
    log = []
    cur = dates[0] + pd.Timedelta(days=TRAIN)
    while True:
        tr_lo = cur - pd.Timedelta(days=TRAIN)
        te_lo = cur + pd.Timedelta(days=EMB)
        te_hi = te_lo + pd.Timedelta(days=TEST)
        if te_hi > dates[-1]:
            break
        # in-sample beste (dir,L,H)
        best, be = None, -np.inf
        for direction in ("mom", "rev"):
            for L in L_GRID:
                for H in H_GRID:
                    r = ls_returns(mat, L, H, direction, fee, lo=tr_lo, hi=cur)
                    if len(r) >= 5 and r.mean() > be:
                        be, best = r.mean(), (direction, L, H)
        if best is None:
            cur += pd.Timedelta(days=TEST); continue
        d, L, H = best
        r = ls_returns(mat, L, H, d, fee, lo=te_lo, hi=te_hi)
        if len(r):
            oos.append(r); log.append((te_lo.date(), d, L, H, len(r)))
        cur += pd.Timedelta(days=TEST)
    pooled = np.concatenate(oos) if oos else np.array([])
    print("=" * 84)
    print("WALK-FORWARD OOS  Cross-Sectional L/S  (in-sample (dir,L,H) gewählt)")
    print("=" * 84)
    for (ts, d, L, H, n) in log:
        print(f"  Test {ts}  -> {d} L{L}/H{H}  n={n}")
    if len(pooled):
        ci = boot_ci(pooled)
        print("-" * 84)
        print(f"  GEPOOLT OOS: n={len(pooled)}  Exp/Reb={pooled.mean()*100:+.3f}%  "
              f"t={tstat(pooled):+.2f}  95%CI=[{ci[0]*100:+.3f}%,{ci[1]*100:+.3f}%]  "
              f"Σ={pooled.sum()*100:+.1f}%")
        if ci[0] > 0:
            print("  -> CI über 0: echtes marktneutrales Alpha (selten!). Weiter prüfen.")
        else:
            print("  -> CI kreuzt 0: kein belastbares OOS-Alpha.")
    else:
        print("  Keine OOS-Trades.")
    print("=" * 84)


if __name__ == "__main__":
    mat = load_close_matrix()
    print(f"Daten: {len(mat)} gemeinsame Tage  {mat.index[0].date()} -> {mat.index[-1].date()}\n")
    if "--wf" in sys.argv:
        walk_forward_ls(mat)
    else:
        grid_scan(mat)
