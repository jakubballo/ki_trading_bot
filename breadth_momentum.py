"""
breadth_momentum.py — Cross-Sectional-Momentum auf BREITE (2026-06-18)

Der 5-Symbol-CS-Test (signal_scan.py) war unterpowert. CS-Momentum ist auf
breitem Universum dokumentiert (Top/Bottom-Dezil über viele Coins). Hier:
alle ~308 handelbaren Kraken-Perps, Daily, marktneutral Long/Short-Dezil.

WICHTIG — SURVIVORSHIP-BIAS: Der Charts-Endpoint liefert nur AKTUELL gelistete
Perps. Tote/delistete Coins fehlen -> Aufwärtsverzerrung. Ergebnisse sind eine
OPTIMISTISCHE Obergrenze. Long/Short mildert (Short-Bein), eliminiert aber nicht.

Schritte:
  --fetch   : Daily-Historie aller Perps holen + cachen (data/perp_daily/).
  (default) : In-Sample-Grid CS-Momentum (L,H,Dezil) Long/Short + Long-only vs Beta.
  --wf      : Walk-Forward OOS + Bootstrap auf bestem Setup.

Reines Offline-Backtesting + lesender Public-API-Fetch. Rührt network.db /
Live-System NICHT an.
"""
import sys
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

CACHE = Path(__file__).parent / "data" / "perp_daily"
CACHE.mkdir(parents=True, exist_ok=True)
INSTR = "https://futures.kraken.com/derivatives/api/v3/instruments"
CHART = "https://futures.kraken.com/api/charts/v1/mark/{}/1d"
FEE = 0.0010            # round-trip pro Position (Perp Taker ~0.05%/Seite)
RNG = np.random.default_rng(42)

L_GRID = [15, 20, 30, 45, 60, 90]
H_GRID = [5, 7, 10, 14, 21]
MIN_HISTORY = 120       # Tage Mindesthistorie zum Zeitpunkt des Signals
MIN_UNIVERSE = 20       # min. Coins im Universum, sonst Rebalance überspringen


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def fetch_all():
    inst = _get(INSTR).get("instruments", [])
    syms = sorted(i["symbol"].upper() for i in inst
                  if i.get("tradeable") and i["symbol"].upper().startswith("PF_"))
    print(f"{len(syms)} Perps. Hole Daily-Historie...")
    ok = 0
    for n, s in enumerate(syms, 1):
        f = CACHE / f"{s}.csv"
        if f.exists():
            ok += 1; continue
        try:
            d = _get(CHART.format(s))
            c = d.get("candles", [])
            if c:
                df = pd.DataFrame(c)[["time", "close"]]
                df.to_csv(f, index=False)
                ok += 1
        except Exception as e:
            print(f"  {s} ERR {e}")
        if n % 50 == 0:
            print(f"  {n}/{len(syms)} ...")
        time.sleep(0.25)
    print(f"fertig: {ok} Symbole gecached in {CACHE}")


def load_matrix():
    """Dates × Symbols Close-Matrix aus dem Cache."""
    series = {}
    for f in CACHE.glob("*.csv"):
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.floor("D")
        s = df.set_index("dt")["close"]
        s = s[~s.index.duplicated(keep="last")]
        series[f.stem] = s
    mat = pd.concat(series, axis=1).sort_index()
    return mat


def cs_returns(mat, L, H, dec=0.1, fee=FEE, lo=None, hi=None, mode="ls"):
    """
    Cross-Sectional-Momentum, nicht-überlappend (Rebalance alle H Tage).
    mode: 'ls' marktneutral Long-Top/Short-Bottom-Dezil; 'long' nur Top-Dezil;
          'beta' gleichgewichtetes Gesamt-Universum (Benchmark).
    Universum je Rebalance: Coins mit >=MIN_HISTORY gültigen Closes bis t.
    """
    c = mat.to_numpy(dtype=float)
    dates = mat.index
    n, m = c.shape
    rets = []
    t = L
    while t + H < n:
        if lo is not None and not (lo <= dates[t] < hi):
            t += H; continue
        past = c[t] / c[t - L] - 1.0
        fwd = c[t + H] / c[t] - 1.0
        # gültig: kein NaN in past/fwd UND genug Historie
        hist_ok = np.array([np.isfinite(c[max(0, t - MIN_HISTORY)][j]) for j in range(m)])
        valid = np.isfinite(past) & np.isfinite(fwd) & hist_ok
        idx = np.where(valid)[0]
        if len(idx) < MIN_UNIVERSE:
            t += H; continue
        order = idx[np.argsort(past[idx])]      # aufsteigend
        k = max(2, int(len(idx) * dec))
        if mode == "beta":
            rets.append(fwd[idx].mean() - fee)
        elif mode == "long":
            longs = order[-k:]
            rets.append(fwd[longs].mean() - fee)
        else:  # ls
            longs, shorts = order[-k:], order[:k]
            rets.append((fwd[longs].mean() - fee + (-fwd[shorts].mean() - fee)) / 2.0)
        t += H
    return np.array(rets, dtype=float)


def tstat(r):
    if len(r) < 2: return np.nan
    sd = r.std(ddof=1)
    return r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else np.nan


def boot_ci(r, nb=3000):
    n = len(r)
    b = np.array([RNG.choice(r, size=n, replace=True).mean() for _ in range(nb)])
    return np.percentile(b, [2.5, 97.5])


def grid(mat):
    print("=" * 86)
    print("CS-MOMENTUM BREITE  In-Sample-Grid  (Long/Short-Dezil, marktneutral)")
    print(f"Fee {FEE*100:.2f}%/Pos | {mat.shape[1]} Symbole | "
          f"{mat.index[0].date()} -> {mat.index[-1].date()}")
    print("!! SURVIVORSHIP-BIAS: nur gelistete Coins -> optimistische Obergrenze !!")
    print("=" * 86)
    print(f"{'L':>4}{'H':>4}{'Exp/Reb%':>11}{'t':>8}{'n':>6}{'Σ%':>9}   (nur |t|>1.8)")
    hits = []
    for L in L_GRID:
        for H in H_GRID:
            r = cs_returns(mat, L, H)
            if len(r) < 10: continue
            t = tstat(r); e = r.mean() * 100
            if abs(t or 0) > 1.8:
                fl = "  <<<+" if t > 2 else ("  <-" if t < -2 else "")
                print(f"{L:>4}{H:>4}{e:>11.3f}{t:>8.2f}{len(r):>6}{r.sum()*100:>9.1f}{fl}")
            if t and t > 2:
                hits.append((L, H, e, t, len(r)))
    print("-" * 86)
    print(f"{len(hits)} Setup(s) t>2 in-sample (Hindsight). "
          + (f"Bestes {max(hits,key=lambda x:x[3])[:2]}" if hits else "KEINS."))


def walk_forward(mat, fee=FEE):
    dates = mat.index
    TRAIN, TEST, EMB = 365, 90, max(H_GRID)
    oos_ls, oos_long, oos_beta, log = [], [], [], []
    cur = dates[0] + pd.Timedelta(days=TRAIN)
    while True:
        tr_lo = cur - pd.Timedelta(days=TRAIN)
        te_lo = cur + pd.Timedelta(days=EMB)
        te_hi = te_lo + pd.Timedelta(days=TEST)
        if te_hi > dates[-1]: break
        best, be = None, -np.inf
        for L in L_GRID:
            for H in H_GRID:
                r = cs_returns(mat, L, H, lo=tr_lo, hi=cur)
                if len(r) >= 3 and r.mean() > be:
                    be, best = r.mean(), (L, H)
        if best is None:
            cur += pd.Timedelta(days=TEST); continue
        L, H = best
        rls = cs_returns(mat, L, H, lo=te_lo, hi=te_hi, mode="ls")
        rlo = cs_returns(mat, L, H, lo=te_lo, hi=te_hi, mode="long")
        rbe = cs_returns(mat, L, H, lo=te_lo, hi=te_hi, mode="beta")
        if len(rls):
            oos_ls.append(rls); oos_long.append(rlo); oos_beta.append(rbe)
            log.append((te_lo.date(), L, H, len(rls)))
        cur += pd.Timedelta(days=TEST)
    print("=" * 86)
    print("WALK-FORWARD OOS  CS-Momentum BREITE  (in-sample (L,H) gewählt)")
    print("=" * 86)
    for (ts, L, H, n) in log:
        print(f"  Test {ts} -> L{L}/H{H}  n={n}")
    def rep(name, parts):
        p = np.concatenate(parts) if parts else np.array([])
        if len(p):
            ci = boot_ci(p)
            print(f"  {name:<22} n={len(p):>4} Exp/Reb={p.mean()*100:+.3f}% "
                  f"t={tstat(p):+.2f} 95%CI=[{ci[0]*100:+.3f}%,{ci[1]*100:+.3f}%] "
                  f"Σ={p.sum()*100:+.1f}%")
        return p
    print("-" * 86)
    ls = rep("Long/Short (neutral):", oos_ls)
    lo = rep("Long-only Top-Dezil:", oos_long)
    be = rep("Beta (Universum eq-wt):", oos_beta)
    if len(lo) and len(be):
        print(f"\n  EDGE des Long-Beins über Beta: {(lo.mean()-be.mean())*100:+.3f}%/Reb")
    if len(ls):
        ci = boot_ci(ls)
        print("  -> " + ("L/S-CI ÜBER 0: marktneutrales Alpha-Signal (Survivorship beachten!)"
                          if ci[0] > 0 else "L/S-CI kreuzt 0: kein belastbares OOS-Alpha"))
    print("=" * 86)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch_all(); sys.exit()
    mat = load_matrix()
    print(f"Matrix: {mat.shape[0]} Tage × {mat.shape[1]} Symbole\n")
    if "--wf" in sys.argv:
        walk_forward(mat)
    else:
        grid(mat)
