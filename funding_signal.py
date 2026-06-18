"""
funding_signal.py — Funding als PRÄDIKTIVES Crowding-Signal (2026-06-18)

Letzter kostenloser Edge-Test. Anders als funding_carry.py (Carry ernten):
Hypothese — extrem hohes positives Funding = überhebelte Longs = Crowding =
Reversal-Risiko -> CONTRARIAN dagegen positionieren (short bei hohem Funding,
long bei negativem). Dokumentierter Effekt; Daten (Funding + Preis) lokal da.

Tests:
  1. Sagt Funding-Level die Forward-Rendite überhaupt voraus? (Korrelation +
     Quartil-Buckets, pro Symbol + gepoolt). Contrarian erwartet NEG. Korrelation.
  2. Time-Series-Contrarian: extremes Funding (Perzentil) -> Gegenposition,
     H Tage halten, nicht-überlappend. Expectancy + t + Bootstrap.
  3. Cross-Sectional-Contrarian (marktneutral, beta-frei): höchstes Funding
     short / niedrigstes long über die 5 Symbole.

Reines Offline-Backtesting (Funding-Cache + 15m-CSVs). Rührt Live-System nicht an.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
HIST = ROOT / "data" / "history"
FUND = ROOT / "data" / "funding"
SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]
FEE = 0.0010
RNG = np.random.default_rng(42)
H_GRID = [3, 5, 7, 10, 14]


def load():
    """Pro Symbol: daily DataFrame mit close + daily Funding (Σ stündlich)."""
    out = {}
    for s in SYMBOLS:
        px = pd.read_csv(HIST / f"{s}_15m.csv")
        px["dt"] = pd.to_datetime(px["timestamp"], unit="ms", utc=True)
        close = px.set_index("dt")["close"].resample("1D").last()
        fd = pd.read_csv(FUND / f"{s}_funding.csv", parse_dates=["timestamp"])
        fd["d"] = fd["timestamp"].dt.floor("D")
        fday = fd.groupby("d")["relativeFundingRate"].sum()   # Tages-Funding
        df = pd.concat({"close": close, "funding": fday}, axis=1).dropna()
        out[s] = df
    return out


def tstat(r):
    if len(r) < 2: return np.nan
    sd = r.std(ddof=1)
    return r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else np.nan


def boot_ci(r, nb=3000):
    n = len(r)
    b = np.array([RNG.choice(r, size=n, replace=True).mean() for _ in range(nb)])
    return np.percentile(b, [2.5, 97.5])


def block(t):
    print("\n" + "=" * 80); print(t); print("=" * 80)


def test_predictive(data):
    block("1. Sagt Funding die Forward-Rendite voraus? (Contrarian -> erwartet NEG corr)")
    print(f"{'Symbol':<12}{'H':>4}{'corr(f,fwd)':>14}{'Q-hoch fwd%':>13}{'Q-tief fwd%':>13}{'Diff%':>9}")
    pooled = {h: [] for h in H_GRID}
    for s in SYMBOLS:
        df = data[s]
        c = df["close"].to_numpy(); f = df["funding"].to_numpy()
        for H in H_GRID:
            fwd = np.full(len(c), np.nan)
            fwd[:-H] = c[H:] / c[:-H] - 1.0
            m = np.isfinite(fwd)
            fv, rv = f[m], fwd[m]
            if len(fv) < 30: continue
            corr = np.corrcoef(fv, rv)[0, 1]
            q_hi = np.percentile(fv, 75); q_lo = np.percentile(fv, 25)
            hi_fwd = rv[fv >= q_hi].mean() * 100
            lo_fwd = rv[fv <= q_lo].mean() * 100
            pooled[H].append((fv, rv))
            if H in (5, 10):
                print(f"{s:<12}{H:>4}{corr:>14.3f}{hi_fwd:>12.2f}%{lo_fwd:>12.2f}%{hi_fwd-lo_fwd:>8.2f}")
    print("  --- gepoolt über alle Symbole ---")
    for H in H_GRID:
        if not pooled[H]: continue
        fv = np.concatenate([p[0] for p in pooled[H]])
        rv = np.concatenate([p[1] for p in pooled[H]])
        corr = np.corrcoef(fv, rv)[0, 1]
        qh, ql = np.percentile(fv, 75), np.percentile(fv, 25)
        hi, lo = rv[fv >= qh].mean()*100, rv[fv <= ql].mean()*100
        print(f"{'POOLED':<12}{H:>4}{corr:>14.3f}{hi:>12.2f}%{lo:>12.2f}%{hi-lo:>8.2f}")


def test_ts_contrarian(data, pct=0.8):
    block(f"2. Time-Series-Contrarian (Funding-Perzentil {int(pct*100)}%, nicht-überlappend)")
    print(f"{'Symbol':<12}{'H':>4}{'n':>6}{'Exp/Tr%':>10}{'t':>7}{'WR%':>6}")
    allr = {h: [] for h in H_GRID}
    for s in SYMBOLS:
        df = data[s]
        c = df["close"].to_numpy(); f = df["funding"].to_numpy()
        hi = np.quantile(f, pct); lo = np.quantile(f, 1 - pct)
        for H in H_GRID:
            rets, t = [], 0
            n = len(c)
            while t + H < n:
                if f[t] >= hi:        # crowded long -> short (contrarian)
                    rets.append(-(c[t + H] / c[t] - 1.0) - FEE); t += H
                elif f[t] <= lo:      # crowded short -> long
                    rets.append((c[t + H] / c[t] - 1.0) - FEE); t += H
                else:
                    t += 1
            r = np.array(rets)
            allr[H].append(r)
            if H in (5, 10) and len(r):
                print(f"{s:<12}{H:>4}{len(r):>6}{r.mean()*100:>9.2f}{tstat(r):>7.2f}{(r>0).mean()*100:>6.0f}")
    print("  --- gepoolt ---")
    for H in H_GRID:
        r = np.concatenate(allr[H]) if allr[H] else np.array([])
        if len(r) < 10: continue
        ci = boot_ci(r)
        print(f"{'POOLED':<12}{H:>4}{len(r):>6}{r.mean()*100:>9.2f}{tstat(r):>7.2f}"
              f"  95%CI=[{ci[0]*100:+.2f}%,{ci[1]*100:+.2f}%]")


def test_xs_contrarian(data, k=2):
    block(f"3. Cross-Sectional-Contrarian (marktneutral: höchstes Funding short / tiefstes long, k={k})")
    # gemeinsame Tage
    closes = pd.concat({s: data[s]["close"] for s in SYMBOLS}, axis=1).dropna()
    funds = pd.concat({s: data[s]["funding"] for s in SYMBOLS}, axis=1).reindex(closes.index)
    C = closes.to_numpy(); F = funds.to_numpy()
    dates = closes.index
    print(f"  {len(dates)} gemeinsame Tage  {dates[0].date()} -> {dates[-1].date()}")
    print(f"{'H':>4}{'n':>6}{'Exp/Reb%':>11}{'t':>7}{'95%CI':>22}")
    for H in H_GRID:
        rets, t, n = [], 0, len(dates)
        while t + H < n:
            past_f = F[t]; fwd = C[t + H] / C[t] - 1.0
            if np.any(~np.isfinite(past_f)) or np.any(~np.isfinite(fwd)):
                t += H; continue
            order = np.argsort(past_f)        # tief -> hoch Funding
            longs = order[:k]                  # tiefstes Funding -> long
            shorts = order[-k:]                # höchstes Funding -> short
            rets.append((fwd[longs].mean() - FEE + (-fwd[shorts].mean() - FEE)) / 2.0)
            t += H
        r = np.array(rets)
        if len(r) < 5: continue
        ci = boot_ci(r)
        flag = "  <<< +" if ci[0] > 0 else ""
        print(f"{H:>4}{len(r):>6}{r.mean()*100:>11.3f}{tstat(r):>7.2f}"
              f"   [{ci[0]*100:+.3f}%,{ci[1]*100:+.3f}%]{flag}")


if __name__ == "__main__":
    data = load()
    print("Geladen:", {s: len(data[s]) for s in SYMBOLS})
    test_predictive(data)
    test_ts_contrarian(data)
    test_xs_contrarian(data)
    block("LESART")
    print("Echtes Signal nur, wenn: (1) corr(f,fwd) konsistent NEGATIV (hohes")
    print("Funding -> schwächere Forward-Rendite), (2) TS/XS-Contrarian-CI ÜBER 0.")
    print("Alles um 0 = Funding trägt keine prädiktive Info (wie der Rest).")
