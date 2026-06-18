"""
daily_backtest.py — Phase-0-Validierung Daily-Long-Momentum (2026-06-18)

Zweck: ehrlicher Walk-Forward auf Daily-Long-Momentum, BEVOR irgendein
produktives Daily-System gebaut wird. Reines Offline-Backtesting auf den
vorhandenen 15m-History-CSVs. Rührt network.db / Exchange / Live-System NICHT an.

Checks (siehe Memory project-pivot-daily-todo, Phase 0):
  1. Walk-Forward: (L,H) IN-SAMPLE wählen, OOS testen, rollen (12M/3M, Embargo>=H).
  2. Bootstrap-95%-CI + t-stat der gepoolten OOS-Expectancy.
  3. Gebühren-Sensitivität (0.14% / 0.2% / 0.3%) + 1 Tag verzögerter Entry.
  4. Pro Symbol einzeln.
  5. Vergleich Hindsight-fix L20/H10 vs ehrlich gewählt vs Buy&Hold.

Kein Schwellen-Senken, kein Optimieren aufs Testfenster. Expectancy nach
Kosten ist das Maß.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HIST = Path(__file__).parent / "data" / "history"
SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]

# Parameter-Gitter fürs In-Sample-Auswählen
L_GRID = [10, 15, 20, 25, 30]
H_GRID = [5, 7, 10]

TRAIN_DAYS = 365
TEST_DAYS = 90
MIN_TRAIN_TRADES = 5     # sonst Fenster überspringen (zu wenig zum Auswählen)
FEE_BASE = 0.0014        # 0.14% round-trip pro Trade (wie 15min-System)
RNG = np.random.default_rng(42)


def load_daily(symbol: str) -> pd.DataFrame:
    """15m-CSV -> Tageskerzen (UTC-Tag)."""
    df = pd.read_csv(HIST / f"{symbol}_15m.csv")
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    daily = df.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return daily


def extract_trades(daily: pd.DataFrame, L: int, H: int, fee: float,
                   delay_entry: bool = False) -> np.ndarray:
    """
    Long-only Time-Series-Momentum, nicht-überlappend.
    Signal am Tag t (close): wenn close[t]/close[t-L]-1 > 0 -> long.
    Entry: close[t] (oder open[t+1] bei delay_entry), Exit: close[t+H].
    Rückgabe: array der Netto-Renditen pro Trade (nach fee).
    """
    c = daily["close"].to_numpy()
    o = daily["open"].to_numpy()
    n = len(c)
    rets = []
    t = L
    while t < n:
        mom = c[t] / c[t - L] - 1.0
        if mom > 0:
            if delay_entry:
                ei = t + 1            # Entry next-day open
                xi = t + 1 + H        # Exit H Tage später (close)
                if xi >= n or ei >= n:
                    break
                entry = o[ei]
                exit_ = c[xi]
                step = (xi - t)
            else:
                xi = t + H
                if xi >= n:
                    break
                entry = c[t]
                exit_ = c[xi]
                step = H
            gross = exit_ / entry - 1.0
            rets.append(gross - fee)
            t += step              # nicht überlappend
        else:
            t += 1
    return np.array(rets, dtype=float)


def trades_in_window(daily: pd.DataFrame, lo, hi, L, H, fee, delay_entry=False):
    """Trades, deren SIGNAL-Tag in [lo, hi) liegt (Slice + lokale Extraktion)."""
    # genug Vorlauf für Lookback mitnehmen
    start_idx = daily.index.searchsorted(lo)
    pad = max(L + 2, 0)
    sub = daily.iloc[max(0, start_idx - pad):daily.index.searchsorted(hi) + H + 2]
    if len(sub) <= L + H:
        return np.array([])
    # Maske: nur Trades mit Signal-Tag >= lo (Vorlauf raus)
    c = sub["close"].to_numpy()
    o = sub["open"].to_numpy()
    idx = sub.index
    n = len(c)
    rets = []
    t = L
    while t < n:
        sig_day = idx[t]
        mom = c[t] / c[t - L] - 1.0
        if mom > 0:
            if delay_entry:
                ei, xi = t + 1, t + 1 + H
                if xi >= n:
                    break
                entry, exit_, step = o[ei], c[xi], (xi - t)
            else:
                xi = t + H
                if xi >= n:
                    break
                entry, exit_, step = c[t], c[xi], H
            if lo <= sig_day < hi:
                rets.append(exit_ / entry - 1.0 - fee)
            t += step
        else:
            t += 1
    return np.array(rets, dtype=float)


def pick_params_insample(daily, lo, hi, fee):
    """Wähle (L,H) mit bester mittlerer Expectancy im Train-Fenster."""
    best, best_exp = None, -np.inf
    for L in L_GRID:
        for H in H_GRID:
            r = trades_in_window(daily, lo, hi, L, H, fee)
            if len(r) >= MIN_TRAIN_TRADES:
                e = r.mean()
                if e > best_exp:
                    best_exp, best = e, (L, H)
    return best


def walk_forward(daily, fee, delay_entry=False, fixed=None):
    """
    Rollender Walk-Forward. fixed=(L,H) -> keine In-Sample-Wahl (Hindsight-Baseline).
    Rückgabe: (oos_returns array, list of (test_start, L, H, n)).
    """
    days = daily.index
    if len(days) < TRAIN_DAYS + TEST_DAYS + 30:
        return np.array([]), []
    oos = []
    log = []
    start = days[0]
    end = days[-1]
    cur = start + pd.Timedelta(days=TRAIN_DAYS)
    while True:
        train_lo = cur - pd.Timedelta(days=TRAIN_DAYS)
        train_hi = cur
        # Embargo: Lücke = max(H_GRID) Tage, damit Train-Trades nicht ins Test lecken
        embargo = max(H_GRID)
        test_lo = cur + pd.Timedelta(days=embargo)
        test_hi = test_lo + pd.Timedelta(days=TEST_DAYS)
        if test_hi > end:
            break
        if fixed is not None:
            L, H = fixed
        else:
            pp = pick_params_insample(daily, train_lo, train_hi, fee)
            if pp is None:
                cur = cur + pd.Timedelta(days=TEST_DAYS)
                continue
            L, H = pp
        r = trades_in_window(daily, test_lo, test_hi, L, H, fee, delay_entry)
        if len(r):
            oos.append(r)
            log.append((test_lo.date(), L, H, len(r)))
        cur = cur + pd.Timedelta(days=TEST_DAYS)
    oos = np.concatenate(oos) if oos else np.array([])
    return oos, log


def stats(r):
    """n, mean%, t-stat, bootstrap-95%-CI der Mean (%)."""
    if len(r) == 0:
        return dict(n=0, mean=np.nan, t=np.nan, lo=np.nan, hi=np.nan, total=np.nan, wr=np.nan)
    n = len(r)
    mean = r.mean()
    sd = r.std(ddof=1) if n > 1 else 0.0
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
    # bootstrap
    boots = np.array([RNG.choice(r, size=n, replace=True).mean() for _ in range(5000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=n, mean=mean * 100, t=t, lo=lo * 100, hi=hi * 100,
                total=r.sum() * 100, wr=(r > 0).mean() * 100)


def fmt(s):
    if s["n"] == 0:
        return "   (keine Trades)"
    return (f"n={s['n']:>4}  Exp/Trade={s['mean']:+.3f}%  t={s['t']:+.2f}  "
            f"95%CI=[{s['lo']:+.3f}%,{s['hi']:+.3f}%]  WR={s['wr']:.0f}%  "
            f"ΣOOS={s['total']:+.1f}%")


def unconditional_long(daily, lo, hi, H, fee):
    """ALLE nicht-überlappenden H-Tage-Longs im Fenster, OHNE Momentum-Filter.
    Isoliert Beta (reine Aufwärtsdrift) vom Momentum-Signal."""
    start_idx = daily.index.searchsorted(lo)
    sub = daily.iloc[max(0, start_idx):daily.index.searchsorted(hi) + H + 2]
    c = sub["close"].to_numpy()
    idx = sub.index
    n = len(c)
    rets, t = [], 0
    while t + H < n:
        if lo <= idx[t] < hi:
            rets.append(c[t + H] / c[t] - 1.0 - fee)
        t += H
    return np.array(rets, dtype=float)


def buyhold_oos(daily, log):
    """Buy&Hold-Rendite über die genutzten Test-Fenster (Vergleich)."""
    if not log:
        return np.nan
    total = 0.0
    for (ts, L, H, n) in log:
        lo = pd.Timestamp(ts, tz="UTC")
        hi = lo + pd.Timedelta(days=TEST_DAYS)
        sub = daily[(daily.index >= lo) & (daily.index < hi)]
        if len(sub) > 1:
            total += sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0
    return total * 100


def wf_daily_curve(daily, fee):
    """Walk-Forward als tägliche Positions-/Renditereihe (OOS).
    Position 0/1 (long/flat) je Tag aus dem in-sample gewählten (L,H);
    Fee fee/2 pro Seite an Positionswechsel-Tagen.
    Rückgabe: DataFrame index=Tag mit mom_ret, bh_ret (close-to-close), pos."""
    days = daily.index
    pct = daily["close"].pct_change().to_numpy()
    if len(days) < TRAIN_DAYS + TEST_DAYS + 30:
        return pd.DataFrame()
    pos = np.zeros(len(days))
    in_oos = np.zeros(len(days), dtype=bool)
    embargo = max(H_GRID)
    cur = days[0] + pd.Timedelta(days=TRAIN_DAYS)
    end = days[-1]
    while True:
        train_lo = cur - pd.Timedelta(days=TRAIN_DAYS)
        test_lo = cur + pd.Timedelta(days=embargo)
        test_hi = test_lo + pd.Timedelta(days=TEST_DAYS)
        if test_hi > end:
            break
        pp = pick_params_insample(daily, train_lo, cur, fee)
        if pp is None:
            cur = cur + pd.Timedelta(days=TEST_DAYS); continue
        L, H = pp
        i0 = days.searchsorted(test_lo)
        i1 = days.searchsorted(test_hi)
        for i in range(i0, i1):
            in_oos[i] = True
        c = daily["close"].to_numpy()
        t = i0
        while t < i1:
            if t - L < 0:
                t += 1; continue
            if c[t] / c[t - L] - 1.0 > 0:           # long-Signal an Tag t
                for d in range(t + 1, min(t + H + 1, len(days))):
                    pos[d] = 1.0
                t += H
            else:
                t += 1
        cur = cur + pd.Timedelta(days=TEST_DAYS)
    df = pd.DataFrame(index=days)
    df["pos"] = pos
    df["in_oos"] = in_oos
    df["bh_ret"] = np.where(in_oos, pct, np.nan)
    chg = np.abs(np.diff(np.concatenate([[0.0], pos])))   # Positionswechsel
    df["mom_ret"] = np.where(in_oos, pos * pct - chg * (fee / 2.0), np.nan)
    return df[df["in_oos"]]


def curve_stats(r):
    """Sharpe (ann., 365), max Drawdown, Gesamt, Zeit-im-Markt."""
    r = pd.Series(r).fillna(0.0).to_numpy()
    if len(r) == 0:
        return dict(sharpe=np.nan, mdd=np.nan, total=np.nan, days=0)
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    return dict(sharpe=sharpe, mdd=mdd * 100, total=(eq[-1] - 1) * 100, days=len(r))


def run_risk():
    print("=" * 78)
    print("RISK-ADJUSTED  Momentum-Long (OOS, walk-forward) vs Buy&Hold")
    print("Sharpe ann.(365) | max Drawdown | Gesamt%  — gleiche OOS-Tage")
    print("=" * 78)
    dailies = {s: load_daily(s) for s in SYMBOLS}
    port_mom, port_bh = [], []
    print(f"\n{'Symbol':<12}{'':<4}{'Sharpe':>8}{'maxDD':>9}{'Gesamt':>10}{'Tage':>7}{'%Markt':>8}")
    for s in SYMBOLS:
        df = wf_daily_curve(dailies[s], FEE_BASE)
        if df.empty:
            continue
        m = curve_stats(df["mom_ret"]); b = curve_stats(df["bh_ret"])
        tim = df["pos"].mean() * 100
        print(f"{s:<12}{'MOM':<4}{m['sharpe']:>8.2f}{m['mdd']:>8.1f}%{m['total']:>9.0f}%{m['days']:>7}{tim:>7.0f}%")
        print(f"{'':<12}{'B&H':<4}{b['sharpe']:>8.2f}{b['mdd']:>8.1f}%{b['total']:>9.0f}%{b['days']:>7}{'100':>7}%")
        port_mom.append(df["mom_ret"].rename(s))
        port_bh.append(df["bh_ret"].rename(s))
    # Equal-Weight-Portfolio (Mittel der täglichen Renditen über Symbole)
    pm = pd.concat(port_mom, axis=1).mean(axis=1)
    pb = pd.concat(port_bh, axis=1).mean(axis=1)
    m = curve_stats(pm); b = curve_stats(pb)
    print("-" * 58)
    print(f"{'PORTFOLIO':<12}{'MOM':<4}{m['sharpe']:>8.2f}{m['mdd']:>8.1f}%{m['total']:>9.0f}%{m['days']:>7}")
    print(f"{'(5 eq-wt)':<12}{'B&H':<4}{b['sharpe']:>8.2f}{b['mdd']:>8.1f}%{b['total']:>9.0f}%{b['days']:>7}")
    print("\nLesart: Wenn MOM-Sharpe > B&H-Sharpe UND |maxDD| deutlich kleiner,")
    print("liegt Momentums Wert im Risikoschutz (nicht in der Per-Trade-Rendite).")
    print("=" * 78)


def main():
    print("=" * 78)
    print("PHASE-0-VALIDIERUNG  Daily-Long-Momentum  Walk-Forward")
    print(f"Train {TRAIN_DAYS}d / Test {TEST_DAYS}d / Embargo {max(H_GRID)}d | "
          f"Gitter L{L_GRID} x H{H_GRID} | Fee-Basis {FEE_BASE*100:.2f}%")
    print("=" * 78)

    dailies = {s: load_daily(s) for s in SYMBOLS}
    for s, d in dailies.items():
        print(f"  {s}: {len(d)} Tageskerzen  {d.index[0].date()} -> {d.index[-1].date()}")

    # ---- CHECK 1+2: ehrlicher Walk-Forward, gepoolt über alle Symbole ----
    print("\n" + "-" * 78)
    print("CHECK 1+2  Ehrlich gewählt (In-Sample (L,H) -> OOS), GEPOOLT, Fee 0.14%")
    print("-" * 78)
    all_oos = []
    all_log = []
    per_symbol_oos = {}
    for s in SYMBOLS:
        oos, log = walk_forward(dailies[s], FEE_BASE)
        per_symbol_oos[s] = oos
        all_oos.append(oos)
        all_log += [(s, *l) for l in log]
    pooled = np.concatenate([o for o in all_oos if len(o)]) if any(len(o) for o in all_oos) else np.array([])
    print("  GEPOOLT:        " + fmt(stats(pooled)))

    # frühe vs späte Hälfte (OOS-Abschwächung prüfen)
    if len(pooled) > 10:
        half = len(pooled) // 2
        print("  1. Hälfte OOS:  " + fmt(stats(pooled[:half])))
        print("  2. Hälfte OOS:  " + fmt(stats(pooled[half:])))

    # ---- CHECK 3: pro Symbol ----
    print("\n" + "-" * 78)
    print("CHECK 3  Pro Symbol (ehrlich gewählt, OOS, Fee 0.14%)")
    print("-" * 78)
    for s in SYMBOLS:
        print(f"  {s}:  " + fmt(stats(per_symbol_oos[s])))

    # ---- CHECK 4: Gebühren-Sensitivität + verzögerter Entry (gepoolt) ----
    print("\n" + "-" * 78)
    print("CHECK 4  Gebühren-Sensitivität + 1-Tag-verzögerter Entry (gepoolt)")
    print("-" * 78)
    for fee in (0.0014, 0.0020, 0.0030):
        outs = []
        for s in SYMBOLS:
            o, _ = walk_forward(dailies[s], fee)
            if len(o):
                outs.append(o)
        p = np.concatenate(outs) if outs else np.array([])
        print(f"  Fee {fee*100:.2f}% Entry@close:    " + fmt(stats(p)))
    for fee in (0.0014, 0.0030):
        outs = []
        for s in SYMBOLS:
            o, _ = walk_forward(dailies[s], fee, delay_entry=True)
            if len(o):
                outs.append(o)
        p = np.concatenate(outs) if outs else np.array([])
        print(f"  Fee {fee*100:.2f}% Entry@open+1d: " + fmt(stats(p)))

    # ---- CHECK 5: Hindsight-fix L20/H10 vs Buy&Hold ----
    print("\n" + "-" * 78)
    print("CHECK 5  Baselines: Hindsight-fix L20/H10 (OOS-Fenster) vs Buy&Hold")
    print("-" * 78)
    fixed_outs = []
    fixed_log = []
    for s in SYMBOLS:
        o, lg = walk_forward(dailies[s], FEE_BASE, fixed=(20, 10))
        if len(o):
            fixed_outs.append(o)
        fixed_log += [(s, *l) for l in lg]
    pf = np.concatenate(fixed_outs) if fixed_outs else np.array([])
    print("  Hindsight L20/H10:  " + fmt(stats(pf)))
    bh = sum(buyhold_oos(dailies[s], [l[1:] for l in all_log if l[0] == s]) for s in SYMBOLS)
    print(f"  Buy&Hold (Σ über genutzte Test-Fenster, alle Symbole): {bh:+.1f}%")

    # ---- CHECK 6: EDGE vs BETA — unkonditionierter Long-Benchmark ----
    print("\n" + "-" * 78)
    print("CHECK 6  EDGE vs BETA: Momentum-gefiltert vs unkond. Long (gleiche Fenster+H)")
    print("-" * 78)
    uncond = []
    for s in SYMBOLS:
        for (sym, ts, L, H, n) in [(x[0], *x[1:]) for x in all_log if x[0] == s]:
            lo = pd.Timestamp(ts, tz="UTC")
            hi = lo + pd.Timedelta(days=TEST_DAYS)
            u = unconditional_long(dailies[s], lo, hi, H, FEE_BASE)
            if len(u):
                uncond.append(u)
    pu = np.concatenate(uncond) if uncond else np.array([])
    print("  Momentum-gefiltert: " + fmt(stats(pooled)))
    print("  Unkond. Long:       " + fmt(stats(pu)))
    if len(pooled) and len(pu):
        diff = pooled.mean() - pu.mean()
        print(f"  --> Edge-Beitrag des Filters: {diff*100:+.3f}%/Trade "
              f"(Momentum {pooled.mean()*100:+.3f}% minus Beta {pu.mean()*100:+.3f}%)")

    print("\n" + "=" * 78)
    print("FERTIG. Maß = Expectancy/Trade nach Kosten + ob CI über 0 liegt.")
    print("=" * 78)


if __name__ == "__main__":
    if "--risk" in sys.argv:
        run_risk()
    else:
        main()
