"""
funding_carry.py — Funding-Carry-Validierung (2026-06-18), Phase-0-Fortsetzung

Testet, ob das Ernten der Perp-Funding-Rate einen echten (marktneutralen)
Edge liefert — der stärkste Rest-Kandidat, nachdem reine Preis-Signale auf
15min/Daily/Cross-Sectional alle leer waren (siehe daily_backtest.py,
signal_scan.py, Memory signal-scan-cross-sectional-no-edge).

Idee: delta-neutral short-perp + long-spot (bei positivem Funding) kassiert
relativeFundingRate je Stunde, Preisbewegung des Perp wird von der Spot-Seite
weggehedged. Netto ≈ Funding − Gebühren − Hedge-Unschärfe.

Datenquelle: Kraken Public historicalfundingrates (v4), keine Auth.
ACHTUNG: Kraken gibt nur ~1 Jahr stündliche Funding-Historie. Daher
Sub-Perioden-Stabilität statt langer Walk-Forward.

Holt + cached Funding lokal nach data/funding/. Rührt network.db / Live-System
NICHT an. Nur lesender Public-API-Fetch + lokale CSVs.
"""
import sys
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]
CACHE = Path(__file__).parent / "data" / "funding"
CACHE.mkdir(parents=True, exist_ok=True)
URL = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates?symbol={}"

# Realistische Kosten (konservativ): ein "Flip" = beide Beine schließen+öffnen.
# Perp Taker ~0.05%, Spot Taker ~0.26% -> round-trip beide Beine grob:
FLIP_COST = 0.0030     # 0.30 % pro Seitenwechsel (4 Fills, konservativ)
ENTRY_COST = 0.0015    # einmaliger Einstieg (beide Beine, einseitig)


def fetch(symbol: str, refresh=False) -> pd.DataFrame:
    f = CACHE / f"{symbol}_funding.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f, parse_dates=["timestamp"])
    req = urllib.request.Request(URL.format(symbol), headers={"User-Agent": "Mozilla/5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    rates = d.get("rates", [])
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "relativeFundingRate"]].dropna().sort_values("timestamp")
    df.to_csv(f, index=False)
    return df


def block(title):
    print("\n" + "=" * 80); print(title); print("=" * 80)


def analyze():
    block("FUNDING-CARRY  Kraken Perp  (relativeFundingRate, stündlich)")
    data = {}
    for s in SYMBOLS:
        df = fetch(s)
        data[s] = df
        r = df["relativeFundingRate"].to_numpy()
        print(f"  {s}: {len(r)} Std.  {df['timestamp'].iloc[0].date()} -> "
              f"{df['timestamp'].iloc[-1].date()}")

    # ---- 1. Größenordnung & Vorzeichen-Persistenz ----
    block("1. Größenordnung (annualisiert) & Vorzeichen")
    print(f"{'Symbol':<12}{'Ø/Std':>11}{'ann.Σ%':>10}{'%pos':>8}{'flips/Tag':>11}{'maxNeg-Strecke':>16}")
    for s in SYMBOLS:
        r = data[s]["relativeFundingRate"].to_numpy()
        ann = r.sum() * 0 + r.mean() * 24 * 365 * 100   # annualisierte Carry, wenn immer Receiver der Ø-Rate
        pos = (r > 0).mean() * 100
        flips = np.sum(np.diff(np.sign(r)) != 0) / (len(r) / 24)
        # längste Strecke mit negativem Funding (Zahler-Phase, wenn man short-perp hält)
        neg = r < 0
        worst = 0; cur = 0
        for x in neg:
            cur = cur + 1 if x else 0
            worst = max(worst, cur)
        print(f"{s:<12}{r.mean()*100:>10.4f}%{ann:>9.1f}%{pos:>7.0f}%{flips:>11.1f}{worst:>13} Std")

    # ---- 2. Carry-Strategien: immer-short-perp vs vorzeichen-folgend ----
    block("2. Netto-Carry nach Kosten  (delta-neutral, ~1J)")
    print("Strategie A = immer Receiver der jeweils aktuellen Funding-Richtung (flippt mit Vorzeichen)")
    print("Strategie B = fixe Seite short-perp (kassiert pos. Funding, zahlt in neg. Phasen)")
    print(f"\n{'Symbol':<12}{'A netto%':>11}{'A Sharpe':>10}{'B netto%':>11}{'B Sharpe':>10}")
    portA, portB = [], []
    for s in SYMBOLS:
        r = data[s]["relativeFundingRate"].to_numpy()
        # A: receive |funding| jede Stunde, aber Flip-Kosten bei Vorzeichenwechsel
        flips_mask = np.concatenate([[False], np.diff(np.sign(r)) != 0])
        a_hourly = np.abs(r) - flips_mask * FLIP_COST
        a_hourly[0] -= ENTRY_COST
        # B: fixe short-perp -> erhält +r wenn r>0, zahlt wenn r<0; nur Einstiegskosten
        b_hourly = r.copy()
        b_hourly[0] -= ENTRY_COST
        portA.append(a_hourly); portB.append(b_hourly)

        def shp(x):
            return x.mean() / x.std() * np.sqrt(24 * 365) if x.std() > 0 else np.nan
        print(f"{s:<12}{a_hourly.sum()*100:>10.1f}%{shp(a_hourly):>10.2f}"
              f"{b_hourly.sum()*100:>10.1f}%{shp(b_hourly):>10.2f}")

    # Portfolio (gleichgewichtet über die 5)
    minlen = min(len(x) for x in portA)
    A = np.mean([x[:minlen] for x in portA], axis=0)
    B = np.mean([x[:minlen] for x in portB], axis=0)
    def shp(x): return x.mean() / x.std() * np.sqrt(24 * 365) if x.std() > 0 else np.nan
    print("-" * 54)
    print(f"{'PORTFOLIO':<12}{A.sum()*100:>10.1f}%{shp(A):>10.2f}{B.sum()*100:>10.1f}%{shp(B):>10.2f}")

    # ---- 3. Sub-Perioden-Stabilität (Quartale) ----
    block("3. Stabilität über Quartale (Strategie B = fixe short-perp, Portfolio)")
    s0 = data["PF_XBTUSD"]
    ts = s0["timestamp"].reset_index(drop=True).iloc[:minlen]
    bser = pd.Series(B, index=ts)
    q = bser.groupby(pd.Grouper(freq="QE")).agg(["sum", "count"])
    for idx, row in q.iterrows():
        print(f"  {idx.date()}  Σ={row['sum']*100:+7.2f}%   ({int(row['count'])} Std)")

    block("LESART")
    print("Carry ist ECHT, wenn: annualisierte Σ deutlich > Kosten, B über alle")
    print("Quartale positiv (nicht nur ein Regime), Flip-Kosten (A) fressen es nicht.")
    print("Restrisiko nicht im Preis (gehedged), sondern in neg. Funding-Strecken +")
    print("Hedge-Unschärfe (hier mit FLIP_COST/ENTRY_COST grob, nicht modelliert: Basis).")


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        for s in SYMBOLS:
            fetch(s, refresh=True)
            print("refreshed", s)
    analyze()
