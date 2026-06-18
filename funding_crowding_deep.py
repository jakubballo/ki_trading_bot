"""
funding_crowding_deep.py — Funding-Crowding vertieft (2026-06-18)

Quetscht die vorhandenen ~1J Funding-Daten auf die PRÄZISE Crowding-Hypothese,
BEVOR wir Infra für Liquidations-/OI-Sammlung bauen. Drei Schärfungen ggü.
funding_signal.py (das grob 80.-Perzentil/symmetrisch testete):

  1. EXTREM-RAND-Sweep: Perzentile 80/85/90/95 — wird der Reversal stärker,
     je extremer das Funding? (Liquidationskaskaden triggern am Rand.)
  2. ASYMMETRIE: Short-bei-hohem-Funding (Long-Crowd) vs Long-bei-negativem
     getrennt — in Krypto sollte die Long-Crowd-Seite die stärkere sein.
  3. SPIKE: schneller Funding-Anstieg (Δ über k Tage) statt absolutem Level.

Pooled über 5 Symbole, Bootstrap-CI. Reines Offline (Funding-Cache + 15m-CSVs).
"""
import sys
import numpy as np
from funding_signal import load, tstat, boot_ci, FEE

sys.stdout.reconfigure(encoding="utf-8")
H_GRID = [5, 7, 10]


def block(t):
    print("\n" + "=" * 82); print(t); print("=" * 82)


def fwd_ret(c, t, H):
    return c[t + H] / c[t] - 1.0


def extreme_sweep(data):
    block("1. EXTREM-RAND-Sweep (contrarian, nicht-überlappend, pooled)")
    print("Hypothese: Reversal stärker, je extremer das Funding-Perzentil.")
    print(f"{'Perzentil':>10}{'H':>4}{'n':>6}{'Exp/Tr%':>10}{'t':>7}{'WR%':>6}{'95%CI':>22}")
    for p in (0.80, 0.85, 0.90, 0.95):
        for H in H_GRID:
            allr = []
            for s in data:
                df = data[s]
                c = df["close"].to_numpy(); f = df["funding"].to_numpy()
                hi = np.quantile(f, p); lo = np.quantile(f, 1 - p)
                t = 0; n = len(c)
                while t + H < n:
                    if f[t] >= hi:
                        allr.append(-fwd_ret(c, t, H) - FEE); t += H
                    elif f[t] <= lo:
                        allr.append(fwd_ret(c, t, H) - FEE); t += H
                    else:
                        t += 1
            r = np.array(allr)
            if len(r) < 10: continue
            ci = boot_ci(r)
            fl = "  <<<+" if ci[0] > 0 else ""
            print(f"{int(p*100):>9}%{H:>4}{len(r):>6}{r.mean()*100:>9.2f}{tstat(r):>7.2f}"
                  f"{(r>0).mean()*100:>6.0f}   [{ci[0]*100:+.2f}%,{ci[1]*100:+.2f}%]{fl}")


def asymmetry(data, p=0.90):
    block(f"2. ASYMMETRIE — getrennte Schenkel (Perzentil {int(p*100)}%)")
    print("HIGH = Funding hoch -> SHORT (Long-Crowd).  LOW = Funding neg -> LONG (Short-Crowd).")
    print(f"{'Schenkel':>10}{'H':>4}{'n':>6}{'Exp/Tr%':>10}{'t':>7}{'WR%':>6}{'95%CI':>22}")
    for leg in ("HIGH->short", "LOW->long"):
        for H in H_GRID:
            allr = []
            for s in data:
                df = data[s]
                c = df["close"].to_numpy(); f = df["funding"].to_numpy()
                hi = np.quantile(f, p); lo = np.quantile(f, 1 - p)
                t = 0; n = len(c)
                while t + H < n:
                    if leg.startswith("HIGH") and f[t] >= hi:
                        allr.append(-fwd_ret(c, t, H) - FEE); t += H
                    elif leg.startswith("LOW") and f[t] <= lo:
                        allr.append(fwd_ret(c, t, H) - FEE); t += H
                    else:
                        t += 1
            r = np.array(allr)
            if len(r) < 8: continue
            ci = boot_ci(r)
            fl = "  <<<+" if ci[0] > 0 else ""
            print(f"{leg:>10}{H:>4}{len(r):>6}{r.mean()*100:>9.2f}{tstat(r):>7.2f}"
                  f"{(r>0).mean()*100:>6.0f}   [{ci[0]*100:+.2f}%,{ci[1]*100:+.2f}%]{fl}")


def spike(data, k=3, p=0.90):
    block(f"3. SPIKE — schneller Funding-Anstieg Δ{k}d (Perzentil {int(p*100)}%) statt Level")
    print("Signal: Funding[t]-Funding[t-k]. Hoher Anstieg = frische Long-Crowd -> short.")
    print(f"{'H':>4}{'n':>6}{'Exp/Tr%':>10}{'t':>7}{'WR%':>6}{'95%CI':>22}")
    for H in H_GRID:
        allr = []
        for s in data:
            df = data[s]
            c = df["close"].to_numpy(); f = df["funding"].to_numpy()
            dacc = np.full(len(f), np.nan)
            dacc[k:] = f[k:] - f[:-k]
            valid = dacc[np.isfinite(dacc)]
            hi = np.quantile(valid, p); lo = np.quantile(valid, 1 - p)
            t = k; n = len(c)
            while t + H < n:
                if not np.isfinite(dacc[t]):
                    t += 1; continue
                if dacc[t] >= hi:
                    allr.append(-fwd_ret(c, t, H) - FEE); t += H
                elif dacc[t] <= lo:
                    allr.append(fwd_ret(c, t, H) - FEE); t += H
                else:
                    t += 1
        r = np.array(allr)
        if len(r) < 10: continue
        ci = boot_ci(r)
        fl = "  <<<+" if ci[0] > 0 else ""
        print(f"{H:>4}{len(r):>6}{r.mean()*100:>9.2f}{tstat(r):>7.2f}"
              f"{(r>0).mean()*100:>6.0f}   [{ci[0]*100:+.2f}%,{ci[1]*100:+.2f}%]{fl}")


if __name__ == "__main__":
    data = load()
    print("Geladen:", {s: len(data[s]) for s in data})
    extreme_sweep(data)
    asymmetry(data)
    spike(data)
    block("LESART")
    print("Grünes Licht für Liquidations-/OI-Sammlung NUR wenn der extreme Rand das")
    print("Signal SCHÄRFT (t steigt mit Perzentil) oder ein Schenkel/Spike klar CI>0.")
    print("Bleibt alles flach -> der Fingerabdruck ist nicht handelbar, ehrliche Absage.")
