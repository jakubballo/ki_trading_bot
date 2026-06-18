# Edge-Untersuchung — Ergebnis-Briefing

*Stand: 2026-06-18 · Kraken-Futures Bot-Netzwerk · reines Paper-Trading, kein echtes Kapital*

## Frage
Lässt sich auf retail-zugänglichen Kraken-Daten ein **nach Kosten handelbarer Edge**
finden, bevor wir ein produktives System darauf bauen?

## Daten
- **5 Majors** (XBT/ETH/SOL/XRP/LINK): 4,2 J × 15min-Kerzen (2022-03 → 2026-06), auf
  Daily resampled (~1547 Tagesbars/Symbol).
- **Breite**: 308 handelbare Kraken-Perps, Daily (Charts-Public-API).
- **Funding**: stündlich, ~1 J (Krakens historisches Limit).

## Methodik (bewusst streng)
- **Out-of-Sample-Walk-Forward**: Parameter *in-sample* wählen (365 T train / 90 T test,
  Embargo ≥ Haltedauer), nur OOS gewertet → kein Hindsight.
- **Bootstrap-95%-CI** + t-Stat auf gepoolten Trades.
- **Edge-vs-Beta-Trennung**: jedes Long-Signal gegen unkonditionierten Long /
  gleichgewichtetes Universum gemessen — um Alpha von reiner Marktdrift zu trennen.
- **Maß = Expectancy/Trade nach Gebühren mit CI über 0.** Nie Accuracy/Win-Rate.
  Schwelle nie gesenkt, um schwache Signale zu retten.

## Ergebnisse

| Ansatz | Daten | Kennzahl | Verdikt |
|---|---|---|---|
| 5 Regel-Strategien | 15min | alle Expectancy < 0 (WR 30–38 %, Break-even 33,3 % bei R:R 1:2) | **kein Edge** |
| Time-Series-Momentum | Daily | WF +1,40 %/Tr (t=2,13) — aber unkond. Long allein +0,95 % | **nur Beta** |
| Cross-Sectional L/S (eng) | Daily, 5 | −0,34 %/Reb, t=−0,88, CI [−1,14, +0,40] | **kein Alpha** |
| Cross-Sectional Momentum (breit) | Daily, 308 | in-sample 0 Setups t>1,8; OOS L/S −0,53 % | **kein Signal** |
| Funding-Carry | stündl., 5 | fix short-perp +2,5 %/J, aber 1 Quartal; aktiv −380 % (Flip-Kosten) | **marginal** |
| **Funding-Crowding** | stündl., 5 | corr(funding, fwd) **5/5 Symbole, alle Horizonte negativ**; XS H5 +0,29 %, t=1,77, CI [−0,02, +0,62] | **Fingerabdruck, unsignifikant** |

### Wichtigste Einzelpunkte

**Momentum = Beta, nicht Alpha.** Der „+1,40 %/Trade" sieht gut aus, aber ~2/3 ist
Krypto-Aufwärtsdrift. Der Momentum-*Filter* trägt nur +0,45 %/Trade bei — statistisch
nicht von 0 trennbar; zweite OOS-Hälfte gar nicht mehr signifikant (t=1,12). Einziger
realer Nutzen: Drawdown (Portfolio-Sharpe 0,76 vs Buy&Hold 0,71; maxDD −45 % statt
−65 %). Pro Symbol inkonsistent (hilft XBT/SOL/XRP, schadet ETH/LINK).

**Breite war nicht die Lösung.** CS-Momentum über 308 Coins — die dokumentierte
Edge-Bedingung — zeigte selbst *in-sample und mit Survivorship-Bias als Rückenwind*
kein einziges signifikantes Setup. Das gleichgewichtete Perp-Universum war selbst
negativ (das breite Altcoin-Universum verlor 2022–26 im Schnitt).

**Funding-Carry** ist real im Vorzeichen, aber ~T-Bill-Niveau, in *einem* Quartal
konzentriert (Q3 2025), und aktiv nicht handelbar: Funding flippt 2–5×/Tag, die
Flip-Kosten (~0,3 %) sind ~600× größer als das stündliche Funding (~0,0005 %) →
Vorzeichen-Folgen ergab −380 % über das Jahr.

**Funding-Crowding** ist das **einzige Lebenszeichen**: hohes Funding → schwächere
Forward-Rendite, konsistent über *alle* Symbole und *alle* Horizonte (Theorie:
überhebelte Longs → Liquidationen → Reversal). Die Asymmetrie ist theoriekonform —
die Long-Crowd-Seite (HIGH→short) trägt das Signal (+1,87 %/Tr bei H=10), die
Gegenseite ist flach. **Aber:** kein Konfidenzintervall räumt die Null (t=1,2–1,8),
und der Effekt schärft sich am *extremen* Rand (90./95. Perzentil) **nicht** → in
~1 J Funding-Historie nicht als handelbar nachweisbar.

## Schlussfolgerung
Auf liquiden Majors gibt es auf **Preis- und Funding-Daten keinen belastbaren,
handelbaren Edge** — das erwartbare Ergebnis für die meist-arbitragierten Instrumente.
Echte Edges sitzen dort, wo **Infrastruktur** (Latenz/Co-Location), **teure Daten**
(on-chain, saubere Historie) oder **Illiquidität** ins Spiel kommen — nicht in Formeln
auf liquiden Preisreihen.

**Einziger weiterverfolgenswerter Thread:** Funding-Crowding direkter messen über
**Liquidations-/OI-Daten** (z. B. Binance `forceOrder`-Stream, gratis, klein),
**forward** über Monate gesammelt. Das ist eine Daten-/Infra-Wette mit unsicherem
Ausgang, kein validierter nächster Schritt — bewusst nicht gestartet.

## Reproduzierbar
Alle Befunde sind durch eigene Skripte belegt (reines Offline-Backtesting + lesende
Public-API-Abrufe, rühren das Live-System nicht an):

| Skript | Deckt ab |
|---|---|
| `threshold_sweep.py` | 15min-Strategien über volle Historie |
| `daily_backtest.py` (`--risk`) | Daily-Momentum: WF, Edge-vs-Beta, Sharpe/Drawdown |
| `signal_scan.py` (`--wf`) | Cross-Sectional L/S, 5 Symbole |
| `breadth_momentum.py` (`--fetch`/`--wf`) | Cross-Sectional Momentum, 308 Perps |
| `funding_carry.py` (`--refresh`) | Funding-Carry |
| `funding_signal.py` | Funding als prädiktives Crowding-Signal |
| `funding_crowding_deep.py` | Extrem-Rand, Asymmetrie, Spike |
