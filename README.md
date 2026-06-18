# Kraken-Futures Bot-Netzwerk — ein ehrliches Krypto-Edge-Forschungsprojekt

Ein Netzwerk aus 50 parallelen Paper-Trading-Bots auf Kraken Futures, gebaut um
systematisch zu prüfen, ob sich auf retail-zugänglichen Marktdaten ein
**handelbarer Edge** finden lässt — mit ML-Veto-Schicht, Walk-Forward-Validierung
und 24/7-Betrieb auf einem VPS.

> ### TL;DR — der ehrliche Kernbefund
> Nach einer systematischen Suche über alle zugänglichen Preis- und Funding-Signale
> (15min bis Daily, Time-Series und Cross-Sectional über bis zu 308 Instrumente,
> Funding-Carry und Funding-Crowding) gilt: **Es wurde kein statistisch belastbarer,
> nach Kosten handelbarer Edge gefunden.** Das ist kein Bug — es ist das erwartbare
> Ergebnis für die liquidesten, meist-arbitragierten Instrumente. Das einzige Signal
> mit einem konsistenten, theoriekonformen Fingerabdruck (Funding-Crowding) blieb
> unter der Signifikanzschwelle. Das Projekt wurde nach diesem Befund **bewusst
> gestoppt** — ohne ein System auf ein unbewiesenes Signal zu bauen.
>
> **Reines Paper-Trading. Kein echtes Kapital wurde je riskiert.**

---

## Warum dieses Repo trotzdem nützlich ist

Die meisten öffentlichen „Trading-Bot"-Repos behaupten zu funktionieren und zeigen
überangepasste Backtests. Dieses Repo dokumentiert das Gegenteil ehrlich: eine
methodisch saubere Edge-Suche, die zu einem **negativen Ergebnis** kam — inklusive
der reproduzierbaren Skripte, die jeden Befund belegen. Ein dokumentiertes ehrliches
Negativ-Ergebnis ist wertvoller als ein geschöntes Positiv-Ergebnis.

---

## Was das System tut (Architektur)

50 unabhängige Bot-Prozesse (5 Symbole × 5 Strategien × 2 Risiko-Varianten), die
gemeinsam über eine geteilte Datenbank lernen. Kein Docker — direkte Python-Prozesse,
orchestriert per `network_manager.py`, 24/7 auf einem Hetzner-VPS (systemd).

```
network_manager.py   Orchestrierung, Telegram-Start/-Shutdown-Report
  ├── data_hub.py    Ein Kraken-WebSocket für alle 50 Bots (Port 8770)
  ├── brain.py       Scheduler: ML-Training, PBT, Dashboard, Daten-Updates
  └── main.py × 50   Bot-Hauptloop (1 Prozess/Bot)

Signal-Pipeline pro Bot:
  Marktdaten → scoring_core.py (5 Strategien, 21 Features, Regime-Gate)
            → layer1_macro (Makro-Filter) → layer2_regime (4h-ADX)
            → layer3_scoring → ML-Veto (Modell A Richtung → Modell B Win-P)
            → risk_gate.py (Verlustlimits, Funding, Volatilität)
            → Paper-Order
```

**ML-Veto, zwei Ebenen** (`ml_network.py`):
- **Modell A** (XGBoost, 3 Klassen): Kerzen-Richtung, trainiert aus 4,2 Jahren Historie.
- **Modell B** (XGBoost, binär): Win-Wahrscheinlichkeit, lernt stündlich aus echten +
  Shadow- + synthetischen Outcomes (`network.db`).

Alle blockierten Signale werden als **Shadow-Trades** virtuell weiterverfolgt, damit
das System auch aus nicht-gehandelten Signalen lernt.

**Symbole:** PF_XBTUSD, PF_ETHUSD, PF_SOLUSD, PF_XRPUSD, PF_LINKUSD (Kraken Perpetuals).

---

## Die Edge-Forschung — vollständige Bilanz

Jede Idee wurde mit derselben Disziplin geprüft: **Out-of-Sample-Walk-Forward**
(Parameter in-sample wählen, OOS testen), **Bootstrap-Konfidenzintervalle**,
**Edge-vs-Beta-Trennung** und realistische Gebühren. Die Messlatte war stets
*Expectancy nach Kosten mit CI über 0* — niemals Accuracy oder Win-Rate, und die
Schwelle wurde nie gesenkt, um ein schwaches Signal schönzureden.

| # | Ansatz | Daten | Ergebnis | Skript |
|---|---|---|---|---|
| 1 | 5 Regel-Strategien | 15min, 5 Sym | Kein Edge — alle Strategien negativ nach Kosten | `threshold_sweep.py` |
| 2 | Time-Series-Momentum | Daily, 5 Sym | Nur Long-**Beta**, kein Timing-Alpha (Filter +0,45 %/Trade, nicht von 0 unterscheidbar) | `daily_backtest.py` |
| 3 | Cross-Sectional L/S (eng) | Daily, 5 Sym | Kein OOS-Alpha (−0,34 %/Reb, t=−0,88) | `signal_scan.py` |
| 4 | Cross-Sectional Momentum (breit) | Daily, **308 Perps** | Kein Signal — in-sample kein t>1,8, OOS negativ | `breadth_momentum.py` |
| 5 | Funding-Carry | stündl., 5 Sym | Real aber marginal (~2,5 %/J, in *einem* Quartal konzentriert); aktiv nicht handelbar (2–5 Vorzeichen-Flips/Tag) | `funding_carry.py` |
| 6 | **Funding-Crowding** (contrarian) | stündl., 5 Sym | **Schwacher, konsistent richtig gerichteter Fingerabdruck** — aber unter Signifikanz | `funding_signal.py`, `funding_crowding_deep.py` |

### Detail-Befunde

**Daily-Momentum war nur Beta.** Ein sauberer Walk-Forward ergab +1,40 %/Trade
(t=2,13), aber der Vergleich gegen einen unkonditionierten Long zeigte: ~2/3 davon
ist reine Krypto-Aufwärtsdrift. Der Momentum-*Filter* trug nur +0,45 %/Trade bei —
statistisch nicht von 0 unterscheidbar. Sein einziger realer Wert ist
Drawdown-Reduktion (Portfolio-Sharpe 0,76 vs Buy&Hold 0,71; max. Drawdown −45 %
statt −65 %) — kein Alpha, das ein System rechtfertigt.

**Breite war nicht die Lösung.** Cross-Sectional-Momentum über alle 308 handelbaren
Kraken-Perps — die dokumentierte Edge-Bedingung — zeigte selbst *in-sample und mit
Survivorship-Bias als Rückenwind* kein einziges signifikantes Setup. Das breite
Altcoin-Universum verlor über den Zeitraum sogar im Schnitt.

**Funding-Crowding war das einzige Lebenszeichen.** Die Korrelation zwischen Funding
und Forward-Rendite war bei *allen 5 Symbolen und allen Horizonten* negativ (genau wie
die Crowding-Theorie vorhersagt: überhebelte Longs → Reversal). Die Asymmetrie war
theoriekonform (die Long-Crowd-Seite trägt das Signal). Aber: kein Konfidenzintervall
räumte die Null, und der Effekt schärfte sich am extremen Rand *nicht* — in ~1 Jahr
Funding-Historie (Krakens Limit) nicht als handelbar nachweisbar.

---

## Befunde reproduzieren

```bash
cd ki_trading_bot
pip install -r requirements.txt   # pandas, numpy, scipy, xgboost, ...

# 1. Daily-Momentum: Walk-Forward, Edge-vs-Beta, Gebühren-Sensitivität
python daily_backtest.py          # Edge-Checks
python daily_backtest.py --risk   # Sharpe / Drawdown vs Buy&Hold

# 2. Cross-Sectional (5 Symbole)
python signal_scan.py             # In-Sample-Grid
python signal_scan.py --wf        # Walk-Forward OOS

# 3. Cross-Sectional auf Breite (308 Perps; lädt Daten von Krakens Public-API)
python breadth_momentum.py --fetch   # Daily-Historie cachen (~3 Min)
python breadth_momentum.py           # In-Sample-Grid
python breadth_momentum.py --wf      # Walk-Forward OOS

# 4. Funding-Carry (lädt Funding-Historie von Krakens Public-API)
python funding_carry.py --refresh

# 5. Funding-Crowding (prädiktives Signal)
python funding_signal.py             # Korrelation + TS/XS-Contrarian
python funding_crowding_deep.py      # Extrem-Rand, Asymmetrie, Spike
```

Alle Validierungs-Skripte sind **reines Offline-Backtesting** (plus lesende
Public-API-Abrufe) und berühren das Live-Trading-System nicht.

---

## Was funktioniert (Infrastruktur)

Auch wenn kein Trading-Edge gefunden wurde, ist das technische Gerüst solide und
lief monatelang stabil:

- 50 parallele Bot-Prozesse über einen geteilten WebSocket-Hub (kein Rate-Limit-Problem)
- Zwei-Ebenen-ML-Veto mit stündlichem Online-Retraining aus Live-Outcomes
- Shadow-Trade-System (Lernen aus blockierten Signalen)
- Walk-Forward-, Expectancy- und Veto-Wirkungs-Validierung als eigene Tools
- 24/7 auf Hetzner-VPS via systemd + Git-Deploy, mit Telegram-Reporting
- Vollständige Outcome-Persistenz in SQLite (WAL), übersteht Neustarts

---

## Status

**Gestoppt.** Die Edge-Suche ist abgeschlossen und ehrlich beantwortet: Auf
retail-zugänglichen Preis- und Funding-Daten über Kraken existiert kein handelbarer
Edge. Echte Edges in Krypto leben dort, wo Infrastruktur (Latenz, Co-Location),
teure Daten (on-chain, saubere Historie) oder Illiquidität ins Spiel kommen — nicht
in cleveren Formeln auf liquiden Preisdaten.

Der einzige Thread mit Substanz (Funding-Crowding, direkter messbar über
Liquidations-Daten) bliebe ein Daten-Sammel-Projekt über Monate mit unsicherem
Ausgang — bewusst nicht weiterverfolgt.

---

## Sicherheit

- `.env` enthält API-Keys und Telegram-Tokens — **niemals committen** (steht in `.gitignore`).
- Das System lief ausschließlich im **Paper-Modus**. Live-Keys waren nie aktiv.
- API-Keys (falls je live) nur mit Trade-, **nie** mit Withdraw-Recht, plus IP-Whitelist.

---

## Disclaimer

Experimentelles Forschungsprojekt. **Keine Anlageberatung.** Trading mit Hebel ist
hochriskant. Das zentrale Ergebnis dieses Projekts ist gerade, dass die hier
untersuchten Strategien **keinen** handelbaren Edge hatten — sie sollten nicht mit
echtem Kapital gehandelt werden.
