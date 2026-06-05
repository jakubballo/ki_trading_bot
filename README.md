# KI-Trading-Bot – Binance Futures

Vollautomatischer Trading-Bot für Binance Futures mit 3-Layer-Analyse (Makro, Regime, Scoring), vollständigem Risikomanagement und Telegram-Benachrichtigungen.

---

## Projektstruktur

```
ki_trading_bot/
├── main.py                  # Einstiegspunkt, Scheduler, Hauptloop, Kill-Switch
├── config.py                # Lädt ki_trading_bot_v4_config.json
├── state.py                 # State-Persistenz (bot_state.json), atomares Schreiben
├── exchange.py              # Binance Futures API-Wrapper
├── websocket_manager.py     # WebSocket Market + User Data Stream
├── layers/
│   ├── layer1_macro.py      # Makro-Filter (yfinance, SPX, DXY)
│   ├── layer2_regime.py     # ADX-Regime-Erkennung (4h, alle 4h)
│   └── layer3_scoring.py    # 15min Scoring (RSI, MACD, Bollinger, F&G)
├── risk_gate.py             # 7 sequentielle Risk-Checks
├── order_manager.py         # Order-Lifecycle, SL/TP, Retry-Logik
├── position_monitor.py      # Liquidation, Haltedauer, Funding
├── watchdog.py              # Heartbeat-Datei
├── notifier.py              # Telegram-Alerts
├── logger_db.py             # SQLite Trade-Logging
├── data/                    # State, Heartbeat, SQLite-DB (wird automatisch erstellt)
└── logs/                    # Log-Dateien (wird automatisch erstellt)
```

---

## Voraussetzungen

- Python 3.11+
- Docker & Docker Compose (für Deployment)
- Binance-API-Key mit Futures-Berechtigung
- Telegram-Bot (optional aber empfohlen)

---

## Schnellstart (lokal)

```bash
# 1. Repository klonen / Dateien kopieren
cd ki_trading_bot

# 2. .env erstellen
cp .env.example .env
# .env mit echten Werten befüllen (TRADING_MODE=paper für Tests!)

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Bot starten
python main.py
```

---

## VPS-Deployment mit Docker

### 1. VPS vorbereiten (Ubuntu 22.04)

```bash
# System aktualisieren
sudo apt update && sudo apt upgrade -y

# Docker installieren
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Docker Compose installieren
sudo apt install docker-compose-plugin -y

# Neu einloggen damit Gruppe aktiv wird
exit
```

### 2. Projektdateien auf VPS kopieren

```bash
# Von lokalem Rechner auf VPS kopieren
scp -r ki_trading_bot/ user@DEINE_VPS_IP:~/

# Oder per Git
ssh user@DEINE_VPS_IP
git clone https://github.com/dein-repo/ki_trading_bot.git
```

### 3. Konfiguration einrichten

```bash
cd ki_trading_bot

# .env aus Vorlage erstellen
cp .env.example .env
nano .env
```

`.env` befüllen:
```
BINANCE_API_KEY=dein_echter_api_key
BINANCE_SECRET=dein_echter_secret
TELEGRAM_BOT_TOKEN=dein_bot_token
TELEGRAM_CHAT_ID=deine_chat_id
TRADING_MODE=paper   # ERST paper testen, dann live!
```

### 4. Docker-Container starten

```bash
# Image bauen und starten
docker compose up -d

# Logs live verfolgen
docker compose logs -f

# Status prüfen
docker compose ps
```

### 5. Bot überwachen

```bash
# Live-Logs
docker compose logs -f trading-bot

# Heartbeat prüfen
cat data/heartbeat.json

# Trades anschauen (SQLite)
sqlite3 data/trades.db "SELECT * FROM trades ORDER BY id DESC LIMIT 10;"
```

### 6. Bot stoppen/neustarten

```bash
# Stoppen
docker compose down

# Neustarten
docker compose restart trading-bot

# Kill-Switch via Telegram: /killswitch senden
```

---

## Konfiguration (ki_trading_bot_v4_config.json)

Erstelle diese Datei im Projektverzeichnis:

```json
{
  "symbols": ["BTCUSDT"],
  "leverage": 3,
  "margin_type": "ISOLATED",
  "risk": {
    "max_position_size_pct": 0.10,
    "daily_loss_limit_pct": 0.03,
    "max_hold_hours": 48,
    "sl_atr_multiplier": 2.0,
    "tp_atr_multiplier": 3.0,
    "max_atr_ratio": 3.0,
    "max_funding_rate": 0.0005,
    "max_consecutive_negative_weeks": 3
  },
  "scoring": {
    "min_score_long": 3,
    "min_score_short": -3
  }
}
```

---

## Telegram-Bot einrichten

1. Bei [@BotFather](https://t.me/BotFather) einen neuen Bot erstellen → `/newbot`
2. Den `BOT_TOKEN` kopieren
3. Bot starten und Chat-ID ermitteln: `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Werte in `.env` eintragen

### Verfügbare Befehle

- `/killswitch` – Bot sofort stoppen und alle Positionen schließen

---

## Sicherheitshinweise

- **Immer erst im Paper-Modus testen!** (`TRADING_MODE=paper`)
- API-Key nur mit Futures-Berechtigung, **KEIN Withdraw-Recht**
- IP-Whitelist für API-Key in Binance-Einstellungen aktivieren
- `.env` niemals in Git committen!
- Regelmäßige Backups der `data/` Verzeichnisses

---

## Monitoring & Troubleshooting

### Bot reagiert nicht
```bash
docker compose restart trading-bot
cat data/heartbeat.json  # Prüfen ob alive=true
```

### Datenbank-Fehler
```bash
sqlite3 data/trades.db ".tables"
sqlite3 data/trades.db "SELECT * FROM errors ORDER BY id DESC LIMIT 5;"
```

### State zurücksetzen
```bash
# Backup erstellen
cp data/bot_state.json data/bot_state_backup_$(date +%Y%m%d).json
# State-Datei löschen (Bot startet mit leerem State)
rm data/bot_state.json
docker compose restart trading-bot
```

---

## Architektur

```
WebSocket (15m Kerze) → trigger_scoring_cycle()
                              ↓
                    Layer 3 Scoring (RSI, MACD, BB, F&G)
                              ↓
                    Layer 2 Regime (gecacht, nur alle 4h)
                              ↓
                    Layer 1 Makro (gecacht, alle 12h)
                              ↓
                    Risk Gate (7 Checks)
                              ↓
                    Entry Order (LIMIT, GTC)
                              ↓
                    On Fill → SL + TP setzen (parallel)
```

---

## Disclaimer

Dieser Bot ist ein experimentelles Tool. Trading mit Hebel ist hochriskant und kann zum vollständigen Kapitalverlust führen. Verwende nur Kapital, dessen Verlust du dir leisten kannst. Keine Anlageberatung.
