"""
config.py – Lädt und verwaltet die Bot-Konfiguration.
Unterstützt --config Argument und BOT_CONFIG Umgebungsvariable für Multi-Bot-Betrieb.
"""

import copy
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Config-Pfad: Reihenfolge: BOT_CONFIG env > CONFIG_PATH env > Default
# Fallback bots/bot1.json (gültige Kraken-Config) – NICHT die alte Binance-Spec
# ki_trading_bot_v4_config.json (die ist nur noch Referenz-Dokument).
_config_path_str = (
    os.environ.get("BOT_CONFIG") or
    os.environ.get("CONFIG_PATH") or
    "bots/bot1.json"
)
CONFIG_PATH = Path(_config_path_str)

BOT_ID = int(os.environ.get("BOT_ID", "0"))

DEFAULT_CONFIG = {
    "bot_id": BOT_ID,
    "symbols": ["PF_XBTUSD", "PF_ETHUSD"],
    "strategy": "momentum",       # momentum | mean_reversion | breakout | contrarian | scalper
    "macro_mode": "filter",       # filter | both | invert
    # Fix 4: Entry nur wenn 4h-Regime die Richtung stützt. AKTIV (Walk-Forward 2026-06-13:
    # half 5/5 Symbolen, +50% Δ PnL). mean_reversion automatisch ausgenommen. Reversibel: False.
    "require_4h_regime_confirmation": True,
    "leverage": 3,
    "trading_mode": os.environ.get("TRADING_MODE", "paper"),

    "risk": {
        "risk_per_trade": 0.01,         # Anteil des Kapitals, der pro Trade riskiert wird
        "max_position_size_pct": 0.10,  # Notional-Cap (× Hebel) als Sicherheitsgrenze
        "daily_loss_limit_pct": 0.03,
        "max_hold_hours": 48,
        "sl_atr_multiplier": 1.5,
        "tp_atr_multiplier": 2.0,
        "max_atr_ratio": 3.0,
        "max_funding_rate": 0.0005,
        "max_consecutive_negative_weeks": 3,
        "fee_taker": 0.0005,      # 0.05% Taker-Gebühr
        "fee_slippage": 0.0002,   # 0.02% Slippage-Puffer
    },

    "scoring": {
        "min_score_long": 5,
        "min_score_short": -5,
    },

    "indicators": {
        "adx_period": 14,
        "adx_trend_threshold": 25,
        "adx_chop_threshold": 18,  # ADX < 18 = Chop
        "atr_period": 14,
    },

    "ml": {
        "veto_threshold": 0.42,       # P(win) < Schwelle → Signal verworfen
        "min_samples_symbol": 150,    # Min Samples für Symbol-Modell (Session-2-Fix)
        "min_samples_base": 200,      # Min Samples für Basis-Modell (Session-2-Fix)
        "retrain_every_n": 50,        # Retrain alle N neuen Outcomes
        "weight_real": 1.0,
        "weight_shadow": 0.5,
        "weight_synthetic": 0.2,
        # B1 — Recency-Gewichtung: ältere Outcomes zählen weniger.
        # Halbwertszeit in Tagen; 0 oder negativ = deaktiviert (kein Decay).
        "weight_half_life_days": 45,
        # B2 — Symbol-balanciertes Sampling: max. Zeilen pro Symbol fürs Win-Training.
        # Verhindert Symbol-Schieflage des alten globalen LIMIT 20000.
        "samples_per_symbol": 8000,
        # A1 — Exploration: ein kleiner Teil knapp-vetoeter High-Score-Signale wird
        # trotzdem (Paper-)gehandelt, um echte Labels in der Veto-Grauzone zu sammeln.
        "exploration_enabled": True,
        "exploration_rate": 0.10,     # Wahrscheinlichkeit, ein Grauzonen-Veto zu überstimmen
        "exploration_band": 0.10,     # nur wenn P(win) >= veto_threshold - band ("knapp daneben")
        "exploration_min_score": 0,   # nur Signale mit |score| >= diesem Wert (0 = aus)
    },

    "pbt_mutable": False,             # Darf der PBT-Selektor diese Config ändern?

    "data": {
        "macro_stale_hours": 26,
        "kline_limit": 200,
    },

    "paths": {
        "state_file": "data/bot_state.json",
        "heartbeat_file": "data/heartbeat.json",
        "db_file": "data/trades.db",
        "log_dir": "logs",
        "network_db": "data/network.db",
        "models_dir": "data/models",
    },
}


class Config:
    """Konfigurationsklasse für eine Bot-Instanz."""

    def __init__(self):
        self._data = copy.deepcopy(DEFAULT_CONFIG)
        self._load_from_file()
        self._normalize_bot_config()
        self._override_from_env()
        self._ensure_bot_paths()

    def _load_from_file(self):
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                self._deep_merge(self._data, file_config)
                logger.info(f"Config geladen: {CONFIG_PATH}")
            except Exception as e:
                logger.warning(f"Config-Datei Fehler: {e} – Verwende Defaults")
        else:
            logger.info(f"Keine Config-Datei ({CONFIG_PATH}) – Verwende Defaults")

    def _normalize_bot_config(self):
        """
        Übersetzt das FLACHE Bot-Config-Schema (symbol, min_score_long,
        atr_sl_multiplier, risk_per_trade, ...) in die VERSCHACHTELTE Struktur,
        die der restliche Code liest (symbols, scoring.*, risk.*, indicators.*).

        Die flachen Keys sind die Quelle der Wahrheit – PBT (brain.py) und
        Learning Factory schreiben/lesen sie. Dieser Layer ist die EINZIGE
        Stelle, an der übersetzt wird. Ohne ihn wurden alle Bot-spezifischen
        Werte still ignoriert (alle Bots liefen mit Default-Symbolen + Defaults).
        """
        d = self._data

        # Symbol (Singular) → symbols-Liste. Jeder Bot handelt GENAU sein Symbol.
        if isinstance(d.get("symbol"), str):
            d["symbols"] = [d["symbol"]]

        # Scoring-Schwellen (oberste Ebene → scoring-Block)
        d.setdefault("scoring", {})
        if "min_score_long" in d:
            d["scoring"]["min_score_long"] = d["min_score_long"]
        if "min_score_short" in d:
            # Bot-Configs speichern die Short-Schwelle positiv (z.B. 6.0);
            # die Scoring-Logik braucht sie negativ (Short bei score <= -6).
            d["scoring"]["min_score_short"] = -abs(d["min_score_short"])

        # Risk-Parameter (oberste Ebene → risk-Block)
        d.setdefault("risk", {})
        if "risk_per_trade" in d:
            d["risk"]["risk_per_trade"] = d["risk_per_trade"]
        if "atr_sl_multiplier" in d:
            d["risk"]["sl_atr_multiplier"] = d["atr_sl_multiplier"]
        if "atr_tp_multiplier" in d:
            d["risk"]["tp_atr_multiplier"] = d["atr_tp_multiplier"]
        if "funding_rate_limit" in d:
            d["risk"]["max_funding_rate"] = d["funding_rate_limit"]
        # Hebel konsistent halten (config.leverage UND risk.leverage lesen ihn)
        if "leverage" in d:
            d["risk"]["leverage"] = d["leverage"]

        # ADX-Chop-Schwelle (oberste Ebene → indicators-Block)
        d.setdefault("indicators", {})
        if "adx_chop_threshold" in d:
            d["indicators"]["adx_chop_threshold"] = d["adx_chop_threshold"]

    def _override_from_env(self):
        if os.environ.get("TRADING_MODE"):
            self._data["trading_mode"] = os.environ["TRADING_MODE"]
        if os.environ.get("BOT_ID"):
            self._data["bot_id"] = int(os.environ["BOT_ID"])

    def _ensure_bot_paths(self):
        """Leitet bot-spezifische Pfade aus bot_id ab wenn nicht explizit gesetzt."""
        bot_id = self._data.get("bot_id", 0)
        if bot_id and bot_id > 0:
            paths = self._data["paths"]
            # Nur setzen wenn noch Default-Pfade (keine bot-spezifischen)
            paths["state_file"] = f"data/bot{bot_id}/bot_state.json"
            paths["heartbeat_file"] = f"data/bot{bot_id}/heartbeat.json"
            paths["db_file"] = f"data/bot{bot_id}/trades.db"
            paths["log_dir"] = f"logs/bot{bot_id}"

    def _deep_merge(self, base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default=None):
        keys = key.split(".")
        data = self._data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default
        return data

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    @property
    def bot_id(self) -> int:
        return self._data.get("bot_id", 0)

    @property
    def symbols(self) -> list:
        return self._data["symbols"]

    @property
    def strategy(self) -> str:
        return self._data.get("strategy", "momentum")

    @property
    def macro_mode(self) -> str:
        return self._data.get("macro_mode", "filter")

    @property
    def require_4h_regime_confirmation(self) -> bool:
        return bool(self._data.get("require_4h_regime_confirmation", False))

    @property
    def leverage(self) -> int:
        return self._data["leverage"]

    @property
    def trading_mode(self) -> str:
        return self._data["trading_mode"]

    @property
    def is_paper(self) -> bool:
        return self.trading_mode.lower() == "paper"

    @property
    def risk(self) -> dict:
        return self._data["risk"]

    @property
    def scoring(self) -> dict:
        return self._data["scoring"]

    @property
    def paths(self) -> dict:
        return self._data["paths"]

    @property
    def data_settings(self) -> dict:
        return self._data["data"]

    @property
    def indicators(self) -> dict:
        return self._data["indicators"]

    @property
    def ml(self) -> dict:
        return self._data.get("ml", {})

    @property
    def pbt_mutable(self) -> bool:
        return self._data.get("pbt_mutable", False)


config = Config()
