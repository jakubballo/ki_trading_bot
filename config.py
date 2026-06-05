"""
config.py – Lädt und verwaltet die Bot-Konfiguration aus ki_trading_bot_v4_config.json
"""

import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# .env laden
load_dotenv()

logger = logging.getLogger(__name__)

# Pfad zur Konfigurationsdatei
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "ki_trading_bot_v4_config.json"))

# Standard-Konfiguration falls keine Datei vorhanden
DEFAULT_CONFIG = {
    "symbols": ["BTCUSDT"],
    "leverage": 3,
    "margin_type": "ISOLATED",
    "trading_mode": os.environ.get("TRADING_MODE", "paper"),

    # Risiko-Parameter
    "risk": {
        "max_position_size_pct": 0.10,       # Max 10% des Kapitals pro Trade
        "daily_loss_limit_pct": 0.03,         # 3% täglicher Verlust-Stop
        "max_hold_hours": 48,                  # Max Haltedauer in Stunden
        "sl_atr_multiplier": 2.0,             # SL = Einstieg ± 2x ATR
        "tp_atr_multiplier": 3.0,             # TP = Einstieg ± 3x ATR
        "max_atr_ratio": 3.0,                  # Extreme Volatilität: ATR > 3x Durchschnitt
        "max_funding_rate": 0.0005,            # Max Funding Rate für Entry
        "max_consecutive_negative_weeks": 3,   # Weekly Stop nach 3 Verlustwochen
    },

    # Scoring-Schwellwerte
    "scoring": {
        "min_score_long": 3,
        "min_score_short": -3,
    },

    # Technische Indikatoren
    "indicators": {
        "adx_period": 14,
        "adx_trend_threshold": 25,            # ADX > 25 = Trending
        "atr_period": 14,
    },

    # Daten-Einstellungen
    "data": {
        "macro_stale_hours": 26,              # Makro-Daten veraltet nach 26h
        "kline_limit": 200,                    # Anzahl Kerzen für Indikatoren
    },

    # Datei-Pfade
    "paths": {
        "state_file": "data/bot_state.json",
        "heartbeat_file": "data/heartbeat.json",
        "db_file": "data/trades.db",
        "log_dir": "logs",
    },
}


class Config:
    """Konfigurationsklasse – lädt JSON-Config und merged mit Defaults."""

    def __init__(self):
        self._data = DEFAULT_CONFIG.copy()
        self._load_from_file()
        self._override_from_env()

    def _load_from_file(self):
        """Lädt Konfiguration aus JSON-Datei, falls vorhanden."""
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                self._deep_merge(self._data, file_config)
                logger.info(f"Konfiguration geladen aus: {CONFIG_PATH}")
            except Exception as e:
                logger.warning(f"Fehler beim Laden der Config-Datei: {e} – Verwende Defaults")
        else:
            logger.info(f"Keine Config-Datei gefunden ({CONFIG_PATH}) – Verwende Defaults")

    def _override_from_env(self):
        """Überschreibt kritische Werte aus Umgebungsvariablen."""
        trading_mode = os.environ.get("TRADING_MODE")
        if trading_mode:
            self._data["trading_mode"] = trading_mode

    def _deep_merge(self, base: dict, override: dict):
        """Merged override rekursiv in base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default=None):
        """Holt einen Konfig-Wert per Punkt-Notation (z.B. 'risk.daily_loss_limit_pct')."""
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

    # Häufig verwendete Properties
    @property
    def symbols(self) -> list:
        return self._data["symbols"]

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


# Globale Config-Instanz
config = Config()
