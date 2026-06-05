"""
state.py – Bot-State-Persistenz mit atomarem Schreiben und Backup-Logik.
Verwaltet den gesamten Zustand des Bots zwischen Neustarts.
"""

import json
import os
import shutil
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from config import config

logger = logging.getLogger(__name__)

# Pfade für State-Dateien
STATE_FILE = Path(config.paths["state_file"])
STATE_TMP = STATE_FILE.with_suffix(".tmp")
STATE_BACKUP = STATE_FILE.with_stem(STATE_FILE.stem + ".backup")

# Standard-State-Schema (exakt nach Spezifikation)
DEFAULT_STATE = {
    "open_position": {
        "symbol": None,
        "side": None,
        "entry_price": None,
        "qty": None,
        "sl_price": None,
        "tp_price": None,
        "sl_order_id": None,
        "tp_order_id": None,
        "entry_order_id": None,
        "entry_time_utc": None,
        "atr_at_entry": None,
        "regime_at_entry": None,
        "liquidation_price": None,
    },
    "daily": {
        "date_utc": None,
        "trade_count": 0,
        "realized_pnl_usdt": 0.0,
        "loss_pct_of_capital": 0.0,
    },
    "weekly": {
        "week_start_utc": None,
        "realized_pnl_usdt": 0.0,
        "is_negative": False,
        "consecutive_negative_weeks": 0,
    },
    "last_regime": None,
    "last_macro_direction": None,
    "last_macro_update_utc": None,
    "account_balance_usdt": 0.0,
    "balance_last_synced_utc": None,
}

# Erlaubte Events für write_on_event
VALID_EVENTS = {
    "order_placed", "order_filled", "order_cancelled",
    "sl_set", "tp_set", "position_closed",
    "daily_loss_update", "weekly_pnl_update",
    "regime_change", "macro_update",
}


class PositionState:
    """Repräsentiert den State einer offenen Position."""

    def __init__(self, data: dict):
        self.symbol: Optional[str] = data.get("symbol")
        self.side: Optional[str] = data.get("side")
        self.entry_price: Optional[float] = data.get("entry_price")
        self.qty: Optional[float] = data.get("qty")
        self.sl_price: Optional[float] = data.get("sl_price")
        self.tp_price: Optional[float] = data.get("tp_price")
        self.sl_order_id: Optional[str] = data.get("sl_order_id")
        self.tp_order_id: Optional[str] = data.get("tp_order_id")
        self.entry_order_id: Optional[str] = data.get("entry_order_id")
        self.entry_time_utc: Optional[str] = data.get("entry_time_utc")
        self.atr_at_entry: Optional[float] = data.get("atr_at_entry")
        self.regime_at_entry: Optional[str] = data.get("regime_at_entry")
        self.liquidation_price: Optional[float] = data.get("liquidation_price")

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "qty": self.qty,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "sl_order_id": self.sl_order_id,
            "tp_order_id": self.tp_order_id,
            "entry_order_id": self.entry_order_id,
            "entry_time_utc": self.entry_time_utc,
            "atr_at_entry": self.atr_at_entry,
            "regime_at_entry": self.regime_at_entry,
            "liquidation_price": self.liquidation_price,
        }

    def reset(self):
        """Setzt die Position zurück (nach Close)."""
        for key in DEFAULT_STATE["open_position"]:
            setattr(self, key, None)

    @property
    def is_open(self) -> bool:
        return self.symbol is not None


class DailyState:
    """Tägliche Statistiken."""

    def __init__(self, data: dict):
        self.date_utc: Optional[str] = data.get("date_utc")
        self.trade_count: int = data.get("trade_count", 0)
        self.realized_pnl_usdt: float = data.get("realized_pnl_usdt", 0.0)
        self.loss_pct_of_capital: float = data.get("loss_pct_of_capital", 0.0)

    def to_dict(self) -> dict:
        return {
            "date_utc": self.date_utc,
            "trade_count": self.trade_count,
            "realized_pnl_usdt": self.realized_pnl_usdt,
            "loss_pct_of_capital": self.loss_pct_of_capital,
        }

    def reset_if_new_day(self):
        """Setzt tägliche Stats zurück wenn neuer Tag."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.date_utc != today:
            logger.info(f"Neuer Tag erkannt ({today}) – Setze tägliche Stats zurück")
            self.date_utc = today
            self.trade_count = 0
            self.realized_pnl_usdt = 0.0
            self.loss_pct_of_capital = 0.0


class WeeklyState:
    """Wöchentliche Statistiken."""

    def __init__(self, data: dict):
        self.week_start_utc: Optional[str] = data.get("week_start_utc")
        self.realized_pnl_usdt: float = data.get("realized_pnl_usdt", 0.0)
        self.is_negative: bool = data.get("is_negative", False)
        self.consecutive_negative_weeks: int = data.get("consecutive_negative_weeks", 0)

    def to_dict(self) -> dict:
        return {
            "week_start_utc": self.week_start_utc,
            "realized_pnl_usdt": self.realized_pnl_usdt,
            "is_negative": self.is_negative,
            "consecutive_negative_weeks": self.consecutive_negative_weeks,
        }

    def reset_if_new_week(self):
        """Setzt wöchentliche Stats zurück wenn neue Woche (Montag)."""
        now = datetime.now(timezone.utc)
        # Montag der aktuellen Woche
        week_start = (now - __import__('datetime').timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        if self.week_start_utc != week_start:
            logger.info(f"Neue Woche erkannt ({week_start}) – Aktualisiere wöchentliche Stats")
            # Vorherige Woche bewerten
            if self.week_start_utc is not None:
                if self.realized_pnl_usdt < 0:
                    self.consecutive_negative_weeks += 1
                    self.is_negative = True
                    logger.warning(f"Verlust-Woche: {self.consecutive_negative_weeks} in Folge")
                else:
                    self.consecutive_negative_weeks = 0
                    self.is_negative = False
            self.week_start_utc = week_start
            self.realized_pnl_usdt = 0.0


class BotState:
    """
    Haupt-State-Klasse. Verwaltet den gesamten Bot-Zustand.
    Atomares Schreiben verhindert korrupte State-Dateien bei Abstürzen.
    """

    def __init__(self):
        self.open_position = PositionState(DEFAULT_STATE["open_position"].copy())
        self.daily = DailyState(DEFAULT_STATE["daily"].copy())
        self.weekly = WeeklyState(DEFAULT_STATE["weekly"].copy())
        self.last_regime: Optional[str] = None
        self.last_macro_direction: Optional[str] = None
        self.last_macro_update_utc: Optional[str] = None
        self.account_balance_usdt: float = 0.0
        self.balance_last_synced_utc: Optional[str] = None
        self._lock = asyncio.Lock()

    def load(self) -> bool:
        """
        Lädt State aus bot_state.json.
        Gibt True zurück wenn erfolgreich, False bei Fehler (dann Defaults verwenden).
        """
        # Sicherstellen dass das Verzeichnis existiert
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        if not STATE_FILE.exists():
            logger.info("Keine State-Datei gefunden – Starte mit leerem State")
            return False

        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.open_position = PositionState(data.get("open_position", DEFAULT_STATE["open_position"]))
            self.daily = DailyState(data.get("daily", DEFAULT_STATE["daily"]))
            self.weekly = WeeklyState(data.get("weekly", DEFAULT_STATE["weekly"]))
            self.last_regime = data.get("last_regime")
            self.last_macro_direction = data.get("last_macro_direction")
            self.last_macro_update_utc = data.get("last_macro_update_utc")
            self.account_balance_usdt = data.get("account_balance_usdt", 0.0)
            self.balance_last_synced_utc = data.get("balance_last_synced_utc")

            # Tages- und Wochen-Reset prüfen
            self.daily.reset_if_new_day()
            self.weekly.reset_if_new_week()

            logger.info(f"State geladen. Position offen: {self.open_position.symbol}, "
                        f"Balance: {self.account_balance_usdt:.2f} USDT")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"State-Datei korrupt: {e} – Versuche Backup zu laden")
            return self._load_backup()
        except Exception as e:
            logger.error(f"Fehler beim Laden des State: {e}")
            return self._load_backup()

    def _load_backup(self) -> bool:
        """Versucht den State aus dem Backup zu laden."""
        if STATE_BACKUP.exists():
            try:
                shutil.copy2(STATE_BACKUP, STATE_FILE)
                logger.warning("State aus Backup wiederhergestellt")
                return self.load()
            except Exception as e:
                logger.error(f"Backup-Laden fehlgeschlagen: {e}")
        return False

    def save(self) -> bool:
        """
        Speichert State atomar: erst in .tmp schreiben, dann umbenennen.
        Erstellt vorher ein Backup der bestehenden Datei.
        """
        try:
            # Verzeichnis anlegen falls nötig
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Backup der aktuellen Datei erstellen
            if STATE_FILE.exists():
                shutil.copy2(STATE_FILE, STATE_BACKUP)

            # Daten serialisieren
            data = self.to_dict()

            # Atomar schreiben: erst in .tmp, dann umbenennen
            with open(STATE_TMP, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            # Atomares Umbenennen
            os.replace(STATE_TMP, STATE_FILE)
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern des State: {e}")
            # .tmp aufräumen falls vorhanden
            if STATE_TMP.exists():
                try:
                    STATE_TMP.unlink()
                except Exception:
                    pass
            return False

    async def save_async(self) -> bool:
        """Thread-sicheres asynchrones Speichern."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.save)

    def write_on_event(self, event_name: str):
        """
        Speichert State nach wichtigen Events.
        Wird synchron aufgerufen nach: order_placed, order_filled, etc.
        """
        if event_name not in VALID_EVENTS:
            logger.warning(f"Unbekanntes Event für State-Speichern: {event_name}")
            return

        success = self.save()
        if success:
            logger.debug(f"State gespeichert nach Event: {event_name}")
        else:
            logger.error(f"State konnte nicht gespeichert werden nach Event: {event_name}")

    async def write_on_event_async(self, event_name: str):
        """Asynchrone Version von write_on_event."""
        if event_name not in VALID_EVENTS:
            logger.warning(f"Unbekanntes Event für State-Speichern: {event_name}")
            return
        await self.save_async()

    def update_balance(self, event: dict):
        """
        Aktualisiert Kontostand aus WebSocket ACCOUNT_UPDATE Event.
        """
        try:
            # Binance ACCOUNT_UPDATE Format
            balances = event.get("a", {}).get("B", [])
            for balance in balances:
                if balance.get("a") == "USDT":
                    self.account_balance_usdt = float(balance.get("wb", 0))
                    self.balance_last_synced_utc = datetime.now(timezone.utc).isoformat()
                    logger.debug(f"Balance aktualisiert: {self.account_balance_usdt:.2f} USDT")
                    break
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Balance: {e}")

    def update_daily_pnl(self, pnl_usdt: float):
        """Aktualisiert täglichen PnL und Loss-Prozent."""
        self.daily.reset_if_new_day()
        self.daily.realized_pnl_usdt += pnl_usdt
        self.daily.trade_count += 1

        if self.account_balance_usdt > 0 and pnl_usdt < 0:
            loss_this_trade = abs(pnl_usdt) / self.account_balance_usdt
            self.daily.loss_pct_of_capital += loss_this_trade

        self.write_on_event("daily_loss_update")

    def update_weekly_pnl(self, pnl_usdt: float):
        """Aktualisiert wöchentlichen PnL."""
        self.weekly.reset_if_new_week()
        self.weekly.realized_pnl_usdt += pnl_usdt
        self.weekly.is_negative = self.weekly.realized_pnl_usdt < 0
        self.write_on_event("weekly_pnl_update")

    def set_position(self, symbol: str, side: str, entry_price: float, qty: float,
                     sl_price: float, tp_price: float, entry_order_id: str,
                     atr_at_entry: float, regime_at_entry: str):
        """Setzt eine neue offene Position."""
        self.open_position.symbol = symbol
        self.open_position.side = side
        self.open_position.entry_price = entry_price
        self.open_position.qty = qty
        self.open_position.sl_price = sl_price
        self.open_position.tp_price = tp_price
        self.open_position.entry_order_id = entry_order_id
        self.open_position.entry_time_utc = datetime.now(timezone.utc).isoformat()
        self.open_position.atr_at_entry = atr_at_entry
        self.open_position.regime_at_entry = regime_at_entry
        self.write_on_event("order_filled")

    def close_position(self):
        """Schließt die aktuelle Position im State."""
        logger.info(f"Position geschlossen: {self.open_position.symbol}")
        self.open_position.reset()
        self.write_on_event("position_closed")

    def to_dict(self) -> dict:
        """Serialisiert den gesamten State zu einem Dictionary."""
        return {
            "open_position": self.open_position.to_dict(),
            "daily": self.daily.to_dict(),
            "weekly": self.weekly.to_dict(),
            "last_regime": self.last_regime,
            "last_macro_direction": self.last_macro_direction,
            "last_macro_update_utc": self.last_macro_update_utc,
            "account_balance_usdt": self.account_balance_usdt,
            "balance_last_synced_utc": self.balance_last_synced_utc,
        }


# Globale State-Instanz
state = BotState()
