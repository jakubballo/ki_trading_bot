"""
layers/layer2_regime.py – ADX-basierte Markt-Regime-Erkennung.
WICHTIG: Wird NUR alle 4h aufgerufen (nicht im 15min-Zyklus!).
Erkennt: trending_up, trending_down, ranging.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import config
from state import state

logger = logging.getLogger(__name__)

# ADX-Schwellwert für Trend-Erkennung
ADX_TREND_THRESHOLD = config.indicators.get("adx_trend_threshold", 25)
ADX_PERIOD = config.indicators.get("adx_period", 14)


def calculate_layer2_regime(symbol: str, klines: list) -> Tuple[str, float]:
    """
    Berechnet das aktuelle Markt-Regime anhand von 4h-Klines und ADX.
    
    WICHTIG: Diese Funktion darf NICHT im 15min-Scoring-Zyklus aufgerufen werden!
    Sie wird nur alle 4h durch den separaten Scheduler-Job ausgeführt.
    
    Args:
        symbol: Trading-Symbol
        klines: Liste von 4h-Klines (Binance-Format)
    
    Returns:
        Tuple[regime, adx_value]
        regime: "trending_up", "trending_down", "ranging"
        adx_value: Aktueller ADX-Wert
    """
    try:
        if len(klines) < ADX_PERIOD + 5:
            logger.warning(f"Zu wenig Klines für ADX-Berechnung: {len(klines)}")
            return "ranging", 0.0

        # Klines in DataFrame umwandeln (Binance-Format)
        df = _klines_to_dataframe(klines)

        # ADX berechnen
        adx, plus_di, minus_di = _calculate_adx(df, ADX_PERIOD)

        if len(adx) == 0:
            return "ranging", 0.0

        current_adx = float(adx.iloc[-1])
        current_plus_di = float(plus_di.iloc[-1])
        current_minus_di = float(minus_di.iloc[-1])

        # Regime bestimmen
        if current_adx > ADX_TREND_THRESHOLD:
            if current_plus_di > current_minus_di:
                regime = "trending_up"
            else:
                regime = "trending_down"
        else:
            regime = "ranging"

        logger.info(f"Regime [{symbol}]: {regime} "
                    f"(ADX={current_adx:.1f}, +DI={current_plus_di:.1f}, -DI={current_minus_di:.1f})")

        # State aktualisieren
        if state.last_regime != regime:
            logger.info(f"Regime-Wechsel: {state.last_regime} → {regime}")
            state.last_regime = regime
            state.write_on_event("regime_change")

        return regime, current_adx

    except Exception as e:
        logger.error(f"Fehler bei Regime-Berechnung: {e}")
        return "ranging", 0.0


def _klines_to_dataframe(klines: list) -> pd.DataFrame:
    """Konvertiert Binance-Klines in einen DataFrame."""
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["high", "low", "close"])
    return df.reset_index(drop=True)


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Berechnet ADX, +DI und -DI manuell.
    
    Returns:
        Tuple[adx, plus_di, minus_di] als Pandas Series
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )

    # Wilder's Smoothing
    atr = _wilders_smooth(tr, period)
    plus_di = 100 * _wilders_smooth(plus_dm, period) / atr
    minus_di = 100 * _wilders_smooth(minus_dm, period) / atr

    # DX und ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    dx = dx.fillna(0)
    adx = _wilders_smooth(dx, period)

    return adx, plus_di, minus_di


def _wilders_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Smoothing (für ADX-Berechnung)."""
    result = series.copy().astype(float)
    result.iloc[:period] = np.nan

    # Erste Berechnung: einfacher Durchschnitt
    if len(series) >= period:
        result.iloc[period - 1] = series.iloc[:period].mean()

        # Wilder's Smoothing Formel
        for i in range(period, len(series)):
            result.iloc[i] = (result.iloc[i - 1] * (period - 1) + series.iloc[i]) / period

    return result


def get_cached_regime() -> str:
    """Gibt das gecachte Regime zurück (oder 'ranging' als Default)."""
    return state.last_regime or "ranging"
