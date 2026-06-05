"""
layers/layer1_macro.py – Makro-Filter.
Wertet übergeordnete Marktbedingungen aus (SPX, DXY, BTC-Dominanz).
Stale-Data-Check: wenn Daten älter als 26h → direction = "both" (neutral).
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Tuple

import pandas as pd

from config import config
from state import state
from notifier import notifier

logger = logging.getLogger(__name__)

# Makro-Assets für yfinance
SPX_TICKER = "^GSPC"       # S&P 500
DXY_TICKER = "DX-Y.NYB"    # Dollar Index
BTC_TICKER = "BTC-USD"     # BTC/USD (für Trendrichtung)

# Stale-Data-Schwelle
STALE_HOURS = config.data_settings.get("macro_stale_hours", 26)


async def calculate_layer1_macro() -> Tuple[str, float]:
    """
    Berechnet die Makro-Richtung.
    
    Returns:
        Tuple[direction, confidence]
        direction: "long", "short", "both" (neutral)
        confidence: 0.0 - 1.0
    """
    try:
        import yfinance as yf

        logger.debug("Makro-Analyse wird berechnet...")

        # Daten laden (letzte 30 Tage für MA-Berechnung)
        spx = await _fetch_ticker_data(yf, SPX_TICKER, period="30d")
        dxy = await _fetch_ticker_data(yf, DXY_TICKER, period="30d")
        btc = await _fetch_ticker_data(yf, BTC_TICKER, period="30d")

        if spx is None or dxy is None or btc is None:
            logger.warning("Makro-Daten konnten nicht geladen werden – neutral (both)")
            return "both", 0.0

        # Scoring
        score = 0
        max_score = 3

        # 1. SPX Trend (über 20-Tage-MA = bullish für Krypto)
        if len(spx) >= 20:
            spx_ma20 = spx["Close"].rolling(20).mean().iloc[-1]
            spx_close = spx["Close"].iloc[-1]
            if spx_close > spx_ma20:
                score += 1
                logger.debug(f"SPX bullish: {spx_close:.1f} > MA20 {spx_ma20:.1f}")
            else:
                score -= 1
                logger.debug(f"SPX bearish: {spx_close:.1f} < MA20 {spx_ma20:.1f}")

        # 2. DXY Trend (fallender Dollar = bullish für Krypto)
        if len(dxy) >= 20:
            dxy_ma20 = dxy["Close"].rolling(20).mean().iloc[-1]
            dxy_close = dxy["Close"].iloc[-1]
            if dxy_close < dxy_ma20:
                score += 1
                logger.debug(f"DXY bullish für Krypto: {dxy_close:.2f} < MA20 {dxy_ma20:.2f}")
            else:
                score -= 1
                logger.debug(f"DXY bearish für Krypto: {dxy_close:.2f} > MA20 {dxy_ma20:.2f}")

        # 3. BTC-Trend (über 50-Tage-MA = bullish)
        if len(btc) >= 20:
            btc_ma20 = btc["Close"].rolling(20).mean().iloc[-1]
            btc_close = btc["Close"].iloc[-1]
            if btc_close > btc_ma20:
                score += 1
                logger.debug(f"BTC bullish: {btc_close:.0f} > MA20 {btc_ma20:.0f}")
            else:
                score -= 1
                logger.debug(f"BTC bearish: {btc_close:.0f} < MA20 {btc_ma20:.0f}")

        # Richtung bestimmen
        confidence = abs(score) / max_score

        if score >= 2:
            direction = "long"
        elif score <= -2:
            direction = "short"
        else:
            direction = "both"  # Neutral – beide Richtungen erlaubt

        logger.info(f"Makro-Analyse: Score={score}/{max_score}, "
                    f"Richtung={direction}, Confidence={confidence:.2f}")

        # State aktualisieren
        state.last_macro_direction = direction
        state.last_macro_update_utc = datetime.now(timezone.utc).isoformat()
        state.write_on_event("macro_update")

        return direction, confidence

    except Exception as e:
        logger.error(f"Fehler bei Makro-Analyse: {e}")
        return "both", 0.0


async def _fetch_ticker_data(yf, ticker: str, period: str = "30d"):
    """
    Lädt Ticker-Daten via yfinance.
    Gibt None zurück bei Fehler.
    """
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        # yfinance ist synchron – im Executor ausführen
        data = await loop.run_in_executor(
            None,
            lambda: yf.download(ticker, period=period, progress=False, auto_adjust=True)
        )
        if data is None or len(data) == 0:
            logger.warning(f"Keine Daten für {ticker}")
            return None
        # yfinance 1.x gibt MultiIndex-Spalten zurück – auf eine Ebene reduzieren
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        logger.error(f"Fehler beim Laden von {ticker}: {e}")
        return None


def get_cached_direction() -> str:
    """
    Gibt die gecachte Makro-Richtung zurück.
    Prüft auf Stale Data (> 26h alt) → gibt "both" zurück.
    """
    if state.last_macro_direction is None:
        return "both"

    if state.last_macro_update_utc is None:
        return "both"

    try:
        last_update = datetime.fromisoformat(
            state.last_macro_update_utc.replace("Z", "+00:00")
        )
        if last_update.tzinfo is None:
            last_update = last_update.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - last_update
        if age > timedelta(hours=STALE_HOURS):
            logger.warning(
                f"Makro-Daten veraltet: {age.total_seconds() / 3600:.1f}h > {STALE_HOURS}h "
                f"– Verwende neutral (both)"
            )
            notifier.send_warning(
                f"⚠️ Makro-Daten veraltet ({age.total_seconds() / 3600:.1f}h alt)!\n"
                f"Verwende neutrale Richtung (both)"
            )
            return "both"

        return state.last_macro_direction

    except Exception as e:
        logger.error(f"Fehler beim Prüfen der Makro-Daten: {e}")
        return "both"
