"""
layers/layer3_scoring.py – 15-Minuten Scoring-System.
Kombiniert technische Indikatoren, On-Chain-Daten und Sentiment.

WICHTIG: Layer 2 (Regime) wird hier NICHT aufgerufen – nur gecachtes Regime verwenden!
BUG-FIX: Fear & Greed Index > 85 Bug korrigiert (score -= 1 statt +=1).
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from config import config
from layers.layer2_regime import get_cached_regime

logger = logging.getLogger(__name__)

# Scoring-Schwellwerte
MIN_SCORE_LONG = config.scoring.get("min_score_long", 3)
MIN_SCORE_SHORT = config.scoring.get("min_score_short", -3)

# ATR-Periode
ATR_PERIOD = config.indicators.get("atr_period", 14)


@dataclass
class ScoringResult:
    """Ergebnis des Scoring-Zyklus."""
    symbol: str
    score: int
    direction: Optional[str]   # "long", "short", None
    signal: bool                # True wenn Score stark genug
    atr: float
    atr_ratio: float
    regime: str
    funding_rate: float
    fg_index: float
    details: dict


def calculate_layer3_score(
    symbol: str,
    klines_15m: list,
    funding_rate: float,
    fg_index: float = 50.0,
) -> ScoringResult:
    """
    Berechnet den 15-Minuten Score für ein Symbol.
    
    Score-Komponenten:
    - RSI: Überkauft/Überverkauft
    - MACD: Momentum
    - Bollinger Bands: Position im Band
    - Fear & Greed Index: Sentiment
    - Regime: Trend-Bestätigung
    
    Args:
        symbol: Trading-Symbol
        klines_15m: 15m-Klines (Binance-Format)
        funding_rate: Aktuelle Funding-Rate
        fg_index: Fear & Greed Index (0-100)
    
    Returns:
        ScoringResult
    """
    try:
        if len(klines_15m) < 50:
            logger.warning(f"Zu wenig Klines für Scoring: {len(klines_15m)}")
            return _empty_result(symbol)

        # DataFrame erstellen
        df = _klines_to_dataframe(klines_15m)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Indikatoren berechnen
        rsi = _calculate_rsi(close, 14)
        macd, macd_signal = _calculate_macd(close)
        bb_upper, bb_mid, bb_lower = _calculate_bollinger(close, 20, 2.0)
        atr = _calculate_atr(high, low, close, ATR_PERIOD)

        # ATR-Ratio (aktueller ATR / 50-Perioden-Durchschnitt)
        atr_avg = atr.rolling(50).mean().iloc[-1] if len(atr) >= 50 else atr.mean()
        atr_ratio = float(atr.iloc[-1] / atr_avg) if atr_avg > 0 else 1.0
        current_atr = float(atr.iloc[-1])

        # Aktuelle Werte
        current_close = float(close.iloc[-1])
        current_rsi = float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0
        current_macd = float(macd.iloc[-1]) if len(macd) > 0 else 0.0
        current_macd_signal = float(macd_signal.iloc[-1]) if len(macd_signal) > 0 else 0.0
        current_bb_upper = float(bb_upper.iloc[-1])
        current_bb_lower = float(bb_lower.iloc[-1])

        # Gecachtes Regime (NICHT Layer 2 aufrufen!)
        regime = get_cached_regime()

        # ─── Score berechnen ────────────────────────────────────────────────

        score = 0
        details = {}

        # Bestätigungs-Flags für Fear & Greed
        confirmed_long = False
        confirmed_short = False

        # 1. RSI (Überkauft/Überverkauft + Momentum)
        if current_rsi < 30:
            score += 2
            details["rsi"] = f"+2 (überverkauft: {current_rsi:.1f})"
            confirmed_long = True
        elif current_rsi < 40:
            score += 1
            details["rsi"] = f"+1 (schwach überverkauft: {current_rsi:.1f})"
        elif current_rsi > 70:
            score -= 2
            details["rsi"] = f"-2 (überkauft: {current_rsi:.1f})"
            confirmed_short = True
        elif current_rsi > 60:
            score -= 1
            details["rsi"] = f"-1 (schwach überkauft: {current_rsi:.1f})"
        else:
            details["rsi"] = f"0 (neutral: {current_rsi:.1f})"

        # 2. MACD Crossover
        prev_macd = float(macd.iloc[-2]) if len(macd) > 1 else current_macd
        prev_signal = float(macd_signal.iloc[-2]) if len(macd_signal) > 1 else current_macd_signal

        if current_macd > current_macd_signal and prev_macd <= prev_signal:
            # Bullisher Crossover
            score += 2
            details["macd"] = f"+2 (bullish Crossover)"
            confirmed_long = True
        elif current_macd < current_macd_signal and prev_macd >= prev_signal:
            # Bearisher Crossover
            score -= 2
            details["macd"] = f"-2 (bearish Crossover)"
            confirmed_short = True
        elif current_macd > current_macd_signal:
            score += 1
            details["macd"] = f"+1 (bullish)"
        elif current_macd < current_macd_signal:
            score -= 1
            details["macd"] = f"-1 (bearish)"
        else:
            details["macd"] = "0 (neutral)"

        # 3. Bollinger Bands Position
        bb_range = current_bb_upper - current_bb_lower
        if bb_range > 0:
            bb_position = (current_close - current_bb_lower) / bb_range

            if bb_position < 0.05:
                score += 2
                details["bollinger"] = f"+2 (unteres Band: {bb_position:.2f})"
                confirmed_long = True
            elif bb_position < 0.25:
                score += 1
                details["bollinger"] = f"+1 (untere Zone)"
            elif bb_position > 0.95:
                score -= 2
                details["bollinger"] = f"-2 (oberes Band: {bb_position:.2f})"
                confirmed_short = True
            elif bb_position > 0.75:
                score -= 1
                details["bollinger"] = f"-1 (obere Zone)"
            else:
                details["bollinger"] = f"0 (Mitte: {bb_position:.2f})"

        # 4. Regime-Bestätigung
        if regime == "trending_up":
            score += 1
            details["regime"] = "+1 (trending_up)"
        elif regime == "trending_down":
            score -= 1
            details["regime"] = "-1 (trending_down)"
        else:
            details["regime"] = "0 (ranging)"

        # 5. Fear & Greed Index – KORRIGIERTER CODE (BUG-FIX)
        # FALSCH (alter Code): score -= 3 if confirmed_short else -1  → -(-1) = +1 wenn nicht bestätigt!
        # RICHTIG:
        if fg_index < 20:
            # Extreme Fear = potenzieller Long-Einstieg
            score += 3 if confirmed_long else 1
            details["fg"] = f"+{3 if confirmed_long else 1} (extreme fear: {fg_index:.0f})"
        elif fg_index < 35:
            score += 1
            details["fg"] = f"+1 (fear: {fg_index:.0f})"
        elif fg_index > 85:
            # Extreme Greed = potenzieller Short-Einstieg
            # BUG-FIX: score -= 1 (nicht += 1) wenn nicht bestätigt
            score -= 3 if confirmed_short else 1   # KORRIGIERT: -1 wenn nicht bestätigt
            details["fg"] = f"-{3 if confirmed_short else 1} (extreme greed: {fg_index:.0f})"
        elif fg_index > 65:
            score -= 1
            details["fg"] = f"-1 (greed: {fg_index:.0f})"
        else:
            details["fg"] = f"0 (neutral: {fg_index:.0f})"

        # ─── Richtung bestimmen ─────────────────────────────────────────────

        direction = None
        signal = False

        if score >= MIN_SCORE_LONG:
            direction = "long"
            signal = True
        elif score <= MIN_SCORE_SHORT:
            direction = "short"
            signal = True

        logger.info(f"Scoring [{symbol}]: Score={score}, Richtung={direction}, "
                    f"Signal={'JA' if signal else 'NEIN'}, "
                    f"ATR={current_atr:.4f}, Regime={regime}")
        logger.debug(f"Score-Details: {details}")

        return ScoringResult(
            symbol=symbol,
            score=score,
            direction=direction,
            signal=signal,
            atr=current_atr,
            atr_ratio=atr_ratio,
            regime=regime,
            funding_rate=funding_rate,
            fg_index=fg_index,
            details=details,
        )

    except Exception as e:
        logger.error(f"Fehler beim Scoring für {symbol}: {e}")
        return _empty_result(symbol)


def _empty_result(symbol: str) -> ScoringResult:
    """Leeres Scoring-Ergebnis bei Fehler."""
    return ScoringResult(
        symbol=symbol, score=0, direction=None, signal=False,
        atr=0.0, atr_ratio=1.0, regime="ranging",
        funding_rate=0.0, fg_index=50.0, details={},
    )


def _klines_to_dataframe(klines: list) -> pd.DataFrame:
    """Konvertiert Binance-Klines in einen DataFrame."""
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI berechnen."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calculate_macd(close: pd.Series,
                    fast: int = 12, slow: int = 26,
                    signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """MACD und Signal-Linie berechnen."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def _calculate_bollinger(close: pd.Series, period: int = 20,
                         std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands berechnen."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def _calculate_atr(high: pd.Series, low: pd.Series,
                   close: pd.Series, period: int = 14) -> pd.Series:
    """ATR (Average True Range) berechnen."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


async def fetch_fear_greed_index() -> float:
    """
    Holt den Fear & Greed Index von der öffentlichen API.
    Gibt 50.0 (neutral) zurück bei Fehler.
    """
    try:
        import aiohttp
        url = "https://api.alternative.me/fng/?limit=1"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    value = float(data["data"][0]["value"])
                    logger.debug(f"Fear & Greed Index: {value}")
                    return value
    except Exception as e:
        logger.warning(f"Fear & Greed Index nicht abrufbar: {e}")
    return 50.0
