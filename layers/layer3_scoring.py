"""
layers/layer3_scoring.py – Wrapper um scoring_core.py für den Live-Bot.
Holt gecachtes Regime und delegiert an scoring_core.score_candles().
"""

import logging
from typing import Optional

from config import config
from layers.layer2_regime import get_cached_regime
from scoring_core import ScoringResult, score_candles

logger = logging.getLogger(__name__)

MIN_SCORE_LONG  = config.scoring.get("min_score_long",  5)
MIN_SCORE_SHORT = config.scoring.get("min_score_short", -5)


def calculate_layer3_score(
    symbol: str,
    klines_15m: list,
    funding_rate: float,
    fg_index: float = 50.0,
    open_interest: float = 0.0,
    vwap24h: float = 0.0,
    high24h: float = 0.0,
    low24h: float = 0.0,
    strategy: str = None,
) -> ScoringResult:
    """
    Live-Scoring-Aufruf.
    Verwendet gecachtes Regime (NICHT Layer 2 aufrufen – nur alle 4h!).
    """
    regime   = get_cached_regime()
    strategy = strategy or config.strategy

    adx_chop = config.indicators.get("adx_chop_threshold", 18)

    result = score_candles(
        symbol=symbol,
        klines=klines_15m,
        funding_rate=funding_rate,
        fg_index=fg_index,
        open_interest=open_interest,
        vwap24h=vwap24h,
        high24h=high24h,
        low24h=low24h,
        strategy=strategy,
        min_score_long=MIN_SCORE_LONG,
        min_score_short=MIN_SCORE_SHORT,
        cached_regime=regime,
        adx_chop_threshold=adx_chop,
    )

    # ML-Veto prüfen
    result = _apply_ml_veto(result, symbol)

    logger.info(
        f"Scoring [{symbol}|{strategy}]: Score={result.score}, "
        f"Signal={'JA' if result.signal else 'NEIN'}, "
        f"Regime={result.regime}, Veto={result.veto_reason or '-'}"
    )
    return result


def _apply_ml_veto(result: ScoringResult, symbol: str) -> ScoringResult:
    """
    Zweistufiges ML-Veto:
      1. Candle-Modell (3-Klassen): Konfidenz < 55% ODER falsche Klasse → veto
      2. Win-Modell (binär): P(win) < 0.42 → veto
    """
    if not result.signal:
        return result
    try:
        from ml_network import ml_network

        # Stufe 1: Candle-Modell (Modell A)
        ml_dir, confidence = ml_network.predict_direction(symbol, result)
        if ml_dir is not None:
            rule_dir = result.direction  # "long" oder "short"
            if ml_dir == "neutral":
                logger.info(f"ML-Veto A: {symbol} neutral (conf={confidence:.3f})")
                result.signal = False
                result.veto_reason = f"ml_neutral(conf={confidence:.3f})"
                return result
            if ml_dir != rule_dir:
                logger.info(f"ML-Veto A: {symbol} Richtung widerspricht "
                            f"(Regel={rule_dir}, ML={ml_dir}, conf={confidence:.3f})")
                result.signal = False
                result.veto_reason = f"ml_conflict({rule_dir}vs{ml_dir})"
                return result

        # Stufe 2: Win-Modell (Modell B)
        prob = ml_network.predict_win_prob(symbol, result)
        threshold = config.ml.get("veto_threshold", 0.42)
        if prob is not None and prob < threshold:
            logger.info(f"ML-Veto B: {symbol} P(win)={prob:.3f} < {threshold}")
            result.signal = False
            result.veto_reason = f"ml_win(p={prob:.3f})"

    except Exception:
        pass  # ML nicht verfügbar → kein Veto
    return result


async def fetch_fear_greed_index() -> float:
    """Holt den Fear & Greed Index von alternative.me."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data["data"][0]["value"])
    except Exception as e:
        logger.warning(f"Fear & Greed nicht abrufbar: {e}")
    return 50.0
