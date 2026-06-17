"""
layers/layer3_scoring.py – Wrapper um scoring_core.py für den Live-Bot.
Holt gecachtes Regime und delegiert an scoring_core.score_candles().
"""

import logging
import random
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


def _exploration_override(result: ScoringResult, prob: float, threshold: float) -> bool:
    """
    A1 — Entscheidet, ob ein durch Modell B vetoetes Signal als Exploration
    trotzdem gehandelt wird. Nur "knapp daneben" + optional starker Score, dann
    mit Wahrscheinlichkeit exploration_rate. Im Paper-Modus risikolos, liefert
    aber echte 1.0-Labels in genau der Grauzone, die das Veto sonst nie sieht.
    """
    if not config.ml.get("exploration_enabled", False):
        return False
    band = config.ml.get("exploration_band", 0.10)
    if prob < threshold - band:        # zu schlecht → kein Exploration, echtes Veto
        return False
    min_score = config.ml.get("exploration_min_score", 0)
    if min_score and abs(result.score) < min_score:
        return False
    return random.random() < config.ml.get("exploration_rate", 0.10)


def _exploration_override_a(result: ScoringResult) -> bool:
    """
    A1b — Erlaubt Exploration, ein Modell-A-Veto zu überstimmen, damit die
    Lernschleife nie vollständig zugeht (auch im confirm-Modus / bei enger
    contradict_conf). Mode-agnostisch: feuert mit exploration_rate, optional
    nur ab |score| >= exploration_min_score. Im Paper-Modus risikolos.
    """
    if not config.ml.get("exploration_enabled", False):
        return False
    if not config.ml.get("exploration_over_candle", True):
        return False
    min_score = config.ml.get("exploration_min_score", 0)
    if min_score and abs(result.score) < min_score:
        return False
    return random.random() < config.ml.get("exploration_rate", 0.10)


def _apply_ml_veto(result: ScoringResult, symbol: str) -> ScoringResult:
    """
    Zweistufiges ML-Veto:
      1. Candle-Modell A (3-Klassen): Politik per config.ml["candle_veto_mode"]
         ("contradict": nur Gegen-Signal blockt | "confirm": muss bestätigen).
      2. Win-Modell B (binär): P(win) < veto_threshold → veto.
    Exploration kann beide Stufen gelegentlich überstimmen (echte Labels sammeln).
    """
    if not result.signal:
        return result
    try:
        from ml_network import ml_network

        # Stufe 1: Candle-Modell (Modell A) — config-getriebene Politik
        a_vetoed, a_reason, _a_info = ml_network.candle_veto(symbol, result)
        if a_vetoed:
            if _exploration_override_a(result):
                result.exploration = True
                result.details["_exploration"] = 1.0
                logger.info(f"ML-Veto A: {symbol} {a_reason} "
                            f"→ EXPLORATION (Veto überstimmt)")
                return result  # Signal bleibt aktiv, Stufe B wird bewusst übersprungen
            logger.info(f"ML-Veto A: {symbol} {a_reason}")
            result.signal = False
            result.veto_reason = a_reason
            return result

        # Stufe 2: Win-Modell (Modell B)
        prob = ml_network.predict_win_prob(symbol, result)
        threshold = config.ml.get("veto_threshold", 0.42)
        if prob is not None and prob < threshold:
            # A1 — Exploration: knapp-vetoete High-Score-Signale gelegentlich
            # trotzdem handeln, um echte Labels in der Grauzone zu sammeln.
            if _exploration_override(result, prob, threshold):
                result.exploration = True
                result.details["_exploration"] = 1.0
                logger.info(f"ML-Veto B: {symbol} P(win)={prob:.3f} < {threshold} "
                            f"→ EXPLORATION (Veto überstimmt)")
            else:
                logger.info(f"ML-Veto B: {symbol} P(win)={prob:.3f} < {threshold}")
                result.signal = False
                result.veto_reason = f"ml_win(p={prob:.3f})"

    except Exception as e:
        # K-H-Fix: Exception NICHT still verschlucken. Bei Feature-Dimension-Mismatch
        # nach einem Retrain (z.B. altes Modell mit 22 Features) wären sonst ALLE
        # ML-Vetos lautlos deaktiviert – Trades gingen ungefiltert durch.
        logger.warning(f"ML-Veto Exception ({symbol}): {type(e).__name__}: {e}")
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
