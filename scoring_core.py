"""
scoring_core.py – Gemeinsamer Scoring-Kern für Live-Bots UND Lernfabrik.
Strategien: momentum | mean_reversion | breakout | contrarian | scalper
Chop-Erkennung: ADX < 18 + enge BB-Range → Signal-Veto.
Wichtig: Live und Lernfabrik nutzen EXAKT denselben Kern.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    symbol: str
    score: int
    direction: Optional[str]     # "long" | "short" | None
    signal: bool
    atr: float
    atr_ratio: float
    regime: str
    funding_rate: float
    fg_index: float
    details: dict = field(default_factory=dict)
    veto_reason: Optional[str] = None
    exploration: bool = False    # A1: Veto bewusst überstimmt, um echtes Label zu sammeln


def score_candles(
    symbol: str,
    klines: list,
    funding_rate: float = 0.0,
    fg_index: float = 50.0,
    open_interest: float = 0.0,
    vwap24h: float = 0.0,
    high24h: float = 0.0,
    low24h: float = 0.0,
    strategy: str = "momentum",
    min_score_long: int = 5,
    min_score_short: int = -5,
    cached_regime: str = "ranging",
    adx_chop_threshold: float = 18.0,
) -> ScoringResult:
    """
    Hauptfunktion: berechnet Score für eine Kerzenreihe.
    Wird von layer3_scoring.py (Live) und learning_factory.py (Offline) aufgerufen.
    """
    if len(klines) < 50:
        return _empty(symbol, funding_rate, fg_index, cached_regime)

    df = _to_df(klines)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    rsi       = _rsi(close, 14)
    rsi7      = _rsi(close, 7)
    macd, sig = _macd(close)
    bb_u, bb_m, bb_l = _bb(close, 20, 2.0)
    atr_s     = _atr(high, low, close, 14)
    adx_v     = _adx(high, low, close, 14)
    ema9      = close.ewm(span=9,  adjust=False).mean()
    ema21     = close.ewm(span=21, adjust=False).mean()

    rsi21     = _rsi(close, 21)
    ema50     = close.ewm(span=50, adjust=False).mean()

    atr_avg   = atr_s.rolling(50).mean().iloc[-1] if len(atr_s) >= 50 else atr_s.mean()
    atr_val   = float(atr_s.iloc[-1])
    atr_ratio = atr_val / atr_avg if atr_avg > 0 else 1.0

    cur_close  = float(close.iloc[-1])
    cur_rsi    = float(rsi.iloc[-1])  if not np.isnan(rsi.iloc[-1])  else 50.0
    cur_rsi7   = float(rsi7.iloc[-1]) if not np.isnan(rsi7.iloc[-1]) else 50.0
    cur_rsi21  = float(rsi21.iloc[-1]) if not np.isnan(rsi21.iloc[-1]) else 50.0
    cur_adx    = float(adx_v.iloc[-1]) if not np.isnan(adx_v.iloc[-1]) else 20.0
    cur_macd   = float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0.0
    cur_sig    = float(sig.iloc[-1])  if not np.isnan(sig.iloc[-1])  else 0.0
    prev_macd  = float(macd.iloc[-2]) if len(macd) > 1 else cur_macd
    prev_sig   = float(sig.iloc[-2])  if len(sig) > 1  else cur_sig
    cur_bbu    = float(bb_u.iloc[-1])
    cur_bbl    = float(bb_l.iloc[-1])
    cur_bbm    = float(bb_m.iloc[-1])
    cur_ema9   = float(ema9.iloc[-1])
    cur_ema21  = float(ema21.iloc[-1])
    cur_ema50  = float(ema50.iloc[-1])
    bb_range   = cur_bbu - cur_bbl
    bb_pct     = (cur_close - cur_bbl) / bb_range if bb_range > 0 else 0.5
    bb_width   = bb_range / cur_close if cur_close > 0 else 0.0
    vol_avg20  = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    vol_ratio  = float(vol.iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0

    # Preis-Returns
    ret_1  = float(close.pct_change(1).iloc[-1])  if len(close) > 1  else 0.0
    ret_4  = float(close.pct_change(4).iloc[-1])  if len(close) > 4  else 0.0
    ret_8  = float(close.pct_change(8).iloc[-1])  if len(close) > 8  else 0.0
    ret_16 = float(close.pct_change(16).iloc[-1]) if len(close) > 16 else 0.0
    ret_1  = 0.0 if np.isnan(ret_1)  else ret_1
    ret_4  = 0.0 if np.isnan(ret_4)  else ret_4
    ret_8  = 0.0 if np.isnan(ret_8)  else ret_8
    ret_16 = 0.0 if np.isnan(ret_16) else ret_16

    # RSI-Steigung (letzte 3 Werte)
    rsi_slope = 0.0
    if len(rsi) >= 4 and not rsi.iloc[-4:].isna().any():
        rsi_slope = float((rsi.iloc[-1] - rsi.iloc[-4]) / 3)

    # Zeit-Features aus Kerzen-Timestamp (ms)
    hour_sin = hour_cos = weekday_sin = weekday_cos = 0.0
    try:
        ts_ms = klines[-1][0]
        if ts_ms and ts_ms > 0:
            from datetime import datetime, timezone as _tz
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=_tz.utc)
            hour_sin     = float(np.sin(2 * np.pi * dt.hour / 24))
            hour_cos     = float(np.cos(2 * np.pi * dt.hour / 24))
            weekday_sin  = float(np.sin(2 * np.pi * dt.weekday() / 7))
            weekday_cos  = float(np.cos(2 * np.pi * dt.weekday() / 7))
    except Exception:
        pass

    # 22 ML-Features in details speichern
    details: dict = {
        "_rsi": cur_rsi, "_adx": cur_adx,
        # ML-Features (Prefix _ damit sie nicht mit Score-Details verwechselt werden)
        "_rsi_7":           cur_rsi7,
        "_rsi_14":          cur_rsi,
        "_rsi_21":          cur_rsi21,
        "_macd_diff":       cur_macd - cur_sig,
        "_macd_signal":     cur_sig,
        "_ema_ratio_9_21":  cur_ema9  / cur_ema21 - 1 if cur_ema21 > 0 else 0.0,
        "_ema_ratio_21_50": cur_ema21 / cur_ema50 - 1 if cur_ema50 > 0 else 0.0,
        "_price_vs_ema50":  cur_close / cur_ema50 - 1 if cur_ema50 > 0 else 0.0,
        "_bb_pct":          bb_pct,
        "_bb_width":        bb_width,
        "_atr_ratio":       atr_ratio,
        "_vol_ratio":       vol_ratio,
        "_ret_1":           ret_1,
        "_ret_4":           ret_4,
        "_ret_8":           ret_8,
        "_ret_16":          ret_16,
        "_hour_sin":        hour_sin,
        "_hour_cos":        hour_cos,
        "_weekday_sin":     weekday_sin,
        "_weekday_cos":     weekday_cos,
        "_fg_index":        fg_index,
        "_rsi_slope":       rsi_slope,
        # S5-3-Fix: Strategie in details, damit predict_win_prob (ml_network) live
        # den echten strategy_enc bekommt statt konstant "momentum" (Train/Live-Mismatch).
        "_strategy":        strategy,
    }

    # ─── Chop-Erkennung (gilt für alle Strategien außer mean_reversion) ───────
    if strategy != "mean_reversion":
        bb_width_ratio = bb_range / cur_close if cur_close > 0 else 1
        if cur_adx < adx_chop_threshold and bb_width_ratio < 0.02:
            return ScoringResult(
                symbol=symbol, score=0, direction=None, signal=False,
                atr=atr_val, atr_ratio=atr_ratio, regime=cached_regime,
                funding_rate=funding_rate, fg_index=fg_index,
                details=details, veto_reason="chop",
            )

    # ─── Strategie-Router ──────────────────────────────────────────────────────
    if strategy == "mean_reversion":
        score, details = _score_mean_reversion(
            cur_close, cur_rsi, cur_rsi7, bb_pct, cur_adx, details)
    elif strategy == "breakout":
        score, details = _score_breakout(
            close, high, low, cur_adx, atr_val, details)
    elif strategy == "contrarian":
        # Contrarian = invertierter Momentum-Score
        score, details = _score_momentum(
            cur_rsi, cur_macd, cur_sig, prev_macd, prev_sig,
            bb_pct, cur_ema9, cur_ema21, cached_regime, fg_index, details)
        score = -score
        details["contrarian"] = "Score invertiert"
    elif strategy == "scalper":
        score, details = _score_scalper(
            cur_rsi, cur_macd, cur_sig, prev_macd, prev_sig, bb_pct, details)
    else:  # momentum (default)
        score, details = _score_momentum(
            cur_rsi, cur_macd, cur_sig, prev_macd, prev_sig,
            bb_pct, cur_ema9, cur_ema21, cached_regime, fg_index, details)

    # ─── Layer 4: Market Structure (Funding, OI, VWAP) ───────────────────────
    score, details = _score_layer4(
        score, details, funding_rate, open_interest,
        cur_close, vwap24h, high24h, low24h,
        close.diff().mean() if len(close) > 1 else 0,  # simple price trend
    )

    # ─── Signal bestimmen ─────────────────────────────────────────────────────
    direction: Optional[str] = None
    signal = False
    if score >= min_score_long:
        direction = "long"
        signal = True
    elif score <= min_score_short:
        direction = "short"
        signal = True

    return ScoringResult(
        symbol=symbol, score=score, direction=direction, signal=signal,
        atr=atr_val, atr_ratio=atr_ratio, regime=cached_regime,
        funding_rate=funding_rate, fg_index=fg_index, details=details,
    )


def passes_regime_gate(direction: Optional[str], regime: str, strategy: str) -> bool:
    """
    Fix 4 (4h-Regime-Bestätigung): erlaubt einen Entry nur, wenn das 4h-Regime
    die Richtung nicht klar widerlegt.
      - Long  blockiert, wenn Regime == 'trending_down'
      - Short blockiert, wenn Regime == 'trending_up'
      - mean_reversion ist AUSGENOMMEN (handelt gewollt gegen den Trend)
    Wird identisch im Live-Bot (main.py) UND im Vergleichstest (test_regime_gate.py) genutzt.
    """
    if strategy == "mean_reversion":
        return True
    if direction == "long" and regime == "trending_down":
        return False
    if direction == "short" and regime == "trending_up":
        return False
    return True


# ─── Momentum-Scoring ─────────────────────────────────────────────────────────

def _score_momentum(
    cur_rsi, cur_macd, cur_sig, prev_macd, prev_sig,
    bb_pct, cur_ema9, cur_ema21, regime, fg_index, details
) -> Tuple[int, dict]:
    score = 0
    confirmed_long = confirmed_short = False

    # 1. RSI
    if cur_rsi < 30:
        score += 2; details["rsi"] = f"+2 (überverkauft {cur_rsi:.1f})"; confirmed_long = True
    elif cur_rsi < 40:
        score += 1; details["rsi"] = f"+1 (schwach {cur_rsi:.1f})"
    elif cur_rsi > 70:
        score -= 2; details["rsi"] = f"-2 (überkauft {cur_rsi:.1f})"; confirmed_short = True
    elif cur_rsi > 60:
        score -= 1; details["rsi"] = f"-1 (schwach {cur_rsi:.1f})"
    else:
        details["rsi"] = f"0 (neutral {cur_rsi:.1f})"

    # 2. MACD Crossover
    if cur_macd > cur_sig and prev_macd <= prev_sig:
        score += 2; details["macd"] = "+2 (bullish Crossover)"; confirmed_long = True
    elif cur_macd < cur_sig and prev_macd >= prev_sig:
        score -= 2; details["macd"] = "-2 (bearish Crossover)"; confirmed_short = True
    elif cur_macd > cur_sig:
        score += 1; details["macd"] = "+1 (bullish)"
    elif cur_macd < cur_sig:
        score -= 1; details["macd"] = "-1 (bearish)"

    # 3. EMA 9/21
    if cur_ema9 > cur_ema21:
        score += 1; details["ema"] = "+1 (bullish)"
    elif cur_ema9 < cur_ema21:
        score -= 1; details["ema"] = "-1 (bearish)"

    # 4. Bollinger Bands
    if bb_pct < 0.05:
        score += 2; details["bb"] = f"+2 (unteres Band {bb_pct:.2f})"; confirmed_long = True
    elif bb_pct < 0.25:
        score += 1; details["bb"] = "+1 (untere Zone)"
    elif bb_pct > 0.95:
        score -= 2; details["bb"] = f"-2 (oberes Band {bb_pct:.2f})"; confirmed_short = True
    elif bb_pct > 0.75:
        score -= 1; details["bb"] = "-1 (obere Zone)"

    # 5. Regime
    if regime == "trending_up":
        score += 1; details["regime"] = "+1"
    elif regime == "trending_down":
        score -= 1; details["regime"] = "-1"

    # 6. Fear & Greed (Korrektur: -1 wenn nicht bestätigt, nicht +1)
    if fg_index < 20:
        v = 3 if confirmed_long else 1
        score += v; details["fg"] = f"+{v} (extreme fear {fg_index:.0f})"
    elif fg_index < 35:
        score += 1; details["fg"] = f"+1 (fear {fg_index:.0f})"
    elif fg_index > 85:
        v = 3 if confirmed_short else 1
        score -= v; details["fg"] = f"-{v} (extreme greed {fg_index:.0f})"
    elif fg_index > 65:
        score -= 1; details["fg"] = f"-1 (greed {fg_index:.0f})"

    return score, details


# ─── Mean-Reversion-Scoring ───────────────────────────────────────────────────

def _score_mean_reversion(
    cur_close, cur_rsi, cur_rsi7, bb_pct, cur_adx, details
) -> Tuple[int, dict]:
    """
    Handelt Übertreibungen Richtung Mittelwert.
    Meidet starke Trends (ADX > 30).
    """
    score = 0

    # Starken Trend meiden
    if cur_adx > 30:
        details["mr_veto"] = f"Trend zu stark (ADX {cur_adx:.1f})"
        return 0, details

    # Long-Signal: übertrieben verkauft, nahe unterem BB
    if cur_rsi < 30 and bb_pct < 0.10:
        score += 4; details["mr"] = f"+4 (RSI={cur_rsi:.1f}, BB={bb_pct:.2f})"
    elif cur_rsi < 35 and bb_pct < 0.15:
        score += 3; details["mr"] = f"+3 (RSI={cur_rsi:.1f}, BB={bb_pct:.2f})"
    elif cur_rsi < 40 and bb_pct < 0.25:
        score += 2; details["mr"] = f"+2 (RSI={cur_rsi:.1f}, BB={bb_pct:.2f})"

    # Short-Signal: übertrieben gekauft, nahe oberem BB
    elif cur_rsi > 70 and bb_pct > 0.90:
        score -= 4; details["mr"] = f"-4 (RSI={cur_rsi:.1f}, BB={bb_pct:.2f})"
    elif cur_rsi > 65 and bb_pct > 0.85:
        score -= 3; details["mr"] = f"-3 (RSI={cur_rsi:.1f}, BB={bb_pct:.2f})"
    elif cur_rsi > 60 and bb_pct > 0.75:
        score -= 2; details["mr"] = f"-2 (RSI={cur_rsi:.1f}, BB={bb_pct:.2f})"

    # RSI7 Momentum als Bestätigung
    if score > 0 and cur_rsi7 < 35:
        score += 1; details["mr_conf"] = "+1 (RSI7 bestätigt)"
    elif score < 0 and cur_rsi7 > 65:
        score -= 1; details["mr_conf"] = "-1 (RSI7 bestätigt)"

    return score, details


# ─── Breakout-Scoring (Turtle/Donchian) ──────────────────────────────────────

def _score_breakout(close, high, low, cur_adx, atr_val, details) -> Tuple[int, dict]:
    """
    Donchian-Breakout: handelt neue Hochs/Tiefs mit Trendbestätigung.
    """
    score = 0
    n = min(20, len(close) - 1)
    if n < 5:
        return 0, details

    high20 = high.iloc[-n-1:-1].max()
    low20  = low.iloc[-n-1:-1].min()
    cur_h  = float(high.iloc[-1])
    cur_l  = float(low.iloc[-1])
    cur_c  = float(close.iloc[-1])

    # Breakout nach oben
    if cur_h > high20 and cur_adx > 20:
        score += 4
        details["breakout"] = f"+4 (Breakout hoch: {cur_h:.2f} > {high20:.2f})"
    elif cur_h > high20:
        score += 2
        details["breakout"] = f"+2 (Breakout hoch ohne Trend)"

    # Breakout nach unten
    elif cur_l < low20 and cur_adx > 20:
        score -= 4
        details["breakout"] = f"-4 (Breakout tief: {cur_l:.2f} < {low20:.2f})"
    elif cur_l < low20:
        score -= 2
        details["breakout"] = f"-2 (Breakout tief ohne Trend)"

    # ADX-Bonus
    if abs(score) > 0 and cur_adx > 30:
        score = int(score * 1.5)
        details["adx_bonus"] = f"Bonus (ADX={cur_adx:.1f})"

    return score, details


# ─── Scalper-Scoring ──────────────────────────────────────────────────────────

def _score_scalper(cur_rsi, cur_macd, cur_sig, prev_macd, prev_sig,
                   bb_pct, details) -> Tuple[int, dict]:
    """
    Scalper: niedrige Score-Schwelle, schnelle Signale.
    Hauptsächlich MACD-Crossovers und extreme RSI.
    """
    score = 0

    # MACD Crossover (Hauptsignal)
    if cur_macd > cur_sig and prev_macd <= prev_sig:
        score += 3; details["scalp_macd"] = "+3 (bullish XO)"
    elif cur_macd < cur_sig and prev_macd >= prev_sig:
        score -= 3; details["scalp_macd"] = "-3 (bearish XO)"
    elif cur_macd > cur_sig:
        score += 1; details["scalp_macd"] = "+1"
    elif cur_macd < cur_sig:
        score -= 1; details["scalp_macd"] = "-1"

    # RSI Extrema
    if cur_rsi < 25:
        score += 2; details["scalp_rsi"] = "+2"
    elif cur_rsi > 75:
        score -= 2; details["scalp_rsi"] = "-2"

    # BB-Extreme
    if bb_pct < 0.05:
        score += 1
    elif bb_pct > 0.95:
        score -= 1

    return score, details


# ─── Layer 4: Market Structure ────────────────────────────────────────────────

def _score_layer4(
    score: int, details: dict,
    funding_rate: float,
    open_interest: float,
    cur_price: float,
    vwap24h: float,
    high24h: float,
    low24h: float,
    price_trend: float,
) -> Tuple[int, dict]:
    """Funding Rate, Open Interest, VWAP, Range-Position."""

    # Funding Rate
    if funding_rate > 0.0020:         # > 0.20%/h  → extremer Short-Druck
        score -= 3; details["funding"] = f"-3 (extrem {funding_rate:.4%})"
    elif funding_rate > 0.0005:        # > 0.05%/h
        score -= 2; details["funding"] = f"-2 ({funding_rate:.4%})"
    elif funding_rate < -0.0005:       # < -0.05%/h → Long-Druck
        score += 2; details["funding"] = f"+2 ({funding_rate:.4%})"
    else:
        details["funding"] = f"0 ({funding_rate:.4%})"

    # Open Interest + Preis-Divergenz
    if open_interest > 0 and price_trend != 0:
        # OI steigt + Preis fällt → neue Shorts öffnen
        if price_trend < 0:
            score -= 2; details["oi"] = "-2 (OI↑ Preis↓)"
        # OI steigt + Preis steigt → neue Longs
        elif price_trend > 0:
            score += 2; details["oi"] = "+2 (OI↑ Preis↑)"

    # VWAP24h
    if vwap24h > 0:
        if cur_price < vwap24h:
            score -= 1; details["vwap"] = "-1 (unter VWAP)"
        else:
            score += 1; details["vwap"] = "+1 (über VWAP)"

    # 24h Range-Position
    range24h = high24h - low24h
    if range24h > 0:
        pos24h = (cur_price - low24h) / range24h
        if pos24h > 0.90:
            score -= 1; details["range24h"] = f"-1 (nahe High {pos24h:.0%})"
        elif pos24h < 0.10:
            score += 1; details["range24h"] = f"+1 (nahe Low {pos24h:.0%})"

    return score, details


# ─── Technische Indikatoren ───────────────────────────────────────────────────

def _to_df(klines: list) -> pd.DataFrame:
    """Konvertiert normalisiertes Kerzenformat in DataFrame."""
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "q", "t", "tbb", "tbq", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=12, slow=26, sig=9) -> Tuple[pd.Series, pd.Series]:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    return macd, macd.ewm(span=sig, adjust=False).mean()


def _bb(close: pd.Series, period=20, std_dev=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std


def _atr(high, low, close, period=14) -> pd.Series:
    prev = close.shift(1)
    tr   = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high, low, close, period=14) -> pd.Series:
    """Vereinfachtes ADX (nur Trend-Stärke, kein +DI/-DI)."""
    prev_h = high.shift(1)
    prev_l = low.shift(1)
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    up_move   = high - prev_h
    down_move = prev_l - low
    plus_dm   = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm  = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr14  = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr14.replace(0, np.nan)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean().fillna(0)


def _empty(symbol, funding_rate=0.0, fg_index=50.0, regime="ranging") -> ScoringResult:
    return ScoringResult(
        symbol=symbol, score=0, direction=None, signal=False,
        atr=0.0, atr_ratio=1.0, regime=regime,
        funding_rate=funding_rate, fg_index=fg_index,
    )
