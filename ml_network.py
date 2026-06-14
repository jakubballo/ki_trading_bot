"""
ml_network.py – Zwei-Ebenen ML auf network.db.

Modell A – Candle-Modell (3 Klassen: Long / Short / Neutral):
  • 22 Features aus Kerzen-Indikatoren (per ML_STRATEGIE_DOKU)
  • Strict Labels: TP in <6 Kerzen ohne SL-Hit vorher
  • Konfidenz-Schwelle: 55%  (< 55% → neutral, kein Trade)
  • Trainiert auf historischen CSVs via learning_factory / train_from_csv()
  • Klassen-Gewichtung: Neutral=1.0, Long≈hochgewichtet, Short≈hochgewichtet

Modell B – Win-Modell (binär P(win)):
  • Trainiert auf network.db Outcomes (real/shadow/synthetic)
  • Gewichte: real 1.0 / shadow 0.5 / synthetic 0.2
  • Veto: P(win) < 0.42

Live-Veto-Logik (in layer3_scoring.py):
  1. Candle-Modell (A) wenn geladen → Konfidenz < 55% → verwerfen
  2. Win-Modell (B) → P(win) < 0.42 → verwerfen
  3. Beide müssen bestehen
"""

import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(os.environ.get("ML_MODEL_DIR", "data/ml_models"))
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Klassen-Mapping
CLASS_NEUTRAL = 0
CLASS_LONG    = 1
CLASS_SHORT   = 2
CLASS_NAMES   = {0: "neutral", 1: "long", 2: "short"}

# 22 Feature-Namen (aus ML_STRATEGIE_DOKU)
CANDLE_FEATURES = [
    "rsi_7", "rsi_14", "rsi_21",
    "macd_diff", "macd_signal_line",
    "ema_ratio_9_21", "ema_ratio_21_50", "price_vs_ema50",
    "bb_pct", "bb_width",
    "atr_ratio",
    "vol_ratio",
    "ret_1", "ret_4", "ret_8", "ret_16",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "fg_index",
    "rsi_slope",
]

# Features für Win-Modell (aus network.db)
WIN_FEATURES = [
    "score", "funding_rate", "rsi", "atr", "fg_index",
    "is_shadow", "is_synthetic", "regime_enc", "strategy_enc",
    # Neue Marktstruktur-Features (Problem 2, ab 2026-06-14)
    "macd_diff", "macd_signal_val",
    "ema_ratio_9_21", "ema_ratio_21_50", "price_vs_ema50",
    "bb_pct", "bb_width", "vol_ratio", "rsi_slope",
    "ret_1", "ret_4", "ret_8", "ret_16",
]

REGIME_MAP   = {"trending_up": 0, "trending_down": 1, "ranging": 2, "volatile": 3}
STRATEGY_MAP = {"momentum": 0, "mean_reversion": 1, "breakout": 2, "contrarian": 3, "scalper": 4}

# XGBoost Hyperparameter (per ML_STRATEGIE_DOKU)
XGB_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 10,
    gamma            = 1.0,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    eval_metric      = "mlogloss",
    use_label_encoder= False,
    n_jobs           = -1,
    verbosity        = 0,
    random_state     = 42,
)


# ────────────────────────────── Feature-Extraktion ────────────────────────────

def _candle_features_from_result(sr) -> Optional[np.ndarray]:
    """Extrahiert 22 Candle-Features aus ScoringResult.details."""
    d = sr.details if hasattr(sr, "details") else {}
    try:
        feat = np.array([
            d.get("_rsi_7",           50.0),
            d.get("_rsi_14",          sr.details.get("_rsi", 50.0)),
            d.get("_rsi_21",          50.0),
            d.get("_macd_diff",       0.0),
            d.get("_macd_signal",     0.0),
            d.get("_ema_ratio_9_21",  0.0),
            d.get("_ema_ratio_21_50", 0.0),
            d.get("_price_vs_ema50",  0.0),
            d.get("_bb_pct",          0.5),
            d.get("_bb_width",        0.0),
            d.get("_atr_ratio",       getattr(sr, "atr_ratio", 1.0)),
            d.get("_vol_ratio",       1.0),
            d.get("_ret_1",           0.0),
            d.get("_ret_4",           0.0),
            d.get("_ret_8",           0.0),
            d.get("_ret_16",          0.0),
            d.get("_hour_sin",        0.0),
            d.get("_hour_cos",        0.0),
            d.get("_weekday_sin",     0.0),
            d.get("_weekday_cos",     0.0),
            float(getattr(sr, "fg_index", d.get("_fg_index", 50.0))),
            d.get("_rsi_slope",       0.0),
        ], dtype=np.float32)
        if np.any(np.isnan(feat)):
            feat = np.nan_to_num(feat, nan=0.0)
        return feat.reshape(1, -1)
    except Exception as e:
        logger.debug(f"Feature-Extraktion Fehler: {e}")
        return None


def _win_features_from_row(row: dict) -> np.ndarray:
    """Baut Win-Modell-Feature-Vektor aus network.db-Zeile (22 Features)."""
    return np.array([
        float(row.get("score")           or 0),
        float(row.get("funding_rate")    or 0),
        float(row.get("rsi")             or 50),
        float(row.get("atr")             or 0),
        float(row.get("fg_index")        or 50),
        float(row.get("is_shadow")       or 0),
        float(row.get("is_synthetic")    or 0),
        float(REGIME_MAP.get(row.get("regime", "ranging"), 2)),
        float(STRATEGY_MAP.get(row.get("strategy", "momentum"), 0)),
        # Neue Features – NULL-Zeilen (alte Synthetik) bekommen Defaults
        float(row.get("macd_diff")       or 0),
        float(row.get("macd_signal_val") or 0),
        float(row.get("ema_ratio_9_21")  or 0),
        float(row.get("ema_ratio_21_50") or 0),
        float(row.get("price_vs_ema50")  or 0),
        float(row.get("bb_pct")          if row.get("bb_pct") is not None else 0.5),
        float(row.get("bb_width")        or 0),
        float(row.get("vol_ratio")       if row.get("vol_ratio") is not None else 1.0),
        float(row.get("rsi_slope")       or 0),
        float(row.get("ret_1")           or 0),
        float(row.get("ret_4")           or 0),
        float(row.get("ret_8")           or 0),
        float(row.get("ret_16")          or 0),
    ], dtype=np.float32)


# ────────────────────────────── Strict Labels (ML_STRATEGIE_DOKU §6) ─────────

def generate_strict_labels(
    klines: list,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_candles: int = 6,
) -> np.ndarray:
    """
    Erzeugt strenge Labels: 0=neutral, 1=long, 2=short.
    TP muss in <6 Kerzen getroffen werden OHNE vorherigen SL-Hit.
    ~26% Long/Short, ~46% Neutral bei realistischen Parametern.
    """
    n = len(klines)
    labels = np.zeros(n, dtype=np.int32)

    closes = np.array([float(k[4]) for k in klines], dtype=np.float64)
    highs  = np.array([float(k[2]) for k in klines], dtype=np.float64)
    lows   = np.array([float(k[3]) for k in klines], dtype=np.float64)

    # ATR (14)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i]  - closes[i-1]))
    atr = np.zeros(n)
    for i in range(14, n):
        atr[i] = tr[i-13:i+1].mean()

    for i in range(14, n - max_candles - 1):
        entry = closes[i]
        a     = atr[i]
        if a <= 0:
            continue

        tp_long  = entry + tp_mult * a
        sl_long  = entry - sl_mult * a
        tp_short = entry - tp_mult * a
        sl_short = entry + sl_mult * a

        long_hit = long_stop = short_hit = short_stop = False

        for j in range(i + 1, min(i + max_candles + 1, n)):
            h, l = highs[j], lows[j]
            if not long_stop and not long_hit:
                if l <= sl_long:   long_stop = True
                elif h >= tp_long: long_hit  = True
            if not short_stop and not short_hit:
                if h >= sl_short:   short_stop = True
                elif l <= tp_short: short_hit  = True

        if long_hit and not long_stop:
            labels[i] = CLASS_LONG
        elif short_hit and not short_stop:
            labels[i] = CLASS_SHORT

    neutral_pct = float((labels == 0).sum()) / max(len(labels), 1) * 100
    logger.debug(f"Labels: Long={( labels==1).sum()}, Short={(labels==2).sum()}, "
                 f"Neutral={(labels==0).sum()} ({neutral_pct:.0f}%)")
    return labels


# ────────────────────────────── ML-Netzwerk ───────────────────────────────────

class MLNetwork:
    """
    Zwei-Ebenen ML:
      Modell A = 3-Klassen Candle-Modell (Long/Short/Neutral)
      Modell B = Binäres Win-Modell (P(win) aus network.db)
    """

    def __init__(self):
        # Symbol-spezifische Candle-Modelle (Modell A)
        self._candle_models:  Dict[str, object] = {}
        self._candle_base:    Optional[object]  = None  # Fallback über alle Symbole

        # Win-Modelle (Modell B)
        self._win_models:     Dict[str, object] = {}
        self._win_base:       Optional[object]  = None

        self._last_trade_id:  int  = 0
        self._retrain_threshold    = 50
        self._load_models()

    # ────────────────── Live-Prediction ──────────────────

    def predict_direction(self, symbol: str, scoring_result) -> Tuple[Optional[str], float]:
        """
        Modell A: Gibt vorhergesagte Richtung + Konfidenz zurück.
        confidence < 0.55 → ("neutral", conf)

        Returns: (direction, confidence)  direction: "long"|"short"|"neutral"
        """
        feat = _candle_features_from_result(scoring_result)
        if feat is None:
            return None, 0.0

        model = self._candle_models.get(symbol) or self._candle_base
        if model is None:
            return None, 0.0

        try:
            proba     = model.predict_proba(feat)[0]
            pred_cls  = int(np.argmax(proba))
            confidence = float(proba[pred_cls])

            if confidence < 0.55:
                return "neutral", confidence

            direction = CLASS_NAMES.get(pred_cls, "neutral")
            return direction, confidence
        except Exception as e:
            logger.debug(f"predict_direction Fehler: {e}")
            return None, 0.0

    def predict_win_prob(self, symbol: str, scoring_result) -> Optional[float]:
        """
        Modell B: P(win) für das aktuelle Signal.
        Returns None wenn kein Modell geladen.
        """
        try:
            d = getattr(scoring_result, "details", {}) or {}
            feat = np.array([[
                float(getattr(scoring_result, "score", 0)),
                float(getattr(scoring_result, "funding_rate", 0)),
                float(d.get("_rsi_14", d.get("_rsi", 50))),
                float(getattr(scoring_result, "atr", 0)),
                float(getattr(scoring_result, "fg_index", 50)),
                0.0, 0.0,  # is_shadow, is_synthetic immer 0 bei Live-Prediction
                float(REGIME_MAP.get(getattr(scoring_result, "regime", "ranging"), 2)),
                float(STRATEGY_MAP.get(d.get("_strategy", "momentum"), 0)),
                # Neue Marktstruktur-Features
                float(d.get("_macd_diff",       0)),
                float(d.get("_macd_signal",     0)),
                float(d.get("_ema_ratio_9_21",  0)),
                float(d.get("_ema_ratio_21_50", 0)),
                float(d.get("_price_vs_ema50",  0)),
                float(d.get("_bb_pct",          0.5)),
                float(d.get("_bb_width",        0)),
                float(d.get("_vol_ratio",       1.0)),
                float(d.get("_rsi_slope",       0)),
                float(d.get("_ret_1",           0)),
                float(d.get("_ret_4",           0)),
                float(d.get("_ret_8",           0)),
                float(d.get("_ret_16",          0)),
            ]], dtype=np.float32)
        except Exception as e:
            logger.debug(f"Win-Feature Fehler: {e}")
            return None

        model = self._win_models.get(symbol) or self._win_base
        if model is None:
            return None

        try:
            return float(model.predict_proba(feat)[0][1])
        except Exception as e:
            logger.debug(f"Win predict Fehler: {e}")
            return None

    # ────────────────── Training ──────────────────

    def maybe_retrain(self):
        """Prüft ob neue network.db-Outcomes vorliegen → Win-Modell retrain."""
        try:
            from network_db import count_new_outcomes_since
            new_count = count_new_outcomes_since(self._last_trade_id)
            if new_count >= self._retrain_threshold:
                logger.info(f"ML Retrain: {new_count} neue Outcomes")
                self.train_win_models()
        except Exception as e:
            logger.error(f"maybe_retrain Fehler: {e}")

    def train_all(self):
        """Vollständiges Training beider Modell-Typen (Win-Modell B + Candle-Modell A)."""
        self.train_win_models()
        self._train_all_candle_models()

    def _train_all_candle_models(self):
        """Trainiert Candle-Modell A aus CSVs für alle 5 Symbole.
        Base-Modell wird einmalig am Ende gebaut (nicht nach jedem Symbol)."""
        history_dir = Path(os.environ.get("HISTORY_DIR", "data/history"))
        symbols = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_LINKUSD"]
        trained = 0
        for sym in symbols:
            csv_path = history_dir / f"{sym}_15m.csv"
            if not csv_path.exists():
                logger.warning(f"Candle-Training: CSV nicht gefunden: {csv_path}")
                continue
            try:
                klines = _load_csv_klines(str(csv_path))
                if len(klines) < 200:
                    logger.warning(f"CSV {csv_path}: zu wenig Kerzen ({len(klines)})")
                    continue
                labels = generate_strict_labels(klines)
                feats  = _compute_candle_features_batch(klines)
                valid  = ~np.isnan(feats).any(axis=1)
                X, y   = feats[valid], labels[valid]
                if len(X) < 50:
                    logger.warning(f"CSV {csv_path}: zu wenig gültige Samples")
                    continue
                logger.info(f"Candle-Training [{sym}]: {len(X)} Samples, "
                            f"L={( y==1).sum()} S={(y==2).sum()} N={(y==0).sum()}")
                model = self._train_candle_model(X, y)
                self._candle_models[sym] = model
                self._save_model(model, f"candle_{sym}")
                logger.info(f"Candle-Modell [{sym}] gespeichert")
                trained += 1
            except Exception as e:
                logger.error(f"train_from_csv Fehler [{sym}]: {e}", exc_info=True)
        # Base-Modell einmalig aus allen trainierten Symbolen
        if trained >= 2:
            self._rebuild_candle_base()
        logger.info(f"Candle-Training abgeschlossen: {trained}/{len(symbols)} Symbole")

    def train_win_models(self):
        """Trainiert Win-Modelle (Modell B) aus network.db."""
        try:
            from network_db import get_training_data, get_max_trade_id
            rows = get_training_data(limit=20_000)
            if len(rows) < 200:
                logger.info(f"Win-Training: zu wenig Daten ({len(rows)} < 200)")
                return

            logger.info(f"Win-Training gestartet: {len(rows)} Trades")
            t0 = time.time()

            self._win_base = self._train_win_model(rows)
            self._save_model(self._win_base, "win_base")

            from collections import defaultdict
            by_sym = defaultdict(list)
            for r in rows:
                by_sym[r["symbol"]].append(r)

            for sym, sym_rows in by_sym.items():
                if len(sym_rows) >= 150:
                    m = self._train_win_model(sym_rows)
                    self._win_models[sym] = m
                    self._save_model(m, f"win_{sym}")

            self._last_trade_id = get_max_trade_id()
            logger.info(f"Win-Training: {time.time()-t0:.1f}s, "
                        f"{len(self._win_models)} Symbol-Modelle")
        except Exception as e:
            logger.error(f"Win-Training Fehler: {e}", exc_info=True)

    def train_from_csv(self, symbol: str, csv_path: str):
        """
        Trainiert Candle-Modell (Modell A) aus einer historischen CSV.
        CSV-Format: timestamp,open,high,low,close,volume
        Wird von brain.py / learning_factory.py aufgerufen.
        """
        try:
            klines = _load_csv_klines(csv_path)
            if len(klines) < 200:
                logger.warning(f"CSV {csv_path}: zu wenig Kerzen ({len(klines)})")
                return

            labels = generate_strict_labels(klines)
            feats  = _compute_candle_features_batch(klines)

            valid  = ~np.isnan(feats).any(axis=1)
            X, y   = feats[valid], labels[valid]

            if len(X) < 50:
                logger.warning(f"CSV {csv_path}: zu wenig gültige Samples")
                return

            logger.info(f"Candle-Training [{symbol}]: {len(X)} Samples, "
                        f"L={( y==1).sum()} S={(y==2).sum()} N={(y==0).sum()}")

            model = self._train_candle_model(X, y)
            self._candle_models[symbol] = model
            self._save_model(model, f"candle_{symbol}")
            logger.info(f"Candle-Modell [{symbol}] gespeichert")
            # Base-Modell nur neu bauen wenn ≥2 Symbol-Modelle vorhanden
            # (beim Batch-Training via _train_all_candle_models übernimmt
            #  jene Methode das Base-Training einmalig am Ende)
            if len(self._candle_models) >= 2:
                self._rebuild_candle_base()
        except Exception as e:
            logger.error(f"train_from_csv Fehler [{symbol}]: {e}", exc_info=True)

    # ────────────────── Interne Trainer ──────────────────

    def _train_candle_model(self, X: np.ndarray, y: np.ndarray):
        """3-Klassen XGBoost mit Klassen-Gewichtung."""
        import xgboost as xgb

        n0 = (y == 0).sum()  # neutral
        n1 = (y == 1).sum()  # long
        n2 = (y == 2).sum()  # short

        weights = np.ones(len(y), dtype=np.float32)
        if n1 > 0: weights[y == 1] = n0 / n1
        if n2 > 0: weights[y == 2] = n0 / n2

        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        w_tr        = weights[:split]

        params = {**XGB_PARAMS, "objective": "multi:softprob", "num_class": 3,
                  "eval_metric": "mlogloss"}
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_val, y_val)], verbose=False)

        if len(X_val) > 0:
            acc = float(np.mean(model.predict(X_val) == y_val))
            logger.debug(f"Candle Val-Accuracy: {acc:.3f}")

        return model

    def _train_win_model(self, rows: list):
        """Binärer XGBoost: P(win) aus network.db."""
        import xgboost as xgb
        from config import config as cfg

        X, y, w = [], [], []
        for r in rows:
            pnl = r.get("pnl")
            if pnl is None:
                continue
            label  = 1 if float(pnl) > 0 else 0
            weight = float(r.get("weight") or 1.0)
            feat   = _win_features_from_row(r)
            X.append(feat); y.append(label); w.append(weight)

        if not X:
            return None

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        w = np.array(w, dtype=np.float32)

        split    = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        w_tr        = w[:split]

        params = {**XGB_PARAMS, "objective": "binary:logistic",
                  "eval_metric": "logloss", "num_class": None}
        params.pop("num_class", None)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_val, y_val)], verbose=False)
        return model

    def _rebuild_candle_base(self):
        """Trainiert Basis-Candle-Modell aus allen Symbol-CSVs (kombiniert)."""
        if len(self._candle_models) < 2:
            return
        history_dir = Path(os.environ.get("HISTORY_DIR", "data/history"))
        all_X, all_y = [], []
        for sym in list(self._candle_models.keys()):
            csv_path = history_dir / f"{sym}_15m.csv"
            if not csv_path.exists():
                continue
            try:
                klines = _load_csv_klines(str(csv_path))
                if len(klines) < 200:
                    continue
                labels = generate_strict_labels(klines)
                feats  = _compute_candle_features_batch(klines)
                valid  = ~np.isnan(feats).any(axis=1)
                all_X.append(feats[valid])
                all_y.append(labels[valid])
            except Exception as e:
                logger.warning(f"Base-Candle: {sym} übersprungen ({e})")
        if not all_X:
            return
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
        logger.info(f"Basis-Candle-Modell: {len(X)} Samples aus {len(all_X)} Symbolen")
        self._candle_base = self._train_candle_model(X, y)
        self._save_model(self._candle_base, "candle_base")
        logger.info("Basis-Candle-Modell gespeichert")

    # ────────────────── Persistenz ──────────────────

    def _save_model(self, model, name: str):
        if model is None:
            return
        path = _MODEL_DIR / f"{name}.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(model, f, protocol=4)
        except Exception as e:
            logger.warning(f"Modell-Save Fehler ({name}): {e}")

    def _load_models(self):
        """Lädt alle persistierten Modelle."""
        for prefix, store, attr in [
            ("candle_PF_", self._candle_models, None),
            ("win_PF_",    self._win_models,    None),
        ]:
            for pkl in _MODEL_DIR.glob(f"{prefix}*.pkl"):
                sym = pkl.stem[len(prefix.rstrip("_PF_")):]
                # Rekonstruiere echten Symbolnamen
                sym = "PF_" + pkl.stem.split("_", 2)[-1]
                try:
                    with open(pkl, "rb") as f:
                        obj = pickle.load(f)
                    if prefix.startswith("candle"):
                        self._candle_models[sym] = obj
                    else:
                        self._win_models[sym] = obj
                except Exception as e:
                    logger.warning(f"Modell-Load Fehler ({pkl.name}): {e}")

        for name, attr in [("candle_base", "_candle_base"), ("win_base", "_win_base")]:
            p = _MODEL_DIR / f"{name}.pkl"
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        setattr(self, attr, pickle.load(f))
                except Exception as e:
                    logger.warning(f"Base-Modell Fehler ({name}): {e}")

        logger.info(f"ML geladen: {len(self._candle_models)} Candle, "
                    f"{len(self._win_models)} Win, "
                    f"Basis-Candle={'ja' if self._candle_base else 'nein'}, "
                    f"Basis-Win={'ja' if self._win_base else 'nein'}")

    def get_status(self) -> dict:
        return {
            "candle_models": list(self._candle_models.keys()),
            "win_models":    list(self._win_models.keys()),
            "candle_base":   self._candle_base is not None,
            "win_base":      self._win_base is not None,
            "last_trade_id": self._last_trade_id,
        }


# ────────────────── CSV-Hilfsfunktionen ──────────────────

def _load_csv_klines(path: str) -> list:
    """Lädt CSV → Binance-kompatible 12-Element-Liste."""
    import csv
    klines = []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    ts = int(float(row[0])) if row[0].replace(".", "").isdigit() else 0
                    o, h, l, c, v = [float(x) for x in row[1:6]]
                    klines.append([ts, o, h, l, c, v, ts + 900_000, v, 0, 0, 0, 0])
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"CSV Ladefehler {path}: {e}")
    return klines


def _compute_candle_features_batch(klines: list) -> np.ndarray:
    """Berechnet 22 Features für alle Kerzen (vektorisiert)."""
    from scoring_core import _to_df, _rsi, _macd, _bb, _atr, _adx
    import pandas as pd

    df     = _to_df(klines)
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    vol    = df["volume"]
    n      = len(df)

    rsi7   = _rsi(close, 7)
    rsi14  = _rsi(close, 14)
    rsi21  = _rsi(close, 21)
    macd, sig = _macd(close)
    bb_u, bb_m, bb_l = _bb(close, 20, 2.0)
    atr_s  = _atr(high, low, close, 14)
    atr_avg = atr_s.rolling(50).mean()

    ema9   = close.ewm(span=9,  adjust=False).mean()
    ema21  = close.ewm(span=21, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()
    vol_avg = vol.rolling(20).mean()

    bb_range = bb_u - bb_l
    bb_pct   = ((close - bb_l) / bb_range.replace(0, np.nan)).fillna(0.5)
    bb_width = (bb_range / close.replace(0, np.nan)).fillna(0.0)
    atr_ratio = (atr_s / atr_avg.replace(0, np.nan)).fillna(1.0)
    vol_ratio = (vol / vol_avg.replace(0, np.nan)).fillna(1.0)

    ret1  = close.pct_change(1).fillna(0)
    ret4  = close.pct_change(4).fillna(0)
    ret8  = close.pct_change(8).fillna(0)
    ret16 = close.pct_change(16).fillna(0)

    rsi_slope = rsi14.diff(3).fillna(0) / 3

    # Zeit-Features aus Timestamps
    hour_sin = hour_cos = weekday_sin = weekday_cos = pd.Series(0.0, index=df.index)
    try:
        ts_col = pd.to_numeric(df["open_time"], errors="coerce")
        dt_idx = pd.to_datetime(ts_col, unit="ms", utc=True)
        hour_sin     = np.sin(2 * np.pi * dt_idx.dt.hour / 24)
        hour_cos     = np.cos(2 * np.pi * dt_idx.dt.hour / 24)
        weekday_sin  = np.sin(2 * np.pi * dt_idx.dt.dayofweek / 7)
        weekday_cos  = np.cos(2 * np.pi * dt_idx.dt.dayofweek / 7)
    except Exception:
        hour_sin = hour_cos = weekday_sin = weekday_cos = pd.Series(0.0, index=df.index)

    fg = pd.Series(50.0, index=df.index)  # F&G nicht in CSV → Platzhalter

    feat = np.column_stack([
        rsi7.values, rsi14.values, rsi21.values,
        (macd - sig).values, sig.values,
        (ema9 / ema21.replace(0, np.nan) - 1).fillna(0).values,
        (ema21 / ema50.replace(0, np.nan) - 1).fillna(0).values,
        (close / ema50.replace(0, np.nan) - 1).fillna(0).values,
        bb_pct.values, bb_width.values,
        atr_ratio.values, vol_ratio.values,
        ret1.values, ret4.values, ret8.values, ret16.values,
        np.array(hour_sin), np.array(hour_cos),
        np.array(weekday_sin), np.array(weekday_cos),
        fg.values, rsi_slope.values,
    ]).astype(np.float32)

    return feat


# Globale Instanz
ml_network = MLNetwork()
