"""
shadow_tracker.py – Verfolgt blockierte Signale als virtuelle Shadow-Trades.
Lernen darf NIE stoppen – auch wenn Risk-Limits das Trading pausieren.
Outcomes werden in network.db geschrieben (is_shadow=True).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShadowTrade:
    trade_id: int              # ID in network.db
    bot_id: int
    symbol: str
    side: str                  # "BUY" | "SELL"
    entry_price: float
    sl_price: float
    tp_price: float
    score: int
    regime: str
    block_reason: str
    is_veto: bool
    funding_rate: float
    fg_index: float
    rsi: float
    opened_at: float = field(default_factory=lambda: __import__("time").time())
    closed: bool = False


class ShadowTracker:
    """
    Verwaltet offene Shadow-Trades im RAM.
    Beim Neustart: cleanup_orphans() markiert verwaiste Shadows.
    Deduplizierung: pro Symbol+Richtung maximal 1 offener Shadow.
    """

    def __init__(self):
        self._open: Dict[str, ShadowTrade] = {}  # key: f"{symbol}_{side}"
        self._lock = asyncio.Lock()

    def register(
        self,
        bot_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        score: int,
        regime: str,
        block_reason: str,
        is_veto: bool = False,
        funding_rate: float = 0.0,
        fg_index: float = 50.0,
        rsi: float = 50.0,
        details: dict = None,
    ):
        """
        Registriert ein blockiertes Signal als Shadow-Trade.
        Deduplizierung: gleiche Symbol+Richtung ersetzt bestehenden Shadow.
        """
        key = f"{symbol}_{side}"

        # Bestehenden Shadow schließen (ersetzt)
        existing = self._open.get(key)
        if existing and not existing.closed:
            logger.debug(f"Shadow dedupe: {key} – alten Shadow ersetzt")

        d = details or {}
        try:
            from network_db import log_network_trade
            from config import config
            trade_id = log_network_trade(
                bot_id=bot_id,
                symbol=symbol,
                side=side,
                entry=entry_price,
                exit_price=None,
                pnl=None,
                exit_reason=None,
                score=score,
                regime=regime,
                funding_rate=funding_rate,
                rsi=rsi,
                fg_index=fg_index,
                strategy=config.strategy,
                is_shadow=True,
                block_reason=block_reason,
                is_veto=is_veto,
                macd_diff=float(d.get("_macd_diff",       0)),
                macd_signal_val=float(d.get("_macd_signal",   0)),
                ema_ratio_9_21=float(d.get("_ema_ratio_9_21",  0)),
                ema_ratio_21_50=float(d.get("_ema_ratio_21_50", 0)),
                price_vs_ema50=float(d.get("_price_vs_ema50",  0)),
                bb_pct=float(d.get("_bb_pct",          0.5)),
                bb_width=float(d.get("_bb_width",        0)),
                vol_ratio=float(d.get("_vol_ratio",       1.0)),
                rsi_slope=float(d.get("_rsi_slope",       0)),
                ret_1=float(d.get("_ret_1",  0)),
                ret_4=float(d.get("_ret_4",  0)),
                ret_8=float(d.get("_ret_8",  0)),
                ret_16=float(d.get("_ret_16", 0)),
            )
        except Exception as e:
            logger.debug(f"Shadow-Trader: network_db nicht erreichbar: {e}")
            trade_id = -1

        shadow = ShadowTrade(
            trade_id=trade_id,
            bot_id=bot_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            regime=regime,
            block_reason=block_reason,
            is_veto=is_veto,
            funding_rate=funding_rate,
            fg_index=fg_index,
            rsi=rsi,
        )
        self._open[key] = shadow
        logger.debug(f"Shadow-Trade registriert: {side} {symbol} @ {entry_price:.4f} "
                     f"(Grund: {block_reason})")

    def update_prices(self, symbol: str, mark_price: float):
        """
        Prüft ob ein Shadow-Trade seinen SL oder TP erreicht hat.
        Wird vom WebSocket-Manager bei jedem Ticker-Update aufgerufen.
        """
        for key in list(self._open.keys()):
            shadow = self._open[key]
            if shadow.symbol != symbol or shadow.closed:
                continue

            is_long  = shadow.side == "BUY"
            sl_hit   = (is_long and mark_price <= shadow.sl_price) or \
                       (not is_long and mark_price >= shadow.sl_price)
            tp_hit   = (is_long and mark_price >= shadow.tp_price) or \
                       (not is_long and mark_price <= shadow.tp_price)

            if sl_hit or tp_hit:
                exit_price  = shadow.sl_price if sl_hit else shadow.tp_price
                exit_reason = "sl" if sl_hit else "tp"
                self._close_shadow(shadow, exit_price, exit_reason)
                del self._open[key]

    def _close_shadow(self, shadow: ShadowTrade, exit_price: float, exit_reason: str):
        """Schreibt das Shadow-Outcome in network.db."""
        shadow.closed = True
        is_long = shadow.side == "BUY"

        # PnL ohne Leverage (nur Preisdifferenz)
        if is_long:
            pnl = shadow.entry_price - exit_price
            pnl = (exit_price - shadow.entry_price) / shadow.entry_price
        else:
            pnl = (shadow.entry_price - exit_price) / shadow.entry_price

        # Gebühren & Slippage abziehen
        try:
            from config import config
            fee_rate = config.risk.get("fee_taker", 0.0005) + config.risk.get("fee_slippage", 0.0002)
        except Exception:
            fee_rate = 0.0007
        pnl_net = pnl - fee_rate * 2  # Einstieg + Ausstieg

        logger.info(f"Shadow-Trade geschlossen: {shadow.side} {shadow.symbol} "
                    f"Entry={shadow.entry_price:.4f} Exit={exit_price:.4f} "
                    f"→ {exit_reason} | pnl={pnl_net:.4f} | Grund: {shadow.block_reason}")

        if shadow.trade_id > 0:
            try:
                from network_db import update_network_trade_outcome
                update_network_trade_outcome(
                    trade_id=shadow.trade_id,
                    exit_price=exit_price,
                    pnl=pnl_net,
                    exit_reason=exit_reason,
                )
            except Exception as e:
                logger.error(f"Shadow-Outcome schreiben fehlgeschlagen: {e}")

    def cleanup_orphans(self):
        """
        Markiert verwaiste Shadow-Trades (aus dem letzten Bot-Lauf) als 'orphaned'.
        Wird beim Bot-Start aufgerufen.
        """
        if not self._open:
            return
        count = len(self._open)
        try:
            from network_db import get_connection
            ids = [s.trade_id for s in self._open.values()
                   if s.trade_id > 0 and not s.closed]
            if ids:
                conn = get_connection()
                conn.execute(
                    f"UPDATE trades_network SET exit_reason='orphaned' "
                    f"WHERE id IN ({','.join('?' * len(ids))}) "
                    f"AND exit_price IS NULL",
                    ids
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.warning(f"Orphan-Cleanup Fehler: {e}")
        self._open.clear()
        logger.info(f"Shadow-Tracker: {count} verwaiste Shadows bereinigt")

    @property
    def open_count(self) -> int:
        return len(self._open)


# Globale Instanz
shadow_tracker = ShadowTracker()
