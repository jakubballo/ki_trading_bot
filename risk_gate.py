"""
risk_gate.py – Sequentielle Risk-Checks vor jedem Trade.
Alle 7 Checks müssen True zurückgeben, sonst wird kein Trade eröffnet.
"""

import logging
from typing import Optional, Tuple

from config import config

logger = logging.getLogger(__name__)


class MarketData:
    """Hilfsklasse für Marktdaten die an den Risk-Gate übergeben werden."""

    def __init__(self, atr: float, atr_ratio: float, funding_rate: float,
                 mark_price: float):
        self.atr = atr
        self.atr_ratio = atr_ratio          # Aktueller ATR / Durchschnittlicher ATR
        self.funding_rate = funding_rate
        self.mark_price = mark_price


class SymbolFilters:
    """Hilfsklasse für Exchange-Symbol-Filter."""

    def __init__(self, min_qty: float, min_notional: float, step_size: float):
        self.min_qty = min_qty
        self.min_notional = min_notional
        self.step_size = step_size


def check_all(state_ref, market: MarketData, symbol_filters: SymbolFilters,
              calculated_qty: float, price: float,
              order_side: str, macro_direction: str,
              macro_mode: str = "filter") -> Tuple[bool, Optional[str]]:
    """
    Führt alle 7 Risk-Checks sequentiell durch.
    
    Returns:
        (True, None) wenn alle Checks bestanden
        (False, "check_name") wenn ein Check fehlschlägt
    """

    # Schwellen aus der (normalisierten) Config – nicht mehr hartkodiert.
    # max_funding stammt aus dem per-Bot funding_rate_limit (PBT-mutierbar).
    max_daily_loss = config.risk.get("daily_loss_limit_pct", 0.03)
    max_atr_ratio  = config.risk.get("max_atr_ratio", 3.0)
    max_funding    = config.risk.get("max_funding_rate", 0.0005)
    max_neg_weeks  = config.risk.get("max_consecutive_negative_weeks", 3)

    checks = [
        # 1. Tägliches Verlust-Limit noch nicht erreicht
        (
            "daily_loss_limit",
            lambda: state_ref.daily.loss_pct_of_capital < max_daily_loss,
            f"Tägliches Verlust-Limit erreicht: "
            f"{state_ref.daily.loss_pct_of_capital * 100:.2f}% >= {max_daily_loss * 100:.1f}%"
        ),

        # 2. Keine offene Position
        (
            "open_position",
            lambda: state_ref.open_position.symbol is None,
            f"Es ist bereits eine Position offen: {state_ref.open_position.symbol}"
        ),

        # 3. Keine extreme Volatilität
        (
            "extreme_volatility",
            lambda: market.atr_ratio <= max_atr_ratio,
            f"Extreme Volatilität erkannt: ATR-Ratio = {market.atr_ratio:.2f} > {max_atr_ratio}"
        ),

        # 4. Mindest-Positionsgröße wird erreicht
        (
            "min_position_size",
            lambda: (
                calculated_qty >= symbol_filters.min_qty and
                calculated_qty * price >= symbol_filters.min_notional
            ),
            f"Positionsgröße zu klein: qty={calculated_qty} (min={symbol_filters.min_qty}), "
            f"notional={calculated_qty * price:.2f} (min={symbol_filters.min_notional})"
        ),

        # 5. Makro-Richtung stimmt überein
        #    filter: BUY muss mit "long" übereinstimmen, SELL mit "short"
        #    both:   immer erlaubt (bot handelt beide Seiten)
        #    invert: contrarian bots – immer erlaubt (Inversion erfolgt im Scoring)
        (
            "macro_direction",
            lambda: (
                macro_mode in ("both", "invert") or
                macro_direction in ("both", "neutral") or
                (order_side.upper() == "BUY"  and macro_direction == "long") or
                (order_side.upper() == "SELL" and macro_direction == "short")
            ),
            f"Makro-Richtung blockiert: Signal={order_side}, Makro={macro_direction}, Modus={macro_mode}"
        ),

        # 6. Funding-Rate nicht zu hoch (per-Bot funding_rate_limit)
        (
            "funding_rate",
            lambda: abs(market.funding_rate) <= max_funding,
            f"Funding-Rate zu hoch: {market.funding_rate:.4%} > {max_funding:.4%}"
        ),

        # 7. Nicht zu viele negative Wochen in Folge
        (
            "weekly_stop",
            lambda: state_ref.weekly.consecutive_negative_weeks < max_neg_weeks,
            f"Wochenstop aktiv: {state_ref.weekly.consecutive_negative_weeks} negative Wochen in Folge (max {max_neg_weeks})"
        ),
    ]

    for check_name, check_fn, fail_message in checks:
        try:
            passed = check_fn()
        except Exception as e:
            logger.error(f"Fehler beim Risk-Check '{check_name}': {e}")
            passed = False
            fail_message = f"Fehler im Check: {e}"

        if not passed:
            logger.info(f"Risk-Gate BLOCKIERT [{check_name}]: {fail_message}")
            return False, fail_message

    logger.debug("Risk-Gate: Alle Checks bestanden ✓")
    return True, None
