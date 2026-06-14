"""
test_macro_gate.py – Unit-Tests für Makro-Richtungs-Prüfung im Risk-Gate.

Testet alle Kombinationen von order_side × macro_direction × macro_mode.
Verhindert Regressions-Bugs beim BUY/SELL vs long/short Mapping.

Aufruf:
  python test_macro_gate.py
  python -m pytest test_macro_gate.py -v
"""

import sys
import types
import unittest


# ─── Minimal-Stubs damit risk_gate.py importierbar ist ───────────────────────

def _make_stubs():
    """Erzeugt minimale Config/State-Stubs."""
    cfg = types.SimpleNamespace(
        risk={
            "daily_loss_limit_pct": 0.03,
            "max_atr_ratio": 3.0,
            "max_funding_rate": 0.0005,
            "max_consecutive_negative_weeks": 3,
        },
        bot_id=1,
    )
    cfg_module = types.ModuleType("config")
    cfg_module.config = cfg
    sys.modules.setdefault("config", cfg_module)


_make_stubs()

from risk_gate import check_all, MarketData, SymbolFilters  # noqa: E402


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def _state(open_pos=False, daily_loss=0.0, neg_weeks=0):
    """Minimaler State-Stub der alle Risk-Checks außer macro_direction besteht."""
    pos = types.SimpleNamespace(symbol="BTC" if open_pos else None)
    daily = types.SimpleNamespace(loss_pct_of_capital=daily_loss)
    weekly = types.SimpleNamespace(consecutive_negative_weeks=neg_weeks)
    return types.SimpleNamespace(
        open_position=pos,
        daily=daily,
        weekly=weekly,
    )


def _market(atr_ratio=1.0, funding=0.0001):
    return MarketData(atr=100.0, atr_ratio=atr_ratio,
                      funding_rate=funding, mark_price=50000.0)


def _filters():
    return SymbolFilters(min_qty=0.001, min_notional=1.0, step_size=0.001)


def _check_macro(order_side: str, macro_direction: str, macro_mode: str) -> bool:
    """Führt nur den Macro-Check durch (alle anderen Checks bewusst passierbar)."""
    passed, reason = check_all(
        state_ref=_state(),
        market=_market(),
        symbol_filters=_filters(),
        calculated_qty=0.01,
        price=50000.0,
        order_side=order_side,
        macro_direction=macro_direction,
        macro_mode=macro_mode,
    )
    # Wenn geblockt und NICHT wegen macro_direction → Fehler im Test-Setup
    if not passed and "Makro-Richtung" not in (reason or ""):
        raise RuntimeError(f"Unerwarteter Block durch anderen Check: {reason}")
    return passed


# ─── Test-Klassen ─────────────────────────────────────────────────────────────

class TestMacroFilterMode(unittest.TestCase):
    """Macro-Modus 'filter': BUY muss mit 'long', SELL mit 'short' übereinstimmen."""

    def test_buy_long_allowed(self):
        """BUY + Makro=long → erlaubt."""
        self.assertTrue(_check_macro("BUY", "long", "filter"),
                        "BUY bei Makro=long muss erlaubt sein")

    def test_sell_short_allowed(self):
        """SELL + Makro=short → erlaubt."""
        self.assertTrue(_check_macro("SELL", "short", "filter"),
                        "SELL bei Makro=short muss erlaubt sein")

    def test_buy_short_blocked(self):
        """BUY + Makro=short → blockiert."""
        self.assertFalse(_check_macro("BUY", "short", "filter"),
                         "BUY bei Makro=short muss blockiert sein")

    def test_sell_long_blocked(self):
        """SELL + Makro=long → blockiert."""
        self.assertFalse(_check_macro("SELL", "long", "filter"),
                         "SELL bei Makro=long muss blockiert sein")

    def test_buy_both_allowed(self):
        """BUY + Makro=both (neutral) → erlaubt."""
        self.assertTrue(_check_macro("BUY", "both", "filter"),
                        "BUY bei Makro=both muss erlaubt sein")

    def test_sell_both_allowed(self):
        """SELL + Makro=both (neutral) → erlaubt."""
        self.assertTrue(_check_macro("SELL", "both", "filter"),
                        "SELL bei Makro=both muss erlaubt sein")


class TestMacroBothMode(unittest.TestCase):
    """Macro-Modus 'both': alle Trades erlaubt."""

    def test_buy_long_allowed(self):
        self.assertTrue(_check_macro("BUY", "long", "both"))

    def test_buy_short_allowed(self):
        self.assertTrue(_check_macro("BUY", "short", "both"))

    def test_sell_long_allowed(self):
        self.assertTrue(_check_macro("SELL", "long", "both"))

    def test_sell_short_allowed(self):
        self.assertTrue(_check_macro("SELL", "short", "both"))


class TestMacroInvertMode(unittest.TestCase):
    """Macro-Modus 'invert': alle Trades erlaubt (Inversion erfolgt im Scoring)."""

    def test_buy_long_allowed(self):
        self.assertTrue(_check_macro("BUY", "long", "invert"))

    def test_buy_short_allowed(self):
        self.assertTrue(_check_macro("BUY", "short", "invert"))

    def test_sell_long_allowed(self):
        self.assertTrue(_check_macro("SELL", "long", "invert"))

    def test_sell_short_allowed(self):
        self.assertTrue(_check_macro("SELL", "short", "invert"))


class TestMacroCaseInsensitivity(unittest.TestCase):
    """order_side muss case-insensitiv funktionieren."""

    def test_lowercase_buy(self):
        self.assertTrue(_check_macro("buy", "long", "filter"))

    def test_lowercase_sell(self):
        self.assertTrue(_check_macro("sell", "short", "filter"))

    def test_mixed_case_buy(self):
        self.assertTrue(_check_macro("Buy", "long", "filter"))

    def test_buy_wrong_direction_case(self):
        self.assertFalse(_check_macro("buy", "short", "filter"))


class TestRegressionBugBuySellVsLongShort(unittest.TestCase):
    """
    Regression-Test für den spezifischen Bug:
    order_side.upper() == macro_direction.upper()
    → 'BUY' == 'LONG' war immer False.
    → 'SELL' == 'SHORT' war immer False.
    → Damit wurden auch korrekt ausgerichtete Trades blockiert.
    """

    def test_sell_short_not_blocked_regression(self):
        """Regression: SELL + Makro=short wurde fälschlich blockiert (Log: 19:00, 19:30 Uhr)."""
        result = _check_macro("SELL", "short", "filter")
        self.assertTrue(result,
                        "REGRESSION: SELL bei Makro=short darf nicht blockiert werden. "
                        "War Bug durch 'SELL' == 'SHORT' → False.")

    def test_buy_long_not_blocked_regression(self):
        """Regression: BUY + Makro=long wurde fälschlich blockiert."""
        result = _check_macro("BUY", "long", "filter")
        self.assertTrue(result,
                        "REGRESSION: BUY bei Makro=long darf nicht blockiert werden. "
                        "War Bug durch 'BUY' == 'LONG' → False.")


# ─── Alle 16 Kombinationen als Matrix-Test ───────────────────────────────────

class TestAllCombinationsMatrix(unittest.TestCase):
    """Vollständige Wahrheitsmatrix aller sinnvollen Kombinationen."""

    # (order_side, macro_direction, macro_mode, expected_pass)
    MATRIX = [
        # filter-Modus
        ("BUY",  "long",  "filter", True),   # ausgerichtet → erlaubt
        ("BUY",  "short", "filter", False),  # gegenläufig → blockiert
        ("BUY",  "both",  "filter", True),   # neutral → erlaubt
        ("SELL", "long",  "filter", False),  # gegenläufig → blockiert
        ("SELL", "short", "filter", True),   # ausgerichtet → erlaubt
        ("SELL", "both",  "filter", True),   # neutral → erlaubt
        # both-Modus
        ("BUY",  "long",  "both",   True),
        ("BUY",  "short", "both",   True),
        ("SELL", "long",  "both",   True),
        ("SELL", "short", "both",   True),
        # invert-Modus
        ("BUY",  "long",  "invert", True),
        ("BUY",  "short", "invert", True),
        ("SELL", "long",  "invert", True),
        ("SELL", "short", "invert", True),
    ]

    def test_matrix(self):
        failures = []
        for order_side, macro_dir, mode, expected in self.MATRIX:
            result = _check_macro(order_side, macro_dir, mode)
            if result != expected:
                status = "ERLAUBT" if result else "BLOCKIERT"
                soll   = "ERLAUBT" if expected else "BLOCKIERT"
                failures.append(
                    f"  {order_side:4} + Makro={macro_dir:5} + Modus={mode:6} "
                    f"→ {status} (erwartet: {soll})"
                )

        if failures:
            self.fail("Folgende Kombinationen sind fehlerhaft:\n" + "\n".join(failures))


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [
        TestMacroFilterMode,
        TestMacroBothMode,
        TestMacroInvertMode,
        TestMacroCaseInsensitivity,
        TestRegressionBugBuySellVsLongShort,
        TestAllCombinationsMatrix,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
