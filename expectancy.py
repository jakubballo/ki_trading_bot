"""
expectancy.py – Edge-Messung pro Strategie x Symbol aus network.db (Session 10).

Beantwortet objektiv "Hat dieser Bot/diese Strategie einen Vorteil?" als EINE Zahl:

    Expectancy = (Trefferquote x Ø-Gewinn) − (Verlustquote x Ø-Verlust)

= durchschnittliche NETTO-Rendite pro Trade (nach Gebühren). > 0 = Edge, < 0 = kein Edge.

WICHTIG (Einheiten): echte Trades speichern pnl in USD, Shadows als Bruchteil — nicht
vergleichbar. Daher wird für ALLE Trades die size-unabhängige Preis-Rendite aus
entry/exit_price gerechnet (BUY: (exit-entry)/entry, SELL: (entry-exit)/entry) und eine
einheitliche Round-Trip-Gebühr abgezogen. Verifiziert: Shadow-Gross − Fee == gespeicherter
Shadow-pnl. So sind echte + Shadow-Outcomes in einer Kennzahl vergleichbar.

Dieses Tool ERZEUGT KEINEN Edge und BLOCKT NICHTS — es misst nur, ob einer da ist
(Weg 2 / nicht-blockierende Mess-Schicht; hindert Modell B nicht am Lernen). Vorstufe BossBot.

Aufruf (auf der VPS, gegen die Live-DB):
  .venv/bin/python expectancy.py                # alle Outcomes (echt + shadow)
  .venv/bin/python expectancy.py --days 7       # letzte 7 Tage
  .venv/bin/python expectancy.py --real-only    # nur echte Trades
  .venv/bin/python expectancy.py --min-n 100    # Urteils-Schwelle (Default 100)
"""

import sys
# Defensive: Box-/Sonderzeichen sicher ausgeben (Windows-cp1252-Konsole). Auf UTF-8/VPS no-op.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
from datetime import datetime, timezone, timedelta

FEE = 0.0007 * 2          # Round-Trip-Gebühr (identisch zu walkforward.py)
MIN_N_DEFAULT = 100       # n darunter => "noch kein Urteil" (Schutz vor Rausch-Edge)
# pnl=0-Artefakte ohne echtes Outcome (würden als Verlust verzerren)
SKIP_REASONS = {"dedup_replaced", "orphaned"}


def net_return(side: str, entry: float, exit_price: float) -> float:
    """Size-unabhängige Netto-Rendite eines Trades (nach Gebühren)."""
    if not entry or entry <= 0 or exit_price is None:
        return None
    gross = (exit_price - entry) / entry
    if (side or "").upper() == "SELL":
        gross = -gross
    return gross - FEE


def compute(rows):
    """rows -> {(strategy,symbol): {n, win_rate, avg_win, avg_loss, payoff, expectancy}}"""
    buckets = {}
    for r in rows:
        if (r.get("exit_reason") or "") in SKIP_REASONS:
            continue
        ret = net_return(r.get("side"), r.get("entry"), r.get("exit_price"))
        if ret is None:
            continue
        key = (r.get("strategy") or "?", r.get("symbol") or "?")
        buckets.setdefault(key, []).append(ret)

    out = {}
    for key, rets in buckets.items():
        n = len(rets)
        wins = [x for x in rets if x > 0]
        losses = [x for x in rets if x <= 0]
        win_rate = len(wins) / n if n else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (abs(sum(losses) / len(losses))) if losses else 0.0   # positive Größe
        payoff = (avg_win / avg_loss) if avg_loss > 0 else float("inf")
        expectancy = sum(rets) / n     # == win_rate*avg_win - (1-win_rate)*avg_loss
        out[key] = {"n": n, "win_rate": win_rate, "avg_win": avg_win,
                    "avg_loss": avg_loss, "payoff": payoff, "expectancy": expectancy}
    return out


def _verdict(s, min_n):
    if s["n"] < min_n:
        return "? n zu klein"
    if s["expectancy"] > 0:
        return "+ EDGE"
    return "- kein Edge"


def _fmt_payoff(p):
    return "inf" if p == float("inf") else f"{p:.2f}"


def print_table(title, agg, min_n):
    print(f"\n{'='*96}")
    print(title)
    print('-'*96)
    print(f"{'Strategie':<16}{'Symbol':<13}{'n':>6}{'WR':>7}{'Ø-Gew%':>9}{'Ø-Verl%':>9}"
          f"{'CRV':>6}{'Expect%':>10}  Urteil")
    print('-'*96)
    # sortiert: erst belastbare (n>=min_n) nach Expectancy absteigend, dann Rest
    def sort_key(item):
        (strat, sym), s = item
        return (0 if s["n"] >= min_n else 1, -s["expectancy"])
    for (strat, sym), s in sorted(agg.items(), key=sort_key):
        print(f"{strat:<16}{sym:<13}{s['n']:>6}{s['win_rate']*100:>6.1f}%"
              f"{s['avg_win']*100:>+8.2f}%{s['avg_loss']*100:>+8.2f}%"
              f"{_fmt_payoff(s['payoff']):>6}{s['expectancy']*100:>+9.3f}%  {_verdict(s, min_n)}")
    print('='*96)


def aggregate_by_strategy(rows, min_n):
    """Aggregiert über alle Symbole je Strategie (eine Edge-Zahl pro Strategie)."""
    by = {}
    for r in rows:
        if (r.get("exit_reason") or "") in SKIP_REASONS:
            continue
        ret = net_return(r.get("side"), r.get("entry"), r.get("exit_price"))
        if ret is None:
            continue
        by.setdefault(r.get("strategy") or "?", []).append(ret)
    print(f"\n{'='*70}")
    print("EDGE JE STRATEGIE (über alle Symbole)")
    print('-'*70)
    print(f"{'Strategie':<16}{'n':>7}{'WR':>8}{'Expect%':>11}  Urteil")
    print('-'*70)
    rowsout = []
    for strat, rets in by.items():
        n = len(rets); wr = sum(1 for x in rets if x > 0)/n if n else 0
        exp = sum(rets)/n if n else 0
        rowsout.append((strat, n, wr, exp))
    for strat, n, wr, exp in sorted(rowsout, key=lambda t: (0 if t[1] >= min_n else 1, -t[3])):
        verdict = "? n zu klein" if n < min_n else ("+ EDGE" if exp > 0 else "- kein Edge")
        print(f"{strat:<16}{n:>7}{wr*100:>7.1f}%{exp*100:>+10.3f}%  {verdict}")
    print('='*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge/Expectancy pro Strategie x Symbol")
    parser.add_argument("--days", type=int, default=None, help="nur letzte N Tage")
    parser.add_argument("--real-only", action="store_true", help="nur echte Trades (kein Shadow)")
    parser.add_argument("--min-n", type=int, default=MIN_N_DEFAULT,
                        help=f"Urteils-Schwelle (Default {MIN_N_DEFAULT})")
    args = parser.parse_args()

    from network_db import get_connection
    conn = get_connection()
    query = """
        SELECT strategy, symbol, side, entry, exit_price, pnl, exit_reason,
               is_shadow, is_synthetic, closed_at
        FROM trades_network
        WHERE exit_price IS NOT NULL AND is_synthetic = 0
    """
    params = []
    if args.real_only:
        query += " AND is_shadow = 0"
    if args.days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        query += " AND closed_at >= ?"
        params.append(cutoff)

    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()

    scope = "echte Trades" if args.real_only else "echte + Shadow-Outcomes"
    period = f"letzte {args.days} Tage" if args.days else "Gesamtzeitraum"
    print(f"\nExpectancy-Tracker | {scope} | {period} | {len(rows)} Outcomes | "
          f"Urteils-Schwelle n>={args.min_n} | Netto nach {FEE*100:.2f}% Fee")

    agg = compute(rows)
    if not agg:
        print("Keine auswertbaren Outcomes gefunden.")
        sys.exit(0)

    aggregate_by_strategy(rows, args.min_n)
    print_table("EDGE JE STRATEGIE x SYMBOL", agg, args.min_n)

    judged = [s for s in agg.values() if s["n"] >= args.min_n]
    pos = sum(1 for s in judged if s["expectancy"] > 0)
    print(f"\nFazit: {len(judged)} von {len(agg)} Kombis belastbar (n>={args.min_n}), "
          f"davon {pos} mit positivem Edge. "
          f"{len(agg)-len(judged)} Kombis noch ohne Urteil (zu wenig Trades).")
