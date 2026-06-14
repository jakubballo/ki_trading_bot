"""
validate_veto.py – Rückwirkende Veto-Wirkungsmessung auf network.db.

Fragt: "Wenn ML-Veto aktiv gewesen wäre – hätte es Verluste reduziert?"
Analysiert abgeschlossene Trades (real + shadow) nach Veto-Kriterien.

Aufruf:
  python validate_veto.py            # alle Daten
  python validate_veto.py --days 7   # letzte 7 Tage (Out-of-Sample)
  python validate_veto.py --bot 3    # nur Bot 3
"""

import argparse
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("validate_veto")


def get_veto_stats(rows: List[dict], veto_threshold: float = 0.42) -> dict:
    """
    Simuliert Veto-Wirkung auf eine Menge abgeschlossener Trades.
    Veto-Kriterium: block_reason LIKE '%veto%' ODER is_veto=1.

    Precision/Recall behandeln das Veto als Loser-Klassifikator:
      - Positive Klasse  = echter Verlierer (pnl <= 0)
      - Veto ausgelöst   = Modell sagt "Verlierer"
      - TP = Veto ausgelöst + war wirklich Verlierer (gut)
      - FP = Veto ausgelöst + war Gewinner (Veto falsch)
      - FN = Kein Veto    + war Verlierer (Veto hat ihn durchgelassen)
      - TN = Kein Veto    + war Gewinner  (richtig freigegeben)
    """
    total   = len(rows)
    if total == 0:
        return {}

    pnls_all   = [float(r["pnl"]) for r in rows if r.get("pnl") is not None]
    vetoed     = [r for r in rows if r.get("is_veto") or
                  (r.get("block_reason") or "").startswith("ml")]
    non_vetoed = [r for r in rows if r not in vetoed]

    pnl_all   = sum(pnls_all)
    pnl_vetoed = sum(float(r["pnl"]) for r in vetoed    if r.get("pnl") is not None)
    pnl_kept   = sum(float(r["pnl"]) for r in non_vetoed if r.get("pnl") is not None)

    wins_all  = sum(1 for p in pnls_all if p > 0)
    wins_kept = sum(1 for r in non_vetoed if (r.get("pnl") or 0) > 0)
    wr_all    = wins_all / total if total else 0
    wr_kept   = wins_kept / len(non_vetoed) if non_vetoed else 0

    # Veto als Loser-Klassifikator
    tp = sum(1 for r in vetoed    if (r.get("pnl") or 0) <= 0)  # richtig blockiert
    fp = sum(1 for r in vetoed    if (r.get("pnl") or 0)  > 0)  # fälschlich blockiert
    fn = sum(1 for r in non_vetoed if (r.get("pnl") or 0) <= 0) # Verlierer durchgelassen
    tn = sum(1 for r in non_vetoed if (r.get("pnl") or 0)  > 0) # richtig freigegeben

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # Profit Impact: wieviel PnL wurde durch das Veto gerettet (blockierte Verlierer)
    profit_impact = sum(abs(float(r["pnl"])) for r in vetoed
                        if (r.get("pnl") or 0) <= 0 and r.get("pnl") is not None)

    return {
        "total":          total,
        "vetoed":         len(vetoed),
        "kept":           len(non_vetoed),
        "pnl_all":        round(pnl_all, 4),
        "pnl_vetoed":     round(pnl_vetoed, 4),
        "pnl_kept":       round(pnl_kept, 4),
        "improvement":    round(pnl_kept - pnl_all, 4),
        "wr_all":         round(wr_all,  3),
        "wr_kept":        round(wr_kept, 3),
        "veto_wrong_pct": round(fp / max(len(vetoed), 1) * 100, 1),
        # Klassifikations-Metriken
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":      round(precision, 3),
        "recall":         round(recall, 3),
        "f1":             round(f1, 3),
        "profit_impact":  round(profit_impact, 4),
    }


def analyze_by_block_reason(rows: List[dict]) -> Dict[str, dict]:
    """Analysiert PnL gruppiert nach Block-Grund (welche Risk-Regel schadet?)."""
    by_reason: Dict[str, list] = {}
    for r in rows:
        reason = r.get("block_reason") or "live"
        by_reason.setdefault(reason, []).append(r)

    result = {}
    for reason, grp in sorted(by_reason.items(), key=lambda x: len(x[1]), reverse=True):
        pnls = [float(r["pnl"]) for r in grp if r.get("pnl") is not None]
        if not pnls:
            continue
        wins = sum(1 for p in pnls if p > 0)
        result[reason] = {
            "count":     len(pnls),
            "win_rate":  round(wins / len(pnls), 3),
            "total_pnl": round(sum(pnls), 4),
            "avg_pnl":   round(sum(pnls) / len(pnls), 4),
        }
    return result


def analyze_by_dimension(rows: List[dict], key: str) -> Dict[str, dict]:
    """Analysiert Veto-Wirkung pro Strategie oder Symbol."""
    by_dim: Dict[str, list] = {}
    for r in rows:
        dim = r.get(key) or "unknown"
        by_dim.setdefault(dim, []).append(r)

    result = {}
    for dim, grp in sorted(by_dim.items()):
        stats = get_veto_stats(grp)
        if stats:
            result[dim] = stats
    return result


def print_report(stats: dict, by_reason: dict, days: Optional[int], bot_id: Optional[int],
                 by_strategy: dict = None, by_symbol: dict = None):
    """Gibt vollständigen Veto-Validierungsbericht aus."""
    scope  = f"Bot {bot_id}" if bot_id else "Alle Bots"
    period = f"letzte {days} Tage" if days else "Gesamtzeitraum"
    print(f"\n{'='*60}")
    print(f"Veto-Validierung | {scope} | {period}")
    print(f"{'='*60}")

    if not stats:
        print("Keine Daten gefunden.")
        return

    print(f"Trades gesamt:      {stats['total']}")
    print(f"Davon veto'd:       {stats['vetoed']} ({stats['vetoed']/stats['total']*100:.1f}%)")
    print(f"Veto FP-Rate:       {stats['veto_wrong_pct']:.1f}% (Gewinner fälschlich blockiert)")
    print()
    print(f"PnL ALLE Trades:    {stats['pnl_all']:+.4f}")
    print(f"PnL OHNE Veto'd:    {stats['pnl_kept']:+.4f}")
    print(f"Verbesserung:       {stats['improvement']:+.4f}  "
          f"({'POSITIV - Veto hilft' if stats['improvement'] > 0 else 'NEGATIV - Veto schadet'})")
    print(f"Profit Impact:      {stats['profit_impact']:+.4f}  (geblockte Verlierer-PnL gerettet)")
    print()
    print(f"Win-Rate ALLE:      {stats['wr_all']*100:.1f}%")
    print(f"Win-Rate OHNE:      {stats['wr_kept']*100:.1f}%")
    print()
    print(f"Veto als Loser-Klassifikator:")
    print(f"  TP={stats['tp']}  FP={stats['fp']}  FN={stats['fn']}  TN={stats['tn']}")
    print(f"  Precision: {stats['precision']:.3f}  "
          f"Recall: {stats['recall']:.3f}  "
          f"F1: {stats['f1']:.3f}")
    print(f"  (Precision = wie oft Veto richtig lag | Recall = wie viele Verlierer gefunden)")

    if by_reason:
        print(f"\n{'─'*60}")
        print("PnL nach Block-Grund:")
        print(f"{'Grund':<25} {'Trades':>7} {'WR':>7} {'∅PnL':>10} {'∑PnL':>10}")
        print(f"{'─'*60}")
        for reason, r in sorted(by_reason.items(), key=lambda x: x[1]["total_pnl"]):
            print(f"{reason[:24]:<25} {r['count']:>7} "
                  f"{r['win_rate']*100:>6.0f}% {r['avg_pnl']:>+10.4f} {r['total_pnl']:>+10.4f}")

    if by_strategy:
        print(f"\n{'─'*60}")
        print("Veto-Wirkung nach Strategie:")
        print(f"{'Strategie':<18} {'Ges':>5} {'Veto':>5} {'WR→':>7} {'P':>6} {'R':>6} {'F1':>6} {'Δ PnL':>10}")
        print(f"{'─'*60}")
        for strat, s in sorted(by_strategy.items()):
            wr_delta = (s["wr_kept"] - s["wr_all"]) * 100
            print(f"{strat:<18} {s['total']:>5} {s['vetoed']:>5} "
                  f"{s['wr_kept']*100:>6.0f}% "
                  f"{s['precision']:>6.2f} {s['recall']:>6.2f} {s['f1']:>6.2f} "
                  f"{s['improvement']:>+10.4f}")

    if by_symbol:
        print(f"\n{'─'*60}")
        print("Veto-Wirkung nach Symbol:")
        print(f"{'Symbol':<16} {'Ges':>5} {'Veto':>5} {'WR→':>7} {'P':>6} {'R':>6} {'F1':>6} {'Δ PnL':>10}")
        print(f"{'─'*60}")
        for sym, s in sorted(by_symbol.items()):
            print(f"{sym:<16} {s['total']:>5} {s['vetoed']:>5} "
                  f"{s['wr_kept']*100:>6.0f}% "
                  f"{s['precision']:>6.2f} {s['recall']:>6.2f} {s['f1']:>6.2f} "
                  f"{s['improvement']:>+10.4f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None,
                        help="Nur Trades der letzten N Tage analysieren")
    parser.add_argument("--bot",  type=int, default=None,
                        help="Nur Trades von Bot ID")
    args = parser.parse_args()

    from network_db import get_connection

    conn = get_connection()
    query = """
        SELECT bot_id, symbol, side, pnl, exit_reason, block_reason, is_veto,
               is_shadow, is_synthetic, regime, strategy, opened_at, closed_at
        FROM trades_network
        WHERE exit_price IS NOT NULL AND pnl IS NOT NULL
    """
    params = []

    if args.days:
        cutoff = (datetime.now(timezone.utc) -
                  timedelta(days=args.days)).isoformat()
        query  += " AND closed_at >= ?"
        params.append(cutoff)

    if args.bot:
        query  += " AND bot_id = ?"
        params.append(args.bot)

    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()

    logger.info(f"Analysiere {len(rows)} abgeschlossene Trades...")

    stats       = get_veto_stats(rows)
    shadow_rows = [r for r in rows if r.get("is_shadow") or r.get("is_veto")]
    by_reason   = analyze_by_block_reason(shadow_rows)
    by_strategy = analyze_by_dimension(rows, "strategy")
    by_symbol   = analyze_by_dimension(rows, "symbol")
    print_report(stats, by_reason, args.days, args.bot,
                 by_strategy=by_strategy, by_symbol=by_symbol)
