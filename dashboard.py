"""
dashboard.py – Generiert data/dashboard.html mit Equity-Kurven + Statistik.

Wird stündlich vom Brain Bot aufgerufen.
Öffne data/dashboard.html im Browser für das Live-Dashboard.

Aufruf:
  python dashboard.py
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("dashboard")

OUTPUT_PATH = Path("data/dashboard.html")


def _sharpe(pnls: list) -> float:
    """Einfache Sharpe-Ratio aus einer PnL-Liste (0.0 bei <5 Werten)."""
    if len(pnls) < 5:
        return 0.0
    avg = sum(pnls) / len(pnls)
    std = (sum((p - avg) ** 2 for p in pnls) / len(pnls)) ** 0.5
    return (avg / std) if std > 0 else 0.0


def _aggregate_per_bot(rows: list) -> list:
    """
    Aggregiert geschlossene Trades zu Pro-Bot-Statistiken.
    `rows` sind sqlite-Rows mit bot_id, symbol, strategy, pnl, closed_at.
    Rückgabe: Liste von Bot-Dicts, sortiert nach Gesamt-PnL (absteigend).
    """
    per_bot: dict = defaultdict(lambda: {
        "pnls": [], "symbol": "", "strategy": "",
    })
    for r in rows:
        b = per_bot[r["bot_id"]]
        b["pnls"].append(float(r["pnl"]))
        b["symbol"] = r["symbol"] or ""
        b["strategy"] = r["strategy"] or ""

    out = []
    for bid, b in per_bot.items():
        pnls = b["pnls"]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        out.append({
            "bot_id":    bid,
            "symbol":    b["symbol"],
            "strategy":  b["strategy"],
            "total":     n,
            "wins":      wins,
            "win_rate":  (wins / n) if n else 0.0,
            "total_pnl": round(sum(pnls), 4),
            "avg_pnl":   round(sum(pnls) / n, 4) if n else 0.0,
            "best":      round(max(pnls), 4) if pnls else 0.0,
            "worst":     round(min(pnls), 4) if pnls else 0.0,
            "sharpe":    round(_sharpe(pnls), 3),
        })
    return sorted(out, key=lambda x: x["total_pnl"], reverse=True)


def load_network_data() -> dict:
    """Lädt alle relevanten Daten aus network.db."""
    from network_db import get_connection

    conn = get_connection()

    # Alle geschlossenen, nicht-synthetischen Trades einmal laden, dann in
    # Python nach echt (is_shadow=0) vs. shadow (is_shadow=1) gruppieren.
    all_rows = conn.execute("""
        SELECT bot_id, symbol, strategy, pnl, is_shadow, closed_at
        FROM trades_network
        WHERE exit_price IS NOT NULL AND pnl IS NOT NULL
          AND is_synthetic = 0
        ORDER BY closed_at
    """).fetchall()

    real_rows   = [r for r in all_rows if r["is_shadow"] == 0]
    shadow_rows = [r for r in all_rows if r["is_shadow"] == 1]

    real_bots   = _aggregate_per_bot(real_rows)
    shadow_bots = _aggregate_per_bot(shadow_rows)

    # Equity-Kurven pro Bot (kumulierter PnL) – nur echte Trades
    equity: dict = defaultdict(list)
    running_pnl: dict = defaultdict(float)
    for row in real_rows:
        bid = row["bot_id"]
        running_pnl[bid] += float(row["pnl"])
        equity[bid].append({
            "t":   row["closed_at"][:16] if row["closed_at"] else "",
            "pnl": round(running_pnl[bid], 4),
        })

    def _totals(bots: list) -> dict:
        n = sum(b["total"] for b in bots)
        wins = sum(b["wins"] for b in bots)
        pnl = sum(b["total_pnl"] for b in bots)
        return {"n": n, "wins": wins, "total_pnl": round(pnl, 4),
                "bots": len(bots)}

    # Reflexionsregeln
    rules = conn.execute("""
        SELECT rule_text, created_at, basis_trades FROM reflection_rules
        ORDER BY created_at DESC LIMIT 5
    """).fetchall()

    conn.close()

    return {
        "real_bots":   real_bots,
        "shadow_bots": shadow_bots,
        "real_total":  _totals(real_bots),
        "shadow_total": _totals(shadow_bots),
        "equity":      dict(equity),
        "rules":       [dict(r) for r in rules],
        "updated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def _short_symbol(sym: str) -> str:
    """PF_XBTUSD -> XBT für kompakte Anzeige."""
    s = sym or ""
    if s.startswith("PF_"):
        s = s[3:]
    if s.endswith("USD"):
        s = s[:-3]
    return s or "—"


def _bot_table(bots: list, with_sharpe: bool) -> str:
    """Baut die Pro-Bot-Tabelle (echte oder Shadow-Trades)."""
    if not bots:
        return "<p style='color:#8b949e;'>Noch keine Trades vorhanden.</p>"

    sharpe_head = "<th>Sharpe</th>" if with_sharpe else ""
    head = (f"<tr><th>Bot</th><th>Symbol</th><th>Strategie</th><th>Trades</th>"
            f"<th>WR</th><th>∑PnL</th><th>Ø PnL</th><th>Best</th><th>Worst</th>"
            f"{sharpe_head}</tr>")

    body = ""
    for r in bots:
        pnl_color = "green" if r["total_pnl"] >= 0 else "red"
        avg_color = "green" if r["avg_pnl"] >= 0 else "red"
        sharpe_cell = f"<td>{r['sharpe']:.3f}</td>" if with_sharpe else ""
        body += f"""
        <tr>
          <td>Bot #{r['bot_id']}</td>
          <td>{_short_symbol(r['symbol'])}</td>
          <td>{r['strategy']}</td>
          <td>{r['total']}</td>
          <td>{r['win_rate']*100:.0f}%</td>
          <td style="color:{pnl_color};">{r['total_pnl']:+.4f}</td>
          <td style="color:{avg_color};">{r['avg_pnl']:+.4f}</td>
          <td class="green">{r['best']:+.4f}</td>
          <td class="red">{r['worst']:+.4f}</td>
          {sharpe_cell}
        </tr>"""
    return f"<table>{head}{body}</table>"


def build_html(data: dict) -> str:
    """Generiert vollständige HTML-Seite."""
    real_bots    = data["real_bots"]
    shadow_bots  = data["shadow_bots"]
    real_total   = data["real_total"]
    shadow_total = data["shadow_total"]
    equity       = data["equity"]
    rules        = data["rules"]
    updated      = data["updated_at"]

    # --- Echte Trades: Kennzahlen ---
    r_pnl  = round(float(real_total.get("total_pnl") or 0), 4)
    r_n    = int(real_total.get("n") or 0)
    r_wins = int(real_total.get("wins") or 0)
    r_wr   = (r_wins / r_n * 100) if r_n else 0
    r_bots = int(real_total.get("bots") or 0)

    # --- Shadow-Trades: Kennzahlen ---
    s_pnl  = round(float(shadow_total.get("total_pnl") or 0), 4)
    s_n    = int(shadow_total.get("n") or 0)
    s_wins = int(shadow_total.get("wins") or 0)
    s_wr   = (s_wins / s_n * 100) if s_n else 0
    s_bots = int(shadow_total.get("bots") or 0)

    equity_json = json.dumps(equity)

    real_table   = _bot_table(real_bots, with_sharpe=True)
    shadow_table = _bot_table(shadow_bots, with_sharpe=False)

    # Reflexionsregeln
    rules_html = ""
    for rule in rules:
        rules_html += f"""
        <div class="rule-card">
          <small>{rule.get('created_at', '')[:16]} | Basis: {rule.get('basis_trades', 0)} Trades</small>
          <p>{rule.get('rule_text', '').replace(chr(10), '<br>')}</p>
        </div>"""
    if not rules_html:
        rules_html = "<p>Noch keine Reflexionsregeln generiert.</p>"

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<title>Bot-Netzwerk Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#0d1117; color:#e6edf3; margin:0; padding:20px; }}
  h1,h2 {{ color:#58a6ff; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin:20px 0; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }}
  .stat {{ font-size:2em; font-weight:bold; }}
  .green {{ color:#3fb950; }} .red {{ color:#f85149; }}
  table {{ width:100%; border-collapse:collapse; }}
  th,td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #21262d; }}
  th {{ color:#58a6ff; }}
  canvas {{ max-height:300px; }}
  .rule-card {{ background:#161b22; border-left:3px solid #58a6ff; padding:10px; margin:8px 0; border-radius:4px; }}
  .rule-card small {{ color:#8b949e; }}
  .updated {{ color:#8b949e; font-size:0.85em; }}
  .section {{ border-top:2px solid #30363d; margin-top:32px; padding-top:8px; }}
  .section.real h2 {{ color:#3fb950; }}
  .section.shadow h2 {{ color:#d29922; }}
  .badge {{ font-size:0.5em; color:#8b949e; font-weight:normal; }}
</style>
</head>
<body>
<h1>🤖 Bot-Netzwerk Dashboard</h1>
<p class="updated">Zuletzt aktualisiert: {updated}</p>

<!-- ===================== ECHTE TRADES ===================== -->
<div class="section real">
<h2>💰 Echte Trades</h2>

<div class="grid">
  <div class="card">
    <div>Gesamt PnL</div>
    <div class="stat {'green' if r_pnl >= 0 else 'red'}">{r_pnl:+.4f}</div>
  </div>
  <div class="card">
    <div>Echte Trades</div>
    <div class="stat">{r_n}</div>
  </div>
  <div class="card">
    <div>Win-Rate</div>
    <div class="stat">{r_wr:.0f}%</div>
  </div>
  <div class="card">
    <div>Bots mit Trades</div>
    <div class="stat">{r_bots}</div>
  </div>
</div>

<h2>Equity-Kurven <span class="badge">(echte Trades)</span></h2>
<div class="card">
  <canvas id="equityChart"></canvas>
</div>

<h2>Pro-Bot-Statistik <span class="badge">(alle Bots mit echten Trades)</span></h2>
<div class="card">
{real_table}
</div>
</div>

<!-- ===================== SHADOW-TRADES ===================== -->
<div class="section shadow">
<h2>👻 Shadow-Trades <span class="badge">(blockierte Signale, virtuell aufgelöst)</span></h2>

<div class="grid">
  <div class="card">
    <div>Gesamt PnL</div>
    <div class="stat {'green' if s_pnl >= 0 else 'red'}">{s_pnl:+.4f}</div>
  </div>
  <div class="card">
    <div>Shadow-Trades</div>
    <div class="stat">{s_n}</div>
  </div>
  <div class="card">
    <div>Win-Rate</div>
    <div class="stat">{s_wr:.0f}%</div>
  </div>
  <div class="card">
    <div>Bots mit Shadows</div>
    <div class="stat">{s_bots}</div>
  </div>
</div>

<h2>Pro-Bot-Statistik <span class="badge">(alle Bots mit Shadow-Trades)</span></h2>
<div class="card">
{shadow_table}
</div>
</div>

<!-- ===================== REFLEXION ===================== -->
<div class="section">
<h2>🧠 LLM-Reflexionsregeln</h2>
<div class="card">
{rules_html}
</div>
</div>

<script>
const equity = {equity_json};
const colors = ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff',
                 '#79c0ff','#56d364','#ff7b72','#e3b341','#d2a8ff'];

const datasets = Object.entries(equity).slice(0, 10).map(([botId, pts], i) => ({{
  label: `Bot #${{botId}}`,
  data: pts.map(p => ({{ x: p.t, y: p.pnl }})),
  borderColor: colors[i % colors.length],
  backgroundColor: 'transparent',
  tension: 0.3,
  pointRadius: 0,
  borderWidth: 1.5,
}}));

new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{ datasets }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#e6edf3', boxWidth: 12 }} }} }},
    scales: {{
      x: {{ type: 'category', ticks: {{ color: '#8b949e', maxTicksLimit: 8 }},
            grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
    }},
  }},
}});
</script>
</body>
</html>"""


def generate_dashboard():
    """Lädt Daten und schreibt Dashboard-HTML."""
    logger.info("Generiere Dashboard...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = load_network_data()
        html = build_html(data)
        OUTPUT_PATH.write_text(html, encoding="utf-8")
        logger.info(f"Dashboard geschrieben: {OUTPUT_PATH.absolute()}")
        return True
    except Exception as e:
        logger.error(f"Dashboard-Fehler: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    generate_dashboard()
    print(f"Dashboard: {OUTPUT_PATH.absolute()}")
