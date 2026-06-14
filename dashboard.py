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


def load_network_data() -> dict:
    """Lädt alle relevanten Daten aus network.db."""
    from network_db import get_connection, get_all_bot_rankings

    conn = get_connection()
    rankings = get_all_bot_rankings(min_trades=3)

    # Equity-Kurven pro Bot (kumulierter PnL)
    equity: dict = defaultdict(list)
    trades_raw = conn.execute("""
        SELECT bot_id, symbol, side, pnl, exit_reason, strategy, regime,
               is_shadow, is_synthetic, closed_at
        FROM trades_network
        WHERE exit_price IS NOT NULL AND pnl IS NOT NULL
          AND is_synthetic = 0
        ORDER BY closed_at
    """).fetchall()

    running_pnl: dict = defaultdict(float)
    for row in trades_raw:
        bid = row["bot_id"]
        running_pnl[bid] += float(row["pnl"])
        equity[bid].append({
            "t":   row["closed_at"][:16] if row["closed_at"] else "",
            "pnl": round(running_pnl[bid], 4),
        })

    # Netzwerk-Gesamtstatistik
    total_row = conn.execute("""
        SELECT COUNT(*) as n,
               SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins,
               SUM(pnl) as total_pnl,
               AVG(pnl) as avg_pnl
        FROM trades_network
        WHERE exit_price IS NOT NULL AND pnl IS NOT NULL
          AND is_synthetic = 0 AND is_shadow = 0
    """).fetchone()

    shadow_row = conn.execute("""
        SELECT COUNT(*) as n, SUM(pnl) as total_pnl
        FROM trades_network
        WHERE exit_price IS NOT NULL AND pnl IS NOT NULL AND is_shadow = 1
    """).fetchone()

    # Reflexionsregeln
    rules = conn.execute("""
        SELECT rule_text, created_at, basis_trades FROM reflection_rules
        ORDER BY created_at DESC LIMIT 5
    """).fetchall()

    conn.close()

    return {
        "rankings":   rankings,
        "equity":     dict(equity),
        "total":      dict(total_row) if total_row else {},
        "shadow":     dict(shadow_row) if shadow_row else {},
        "rules":      [dict(r) for r in rules],
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def build_html(data: dict) -> str:
    """Generiert vollständige HTML-Seite."""
    rankings  = data["rankings"]
    equity    = data["equity"]
    total     = data["total"]
    shadow    = data["shadow"]
    rules     = data["rules"]
    updated   = data["updated_at"]

    total_pnl = round(float(total.get("total_pnl") or 0), 4)
    total_n   = int(total.get("n") or 0)
    total_wins = int(total.get("wins") or 0)
    wr_pct    = (total_wins / total_n * 100) if total_n else 0

    # Equity-Kurven-Daten als JSON für Chart.js
    equity_json = json.dumps(equity)

    # Rankings-Tabelle
    ranking_rows = ""
    for i, r in enumerate(rankings[:20]):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        pnl_color = "green" if r.get("total_pnl", 0) >= 0 else "red"
        ranking_rows += f"""
        <tr>
          <td>{emoji}</td>
          <td>Bot #{r['bot_id']}</td>
          <td>{r.get('total', 0)}</td>
          <td>{r.get('win_rate', 0)*100:.0f}%</td>
          <td style="color:{pnl_color};">{r.get('total_pnl', 0):+.4f}</td>
          <td>{r.get('sharpe', 0):.3f}</td>
        </tr>"""

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
</style>
</head>
<body>
<h1>🤖 Bot-Netzwerk Dashboard</h1>
<p class="updated">Zuletzt aktualisiert: {updated}</p>

<div class="grid">
  <div class="card">
    <div>Gesamt PnL</div>
    <div class="stat {'green' if total_pnl >= 0 else 'red'}">{total_pnl:+.4f}</div>
  </div>
  <div class="card">
    <div>Echte Trades</div>
    <div class="stat">{total_n}</div>
  </div>
  <div class="card">
    <div>Win-Rate</div>
    <div class="stat">{wr_pct:.0f}%</div>
  </div>
  <div class="card">
    <div>Shadow-Trades</div>
    <div class="stat">{int(shadow.get('n') or 0)}</div>
    <small>PnL: {float(shadow.get('total_pnl') or 0):+.4f}</small>
  </div>
  <div class="card">
    <div>Aktive Bots</div>
    <div class="stat">{len(rankings)}</div>
  </div>
</div>

<h2>Equity-Kurven</h2>
<div class="card">
  <canvas id="equityChart"></canvas>
</div>

<h2>Bot-Ranking (Top 20)</h2>
<div class="card">
<table>
  <tr><th>#</th><th>Bot</th><th>Trades</th><th>WR</th><th>∑PnL</th><th>Sharpe</th></tr>
  {ranking_rows}
</table>
</div>

<h2>LLM-Reflexionsregeln</h2>
<div class="card">
{rules_html}
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
