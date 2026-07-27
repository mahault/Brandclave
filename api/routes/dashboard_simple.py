"""Simple working dashboard for BrandClave."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/monitoring/dashboard-v2", response_class=HTMLResponse)
async def dashboard_v2():
    """Simple working dashboard."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BrandClave Intelligence</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg: #0e0c09;
            --surface: #17140f;
            --surface-2: #201a12;
            --surface-3: #2a2318;
            --ink: #f2ecdf;
            --ink-2: #b9ae9c;
            --ink-3: #857a68;
            --line: rgba(212,175,106,0.16);
            --line-strong: rgba(212,175,106,0.34);
            --gold: #d4af6a;
            --gold-deep: #b8862e;
            --gold-ink: #141008;
            --violet: #8b7ce0;
            --teal: #3aa88d;
            --rose: #c25a78;
            --blue: #4a8bc2;
            --good: #3aa88d;
            --warn: #b8862e;
            --bad: #c25a78;
            --grad: linear-gradient(90deg, #c25a78, #8b7ce0, #3aa88d, #d4af6a);
            --font-display: 'Archivo', 'Segoe UI', sans-serif;
            --font-body: 'Inter', -apple-system, 'Segoe UI', sans-serif;
            --font-mono: 'JetBrains Mono', 'Consolas', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-body);
            background: var(--bg);
            background-image: radial-gradient(ellipse 80% 40% at 50% -10%, rgba(212,175,106,0.07), transparent);
            min-height: 100vh;
            color: var(--ink);
        }
        ::selection { background: rgba(212,175,106,0.30); }

        .hero {
            padding: 56px 20px 36px;
            text-align: center;
            border-bottom: 1px solid var(--line);
        }
        .hero::before {
            content: 'HOSPITALITY DEMAND INTELLIGENCE';
            display: block;
            font-family: var(--font-mono);
            font-size: 0.7em;
            letter-spacing: 0.35em;
            color: var(--gold);
            margin-bottom: 14px;
        }
        .hero h1 {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 2.4em;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--ink);
        }
        .hero p { color: var(--ink-2); margin-top: 10px; font-size: 0.95em; }
        .hero::after {
            content: '';
            display: block;
            width: 120px;
            height: 3px;
            margin: 22px auto 0;
            background: var(--grad);
            border-radius: 2px;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 28px 20px; }

        .status-bar {
            background: var(--surface);
            border: 1px solid var(--line);
            padding: 12px 18px;
            border-radius: 10px;
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--ink-2);
            font-family: var(--font-mono);
            font-size: 0.85em;
        }
        .status-bar .icon { font-size: 1.1em; }
        .status-bar button {
            margin-left: auto;
            padding: 8px 18px;
            background: transparent;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            color: var(--gold);
            font-family: var(--font-mono);
            font-size: 0.9em;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
            transition: all 0.2s;
        }
        .status-bar button:hover { background: var(--gold); color: var(--gold-ink); border-color: var(--gold); }

        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 26px;
            flex-wrap: wrap;
            border-bottom: 1px solid var(--line);
        }
        .tab {
            padding: 12px 16px;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--ink-2);
            font-family: var(--font-mono);
            font-size: 0.78em;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            cursor: pointer;
            transition: color 0.2s, border-color 0.2s;
        }
        .tab:hover { color: var(--ink); }
        .tab.active { color: var(--gold); border-bottom-color: var(--gold); }

        .section { display: none; }
        .section.active { display: block; }

        .card {
            background: var(--surface);
            border: 1px solid var(--line);
            padding: 26px;
            margin-bottom: 18px;
            border-radius: 12px;
        }
        .card h2 {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1.15em;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--ink);
            margin-bottom: 16px;
        }
        .card h2::before {
            content: '';
            display: block;
            width: 48px;
            height: 3px;
            background: var(--grad);
            border-radius: 2px;
            margin-bottom: 12px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-bottom: 8px;
        }
        .metric {
            text-align: center;
            padding: 18px 15px;
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 10px;
        }
        .metric-value {
            font-family: var(--font-display);
            font-size: 2em;
            font-weight: 800;
            color: var(--gold);
        }
        .metric-label {
            color: var(--ink-3);
            font-family: var(--font-mono);
            font-size: 0.68em;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-top: 6px;
        }

        .trend-card {
            background: rgba(139,124,224,0.08);
            border: 1px solid rgba(139,124,224,0.30);
            border-left: 3px solid var(--violet);
            color: var(--ink);
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 12px;
            transition: transform 0.15s, border-color 0.15s;
        }
        .trend-card h3 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }
        .trend-card p { color: var(--ink-2); font-size: 0.9em; line-height: 1.5; }
        .trend-card:hover { transform: translateY(-2px); border-color: var(--violet); }
        .trend-meta { margin-top: 10px; font-size: 0.82em; color: var(--ink-2); font-family: var(--font-mono); }

        .move-card {
            background: rgba(58,168,141,0.08);
            border: 1px solid rgba(58,168,141,0.30);
            border-left: 3px solid var(--teal);
            color: var(--ink);
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 12px;
            transition: transform 0.15s, border-color 0.15s;
        }
        .move-card h3 { margin-bottom: 5px; font-family: var(--font-display); font-weight: 600; }
        .move-card .company { font-size: 0.85em; color: var(--teal); font-family: var(--font-mono); letter-spacing: 0.06em; margin-bottom: 8px; }
        .move-card p { font-size: 0.9em; line-height: 1.5; color: var(--ink-2); }
        .move-card:hover { transform: translateY(-2px); border-color: var(--teal); }
        .move-badges { display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
        .move-type-badge {
            background: rgba(58,168,141,0.18);
            color: var(--teal);
            padding: 3px 10px;
            border-radius: 999px;
            font-family: var(--font-mono);
            font-size: 0.7em;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .market-badge {
            background: var(--surface-2);
            border: 1px solid var(--line);
            color: var(--ink-2);
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.78em;
        }
        .move-actions { margin-top: 10px; display: flex; gap: 8px; }
        .move-action-btn {
            padding: 6px 14px;
            border: 1px solid var(--line-strong);
            background: transparent;
            color: var(--ink-2);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.82em;
            transition: all 0.2s;
        }
        .move-action-btn.btn-save { background: transparent; color: var(--ink-2); }
        .move-action-btn.btn-save:hover { color: var(--ink); border-color: var(--gold); }
        .move-action-btn.btn-save.saved { background: var(--teal); border-color: var(--teal); color: var(--gold-ink); }

        .content-item { padding: 14px 4px; border-bottom: 1px solid var(--line); }
        .content-item:last-child { border-bottom: none; }
        .content-item h4 { color: var(--ink); margin-bottom: 5px; font-weight: 600; }
        .content-item p { color: var(--ink-2); font-size: 0.9em; }
        .content-item .meta { font-size: 0.78em; color: var(--ink-3); margin-top: 5px; font-family: var(--font-mono); }
        .content-item .source {
            background: rgba(212,175,106,0.16);
            color: var(--gold);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-size: 0.72em;
            letter-spacing: 0.06em;
        }

        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--line); }
        th {
            background: transparent;
            color: var(--ink-3);
            font-family: var(--font-mono);
            font-size: 0.72em;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 500;
        }
        td { color: var(--ink-2); font-size: 0.9em; }

        .badge { padding: 3px 10px; border-radius: 4px; font-size: 0.78em; font-family: var(--font-mono); }
        .badge-success { background: rgba(58,168,141,0.16); color: var(--teal); }
        .badge-warning { background: rgba(184,134,46,0.16); color: var(--gold); }

        .empty { text-align: center; padding: 44px; color: var(--ink-3); }
        .empty .icon { font-size: 2em; margin-bottom: 10px; opacity: 0.7; }

        .error {
            background: rgba(194,90,120,0.12);
            border: 1px solid rgba(194,90,120,0.35);
            color: var(--rose);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        /* White Space Badge */
        .white-space-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: var(--surface-2);
            border: 1px solid var(--line);
            color: var(--ink-2);
            padding: 3px 9px;
            border-radius: 999px;
            font-family: var(--font-mono);
            font-size: 0.72em;
            letter-spacing: 0.04em;
        }
        .white-space-high { background: rgba(58,168,141,0.18); color: var(--teal); border-color: rgba(58,168,141,0.35); }
        .white-space-medium { background: rgba(184,134,46,0.18); color: var(--gold); border-color: rgba(212,175,106,0.35); }
        .white-space-low { background: rgba(194,90,120,0.14); color: var(--rose); border-color: rgba(194,90,120,0.35); }

        /* Filter Bar */
        .filter-bar {
            display: flex;
            gap: 12px;
            margin-bottom: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filter-select, select {
            padding: 8px 12px;
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            background: var(--surface-2);
            color: var(--ink);
            font-family: var(--font-body);
            font-size: 0.88em;
            min-width: 140px;
            cursor: pointer;
        }
        .filter-select:focus, select:focus { outline: none; border-color: var(--gold); }
        .filter-reset, button.filter-reset {
            padding: 8px 16px;
            background: transparent;
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            color: var(--ink-2);
            cursor: pointer;
            font-size: 0.88em;
        }
        .filter-reset:hover { color: var(--ink); border-color: var(--gold); }
        .saved-count {
            background: rgba(139,124,224,0.14);
            color: var(--violet);
            padding: 4px 10px;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-size: 0.78em;
            margin-left: auto;
        }

        input[type="text"], input[type="number"], textarea {
            background: var(--surface-2);
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            color: var(--ink);
            font-family: var(--font-body);
        }
        input[type="text"]:focus, input[type="number"]:focus, textarea:focus { outline: none; border-color: var(--gold); }
        input::placeholder, textarea::placeholder { color: var(--ink-3); }

        /* Trend Action Buttons */
        .trend-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--line);
        }
        .trend-action-btn {
            padding: 6px 14px;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            font-size: 0.82em;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: transform 0.1s, border-color 0.15s;
        }
        .trend-action-btn:hover { transform: translateY(-1px); border-color: var(--gold); }
        .btn-save { background: transparent; color: var(--ink-2); }
        .btn-save.saved { background: var(--teal); border-color: var(--teal); color: var(--gold-ink); }
        .btn-brand { background: var(--gold); border-color: var(--gold); color: var(--gold-ink); font-weight: 600; }

        /* Chat */
        .chat-message { margin-bottom: 15px; }
        .chat-message.user { text-align: right; }
        .chat-message.assistant { text-align: left; }
        .chat-bubble {
            display: inline-block;
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 14px;
            line-height: 1.5;
            text-align: left;
        }
        .chat-message.user .chat-bubble {
            background: rgba(212,175,106,0.16);
            border: 1px solid rgba(212,175,106,0.30);
            color: var(--ink);
            border-bottom-right-radius: 4px;
        }
        .chat-message.assistant .chat-bubble {
            background: var(--surface-2);
            color: var(--ink);
            border: 1px solid var(--line);
            border-bottom-left-radius: 4px;
        }
        .chat-confidence { font-size: 0.72em; margin-top: 4px; font-family: var(--font-mono); }
        .confidence-high { color: var(--teal); }
        .confidence-medium { color: var(--gold); }
        .confidence-low { color: var(--rose); }
        .suggestion-chip {
            padding: 8px 16px;
            background: transparent;
            border: 1px solid var(--line-strong);
            color: var(--gold);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.88em;
            transition: all 0.2s;
        }
        .suggestion-chip:hover { background: var(--gold); color: var(--gold-ink); border-color: var(--gold); }
        .chat-typing { display: flex; gap: 4px; padding: 10px 15px; }
        .chat-typing span {
            width: 8px;
            height: 8px;
            background: var(--gold);
            border-radius: 50%;
            animation: typing 1s infinite;
        }
        .chat-typing span:nth-child(2) { animation-delay: 0.2s; }
        .chat-typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }

        /* My Projects */
        .profile-insights-card {
            background: var(--surface-2);
            border: 1px solid rgba(139,124,224,0.30);
            border-left: 3px solid var(--violet);
            color: var(--ink);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        .profile-insights-card h3 { margin-bottom: 15px; font-family: var(--font-display); font-weight: 600; }
        .profile-tag {
            display: inline-block;
            background: rgba(139,124,224,0.14);
            color: var(--violet);
            padding: 5px 12px;
            border-radius: 999px;
            margin: 3px;
            font-size: 0.85em;
        }
        .profile-section { margin-bottom: 12px; }
        .profile-section-title {
            font-family: var(--font-mono);
            font-size: 0.72em;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--ink-3);
            margin-bottom: 6px;
        }
        .btn-primary {
            padding: 12px 24px;
            background: var(--gold);
            color: var(--gold-ink);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95em;
            transition: background 0.2s;
        }
        .btn-primary:hover { background: #e2c184; }
        .btn-primary:disabled { background: var(--surface-3); color: var(--ink-3); cursor: not-allowed; }
        .btn-secondary {
            padding: 12px 24px;
            background: transparent;
            color: var(--ink-2);
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95em;
        }
        .btn-secondary:hover { color: var(--ink); border-color: var(--gold); }
        .saved-item-card {
            background: var(--surface-2);
            border: 1px solid var(--line);
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .saved-item-card h4 { margin-bottom: 4px; color: var(--ink); }
        .saved-item-meta { font-size: 0.82em; color: var(--ink-3); }
        .saved-item-actions { display: flex; gap: 8px; }
        .btn-remove {
            padding: 5px 12px;
            background: transparent;
            color: var(--rose);
            border: 1px solid rgba(194,90,120,0.4);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.82em;
        }
        .btn-remove:hover { background: var(--rose); color: var(--gold-ink); }

        /* Modal */
        .modal-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(8,6,4,0.8); z-index: 1000; overflow-y: auto; padding: 20px;
        }
        .modal-overlay.active { display: flex; justify-content: center; align-items: flex-start; }
        .modal-content {
            background: var(--surface);
            border: 1px solid var(--line-strong);
            border-radius: 12px;
            max-width: 700px; width: 100%; margin: 40px auto; position: relative;
            overflow: hidden;
        }
        .modal-header {
            background: var(--surface-2);
            border-bottom: 1px solid var(--line);
            border-top: 3px solid var(--violet);
            color: var(--ink);
            padding: 20px;
        }
        .modal-header.move-header { background: var(--surface-2); border-top-color: var(--teal); }
        .modal-header h2 { margin: 0; font-size: 1.3em; line-height: 1.3; font-family: var(--font-display); font-weight: 700; }
        .modal-header .meta { color: var(--ink-2); margin-top: 8px; font-size: 0.85em; font-family: var(--font-mono); }
        .modal-body { padding: 20px; max-height: 60vh; overflow-y: auto; }
        .modal-section { margin-bottom: 20px; }
        .modal-section:last-child { margin-bottom: 0; }
        .modal-section h3 {
            color: var(--gold);
            margin-bottom: 10px;
            font-family: var(--font-mono);
            font-size: 0.78em;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 500;
        }
        .modal-section p { color: var(--ink-2); line-height: 1.6; }
        .modal-close {
            position: absolute; top: 15px; right: 15px;
            background: transparent; border: 1px solid var(--line-strong);
            color: var(--ink-2); width: 32px; height: 32px; border-radius: 50%;
            cursor: pointer; font-size: 1.1em;
        }
        .modal-close:hover { color: var(--ink); border-color: var(--gold); }
        .source-quote {
            background: var(--surface-2);
            border-left: 3px solid var(--gold);
            padding: 12px 15px;
            margin-bottom: 10px;
            border-radius: 0 8px 8px 0;
            font-style: italic;
            color: var(--ink-2);
            font-size: 0.9em;
        }
        .topic-tag {
            display: inline-block;
            background: rgba(139,124,224,0.14);
            color: var(--violet);
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.82em;
            margin: 3px;
        }

        .quick-city {
            padding: 5px 14px;
            background: transparent;
            border: 1px solid var(--line-strong);
            color: var(--ink-2);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.82em;
            margin: 2px;
            transition: all 0.2s;
        }
        .quick-city:hover { color: var(--gold); border-color: var(--gold); }

        .desire-card {
            background: rgba(194,90,120,0.08);
            border: 1px solid rgba(194,90,120,0.30);
            border-left: 3px solid var(--rose);
            color: var(--ink);
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 12px;
        }
        .desire-card h4 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }
        .desire-card p { font-size: 0.9em; color: var(--ink-2); line-height: 1.5; }
        .desire-meta { margin-top: 10px; font-size: 0.82em; color: var(--ink-2); font-family: var(--font-mono); }

        .opportunity-card {
            background: rgba(74,139,194,0.08);
            border: 1px solid rgba(74,139,194,0.30);
            border-left: 3px solid var(--blue);
            color: var(--ink);
            padding: 14px 16px;
            border-radius: 10px;
            margin-bottom: 8px;
        }

        .concept-card {
            background: rgba(212,175,106,0.08);
            border: 1px solid rgba(212,175,106,0.35);
            border-left: 3px solid var(--gold);
            color: var(--ink);
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 12px;
        }
        .concept-card h4 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }

        /* Demand Scan */
        .property-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            transition: border-color 0.2s, transform 0.15s;
        }
        .property-card:hover { border-color: var(--line-strong); transform: translateY(-2px); }
        .property-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }
        .property-card h3 { margin: 0; color: var(--ink); font-size: 1.15em; font-family: var(--font-display); font-weight: 600; }
        .property-card .location { color: var(--ink-3); font-size: 0.85em; margin-top: 4px; font-family: var(--font-mono); }

        /* Demand Fit Score Badge */
        .demand-score {
            padding: 8px 15px;
            border-radius: 999px;
            font-family: var(--font-mono);
            font-weight: 500;
            font-size: 0.9em;
        }
        .demand-high { background: rgba(58,168,141,0.18); color: var(--teal); border: 1px solid rgba(58,168,141,0.35); }
        .demand-medium { background: rgba(184,134,46,0.18); color: var(--gold); border: 1px solid rgba(212,175,106,0.35); }
        .demand-low { background: rgba(194,90,120,0.14); color: var(--rose); border: 1px solid rgba(194,90,120,0.35); }

        /* Misalignment Flags */
        .misalignment-flag {
            display: inline-flex;
            align-items: center;
            background: rgba(194,90,120,0.12);
            color: var(--rose);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.78em;
            margin: 3px;
        }
        .misalignment-flag::before { content: "! "; font-weight: 700; margin-right: 4px; }

        /* Property Sections */
        .property-section { margin-bottom: 15px; }
        .property-section-title {
            font-family: var(--font-mono);
            font-size: 0.72em;
            font-weight: 500;
            color: var(--ink-3);
            margin-bottom: 8px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .gap-item {
            display: inline-block;
            background: rgba(184,134,46,0.14);
            color: var(--gold);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.82em;
            margin: 2px;
        }
        .opportunity-item {
            display: flex;
            align-items: center;
            background: rgba(74,139,194,0.10);
            color: var(--blue);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.88em;
            margin-bottom: 6px;
        }
        .opportunity-item::before { content: "→ "; font-weight: bold; margin-right: 6px; }

        .property-actions {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid var(--line);
        }
        .property-action-btn {
            padding: 8px 18px;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.88em;
            font-weight: 500;
            transition: all 0.2s;
        }
        .property-action-btn.btn-brand {
            background: var(--gold);
            border-color: var(--gold);
            color: var(--gold-ink);
        }
        .property-action-btn.btn-brand:hover { background: #e2c184; }
        .property-action-btn.btn-save {
            background: transparent;
            color: var(--ink-2);
        }
        .property-action-btn.btn-save:hover { color: var(--ink); border-color: var(--gold); }
        .property-action-btn.btn-save.saved {
            background: rgba(58,168,141,0.18);
            border-color: rgba(58,168,141,0.4);
            color: var(--teal);
        }
    </style>
</head>
<body>
    <div class="hero">
        <h1>BrandClave Intelligence</h1>
        <p>Hospitality Trends, Strategic Moves & Demand Signals</p>
    </div>

    <div class="container">
        <div class="status-bar">
            <span class="icon" id="status-icon">⏳</span>
            <span id="status-text">Loading...</span>
            <button onclick="loadAllData()">Refresh</button>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">Overview</button>
            <button class="tab" onclick="showTab('citydesires')">City Desires</button>
            <button class="tab" onclick="showTab('trends')">Social Pulse</button>
            <button class="tab" onclick="showTab('moves')">Hotelier Bets</button>
            <button class="tab" onclick="showTab('demandscan')">Demand Scan</button>
            <button class="tab" onclick="showTab('content')">Content</button>
            <button class="tab" onclick="showTab('scrapers')">Scrapers</button>
            <button class="tab" onclick="showTab('chat')">Chat</button>
            <button class="tab" onclick="showTab('projects')" id="projects-tab">My Projects</button>
        </div>

        <div id="overview" class="section active">
            <div class="card">
                <h2>Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="m-content">-</div>
                        <div class="metric-label">Content Items</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="m-processed">-</div>
                        <div class="metric-label">Processed</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="m-trends">-</div>
                        <div class="metric-label">Trends</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="m-moves">-</div>
                        <div class="metric-label">Moves</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>Latest Trend</h2>
                <div id="latest-trend"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
            <div class="card">
                <h2>Latest Move</h2>
                <div id="latest-move"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="citydesires" class="section">
            <div class="card">
                <h2>City Desires</h2>
                <p style="color:var(--ink-2);margin-bottom:15px;">Discover what travelers are craving in specific destinations. Uncover unmet needs, frustrations, and white-space opportunities from social conversations.</p>
                <p style="color:var(--ink-2);margin-bottom:15px;">Type a city to discover what travelers want but can't find.</p>
                <div style="display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;">
                    <input type="text" id="city-input" placeholder="City name (e.g., Lisbon)"
                           style="padding:10px 15px;border:1px solid var(--line-strong);border-radius:6px;font-size:1em;flex:1;min-width:150px;">
                    <input type="text" id="country-input" placeholder="Country (optional)"
                           style="padding:10px 15px;border:1px solid var(--line-strong);border-radius:6px;font-size:1em;width:150px;">
                    <button onclick="analyzeCity()" id="analyze-btn"
                            style="padding:10px 20px;background:var(--gold);color:var(--gold-ink);border:none;border-radius:6px;cursor:pointer;font-size:1em;">
                        Analyze
                    </button>
                </div>
                <div style="margin-bottom:15px;">
                    <span style="color:var(--ink-3);font-size:0.9em;">Popular: </span>
                    <button onclick="quickCity('Lisbon','Portugal')" class="quick-city">Lisbon</button>
                    <button onclick="quickCity('Barcelona','Spain')" class="quick-city">Barcelona</button>
                    <button onclick="quickCity('Tokyo','Japan')" class="quick-city">Tokyo</button>
                    <button onclick="quickCity('Bali','Indonesia')" class="quick-city">Bali</button>
                    <button onclick="quickCity('Paris','France')" class="quick-city">Paris</button>
                </div>
                <div id="city-results">
                    <div class="empty"><div class="icon">🔍</div>Enter a city to analyze traveler desires</div>
                </div>
            </div>
        </div>

        <div id="trends" class="section">
            <div class="card">
                <h2>Social Pulse Trends</h2>
                <p style="color:var(--ink-2);margin-bottom:15px;">Track emerging hospitality trends from Reddit, industry news, and social conversations. Discover what's gaining momentum and find white-space opportunities before your competitors.</p>
                <div class="filter-bar">
                    <select id="filter-region" class="filter-select" onchange="applyFilters()">
                        <option value="">All Regions</option>
                    </select>
                    <select id="filter-audience" class="filter-select" onchange="applyFilters()">
                        <option value="">All Segments</option>
                    </select>
                    <select id="filter-time" class="filter-select" onchange="applyFilters()">
                        <option value="">All Time</option>
                        <option value="7">Last 7 Days</option>
                        <option value="14">Last 14 Days</option>
                        <option value="30">Last 30 Days</option>
                    </select>
                    <button class="filter-reset" onclick="resetFilters()">Reset</button>
                    <span id="saved-count" class="saved-count"></span>
                </div>
                <div id="trends-list"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="moves" class="section">
            <div class="card">
                <h2>Hotelier Bets</h2>
                <p style="color:var(--ink-2);margin-bottom:15px;">Monitor strategic moves by hotel companies worldwide. Track launches, acquisitions, repositionings, and partnerships to understand where the industry is heading and identify competitive signals.</p>
                <div class="filter-bar">
                    <select id="filter-company" onchange="applyMoveFilters()">
                        <option value="">All Companies</option>
                    </select>
                    <select id="filter-move-type" onchange="applyMoveFilters()">
                        <option value="">All Move Types</option>
                    </select>
                    <select id="filter-market" onchange="applyMoveFilters()">
                        <option value="">All Markets</option>
                    </select>
                    <button onclick="resetMoveFilters()" class="filter-reset">Reset</button>
                    <span id="moves-saved-count" class="saved-count"></span>
                </div>
                <div id="moves-list"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="demandscan" class="section">
            <div class="card">
                <h2>Demand Scan</h2>
                <p style="color:var(--ink-2);margin-bottom:15px;">Analyze any hotel website against current demand trends. Get fit scores, experience gaps, and opportunities.</p>

                <!-- URL Input Form -->
                <div style="display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;">
                    <input type="text" id="property-url-input" placeholder="Enter hotel website URL (e.g., https://www.example-hotel.com)"
                           style="padding:12px 15px;border:1px solid var(--line-strong);border-radius:6px;font-size:1em;flex:1;min-width:250px;">
                    <button onclick="scanProperty()" id="scan-btn"
                            style="padding:12px 24px;background:var(--gold);color:var(--gold-ink);border:none;border-radius:6px;cursor:pointer;font-size:1em;font-weight:600;">
                        Scan Property
                    </button>
                </div>

                <!-- Scan Status -->
                <div id="scan-status" style="display:none;margin-bottom:20px;padding:15px;border-radius:8px;"></div>

                <!-- Previously Scanned Properties -->
                <h3 style="margin:20px 0 15px;color:var(--ink);">Previously Scanned Properties</h3>
                <div id="properties-list"><div class="empty"><div class="icon">🏨</div>No properties scanned yet. Enter a URL above to analyze a property.</div></div>
            </div>
        </div>

        <div id="content" class="section">
            <div class="card">
                <h2>Recent Content</h2>
                <p style="color:var(--ink-2);margin-bottom:15px;">Browse the latest scraped articles, social posts, and news from our 12+ hospitality sources. This raw content feeds our trend detection and move extraction engines.</p>
                <div id="content-list"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="scrapers" class="section">
            <div class="card">
                <h2>Scraper Status</h2>
                <p style="color:var(--ink-2);margin-bottom:15px;">Monitor the health and activity of our data collection system. Our POMDP-driven scheduler intelligently prioritizes sources based on expected information gain.</p>
                <div id="scrapers-list"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="chat" class="section">
            <div class="card">
                <h2>BrandClave Chat</h2>
                <p style="margin-bottom:15px;color:var(--ink-2);">Your AI-powered hospitality intelligence assistant. Ask about market trends, explore opportunities in specific cities, or get help ideating a brand concept with RAG-powered insights from our data.</p>

                <div id="chat-messages" style="min-height:300px;max-height:500px;overflow-y:auto;border:1px solid var(--line);border-radius:8px;padding:15px;margin-bottom:15px;background:var(--surface-2);">
                    <div class="chat-welcome">
                        <div style="text-align:center;padding:40px 20px;">
                            <div style="font-size:3em;margin-bottom:15px;">🤖</div>
                            <h3 style="color:var(--ink);margin-bottom:10px;">Hello! I'm your hospitality intelligence assistant.</h3>
                            <p style="color:var(--ink-2);">Try asking me about:</p>
                            <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:15px;">
                                <button onclick="sendSuggestion('What are the top wellness trends in hotels?')" class="suggestion-chip">Wellness trends</button>
                                <button onclick="sendSuggestion('What opportunities exist in the boutique hotel market in Lisbon?')" class="suggestion-chip">Lisbon opportunities</button>
                                <button onclick="startBrandBuild()" class="suggestion-chip">Build a brand</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div style="display:flex;gap:10px;">
                    <input type="text" id="chat-input" placeholder="Ask about trends, opportunities, or help building a brand..."
                           style="flex:1;padding:12px 15px;border:1px solid var(--line-strong);border-radius:8px;font-size:1em;"
                           onkeypress="if(event.key==='Enter')sendMessage()">
                    <button onclick="sendMessage()" style="padding:12px 25px;background:var(--gold);color:var(--gold-ink);border:none;border-radius:8px;cursor:pointer;font-weight:600;">Send</button>
                </div>

                <div id="chat-state" style="margin-top:10px;font-size:0.85em;color:var(--ink-3);"></div>
            </div>
        </div>

        <div id="projects" class="section">
            <div class="card">
                <h2>My Projects</h2>
                <p style="margin-bottom:15px;color:var(--ink-2);">Save trends and strategic moves to build a research profile. Your saved items inform brand generation, helping BrandClave understand your interests and create more relevant concepts.</p>

                <!-- Profile Insights -->
                <div id="profile-insights" class="profile-insights-card">
                    <h3>Your Interest Profile</h3>
                    <div id="profile-content">
                        <div class="empty"><div class="icon">💡</div>Save trends and moves to build your profile</div>
                    </div>
                </div>

                <!-- Actions -->
                <div style="display:flex;gap:10px;margin:20px 0;">
                    <button onclick="buildBrandFromProfile()" class="btn-primary" id="build-from-profile-btn" disabled>
                        Build Brand from Profile
                    </button>
                    <button onclick="clearAllSaved()" class="btn-secondary">
                        Clear All
                    </button>
                </div>

                <!-- Saved Trends -->
                <div style="margin-top:20px;">
                    <h3 style="margin-bottom:10px;">Saved Trends <span id="saved-trends-count" style="font-weight:normal;color:var(--ink-2);"></span></h3>
                    <div id="saved-trends-list">
                        <div class="empty"><div class="icon">📊</div>No saved trends yet</div>
                    </div>
                </div>

                <!-- Saved Moves -->
                <div style="margin-top:20px;">
                    <h3 style="margin-bottom:10px;">Saved Moves <span id="saved-moves-count" style="font-weight:normal;color:var(--ink-2);"></span></h3>
                    <div id="saved-moves-list">
                        <div class="empty"><div class="icon">♟️</div>No saved moves yet</div>
                    </div>
                </div>

                <!-- My Blueprints -->
                <div style="margin-top:30px;padding-top:20px;border-top:1px solid var(--line);">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
                        <h3>My Blueprints <span id="saved-blueprints-count" style="font-weight:normal;color:var(--ink-2);"></span></h3>
                        <a href="/api/monitoring/build-a-brand" style="color:var(--gold);text-decoration:none;font-size:0.9em;">+ Create New Blueprint</a>
                    </div>
                    <div id="my-blueprints-list" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:15px;">
                        <div class="empty"><div class="icon">🏨</div>No blueprints yet. Create your first brand concept!</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Store data globally for modal access
        var allTrends = [];
        var allMoves = [];

        function openTrendModal(i) {
            var t = allTrends[i];
            if (!t) return;
            document.getElementById('modal-header').className = 'modal-header';
            document.getElementById('modal-title').textContent = t.name || t.trend_name || 'Unnamed Trend';
            var score = t.strength_score ? Math.round(t.strength_score * 100) + '%' : 'N/A';
            var whiteSpace = t.white_space_score ? Math.round(t.white_space_score * 100) : 0;
            document.getElementById('modal-meta').innerHTML = 'Strength: ' + score + ' | White Space: ' + whiteSpace + '% | ' + (t.volume || 0) + ' sources';
            var h = '';
            if (t.description) h += '<div class="modal-section"><h3>Description</h3><p>' + t.description + '</p></div>';
            if (t.why_it_matters) h += '<div class="modal-section"><h3>Why It Matters</h3><p>' + t.why_it_matters + '</p></div>';
            // White Space Analysis section
            if (t.white_space_score !== undefined) {
                var ws = Math.round((t.white_space_score || 0) * 100);
                var wsClass = ws >= 70 ? 'white-space-high' : ws >= 40 ? 'white-space-medium' : 'white-space-low';
                var wsLabel = ws >= 70 ? 'High Opportunity - underserved market' : ws >= 40 ? 'Moderate Opportunity' : 'Low - competitive market';
                h += '<div class="modal-section"><h3>White Space Analysis</h3>';
                h += '<p><span class="white-space-badge ' + wsClass + '" style="font-size:1em;padding:6px 12px;">' + ws + '% - ' + wsLabel + '</span></p>';
                if (t.region) h += '<p style="margin-top:10px;"><strong>Region:</strong> ' + t.region + '</p>';
                if (t.audience_segment) h += '<p><strong>Segment:</strong> ' + t.audience_segment + '</p>';
                h += '</div>';
            }
            if (t.topics && t.topics.length) {
                var topicsHtml = '';
                for (var ti = 0; ti < t.topics.length; ti++) { topicsHtml += '<span class="topic-tag">' + t.topics[ti] + '</span>'; }
                h += '<div class="modal-section"><h3>Topics</h3><div>' + topicsHtml + '</div></div>';
            }
            if (t.sample_quotes && t.sample_quotes.length) {
                h += '<div class="modal-section"><h3>Source Quotes</h3>';
                for (var qi = 0; qi < t.sample_quotes.length; qi++) { h += '<div class="source-quote">"' + t.sample_quotes[qi] + '"</div>'; }
                h += '</div>';
            }
            document.getElementById('modal-body').innerHTML = h || '<p>No additional details.</p>';
            document.getElementById('modal-overlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function openMoveModal(i) {
            var m = allMoves[i];
            if (!m) return;
            document.getElementById('modal-header').className = 'modal-header move-header';
            document.getElementById('modal-title').textContent = m.title || 'Untitled';

            // Enhanced meta with market and investment
            var metaParts = [m.company || 'Unknown', m.move_type ? m.move_type.replace('_', ' ') : 'Move'];
            if (m.market) metaParts.push('📍 ' + m.market);
            if (m.investment_amount) metaParts.push('💰 ' + m.investment_amount);
            document.getElementById('modal-meta').innerHTML = metaParts.join(' | ');

            var h = '';

            // Summary
            if (m.summary) h += '<div class="modal-section"><h3>Summary</h3><p>' + m.summary + '</p></div>';

            // Why It Matters
            if (m.why_it_matters) h += '<div class="modal-section"><h3>Why It Matters</h3><p>' + m.why_it_matters + '</p></div>';

            // Strategic Implications
            if (m.strategic_implications && m.strategic_implications.length > 0) {
                h += '<div class="modal-section"><h3>Strategic Implications</h3><ul style="margin:0;padding-left:20px;">';
                for (var i = 0; i < m.strategic_implications.length; i++) {
                    h += '<li style="margin-bottom:6px;">' + m.strategic_implications[i] + '</li>';
                }
                h += '</ul></div>';
            }

            // Competitive Impact
            if (m.competitive_impact) {
                h += '<div class="modal-section"><h3>Competitive Impact</h3><p>' + m.competitive_impact + '</p></div>';
            }

            // Source
            if (m.source_url) h += '<div class="modal-section"><h3>Source</h3><p><a href="' + m.source_url + '" target="_blank" style="color:var(--violet);">' + (m.source_name || 'View article') + '</a></p></div>';

            document.getElementById('modal-body').innerHTML = h || '<p>No additional details.</p>';
            document.getElementById('modal-overlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeModal(e) {
            if (e && e.target !== e.currentTarget) return;
            document.getElementById('modal-overlay').classList.remove('active');
            document.body.style.overflow = '';
        }
        document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(); });


        function showTab(tabId) {
            var sections = document.querySelectorAll('.section');
            for (var i = 0; i < sections.length; i++) { sections[i].classList.remove('active'); }
            var tabs = document.querySelectorAll('.tab');
            for (var j = 0; j < tabs.length; j++) { tabs[j].classList.remove('active'); }
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }

        function setStatus(icon, text) {
            document.getElementById('status-icon').textContent = icon;
            document.getElementById('status-text').textContent = text;
        }

        function truncate(str, len) {
            if (!str) return '';
            return str.length > len ? str.substring(0, len) + '...' : str;
        }

        async function loadAllData() {
            setStatus('⏳', 'Loading data...');

            try {
                // Fetch all data in parallel
                var responses = await Promise.all([
                    fetch('/api/monitoring/metrics'),
                    fetch('/api/social-pulse?limit=10'),
                    fetch('/api/hotelier-bets?limit=10'),
                    fetch('/api/monitoring/content?limit=30'),
                    fetch('/api/monitoring/scrapers')
                ]);
                var metricsRes = responses[0];
                var trendsRes = responses[1];
                var movesRes = responses[2];
                var contentRes = responses[3];
                var scrapersRes = responses[4];

                var metrics = await metricsRes.json();
                var trendsData = await trendsRes.json();
                var movesData = await movesRes.json();
                var contentData = await contentRes.json();
                var scrapers = await scrapersRes.json();

                // Update metrics
                document.getElementById('m-content').textContent = (metrics.total_content ? metrics.total_content.toLocaleString() : '0');
                document.getElementById('m-processed').textContent = (metrics.processed_content ? metrics.processed_content.toLocaleString() : '0');
                document.getElementById('m-trends').textContent = metrics.trends_count || '0';
                document.getElementById('m-moves').textContent = metrics.moves_count || '0';

                // Render trends
                allTrends = trendsData.trends || [];
                var trends = allTrends;
                if (trends.length > 0) {
                    document.getElementById('latest-trend').innerHTML = renderTrend(trends[0], 0);
                    var trendsHtml = '';
                    for (var ti = 0; ti < trends.length; ti++) { trendsHtml += renderTrend(trends[ti], ti); }
                    document.getElementById('trends-list').innerHTML = trendsHtml;
                } else {
                    document.getElementById('latest-trend').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends yet. Run POPULATE_DATA.bat</div>';
                    document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends yet</div>';
                }

                // Render moves
                allMoves = movesData.moves || [];
                var moves = allMoves;
                if (moves.length > 0) {
                    document.getElementById('latest-move').innerHTML = renderMove(moves[0], 0);
                    var movesHtml = '';
                    for (var mi = 0; mi < moves.length; mi++) { movesHtml += renderMove(moves[mi], mi); }
                    document.getElementById('moves-list').innerHTML = movesHtml;
                } else {
                    document.getElementById('latest-move').innerHTML = '<div class="empty"><div class="icon">♟️</div>No moves yet. Run POPULATE_DATA.bat</div>';
                    document.getElementById('moves-list').innerHTML = '<div class="empty"><div class="icon">♟️</div>No moves yet</div>';
                }

                // Render content
                var content = contentData.items || [];
                if (content.length > 0) {
                    var contentHtml = '';
                    for (var ci = 0; ci < content.length; ci++) { contentHtml += renderContent(content[ci]); }
                    document.getElementById('content-list').innerHTML = contentHtml;
                } else {
                    document.getElementById('content-list').innerHTML = '<div class="empty"><div class="icon">📰</div>No content yet</div>';
                }

                // Render scrapers
                if (scrapers.length > 0) {
                    var scrapersHtml = '<table><thead><tr><th>Source</th><th>Total</th><th>Last Run</th><th>New Items</th><th>Status</th></tr></thead><tbody>';
                    for (var si = 0; si < scrapers.length; si++) {
                        var s = scrapers[si];
                        var lastRunTime = s.last_run_at ? new Date(s.last_run_at).toLocaleString() : 'Never';
                        var statusClass = s.last_run_status === 'completed' ? 'success' : 'warning';
                        var statusText = s.last_run_status || 'N/A';
                        scrapersHtml += '<tr><td><strong>' + s.source + '</strong></td><td>' + s.total_items.toLocaleString() + '</td><td>' + lastRunTime + '</td><td>' + (s.last_run_items || 0) + '</td><td><span class="badge badge-' + statusClass + '">' + statusText + '</span></td></tr>';
                    }
                    scrapersHtml += '</tbody></table>';
                    document.getElementById('scrapers-list').innerHTML = scrapersHtml;
                } else {
                    document.getElementById('scrapers-list').innerHTML = '<div class="empty"><div class="icon">🔧</div>No scraper data</div>';
                }

                // Load scanned properties (don't wait for it)
                loadScannedProperties();

                setStatus('✅', 'Data loaded at ' + new Date().toLocaleTimeString());

            } catch (err) {
                console.error('Load error:', err);
                setStatus('❌', 'Error: ' + err.message);
                document.getElementById('latest-trend').innerHTML = '<div class="error">Failed to load data: ' + err.message + '</div>';
                document.getElementById('latest-move').innerHTML = '<div class="error">Failed to load data: ' + err.message + '</div>';
            }
        }

        function renderTrend(t, idx) {
            var name = t.name || t.trend_name || 'Unnamed Trend';
            var score = t.strength_score ? Math.round(t.strength_score * 100) + '%' : 'N/A';
            var whiteSpace = t.white_space_score ? Math.round(t.white_space_score * 100) : 0;
            var wsClass = whiteSpace >= 70 ? 'white-space-high' : whiteSpace >= 40 ? 'white-space-medium' : 'white-space-low';
            var wsLabel = whiteSpace >= 70 ? 'High Opportunity' : whiteSpace >= 40 ? 'Moderate' : 'Low';
            var isSaved = isProjectSaved(t.id);
            var savedClass = isSaved ? 'saved' : '';
            var savedText = isSaved ? '✓ Saved' : '💾 Save';

            return '<div class="trend-card" data-trend-id="' + t.id + '">' +
                '<div onclick="openTrendModal(' + idx + ')" style="cursor:pointer;">' +
                '<h3>' + truncate(name, 60) + '</h3>' +
                '<p>' + truncate(t.description || t.why_it_matters || '', 200) + '</p>' +
                '<div class="trend-meta">' +
                '<span class="white-space-badge ' + wsClass + '">' + whiteSpace + '% ' + wsLabel + '</span>' +
                ' | Strength: ' + score + ' | ' + (t.volume || 0) + ' sources' +
                '</div></div>' +
                '<div class="trend-actions">' +
                '<button class="trend-action-btn btn-save ' + savedClass + '" onclick="event.stopPropagation(); toggleSaveProject(' + idx + ')">' + savedText + '</button>' +
                '<button class="trend-action-btn btn-brand" onclick="event.stopPropagation(); turnIntoBrand(' + idx + ')">Build a Brand</button>' +
                '</div></div>';
        }

        function renderMove(m, idx) {
            var isSaved = isMoveSaved(m.id);
            var moveTypeBadge = m.move_type ? '<span class="move-type-badge">' + m.move_type.replace('_', ' ') + '</span>' : '';
            var marketBadge = m.market ? '<span class="market-badge">📍 ' + m.market + '</span>' : '';
            var savedClass = isSaved ? 'saved' : '';
            var savedText = isSaved ? '✓ Saved' : '💾 Save';

            return '<div class="move-card" onclick="openMoveModal(' + idx + ')">' +
                '<div class="move-badges">' + moveTypeBadge + marketBadge + '</div>' +
                '<h3>' + truncate(m.title || 'Untitled', 60) + '</h3>' +
                '<div class="company">' + (m.company || 'Unknown') + '</div>' +
                '<p>' + truncate(m.summary || m.why_it_matters || '', 180) + '</p>' +
                '<div class="move-actions">' +
                '<button class="move-action-btn btn-save ' + savedClass + '" onclick="event.stopPropagation(); toggleSaveMove(' + idx + ')">' + savedText + '</button>' +
                '</div></div>';
        }

        function renderContent(c) {
            var dateStr = c.published_at ? ' • ' + new Date(c.published_at).toLocaleDateString() : '';
            return '<div class="content-item">' +
                '<h4>' + truncate(c.title || 'Untitled', 70) + '</h4>' +
                '<p>' + truncate(c.content || '', 150) + '</p>' +
                '<div class="meta"><span class="source">' + (c.source || 'unknown') + '</span>' + dateStr + '</div>' +
                '</div>';
        }

        // =============================================
        // Demand Scan Functions
        // =============================================

        var allProperties = [];

        async function scanProperty() {
            var urlInput = document.getElementById('property-url-input');
            var url = urlInput.value.trim();

            if (!url) {
                showScanStatus('error', 'Please enter a valid URL');
                return;
            }

            // Validate URL format
            try {
                new URL(url);
            } catch (e) {
                showScanStatus('error', 'Invalid URL format. Please enter a complete URL including https://');
                return;
            }

            var scanBtn = document.getElementById('scan-btn');
            scanBtn.disabled = true;
            scanBtn.textContent = 'Scanning...';
            showScanStatus('info', 'Analyzing property website... This may take 30-60 seconds.');

            try {
                var response = await fetch('/api/demand-scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });

                var data = await response.json();

                if (response.ok) {
                    if (data.status === 'exists') {
                        showScanStatus('warning', 'Property was previously scanned. Showing existing analysis.');
                    } else {
                        showScanStatus('success', 'Property analyzed successfully!');
                    }
                    urlInput.value = '';
                    loadScannedProperties();
                } else {
                    showScanStatus('error', data.detail || 'Failed to scan property. Please check the URL and try again.');
                }
            } catch (err) {
                console.error('Scan error:', err);
                showScanStatus('error', 'Network error. Please try again.');
            } finally {
                scanBtn.disabled = false;
                scanBtn.textContent = 'Scan Property';
            }
        }

        function showScanStatus(type, message) {
            var statusEl = document.getElementById('scan-status');
            var colors = {
                error: { bg: 'rgba(194,90,120,0.12)', color: 'var(--bad)' },
                success: { bg: 'rgba(58,168,141,0.16)', color: 'var(--good)' },
                warning: { bg: 'rgba(184,134,46,0.16)', color: 'var(--warn)' },
                info: { bg: 'rgba(139,124,224,0.14)', color: 'var(--violet)' }
            };
            var style = colors[type] || colors.info;
            statusEl.style.display = 'block';
            statusEl.style.background = style.bg;
            statusEl.style.color = style.color;
            statusEl.textContent = message;

            if (type !== 'info') {
                setTimeout(function() { statusEl.style.display = 'none'; }, 5000);
            }
        }

        async function loadScannedProperties() {
            try {
                var response = await fetch('/api/demand-scan?limit=20');
                var data = await response.json();
                allProperties = data.properties || [];

                var container = document.getElementById('properties-list');
                if (allProperties.length > 0) {
                    var propsHtml = '';
                    for (var pi = 0; pi < allProperties.length; pi++) { propsHtml += renderPropertyCard(allProperties[pi], pi); }
                    container.innerHTML = propsHtml;
                } else {
                    container.innerHTML = '<div class="empty"><div class="icon">🏨</div>No properties scanned yet. Enter a URL above to analyze a property.</div>';
                }
            } catch (err) {
                console.error('Load properties error:', err);
            }
        }

        function renderPropertyCard(p, idx) {
            // Calculate demand score as 0-100
            var score = p.demand_fit_score ? Math.round(p.demand_fit_score * 100) : 0;
            var scoreClass = score >= 70 ? 'demand-high' : score >= 40 ? 'demand-medium' : 'demand-low';
            var scoreLabel = score >= 70 ? 'High Fit' : score >= 40 ? 'Moderate Fit' : 'Low Fit';

            // Experience gaps (top 3)
            var gaps = (p.experience_gaps || []).slice(0, 3);
            var gapsHtml = '';
            if (gaps.length > 0) {
                for (var gi = 0; gi < gaps.length; gi++) {
                    gapsHtml += '<span class="gap-item">' + truncate(gaps[gi].split(' (')[0], 50) + '</span>';
                }
            } else {
                gapsHtml = '<span style="color:var(--ink-3);font-size:0.85em;">No major gaps identified</span>';
            }

            // Opportunity lanes (top 2)
            var opportunities = (p.opportunity_lanes || []).slice(0, 2);
            var oppsHtml = '';
            if (opportunities.length > 0) {
                for (var oi = 0; oi < opportunities.length; oi++) {
                    oppsHtml += '<div class="opportunity-item">' + truncate(opportunities[oi], 120) + '</div>';
                }
            } else {
                oppsHtml = '<span style="color:var(--ink-3);font-size:0.85em;">No opportunities identified</span>';
            }

            // Misalignment flags
            var flags = p.positioning_misalignment_flags || [];
            var flagsHtml = '';
            if (flags.length > 0) {
                flagsHtml = '<div class="property-section"><div class="property-section-title">Positioning Issues</div>';
                for (var fi = 0; fi < flags.length; fi++) {
                    var flagText = flags[fi].split(':')[1] || flags[fi];
                    flagsHtml += '<span class="misalignment-flag">' + truncate(flagText, 100) + '</span>';
                }
                flagsHtml += '</div>';
            }

            // Property themes
            var themes = (p.themes || []).slice(0, 4);
            var themesHtml = '';
            if (themes.length > 0) {
                for (var thi = 0; thi < themes.length; thi++) {
                    themesHtml += '<span class="topic-tag">' + themes[thi] + '</span>';
                }
            }

            return '<div class="property-card" data-property-id="' + p.id + '">' +
                '<div class="property-card-header"><div>' +
                '<h3>' + (p.name || 'Unnamed Property') + '</h3>' +
                '<div class="location">' + (p.location || p.region || 'Location unknown') + '</div>' +
                '<div style="margin-top:8px;">' + themesHtml + '</div></div>' +
                '<div class="demand-score ' + scoreClass + '">' + score + '% ' + scoreLabel + '</div></div>' +
                flagsHtml +
                '<div class="property-section"><div class="property-section-title">Experience Gaps</div>' + gapsHtml + '</div>' +
                '<div class="property-section"><div class="property-section-title">Opportunity Lanes</div>' + oppsHtml + '</div>' +
                '<div class="property-actions">' +
                '<button class="property-action-btn btn-brand" onclick="sendPropertyToBuildBrand(' + idx + ')">Build a Brand</button>' +
                '<button class="property-action-btn btn-save" onclick="savePropertyToProject(' + idx + ')">💾 Save to Project</button>' +
                '<a href="' + p.url + '" target="_blank" class="property-action-btn btn-save" style="text-decoration:none;">🔗 Visit Site</a>' +
                '</div></div>';
        }

        function sendPropertyToBuildBrand(idx) {
            var p = allProperties[idx];
            if (!p) return;

            // Store property data for Build a Brand page
            var brandData = {
                type: 'property',
                property_name: p.name,
                location: p.location || '',
                segment: p.price_segment || (p.themes && p.themes[0]) || '',
                context: 'Property analysis of ' + p.name + '. Demand fit: ' + Math.round((p.demand_fit_score || 0) * 100) + '%',
                gaps: p.experience_gaps || [],
                opportunities: p.opportunity_lanes || [],
                themes: p.themes || []
            };
            localStorage.setItem('brandclave_brand_prefill', JSON.stringify(brandData));

            // Navigate to Build a Brand page
            window.location.href = '/monitoring/build-a-brand';
        }

        function savePropertyToProject(idx) {
            var p = allProperties[idx];
            if (!p) return;

            // Get existing saved properties
            var saved = JSON.parse(localStorage.getItem('brandclave_saved_properties') || '[]');

            // Check if already saved
            var existingIdx = -1;
            for (var i = 0; i < saved.length; i++) {
                if (saved[i].id === p.id) { existingIdx = i; break; }
            }
            if (existingIdx >= 0) {
                // Remove if already saved
                saved.splice(existingIdx, 1);
                localStorage.setItem('brandclave_saved_properties', JSON.stringify(saved));
                loadScannedProperties();
                return;
            }

            // Add to saved
            saved.push({
                id: p.id,
                type: 'property',
                name: p.name,
                location: p.location,
                demand_fit_score: p.demand_fit_score,
                saved_at: new Date().toISOString()
            });
            localStorage.setItem('brandclave_saved_properties', JSON.stringify(saved));
            loadScannedProperties();
        }

        // Allow Enter key to trigger scan
        document.addEventListener('DOMContentLoaded', function() {
            var urlInput = document.getElementById('property-url-input');
            if (urlInput) {
                urlInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') scanProperty();
                });
            }
        });

        // =============================================
        // Filter Functions
        // =============================================
        var currentFilters = { region: '', audience: '', time: '' };

        async function loadFilterOptions() {
            try {
                var filterResponses = await Promise.all([
                    fetch('/api/social-pulse/regions'),
                    fetch('/api/social-pulse/audiences')
                ]);
                var regionsRes = filterResponses[0];
                var audiencesRes = filterResponses[1];

                var regionsData = await regionsRes.json();
                var audiencesData = await audiencesRes.json();

                // Populate region dropdown
                var regionSelect = document.getElementById('filter-region');
                regionSelect.innerHTML = '<option value="">All Regions</option>';
                var regArr = regionsData.regions || [];
                for (var ri = 0; ri < regArr.length; ri++) {
                    var r = regArr[ri];
                    if (r.region) {
                        regionSelect.innerHTML += '<option value="' + r.region + '">' + r.region + ' (' + r.count + ')</option>';
                    }
                }

                // Populate audience dropdown
                var audienceSelect = document.getElementById('filter-audience');
                audienceSelect.innerHTML = '<option value="">All Segments</option>';
                var audArr = audiencesData.audiences || [];
                for (var ai = 0; ai < audArr.length; ai++) {
                    var a = audArr[ai];
                    if (a.segment) {
                        audienceSelect.innerHTML += '<option value="' + a.segment + '">' + a.segment + ' (' + a.count + ')</option>';
                    }
                }
            } catch (err) {
                console.error('Failed to load filter options:', err);
            }
        }

        async function applyFilters() {
            currentFilters.region = document.getElementById('filter-region').value;
            currentFilters.audience = document.getElementById('filter-audience').value;
            currentFilters.time = document.getElementById('filter-time').value;
            await loadTrendsWithFilters();
        }

        async function loadTrendsWithFilters() {
            var params = new URLSearchParams({ limit: '20' });
            if (currentFilters.region) params.append('region', currentFilters.region);
            if (currentFilters.audience) params.append('audience', currentFilters.audience);

            document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">⏳</div>Loading...</div>';

            try {
                var res = await fetch('/api/social-pulse?' + params.toString());
                var data = await res.json();

                var trends = data.trends || [];

                // Client-side time filtering
                if (currentFilters.time) {
                    var daysAgo = parseInt(currentFilters.time);
                    var cutoff = new Date();
                    cutoff.setDate(cutoff.getDate() - daysAgo);
                    var filteredTrends = [];
                    for (var fi = 0; fi < trends.length; fi++) {
                        if (!trends[fi].first_seen || new Date(trends[fi].first_seen) >= cutoff) {
                            filteredTrends.push(trends[fi]);
                        }
                    }
                    trends = filteredTrends;
                }

                allTrends = trends;

                if (trends.length > 0) {
                    var trendsHtml = '';
                    for (var ti = 0; ti < trends.length; ti++) { trendsHtml += renderTrend(trends[ti], ti); }
                    document.getElementById('trends-list').innerHTML = trendsHtml;
                } else {
                    document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends match your filters</div>';
                }
            } catch (err) {
                document.getElementById('trends-list').innerHTML = '<div class="error">Failed to load trends: ' + err.message + '</div>';
            }
        }

        function resetFilters() {
            document.getElementById('filter-region').value = '';
            document.getElementById('filter-audience').value = '';
            document.getElementById('filter-time').value = '';
            currentFilters = { region: '', audience: '', time: '' };
            loadTrendsWithFilters();
        }

        // =============================================
        // Move Filter Functions
        // =============================================
        var moveFilters = { company: '', move_type: '', market: '' };

        async function loadMoveFilterOptions() {
            try {
                var moveFilterResponses = await Promise.all([
                    fetch('/api/hotelier-bets/companies'),
                    fetch('/api/hotelier-bets/move-types'),
                    fetch('/api/hotelier-bets/markets')
                ]);
                var companiesRes = moveFilterResponses[0];
                var moveTypesRes = moveFilterResponses[1];
                var marketsRes = moveFilterResponses[2];

                var companiesData = await companiesRes.json();
                var moveTypesData = await moveTypesRes.json();
                var marketsData = await marketsRes.json();

                // Populate company dropdown
                var companySelect = document.getElementById('filter-company');
                companySelect.innerHTML = '<option value="">All Companies</option>';
                var compArr = companiesData.companies || [];
                for (var ci = 0; ci < compArr.length; ci++) {
                    companySelect.innerHTML += '<option value="' + compArr[ci] + '">' + compArr[ci] + '</option>';
                }

                // Populate move type dropdown
                var moveTypeSelect = document.getElementById('filter-move-type');
                moveTypeSelect.innerHTML = '<option value="">All Move Types</option>';
                var mtArr = moveTypesData.move_types || [];
                for (var mti = 0; mti < mtArr.length; mti++) {
                    var mt = mtArr[mti];
                    // Handle both object format {type, label} and string format
                    var mtValue = typeof mt === 'object' ? mt.type : mt;
                    var mtDisplay = typeof mt === 'object' ? mt.label : mt.replace('_', ' ').replace(/\b\w/g, function(l) { return l.toUpperCase(); });
                    if (mtValue) moveTypeSelect.innerHTML += '<option value="' + mtValue + '">' + mtDisplay + '</option>';
                }

                // Populate market dropdown
                var marketSelect = document.getElementById('filter-market');
                marketSelect.innerHTML = '<option value="">All Markets</option>';
                var marketArr = marketsData.markets || [];
                for (var mi = 0; mi < marketArr.length; mi++) {
                    if (marketArr[mi]) marketSelect.innerHTML += '<option value="' + marketArr[mi] + '">' + marketArr[mi] + '</option>';
                }
            } catch (err) {
                console.error('Failed to load move filter options:', err);
            }
        }

        async function applyMoveFilters() {
            moveFilters.company = document.getElementById('filter-company').value;
            moveFilters.move_type = document.getElementById('filter-move-type').value;
            moveFilters.market = document.getElementById('filter-market').value;
            await loadMovesWithFilters();
        }

        async function loadMovesWithFilters() {
            var params = new URLSearchParams({ limit: '20' });
            if (moveFilters.company) params.append('company', moveFilters.company);
            if (moveFilters.move_type) params.append('move_type', moveFilters.move_type);
            if (moveFilters.market) params.append('market', moveFilters.market);

            document.getElementById('moves-list').innerHTML = '<div class="empty"><div class="icon">⏳</div>Loading...</div>';

            try {
                var res = await fetch('/api/hotelier-bets?' + params.toString());
                var data = await res.json();

                allMoves = data.moves || [];

                if (allMoves.length > 0) {
                    var movesHtml = '';
                    for (var mi = 0; mi < allMoves.length; mi++) { movesHtml += renderMove(allMoves[mi], mi); }
                    document.getElementById('moves-list').innerHTML = movesHtml;
                } else {
                    document.getElementById('moves-list').innerHTML = '<div class="empty"><div class="icon">♟️</div>No moves match your filters</div>';
                }
                updateMovesSavedCount();
            } catch (err) {
                document.getElementById('moves-list').innerHTML = '<div class="error">Failed to load moves: ' + err.message + '</div>';
            }
        }

        function resetMoveFilters() {
            document.getElementById('filter-company').value = '';
            document.getElementById('filter-move-type').value = '';
            document.getElementById('filter-market').value = '';
            moveFilters = { company: '', move_type: '', market: '' };
            loadMovesWithFilters();
        }

        // =============================================
        // LocalStorage Save to Project Functions
        // =============================================
        var STORAGE_KEY = 'brandclave_saved_trends';

        function getSavedProjects() {
            try {
                var data = localStorage.getItem(STORAGE_KEY);
                return data ? JSON.parse(data) : [];
            } catch (e) {
                console.error('Error reading saved projects:', e);
                return [];
            }
        }

        function saveProject(trend) {
            var saved = getSavedProjects();
            for (var i = 0; i < saved.length; i++) { if (saved[i].id === trend.id) return false; }

            saved.push({
                id: trend.id,
                name: trend.name || trend.trend_name,
                description: trend.description,
                white_space_score: trend.white_space_score,
                strength_score: trend.strength_score,
                region: trend.region,
                audience_segment: trend.audience_segment,
                topics: trend.topics,
                saved_at: new Date().toISOString()
            });

            localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
            return true;
        }

        function removeProject(trendId) {
            var saved = getSavedProjects();
            var filtered = [];
            for (var i = 0; i < saved.length; i++) { if (saved[i].id !== trendId) filtered.push(saved[i]); }
            localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
        }

        function isProjectSaved(trendId) {
            var saved = getSavedProjects();
            for (var i = 0; i < saved.length; i++) { if (saved[i].id === trendId) return true; }
            return false;
        }

        function toggleSaveProject(idx) {
            var trend = allTrends[idx];
            if (!trend) return;

            if (isProjectSaved(trend.id)) {
                removeProject(trend.id);
            } else {
                saveProject(trend);
            }

            // Re-render the trends list
            var trendsHtml = '';
            for (var i = 0; i < allTrends.length; i++) { trendsHtml += renderTrend(allTrends[i], i); }
            document.getElementById('trends-list').innerHTML = trendsHtml;
            updateSavedCount();
            renderMyProjects(); // Auto-update My Projects tab
        }

        function updateSavedCount() {
            var count = getSavedProjects().length;
            var countEl = document.getElementById('saved-count');
            if (countEl) {
                countEl.textContent = count > 0 ? count + ' saved' : '';
            }
        }

        // =============================================
        // LocalStorage Save Moves Functions
        // =============================================
        var MOVES_STORAGE_KEY = 'brandclave_saved_moves';

        function getSavedMoves() {
            try {
                var data = localStorage.getItem(MOVES_STORAGE_KEY);
                return data ? JSON.parse(data) : [];
            } catch (e) {
                console.error('Error reading saved moves:', e);
                return [];
            }
        }

        function saveMove(move) {
            var saved = getSavedMoves();
            for (var i = 0; i < saved.length; i++) { if (saved[i].id === move.id) return false; }

            saved.push({
                id: move.id,
                title: move.title,
                summary: move.summary,
                company: move.company,
                move_type: move.move_type,
                market: move.market,
                strategic_implications: move.strategic_implications,
                source_name: move.source_name,
                saved_at: new Date().toISOString()
            });

            localStorage.setItem(MOVES_STORAGE_KEY, JSON.stringify(saved));
            return true;
        }

        function removeMove(moveId) {
            var saved = getSavedMoves();
            var filtered = [];
            for (var i = 0; i < saved.length; i++) { if (saved[i].id !== moveId) filtered.push(saved[i]); }
            localStorage.setItem(MOVES_STORAGE_KEY, JSON.stringify(filtered));
        }

        function isMoveSaved(moveId) {
            var saved = getSavedMoves();
            for (var i = 0; i < saved.length; i++) { if (saved[i].id === moveId) return true; }
            return false;
        }

        function toggleSaveMove(idx) {
            var move = allMoves[idx];
            if (!move) return;

            if (isMoveSaved(move.id)) {
                removeMove(move.id);
            } else {
                saveMove(move);
            }

            // Re-render the moves list
            var movesHtml = '';
            for (var i = 0; i < allMoves.length; i++) { movesHtml += renderMove(allMoves[i], i); }
            document.getElementById('moves-list').innerHTML = movesHtml;
            updateMovesSavedCount();
            renderMyProjects(); // Auto-update My Projects tab
        }

        function updateMovesSavedCount() {
            var count = getSavedMoves().length;
            var countEl = document.getElementById('moves-saved-count');
            if (countEl) {
                countEl.textContent = count > 0 ? count + ' saved' : '';
            }
        }

        // =============================================
        // My Projects Functions
        // =============================================
        function renderMyProjects() {
            var savedTrends = getSavedProjects();
            var savedMoves = getSavedMoves();

            // Update counts
            document.getElementById('saved-trends-count').textContent = savedTrends.length > 0 ? '(' + savedTrends.length + ')' : '';
            document.getElementById('saved-moves-count').textContent = savedMoves.length > 0 ? '(' + savedMoves.length + ')' : '';

            // Render saved trends
            var trendsListEl = document.getElementById('saved-trends-list');
            if (savedTrends.length > 0) {
                var trendsHtml = '';
                for (var i = 0; i < savedTrends.length; i++) {
                    var t = savedTrends[i];
                    var tMeta = (t.region ? t.region + ' • ' : '') + (t.audience_segment || 'General');
                    if (t.white_space_score) tMeta += ' • White Space: ' + (t.white_space_score * 100).toFixed(0) + '%';
                    trendsHtml += '<div class="saved-item-card"><div><h4>' + (t.name || 'Unnamed Trend') + '</h4><div class="saved-item-meta">' + tMeta + '</div></div><div class="saved-item-actions"><button class="btn-remove" onclick="removeSavedTrend(\\'' + t.id + '\\')">Remove</button></div></div>';
                }
                trendsListEl.innerHTML = trendsHtml;
            } else {
                trendsListEl.innerHTML = '<div class="empty"><div class="icon">📊</div>No saved trends yet</div>';
            }

            // Render saved moves
            var movesListEl = document.getElementById('saved-moves-list');
            if (savedMoves.length > 0) {
                var movesHtml = '';
                for (var j = 0; j < savedMoves.length; j++) {
                    var m = savedMoves[j];
                    var mMeta = (m.company || 'Unknown') + ' • ' + (m.move_type ? m.move_type.replace('_', ' ') : 'Move');
                    if (m.market) mMeta += ' • ' + m.market;
                    movesHtml += '<div class="saved-item-card"><div><h4>' + (m.title || 'Unnamed Move') + '</h4><div class="saved-item-meta">' + mMeta + '</div></div><div class="saved-item-actions"><button class="btn-remove" onclick="removeSavedMove(\\'' + m.id + '\\')">Remove</button></div></div>';
                }
                movesListEl.innerHTML = movesHtml;
            } else {
                movesListEl.innerHTML = '<div class="empty"><div class="icon">♟️</div>No saved moves yet</div>';
            }

            // Load and render blueprints from database
            loadMyBlueprints();

            // Update profile insights
            updateProfileInsights(savedTrends, savedMoves);

            // Enable/disable build button
            var buildBtn = document.getElementById('build-from-profile-btn');
            buildBtn.disabled = (savedTrends.length + savedMoves.length) === 0;
        }

        function updateProfileInsights(trends, moves) {
            var profileEl = document.getElementById('profile-content');

            if (trends.length === 0 && moves.length === 0) {
                profileEl.innerHTML = '<div class="empty" style="color:rgba(255,255,255,0.8);"><div class="icon">💡</div>Save trends and moves to build your profile</div>';
                return;
            }

            // Analyze patterns
            var regions = {};
            var segments = {};
            var topics = {};
            var companies = {};
            var moveTypes = {};
            var markets = {};

            // From trends
            for (var ti = 0; ti < trends.length; ti++) {
                var t = trends[ti];
                if (t.region) regions[t.region] = (regions[t.region] || 0) + 1;
                if (t.audience_segment) segments[t.audience_segment] = (segments[t.audience_segment] || 0) + 1;
                var tTopics = t.topics || [];
                for (var tpi = 0; tpi < tTopics.length; tpi++) {
                    topics[tTopics[tpi]] = (topics[tTopics[tpi]] || 0) + 1;
                }
            }

            // From moves
            for (var mi = 0; mi < moves.length; mi++) {
                var m = moves[mi];
                if (m.company) companies[m.company] = (companies[m.company] || 0) + 1;
                if (m.move_type) moveTypes[m.move_type] = (moveTypes[m.move_type] || 0) + 1;
                if (m.market) markets[m.market] = (markets[m.market] || 0) + 1;
            }

            // Sort by frequency and take top items
            function sortByFreq(obj) {
                var entries = [];
                for (var k in obj) { if (obj.hasOwnProperty(k)) entries.push([k, obj[k]]); }
                entries.sort(function(a, b) { return b[1] - a[1]; });
                return entries;
            }
            var topRegions = sortByFreq(regions).slice(0, 3);
            var topSegments = sortByFreq(segments).slice(0, 3);
            var topTopics = sortByFreq(topics).slice(0, 5);
            var topCompanies = sortByFreq(companies).slice(0, 3);
            var topMoveTypes = sortByFreq(moveTypes).slice(0, 3);
            var topMarkets = sortByFreq(markets).slice(0, 3);

            var html = '';

            if (topRegions.length > 0 || topMarkets.length > 0) {
                var allLocations = topRegions.concat(topMarkets).slice(0, 4);
                var locHtml = '';
                for (var li = 0; li < allLocations.length; li++) { locHtml += '<span class="profile-tag">' + allLocations[li][0] + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">📍 Locations of Interest</div>' + locHtml + '</div>';
            }

            if (topSegments.length > 0) {
                var segHtml = '';
                for (var si = 0; si < topSegments.length; si++) { segHtml += '<span class="profile-tag">' + topSegments[si][0] + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">👥 Target Segments</div>' + segHtml + '</div>';
            }

            if (topTopics.length > 0) {
                var topHtml = '';
                for (var toi = 0; toi < topTopics.length; toi++) { topHtml += '<span class="profile-tag">' + topTopics[toi][0] + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">🔥 Key Themes</div>' + topHtml + '</div>';
            }

            if (topCompanies.length > 0) {
                var coHtml = '';
                for (var coi = 0; coi < topCompanies.length; coi++) { coHtml += '<span class="profile-tag">' + topCompanies[coi][0] + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">Companies Watched</div>' + coHtml + '</div>';
            }

            if (topMoveTypes.length > 0) {
                var mtHtml = '';
                for (var mti = 0; mti < topMoveTypes.length; mti++) { mtHtml += '<span class="profile-tag">' + topMoveTypes[mti][0].replace('_', ' ') + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">Move Types</div>' + mtHtml + '</div>';
            }

            profileEl.innerHTML = html || '<div style="opacity:0.8;">Collecting insights...</div>';
        }

        function removeSavedTrend(trendId) {
            removeProject(trendId);
            renderMyProjects();
            updateSavedCount();
            // Re-render trends if visible
            if (allTrends.length > 0) {
                var trendsHtml = '';
                for (var i = 0; i < allTrends.length; i++) { trendsHtml += renderTrend(allTrends[i], i); }
                document.getElementById('trends-list').innerHTML = trendsHtml;
            }
        }

        function removeSavedMove(moveId) {
            removeMove(moveId);
            renderMyProjects();
            updateMovesSavedCount();
            // Re-render moves if visible
            if (allMoves.length > 0) {
                var movesHtml = '';
                for (var i = 0; i < allMoves.length; i++) { movesHtml += renderMove(allMoves[i], i); }
                document.getElementById('moves-list').innerHTML = movesHtml;
            }
        }

        function clearAllSaved() {
            if (!confirm('Are you sure you want to clear all saved items?')) return;
            localStorage.removeItem(STORAGE_KEY);
            localStorage.removeItem(MOVES_STORAGE_KEY);
            renderMyProjects();
            updateSavedCount();
            updateMovesSavedCount();
            // Re-render lists
            if (allTrends.length > 0) {
                var trendsHtml = '';
                for (var i = 0; i < allTrends.length; i++) { trendsHtml += renderTrend(allTrends[i], i); }
                document.getElementById('trends-list').innerHTML = trendsHtml;
            }
            if (allMoves.length > 0) {
                var movesHtml = '';
                for (var j = 0; j < allMoves.length; j++) { movesHtml += renderMove(allMoves[j], j); }
                document.getElementById('moves-list').innerHTML = movesHtml;
            }
        }

        // =============================================
        // My Blueprints Functions
        // =============================================
        var allBlueprints = [];

        async function loadMyBlueprints() {
            var listEl = document.getElementById('my-blueprints-list');
            var countEl = document.getElementById('saved-blueprints-count');
            if (!listEl) return;

            try {
                var response = await fetch('/api/brand-blueprint?limit=20');
                var data = await response.json();

                allBlueprints = data.blueprints || [];
                countEl.textContent = allBlueprints.length > 0 ? '(' + allBlueprints.length + ')' : '';

                if (allBlueprints.length === 0) {
                    listEl.innerHTML = '<div class="empty"><div class="icon">🏨</div>No blueprints yet. Create your first brand concept!</div>';
                    return;
                }

                var html = '';
                for (var i = 0; i < allBlueprints.length; i++) {
                    var bp = allBlueprints[i];
                    var names = bp.brand_names || {};
                    var inputs = bp.inputs || {};
                    var created = new Date(bp.generated_at).toLocaleDateString();
                    var confidence = Math.round((bp.confidence || 0) * 100);

                    html += '<div class="blueprint-card-mini" style="background:var(--surface);border:1px solid var(--surface-3);border-radius:10px;padding:15px;transition:all 0.2s;">' +
                        '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">' +
                        '<h4 style="margin:0;color:var(--ink);font-size:1.1em;">' + (names.primary || 'Unnamed Brand') + '</h4>' +
                        '<span style="background:var(--good);color:var(--ink);padding:2px 8px;border-radius:4px;font-size:0.75em;">' + confidence + '%</span>' +
                        '</div>' +
                        '<p style="font-size:0.85em;color:var(--ink-2);margin:0 0 10px 0;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">' + (bp.one_liner || '') + '</p>' +
                        '<div style="font-size:0.8em;color:var(--ink-3);margin-bottom:12px;">' +
                        '<span>' + (inputs.location || '-') + '</span> • ' +
                        '<span>' + (inputs.segment || '-') + '</span> • ' +
                        '<span>$' + (inputs.adr || '-') + ' ADR</span>' +
                        '</div>' +
                        '<div style="display:flex;gap:8px;justify-content:space-between;align-items:center;">' +
                        '<span style="font-size:0.75em;color:var(--ink-3);">' + created + '</span>' +
                        '<div style="display:flex;gap:8px;">' +
                        '<button onclick="viewBlueprint(\\'' + bp.id + '\\')" style="padding:6px 12px;background:var(--gold);color:var(--ink);border:none;border-radius:6px;cursor:pointer;font-size:0.85em;">View</button>' +
                        '<button onclick="deleteBlueprint(\\'' + bp.id + '\\')" style="padding:6px 12px;background:var(--surface);color:var(--bad);border:1px solid var(--bad);border-radius:6px;cursor:pointer;font-size:0.85em;">Delete</button>' +
                        '</div>' +
                        '</div>' +
                        '</div>';
                }
                listEl.innerHTML = html;

                // Add hover effects
                var cards = listEl.querySelectorAll('.blueprint-card-mini');
                cards.forEach(function(card) {
                    card.addEventListener('mouseenter', function() {
                        this.style.borderColor = 'var(--gold)';
                        this.style.transform = 'translateY(-2px)';
                        this.style.boxShadow = '0 4px 15px rgba(233,69,96,0.15)';
                    });
                    card.addEventListener('mouseleave', function() {
                        this.style.borderColor = 'var(--surface-3)';
                        this.style.transform = 'translateY(0)';
                        this.style.boxShadow = 'none';
                    });
                });

            } catch (e) {
                console.error('Error loading blueprints:', e);
                listEl.innerHTML = '<div class="empty" style="color:var(--bad);"><div class="icon">⚠️</div>Failed to load blueprints</div>';
            }
        }

        function viewBlueprint(id) {
            // Navigate to Build a Brand page with the blueprint loaded
            window.location.href = '/api/monitoring/build-a-brand?blueprint=' + id;
        }

        async function deleteBlueprint(id) {
            if (!confirm('Are you sure you want to delete this blueprint?')) return;

            try {
                var response = await fetch('/api/brand-blueprint/' + id, { method: 'DELETE' });
                if (response.ok) {
                    loadMyBlueprints(); // Refresh the list
                } else {
                    alert('Failed to delete blueprint');
                }
            } catch (e) {
                console.error('Error deleting blueprint:', e);
                alert('Error deleting blueprint: ' + e.message);
            }
        }

        function buildBrandFromProfile() {
            var savedTrends = getSavedProjects();
            var savedMoves = getSavedMoves();

            if (savedTrends.length === 0 && savedMoves.length === 0) {
                alert('Save some trends or moves first to build your profile.');
                return;
            }

            // Helper to get unique values
            function getUnique(arr) {
                var seen = {};
                var result = [];
                for (var i = 0; i < arr.length; i++) {
                    if (arr[i] && !seen[arr[i]]) {
                        seen[arr[i]] = true;
                        result.push(arr[i]);
                    }
                }
                return result;
            }

            // Extract values from trends
            var trendRegions = [];
            var trendSegments = [];
            var trendTopics = [];
            for (var ti = 0; ti < savedTrends.length; ti++) {
                if (savedTrends[ti].region) trendRegions.push(savedTrends[ti].region);
                if (savedTrends[ti].audience_segment) trendSegments.push(savedTrends[ti].audience_segment);
                var topics = savedTrends[ti].topics || [];
                for (var tpi = 0; tpi < topics.length; tpi++) { trendTopics.push(topics[tpi]); }
            }

            // Extract values from moves
            var moveCompanies = [];
            var moveMarkets = [];
            var moveTypes = [];
            for (var mi = 0; mi < savedMoves.length; mi++) {
                if (savedMoves[mi].company) moveCompanies.push(savedMoves[mi].company);
                if (savedMoves[mi].market) moveMarkets.push(savedMoves[mi].market);
                if (savedMoves[mi].move_type) moveTypes.push(savedMoves[mi].move_type);
            }

            // Build profile data for brand generation
            var profileData = {
                trends: savedTrends,
                moves: savedMoves,
                regions: getUnique(trendRegions),
                segments: getUnique(trendSegments),
                topics: getUnique(trendTopics),
                companies: getUnique(moveCompanies),
                markets: getUnique(moveMarkets),
                move_types: getUnique(moveTypes)
            };

            // Store for Build a Brand page
            sessionStorage.setItem('brandclave_profile_data', JSON.stringify(profileData));
            sessionStorage.setItem('brandclave_brand_input', JSON.stringify({
                from_profile: true,
                initial_region: profileData.regions[0] || '',
                initial_segment: profileData.segments[0] || 'lifestyle',
                topics: profileData.topics.slice(0, 5)
            }));

            // Navigate to Build a Brand
            window.location.href = '/api/monitoring/build-a-brand';
        }

        // =============================================
        // Turn Into Brand Function
        // =============================================
        function turnIntoBrand(idx) {
            var trend = allTrends[idx];
            if (!trend) return;

            // Store trend data for the Build a Brand page
            var brandInput = {
                source_trend_id: trend.id,
                source_trend_name: trend.name || trend.trend_name,
                initial_segment: trend.audience_segment || 'lifestyle',
                initial_region: trend.region || '',
                topics: trend.topics || [],
                white_space_score: trend.white_space_score,
                description: trend.description,
                why_it_matters: trend.why_it_matters
            };

            // Store in sessionStorage
            sessionStorage.setItem('brandclave_brand_input', JSON.stringify(brandInput));

            // Navigate to Build a Brand page
            window.location.href = '/api/monitoring/build-a-brand';
        }

        // City Desires functions
        function quickCity(city, country) {
            document.getElementById('city-input').value = city;
            document.getElementById('country-input').value = country;
            analyzeCity();
        }

        // Event delegation for desire quotes interactions
        document.addEventListener('click', function(e) {
            // Handle source badge clicks
            if (e.target.classList.contains('source-badge')) {
                var cardId = e.target.getAttribute('data-card');
                var sourceName = e.target.getAttribute('data-source');
                filterDesireQuotes(cardId, sourceName);
            }
            // Handle expand button clicks
            if (e.target.classList.contains('expand-btn')) {
                var cardId = e.target.getAttribute('data-card');
                var moreCount = parseInt(e.target.getAttribute('data-count'));
                toggleDesireQuotes(cardId, moreCount, e.target);
            }
            // Handle show all button clicks
            if (e.target.classList.contains('show-all-btn')) {
                var cardId = e.target.getAttribute('data-card');
                filterDesireQuotes(cardId, 'all');
            }
        });

        // Toggle expand/collapse for desire quotes
        function toggleDesireQuotes(cardId, moreCount, btn) {
            var card = document.getElementById(cardId);
            if (!card) return;

            var allQuotes = card.querySelectorAll('.desire-quote');
            var hiddenQuotes = [];
            for (var i = 0; i < allQuotes.length; i++) {
                if (allQuotes[i].style.display === 'none') {
                    hiddenQuotes.push(allQuotes[i]);
                }
            }
            var isExpanded = hiddenQuotes.length === 0;

            if (isExpanded) {
                // Collapse: hide quotes beyond initial 2
                for (var i = 0; i < allQuotes.length; i++) {
                    if (i >= 2) allQuotes[i].style.display = 'none';
                }
                if (btn) btn.textContent = 'Show ' + moreCount + ' more quotes';
            } else {
                // Expand: show all quotes
                for (var i = 0; i < hiddenQuotes.length; i++) {
                    hiddenQuotes[i].style.display = 'block';
                }
                if (btn) btn.textContent = 'Show less';
            }
        }

        // Filter quotes by source when clicking source badge
        function filterDesireQuotes(cardId, sourceName) {
            var card = document.getElementById(cardId);
            if (!card) return;

            var allQuotes = card.querySelectorAll('.desire-quote');
            var expandBtn = card.querySelector('.expand-btn');

            if (sourceName === 'all') {
                // Show first 2, hide rest (reset to initial state)
                for (var i = 0; i < allQuotes.length; i++) {
                    allQuotes[i].style.display = i < 2 ? 'block' : 'none';
                }
                if (expandBtn) {
                    var hiddenCount = Math.max(0, allQuotes.length - 2);
                    expandBtn.style.display = hiddenCount > 0 ? 'block' : 'none';
                    expandBtn.textContent = 'Show ' + hiddenCount + ' more quotes';
                }
            } else {
                // Filter by source - show all matching, hide expand button
                for (var i = 0; i < allQuotes.length; i++) {
                    var qSource = allQuotes[i].getAttribute('data-source');
                    if (qSource === sourceName) {
                        allQuotes[i].style.display = 'block';
                    } else {
                        allQuotes[i].style.display = 'none';
                    }
                }
                if (expandBtn) {
                    expandBtn.style.display = 'none';
                }
            }
        }

        async function analyzeCity() {
            var city = document.getElementById('city-input').value.trim();
            var country = document.getElementById('country-input').value.trim();

            if (!city) {
                alert('Please enter a city name');
                return;
            }

            var btn = document.getElementById('analyze-btn');
            var resultsDiv = document.getElementById('city-results');

            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            resultsDiv.innerHTML = '<div class="empty"><div class="icon">⏳</div>Analyzing ' + city + '... This may take 60-120 seconds (using semantic clustering).</div>';

            try {
                // Use adaptive endpoint with semantic clustering for better results
                var response = await fetch('/api/city-desires/adaptive', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ city: city, country: country })
                });

                if (!response.ok) {
                    throw new Error('Analysis failed: ' + response.status);
                }

                var data = await response.json();
                renderCityResults(data);

            } catch (err) {
                resultsDiv.innerHTML = '<div class="error">Analysis failed: ' + err.message + '</div>';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Analyze';
            }
        }

        function renderCityResults(data) {
            var resultsDiv = document.getElementById('city-results');

            // Format sources summary
            var sourcesSummary = data.sources_summary || {};
            var sourcesEntries = [];
            for (var k in sourcesSummary) { if (sourcesSummary.hasOwnProperty(k)) sourcesEntries.push([k, sourcesSummary[k]]); }
            sourcesEntries.sort(function(a, b) { return b[1] - a[1]; });
            var sourcesHtml = sourcesEntries.map(function(entry) { return '<span style="background:var(--surface-2);padding:3px 8px;border-radius:12px;font-size:0.85em;margin-right:6px;">' + entry[0] + ': ' + entry[1] + '</span>'; }).join('');

            var html = '<div style="background:var(--surface-2);padding:15px;border-radius:8px;margin-bottom:20px;">' +
                '<h3 style="margin-bottom:10px;">' + data.city + ', ' + data.country + '</h3>' +
                '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:10px;">' +
                '<div><strong>' + (data.total_signals || 0) + '</strong> signals</div>' +
                '<div><strong>' + (data.num_learned_categories || 0) + '</strong> themes discovered</div>' +
                '<div>Confidence: <strong>' + ((data.model_confidence || 0) * 100).toFixed(0) + '%</strong></div>' +
                '</div>' +
                (sourcesHtml ? '<div style="margin-top:10px;">Sources: ' + sourcesHtml + '</div>' : '') +
                '</div>';

            // Top Desires with source attribution
            if (data.top_desires && data.top_desires.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">🔥 Top Desires</h3>';
                html += data.top_desires.slice(0, 5).map(function(d, idx) {
                    // Format sources as visual badges
                    var sourceColors = {
                        'reddit': '#ff4500',
                        'youtube': '#ff0000',
                        'tripadvisor': 'var(--good)',
                        'twitter': '#1da1f2',
                        'instagram': '#e1306c'
                    };
                    var cardId = 'desire-card-' + idx;
                    var sourceBadges = (d.sources || [])
                        .map(function(s) {
                            var color = sourceColors[s.name.toLowerCase()] || 'var(--ink-2)';
                            var sourceName = s.name.toLowerCase();
                            return '<span class="source-badge" data-card="' + cardId + '" data-source="' + sourceName + '" style="display:inline-block;background:' + color + ';color:var(--ink);padding:2px 8px;border-radius:12px;font-size:0.75em;margin-right:4px;cursor:pointer;">' + s.name + ' (' + s.count + ')</span>';
                        })
                        .join('') || '<span style="color:var(--ink-3);font-size:0.85em;">No source data</span>';

                    // Get all example snippets for expandable section
                    var examplesHtml = '';
                    if (d.example_snippets && d.example_snippets.length > 0) {
                        var allSnippets = d.example_snippets.slice(0, 10);
                        var initialCount = 2;

                        function renderQuote(snippet, hidden) {
                            var text = typeof snippet === 'string' ? snippet : (snippet.text || '');
                            var source = typeof snippet === 'object' ? snippet.source : 'traveler';
                            var sourceColor = sourceColors[source.toLowerCase()] || 'var(--ink-2)';
                            if (text) {
                                var hiddenStyle = hidden ? 'display:none;' : '';
                                return '<div class="desire-quote" data-source="' + source.toLowerCase() + '" style="' + hiddenStyle + 'margin-bottom:8px;padding:8px 10px;background:rgba(0,0,0,0.02);border-radius:4px;font-size:0.85em;border-left:3px solid ' + sourceColor + ';"><span style="font-style:italic;">"' + text.substring(0, 180) + '..."</span> <span style="color:' + sourceColor + ';font-weight:500;">- ' + source + '</span></div>';
                            }
                            return '';
                        }

                        // Initial visible quotes
                        var visibleQuotes = allSnippets.slice(0, initialCount).map(function(s) { return renderQuote(s, false); }).filter(function(h) { return h; }).join('');

                        // Hidden quotes (for expansion)
                        var hiddenQuotes = allSnippets.slice(initialCount).map(function(s) { return renderQuote(s, true); }).filter(function(h) { return h; }).join('');

                        var hasMore = allSnippets.length > initialCount;
                        var moreCount = allSnippets.length - initialCount;

                        if (visibleQuotes || hiddenQuotes) {
                            var expandBtnHtml = '';
                            if (hasMore) {
                                expandBtnHtml = '<div class="expand-btn" data-card="' + cardId + '" data-count="' + moreCount + '" style="margin-top:8px;padding:6px 12px;background:var(--surface-2);border-radius:4px;font-size:0.8em;color:var(--blue);cursor:pointer;text-align:center;">Show ' + moreCount + ' more quotes</div>';
                            }
                            examplesHtml = '<div style="margin-top:12px;" id="' + cardId + '-quotes">' +
                                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">' +
                                '<span style="font-size:0.8em;color:var(--ink-2);text-transform:uppercase;letter-spacing:0.5px;">What travelers are saying:</span>' +
                                '<span class="show-all-btn" data-card="' + cardId + '" style="font-size:0.75em;color:var(--blue);cursor:pointer;">Show all</span>' +
                                '</div>' +
                                '<div class="quotes-container">' + visibleQuotes + hiddenQuotes + '</div>' +
                                expandBtnHtml +
                                '</div>';
                        }
                    }

                    // Build insights section
                    var insightsHtml = '';
                    if (d.unmet_need) {
                        insightsHtml += '<div style="margin-top:10px;"><strong style="color:var(--bad);">Unmet Need:</strong> ' + d.unmet_need + '</div>';
                    }
                    if (d.why_supply_fails) {
                        insightsHtml += '<div style="margin-top:6px;"><strong style="color:var(--warn);">Why Supply Fails:</strong> ' + d.why_supply_fails + '</div>';
                    }
                    if (d.solving_features && d.solving_features.length > 0) {
                        insightsHtml += '<div style="margin-top:6px;"><strong style="color:var(--good);">What Would Solve This:</strong> ' + d.solving_features.slice(0, 3).join(' • ') + '</div>';
                    }
                    if (d.target_guest) {
                        insightsHtml += '<div style="margin-top:6px;"><strong style="color:var(--blue);">Target Guest:</strong> ' + d.target_guest + '</div>';
                    }

                    // Fallback to description if no structured insights
                    if (!insightsHtml && d.description) {
                        insightsHtml = '<div style="margin-top:8px;">' + d.description + '</div>';
                    }

                    return '<div class="desire-card" id="' + cardId + '">' +
                        '<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">' +
                        '<h4 style="margin:0;">' + (d.theme_name || d.theme || 'Desire') + '</h4>' +
                        '<div>' + sourceBadges + '</div>' +
                        '</div>' +
                        insightsHtml +
                        '<div class="desire-meta" style="margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-size:0.9em;color:var(--ink-2);">' +
                        '<strong>' + (d.frequency || 0) + '</strong> mentions • ' +
                        '<strong>' + ((d.intensity_score || 0) * 100).toFixed(0) + '%</strong> intensity' +
                        '</div>' +
                        examplesHtml +
                        '</div>';
                }).join('');
            }

            // White Space Opportunities
            if (data.white_space_opportunities && data.white_space_opportunities.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">💡 White Space Opportunities</h3>';
                html += data.white_space_opportunities.slice(0, 5).map(function(o) {
                    return '<div class="opportunity-card">' + o + '</div>';
                }).join('');
            }

            // Concept Lanes
            if (data.concept_lanes && data.concept_lanes.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">Concept Lanes</h3>';
                html += data.concept_lanes.slice(0, 3).map(function(c) {
                    var name = c.name || c.concept || 'Hotel Concept';
                    var positioning = c.positioning || c.rationale || '';
                    var solves = c.solves || '';
                    var differentiators = c.key_differentiators || c.key_features || [];
                    var targetGuest = c.target_guest || '';
                    var pricePosition = c.price_position || '';
                    var whyWins = c.why_it_wins || '';

                    var detailsHtml = '';
                    if (positioning) {
                        detailsHtml += '<div style="margin-top:8px;font-style:italic;color:var(--ink-2);">' + positioning + '</div>';
                    }
                    if (solves) {
                        detailsHtml += '<div style="margin-top:8px;"><strong>Solves:</strong> ' + solves + '</div>';
                    }
                    if (differentiators.length > 0) {
                        detailsHtml += '<div style="margin-top:8px;"><strong>Key Differentiators:</strong> ' + differentiators.slice(0, 3).join(' • ') + '</div>';
                    }
                    if (targetGuest) {
                        detailsHtml += '<div style="margin-top:6px;"><strong>Target:</strong> ' + targetGuest + '</div>';
                    }
                    if (pricePosition) {
                        detailsHtml += '<span style="display:inline-block;margin-top:8px;background:var(--surface-2);padding:3px 10px;border-radius:12px;font-size:0.85em;">' + pricePosition + '</span>';
                    }
                    if (whyWins) {
                        detailsHtml += '<div style="margin-top:10px;padding:8px;background:var(--surface-2);border-radius:4px;font-size:0.9em;"><strong>Why it wins:</strong> ' + whyWins + '</div>';
                    }

                    return '<div class="concept-card"><h4>' + name + '</h4>' + detailsHtml + '</div>';
                }).join('');
            }

            // Underserved Segments
            if (data.underserved_segments && data.underserved_segments.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">👥 Underserved Segments</h3>';
                html += '<div style="display:flex;flex-wrap:wrap;gap:8px;">';
                html += data.underserved_segments.map(function(s) {
                    return '<span style="background:var(--surface-3);padding:5px 12px;border-radius:20px;font-size:0.9em;">' + s + '</span>';
                }).join('');
                html += '</div>';
            }

            resultsDiv.innerHTML = html;
        }

        // Chat functionality
        var chatConversationId = null;

        function buildProfileContext() {
            var savedTrends = getSavedProjects();
            var savedMoves = getSavedMoves();

            if (savedTrends.length === 0 && savedMoves.length === 0) {
                return null;
            }

            // Helper function to get unique values
            function getUniqueVals(arr) {
                var seen = {};
                var result = [];
                for (var i = 0; i < arr.length; i++) {
                    if (arr[i] && !seen[arr[i]]) {
                        seen[arr[i]] = true;
                        result.push(arr[i]);
                    }
                }
                return result;
            }

            var context = 'User research profile: ';

            // Trends summary
            if (savedTrends.length > 0) {
                var trendNames = [];
                var trendRegions = [];
                var trendSegments = [];
                var trendTopics = [];
                for (var i = 0; i < savedTrends.length && i < 3; i++) {
                    if (savedTrends[i].name) trendNames.push(savedTrends[i].name);
                }
                for (var j = 0; j < savedTrends.length; j++) {
                    if (savedTrends[j].region) trendRegions.push(savedTrends[j].region);
                    if (savedTrends[j].audience_segment) trendSegments.push(savedTrends[j].audience_segment);
                    var topics = savedTrends[j].topics || [];
                    for (var k = 0; k < topics.length; k++) { trendTopics.push(topics[k]); }
                }
                var regions = getUniqueVals(trendRegions);
                var segments = getUniqueVals(trendSegments);
                var topics = getUniqueVals(trendTopics).slice(0, 5);

                context += 'Tracking ' + savedTrends.length + ' trends';
                if (trendNames.length) context += ' including "' + trendNames.join('", "') + '"';
                if (regions.length) context += '. Interested in regions: ' + regions.join(', ');
                if (segments.length) context += '. Target segments: ' + segments.join(', ');
                if (topics.length) context += '. Key themes: ' + topics.join(', ');
                context += '. ';
            }

            // Moves summary
            if (savedMoves.length > 0) {
                var moveCompanies = [];
                var moveMarkets = [];
                var moveMoveTypes = [];
                for (var mi = 0; mi < savedMoves.length; mi++) {
                    if (savedMoves[mi].company) moveCompanies.push(savedMoves[mi].company);
                    if (savedMoves[mi].market) moveMarkets.push(savedMoves[mi].market);
                    if (savedMoves[mi].move_type) moveMoveTypes.push(savedMoves[mi].move_type);
                }
                var companies = getUniqueVals(moveCompanies).slice(0, 3);
                var markets = getUniqueVals(moveMarkets).slice(0, 3);
                var moveTypes = getUniqueVals(moveMoveTypes);

                context += 'Watching ' + savedMoves.length + ' strategic moves';
                if (companies.length) context += ' by companies like ' + companies.join(', ');
                if (markets.length) context += ' in markets: ' + markets.join(', ');
                if (moveTypes.length) {
                    var mtFormatted = [];
                    for (var mti = 0; mti < moveTypes.length; mti++) { mtFormatted.push(moveTypes[mti].replace('_', ' ')); }
                    context += '. Move types: ' + mtFormatted.join(', ');
                }
                context += '.';
            }

            return context;
        }

        function sendSuggestion(text) {
            document.getElementById('chat-input').value = text;
            sendMessage();
        }

        function startBrandBuild() {
            var savedTrends = getSavedProjects();
            var savedMoves = getSavedMoves();

            // Helper function
            function getUniqueVals(arr) {
                var seen = {};
                var result = [];
                for (var i = 0; i < arr.length; i++) {
                    if (arr[i] && !seen[arr[i]]) {
                        seen[arr[i]] = true;
                        result.push(arr[i]);
                    }
                }
                return result;
            }

            var message;

            if (savedTrends.length > 0 || savedMoves.length > 0) {
                // Has profile - build from it
                var trendRegions = [];
                var trendSegments = [];
                var trendTopics = [];
                for (var i = 0; i < savedTrends.length; i++) {
                    if (savedTrends[i].region) trendRegions.push(savedTrends[i].region);
                    if (savedTrends[i].audience_segment) trendSegments.push(savedTrends[i].audience_segment);
                    var topics = savedTrends[i].topics || [];
                    for (var j = 0; j < topics.length; j++) { trendTopics.push(topics[j]); }
                }
                var regions = getUniqueVals(trendRegions);
                var segments = getUniqueVals(trendSegments);
                var topicsArr = getUniqueVals(trendTopics).slice(0, 3);

                message = 'Help me build a hotel brand based on my research profile.';
                if (regions.length) message += ' I am interested in ' + regions.slice(0, 2).join(' and ') + '.';
                if (segments.length) message += ' Target segment: ' + segments[0] + '.';
                if (topicsArr.length) message += ' Key themes I have been tracking: ' + topicsArr.join(', ') + '.';
            } else {
                // No profile - ask for guidance
                message = 'I want to build a hotel brand but I am not sure where to start. Can you help me figure out what kind of brand would be right?';
            }

            document.getElementById('chat-input').value = message;
            sendMessage();
        }

        async function sendMessage() {
            var input = document.getElementById('chat-input');
            var message = input.value.trim();
            if (!message) return;

            var messagesDiv = document.getElementById('chat-messages');

            // Clear welcome message if present
            var welcome = messagesDiv.querySelector('.chat-welcome');
            if (welcome) welcome.remove();

            // Add user message
            messagesDiv.innerHTML += '<div class="chat-message user"><div class="chat-bubble">' + escapeHtml(message) + '</div></div>';
            input.value = '';

            // Add typing indicator
            messagesDiv.innerHTML += '<div class="chat-message assistant" id="typing-indicator"><div class="chat-bubble chat-typing"><span></span><span></span><span></span></div></div>';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Build profile context from saved items
            var profileContext = buildProfileContext();

            try {
                var res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        conversation_id: chatConversationId,
                        user_context: profileContext
                    })
                });

                var data = await res.json();

                // Remove typing indicator
                var typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();

                if (res.ok) {
                    chatConversationId = data.conversation_id;

                    // Add assistant message
                    var confClass = data.confidence === 'High' ? 'confidence-high' :
                                      data.confidence === 'Medium' ? 'confidence-medium' : 'confidence-low';
                    messagesDiv.innerHTML += '<div class="chat-message assistant"><div class="chat-bubble">' + formatResponse(data.response) + '</div><div class="chat-confidence ' + confClass + '">' + data.confidence + ' confidence | ' + data.sources_used + ' sources | Mode: ' + data.mode + '</div></div>';

                    // Show state info
                    if (data.state) {
                        var slotLoc = (data.state.slots && data.state.slots.location) ? data.state.slots.location : '-';
                        var slotSeg = (data.state.slots && data.state.slots.segment) ? data.state.slots.segment : '-';
                        document.getElementById('chat-state').innerHTML = 'Mode: ' + data.mode + ' (' + Math.round(data.state.mode_confidence * 100) + '%) | Location: ' + slotLoc + ' | Segment: ' + slotSeg;
                    }

                    // Show suggested action
                    if (data.suggested_action) {
                        messagesDiv.innerHTML += '<div class="chat-message assistant"><button onclick="window.location.href=\\'/api/monitoring/dashboard-v2#build\\'" class="suggestion-chip" style="margin-top:10px;">➡️ Continue to Build a Brand</button></div>';
                    }
                } else {
                    messagesDiv.innerHTML += '<div class="chat-message assistant"><div class="chat-bubble" style="background:rgba(194,90,120,0.12);color:var(--bad);">Error: ' + (data.detail || 'Something went wrong') + '</div></div>';
                }

            } catch (err) {
                var typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();
                messagesDiv.innerHTML += '<div class="chat-message assistant"><div class="chat-bubble" style="background:rgba(194,90,120,0.12);color:var(--bad);">Connection error: ' + err.message + '</div></div>';
            }

            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function escapeHtml(text) {
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function formatResponse(text) {
            // Basic markdown-like formatting
            return escapeHtml(text)
                .replace(/\\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
        }

        // Load data on page load
        loadAllData();
        loadFilterOptions();
        loadMoveFilterOptions();
        updateSavedCount();
        updateMovesSavedCount();
        renderMyProjects();

        // Auto-refresh every 60 seconds
        setInterval(loadAllData, 60000);
    </script>

    <!-- Modal -->
    <div id="modal-overlay" class="modal-overlay" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <div id="modal-header" class="modal-header"><h2 id="modal-title">Title</h2><div class="meta" id="modal-meta"></div></div>
            <div class="modal-body" id="modal-body"></div>
        </div>
    </div>

</body>
</html>
"""
    return HTMLResponse(content=html)


@router.get("/monitoring/build-a-brand", response_class=HTMLResponse)
async def build_a_brand_page():
    """Build a Brand concept page - create hotel brand from trends."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Build a Brand | BrandClave</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg: #0e0c09;
            --surface: #17140f;
            --surface-2: #201a12;
            --surface-3: #2a2318;
            --ink: #f2ecdf;
            --ink-2: #b9ae9c;
            --ink-3: #857a68;
            --line: rgba(212,175,106,0.16);
            --line-strong: rgba(212,175,106,0.34);
            --gold: #d4af6a;
            --gold-deep: #b8862e;
            --gold-ink: #141008;
            --violet: #8b7ce0;
            --teal: #3aa88d;
            --rose: #c25a78;
            --blue: #4a8bc2;
            --grad: linear-gradient(90deg, #c25a78, #8b7ce0, #3aa88d, #d4af6a);
            --font-display: 'Archivo', 'Segoe UI', sans-serif;
            --font-body: 'Inter', -apple-system, 'Segoe UI', sans-serif;
            --font-mono: 'JetBrains Mono', 'Consolas', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-body);
            background: var(--bg);
            background-image: radial-gradient(ellipse 80% 40% at 50% -10%, rgba(212,175,106,0.07), transparent);
            min-height: 100vh;
            color: var(--ink);
        }
        ::selection { background: rgba(212,175,106,0.30); }

        .hero {
            padding: 56px 20px 36px;
            text-align: center;
            border-bottom: 1px solid var(--line);
        }
        .hero::before {
            content: 'BRANDCLAVE / CONCEPT STUDIO';
            display: block;
            font-family: var(--font-mono);
            font-size: 0.7em;
            letter-spacing: 0.35em;
            color: var(--gold);
            margin-bottom: 14px;
        }
        .hero h1 {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 2.4em;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--ink);
        }
        .hero p { color: var(--ink-2); margin-top: 10px; font-size: 0.95em; }
        .back-link {
            display: inline-block;
            margin-top: 16px;
            color: var(--gold);
            text-decoration: none;
            font-family: var(--font-mono);
            font-size: 0.8em;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            padding: 7px 16px;
            transition: all 0.2s;
        }
        .back-link:hover { background: var(--gold); color: var(--gold-ink); border-color: var(--gold); }
        .container { max-width: 900px; margin: 0 auto; padding: 28px 20px; }

        .card {
            background: var(--surface);
            border: 1px solid var(--line);
            padding: 28px;
            margin-bottom: 20px;
            border-radius: 12px;
        }
        .card h2 {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1.15em;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--ink);
            margin-bottom: 20px;
        }
        .card h2::before {
            content: '';
            display: block;
            width: 48px;
            height: 3px;
            background: var(--grad);
            border-radius: 2px;
            margin-bottom: 12px;
        }

        .source-trend {
            background: rgba(139,124,224,0.08);
            border: 1px solid rgba(139,124,224,0.30);
            border-left: 3px solid var(--violet);
            color: var(--ink);
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .source-trend h3 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }
        .source-trend p { color: var(--ink-2); font-size: 0.9em; }

        .profile-source-card {
            background: rgba(58,168,141,0.08);
            border: 1px solid rgba(58,168,141,0.30);
            border-left: 3px solid var(--teal);
            color: var(--ink);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .profile-source-card h3 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }
        .profile-row { display: flex; gap: 8px; margin-bottom: 4px; color: var(--ink-2); }
        .profile-label { color: var(--ink-3); }
        .profile-theme-tag {
            display: inline-block;
            background: rgba(58,168,141,0.16);
            color: var(--teal);
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.82em;
            margin: 2px;
        }

        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-family: var(--font-mono);
            font-size: 0.72em;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--ink-3);
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            background: var(--surface-2);
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            color: var(--ink);
            font-family: var(--font-body);
            font-size: 0.95em;
        }
        .form-group textarea { min-height: 100px; resize: vertical; }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none;
            border-color: var(--gold);
        }
        .form-group input::placeholder, .form-group textarea::placeholder { color: var(--ink-3); }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        @media (max-width: 600px) {
            .form-row { grid-template-columns: 1fr; }
        }

        .topics-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .topic-tag {
            background: rgba(139,124,224,0.14);
            color: var(--violet);
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.82em;
        }

        .btn-generate {
            width: 100%;
            padding: 16px 30px;
            background: var(--gold);
            color: var(--gold-ink);
            border: none;
            border-radius: 8px;
            font-family: var(--font-display);
            font-size: 0.95em;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
            transition: transform 0.2s, background 0.2s, box-shadow 0.2s;
        }
        .btn-generate:hover {
            background: #e2c184;
            transform: translateY(-2px);
            box-shadow: 0 6px 22px rgba(212,175,106,0.25);
        }
        .btn-generate:disabled {
            background: var(--surface-3);
            color: var(--ink-3);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        #result-container { display: none; }

        .blueprint-card {
            background: var(--surface);
            border: 1px solid var(--line-strong);
            border-top: 3px solid transparent;
            border-image: linear-gradient(90deg, #c25a78, #8b7ce0, #3aa88d, #d4af6a) 1;
            padding: 28px;
            border-radius: 0 0 12px 12px;
            color: var(--ink);
        }
        .blueprint-card h2 {
            color: var(--ink);
            margin-bottom: 5px;
            font-family: var(--font-display);
            font-weight: 800;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .blueprint-oneliner { font-size: 1.05em; color: var(--gold); margin-bottom: 24px; }

        .blueprint-section { margin-bottom: 24px; }
        .blueprint-section h3 {
            font-family: var(--font-mono);
            font-size: 0.75em;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 500;
            color: var(--gold);
            margin-bottom: 10px;
        }
        .blueprint-section p { line-height: 1.65; color: var(--ink-2); }
        .blueprint-section ul { padding-left: 20px; color: var(--ink-2); }
        .blueprint-section li { margin-bottom: 6px; }

        .experience-card {
            background: var(--surface-2);
            border: 1px solid var(--line);
            padding: 14px 16px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .experience-card h4 { margin-bottom: 5px; color: var(--ink); }
        .experience-card p { font-size: 0.9em; color: var(--ink-2); }

        .loading-indicator { text-align: center; padding: 40px; }
        .loading-indicator .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid var(--line);
            border-top-color: var(--gold);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .stage-item { color: var(--ink-2); padding: 3px 0; font-family: var(--font-mono); font-size: 0.85em; }
        .stage-icon { color: var(--gold); }

        .btn-actions {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .btn-secondary {
            flex: 1;
            min-width: 150px;
            padding: 12px;
            background: transparent;
            color: var(--ink-2);
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.92em;
            transition: all 0.2s;
        }
        .btn-secondary:hover { color: var(--ink); border-color: var(--gold); }

        .white-space-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: var(--surface-2);
            border: 1px solid var(--line);
            color: var(--ink-2);
            padding: 3px 9px;
            border-radius: 999px;
            font-family: var(--font-mono);
            font-size: 0.78em;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="hero">
        <h1>Build a Brand</h1>
        <p>Transform market trends into unique hotel brand concepts</p>
        <a href="/api/monitoring/dashboard-v2" class="back-link">Back to Dashboard</a>
    </div>

    <div class="container">
        <div id="source-trend-card" class="source-trend" style="display:none;">
            <h3 id="source-trend-name">Source Trend</h3>
            <p id="source-trend-desc">Description</p>
            <div class="topics-list" id="source-topics"></div>
            <div class="white-space-badge" id="source-ws"></div>
        </div>

        <div id="profile-card" class="profile-source-card" style="display:none;">
            <h3>Building from Your Profile</h3>
            <p style="margin-bottom:12px;opacity:0.9;">Your saved trends and moves are informing this brand.</p>
            <div id="profile-summary">
                <div class="profile-row">
                    <span class="profile-label">Trends:</span>
                    <span id="profile-trend-count">0</span>
                </div>
                <div class="profile-row">
                    <span class="profile-label">Moves:</span>
                    <span id="profile-move-count">0</span>
                </div>
                <div id="profile-themes" style="margin-top:10px;"></div>
            </div>
        </div>

        <div class="card">
            <h2>Brand Inputs</h2>

            <div class="form-row">
                <div class="form-group">
                    <label for="brand-location">Target Location</label>
                    <input type="text" id="brand-location" placeholder="e.g., Lisbon, Portugal">
                </div>
                <div class="form-group">
                    <label for="brand-segment">Target Segment</label>
                    <select id="brand-segment">
                        <option value="lifestyle">Lifestyle</option>
                        <option value="luxury">Luxury</option>
                        <option value="boutique">Boutique</option>
                        <option value="wellness">Wellness</option>
                        <option value="eco">Eco / Sustainable</option>
                        <option value="business">Business</option>
                        <option value="family">Family</option>
                        <option value="adventure">Adventure</option>
                    </select>
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label for="brand-adr">Target ADR ($)</label>
                    <input type="number" id="brand-adr" placeholder="e.g., 350">
                </div>
                <div class="form-group">
                    <label for="brand-rooms">Room Count</label>
                    <input type="number" id="brand-rooms" placeholder="e.g., 80">
                </div>
            </div>

            <div class="form-group">
                <label for="brand-goal">Developer Goal / Vision</label>
                <textarea id="brand-goal" placeholder="What makes this project special? What's your vision?"></textarea>
            </div>

            <button class="btn-generate" id="generate-btn" onclick="generateBrandConcept()">
                Generate Brand Concept
            </button>
        </div>

        <div class="card" id="saved-blueprints-section">
            <h2 style="margin-bottom:15px;">Saved Blueprints</h2>
            <div id="saved-blueprints-list" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:15px;">
                <p style="color:var(--ink-3);grid-column:1/-1;">Loading saved blueprints...</p>
            </div>
        </div>

        <div id="loading-container" class="card" style="display:none;">
            <div class="loading-indicator">
                <div class="spinner"></div>
                <p id="loading-stage">Generating your brand concept...</p>
                <div id="stage-progress" style="margin-top:15px;text-align:left;max-width:300px;margin-left:auto;margin-right:auto;">
                    <div class="stage-item" data-stage="foundation"><span class="stage-icon">&#9679;</span> Foundation (names, thesis)</div>
                    <div class="stage-item" data-stage="strategic"><span class="stage-icon">&#9675;</span> Strategic (pillars, positioning)</div>
                    <div class="stage-item" data-stage="experience"><span class="stage-icon">&#9675;</span> Experience (personas, journey)</div>
                    <div class="stage-item" data-stage="atmosphere"><span class="stage-icon">&#9675;</span> Atmosphere (design, F&B)</div>
                    <div class="stage-item" data-stage="summary"><span class="stage-icon">&#9675;</span> Summary (investor pitch)</div>
                </div>
            </div>
        </div>

        <div id="result-container">
            <div class="blueprint-card">
                <div id="bp-name-options" style="margin-bottom:15px;">
                    <h2 id="bp-name">Brand Name</h2>
                    <div id="bp-alternates" style="font-size:0.9em;color:var(--ink-2);margin-top:5px;"></div>
                </div>
                <p class="blueprint-oneliner" id="bp-oneliner">One-liner</p>

                <div id="bp-inputs-section" class="blueprint-section" style="background:var(--surface-2);border:1px solid var(--line-strong);border-radius:8px;padding:15px;display:none;">
                    <h3 style="font-size:0.95em;color:var(--blue);margin-bottom:10px;">Blueprint Parameters</h3>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;font-size:0.9em;">
                        <div><strong style="color:var(--ink-2);">Location:</strong> <span id="bp-input-location">-</span></div>
                        <div><strong style="color:var(--ink-2);">Segment:</strong> <span id="bp-input-segment">-</span></div>
                        <div><strong style="color:var(--ink-2);">Target ADR:</strong> $<span id="bp-input-adr">-</span></div>
                        <div><strong style="color:var(--ink-2);">Rooms:</strong> <span id="bp-input-rooms">-</span></div>
                    </div>
                    <div id="bp-input-goal-container" style="margin-top:10px;display:none;">
                        <strong style="color:var(--ink-2);">Developer Goal:</strong>
                        <p id="bp-input-goal" style="margin:5px 0 0 0;font-style:italic;color:var(--ink-2);"></p>
                    </div>
                </div>

                <div class="blueprint-section">
                    <h3>Brand Thesis</h3>
                    <p id="bp-thesis"></p>
                </div>

                <div class="blueprint-section">
                    <h3>Brand Pillars</h3>
                    <ul id="bp-pillars"></ul>
                </div>

                <div class="blueprint-section">
                    <h3>Positioning Statement</h3>
                    <p id="bp-positioning"></p>
                </div>

                <div class="blueprint-section">
                    <h3>Unmet Desires Solved</h3>
                    <p style="font-size:0.9em;color:var(--ink-2);margin-bottom:12px;">Guest needs identified from market trends that this brand addresses</p>
                    <div id="bp-desires"></div>
                </div>

                <div class="blueprint-section">
                    <h3>Target Guest Personas</h3>
                    <div id="bp-personas"></div>
                </div>

                <div class="blueprint-section">
                    <h3>Signature Experiences</h3>
                    <div id="bp-experiences"></div>
                </div>

                <div class="blueprint-section">
                    <h3>Guest Journey</h3>
                    <div id="bp-journey"></div>
                </div>

                <div class="blueprint-section">
                    <h3>Design Direction</h3>
                    <p id="bp-design"></p>
                </div>

                <div class="blueprint-section">
                    <h3>F&B Concepts</h3>
                    <div id="bp-fnb"></div>
                </div>

                <div class="blueprint-section">
                    <h3>Revenue Logic</h3>
                    <p id="bp-revenue"></p>
                </div>

                <div class="blueprint-section">
                    <h3>Investor Summary</h3>
                    <p id="bp-investor" style="background:var(--surface-2);padding:15px;border-radius:8px;"></p>
                </div>

                <div id="bp-metadata" style="margin-top:20px;font-size:0.85em;color:var(--ink-3);">
                    <span id="bp-confidence"></span>
                    <span id="bp-tokens" style="margin-left:15px;"></span>
                </div>
            </div>

            <div class="btn-actions">
                <button class="btn-secondary" onclick="saveBlueprintToProject()">
                    Save to Project
                </button>
                <button class="btn-secondary" onclick="regenerateConcept()">
                    Regenerate
                </button>
                <button class="btn-secondary" onclick="window.print()">
                    Print / Export
                </button>
            </div>
        </div>
    </div>

    <script>
        var sourceTrend = null;
        var profileData = null;
        var currentBlueprint = null;

        // Load source trend or profile from sessionStorage
        function loadSourceTrend() {
            try {
                var data = sessionStorage.getItem('brandclave_brand_input');
                var profile = sessionStorage.getItem('brandclave_profile_data');

                if (data) {
                    sourceTrend = JSON.parse(data);

                    // Check if coming from profile
                    if (sourceTrend.from_profile && profile) {
                        profileData = JSON.parse(profile);
                        displayProfileCard(profileData);
                    } else if (sourceTrend.source_trend_name) {
                        // Show single trend card
                        document.getElementById('source-trend-card').style.display = 'block';
                        document.getElementById('source-trend-name').textContent = sourceTrend.source_trend_name || 'Selected Trend';
                        document.getElementById('source-trend-desc').textContent = sourceTrend.description || '';

                        // Show topics
                        var topicsEl = document.getElementById('source-topics');
                        if (sourceTrend.topics && sourceTrend.topics.length) {
                            var topicsHtml = '';
                            for (var ti = 0; ti < sourceTrend.topics.length; ti++) {
                                topicsHtml += '<span class="topic-tag">' + sourceTrend.topics[ti] + '</span>';
                            }
                            topicsEl.innerHTML = topicsHtml;
                        }

                        // Show white space score
                        if (sourceTrend.white_space_score) {
                            var ws = Math.round(sourceTrend.white_space_score * 100);
                            document.getElementById('source-ws').textContent = 'White Space: ' + ws + '%';
                        }
                    }

                    // Pre-fill inputs
                    if (sourceTrend.initial_region) {
                        document.getElementById('brand-location').value = sourceTrend.initial_region;
                    }
                    if (sourceTrend.initial_segment) {
                        document.getElementById('brand-segment').value = sourceTrend.initial_segment;
                    }
                }
            } catch (e) {
                console.error('Error loading source trend:', e);
            }
        }

        function displayProfileCard(profile) {
            document.getElementById('profile-card').style.display = 'block';
            document.getElementById('profile-trend-count').textContent = profile.trends ? profile.trends.length : 0;
            document.getElementById('profile-move-count').textContent = profile.moves ? profile.moves.length : 0;

            // Show key themes
            var themesEl = document.getElementById('profile-themes');
            var allThemes = [];
            var topics = (profile.topics || []).slice(0, 3);
            var segments = (profile.segments || []).slice(0, 2);
            var markets = (profile.markets || []).slice(0, 2);
            for (var i = 0; i < topics.length; i++) allThemes.push(topics[i]);
            for (var j = 0; j < segments.length; j++) allThemes.push(segments[j]);
            for (var k = 0; k < markets.length; k++) allThemes.push(markets[k]);

            if (allThemes.length > 0) {
                var themesHtml = '<div style="margin-top:8px;">';
                for (var ti = 0; ti < allThemes.length; ti++) {
                    themesHtml += '<span class="profile-theme-tag">' + allThemes[ti] + '</span>';
                }
                themesHtml += '</div>';
                themesEl.innerHTML = themesHtml;
            }
        }

        async function generateBrandConcept() {
            var btn = document.getElementById('generate-btn');
            var loadingEl = document.getElementById('loading-container');
            var resultEl = document.getElementById('result-container');

            // Validate inputs
            var location = document.getElementById('brand-location').value;
            var segment = document.getElementById('brand-segment').value;
            var adr = document.getElementById('brand-adr').value;
            var rooms = document.getElementById('brand-rooms').value || 100;
            var goal = document.getElementById('brand-goal').value;

            if (!location || !adr || !goal) {
                alert('Please fill in Location, Target ADR, and Developer Goal.');
                return;
            }

            btn.disabled = true;
            loadingEl.style.display = 'block';
            resultEl.style.display = 'none';

            // Reset stage indicators
            var stageItems = document.querySelectorAll('.stage-item');
            for (var si = 0; si < stageItems.length; si++) {
                stageItems[si].querySelector('.stage-icon').innerHTML = '&#9675;';
                stageItems[si].style.color = 'var(--ink-2)';
            }

            try {
                // Call the new blueprint generation API
                var res = await fetch('/api/brand-blueprint/generate-simple', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        location: location,
                        segment: segment,
                        adr: parseFloat(adr),
                        rooms: parseInt(rooms),
                        developer_goal: goal,
                        source_trend_id: sourceTrend ? sourceTrend.source_trend_id : null
                    })
                });

                var data = await res.json();

                if (res.ok && data.blueprint) {
                    // Update all stage indicators to complete
                    var completeItems = document.querySelectorAll('.stage-item');
                    for (var ci = 0; ci < completeItems.length; ci++) {
                        completeItems[ci].querySelector('.stage-icon').innerHTML = '&#10003;';
                        completeItems[ci].style.color = 'var(--good)';
                    }

                    displayBlueprint(data.blueprint);
                    resultEl.style.display = 'block';
                } else {
                    alert('Generation failed: ' + (data.detail || 'Unknown error'));
                }

            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                loadingEl.style.display = 'none';
            }
        }

        function buildBrandPrompt(inputs) {
            var prompt = 'Help me create a detailed hotel brand concept';

            if (inputs.location) prompt += ' in ' + inputs.location;
            if (inputs.segment) prompt += ' targeting the ' + inputs.segment + ' segment';
            if (inputs.adr) prompt += ' with a target ADR of $' + inputs.adr;
            if (inputs.rooms) prompt += ' and approximately ' + inputs.rooms + ' rooms';

            // Use profile data if available
            if (inputs.profile) {
                prompt += '. IMPORTANT CONTEXT - This brand should be informed by my research:';

                // Add trend insights
                if (inputs.profile.trends && inputs.profile.trends.length > 0) {
                    prompt += ' I have been tracking ' + inputs.profile.trends.length + ' trends including: ';
                    var trendNames = [];
                    var trendsSlice = inputs.profile.trends.slice(0, 3);
                    for (var ti = 0; ti < trendsSlice.length; ti++) {
                        var tName = trendsSlice[ti].name || trendsSlice[ti].description;
                        if (tName) trendNames.push(tName);
                    }
                    if (trendNames.length) prompt += trendNames.join(', ');
                }

                // Add topic themes
                if (inputs.profile.topics && inputs.profile.topics.length > 0) {
                    prompt += '. Key themes I am interested in: ' + inputs.profile.topics.slice(0, 5).join(', ');
                }

                // Add move insights
                if (inputs.profile.moves && inputs.profile.moves.length > 0) {
                    prompt += '. I have been watching ' + inputs.profile.moves.length + ' strategic moves by companies like: ';
                    var companiesSet = {};
                    var companies = [];
                    for (var mi = 0; mi < inputs.profile.moves.length; mi++) {
                        var comp = inputs.profile.moves[mi].company;
                        if (comp && !companiesSet[comp]) {
                            companiesSet[comp] = true;
                            companies.push(comp);
                            if (companies.length >= 3) break;
                        }
                    }
                    if (companies.length) prompt += companies.join(', ');
                }

                // Add markets of interest
                if (inputs.profile.markets && inputs.profile.markets.length > 0) {
                    prompt += '. Markets I am researching: ' + inputs.profile.markets.slice(0, 3).join(', ');
                }
            }
            // Use single trend if available (old flow)
            else if (inputs.source_trend && inputs.source_trend.source_trend_name) {
                prompt += '. This brand should capitalize on the trend: "' +
                    inputs.source_trend.source_trend_name + '"';
                if (inputs.source_trend.description) {
                    prompt += ' - ' + inputs.source_trend.description.substring(0, 200);
                }
            }

            if (inputs.goal) prompt += '. My vision: ' + inputs.goal;

            prompt += '. Please provide: 1) A unique brand name 2) One-liner essence 3) Brand thesis 4) 3-5 brand pillars 5) Signature experiences 6) Design direction 7) Target guest personas 8) Why this concept will succeed in the market.';

            return prompt;
        }

        function displayBlueprint(blueprint) {
            currentBlueprint = blueprint;

            // Brand names with alternates
            var names = blueprint.brand_names || {};
            document.getElementById('bp-name').textContent = names.primary || 'Brand Concept';
            if (names.alternate_1 || names.alternate_2) {
                var alts = [];
                if (names.alternate_1) alts.push(names.alternate_1);
                if (names.alternate_2) alts.push(names.alternate_2);
                document.getElementById('bp-alternates').textContent = 'Alternates: ' + alts.join(', ');
            }

            document.getElementById('bp-oneliner').textContent = blueprint.one_liner || '';

            // Input parameters display
            var inputs = blueprint.inputs || {};
            var inputsSection = document.getElementById('bp-inputs-section');
            if (inputs.location || inputs.segment || inputs.adr) {
                document.getElementById('bp-input-location').textContent = inputs.location || '-';
                document.getElementById('bp-input-segment').textContent = inputs.segment || '-';
                document.getElementById('bp-input-adr').textContent = inputs.adr || '-';
                document.getElementById('bp-input-rooms').textContent = inputs.rooms || '-';
                if (inputs.developer_goal) {
                    document.getElementById('bp-input-goal').textContent = inputs.developer_goal;
                    document.getElementById('bp-input-goal-container').style.display = 'block';
                } else {
                    document.getElementById('bp-input-goal-container').style.display = 'none';
                }
                inputsSection.style.display = 'block';
            } else {
                inputsSection.style.display = 'none';
            }

            document.getElementById('bp-thesis').textContent = blueprint.thesis || '';

            // Pillars
            var pillars = blueprint.pillars || [];
            var pillarsHtml = '';
            for (var pi = 0; pi < pillars.length; pi++) {
                pillarsHtml += '<li>' + pillars[pi] + '</li>';
            }
            document.getElementById('bp-pillars').innerHTML = pillarsHtml;

            // Positioning
            document.getElementById('bp-positioning').textContent = blueprint.positioning_statement || '';

            // Unmet desires solved
            var desires = blueprint.unmet_desires_solved || [];
            var desiresHtml = '';
            if (desires.length > 0) {
                for (var di = 0; di < desires.length; di++) {
                    var d = desires[di];
                    var strength = Math.round((d.demand_strength || 0.5) * 100);
                    var strengthColor = strength >= 70 ? 'var(--good)' : (strength >= 40 ? 'var(--warn)' : 'var(--bad)');
                    desiresHtml += '<div class="experience-card">' +
                        '<div style="display:flex;justify-content:space-between;align-items:center;">' +
                        '<strong>' + (d.desire || '') + '</strong>' +
                        '<span style="background:' + strengthColor + ';color:var(--ink);padding:2px 8px;border-radius:4px;font-size:0.8em;">' + strength + '% demand</span>' +
                        '</div>' +
                        '<p style="margin-top:8px;">' + (d.how_solved || '') + '</p>' +
                        '</div>';
                }
            } else {
                desiresHtml = '<p style="color:var(--ink-3);font-style:italic;">No specific unmet desires identified</p>';
            }
            document.getElementById('bp-desires').innerHTML = desiresHtml;

            // Guest personas
            var personas = blueprint.guest_personas || [];
            var personasHtml = '';
            for (var psi = 0; psi < personas.length; psi++) {
                var p = personas[psi];
                personasHtml += '<div class="experience-card">' +
                    '<strong>' + (p.name || '') + '</strong>' +
                    '<p>' + (p.description || '') + '</p>' +
                    '<p style="font-size:0.9em;color:var(--ink-2);">Spend: ' + (p.spend_behavior || '') + '</p>' +
                    '</div>';
            }
            document.getElementById('bp-personas').innerHTML = personasHtml;

            // Signature experiences
            var experiences = blueprint.signature_experiences || [];
            var expHtml = '';
            for (var ei = 0; ei < experiences.length; ei++) {
                var e = experiences[ei];
                expHtml += '<div class="experience-card">' +
                    '<strong>' + (e.name || '') + '</strong>' +
                    '<p>' + (e.description || '') + '</p>' +
                    '<p style="font-size:0.9em;color:var(--good);">' + (e.why_it_matters || '') + '</p>' +
                    '</div>';
            }
            document.getElementById('bp-experiences').innerHTML = expHtml;

            // Guest journey
            var journey = blueprint.guest_journey || {};
            if (journey.arrival || journey.stay || journey.departure) {
                document.getElementById('bp-journey').innerHTML =
                    '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:15px;">' +
                    '<div class="experience-card"><strong>Arrival</strong><p>' + (journey.arrival || '') + '</p></div>' +
                    '<div class="experience-card"><strong>Stay</strong><p>' + (journey.stay || '') + '</p></div>' +
                    '<div class="experience-card"><strong>Departure</strong><p>' + (journey.departure || '') + '</p></div>' +
                    '</div>';
            }

            // Design direction
            document.getElementById('bp-design').textContent = blueprint.design_direction || '';

            // F&B concepts
            var fnb = blueprint.fnb_concepts || [];
            var fnbHtml = '';
            for (var fi = 0; fi < fnb.length; fi++) {
                var f = fnb[fi];
                fnbHtml += '<div class="experience-card">' +
                    '<strong>' + (f.name || '') + '</strong>' +
                    '<p>' + (f.concept || '') + '</p>' +
                    '<p style="font-size:0.9em;color:var(--ink-2);">Vibe: ' + (f.vibe || '') + '</p>' +
                    '</div>';
            }
            document.getElementById('bp-fnb').innerHTML = fnbHtml;

            // Revenue logic
            document.getElementById('bp-revenue').textContent = blueprint.revenue_logic || '';

            // Investor summary
            document.getElementById('bp-investor').textContent = blueprint.investor_summary || '';

            // Metadata
            var confidence = Math.round((blueprint.confidence || 0) * 100);
            document.getElementById('bp-confidence').textContent = 'Confidence: ' + confidence + '%';

            var tokens = blueprint.token_usage || {};
            if (tokens.total_tokens) {
                document.getElementById('bp-tokens').textContent =
                    'Tokens: ' + tokens.total_tokens + ' (~$' + (tokens.estimated_cost_usd || 0).toFixed(3) + ')';
            }
        }

        function parseResponse(text) {
            var result = {
                name: '',
                oneliner: '',
                thesis: '',
                pillars: [],
                experiences: [],
                design: '',
                personas: '',
                success: ''
            };

            // Helper to strip markdown formatting
            function cleanMarkdown(str) {
                if (!str) return '';
                return str
                    .replace(/\\*\\*\\*([^*]+)\\*\\*\\*/g, '$1')  // ***bold italic***
                    .replace(/\\*\\*([^*]+)\\*\\*/g, '$1')        // **bold**
                    .replace(/\\*([^*]+)\\*/g, '$1')              // *italic*
                    .replace(/^#{1,6}\\s*/gm, '')                 // ### headers
                    .replace(/^[-*]\\s+/gm, '')                   // - bullet points
                    .replace(/^\\d+\\.\\s+/gm, '')                // 1. numbered lists
                    .replace(/\\n{3,}/g, '\\n\\n')                // multiple newlines
                    .trim();
            }

            // Extract brand name - look for quoted name or "The X" pattern
            var namePatterns = [
                /brand\\s*name[:\\s]*[""]([^""]+)[""]|brand\\s*name[:\\s]*["']([^"']+)["']/i,
                /[""]The\\s+([^""]+)[""]|["']The\\s+([^"']+)["']/i,
                /called\\s+[""]([^""]+)[""]|called\\s+["']([^"']+)["']/i,
                /\\*\\*[""]?([^"*]+)[""]?\\*\\*/
            ];
            for (var npi = 0; npi < namePatterns.length; npi++) {
                var pattern = namePatterns[npi];
                var match = text.match(pattern);
                if (match) {
                    result.name = cleanMarkdown(match[1] || match[2] || '');
                    if (result.name) break;
                }
            }

            // Extract one-liner/essence
            var essenceMatch = text.match(/one[- ]liner[^:]*:[\\s]*[""]?([^""\\n]+)/i) ||
                               text.match(/essence[^:]*:[\\s]*[""]?([^""\\n]+)/i);
            if (essenceMatch) result.oneliner = cleanMarkdown(essenceMatch[1]);

            // Split into sections by markdown headers
            var sections = text.split(/(?=#{2,3}\\s|\\*\\*\\d+\\.|\\*\\*[A-Z])/);

            for (var sci = 0; sci < sections.length; sci++) {
                var section = sections[sci];
                var lower = section.toLowerCase();
                var content = cleanMarkdown(section.replace(/^[#*\\d.\\s]+[^\\n]*\\n?/, ''));

                // Only match section headers, not content
                var isHeader = section.match(/^#{2,3}\\s|^\\*\\*\\d+\\.|^\\*\\*[A-Z]/);
                if (!isHeader) continue;

                if (lower.includes('thesis') || lower.includes('philosophy') || lower.includes('core concept')) {
                    result.thesis = content;
                } else if (lower.includes('pillar')) {
                    // Extract bullet points
                    var bullets = section.match(/[-*]\\s+\\*?\\*?([^\\n*]+)/g) || [];
                    result.pillars = [];
                    for (var bi = 0; bi < bullets.length; bi++) {
                        var cleaned = cleanMarkdown(bullets[bi]);
                        if (cleaned.length > 3) result.pillars.push(cleaned);
                    }
                } else if ((lower.includes('experience') || lower.includes('signature')) && !lower.includes('target')) {
                    var expBullets = section.match(/[-*]\\s+\\*?\\*?([^\\n*]+)/g) ||
                                     section.match(/\\*\\*([^*]+)\\*\\*[^\\n]*/g) || [];
                    result.experiences = [];
                    for (var ebi = 0; ebi < expBullets.length; ebi++) {
                        var expCleaned = cleanMarkdown(expBullets[ebi]);
                        if (expCleaned.length > 3) result.experiences.push(expCleaned);
                    }
                } else if (lower.includes('design') || lower.includes('aesthetic') || lower.includes('visual')) {
                    result.design = content;
                } else if ((lower.includes('guest') || lower.includes('persona')) && lower.includes('target')) {
                    result.personas = content;
                } else if (lower.includes('succeed') || lower.includes('success') || lower.includes('why it will')) {
                    result.success = content;
                }
            }

            // Try alternate extraction for experiences if empty
            if (result.experiences.length === 0) {
                var expMatch = text.match(/signature\\s+experience[^:]*:([\\s\\S]*?)(?=###|\\*\\*\\d|$)/i);
                if (expMatch) {
                    var altBullets = expMatch[1].match(/[-*]\\s+\\*?\\*?([^\\n]+)/g) || [];
                    result.experiences = [];
                    for (var abi = 0; abi < altBullets.length; abi++) {
                        var altCleaned = cleanMarkdown(altBullets[abi]);
                        if (altCleaned.length > 5) result.experiences.push(altCleaned);
                    }
                }
            }

            // Try alternate extraction for personas if empty
            if (!result.personas) {
                var personaMatch = text.match(/target\\s+guest[^:]*:([\\s\\S]*?)(?=###|\\*\\*\\d|$)/i);
                if (personaMatch) {
                    result.personas = cleanMarkdown(personaMatch[1]);
                }
            }

            // Fallbacks with cleaner defaults
            if (result.pillars.length === 0) {
                result.pillars = ['Authentic Local Experience', 'Community Connection', 'Distinctive Design', 'Personalized Service'];
            }
            if (result.experiences.length === 0) {
                result.experiences = ['Curated neighborhood discoveries', 'Signature welcome ritual', 'Local artisan collaborations'];
            }
            if (!result.thesis) {
                // Extract first meaningful paragraph
                var firstPara = text.split('\\n\\n')[0];
                result.thesis = cleanMarkdown(firstPara).substring(0, 400);
            }
            if (!result.success) {
                result.success = 'This concept fills a clear market gap by combining trending traveler preferences with authentic local experiences.';
            }

            return result;
        }

        function saveBlueprintToProject() {
            if (!currentBlueprint) return;

            var saved = JSON.parse(localStorage.getItem('brandclave_saved_blueprints') || '[]');
            saved.push(currentBlueprint);
            localStorage.setItem('brandclave_saved_blueprints', JSON.stringify(saved));

            alert('Blueprint saved!');
        }

        function regenerateConcept() {
            generateBrandConcept();
        }

        async function loadSavedBlueprints() {
            var listEl = document.getElementById('saved-blueprints-list');
            if (!listEl) return;

            try {
                var response = await fetch('/api/brand-blueprint?limit=10');
                var data = await response.json();

                if (!data.blueprints || data.blueprints.length === 0) {
                    listEl.innerHTML = '<p style="color:var(--ink-3);grid-column:1/-1;font-style:italic;">No saved blueprints yet. Generate your first brand concept above!</p>';
                    return;
                }

                var html = '';
                for (var i = 0; i < data.blueprints.length; i++) {
                    var bp = data.blueprints[i];
                    var names = bp.brand_names || {};
                    var created = new Date(bp.generated_at).toLocaleDateString();
                    html += '<div class="saved-blueprint-card" onclick="loadSavedBlueprint(\\'' + bp.id + '\\')" style="' +
                        'background:var(--surface-2);border:1px solid var(--surface-3);border-radius:8px;padding:15px;cursor:pointer;transition:all 0.2s;' +
                        '">' +
                        '<h4 style="margin:0 0 8px 0;color:var(--ink);">' + (names.primary || 'Unnamed') + '</h4>' +
                        '<p style="font-size:0.85em;color:var(--ink-2);margin:0 0 8px 0;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">' + (bp.one_liner || '') + '</p>' +
                        '<div style="display:flex;justify-content:space-between;align-items:center;font-size:0.8em;color:var(--ink-3);">' +
                        '<span>' + (bp.inputs?.location || '') + '</span>' +
                        '<span>' + created + '</span>' +
                        '</div>' +
                        '</div>';
                }
                listEl.innerHTML = html;

                // Add hover effects
                var cards = listEl.querySelectorAll('.saved-blueprint-card');
                cards.forEach(function(card) {
                    card.addEventListener('mouseenter', function() {
                        this.style.borderColor = 'var(--good)';
                        this.style.transform = 'translateY(-2px)';
                        this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
                    });
                    card.addEventListener('mouseleave', function() {
                        this.style.borderColor = 'var(--surface-3)';
                        this.style.transform = 'translateY(0)';
                        this.style.boxShadow = 'none';
                    });
                });

            } catch (e) {
                console.error('Error loading saved blueprints:', e);
                listEl.innerHTML = '<p style="color:var(--bad);grid-column:1/-1;">Failed to load saved blueprints</p>';
            }
        }

        async function loadSavedBlueprint(id) {
            try {
                var response = await fetch('/api/brand-blueprint/' + id);
                if (!response.ok) throw new Error('Blueprint not found');
                var blueprint = await response.json();

                // Display the blueprint
                displayBlueprint(blueprint);

                // Show result container, hide others
                document.getElementById('result-container').style.display = 'block';
                document.getElementById('loading-container').style.display = 'none';

                // Scroll to result
                document.getElementById('result-container').scrollIntoView({ behavior: 'smooth' });

            } catch (e) {
                console.error('Error loading blueprint:', e);
                alert('Failed to load blueprint: ' + e.message);
            }
        }

        // Initialize
        loadSourceTrend();
        loadSavedBlueprints();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
