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
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800;900&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg: #0d0b08;
            --surface: #17140f;
            --surface-2: #201a12;
            --surface-3: #2a2318;
            --ink: #f2ecdf;
            --ink-2: #b9ae9c;
            --ink-3: #857a68;
            --line: rgba(212,175,106,0.15);
            --line-strong: rgba(212,175,106,0.34);
            --gold: #d4af6a;
            --gold-deep: #b8862e;
            --gold-ink: #141008;
            --gold-grad: linear-gradient(135deg, #ecd7a8 0%, #d4af6a 48%, #b18a41 100%);
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
            min-height: 100vh;
            color: var(--ink);
            background:
                radial-gradient(ellipse 75% 50% at 50% -14%, rgba(216,180,120,0.13), transparent 62%),
                radial-gradient(ellipse 42% 34% at 88% 4%, rgba(139,124,224,0.06), transparent 70%),
                radial-gradient(ellipse 42% 34% at 8% 10%, rgba(58,168,141,0.05), transparent 70%),
                radial-gradient(1.2px 1.2px at 21% 26%, rgba(242,236,223,0.32), transparent 100%),
                radial-gradient(1px 1px at 67% 14%, rgba(242,236,223,0.24), transparent 100%),
                radial-gradient(1.4px 1.4px at 84% 41%, rgba(226,193,132,0.20), transparent 100%),
                radial-gradient(1px 1px at 39% 57%, rgba(242,236,223,0.16), transparent 100%),
                radial-gradient(1.2px 1.2px at 9% 74%, rgba(226,193,132,0.14), transparent 100%),
                radial-gradient(1px 1px at 55% 88%, rgba(242,236,223,0.12), transparent 100%),
                var(--bg);
            /* background-attachment: fixed removed: nine stacked radial gradients repainted on every scroll tick and stalled the renderer */
        }
        ::selection { background: rgba(212,175,106,0.30); }
        ::-webkit-scrollbar { width: 11px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 6px; border: 3px solid var(--bg); }
        ::-webkit-scrollbar-thumb:hover { background: #3a3120; }

        .hero {
            position: relative;
            padding: 34px 20px 26px;
            text-align: center;
            border-bottom: 1px solid var(--line);
            background:
                radial-gradient(closest-side circle at 50% 128%, rgba(236,215,168,0.20), rgba(212,175,106,0.07) 52%, transparent 78%);
            overflow: hidden;
        }
        .hero::before {
            content: 'HOSPITALITY DEMAND INTELLIGENCE';
            display: block;
            font-family: var(--font-mono);
            font-size: 0.66em;
            letter-spacing: 0.42em;
            text-indent: 0.42em;
            color: var(--gold);
            margin-bottom: 10px;
            text-shadow: 0 0 24px rgba(212,175,106,0.45);
        }
        .hero h1 {
            font-family: var(--font-display);
            font-weight: 900;
            font-size: clamp(1.5em, 3.2vw, 2.1em);
            letter-spacing: 0.06em;
            text-transform: uppercase;
            line-height: 1.08;
            background: linear-gradient(180deg, #fbf6e9 18%, #e8dcc0 52%, #bda87c 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            /* drop-shadow filter removed: on background-clip:text it makes Chrome rasterise the layer per scroll tick and stalls capture */
        }
        .hero p {
            color: var(--ink-2);
            margin-top: 14px;
            font-size: 0.98em;
            letter-spacing: 0.02em;
        }
        .hero::after {
            content: '';
            display: block;
            width: 148px;
            height: 3px;
            margin: 16px auto 0;
            background: var(--grad);
            border-radius: 2px;
            box-shadow: 0 0 18px rgba(139,124,224,0.35);
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 32px 20px; }

        /* Account */
        .auth-area { display: flex; align-items: center; gap: 10px; }
        .status-bar .auth-area button { margin-left: 0; }
        .auth-chip {
            font-family: var(--font-mono);
            font-size: 0.92em;
            color: var(--gold);
            letter-spacing: 0.04em;
            text-shadow: 0 0 14px rgba(212,175,106,0.35);
        }
        .auth-modal-card {
            background: var(--surface);
            border: 1px solid var(--line-strong);
            border-radius: 16px;
            max-width: 400px;
            width: 100%;
            margin: 90px auto;
            padding: 30px;
            position: relative;
            box-shadow: 0 40px 90px -30px rgba(0,0,0,0.9), 0 0 50px -20px rgba(212,175,106,0.25);
        }
        .auth-modal-card h2 {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1.2em;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .auth-modal-card h2::before {
            content: '';
            display: block;
            width: 48px;
            height: 3px;
            background: var(--grad);
            border-radius: 2px;
            margin-bottom: 12px;
        }
        .auth-modal-sub { color: var(--ink-3); font-size: 0.85em; margin: 8px 0 18px; }
        .auth-field {
            display: block;
            width: 100%;
            padding: 12px 14px;
            margin-bottom: 12px;
            background: var(--surface-2);
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            color: var(--ink);
            font-family: var(--font-body);
            font-size: 0.95em;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .auth-field:focus { outline: none; border-color: var(--gold); box-shadow: 0 0 0 3px rgba(212,175,106,0.14); }
        .auth-field::placeholder { color: var(--ink-3); }
        .auth-error { color: var(--rose); font-size: 0.85em; margin: 2px 0 12px; min-height: 1.2em; }
        .auth-switch { font-size: 0.84em; color: var(--ink-3); margin-top: 16px; text-align: center; }
        .auth-switch a { color: var(--gold); cursor: pointer; }
        .auth-switch a:hover { text-decoration: underline; }

        .status-bar {
            background: linear-gradient(180deg, rgba(255,244,222,0.03), transparent), var(--surface);
            border: 1px solid var(--line);
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.05), 0 14px 30px -22px rgba(0,0,0,0.8);
            padding: 13px 20px;
            border-radius: 999px;
            margin-bottom: 26px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--ink-2);
            font-family: var(--font-mono);
            font-size: 0.85em;
        }
        .status-bar .icon { font-size: 1.05em; }
        .status-bar button {
            margin-left: auto;
            padding: 8px 20px;
            background: transparent;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            color: var(--gold);
            font-family: var(--font-mono);
            font-size: 0.9em;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            cursor: pointer;
            transition: all 0.25s;
        }
        .status-bar button:hover {
            background: var(--gold-grad);
            color: var(--gold-ink);
            border-color: transparent;
            box-shadow: 0 4px 18px -6px rgba(212,175,106,0.6);
        }

        .tabs {
            display: flex;
            gap: 2px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            border-bottom: 1px solid var(--line);
            align-items: center;
        }
        .tabs .tab-group-label { font-family: var(--font-mono); font-size: 0.62em; letter-spacing: 0.2em; text-transform: uppercase; color: var(--ink-3); padding: 0 10px 0 22px; border-left: 1px solid var(--line); margin-left: 12px; }
        .tabs .tab.secondary { font-size: 0.68em; color: var(--ink-3); padding: 13px 12px; }
        .funnel {
            display: grid; grid-template-columns: repeat(4, 1fr) auto; gap: 0; align-items: stretch;
            margin: 0 0 22px; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; background: var(--surface);
        }
        .funnel .step { padding: 12px 16px; border-right: 1px solid var(--line); cursor: pointer; transition: background 0.2s; }
        .funnel .step:hover { background: var(--surface-2); }
        .funnel .step .n { font-family: var(--font-mono); font-size: 0.62em; letter-spacing: 0.18em; text-transform: uppercase; color: var(--gold); }
        .funnel .step .t { font-family: var(--font-display); font-weight: 700; font-size: 0.92em; color: var(--ink); margin-top: 3px; }
        .funnel .step .d { color: var(--ink-3); font-size: 0.76em; margin-top: 2px; }
        .funnel .picks { padding: 12px 18px; display: flex; flex-direction: column; justify-content: center; gap: 6px; min-width: 200px; background: linear-gradient(90deg, rgba(212,175,106,0.08), transparent); }
        .funnel .picks .k { font-family: var(--font-mono); font-size: 0.62em; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-3); }
        .funnel .picks .v { color: var(--ink); font-size: 0.86em; }
        .funnel .picks a { color: var(--gold); text-decoration: none; font-family: var(--font-mono); font-size: 0.72em; letter-spacing: 0.12em; text-transform: uppercase; }
        @media (max-width: 1000px) { .funnel { grid-template-columns: 1fr 1fr; } .funnel .picks { grid-column: 1 / -1; } }
        .tab {
            padding: 13px 17px;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            color: var(--ink-3);
            font-family: var(--font-mono);
            font-size: 0.76em;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            cursor: pointer;
            transition: color 0.2s, border-color 0.2s, text-shadow 0.2s;
        }
        .tab:hover { color: var(--ink); }
        .tab.active {
            color: var(--gold);
            border-bottom-color: var(--gold);
            text-shadow: 0 0 16px rgba(212,175,106,0.55);
        }

        .section { display: none; }
        .section.active { display: block; animation: rise 0.35s ease both; }
        @keyframes rise {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: none; }
        }
        @media (prefers-reduced-motion: reduce) {
            .section.active { animation: none; }
        }

        .card {
            position: relative;
            background: linear-gradient(180deg, rgba(255,244,222,0.035), rgba(255,244,222,0.008) 38%, transparent), var(--surface);
            border: 1px solid var(--line);
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.05), 0 24px 48px -30px rgba(0,0,0,0.85);
            padding: 30px;
            margin-bottom: 22px;
            border-radius: 18px;
        }
        .card h2 {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1.2em;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--ink);
            margin-bottom: 18px;
        }
        .card h2::before {
            content: '';
            display: block;
            width: 52px;
            height: 3px;
            background: var(--grad);
            border-radius: 2px;
            margin-bottom: 14px;
            box-shadow: 0 0 12px rgba(139,124,224,0.4);
        }
        .card > p { max-width: 72ch; }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 14px;
            margin-bottom: 8px;
        }
        .metric {
            text-align: center;
            padding: 24px 15px 20px;
            background: linear-gradient(180deg, rgba(226,193,132,0.07), rgba(226,193,132,0.015) 55%, transparent), var(--surface-2);
            border: 1px solid var(--line-strong);
            border-radius: 14px;
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.07);
        }
        .metric-value {
            font-family: var(--font-display);
            font-size: 2.5em;
            font-weight: 800;
            letter-spacing: 0.01em;
            background: var(--gold-grad);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            /* drop-shadow filter removed: on background-clip:text it makes Chrome rasterise the layer per scroll tick and stalls capture */
            font-variant-numeric: tabular-nums;
        }
        .metric-label {
            color: var(--ink-3);
            font-family: var(--font-mono);
            font-size: 0.68em;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-top: 8px;
        }

        .trend-card {
            background: linear-gradient(135deg, rgba(139,124,224,0.16), rgba(139,124,224,0.05) 48%, rgba(139,124,224,0.02)), var(--surface);
            border: 1px solid rgba(139,124,224,0.32);
            border-left: 3px solid var(--violet);
            color: var(--ink);
            padding: 20px;
            border-radius: 14px;
            margin-bottom: 14px;
            box-shadow: 0 18px 36px -26px rgba(139,124,224,0.55);
            transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
        }
        .trend-card h3 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; letter-spacing: 0.01em; }
        .trend-card p { color: var(--ink-2); font-size: 0.9em; line-height: 1.55; }
        .trend-card:hover {
            transform: translateY(-3px);
            border-color: rgba(139,124,224,0.6);
            box-shadow: 0 24px 44px -24px rgba(139,124,224,0.7);
        }
        .trend-meta { margin-top: 10px; font-size: 0.8em; color: var(--ink-2); font-family: var(--font-mono); }

        .move-card {
            background: linear-gradient(135deg, rgba(58,168,141,0.15), rgba(58,168,141,0.05) 48%, rgba(58,168,141,0.02)), var(--surface);
            border: 1px solid rgba(58,168,141,0.32);
            border-left: 3px solid var(--teal);
            color: var(--ink);
            padding: 20px;
            border-radius: 14px;
            margin-bottom: 14px;
            box-shadow: 0 18px 36px -26px rgba(58,168,141,0.5);
            transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
        }
        .move-card h3 { margin-bottom: 5px; font-family: var(--font-display); font-weight: 600; }
        .move-card .company { font-size: 0.84em; color: var(--teal); font-family: var(--font-mono); letter-spacing: 0.08em; margin-bottom: 8px; }
        .move-card p { font-size: 0.9em; line-height: 1.55; color: var(--ink-2); }
        .move-card:hover {
            transform: translateY(-3px);
            border-color: rgba(58,168,141,0.6);
            box-shadow: 0 24px 44px -24px rgba(58,168,141,0.65);
        }
        .move-badges { display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
        .move-type-badge {
            background: rgba(58,168,141,0.18);
            color: var(--teal);
            padding: 4px 11px;
            border-radius: 999px;
            font-family: var(--font-mono);
            font-size: 0.7em;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .market-badge {
            background: var(--surface-2);
            border: 1px solid var(--line);
            color: var(--ink-2);
            padding: 4px 11px;
            border-radius: 999px;
            font-size: 0.78em;
        }
        .move-actions { margin-top: 12px; display: flex; gap: 8px; }
        .move-action-btn {
            padding: 7px 15px;
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

        .content-item { padding: 15px 4px; border-bottom: 1px solid var(--line); }
        .content-item:last-child { border-bottom: none; }
        .content-item h4 { color: var(--ink); margin-bottom: 5px; font-weight: 600; }
        .content-item p { color: var(--ink-2); font-size: 0.9em; }
        .content-item .meta { font-size: 0.76em; color: var(--ink-3); margin-top: 6px; font-family: var(--font-mono); }
        .content-item .source {
            background: rgba(212,175,106,0.15);
            color: var(--gold);
            padding: 2px 9px;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-size: 0.72em;
            letter-spacing: 0.06em;
        }

        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 11px 10px; text-align: left; border-bottom: 1px solid var(--line); }
        th {
            background: transparent;
            color: var(--ink-3);
            font-family: var(--font-mono);
            font-size: 0.7em;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 500;
        }
        td { color: var(--ink-2); font-size: 0.9em; font-variant-numeric: tabular-nums; }

        .badge { padding: 3px 10px; border-radius: 4px; font-size: 0.78em; font-family: var(--font-mono); }
        .badge-success { background: rgba(58,168,141,0.16); color: var(--teal); }
        .badge-warning { background: rgba(184,134,46,0.16); color: var(--gold); }

        .empty { text-align: center; padding: 48px; color: var(--ink-3); }
        .empty .icon { font-size: 0; display: block; margin-bottom: 12px; }
        .empty .icon::before {
            content: '\\2726';
            font-size: 1.6rem;
            color: var(--gold);
            opacity: 0.65;
            text-shadow: 0 0 18px rgba(212,175,106,0.5);
        }

        .error {
            background: rgba(194,90,120,0.12);
            border: 1px solid rgba(194,90,120,0.35);
            color: var(--rose);
            padding: 15px;
            border-radius: 10px;
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
            padding: 3px 10px;
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
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filter-select, select {
            padding: 9px 13px;
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            background: var(--surface-2);
            color: var(--ink);
            font-family: var(--font-body);
            font-size: 0.88em;
            min-width: 140px;
            cursor: pointer;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .filter-select:focus, select:focus {
            outline: none;
            border-color: var(--gold);
            box-shadow: 0 0 0 3px rgba(212,175,106,0.14);
        }
        .filter-reset, button.filter-reset {
            padding: 9px 17px;
            background: transparent;
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            color: var(--ink-2);
            cursor: pointer;
            font-size: 0.88em;
            transition: all 0.2s;
        }
        .filter-reset:hover { color: var(--ink); border-color: var(--gold); }
        .saved-count {
            background: rgba(139,124,224,0.14);
            color: var(--violet);
            padding: 4px 11px;
            border-radius: 999px;
            font-family: var(--font-mono);
            font-size: 0.76em;
            margin-left: auto;
        }

        input[type="text"], input[type="number"], textarea {
            background: var(--surface-2);
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            color: var(--ink);
            font-family: var(--font-body);
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input[type="text"]:focus, input[type="number"]:focus, textarea:focus {
            outline: none;
            border-color: var(--gold);
            box-shadow: 0 0 0 3px rgba(212,175,106,0.14);
        }
        input::placeholder, textarea::placeholder { color: var(--ink-3); }

        /* Trend Action Buttons */
        .trend-actions {
            display: flex;
            gap: 8px;
            margin-top: 14px;
            padding-top: 14px;
            border-top: 1px solid rgba(139,124,224,0.18);
        }
        .trend-action-btn {
            padding: 7px 15px;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            font-size: 0.82em;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: transform 0.15s, border-color 0.2s, box-shadow 0.2s;
        }
        .trend-action-btn:hover { transform: translateY(-1px); border-color: var(--gold); }
        .btn-save { background: transparent; color: var(--ink-2); }
        .btn-save.saved { background: var(--teal); border-color: var(--teal); color: var(--gold-ink); }
        .btn-brand {
            background: var(--gold-grad);
            border-color: transparent;
            color: var(--gold-ink);
            font-weight: 600;
            box-shadow: 0 6px 16px -8px rgba(212,175,106,0.55), inset 0 1px 0 rgba(255,255,255,0.28);
        }
        .btn-brand:hover { box-shadow: 0 8px 22px -8px rgba(212,175,106,0.75), inset 0 1px 0 rgba(255,255,255,0.28); }

        /* Chat */
        .chat-message { margin-bottom: 15px; }
        .chat-message.user { text-align: right; }
        .chat-message.assistant { text-align: left; }
        .chat-bubble {
            display: inline-block;
            max-width: 80%;
            padding: 12px 17px;
            border-radius: 16px;
            line-height: 1.55;
            text-align: left;
        }
        .chat-message.user .chat-bubble {
            background: linear-gradient(135deg, rgba(226,193,132,0.20), rgba(212,175,106,0.10));
            border: 1px solid rgba(212,175,106,0.32);
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
            padding: 8px 17px;
            background: transparent;
            border: 1px solid var(--line-strong);
            color: var(--gold);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.88em;
            transition: all 0.22s;
        }
        .suggestion-chip:hover {
            background: var(--gold-grad);
            color: var(--gold-ink);
            border-color: transparent;
            box-shadow: 0 6px 16px -8px rgba(212,175,106,0.6);
        }
        .chat-typing { display: flex; gap: 4px; padding: 10px 15px; }
        .chat-typing span {
            width: 7px;
            height: 7px;
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
            background: linear-gradient(135deg, rgba(139,124,224,0.14), rgba(139,124,224,0.04) 55%, transparent), var(--surface-2);
            border: 1px solid rgba(139,124,224,0.30);
            border-left: 3px solid var(--violet);
            color: var(--ink);
            padding: 22px;
            border-radius: 14px;
            margin-bottom: 15px;
            box-shadow: 0 18px 36px -26px rgba(139,124,224,0.5);
        }
        .profile-insights-card h3 { margin-bottom: 15px; font-family: var(--font-display); font-weight: 600; }
        .profile-tag {
            display: inline-block;
            background: rgba(139,124,224,0.15);
            color: #b1a5ec;
            padding: 5px 13px;
            border-radius: 999px;
            margin: 3px;
            font-size: 0.84em;
        }
        .profile-section { margin-bottom: 12px; }
        .profile-section-title {
            font-family: var(--font-mono);
            font-size: 0.7em;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--ink-3);
            margin-bottom: 6px;
        }
        .btn-primary {
            padding: 13px 26px;
            background: var(--gold-grad);
            color: var(--gold-ink);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95em;
            box-shadow: 0 8px 22px -10px rgba(212,175,106,0.6), inset 0 1px 0 rgba(255,255,255,0.28);
            transition: transform 0.15s, box-shadow 0.2s;
        }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 12px 28px -10px rgba(212,175,106,0.75), inset 0 1px 0 rgba(255,255,255,0.28); }
        .btn-primary:disabled { background: var(--surface-3); color: var(--ink-3); cursor: not-allowed; box-shadow: none; transform: none; }
        .btn-secondary {
            padding: 13px 26px;
            background: transparent;
            color: var(--ink-2);
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95em;
            transition: all 0.2s;
        }
        .btn-secondary:hover { color: var(--ink); border-color: var(--gold); }
        .saved-item-card {
            background: var(--surface-2);
            border: 1px solid var(--line);
            padding: 13px 16px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .saved-item-card h4 { margin-bottom: 4px; color: var(--ink); }
        .saved-item-meta { font-size: 0.82em; color: var(--ink-3); }
        .saved-item-actions { display: flex; gap: 8px; }
        .btn-remove {
            padding: 5px 13px;
            background: transparent;
            color: var(--rose);
            border: 1px solid rgba(194,90,120,0.4);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.82em;
            transition: all 0.2s;
        }
        .btn-remove:hover { background: var(--rose); color: var(--gold-ink); }

        /* Modal */
        .modal-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(8,6,4,0.72); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
            z-index: 1000; overflow-y: auto; padding: 20px;
        }
        .modal-overlay.active { display: flex; justify-content: center; align-items: flex-start; }
        .modal-content {
            background: var(--surface);
            border: 1px solid var(--line-strong);
            border-radius: 16px;
            box-shadow: 0 40px 90px -30px rgba(0,0,0,0.9);
            max-width: 700px; width: 100%; margin: 40px auto; position: relative;
            overflow: hidden;
        }
        .modal-header {
            background: linear-gradient(135deg, rgba(139,124,224,0.14), transparent 60%), var(--surface-2);
            border-bottom: 1px solid var(--line);
            border-top: 3px solid var(--violet);
            color: var(--ink);
            padding: 22px;
        }
        .modal-header.move-header {
            background: linear-gradient(135deg, rgba(58,168,141,0.14), transparent 60%), var(--surface-2);
            border-top-color: var(--teal);
        }
        .modal-header h2 { margin: 0; font-size: 1.3em; line-height: 1.3; font-family: var(--font-display); font-weight: 700; }
        .modal-header .meta { color: var(--ink-2); margin-top: 8px; font-size: 0.84em; font-family: var(--font-mono); }
        .modal-body { padding: 22px; max-height: 60vh; overflow-y: auto; }
        .modal-section { margin-bottom: 20px; }
        .modal-section:last-child { margin-bottom: 0; }
        .modal-section h3 {
            color: var(--gold);
            margin-bottom: 10px;
            font-family: var(--font-mono);
            font-size: 0.75em;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-weight: 500;
        }
        .modal-section p { color: var(--ink-2); line-height: 1.65; }
        .modal-close {
            position: absolute; top: 16px; right: 16px;
            background: rgba(13,11,8,0.5); border: 1px solid var(--line-strong);
            color: var(--ink-2); width: 32px; height: 32px; border-radius: 50%;
            cursor: pointer; font-size: 1.05em;
            transition: all 0.2s;
        }
        .modal-close:hover { color: var(--ink); border-color: var(--gold); }
        .source-quote {
            background: var(--surface-2);
            border-left: 3px solid var(--gold);
            padding: 13px 16px;
            margin-bottom: 10px;
            border-radius: 0 10px 10px 0;
            font-style: italic;
            color: var(--ink-2);
            font-size: 0.9em;
        }
        .topic-tag {
            display: inline-block;
            background: rgba(139,124,224,0.15);
            color: #b1a5ec;
            padding: 4px 11px;
            border-radius: 999px;
            font-size: 0.8em;
            margin: 3px;
        }

        .quick-city {
            padding: 6px 15px;
            background: transparent;
            border: 1px solid var(--line-strong);
            color: var(--ink-2);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.82em;
            margin: 2px;
            transition: all 0.22s;
        }
        .quick-city:hover {
            color: var(--gold);
            border-color: var(--gold);
            box-shadow: 0 0 14px -4px rgba(212,175,106,0.5);
        }

        .desire-card {
            background: linear-gradient(135deg, rgba(194,90,120,0.15), rgba(194,90,120,0.05) 48%, rgba(194,90,120,0.02)), var(--surface);
            border: 1px solid rgba(194,90,120,0.32);
            border-left: 3px solid var(--rose);
            color: var(--ink);
            padding: 20px;
            border-radius: 14px;
            margin-bottom: 14px;
            box-shadow: 0 18px 36px -26px rgba(194,90,120,0.5);
        }
        .desire-card h4 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }
        .desire-card p { font-size: 0.9em; color: var(--ink-2); line-height: 1.55; }
        .desire-meta { margin-top: 10px; font-size: 0.8em; color: var(--ink-2); font-family: var(--font-mono); }

        .opportunity-card {
            background: linear-gradient(135deg, rgba(74,139,194,0.14), rgba(74,139,194,0.04) 55%, transparent), var(--surface);
            border: 1px solid rgba(74,139,194,0.32);
            border-left: 3px solid var(--blue);
            color: var(--ink);
            padding: 15px 17px;
            border-radius: 12px;
            margin-bottom: 9px;
            box-shadow: 0 14px 30px -24px rgba(74,139,194,0.5);
        }

        .concept-card {
            background: linear-gradient(135deg, rgba(226,193,132,0.15), rgba(212,175,106,0.05) 48%, rgba(212,175,106,0.02)), var(--surface);
            border: 1px solid rgba(212,175,106,0.36);
            border-left: 3px solid var(--gold);
            color: var(--ink);
            padding: 20px;
            border-radius: 14px;
            margin-bottom: 14px;
            box-shadow: 0 18px 36px -26px rgba(212,175,106,0.45);
        }
        .concept-card h4 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }

        /* Demand Scan */
        .property-card {
            background: linear-gradient(180deg, rgba(255,244,222,0.03), transparent 45%), var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 22px;
            margin-bottom: 15px;
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.05), 0 20px 40px -30px rgba(0,0,0,0.8);
            transition: border-color 0.2s, transform 0.18s, box-shadow 0.2s;
        }
        .property-card:hover {
            border-color: var(--line-strong);
            transform: translateY(-2px);
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.05), 0 26px 50px -30px rgba(0,0,0,0.9);
        }
        .property-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }
        .property-card h3 { margin: 0; color: var(--ink); font-size: 1.15em; font-family: var(--font-display); font-weight: 600; }
        .property-card .location { color: var(--ink-3); font-size: 0.84em; margin-top: 4px; font-family: var(--font-mono); }

        /* Demand Fit Score Badge */
        .demand-score {
            padding: 8px 16px;
            border-radius: 999px;
            font-family: var(--font-mono);
            font-weight: 500;
            font-size: 0.9em;
            font-variant-numeric: tabular-nums;
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
            padding: 4px 11px;
            border-radius: 6px;
            font-size: 0.78em;
            margin: 3px;
        }
        .misalignment-flag::before { content: "! "; font-weight: 700; margin-right: 4px; }

        /* Property Sections */
        .property-section { margin-bottom: 15px; }
        .property-section-title {
            font-family: var(--font-mono);
            font-size: 0.7em;
            font-weight: 500;
            color: var(--ink-3);
            margin-bottom: 8px;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }
        .gap-item {
            display: inline-block;
            background: rgba(184,134,46,0.14);
            color: var(--gold);
            padding: 4px 11px;
            border-radius: 6px;
            font-size: 0.82em;
            margin: 2px;
        }
        .opportunity-item {
            display: flex;
            align-items: center;
            background: rgba(74,139,194,0.10);
            color: #7fb3dd;
            padding: 9px 13px;
            border-radius: 8px;
            font-size: 0.88em;
            margin-bottom: 6px;
        }
        .opportunity-item::before { content: "\\2192 "; font-weight: bold; margin-right: 7px; color: var(--blue); }

        .property-actions {
            display: flex;
            gap: 10px;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--line);
        }
        .property-action-btn {
            padding: 9px 19px;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.88em;
            font-weight: 500;
            transition: all 0.2s;
        }
        .property-action-btn.btn-brand {
            background: var(--gold-grad);
            border-color: transparent;
            color: var(--gold-ink);
            box-shadow: 0 6px 16px -8px rgba(212,175,106,0.55), inset 0 1px 0 rgba(255,255,255,0.28);
        }
        .property-action-btn.btn-brand:hover { box-shadow: 0 8px 22px -8px rgba(212,175,106,0.75), inset 0 1px 0 rgba(255,255,255,0.28); }
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

        /* =============================================
           Signal Room (overview) + Signal Ledger
           Marks stay thin; the data is the only loud thing.
           ============================================= */
        .room-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 22px; }
        .room-grid > .card { margin-bottom: 0; }
        .span-12 { grid-column: span 12; }
        .span-8 { grid-column: span 8; }
        .span-7 { grid-column: span 7; }
        .span-6 { grid-column: span 6; }
        .span-5 { grid-column: span 5; }
        .span-4 { grid-column: span 4; }
        @media (max-width: 1100px) { .span-8, .span-7, .span-6, .span-5, .span-4 { grid-column: span 12; } }
        .room-grid + .room-grid { margin-top: 22px; }

        .card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
        .card-head h2 { margin-bottom: 6px; }
        .card-sub { color: var(--ink-3); font-size: 0.86em; max-width: 64ch; margin-bottom: 18px; line-height: 1.5; }
        .card-tools { display: flex; gap: 8px; align-items: center; }
        .tool-btn {
            background: transparent; border: 1px solid var(--line-strong); color: var(--ink-2);
            font-family: var(--font-mono); font-size: 0.7em; letter-spacing: 0.14em; text-transform: uppercase;
            padding: 7px 12px; border-radius: 8px; cursor: pointer; transition: all 0.2s;
        }
        .tool-btn:hover, .tool-btn.active { color: var(--gold); border-color: var(--gold); }

        /* Stat tiles: value + delta + sparkline */
        .stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }
        .stat {
            position: relative; padding: 22px 22px 18px;
            background: linear-gradient(180deg, rgba(226,193,132,0.07), rgba(226,193,132,0.015) 55%, transparent), var(--surface-2);
            border: 1px solid var(--line-strong); border-radius: 14px;
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.07);
            min-height: 132px; display: flex; flex-direction: column; justify-content: space-between;
        }
        .stat-label { color: var(--ink-3); font-family: var(--font-mono); font-size: 0.68em; letter-spacing: 0.18em; text-transform: uppercase; }
        .stat-value {
            font-family: var(--font-display); font-weight: 800; font-size: 2.3em; line-height: 1.1; margin-top: 8px;
            background: var(--gold-grad); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
            /* drop-shadow filter removed: on background-clip:text it makes Chrome rasterise the layer per scroll tick and stalls capture */
        }
        .stat-foot { display: flex; align-items: flex-end; justify-content: space-between; gap: 10px; margin-top: 10px; }
        .stat-delta { font-family: var(--font-mono); font-size: 0.72em; color: var(--ink-2); letter-spacing: 0.02em; line-height: 1.4; white-space: nowrap; }
        .stat-delta.up { color: var(--teal); }
        .stat-delta.down { color: var(--rose); }
        .stat-delta .vs { color: var(--ink-3); }
        .stat svg.spark { width: 72px; height: 30px; overflow: hidden; flex: none; }

        /* Demand curves */
        .chart-legend { display: flex; flex-wrap: wrap; gap: 14px 18px; margin: 4px 0 12px; font-size: 0.8em; color: var(--ink-2); }
        .chart-legend .key { display: inline-flex; align-items: center; gap: 8px; }
        .chart-legend .key i { display: inline-block; width: 18px; height: 2px; border-radius: 1px; background: var(--ink-3); }
        .chart-legend .key.rest i { opacity: 0.45; }
        .chart-wrap { position: relative; }
        .chart-wrap svg.curves { width: 100%; height: auto; display: block; overflow: hidden; }
        .chart-wrap .grid line { stroke: rgba(242,236,223,0.07); stroke-width: 1; }
        .chart-wrap .baseline { stroke: rgba(212,175,106,0.35); stroke-width: 1; }
        .chart-wrap .axis text { fill: var(--ink-3); font-family: var(--font-mono); font-size: 11px; }
        .chart-wrap .series { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
        .chart-wrap .series.rest { stroke: var(--ink-3); stroke-opacity: 0.22; stroke-width: 1.5; }
        .chart-wrap .end-dot { stroke: var(--surface); stroke-width: 2; }
        .chart-wrap .end-label { fill: var(--ink); font-family: var(--font-body); font-size: 12px; font-weight: 600; }
        .chart-wrap .end-label tspan.pct { fill: var(--ink-2); font-family: var(--font-mono); font-weight: 400; }
        .chart-wrap .crosshair { stroke: rgba(242,236,223,0.35); stroke-width: 1; pointer-events: none; }
        .chart-wrap .hit { fill: transparent; cursor: crosshair; }
        .chart-tip {
            position: absolute; pointer-events: none; z-index: 5; min-width: 170px;
            background: var(--surface-3); border: 1px solid var(--line-strong); border-radius: 10px;
            padding: 10px 12px; box-shadow: 0 18px 40px -20px rgba(0,0,0,0.9); font-size: 0.8em;
            transform: translate(-50%, 0);
        }
        .chart-tip[hidden] { display: none; }
        .chart-tip .tip-date { color: var(--ink-3); font-family: var(--font-mono); font-size: 0.86em; margin-bottom: 6px; letter-spacing: 0.04em; }
        .chart-tip .tip-row { display: flex; align-items: center; gap: 8px; padding: 2px 0; }
        .chart-tip .tip-row i { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }
        .chart-tip .tip-row b { color: var(--ink); font-family: var(--font-mono); font-weight: 500; margin-left: auto; }
        .chart-tip .tip-row span { color: var(--ink-2); }
        .chart-table { width: 100%; border-collapse: collapse; font-size: 0.84em; margin-top: 6px; }
        .chart-table th { text-align: left; color: var(--ink-3); font-family: var(--font-mono); font-weight: 400; font-size: 0.8em; letter-spacing: 0.12em; text-transform: uppercase; padding: 8px 6px; border-bottom: 1px solid var(--line); }
        .chart-table td { padding: 8px 6px; border-bottom: 1px solid rgba(242,236,223,0.05); font-variant-numeric: tabular-nums; }
        .chart-table td.num { text-align: right; font-family: var(--font-mono); }

        /* Movers */
        .movers { display: grid; grid-template-columns: 1fr 1fr; gap: 18px 28px; margin-top: 18px; }
        @media (max-width: 700px) { .movers { grid-template-columns: 1fr; } }
        .movers h4 { font-family: var(--font-mono); font-weight: 400; font-size: 0.68em; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 10px; }
        .mover { display: grid; grid-template-columns: 120px 1fr 62px; align-items: center; gap: 10px; padding: 5px 0; font-size: 0.86em; }
        .mover .name { color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .mover .bar { height: 6px; border-radius: 0 3px 3px 0; background: var(--teal); min-width: 3px; }
        .mover.down .bar { background: var(--rose); }
        .mover .pct { text-align: right; font-family: var(--font-mono); color: var(--ink-2); font-size: 0.92em; }

        /* Trend movers / bets lists */
        .signal-list { display: flex; flex-direction: column; gap: 10px; }
        .signal-item {
            display: grid; grid-template-columns: 1fr auto; gap: 6px 14px; align-items: start;
            padding: 14px 16px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface-2);
            cursor: pointer; transition: border-color 0.2s, transform 0.2s;
        }
        .signal-item:hover { border-color: var(--line-strong); transform: translateY(-1px); }
        .signal-item h3 { font-family: var(--font-body); font-weight: 600; font-size: 0.98em; color: var(--ink); line-height: 1.35; }
        .signal-item p { color: var(--ink-2); font-size: 0.82em; line-height: 1.45; grid-column: 1 / -1; }
        .signal-item .kicker { font-family: var(--font-mono); font-size: 0.68em; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3); grid-column: 1 / -1; }
        .signal-item .kicker b { color: var(--gold); font-weight: 500; }
        .meter { display: flex; flex-direction: column; gap: 5px; min-width: 110px; }
        .meter .m-label { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.66em; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
        .meter .m-track { height: 5px; border-radius: 3px; background: rgba(212,175,106,0.14); overflow: hidden; }
        .meter .m-fill { height: 100%; border-radius: 3px; background: var(--gold); }
        .meter.violet .m-track { background: rgba(139,124,224,0.16); }
        .meter.violet .m-fill { background: var(--violet); }

        /* Coverage strip */
        .coverage { display: flex; flex-wrap: wrap; gap: 8px; }
        .chip {
            display: inline-flex; align-items: center; gap: 8px; padding: 7px 11px; border-radius: 999px;
            border: 1px solid var(--line); background: var(--surface-2); font-size: 0.78em; color: var(--ink-2);
        }
        .chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ink-3); box-shadow: 0 0 0 2px var(--surface-2); }
        .chip.live .dot { background: var(--good); box-shadow: 0 0 0 2px var(--surface-2), 0 0 10px rgba(58,168,141,0.6); }
        .chip.stale .dot { background: var(--warn); }
        .chip.silent .dot { background: var(--bad); }
        .chip.blocked { opacity: 0.6; border-style: dashed; }
        .chip.blocked .dot { background: var(--ink-3); }
        .chip .n { font-family: var(--font-mono); color: var(--ink-3); font-size: 0.9em; }
        .chip .age { font-family: var(--font-mono); color: var(--ink-3); font-size: 0.86em; }
        .coverage-legend { display: flex; gap: 16px; margin-top: 14px; font-family: var(--font-mono); font-size: 0.68em; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
        .coverage-legend span::before { content: ''; display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; background: var(--ink-3); vertical-align: middle; }
        .coverage-legend .l-live::before { background: var(--good); }
        .coverage-legend .l-stale::before { background: var(--warn); }
        .coverage-legend .l-silent::before { background: var(--bad); }
        .coverage-legend .l-blocked::before { background: var(--ink-3); }

        /* Active inference */
        .ai-head { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }
        @media (max-width: 700px) { .ai-head { grid-template-columns: 1fr; } }
        .ai-fig { padding: 14px 16px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface-2); }
        .ai-fig .f-label { font-family: var(--font-mono); font-size: 0.66em; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-3); }
        .ai-fig .f-value { font-family: var(--font-display); font-weight: 700; font-size: 1.5em; color: var(--ink); margin-top: 4px; }
        .ai-fig .f-value small { font-family: var(--font-mono); font-weight: 400; font-size: 0.55em; color: var(--ink-3); margin-left: 6px; }
        .belief { display: grid; grid-template-columns: 150px 1fr 110px 24px; gap: 12px; align-items: center; padding: 7px 0; border-bottom: 1px solid rgba(242,236,223,0.05); font-size: 0.84em; }
        .belief .b-name { color: var(--ink); font-family: var(--font-mono); font-size: 0.9em; }
        .belief .b-track { height: 6px; border-radius: 3px; background: rgba(139,124,224,0.16); overflow: hidden; position: relative; }
        .belief .b-fill { height: 100%; background: var(--violet); border-radius: 3px; }
        .belief .b-val { text-align: right; font-family: var(--font-mono); color: var(--ink-2); font-size: 0.9em; }
        .belief .b-flag { color: var(--gold); text-align: center; }
        .ai-explain { color: var(--ink-3); font-size: 0.8em; line-height: 1.5; margin-top: 14px; }
        .ai-explain code { font-family: var(--font-mono); color: var(--ink-2); }

        /* Ledger */
        .ledger-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 22px; }
        .ledger-kpi { padding: 16px 18px; border: 1px solid var(--line-strong); border-radius: 12px; background: var(--surface-2); }
        .ledger-kpi .k-label { font-family: var(--font-mono); font-size: 0.66em; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-3); }
        .ledger-kpi .k-value { font-family: var(--font-display); font-weight: 800; font-size: 1.9em; color: var(--ink); margin-top: 4px; }
        .ledger-kpi .k-note { color: var(--ink-3); font-size: 0.74em; margin-top: 4px; }
        .pred {
            border: 1px solid var(--line); border-left: 3px solid var(--gold); border-radius: 12px;
            background: linear-gradient(135deg, rgba(212,175,106,0.10), rgba(212,175,106,0.02) 50%, transparent), var(--surface);
            padding: 18px 20px; margin-bottom: 12px;
        }
        .pred-top { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
        .pred h3 { font-family: var(--font-body); font-weight: 600; font-size: 1.02em; color: var(--ink); }
        .pred .seal { font-family: var(--font-mono); font-size: 0.72em; color: var(--ink-3); letter-spacing: 0.04em; margin-top: 4px; }
        .pred .seal b { color: var(--gold); font-weight: 500; }
        .pred .status { font-family: var(--font-mono); font-size: 0.68em; letter-spacing: 0.14em; text-transform: uppercase; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line-strong); color: var(--ink-2); white-space: nowrap; }
        .pred .status.open { color: var(--teal); border-color: rgba(58,168,141,0.5); }
        .pred .status.hit { color: var(--good); }
        .pred .status.miss, .pred .status.falsified { color: var(--bad); }
        .pred p.hyp { color: var(--ink-2); font-size: 0.86em; line-height: 1.5; margin: 12px 0 10px; max-width: 90ch; }
        .forecasts { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-top: 8px; }
        .forecast { padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface-2); font-size: 0.8em; }
        .forecast .f-metric { font-family: var(--font-mono); font-size: 0.82em; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
        .forecast .f-range { color: var(--ink); font-weight: 600; margin-top: 3px; font-size: 1.05em; }
        .forecast .f-range small { color: var(--ink-3); font-weight: 400; font-family: var(--font-mono); }
        .forecast .f-fals { color: var(--ink-3); margin-top: 5px; line-height: 1.4; }
        .pred-foot { display: flex; gap: 14px; align-items: center; margin-top: 12px; flex-wrap: wrap; font-size: 0.76em; color: var(--ink-3); font-family: var(--font-mono); }
        .pred-foot button { background: transparent; border: 1px solid var(--line-strong); color: var(--ink-2); font-family: var(--font-mono); font-size: 0.9em; letter-spacing: 0.1em; text-transform: uppercase; padding: 5px 10px; border-radius: 7px; cursor: pointer; }
        .pred-foot button:hover { color: var(--gold); border-color: var(--gold); }
        .events { margin-top: 12px; border-top: 1px solid var(--line); padding-top: 10px; font-size: 0.82em; }
        .events .ev { display: grid; grid-template-columns: 120px 90px 1fr; gap: 12px; padding: 5px 0; color: var(--ink-2); }
        .events .ev .t { font-family: var(--font-mono); color: var(--ink-3); font-size: 0.9em; }
        .events .ev .k { font-family: var(--font-mono); color: var(--gold); font-size: 0.86em; letter-spacing: 0.1em; text-transform: uppercase; }
        .ledger-how { color: var(--ink-3); font-size: 0.84em; line-height: 1.55; max-width: 80ch; margin-bottom: 20px; }
        .ledger-how b { color: var(--ink-2); font-weight: 600; }

        /* Scatter figures */
        .fig-wrap { position: relative; }
        .fig-wrap svg { width: 100%; height: auto; display: block; overflow: hidden; }
        .fig-wrap .grid line { stroke: rgba(242,236,223,0.07); stroke-width: 1; }
        .fig-wrap .median { stroke: rgba(212,175,106,0.28); stroke-width: 1; }
        .fig-wrap .axis text { fill: var(--ink-3); font-family: var(--font-mono); font-size: 11px; }
        .fig-wrap .axis-title { fill: var(--ink-3); font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; }
        .fig-wrap .quad { fill: var(--ink-3); font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.8; }
        .fig-wrap .dot { stroke: var(--surface); stroke-width: 2; cursor: pointer; transition: opacity 0.15s; }
        .fig-wrap .dot.dim { opacity: 0.25; }
        .fig-wrap .lbl { fill: var(--ink); font-family: var(--font-body); font-size: 11.5px; font-weight: 600; pointer-events: none; paint-order: stroke; stroke: var(--surface); stroke-width: 3px; stroke-linejoin: round; }
        .fig-wrap .bar { cursor: pointer; }
        .fig-wrap .bar:hover { filter: brightness(1.15); }
        .fig-legend { display: flex; flex-wrap: wrap; gap: 10px 18px; margin: 2px 0 10px; font-size: 0.8em; color: var(--ink-2); }
        .fig-legend .key { display: inline-flex; align-items: center; gap: 8px; }
        .fig-legend .key i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
        .fig-legend .key i.sq { border-radius: 2px; width: 12px; height: 12px; }
        .fig-foot { color: var(--ink-3); font-size: 0.78em; line-height: 1.5; margin-top: 10px; }

        /* Stake a prediction */
        .stake-form { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
        .stake-form .full { grid-column: 1 / -1; }
        .stake-form label { display: block; font-family: var(--font-mono); font-size: 0.66em; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 6px; }
        .stake-form input, .stake-form textarea, .stake-form select {
            width: 100%; background: var(--surface-2); border: 1px solid var(--line-strong); border-radius: 9px;
            color: var(--ink); font-family: var(--font-body); font-size: 0.9em; padding: 9px 11px;
        }
        .stake-form textarea { min-height: 72px; resize: vertical; }
        .stake-form input:focus, .stake-form textarea:focus { outline: none; border-color: var(--gold); }
        .stake-actions { display: flex; gap: 10px; align-items: center; justify-content: flex-end; margin-top: 16px; }
        .stake-hint { color: var(--ink-3); font-size: 0.8em; line-height: 1.5; margin-bottom: 14px; max-width: 70ch; }
        .stake-result { margin-top: 14px; padding: 14px 16px; border: 1px solid rgba(58,168,141,0.45); border-radius: 12px; background: rgba(58,168,141,0.08); font-size: 0.86em; color: var(--ink-2); line-height: 1.5; }
        .stake-result b { font-family: var(--font-mono); color: var(--teal); font-weight: 500; word-break: break-all; }
        .btn-stake { background: transparent; border: 1px solid var(--line-strong); color: var(--ink-2); font-family: var(--font-mono); font-size: 0.7em; letter-spacing: 0.12em; text-transform: uppercase; padding: 6px 11px; border-radius: 8px; cursor: pointer; }
        .btn-stake:hover { color: var(--gold); border-color: var(--gold); }

        /* Demand Scan: brief + alignment */
        .brief { margin: 14px 0 6px; padding: 14px 16px; border-left: 3px solid var(--gold); border-radius: 0 12px 12px 0; background: linear-gradient(90deg, rgba(212,175,106,0.10), transparent 70%); }
        .brief .b-head { font-family: var(--font-body); font-weight: 600; font-size: 1.02em; color: var(--ink); line-height: 1.4; }
        .brief .b-read { color: var(--ink-2); font-size: 0.86em; line-height: 1.5; margin-top: 6px; }
        .brief ol { margin: 8px 0 0 18px; color: var(--ink-2); font-size: 0.84em; line-height: 1.5; }
        .brief ol li { margin: 3px 0; }
        .brief .b-model { font-family: var(--font-mono); font-size: 0.66em; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3); margin-top: 8px; }
        .align-row { display: grid; grid-template-columns: minmax(120px, 1fr) 120px 52px; gap: 10px; align-items: center; padding: 4px 0; font-size: 0.82em; }
        .align-row .a-name { color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .align-row .a-track { height: 5px; border-radius: 3px; background: rgba(212,175,106,0.14); overflow: hidden; }
        .align-row .a-fill { height: 100%; border-radius: 3px; background: var(--gold); }
        .align-row.gap .a-fill { background: var(--ink-3); }
        .align-row .a-val { text-align: right; font-family: var(--font-mono); color: var(--ink-2); font-size: 0.9em; }
        .fit-method { font-family: var(--font-mono); font-size: 0.64em; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3); margin-top: 4px; }
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
            <span class="auth-area" id="auth-area"></span>
        </div>

        <div class="modal-overlay" id="auth-modal">
            <div class="auth-modal-card">
                <button class="modal-close" onclick="closeAuthModal()">&times;</button>
                <h2 id="auth-modal-title">Sign in</h2>
                <p class="auth-modal-sub">Your saved research and blueprints follow your account.</p>
                <input type="text" id="auth-name" class="auth-field" placeholder="Display name" style="display:none;" autocomplete="name">
                <input type="text" id="auth-email" class="auth-field" placeholder="Email" autocomplete="email">
                <input type="password" id="auth-password" class="auth-field" placeholder="Password (8+ characters)" autocomplete="current-password" onkeypress="if(event.key==='Enter')submitAuth()">
                <div class="auth-error" id="auth-error"></div>
                <button class="btn-primary" style="width:100%;" id="auth-submit" onclick="submitAuth()">Sign in</button>
                <div class="auth-switch" id="auth-switch">New here? <a onclick="toggleAuthMode()">Create an account</a></div>
            </div>
        </div>

        <div class="funnel" id="funnel">
            <div class="step" onclick="showTab('overview')"><div class="n">1 · See</div><div class="t">Signal Room</div><div class="d">where demand is moving</div></div>
            <div class="step" onclick="showTab('trends')"><div class="n">2 · Pick</div><div class="t">Trends &amp; cities</div><div class="d">save the signals that matter</div></div>
            <div class="step" onclick="showTab('demandscan')"><div class="n">3 · Test</div><div class="t">Scan a property</div><div class="d">fit, gaps, white space</div></div>
            <div class="step" onclick="window.location.href='/api/monitoring/build-a-brand'"><div class="n">4 · Make</div><div class="t">Build the brand</div><div class="d">blueprint, then renders</div></div>
            <div class="picks"><div class="k">Your picks</div><div class="v" id="funnel-picks">nothing saved yet</div><a href="/api/monitoring/build-a-brand" id="funnel-build">Build from picks &rarr;</a></div>
        </div>
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">Signal Room</button>
            <button class="tab" onclick="showTab('trends')">Trends</button>
            <button class="tab" onclick="showTab('citydesires')">Cities</button>
            <button class="tab" onclick="showTab('moves')">Market Moves</button>
            <button class="tab" onclick="showTab('demandscan')">Demand Scan</button>
            <button class="tab" onclick="showTab('ledger')">Signal Ledger</button>
            <span class="tab-group-label">Data &amp; tools</span>
            <button class="tab secondary" onclick="showTab('chat')">Chat</button>
            <button class="tab secondary" onclick="showTab('content')">Content</button>
            <button class="tab secondary" onclick="showTab('scrapers')">Sources</button>
            <button class="tab secondary" onclick="showTab('projects')" id="projects-tab">My Projects</button>
        </div>

        <div id="overview" class="section active">
            <div class="card">
                <div class="card-head">
                    <div>
                        <h2>Signal Room</h2>
                        <div class="card-sub" id="room-sub">What the platform is watching right now, and what changed in the last seven days.</div>
                    </div>
                </div>
                <div class="stat-row" id="stat-row">
                    <div class="stat"><div class="stat-label">Corpus</div><div class="stat-value">&ndash;</div><div class="stat-foot"><div class="stat-delta">loading</div></div></div>
                    <div class="stat"><div class="stat-label">Sources live</div><div class="stat-value">&ndash;</div><div class="stat-foot"><div class="stat-delta">loading</div></div></div>
                    <div class="stat"><div class="stat-label">Trends tracked</div><div class="stat-value">&ndash;</div><div class="stat-foot"><div class="stat-delta">loading</div></div></div>
                    <div class="stat"><div class="stat-label">Predictions staked</div><div class="stat-value">&ndash;</div><div class="stat-foot"><div class="stat-delta">loading</div></div></div>
                </div>
            </div>

            <div class="room-grid">
                <div class="card span-7">
                    <div class="card-head">
                        <div>
                            <h2>Opportunity Map</h2>
                            <div class="card-sub">Every demand cluster the platform tracks. Right is stronger demand; up is less supply answering it. Bubble size is source volume. The upper-right is where a concept should be built; click any bubble to open it or build from it.</div>
                        </div>
                        <div class="card-tools"><button class="tool-btn" id="omap-table-btn" onclick="toggleFigTable('omap')">Table</button></div>
                    </div>
                    <div class="fig-legend" id="omap-legend"></div>
                    <div class="fig-wrap" id="omap"></div>
                    <div id="omap-table" hidden></div>
                    <div class="fig-foot" id="omap-foot"></div>
                </div>
                <div class="card span-5">
                    <div class="card-head">
                        <div>
                            <h2>City Matrix</h2>
                            <div class="card-sub">Where attention is moving against how much supply already exists. Right is rising attention this week (Wikipedia); up is more hotels on the map (OpenStreetMap). Bubble size is Airbnb listings. Lower-right is thin supply meeting rising interest.</div>
                        </div>
                        <div class="card-tools"><button class="tool-btn" id="cmat-table-btn" onclick="toggleFigTable('cmat')">Table</button></div>
                    </div>
                    <div class="fig-legend" id="cmat-legend"></div>
                    <div class="fig-wrap" id="cmat"></div>
                    <div id="cmat-table" hidden></div>
                    <div class="fig-foot" id="cmat-foot"></div>
                </div>
            </div>

            <div class="room-grid">
                <div class="card span-12">
                    <div class="card-head">
                        <div>
                            <h2>Demand Curves</h2>
                            <div class="card-sub" id="curves-sub">Daily destination attention across tracked cities, each indexed to its own 30-day average (100). The three strongest week-over-week risers are highlighted; every other city is the grey field behind them.</div>
                        </div>
                        <div class="card-tools">
                            <select id="curves-metric" class="filter-select" onchange="setCurvesMetric(this.value)" title="Demand metric">
                                <option value="wikipedia_pageviews">Attention · Wikipedia pageviews (daily)</option>
                                <option value="eurostat_nights_spent">Nights spent · Eurostat (monthly, by country)</option>
                                <option value="airbnb_reviews_per_month">Airbnb review velocity (quarterly)</option>
                                <option value="airbnb_median_price">Airbnb median price (quarterly)</option>
                                <option value="osm_hotels">Hotel supply · OpenStreetMap</option>
                            </select>
                            <button class="tool-btn active" id="curves-chart-btn" onclick="setCurvesView('chart')">Chart</button>
                            <button class="tool-btn" id="curves-table-btn" onclick="setCurvesView('table')">Table</button>
                        </div>
                    </div>
                    <div id="curves-legend" class="chart-legend"></div>
                    <div class="chart-wrap" id="curves-chart">
                        <div class="empty"><div class="icon"></div>Loading demand series&hellip;</div>
                    </div>
                    <div id="curves-table" hidden></div>
                    <div class="movers" id="movers"></div>
                </div>
            </div>

            <div class="room-grid">
                <div class="card span-7">
                    <div class="card-head">
                        <div>
                            <h2>Where Capital Is Moving</h2>
                            <div class="card-sub">Operator moves per week, grouped by kind. Deals are acquisitions, expansions, reflags and partnerships; product is launches, concepts, renovations and repositionings. Filings count alongside press.</div>
                        </div>
                        <div class="card-tools"><button class="tool-btn" id="mvw-table-btn" onclick="toggleFigTable('mvw')">Table</button></div>
                    </div>
                    <div class="fig-legend" id="mvw-legend"></div>
                    <div class="fig-wrap" id="mvw"></div>
                    <div id="mvw-table" hidden></div>
                    <div class="fig-foot" id="mvw-foot"></div>
                </div>
                <div class="card span-5">
                    <div class="card-head">
                        <div>
                            <h2>Operator Bets</h2>
                            <div class="card-sub">The latest strategic moves extracted from trade press. Where capital is already going.</div>
                        </div>
                    </div>
                    <div class="signal-list" id="room-moves"><div class="empty"><div class="icon"></div>Loading&hellip;</div></div>
                </div>
            </div>

            <div class="room-grid">
                <div class="card span-7">
                    <div class="card-head">
                        <div>
                            <h2>Trend Movers</h2>
                            <div class="card-sub">Most recently strengthened demand clusters. Strength is cluster cohesion and volume; white space is how little supply answers it.</div>
                        </div>
                    </div>
                    <div class="signal-list" id="room-trends"><div class="empty"><div class="icon"></div>Loading&hellip;</div></div>
                </div>
                <div class="card span-5">
                    <div class="card-head">
                        <div>
                            <h2>Attention Model</h2>
                            <div class="card-sub">BrandClave decides what to read next with active inference: each source carries a belief about how productive it is, and the scheduler picks the action that minimises expected free energy, trading exploitation of known-good sources against exploring uncertain ones.</div>
                        </div>
                    </div>
                    <div id="room-ai"><div class="empty"><div class="icon"></div>Loading beliefs&hellip;</div></div>
                </div>
            </div>

            <div class="room-grid">
                <div class="card span-12">
                    <div class="card-head">
                        <div>
                            <h2>Coverage</h2>
                            <div class="card-sub">Every registered source and when it last delivered. Fresh means within 24 hours.</div>
                        </div>
                    </div>
                    <div class="coverage" id="room-coverage"></div>
                    <div class="coverage-legend"><span class="l-live">Fresh</span><span class="l-stale">Ageing</span><span class="l-silent">Silent</span><span class="l-blocked">Blocked</span></div>
                    <div id="room-registry" class="ai-explain"></div>
                </div>
            </div>
        </div>

        <div id="ledger" class="section">
            <div class="card">
                <div class="card-head">
                    <div>
                        <h2>Signal Ledger</h2>
                        <div class="card-sub">BrandClave stakes its forecasts before outcomes are known, so accuracy can be audited rather than claimed.</div>
                    </div>
                    <div class="card-tools"><button class="tool-btn" onclick="loadLedger(true)">Refresh</button></div>
                </div>
                <div class="ledger-how">
                    Every record is <b>sealed</b> with a SHA-256 hash of its content at the moment it is written and can never be edited. Evidence and outcomes are <b>appended</b> as events, each timestamped. When a forecast horizon passes, the realised value is scored against the sealed range, and the hit rate, error and calibration below are computed from those scores only. An empty hit rate means no horizon has been reached yet; it is not filled in by hand.
                </div>
                <div class="ledger-kpis" id="ledger-kpis"></div>
                <div id="ledger-list"><div class="empty"><div class="icon"></div>Loading ledger&hellip;</div></div>
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
                <h2>Demand Trends</h2>
                <p style="color:var(--ink-2);margin-bottom:15px;">Demand clusters found across consumer conversation (Bluesky, Mastodon, YouTube), trade press, the global news index and culture feeds. Strength is cluster cohesion and volume; white space is how little supply answers it. Save the ones that matter, then build from them.</p>
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
                <div class="card-head">
                    <div>
                        <h2>Who Is Moving</h2>
                        <div class="card-sub">Operators, REITs and platforms ranked by moves extracted in the window, with the mix of what they are doing. Filings are read from the SEC directly; the rest from trade press and the global news index.</div>
                    </div>
                </div>
                <div class="fig-legend" id="company-legend"></div>
                <div id="company-league"></div>
            </div>
            <div class="card">
                <h2>Market Moves</h2>
                <p style="color:var(--ink-2);margin-bottom:15px;">Every strategic move on file: launches, acquisitions, renovations, repositionings, partnerships and technology bets, from trade press, the global news index and SEC filings.</p>
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

        function setTextIf(id, text) { var el = document.getElementById(id); if (el) el.textContent = text; }
        function setHtmlIf(id, html) { var el = document.getElementById(id); if (el) el.innerHTML = html; }

        function setStatus(icon, text) {
            document.getElementById('status-icon').textContent = icon;
            document.getElementById('status-text').textContent = text;
        }

        function truncate(str, len) {
            if (!str) return '';
            return str.length > len ? str.substring(0, len) + '...' : str;
        }

        // =============================================
        // Account & saved-research sync
        // localStorage stays the fast render cache; the API is the durable
        // store when signed in. Existing anonymous saves migrate up on login.
        // =============================================
        var AUTH_TOKEN_KEY = 'brandclave_token';
        var AUTH_USER_KEY = 'brandclave_user';
        var authRegisterMode = false;
        var serverSavedMap = {}; // "type:itemId" -> server saved-item id

        function getAuthToken() { try { return localStorage.getItem(AUTH_TOKEN_KEY); } catch (e) { return null; } }
        function getAuthUser() { try { return JSON.parse(localStorage.getItem(AUTH_USER_KEY) || 'null'); } catch (e) { return null; } }
        function authHeaders(extra) { var h = extra || {}; var t = getAuthToken(); if (t) { h['Authorization'] = 'Bearer ' + t; } return h; }

        function renderAuthArea() {
            var el = document.getElementById('auth-area');
            if (!el) return;
            var user = getAuthUser();
            if (getAuthToken() && user) {
                el.innerHTML = '<span class="auth-chip">' + escapeHtml(user.display_name || user.email) + '</span>' +
                    '<button onclick="signOut()">Sign out</button>';
            } else {
                el.innerHTML = '<button onclick="openAuthModal()">Sign in</button>';
            }
        }

        function openAuthModal() {
            document.getElementById('auth-error').textContent = '';
            document.getElementById('auth-modal').classList.add('active');
            document.getElementById('auth-email').focus();
        }
        function closeAuthModal() { document.getElementById('auth-modal').classList.remove('active'); }
        function toggleAuthMode() {
            authRegisterMode = !authRegisterMode;
            document.getElementById('auth-name').style.display = authRegisterMode ? 'block' : 'none';
            document.getElementById('auth-modal-title').textContent = authRegisterMode ? 'Create your account' : 'Sign in';
            document.getElementById('auth-submit').textContent = authRegisterMode ? 'Create account' : 'Sign in';
            document.getElementById('auth-switch').innerHTML = authRegisterMode
                ? 'Already have an account? <a onclick="toggleAuthMode()">Sign in</a>'
                : 'New here? <a onclick="toggleAuthMode()">Create an account</a>';
        }

        async function submitAuth() {
            var email = document.getElementById('auth-email').value.trim();
            var password = document.getElementById('auth-password').value;
            var errEl = document.getElementById('auth-error');
            errEl.textContent = '';
            if (!email || !password) { errEl.textContent = 'Email and password are required.'; return; }

            var payload = { email: email, password: password };
            if (authRegisterMode) {
                var name = document.getElementById('auth-name').value.trim();
                if (name) payload.display_name = name;
            }
            try {
                var res = await fetch(authRegisterMode ? '/api/auth/register' : '/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                var data = await res.json();
                if (!res.ok) {
                    var detail = data.detail;
                    if (Array.isArray(detail)) detail = detail[0] && detail[0].msg;
                    errEl.textContent = detail || 'That did not work. Check your details.';
                    return;
                }
                localStorage.setItem(AUTH_TOKEN_KEY, data.token);
                localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user));
                document.getElementById('auth-password').value = '';
                closeAuthModal();
                renderAuthArea();
                await syncSavedItems();
                updateSavedCount();
                if (typeof renderMyProjects === 'function') renderMyProjects();
            } catch (e) {
                errEl.textContent = 'Could not reach the server.';
            }
        }

        function signOut() {
            localStorage.removeItem(AUTH_TOKEN_KEY);
            localStorage.removeItem(AUTH_USER_KEY);
            serverSavedMap = {};
            renderAuthArea();
        }

        async function syncSavedItems() {
            if (!getAuthToken()) return;
            try {
                // Push local anonymous saves up (server 409s duplicates, which is fine)
                var toPush = [];
                var trends = getSavedProjects();
                for (var i = 0; i < trends.length; i++) toPush.push({ item_type: 'trend', item_id: trends[i].id, title: trends[i].name || 'Untitled trend', snapshot: trends[i] });
                var moves = getSavedMoves();
                for (var j = 0; j < moves.length; j++) toPush.push({ item_type: 'move', item_id: moves[j].id, title: moves[j].title || 'Untitled move', snapshot: moves[j] });
                for (var k = 0; k < toPush.length; k++) {
                    await fetch('/api/projects/saved', {
                        method: 'POST',
                        headers: authHeaders({ 'Content-Type': 'application/json' }),
                        body: JSON.stringify(toPush[k])
                    }).catch(function () {});
                }

                // Pull the durable copy down and make it the local cache
                var res = await fetch('/api/projects/saved', { headers: authHeaders() });
                if (!res.ok) { if (res.status === 401) signOut(); return; }
                var data = await res.json();
                var items = data.items || [];
                serverSavedMap = {};
                var newTrends = [], newMoves = [];
                for (var m = 0; m < items.length; m++) {
                    var it = items[m];
                    serverSavedMap[it.item_type + ':' + it.item_id] = it.id;
                    var snap = it.snapshot || {};
                    snap.id = it.item_id;
                    if (it.item_type === 'trend') newTrends.push(snap);
                    else if (it.item_type === 'move') newMoves.push(snap);
                }
                localStorage.setItem(STORAGE_KEY, JSON.stringify(newTrends));
                localStorage.setItem(MOVES_STORAGE_KEY, JSON.stringify(newMoves));
            } catch (e) {
                console.error('Saved-items sync failed:', e);
            }
        }

        function pushSavedItem(itemType, item, title) {
            if (!getAuthToken() || !item || !item.id) return;
            fetch('/api/projects/saved', {
                method: 'POST',
                headers: authHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify({ item_type: itemType, item_id: item.id, title: title, snapshot: item })
            }).then(function (res) { return res.ok ? res.json() : null; })
              .then(function (data) { if (data && data.id) serverSavedMap[itemType + ':' + item.id] = data.id; })
              .catch(function () {});
        }

        function deleteSavedItemRemote(itemType, itemId) {
            if (!getAuthToken()) return;
            var sid = serverSavedMap[itemType + ':' + itemId];
            if (!sid) return;
            delete serverSavedMap[itemType + ':' + itemId];
            fetch('/api/projects/saved/' + sid, { method: 'DELETE', headers: authHeaders() }).catch(function () {});
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
                // Headline tiles now live in the Signal Room (loadOverview); the legacy
                // metric ids are optional so older markup keeps working.
                setTextIf('m-content', metrics.total_content ? metrics.total_content.toLocaleString() : '0');
                setTextIf('m-processed', metrics.processed_content ? metrics.processed_content.toLocaleString() : '0');
                setTextIf('m-trends', metrics.trends_count || '0');
                setTextIf('m-moves', metrics.moves_count || '0');

                // Render trends
                allTrends = trendsData.trends || [];
                var trends = allTrends;
                if (trends.length > 0) {
                    setHtmlIf('latest-trend', renderTrend(trends[0], 0));
                    var trendsHtml = '';
                    for (var ti = 0; ti < trends.length; ti++) { trendsHtml += renderTrend(trends[ti], ti); }
                    document.getElementById('trends-list').innerHTML = trendsHtml;
                } else {
                    setHtmlIf('latest-trend', '<div class="empty"><div class="icon">📈</div>No trends yet. Run POPULATE_DATA.bat</div>');
                    document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends yet</div>';
                }

                // Render moves
                allMoves = movesData.moves || [];
                var moves = allMoves;
                if (moves.length > 0) {
                    setHtmlIf('latest-move', renderMove(moves[0], 0));
                    var movesHtml = '';
                    for (var mi = 0; mi < moves.length; mi++) { movesHtml += renderMove(moves[mi], mi); }
                    document.getElementById('moves-list').innerHTML = movesHtml;
                } else {
                    setHtmlIf('latest-move', '<div class="empty"><div class="icon">♟️</div>No moves yet. Run POPULATE_DATA.bat</div>');
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
                setHtmlIf('latest-trend', '<div class="error">Failed to load data: ' + err.message + '</div>');
                setHtmlIf('latest-move', '<div class="error">Failed to load data: ' + err.message + '</div>');
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

            // Executive brief (LLM read of the measured evidence)
            var briefHtml = '';
            var b = p.demand_brief;
            if (b && (b.headline || b.read)) {
                var movesHtml = (b.moves || []).map(function (m) { return '<li>' + esc(m) + '</li>'; }).join('');
                briefHtml = '<div class="brief"><div class="b-head">' + esc(b.headline || '') + '</div>' +
                    (b.read ? '<div class="b-read">' + esc(b.read) + '</div>' : '') +
                    (movesHtml ? '<ol>' + movesHtml + '</ol>' : '') +
                    '<div class="b-model">Brief by ' + esc(b.model || 'llm') + ' from measured alignment only</div></div>';
            }

            // Trend alignment: the semantic evidence behind the score
            var alignHtml = '';
            var al = (p.trend_alignment || []).slice(0, 6);
            if (al.length) {
                alignHtml = '<div class="property-section"><div class="property-section-title">Demand Alignment</div>' + al.map(function (a) {
                    var pct = Math.round((a.similarity || 0) * 100);
                    var w = Math.max(2, Math.round(((a.similarity - 0.6) / 0.28) * 100));
                    var cls = a.similarity >= 0.77 ? '' : ' gap';
                    return '<div class="align-row' + cls + '" title="demand strength ' + Math.round((a.strength_score || 0) * 100) + '%, white space ' + Math.round((a.white_space_score || 0) * 100) + '%"><span class="a-name">' + esc(a.name) + '</span><div class="a-track"><div class="a-fill" style="width:' + Math.min(100, w) + '%"></div></div><span class="a-val">' + pct + '%</span></div>';
                }).join('') + '<div class="fit-method">' + (p.fit_method === 'embedding' ? 'semantic match, mistral-embed' : 'keyword match') + '</div></div>';
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
                briefHtml +
                flagsHtml +
                alignHtml +
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

            var snapshot = {
                id: trend.id,
                name: trend.name || trend.trend_name,
                description: trend.description,
                white_space_score: trend.white_space_score,
                strength_score: trend.strength_score,
                region: trend.region,
                audience_segment: trend.audience_segment,
                topics: trend.topics,
                saved_at: new Date().toISOString()
            };
            saved.push(snapshot);

            localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
            pushSavedItem('trend', snapshot, snapshot.name || 'Untitled trend');
            return true;
        }

        function removeProject(trendId) {
            var saved = getSavedProjects();
            var filtered = [];
            for (var i = 0; i < saved.length; i++) { if (saved[i].id !== trendId) filtered.push(saved[i]); }
            localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
            deleteSavedItemRemote('trend', trendId);
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

            var snapshot = {
                id: move.id,
                title: move.title,
                summary: move.summary,
                company: move.company,
                move_type: move.move_type,
                market: move.market,
                strategic_implications: move.strategic_implications,
                source_name: move.source_name,
                saved_at: new Date().toISOString()
            };
            saved.push(snapshot);

            localStorage.setItem(MOVES_STORAGE_KEY, JSON.stringify(saved));
            pushSavedItem('move', snapshot, snapshot.title || 'Untitled move');
            return true;
        }

        function removeMove(moveId) {
            var saved = getSavedMoves();
            var filtered = [];
            for (var i = 0; i < saved.length; i++) { if (saved[i].id !== moveId) filtered.push(saved[i]); }
            localStorage.setItem(MOVES_STORAGE_KEY, JSON.stringify(filtered));
            deleteSavedItemRemote('move', moveId);
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
            // Clear the durable copies too when signed in
            for (var key in serverSavedMap) {
                fetch('/api/projects/saved/' + serverSavedMap[key], { method: 'DELETE', headers: authHeaders() }).catch(function () {});
            }
            serverSavedMap = {};
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
                var response = await fetch('/api/brand-blueprint?limit=20', { headers: authHeaders() });
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
        renderAuthArea();
        syncSavedItems().then(function () {
            updateSavedCount();
            updateMovesSavedCount();
            renderMyProjects();
        });

        // Auto-refresh every 60 seconds
        setInterval(loadAllData, 60000);

        // =============================================
        // Signal Room (overview) + Signal Ledger
        // Inline SVG, no chart library. Categorical slots are fixed:
        // gold, violet, teal; everything else is the grey field.
        // =============================================
        var SERIES_COLORS = ['#d4af6a', '#8b7ce0', '#3aa88d'];
        // Copy per demand metric: what the series is, its cadence, and what a "mover" means.
        var METRIC_COPY = {
            wikipedia_pageviews: { what: 'Daily destination attention (Wikipedia pageviews)', unit: 'cities', base: 'its own 30-day average', movers: 'week-over-week', period: 'this week' },
            eurostat_nights_spent: { what: 'Monthly nights spent at hotels and similar accommodation (Eurostat tour_occ_nim)', unit: 'countries', base: 'its own multi-year average', movers: 'month-over-month (seasonal by nature)', period: 'latest month' },
            airbnb_reviews_per_month: { what: 'Quarterly Airbnb review velocity, the standard short-term-rental occupancy proxy (Inside Airbnb, CC BY 4.0)', unit: 'cities', base: 'its own average', movers: 'snapshot-over-snapshot', period: 'latest snapshot' },
            airbnb_median_price: { what: 'Quarterly Airbnb median nightly price in each city’s local currency (Inside Airbnb, CC BY 4.0) — compare a city with itself over time, not cities with each other', unit: 'cities', base: 'its own average', movers: 'snapshot-over-snapshot', period: 'latest snapshot' },
            osm_hotels: { what: 'Hotel, hostel and guest-house supply inside each city boundary (OpenStreetMap)', unit: 'cities', base: 'its own average', movers: 'run-over-run', period: 'latest run' }
        };
        function metricCopy(metric) { return METRIC_COPY[metric] || { what: metric.replace(/_/g, ' '), unit: 'series', base: 'its own average', movers: 'latest vs previous', period: 'latest' }; }
        var roomData = null;
        var curvesView = 'chart';

        function esc(s) {
            return String(s == null ? '' : s)
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }
        function fmtInt(n) { return (n || 0).toLocaleString(); }
        // LLM output occasionally leaks markdown emphasis into stored summaries.
        function stripMd(s) { return String(s || '').replace(/\*\*/g, '').replace(/^#+\s*/gm, '').replace(/^\s*[-*]\s+/gm, ''); }
        function fmtPct(p, digits) {
            if (p == null || isNaN(p)) return '&ndash;';
            var v = p * 100;
            return (v > 0 ? '+' : '') + v.toFixed(digits == null ? 0 : digits) + '%';
        }
        // The API emits naive ISO timestamps that are UTC; without a zone suffix
        // the browser would read them as local time and shift them by the offset.
        function parseUtc(iso) {
            if (iso instanceof Date) return iso;
            var s = String(iso);
            return new Date(/[Zz]$|[+-][0-9]{2}:?[0-9]{2}$/.test(s) ? s : s + 'Z');
        }
        function fmtDate(iso) {
            if (!iso) return '';
            return parseUtc(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        }
        function fmtDateYear(iso) {
            if (!iso) return '';
            return parseUtc(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
        }
        function hoursAgo(h) {
            if (h == null) return 'never';
            if (h < 1) return Math.max(1, Math.round(h * 60)) + 'm';
            if (h < 48) return Math.round(h) + 'h';
            return Math.round(h / 24) + 'd';
        }

        function sparkline(series, width, height) {
            var vals = series.map(function (p) { return p.count; });
            var max = Math.max.apply(null, vals.concat([1]));
            var n = vals.length;
            var pts = vals.map(function (v, i) {
                var x = (i / (n - 1)) * width;
                var y = height - (v / max) * (height - 4) - 2;
                return [x, y];
            });
            var path = pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ');
            var last = pts[pts.length - 1];
            return '<svg class="spark" viewBox="0 0 ' + width + ' ' + height + '" aria-hidden="true">' +
                '<path d="' + path + '" fill="none" stroke="#857a68" stroke-width="1.5" stroke-linejoin="round"/>' +
                '<circle cx="' + last[0].toFixed(1) + '" cy="' + last[1].toFixed(1) + '" r="3.5" fill="#d4af6a" stroke="#201a12" stroke-width="2"/>' +
                '</svg>';
        }

        function statTile(label, value, deltaHtml, sparkHtml) {
            return '<div class="stat"><div class="stat-label">' + label + '</div>' +
                '<div class="stat-value">' + value + '</div>' +
                '<div class="stat-foot"><div class="stat-delta">' + deltaHtml + '</div>' + (sparkHtml || '') + '</div></div>';
        }

        function renderStats(d) {
            var k = d.kpis;
            var fresh = d.sources.freshness.filter(function (s) { return s.hours_since != null && s.hours_since <= 24; }).length;
            var active = d.sources.freshness.filter(function (s) { return s.status === 'active'; }).length;
            var contentDelta = k.content.last_7d > 0
                ? '<span class="up">+' + fmtInt(k.content.last_7d) + '</span> <span class="vs">last 7 days</span>'
                : '<span class="vs">no intake in 7 days</span>';
            var trendDelta = k.trends.last_7d > 0
                ? '<span class="up">+' + fmtInt(k.trends.last_7d) + '</span> <span class="vs">refreshed this week</span>'
                : '<span class="vs">' + fmtInt(k.trends.total) + ' clusters on file</span>';
            var L = d.ledger;
            var ledgerDelta = L.resolved_predictions > 0
                ? '<span class="up">' + Math.round((L.hit_rate || 0) * 100) + '% hit rate</span> <span class="vs">' + L.resolved_predictions + ' resolved</span>'
                : '<span class="vs">' + L.open_predictions + ' open &middot; first horizon pending</span>';

            document.getElementById('stat-row').innerHTML =
                statTile('Corpus', fmtInt(k.content.total), contentDelta + (k.content.archived ? ' <span class="vs">&middot; ' + fmtInt(k.content.archived) + ' archived (retired sources)</span>' : ''), sparkline(d.intake, 72, 30)) +
                statTile('Sources fresh · 24h', fresh + '<small style="font-size:0.45em;color:var(--ink-3);-webkit-text-fill-color:var(--ink-3);margin-left:6px;">of ' + active + ' active</small>',
                    '<span class="vs">' + (d.sources.registry.planned || 0) + ' planned &middot; ' + (d.sources.registry.blocked || 0) + ' blocked by ToS</span>') +
                statTile('Trends tracked', fmtInt(k.trends.total), trendDelta) +
                statTile('Predictions staked', fmtInt(L.total_predictions), ledgerDelta);
            document.getElementById('room-sub').textContent = 'Snapshot ' + parseUtc(d.generated_at).toLocaleString() + ' · ' + fmtInt(k.moves.total) + ' operator moves on file';
        }

        // ---------- Demand curves ----------
        function renderCurves(demand) {
            var wrap = document.getElementById('curves-chart');
            var legend = document.getElementById('curves-legend');
            var cities = demand.cities || [];
            if (!cities.length) {
                wrap.innerHTML = '<div class="empty"><div class="icon"></div>No demand series yet. The Wikimedia pageviews source fills this in.</div>';
                legend.innerHTML = '';
                return;
            }
            var highlight = demand.movers_up.slice(0, 3);
            var byName = {};
            cities.forEach(function (c) { byName[c.city] = c; });
            var dates = cities[0].series.map(function (p) { return p.date; });
            var n = dates.length;

            if (demand.snapshot || n < 2) {
                // One observation per city: a curve has nothing to draw, so the
                // league table is the chart. Movers become "highest / lowest".
                var mcS = metricCopy(demand.metric);
                legend.innerHTML = '';
                wrap.innerHTML = '<div class="empty"><div class="icon"></div>' + esc(mcS.what) + ': one snapshot per city so far (' + esc(cities[0].latest_date) + '). Levels are in the table; a curve appears once the next snapshot lands.</div>';
                var rowsS = cities.slice().sort(function (a, b) { return b.recent_7d_avg - a.recent_7d_avg; }).map(function (c) {
                    return '<tr><td>' + esc(c.city) + '</td><td>' + esc(c.country || '') + '</td><td class="num">' + esc(c.latest_date) + '</td><td class="num">' + (c.recent_7d_avg >= 100 ? fmtInt(c.recent_7d_avg) : c.recent_7d_avg) + '</td></tr>';
                }).join('');
                document.getElementById('curves-table').innerHTML = '<table class="chart-table"><thead><tr><th>City</th><th>Country</th><th style="text-align:right">Snapshot</th><th style="text-align:right">Value</th></tr></thead><tbody>' + rowsS + '</tbody></table>';
                var maxV = Math.max.apply(null, cities.map(function (c) { return c.recent_7d_avg; }).concat([0.01]));
                var lv = function (list) {
                    return list.map(function (name) { var c = byName[name]; if (!c) return ''; var w = Math.max(3, Math.round((c.recent_7d_avg / maxV) * 100)); return '<div class="mover up"><span class="name">' + esc(c.city) + '</span><div><div class="bar" style="width:' + w + '%"></div></div><span class="pct">' + (c.recent_7d_avg >= 100 ? fmtInt(c.recent_7d_avg) : c.recent_7d_avg) + '</span></div>'; }).join('');
                };
                document.getElementById('movers').innerHTML = '<div><h4>Highest</h4>' + lv(demand.movers_up) + '</div><div><h4>Lowest</h4>' + lv(demand.movers_down) + '</div>';
                document.getElementById('curves-sub').textContent = mcS.what + ' across ' + cities.length + ' ' + mcS.unit + '. Single snapshot per city: shown as levels, ranked.';
                setCurvesView('table');
                return;
            }

            var W = 960, H = 320, padL = 44, padR = 150, padT = 14, padB = 30;
            var plotW = W - padL - padR, plotH = H - padT - padB;
            var allIdx = [];
            cities.forEach(function (c) { c.series.forEach(function (p) { if (p.index != null) allIdx.push(p.index); }); });
            // Domain from the 2nd-98th percentile of all points: a single viral day in
            // one city would otherwise flatten every other curve into the baseline.
            // Clipped spikes run off the top; the subtitle says so.
            var sortedIdx = allIdx.slice().sort(function (a, b) { return a - b; });
            var pct = function (q) { return sortedIdx[Math.min(sortedIdx.length - 1, Math.floor(q * sortedIdx.length))]; };
            var hiIdx = Math.max.apply(null, highlight.map(function (nm) { var c = byName[nm]; return c ? Math.max.apply(null, c.series.map(function (p) { return p.index; })) : 0; }));
            var lo = Math.floor(Math.min(pct(0.02), 85) / 10) * 10;
            var hi = Math.ceil(Math.max(pct(0.98), hiIdx, 120) / 10) * 10;
            var clipped = sortedIdx[sortedIdx.length - 1] > hi;
            var step = hi - lo > 160 ? 50 : hi - lo > 80 ? 25 : 10;
            var x = function (i) { return padL + (i / (n - 1)) * plotW; };
            var y = function (v) { return padT + plotH - ((v - lo) / (hi - lo)) * plotH; };

            var svg = '<svg class="curves" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Indexed destination attention, last 30 days">';
            // grid + axis
            svg += '<g class="grid">';
            var ticks = [];
            for (var t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
            ticks.forEach(function (t) { svg += '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + y(t) + '" y2="' + y(t) + '"/>'; });
            svg += '</g><g class="axis">';
            ticks.forEach(function (t) { svg += '<text x="' + (padL - 8) + '" y="' + (y(t) + 4) + '" text-anchor="end">' + t + '</text>'; });
            svg += '<line class="baseline" x1="' + padL + '" x2="' + (W - padR) + '" y1="' + y(100) + '" y2="' + y(100) + '"/>';
            svg += '<text x="' + (padL + 6) + '" y="' + (y(100) - 5) + '" style="fill:rgba(212,175,106,0.7)">series average = 100</text>';
            [0, Math.floor(n / 2), n - 1].forEach(function (i) {
                svg += '<text x="' + x(i) + '" y="' + (H - 8) + '" text-anchor="' + (i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle') + '">' + fmtDate(dates[i]) + '</text>';
            });
            svg += '</g>';
            // grey field first, highlighted on top
            var pathFor = function (c) {
                return c.series.map(function (p, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(p.index).toFixed(1); }).join(' ');
            };
            cities.forEach(function (c) {
                if (highlight.indexOf(c.city) !== -1) return;
                svg += '<path class="series rest" d="' + pathFor(c) + '"/>';
            });
            var labelYs = [];
            highlight.forEach(function (name, k) {
                var c = byName[name]; if (!c) return;
                var color = SERIES_COLORS[k];
                svg += '<path class="series" stroke="' + color + '" d="' + pathFor(c) + '"/>';
                var last = c.series[c.series.length - 1];
                var ly = y(last.index);
                // nudge colliding end-labels apart without detaching them far from the line
                labelYs.forEach(function (prev) { if (Math.abs(prev - ly) < 16) ly = prev + 16; });
                labelYs.push(ly);
                svg += '<circle class="end-dot" cx="' + x(n - 1) + '" cy="' + y(last.index) + '" r="4.5" fill="' + color + '"/>';
                svg += '<text class="end-label" x="' + (x(n - 1) + 12) + '" y="' + (ly + 4) + '">' + esc(c.city) + ' <tspan class="pct">' + fmtPct(c.change_pct) + '</tspan></text>';
            });
            svg += '<line id="curves-crosshair" class="crosshair" x1="0" x2="0" y1="' + padT + '" y2="' + (padT + plotH) + '" visibility="hidden"/>';
            svg += '<rect class="hit" x="' + padL + '" y="' + padT + '" width="' + plotW + '" height="' + plotH + '"/>';
            svg += '</svg><div class="chart-tip" id="curves-tip" hidden></div>';
            wrap.innerHTML = svg;

            legend.innerHTML = highlight.map(function (name, k) {
                return '<span class="key"><i style="background:' + SERIES_COLORS[k] + '"></i>' + esc(name) + '</span>';
            }).join('') + '<span class="key rest"><i></i>' + (cities.length - highlight.length) + ' other cities</span>';

            // crosshair + tooltip: pointer finds the X, readout lists every highlighted series
            var svgEl = wrap.querySelector('svg');
            var hit = wrap.querySelector('.hit');
            var cross = wrap.querySelector('#curves-crosshair');
            var tip = document.getElementById('curves-tip');
            function move(ev) {
                var rect = svgEl.getBoundingClientRect();
                var px = (ev.clientX - rect.left) * (W / rect.width);
                var i = Math.round(((px - padL) / plotW) * (n - 1));
                i = Math.max(0, Math.min(n - 1, i));
                cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
                var rows = highlight.map(function (name, k) {
                    var c = byName[name]; var p = c.series[i];
                    return '<div class="tip-row"><i style="background:' + SERIES_COLORS[k] + '"></i><span>' + esc(name) + '</span><b>' + p.index.toFixed(0) + ' <small style="color:var(--ink-3)">(' + fmtInt(Math.round(p.value)) + ')</small></b></div>';
                }).join('');
                tip.innerHTML = '<div class="tip-date">' + fmtDateYear(dates[i]) + '</div>' + rows;
                tip.hidden = false;
                var leftPx = (x(i) / W) * rect.width;
                tip.style.left = leftPx + 'px';
                tip.style.top = (rect.height * 0.08) + 'px';
            }
            hit.addEventListener('pointermove', move);
            hit.addEventListener('pointerleave', function () { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; });

            // table twin
            var rowsHtml = cities.slice().sort(function (a, b) { return (b.change_pct || 0) - (a.change_pct || 0); }).map(function (c) {
                return '<tr><td>' + esc(c.city) + '</td><td>' + esc(c.country || '') + '</td><td class="num">' + fmtInt(c.prior_7d_avg) + '</td><td class="num">' + fmtInt(c.recent_7d_avg) + '</td><td class="num">' + fmtPct(c.change_pct, 1) + '</td></tr>';
            }).join('');
            document.getElementById('curves-table').innerHTML = '<table class="chart-table"><thead><tr><th>City</th><th>Country</th><th style="text-align:right">Prior 7d avg</th><th style="text-align:right">Last 7d avg</th><th style="text-align:right">Change</th></tr></thead><tbody>' + rowsHtml + '</tbody></table>';

            // movers
            var maxAbs = Math.max.apply(null, cities.map(function (c) { return Math.abs(c.change_pct || 0); }).concat([0.01]));
            var mv = function (list, cls) {
                return list.map(function (name) {
                    var c = byName[name]; if (!c) return '';
                    var w = Math.max(3, Math.round((Math.abs(c.change_pct) / maxAbs) * 100));
                    return '<div class="mover ' + cls + '"><span class="name" title="' + esc(c.city) + '">' + esc(c.city) + '</span><div><div class="bar" style="width:' + w + '%"></div></div><span class="pct">' + fmtPct(c.change_pct) + '</span></div>';
                }).join('');
            };
            var period = metricCopy(demand.metric).period;
            document.getElementById('movers').innerHTML =
                '<div><h4>Rising, ' + period + '</h4>' + mv(demand.movers_up, 'up') + '</div>' +
                '<div><h4>Cooling, ' + period + '</h4>' + mv(demand.movers_down, 'down') + '</div>';
            var mc = metricCopy(demand.metric);
            document.getElementById('curves-sub').textContent = mc.what + ' across ' + cities.length + ' ' + mc.unit + ', each indexed to ' + mc.base + ' (= 100). The three strongest ' + mc.movers + ' risers are highlighted; the grey field is everyone else.' + (clipped ? ' Axis capped at ' + hi + '; spikes above it run off the top.' : '');
        }

        var curvesMetric = 'wikipedia_pageviews';
        async function setCurvesMetric(metric) {
            curvesMetric = metric;
            try {
                var res = await fetch('/api/overview?demand_metric=' + encodeURIComponent(metric));
                var d = await res.json();
                renderCurves(d.demand);
                setCurvesView(curvesView);
            } catch (e) { console.error('metric switch failed', e); }
        }
        function setCurvesView(v) {
            curvesView = v;
            document.getElementById('curves-chart').hidden = v !== 'chart';
            document.getElementById('curves-legend').hidden = v !== 'chart';
            document.getElementById('curves-table').hidden = v !== 'table';
            document.getElementById('curves-chart-btn').classList.toggle('active', v === 'chart');
            document.getElementById('curves-table-btn').classList.toggle('active', v === 'table');
        }

        // ---------- Trends + moves ----------
        function meter(label, value, cls) {
            var pct = Math.round((value || 0) * 100);
            return '<div class="meter ' + (cls || '') + '"><div class="m-label"><span>' + label + '</span><span>' + pct + '%</span></div><div class="m-track"><div class="m-fill" style="width:' + pct + '%"></div></div></div>';
        }
        function renderRoomTrends(trends) {
            var el = document.getElementById('room-trends');
            if (!trends.length) { el.innerHTML = '<div class="empty"><div class="icon"></div>No trends yet</div>'; return; }
            el.innerHTML = trends.map(function (t) {
                var idx = allTrends ? allTrends.findIndex(function (x) { return x.id === t.id; }) : -1;
                var open = idx >= 0 ? 'openTrendModal(' + idx + ')' : "showTab('trends')";
                return '<div class="signal-item" onclick="' + open + '">' +
                    '<div><h3>' + esc(t.name) + '</h3></div>' +
                    '<div style="display:flex;gap:12px">' + meter('Strength', t.strength_score) + meter('White space', t.white_space_score, 'violet') + '</div>' +
                    '<p>' + esc(truncate(stripMd(t.why_it_matters || ''), 170)) + '</p>' +
                    '<div class="kicker">' + (t.region ? '<b>' + esc(t.region) + '</b> &middot; ' : '') + fmtInt(t.volume) + ' sources &middot; updated ' + fmtDate(t.last_updated) +
                    ' <button class="btn-stake" style="margin-left:10px" onclick="event.stopPropagation(); openStakeModal(&#39;' + esc(t.id) + '&#39;)">Stake prediction</button></div>' +
                    '</div>';
            }).join('');
        }
        function renderRoomMoves(moves) {
            var el = document.getElementById('room-moves');
            if (!moves.length) { el.innerHTML = '<div class="empty"><div class="icon"></div>No moves yet</div>'; return; }
            el.innerHTML = moves.map(function (m) {
                var idx = allMoves ? allMoves.findIndex(function (x) { return x.id === m.id; }) : -1;
                var open = idx >= 0 ? 'openMoveModal(' + idx + ')' : "showTab('moves')";
                return '<div class="signal-item" onclick="' + open + '">' +
                    '<div><h3>' + esc(truncate(m.title, 90)) + '</h3></div>' +
                    '<div class="kicker" style="grid-column:auto;text-align:right">' + esc((m.move_type || '').replace('_', ' ')) + '</div>' +
                    '<div class="kicker"><b>' + esc(m.company) + '</b>' + (m.market ? ' &middot; ' + esc(m.market) : '') + (m.investment_amount ? ' &middot; ' + esc(m.investment_amount) : '') + ' &middot; ' + esc(m.source_name) + (m.published_at ? ' &middot; ' + fmtDate(m.published_at) : '') + '</div>' +
                    '</div>';
            }).join('');
        }

        // ---------- Coverage ----------
        function renderCoverage(sources) {
            var el = document.getElementById('room-coverage');
            el.innerHTML = sources.freshness.map(function (s) {
                var cls = s.status === 'blocked' ? 'blocked' : s.hours_since == null ? 'silent' : s.hours_since <= 24 ? 'live' : s.hours_since <= 24 * 7 ? 'stale' : 'silent';
                var age = s.status === 'blocked' ? 'ToS' : hoursAgo(s.hours_since);
                return '<span class="chip ' + cls + '" title="' + esc(s.source) + ': ' + fmtInt(s.total_items) + ' items on file, ' + fmtInt(s.items_7d) + ' this week' + (s.status === 'blocked' ? ' (retained history; source blocked on terms of service)' : '') + '"><span class="dot"></span>' + esc(s.source.replace(/_/g, ' ')) + ' <span class="n">' + fmtInt(s.items_7d) + '</span><span class="age">' + age + '</span></span>';
            }).join('');
            var r = sources.registry;
            document.getElementById('room-registry').innerHTML = 'Registry: <code>' + (r.active || 0) + ' active</code> &middot; <code>' + (r.planned || 0) + ' planned</code> &middot; <code>' + (r.blocked || 0) + ' blocked</code>. Blocked sources (Reddit, OTA reviews) are excluded on terms-of-service grounds, not for lack of a scraper.';
        }

        // ---------- Active inference ----------
        async function loadAttention() {
            var el = document.getElementById('room-ai');
            try {
                var res = await fetch('/api/scheduler/pomdp');
                var d = await res.json();
                if (!d.enabled || !d.status || d.error) {
                    el.innerHTML = '<div class="empty"><div class="icon"></div>The scheduler is not running in this process, so no beliefs have been updated yet.</div>';
                    return;
                }
                var st = d.status;
                var srcs = Object.keys(st.sources || {}).map(function (k) { var s = st.sources[k]; s.name = k; return s; });
                srcs.sort(function (a, b) { return (b.productivity || 0) - (a.productivity || 0); });
                var fe = st.free_energy != null ? Number(st.free_energy).toFixed(3) : '&ndash;';
                var rec = d.next_recommended_source;
                var nx = rec && typeof rec === 'object' ? (rec.source || '') : (rec || '');
                nx = String(nx || 'pending');
                var reason = rec && typeof rec === 'object' && rec.reason ? rec.reason : '';
                var efe = rec && typeof rec === 'object' && rec.efe_values ? rec.efe_values : {};
                var head = '<div class="ai-head">' +
                    '<div class="ai-fig"><div class="f-label">Expected free energy</div><div class="f-value">' + fe + '<small>lower is better</small></div></div>' +
                    '<div class="ai-fig"><div class="f-label">Observations absorbed</div><div class="f-value">' + fmtInt(st.total_observations) + '</div></div>' +
                    '<div class="ai-fig"><div class="f-label">Next to read</div><div class="f-value" style="font-size:1.15em">' + esc(nx.replace(/_/g, ' ')) + (reason ? '<small>' + esc(reason) + '</small>' : '') + '</div></div>' +
                    '</div>';
                var rows = srcs.map(function (s) {
                    var p = Math.round((s.productivity || 0) * 100);
                    var flag = s.observations === 0 ? '<span class="b-flag" title="Unobserved: exploration candidate">?</span>' : (s.error_rate > 0.3 ? '<span class="b-flag" title="Elevated error rate">!</span>' : '<span></span>');
                    var g = efe[s.name] != null ? ' <small style="color:var(--ink-3)">G ' + Number(efe[s.name]).toFixed(2) + '</small>' : '';
                    return '<div class="belief"><span class="b-name">' + esc(s.name.replace(/_/g, ' ')) + '</span><div class="b-track"><div class="b-fill" style="width:' + p + '%"></div></div><span class="b-val">' + p + '%' + g + '</span>' + flag + '</div>';
                }).join('');
                el.innerHTML = head + rows +
                    '<div class="ai-explain">Bars are the current belief that a source will yield new, non-duplicate items on the next visit. <code>?</code> marks a source the agent has not yet observed &mdash; the epistemic term makes those attractive to try. ' +
                    'Uniform 50% across the board means the scheduler has just started and the beliefs are priors.</div>';
            } catch (e) {
                el.innerHTML = '<div class="empty"><div class="icon"></div>Beliefs unavailable: ' + esc(e.message) + '</div>';
            }
        }


        // ---------- Figures: Opportunity Map, City Matrix, Moves timeline ----------
        var REGION_SLOTS = { europe: { color: '#d4af6a', label: 'Europe' }, asia: { color: '#8b7ce0', label: 'Asia' }, north_america: { color: '#3aa88d', label: 'North America' }, other: { color: '#857a68', label: 'Other / global' } };
        var MOVE_GROUP_SLOTS = { deals: { color: '#d4af6a', label: 'Deals' }, product: { color: '#8b7ce0', label: 'Product' }, technology: { color: '#3aa88d', label: 'Technology' }, other: { color: '#857a68', label: 'Other' } };
        var figTip = null;
        function tip(wrap, html, x, y) {
            if (!figTip) { figTip = document.createElement('div'); figTip.className = 'chart-tip'; document.body.appendChild(figTip); }
            figTip.innerHTML = html; figTip.hidden = false;
            figTip.style.position = 'fixed'; figTip.style.left = (x + 14) + 'px'; figTip.style.top = (y + 14) + 'px'; figTip.style.transform = 'none';
        }
        function hideTip() { if (figTip) figTip.hidden = true; }
        function toggleFigTable(key) {
            var t = document.getElementById(key + '-table'), f = document.getElementById(key), b = document.getElementById(key + '-table-btn');
            var showTable = t.hidden; t.hidden = !showTable; f.hidden = showTable; b.classList.toggle('active', showTable); b.textContent = showTable ? 'Chart' : 'Table';
        }
        function legendHtml(slots, shape) {
            return Object.keys(slots).map(function (k) { return '<span class="key"><i class="' + (shape || '') + '" style="background:' + slots[k].color + '"></i>' + esc(slots[k].label) + '</span>'; }).join('');
        }
        function nice(v) { return Math.abs(v) >= 100 ? fmtInt(Math.round(v)) : (Math.round(v * 10) / 10); }

        function renderOpportunityMap(points) {
            var wrap = document.getElementById('omap');
            if (!points || !points.length) { wrap.innerHTML = '<div class="empty"><div class="icon"></div>No trends yet</div>'; return; }
            var W = 640, H = 420, pl = 46, pr = 18, pt = 16, pb = 44;
            var xs = points.map(function (p) { return p.strength; }), ys = points.map(function (p) { return p.white_space; });
            var xmin = Math.max(0, Math.min.apply(null, xs) - 0.05), xmax = Math.min(1, Math.max.apply(null, xs) + 0.05);
            var ymin = 0, ymax = Math.min(1, Math.max.apply(null, ys) + 0.06);
            var X = function (v) { return pl + (v - xmin) / (xmax - xmin) * (W - pl - pr); };
            var Y = function (v) { return pt + (1 - (v - ymin) / (ymax - ymin)) * (H - pt - pb); };
            var sorted = function (a) { return a.slice().sort(function (p, q) { return p - q; }); };
            var med = function (a) { var b = sorted(a); return b[Math.floor(b.length / 2)]; };
            var mx = med(xs), my = med(ys);
            var maxVol = Math.max.apply(null, points.map(function (p) { return p.volume; }).concat([1]));
            var R = function (v) { return 4 + Math.sqrt(v / maxVol) * 22; };
            var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Trend opportunity map">';
            svg += '<g class="grid">';
            [0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) { if (v >= xmin && v <= xmax) svg += '<line x1="' + X(v) + '" x2="' + X(v) + '" y1="' + pt + '" y2="' + (H - pb) + '"/>'; });
            [0.2, 0.4, 0.6, 0.8].forEach(function (v) { if (v <= ymax) svg += '<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/>'; });
            svg += '</g>';
            svg += '<line class="median" x1="' + X(mx) + '" x2="' + X(mx) + '" y1="' + pt + '" y2="' + (H - pb) + '"/><line class="median" x1="' + pl + '" x2="' + (W - pr) + '" y1="' + Y(my) + '" y2="' + Y(my) + '"/>';
            svg += '<text class="quad" x="' + (W - pr - 4) + '" y="' + (pt + 12) + '" text-anchor="end">white space</text>';
            svg += '<text class="quad" x="' + (W - pr - 4) + '" y="' + (H - pb - 6) + '" text-anchor="end">crowded</text>';
            svg += '<text class="quad" x="' + (pl + 4) + '" y="' + (pt + 12) + '">emerging</text>';
            svg += '<text class="quad" x="' + (pl + 4) + '" y="' + (H - pb - 6) + '">noise</text>';
            svg += '<g class="axis">';
            [0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) { if (v >= xmin && v <= xmax) svg += '<text x="' + X(v) + '" y="' + (H - pb + 16) + '" text-anchor="middle">' + Math.round(v * 100) + '%</text>'; });
            [0.2, 0.4, 0.6, 0.8].forEach(function (v) { if (v <= ymax) svg += '<text x="' + (pl - 8) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + Math.round(v * 100) + '%</text>'; });
            svg += '<text class="axis-title" x="' + (W / 2) + '" y="' + (H - 6) + '" text-anchor="middle">demand strength</text>';
            svg += '<text class="axis-title" transform="translate(12 ' + (H / 2) + ') rotate(-90)" text-anchor="middle">white space</text></g>';
            var byScore = points.slice().sort(function (a, b) { return (b.strength * b.white_space) - (a.strength * a.white_space); });
            var labelled = byScore.slice(0, 5).map(function (p) { return p.id; });
            points.slice().sort(function (a, b) { return b.volume - a.volume; }).forEach(function (p, i) {
                var c = REGION_SLOTS[p.region] || REGION_SLOTS.other;
                svg += '<circle class="dot" data-i="' + esc(p.id) + '" cx="' + X(p.strength).toFixed(1) + '" cy="' + Y(p.white_space).toFixed(1) + '" r="' + R(p.volume).toFixed(1) + '" fill="' + c.color + '" fill-opacity="0.78"/>';
            });
            var placedL = [];
            labelled.forEach(function (id) {
                var p = points.filter(function (q) { return q.id === id; })[0];
                var name = truncate(p.name, 28), cx = X(p.strength), r = R(p.volume);
                // ~6.3px per character at this size; flip to the left when the label would leave the plot
                var fitsRight = cx + r + 4 + name.length * 6.3 < W - pr;
                var lx = fitsRight ? cx + r + 4 : cx - r - 4, ly = Y(p.white_space) + 4;
                var width = name.length * 6.3, x0 = fitsRight ? lx : lx - width, x1 = x0 + width;
                // push down when another label occupies the same band
                placedL.forEach(function (q) { if (Math.abs(q.y - ly) < 14 && x0 < q.x1 && x1 > q.x0) ly = q.y + 14; });
                placedL.push({ y: ly, x0: x0, x1: x1 });
                svg += '<text class="lbl" x="' + lx.toFixed(1) + '" y="' + ly.toFixed(1) + '"' + (fitsRight ? '' : ' text-anchor="end"') + '>' + esc(name) + '</text>';
            });
            svg += '</svg>';
            wrap.innerHTML = svg;
            document.getElementById('omap-legend').innerHTML = legendHtml(REGION_SLOTS) + '<span class="key" style="color:var(--ink-3)">bubble = source volume</span>';
            document.getElementById('omap-foot').textContent = points.length + ' clusters. Medians drawn at ' + Math.round(mx * 100) + '% strength and ' + Math.round(my * 100) + '% white space.';
            var byId = {}; points.forEach(function (p) { byId[p.id] = p; });
            wrap.querySelectorAll('.dot').forEach(function (el) {
                el.addEventListener('pointermove', function (ev) { var p = byId[el.getAttribute('data-i')]; tip(wrap, '<div class="tip-date">' + esc((REGION_SLOTS[p.region] || REGION_SLOTS.other).label) + '</div><div class="tip-row"><span>' + esc(p.name) + '</span></div><div class="tip-row"><span>strength</span><b>' + Math.round(p.strength * 100) + '%</b></div><div class="tip-row"><span>white space</span><b>' + Math.round(p.white_space * 100) + '%</b></div><div class="tip-row"><span>sources</span><b>' + fmtInt(p.volume) + '</b></div><div class="tip-row"><span style="color:var(--ink-3)">click to open / build</span></div>', ev.clientX, ev.clientY); });
                el.addEventListener('pointerleave', hideTip);
                el.addEventListener('click', function () { var p = byId[el.getAttribute('data-i')]; var idx = allTrends ? allTrends.findIndex(function (x) { return x.id === p.id; }) : -1; if (idx >= 0) openTrendModal(idx); else showTab('trends'); });
            });
            document.getElementById('omap-table').innerHTML = '<table class="chart-table"><thead><tr><th>Trend</th><th>Region</th><th style="text-align:right">Strength</th><th style="text-align:right">White space</th><th style="text-align:right">Sources</th></tr></thead><tbody>' +
                byScore.map(function (p) { return '<tr><td>' + esc(p.name) + '</td><td>' + esc((REGION_SLOTS[p.region] || REGION_SLOTS.other).label) + '</td><td class="num">' + Math.round(p.strength * 100) + '%</td><td class="num">' + Math.round(p.white_space * 100) + '%</td><td class="num">' + fmtInt(p.volume) + '</td></tr>'; }).join('') + '</tbody></table>';
        }

        function renderCityMatrix(points) {
            var wrap = document.getElementById('cmat');
            if (!points || !points.length) { wrap.innerHTML = '<div class="empty"><div class="icon"></div>Needs attention and supply metrics for the same cities.</div>'; return; }
            var W = 520, H = 420, pl = 50, pr = 16, pt = 16, pb = 44;
            var xs = points.map(function (p) { return p.attention_change_pct; });
            var xr = Math.max(0.1, Math.max.apply(null, xs.map(Math.abs)) * 1.15);
            var ys = points.map(function (p) { return Math.log10(Math.max(1, p.hotels)); });
            var ymin = Math.floor(Math.min.apply(null, ys) * 2) / 2 - 0.1, ymax = Math.ceil(Math.max.apply(null, ys) * 2) / 2 + 0.1;
            // Signed square-root scale: one city at +46% must not flatten the
            // dozen sitting between -5% and +5% into a single blob.
            var sq = function (v) { return (v < 0 ? -1 : 1) * Math.sqrt(Math.abs(v)); };
            var X = function (v) { return pl + (sq(v) + sq(xr)) / (2 * sq(xr)) * (W - pl - pr); };
            var Y = function (v) { return pt + (1 - (v - ymin) / (ymax - ymin)) * (H - pt - pb); };
            var maxL = Math.max.apply(null, points.map(function (p) { return p.airbnb_listings || 0; }).concat([1]));
            var R = function (p) { return p.airbnb_listings ? 5 + Math.sqrt(p.airbnb_listings / maxL) * 20 : 5; };
            var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="City attention vs supply">';
            svg += '<g class="grid">';
            var yticks = []; for (var t = Math.ceil(ymin); t <= ymax; t += 0.5) yticks.push(t);
            yticks.forEach(function (v) { svg += '<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/>'; });
            svg += '</g>';
            svg += '<line class="median" x1="' + X(0) + '" x2="' + X(0) + '" y1="' + pt + '" y2="' + (H - pb) + '"/>';
            svg += '<text class="quad" x="' + (W - pr - 4) + '" y="' + (H - pb - 6) + '" text-anchor="end">rising, thin supply</text>';
            svg += '<text class="quad" x="' + (pl + 4) + '" y="' + (pt + 12) + '">cooling, dense supply</text>';
            svg += '<g class="axis">';
            var xt = [-0.4, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.4].filter(function (v) { return Math.abs(v) <= xr; });
            xt.forEach(function (v) { svg += '<text x="' + X(v) + '" y="' + (H - pb + 16) + '" text-anchor="middle">' + (v > 0 ? '+' : '') + Math.round(v * 100) + '%</text>'; });
            yticks.forEach(function (v) { svg += '<text x="' + (pl - 8) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + fmtInt(Math.round(Math.pow(10, v))) + '</text>'; });
            svg += '<text class="axis-title" x="' + (W / 2) + '" y="' + (H - 6) + '" text-anchor="middle">attention, week over week (square-root scale)</text>';
            svg += '<text class="axis-title" transform="translate(12 ' + (H / 2) + ') rotate(-90)" text-anchor="middle">hotels (log)</text></g>';
            var placed = [];
            points.slice().sort(function (a, b) { return R(b) - R(a); }).forEach(function (p) {
                var c = REGION_SLOTS[p.region] || REGION_SLOTS.other;
                var y = Math.log10(Math.max(1, p.hotels));
                svg += '<circle class="dot" data-c="' + esc(p.city) + '" cx="' + X(p.attention_change_pct).toFixed(1) + '" cy="' + Y(y).toFixed(1) + '" r="' + R(p).toFixed(1) + '" fill="' + c.color + '" fill-opacity="' + (p.airbnb_listings ? 0.8 : 0.35) + '"' + (p.airbnb_listings ? '' : ' stroke-dasharray="3 2"') + '/>';
            });
            // Labels after all dots (so they sit on top), nudged apart when two land within 13px.
            points.slice().sort(function (a, b) { return Math.log10(Math.max(1, a.hotels)) - Math.log10(Math.max(1, b.hotels)); }).forEach(function (p) {
                var cx = X(p.attention_change_pct), cy = Y(Math.log10(Math.max(1, p.hotels))), r = R(p);
                var right = cx + r + 4 + p.city.length * 6.3 < W - pr;
                var ly = cy + 4;
                placed.forEach(function (q) { if (q.right === right && Math.abs(q.y - ly) < 13 && Math.abs(q.x - cx) < 140) ly = q.y + 13; });
                placed.push({ x: cx, y: ly, right: right });
                svg += '<text class="lbl" x="' + (right ? cx + r + 4 : cx - r - 4).toFixed(1) + '" y="' + ly.toFixed(1) + '"' + (right ? '' : ' text-anchor="end"') + '>' + esc(p.city) + '</text>';
            });
            svg += '</svg>';
            wrap.innerHTML = svg;
            document.getElementById('cmat-legend').innerHTML = legendHtml(REGION_SLOTS) + '<span class="key" style="color:var(--ink-3)">bubble = Airbnb listings (dashed: no snapshot)</span>';
            document.getElementById('cmat-foot').textContent = points.length + ' cities with both an attention series and a supply count. Supply is hotels, hostels and guest houses inside the administrative boundary; boundaries differ in size, so read positions within a region.';
            var byCity = {}; points.forEach(function (p) { byCity[p.city] = p; });
            wrap.querySelectorAll('.dot').forEach(function (el) {
                el.addEventListener('pointermove', function (ev) { var p = byCity[el.getAttribute('data-c')]; tip(wrap, '<div class="tip-date">' + esc(p.city) + (p.country ? ', ' + esc(p.country) : '') + '</div><div class="tip-row"><span>attention w/w</span><b>' + fmtPct(p.attention_change_pct) + '</b></div><div class="tip-row"><span>hotels (OSM)</span><b>' + fmtInt(p.hotels) + '</b></div>' + (p.airbnb_listings ? '<div class="tip-row"><span>Airbnb listings</span><b>' + fmtInt(p.airbnb_listings) + '</b></div><div class="tip-row"><span>reviews / month</span><b>' + p.airbnb_reviews_per_month + '</b></div>' : '') + '<div class="tip-row"><span style="color:var(--ink-3)">click to explore desires</span></div>', ev.clientX, ev.clientY); });
                el.addEventListener('pointerleave', hideTip);
                el.addEventListener('click', function () { exploreCity(el.getAttribute('data-c')); });
            });
            document.getElementById('cmat-table').innerHTML = '<table class="chart-table"><thead><tr><th>City</th><th style="text-align:right">Attention w/w</th><th style="text-align:right">Hotels</th><th style="text-align:right">Airbnb listings</th><th style="text-align:right">Reviews / mo</th></tr></thead><tbody>' +
                points.slice().sort(function (a, b) { return b.attention_change_pct - a.attention_change_pct; }).map(function (p) { return '<tr><td>' + esc(p.city) + '</td><td class="num">' + fmtPct(p.attention_change_pct) + '</td><td class="num">' + fmtInt(p.hotels) + '</td><td class="num">' + (p.airbnb_listings ? fmtInt(p.airbnb_listings) : '&ndash;') + '</td><td class="num">' + (p.airbnb_reviews_per_month != null ? p.airbnb_reviews_per_month : '&ndash;') + '</td></tr>'; }).join('') + '</tbody></table>';
        }
        function exploreCity(name) {
            showTab('citydesires');
            var input = document.querySelector('#citydesires input[type="text"], #citydesires input');
            if (input) { input.value = name; input.focus(); }
        }

        function renderMovesByWeek(data) {
            var wrap = document.getElementById('mvw');
            var weeks = (data && data.weeks) || [];
            if (!weeks.length) { wrap.innerHTML = '<div class="empty"><div class="icon"></div>No moves in window</div>'; return; }
            var groups = data.groups;
            var W = 640, H = 260, pl = 40, pr = 12, pt = 12, pb = 34;
            var totals = weeks.map(function (w) { return groups.reduce(function (a, g) { return a + (w[g] || 0); }, 0); });
            var max = Math.max.apply(null, totals.concat([1]));
            var step = max > 40 ? 20 : max > 16 ? 10 : max > 8 ? 5 : 2;
            var ymax = Math.ceil(max / step) * step;
            var slot = (W - pl - pr) / weeks.length, bw = Math.min(24, slot * 0.6);
            var Y = function (v) { return pt + (1 - v / ymax) * (H - pt - pb); };
            var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Operator moves per week">';
            svg += '<g class="grid">'; for (var v = 0; v <= ymax; v += step) svg += '<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/>'; svg += '</g>';
            svg += '<g class="axis">'; for (var v2 = 0; v2 <= ymax; v2 += step) svg += '<text x="' + (pl - 8) + '" y="' + (Y(v2) + 4) + '" text-anchor="end">' + v2 + '</text>';
            weeks.forEach(function (w, i) { if (i % 2 === (weeks.length - 1) % 2) svg += '<text x="' + (pl + slot * i + slot / 2) + '" y="' + (H - pb + 16) + '" text-anchor="middle">' + fmtDate(w.week) + '</text>'; });
            svg += '</g>';
            weeks.forEach(function (w, i) {
                var x = pl + slot * i + (slot - bw) / 2, acc = 0;
                groups.forEach(function (g, gi) {
                    var v = w[g] || 0; if (!v) return;
                    var y0 = Y(acc + v), y1 = Y(acc); acc += v;
                    var h = Math.max(0, y1 - y0 - 2); // 2px surface gap between segments
                    var isTop = acc === totals[i];
                    svg += '<rect class="bar" data-w="' + i + '" x="' + x.toFixed(1) + '" y="' + y0.toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + h.toFixed(1) + '" fill="' + MOVE_GROUP_SLOTS[g].color + '"' + (isTop ? ' rx="4"' : '') + '/>';
                });
            });
            svg += '</svg>';
            wrap.innerHTML = svg;
            document.getElementById('mvw-legend').innerHTML = legendHtml(MOVE_GROUP_SLOTS, 'sq');
            document.getElementById('mvw-foot').textContent = fmtInt(data.total) + ' moves in the last ' + weeks.length + ' weeks, ' + fmtInt(data.from_filings) + ' read directly from SEC filings. Weeks with no extraction run show as empty, not as quiet markets.';
            wrap.querySelectorAll('.bar').forEach(function (el) {
                el.addEventListener('pointermove', function (ev) { var w = weeks[Number(el.getAttribute('data-w'))]; tip(wrap, '<div class="tip-date">week of ' + fmtDateYear(w.week) + '</div>' + groups.map(function (g) { return '<div class="tip-row"><i style="background:' + MOVE_GROUP_SLOTS[g].color + '"></i><span>' + MOVE_GROUP_SLOTS[g].label + '</span><b>' + (w[g] || 0) + '</b></div>'; }).join(''), ev.clientX, ev.clientY); });
                el.addEventListener('pointerleave', hideTip);
                el.addEventListener('click', function () { showTab('moves'); });
            });
            document.getElementById('mvw-table').innerHTML = '<table class="chart-table"><thead><tr><th>Week of</th>' + groups.map(function (g) { return '<th style="text-align:right">' + MOVE_GROUP_SLOTS[g].label + '</th>'; }).join('') + '<th style="text-align:right">Total</th></tr></thead><tbody>' +
                weeks.slice().reverse().map(function (w, i) { return '<tr><td>' + fmtDateYear(w.week) + '</td>' + groups.map(function (g) { return '<td class="num">' + (w[g] || 0) + '</td>'; }).join('') + '<td class="num">' + totals[weeks.length - 1 - i] + '</td></tr>'; }).join('') + '</tbody></table>';
        }

        // ---------- Company league (Market Moves tab) ----------
        function renderCompanyLeague(companies) {
            var el = document.getElementById('company-league');
            var lg = document.getElementById('company-legend');
            if (!el) return;
            if (lg) lg.innerHTML = legendHtml(MOVE_GROUP_SLOTS, 'sq');
            if (!companies || !companies.length) { el.innerHTML = '<div class="empty"><div class="icon"></div>No moves extracted yet</div>'; return; }
            var max = Math.max.apply(null, companies.map(function (c) { return c.moves; }));
            el.innerHTML = '<table class="chart-table"><thead><tr><th>Company</th><th>Moves</th><th style="width:34%">Mix</th><th>Markets</th><th>Latest</th></tr></thead><tbody>' + companies.map(function (c) {
                var order = Object.keys(MOVE_GROUP_SLOTS);
                var bar = '<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;gap:2px;width:' + Math.max(8, Math.round(c.moves / max * 100)) + '%">' + order.map(function (g) { var v = c.groups[g] || 0; return v ? '<span title="' + MOVE_GROUP_SLOTS[g].label + ': ' + v + '" style="flex:' + v + ';background:' + MOVE_GROUP_SLOTS[g].color + '"></span>' : ''; }).join('') + '</div>';
                return '<tr><td><strong>' + esc(c.company) + '</strong>' + (c.filings ? ' <span class="badge badge-success" title="read from SEC filings">' + c.filings + ' filing' + (c.filings > 1 ? 's' : '') + '</span>' : '') + '</td><td class="num">' + c.moves + '</td><td>' + bar + '</td><td style="color:var(--ink-2)">' + esc((c.markets || []).join(', ')) + '</td><td style="color:var(--ink-2)">' + esc(truncate(c.latest || '', 60)) + '</td></tr>';
            }).join('') + '</tbody></table>';
        }

        // ---------- Overview loader ----------
        async function loadOverview() {
            try {
                var res = await fetch('/api/overview');
                if (!res.ok) throw new Error('HTTP ' + res.status);
                roomData = await res.json();
                renderStats(roomData);
                renderOpportunityMap(roomData.trend_map || []);
                renderCityMatrix(roomData.city_matrix || []);
                renderMovesByWeek(roomData.moves_by_week);
                renderCompanyLeague(roomData.companies || []);
                renderCurves(roomData.demand);
                renderRoomTrends(roomData.trends || []);
                renderRoomMoves(roomData.moves || []);
                renderCoverage(roomData.sources);
                setCurvesView(curvesView);
            } catch (e) {
                console.error('overview failed', e);
                document.getElementById('curves-chart').innerHTML = '<div class="error">Overview failed: ' + esc(e.message) + '</div>';
            }
            loadAttention();
        }

        // ---------- Ledger ----------
        var ledgerLoaded = false;
        async function loadLedger(force) {
            if (ledgerLoaded && !force) return;
            var list = document.getElementById('ledger-list');
            var kpis = document.getElementById('ledger-kpis');
            try {
                var rs = await Promise.all([fetch('/api/signal-ledger/metrics'), fetch('/api/signal-ledger/predictions?limit=50')]);
                var m = await rs[0].json();
                var p = await rs[1].json();
                var preds = p.predictions || p.items || (Array.isArray(p) ? p : []);
                var hit = m.hit_rate == null ? '&ndash;' : Math.round(m.hit_rate * 100) + '%';
                var err = m.mean_abs_error_pct == null ? '&ndash;' : Math.round(m.mean_abs_error_pct * 100) + '%';
                var cal = m.calibration_gap == null ? '&ndash;' : ((m.calibration_gap > 0 ? '+' : '') + Math.round(m.calibration_gap * 100) + ' pts');
                var horizons = [];
                preds.forEach(function (q) { (q.forecasts || []).forEach(function (f) { if (f.horizon_date) horizons.push(new Date(f.horizon_date)); }); });
                horizons.sort(function (a, b) { return a - b; });
                var firstH = horizons.length ? horizons[0].toLocaleDateString(undefined, { month: 'short', year: 'numeric' }) : '&ndash;';
                kpis.innerHTML =
                    '<div class="ledger-kpi"><div class="k-label">Staked</div><div class="k-value">' + fmtInt(m.total_predictions) + '</div><div class="k-note">sealed records</div></div>' +
                    '<div class="ledger-kpi"><div class="k-label">Open</div><div class="k-value">' + fmtInt(m.open_predictions) + '</div><div class="k-note">first horizon ' + firstH + '</div></div>' +
                    '<div class="ledger-kpi"><div class="k-label">Resolved</div><div class="k-value">' + fmtInt(m.resolved_predictions) + '</div><div class="k-note">scored against sealed ranges</div></div>' +
                    '<div class="ledger-kpi"><div class="k-label">Hit rate</div><div class="k-value">' + hit + '</div><div class="k-note">in-range outcomes</div></div>' +
                    '<div class="ledger-kpi"><div class="k-label">Mean abs error</div><div class="k-value">' + err + '</div><div class="k-note">vs range midpoint</div></div>' +
                    '<div class="ledger-kpi"><div class="k-label">Calibration</div><div class="k-value">' + cal + '</div><div class="k-note">confidence minus hit rate</div></div>';
                if (!preds.length) {
                    list.innerHTML = '<div class="empty"><div class="icon"></div>No predictions staked yet.</div>';
                } else {
                    list.innerHTML = preds.map(renderPrediction).join('');
                }
                ledgerLoaded = true;
            } catch (e) {
                list.innerHTML = '<div class="error">Ledger failed: ' + esc(e.message) + '</div>';
            }
        }
        function renderPrediction(q) {
            var status = (q.status || 'open').replace('resolved_', '');
            var forecasts = (q.forecasts || []).map(function (f) {
                var unit = f.unit ? ' ' + esc(f.unit) : '';
                return '<div class="forecast"><div class="f-metric">' + esc((f.metric || '').replace(/_/g, ' ')) + '</div>' +
                    '<div class="f-range">' + fmtInt(Math.round(f.predicted_low)) + ' &ndash; ' + fmtInt(Math.round(f.predicted_high)) + unit + ' <small>by ' + fmtDateYear(f.horizon_date) + ' &middot; p=' + Number(f.confidence).toFixed(2) + '</small></div>' +
                    (f.falsifier ? '<div class="f-fals">Falsified if: ' + esc(f.falsifier) + '</div>' : '') + '</div>';
            }).join('');
            return '<div class="pred" id="pred-' + esc(q.id) + '">' +
                '<div class="pred-top"><div><h3>' + esc(q.title) + '</h3>' +
                '<div class="seal">SEALED <b>' + esc((q.content_hash || '').slice(0, 16)) + '</b> &middot; recorded ' + fmtDateYear(q.recorded_at) + ' &middot; signal first seen ' + fmtDateYear(q.signal_date) + ' &middot; ' + esc(q.methodology_version || '') + '</div></div>' +
                '<span class="status ' + esc(status) + '">' + esc(status) + '</span></div>' +
                '<p class="hyp">' + esc(truncate(q.hypothesis || '', 260)) + '</p>' +
                '<div class="forecasts">' + forecasts + '</div>' +
                '<div class="pred-foot"><span>source: ' + esc(q.signal_source || '') + '</span>' + (q.location_thesis ? '<span>where: ' + esc(q.location_thesis) + '</span>' : '') +
                (q.highest_evidence_stage ? '<span>stage: ' + esc(q.highest_evidence_stage) + '</span>' : '<span>stage: awareness</span>') +
                '<button onclick="toggleEvents(&#39;' + esc(q.id) + '&#39;)">Events</button></div>' +
                '<div class="events" id="ev-' + esc(q.id) + '" hidden></div></div>';
        }
        async function toggleEvents(id) {
            var el = document.getElementById('ev-' + id);
            if (!el.hidden) { el.hidden = true; return; }
            el.hidden = false;
            el.innerHTML = '<div class="ev"><span class="t">loading</span></div>';
            try {
                var r = await fetch('/api/signal-ledger/predictions/' + id);
                var d = await r.json();
                var evs = d.events || [];
                var sealRow = '<div class="ev"><span class="t">' + fmtDateYear(d.recorded_at) + '</span><span class="k">sealed</span><span>' +
                    (d.hash_verified ? 'Content hash re-derived and verified against the stored seal.' : 'Warning: stored hash does not match the record content.') + '</span></div>';
                if (!evs.length) { el.innerHTML = sealRow + '<div class="ev"><span class="t"></span><span class="k">open</span><span>No evidence appended yet. Outcomes are scored here when horizons pass.</span></div>'; return; }
                el.innerHTML = sealRow + evs.map(function (e) {
                    return '<div class="ev"><span class="t">' + fmtDateYear(e.recorded_at) + '</span><span class="k">' + esc(e.event_type) + (e.stage ? ' / ' + esc(e.stage) : '') + '</span><span>' + esc(e.description) + (e.actual_value != null ? ' (' + e.actual_value + ')' : '') + '</span></div>';
                }).join('');
            } catch (e) { el.innerHTML = '<div class="ev"><span class="t">error</span><span></span><span>' + esc(e.message) + '</span></div>'; }
        }

        // ---------- Stake a prediction ----------
        // A trend becomes a sealed, falsifiable record. The form pre-fills from
        // the trend; the numbers are the user's call, and the hash returned by
        // the API is shown so the room can see the seal happen.
        function openStakeModal(trendId) {
            var t = (roomData && roomData.trends || []).concat(allTrends || []).filter(function (x) { return x.id === trendId; })[0];
            if (!t) { showTab('trends'); return; }
            var horizon = new Date(); horizon.setDate(horizon.getDate() + 90);
            var conf = Math.round((0.45 + 0.4 * (t.strength_score || 0.5)) * 100) / 100;
            var vol = Math.max(1, t.volume || 1);
            document.getElementById('modal-title').textContent = 'Stake a prediction';
            document.getElementById('modal-meta').textContent = 'From trend: ' + (t.name || '');
            document.getElementById('modal-body').innerHTML =
                '<div class="stake-hint">Sealed at submission with a SHA-256 hash of every field below. It cannot be edited afterwards; it can only be resolved against real outcomes. Make it falsifiable.</div>' +
                '<div class="stake-form">' +
                '<div class="full"><label>Hypothesis</label><textarea id="st-hyp">' + esc(stripMd(t.description || t.why_it_matters || '')).slice(0, 600) + '</textarea></div>' +
                '<div class="full"><label>Product implication</label><textarea id="st-imp">' + esc(stripMd(t.why_it_matters || '')).slice(0, 400) + '</textarea></div>' +
                '<div><label>Metric</label><input id="st-metric" value="source_volume"></div>' +
                '<div><label>Unit</label><input id="st-unit" value="items"></div>' +
                '<div><label>Predicted low</label><input id="st-low" type="number" value="' + Math.round(vol * 1.1) + '"></div>' +
                '<div><label>Predicted high</label><input id="st-high" type="number" value="' + Math.round(vol * 1.8) + '"></div>' +
                '<div><label>Horizon date</label><input id="st-horizon" type="date" value="' + horizon.toISOString().slice(0, 10) + '"></div>' +
                '<div><label>Confidence (0-1)</label><input id="st-conf" type="number" step="0.05" min="0" max="1" value="' + conf + '"></div>' +
                '<div class="full"><label>Falsifier</label><input id="st-fals" value="Fewer than ' + Math.round(vol * 1.1) + ' corpus items match this cluster at the horizon date"></div>' +
                '<div><label>Where</label><input id="st-where" value="' + esc(t.region || '') + '"></div>' +
                '<div><label>Project</label><input id="st-proj" value="BrandClave platform"></div>' +
                '</div>' +
                '<div class="stake-actions"><span id="st-status" style="color:var(--ink-3);font-size:0.8em"></span><button class="tool-btn active" onclick="submitStake(&#39;' + esc(t.id) + '&#39;)">Seal prediction</button></div>' +
                '<div id="st-result"></div>';
            document.getElementById('modal-overlay').style.display = 'block';
        }
        async function submitStake(trendId) {
            var t = (roomData && roomData.trends || []).concat(allTrends || []).filter(function (x) { return x.id === trendId; })[0] || {};
            var v = function (id) { return document.getElementById(id).value; };
            var status = document.getElementById('st-status');
            status.textContent = 'sealing...';
            var body = {
                title: t.name || 'Untitled prediction',
                signal_date: (t.first_seen || t.last_updated || new Date().toISOString()),
                signal_source: 'social + trade press clustering (' + (t.volume || 0) + ' items)',
                hypothesis: v('st-hyp'),
                product_implication: v('st-imp'),
                location_thesis: v('st-where') || null,
                forecasts: [{
                    metric: v('st-metric'), unit: v('st-unit'),
                    predicted_low: Number(v('st-low')), predicted_high: Number(v('st-high')),
                    horizon_date: new Date(v('st-horizon')).toISOString(),
                    confidence: Number(v('st-conf')), falsifier: v('st-fals') || null
                }],
                uncertainty_notes: 'Staked from the dashboard by a signed-in analyst; forecast range set by hand.',
                methodology_version: 'v1.1-analyst',
                project: v('st-proj') || null,
                source_trend_ids: [trendId],
                source_content_ids: [],
                metadata: { staked_from: 'signal-room' }
            };
            try {
                var res = await fetch('/api/signal-ledger/predictions', { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify(body) });
                if (!res.ok) { var err = await res.text(); throw new Error('HTTP ' + res.status + ' ' + err.slice(0, 200)); }
                var rec = await res.json();
                status.textContent = '';
                document.getElementById('st-result').innerHTML = '<div class="stake-result">Sealed. Content hash <b>' + esc(rec.content_hash) + '</b><br>Recorded ' + esc(parseUtc(rec.recorded_at).toLocaleString()) + ' &middot; status ' + esc(rec.status) + '. It now appears in the Signal Ledger tab; append evidence there as it arrives.</div>';
                ledgerLoaded = false;
                loadOverview();
            } catch (e) {
                status.textContent = 'failed: ' + e.message;
            }
        }

        // Wire into the page lifecycle: overview on load and on the refresh
        // cadence; ledger lazily when its tab opens.
        var _origShowTab = showTab;
        showTab = function (tabId) {
            _origShowTab(tabId);
            if (tabId === 'ledger') loadLedger(false);
        };
        function updateFunnelPicks() {
            try {
                // Count whatever the save buttons have stored; keys differ per item type.
                var counts = { trend: 0, move: 0, property: 0, blueprint: 0 };
                for (var i = 0; i < localStorage.length; i++) {
                    var key = localStorage.key(i);
                    if (!/^brandclave_/.test(key) || /token|user|prefill/.test(key)) continue;
                    var val; try { val = JSON.parse(localStorage.getItem(key)); } catch (e2) { continue; }
                    var n = Array.isArray(val) ? val.length : (val && typeof val === 'object' ? Object.keys(val).length : 0);
                    if (!n) continue;
                    if (/move/.test(key)) counts.move += n; else if (/propert/.test(key)) counts.property += n; else if (/blueprint/.test(key)) counts.blueprint += n; else counts.trend += n;
                }
                var parts = [];
                if (counts.trend) parts.push(counts.trend + ' trend' + (counts.trend > 1 ? 's' : ''));
                if (counts.property) parts.push(counts.property + ' scanned propert' + (counts.property > 1 ? 'ies' : 'y'));
                if (counts.move) parts.push(counts.move + ' move' + (counts.move > 1 ? 's' : ''));
                if (counts.blueprint) parts.push(counts.blueprint + ' blueprint' + (counts.blueprint > 1 ? 's' : ''));
                var el = document.getElementById('funnel-picks');
                if (el) el.textContent = parts.length ? parts.join(' · ') + ' saved' : 'nothing saved yet — save a trend or scan a property to start a brief';
            } catch (e) {}
        }
        document.addEventListener('DOMContentLoaded', function () { loadOverview(); updateFunnelPicks(); });
        window.addEventListener('storage', updateFunnelPicks);
        setInterval(updateFunnelPicks, 5000);
        setInterval(loadOverview, 120000);
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
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800;900&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg: #0d0b08;
            --surface: #17140f;
            --surface-2: #201a12;
            --surface-3: #2a2318;
            --ink: #f2ecdf;
            --ink-2: #b9ae9c;
            --ink-3: #857a68;
            --line: rgba(212,175,106,0.15);
            --line-strong: rgba(212,175,106,0.34);
            --gold: #d4af6a;
            --gold-deep: #b8862e;
            --gold-ink: #141008;
            --gold-grad: linear-gradient(135deg, #ecd7a8 0%, #d4af6a 48%, #b18a41 100%);
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
            min-height: 100vh;
            color: var(--ink);
            background:
                radial-gradient(ellipse 75% 50% at 50% -14%, rgba(216,180,120,0.13), transparent 62%),
                radial-gradient(ellipse 42% 34% at 88% 4%, rgba(139,124,224,0.06), transparent 70%),
                radial-gradient(ellipse 42% 34% at 8% 10%, rgba(58,168,141,0.05), transparent 70%),
                radial-gradient(1.2px 1.2px at 21% 26%, rgba(242,236,223,0.32), transparent 100%),
                radial-gradient(1px 1px at 67% 14%, rgba(242,236,223,0.24), transparent 100%),
                radial-gradient(1.4px 1.4px at 84% 41%, rgba(226,193,132,0.20), transparent 100%),
                radial-gradient(1px 1px at 39% 57%, rgba(242,236,223,0.16), transparent 100%),
                var(--bg);
            /* background-attachment: fixed removed: nine stacked radial gradients repainted on every scroll tick and stalled the renderer */
        }
        ::selection { background: rgba(212,175,106,0.30); }
        ::-webkit-scrollbar { width: 11px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 6px; border: 3px solid var(--bg); }

        .hero {
            position: relative;
            padding: 88px 20px 56px;
            text-align: center;
            border-bottom: 1px solid var(--line);
            background:
                radial-gradient(closest-side circle at 50% 132%, rgba(236,215,168,0.20), rgba(212,175,106,0.07) 52%, transparent 78%);
            overflow: hidden;
        }
        .hero::before {
            content: 'BRANDCLAVE / STEP 4 · MAKE';
            display: block;
            font-family: var(--font-mono);
            font-size: 0.72em;
            letter-spacing: 0.42em;
            text-indent: 0.42em;
            color: var(--gold);
            margin-bottom: 20px;
            text-shadow: 0 0 24px rgba(212,175,106,0.45);
        }
        .hero h1 {
            font-family: var(--font-display);
            font-weight: 900;
            font-size: clamp(2.4em, 6vw, 3.4em);
            letter-spacing: 0.06em;
            text-transform: uppercase;
            line-height: 1.08;
            background: linear-gradient(180deg, #fbf6e9 18%, #e8dcc0 52%, #bda87c 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            /* drop-shadow filter removed: on background-clip:text it makes Chrome rasterise the layer per scroll tick and stalls capture */
        }
        .hero p { color: var(--ink-2); margin-top: 14px; font-size: 0.98em; }
        .back-link {
            display: inline-block;
            margin-top: 18px;
            color: var(--gold);
            text-decoration: none;
            font-family: var(--font-mono);
            font-size: 0.78em;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            padding: 8px 18px;
            transition: all 0.25s;
        }
        .back-link:hover {
            background: var(--gold-grad);
            color: var(--gold-ink);
            border-color: transparent;
            box-shadow: 0 6px 18px -8px rgba(212,175,106,0.6);
        }
        .container { max-width: 900px; margin: 0 auto; padding: 32px 20px; }

        .card {
            background: linear-gradient(180deg, rgba(255,244,222,0.035), rgba(255,244,222,0.008) 38%, transparent), var(--surface);
            border: 1px solid var(--line);
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.05), 0 24px 48px -30px rgba(0,0,0,0.85);
            padding: 30px;
            margin-bottom: 22px;
            border-radius: 18px;
        }
        .card h2 {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1.2em;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--ink);
            margin-bottom: 20px;
        }
        .card h2::before {
            content: '';
            display: block;
            width: 52px;
            height: 3px;
            background: var(--grad);
            border-radius: 2px;
            margin-bottom: 14px;
            box-shadow: 0 0 12px rgba(139,124,224,0.4);
        }

        .source-trend {
            background: linear-gradient(135deg, rgba(139,124,224,0.16), rgba(139,124,224,0.05) 48%, rgba(139,124,224,0.02)), var(--surface);
            border: 1px solid rgba(139,124,224,0.32);
            border-left: 3px solid var(--violet);
            color: var(--ink);
            padding: 20px;
            border-radius: 14px;
            margin-bottom: 20px;
            box-shadow: 0 18px 36px -26px rgba(139,124,224,0.55);
        }
        .source-trend h3 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }
        .source-trend p { color: var(--ink-2); font-size: 0.9em; }

        .profile-source-card {
            background: linear-gradient(135deg, rgba(58,168,141,0.15), rgba(58,168,141,0.05) 48%, rgba(58,168,141,0.02)), var(--surface);
            border: 1px solid rgba(58,168,141,0.32);
            border-left: 3px solid var(--teal);
            color: var(--ink);
            padding: 22px;
            border-radius: 14px;
            margin-bottom: 20px;
            box-shadow: 0 18px 36px -26px rgba(58,168,141,0.5);
        }
        .profile-source-card h3 { margin-bottom: 8px; font-family: var(--font-display); font-weight: 600; }
        .profile-row { display: flex; gap: 8px; margin-bottom: 4px; color: var(--ink-2); }
        .profile-label { color: var(--ink-3); }
        .profile-theme-tag {
            display: inline-block;
            background: rgba(58,168,141,0.16);
            color: var(--teal);
            padding: 4px 11px;
            border-radius: 999px;
            font-size: 0.82em;
            margin: 2px;
        }

        .form-group { margin-bottom: 22px; }
        .form-group label {
            display: block;
            margin-bottom: 9px;
            font-family: var(--font-mono);
            font-size: 0.7em;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--ink-3);
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 13px 14px;
            background: var(--surface-2);
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            color: var(--ink);
            font-family: var(--font-body);
            font-size: 0.95em;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .form-group textarea { min-height: 100px; resize: vertical; }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none;
            border-color: var(--gold);
            box-shadow: 0 0 0 3px rgba(212,175,106,0.14);
        }
        .form-group input::placeholder, .form-group textarea::placeholder { color: var(--ink-3); }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
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
            background: rgba(139,124,224,0.15);
            color: #b1a5ec;
            padding: 5px 13px;
            border-radius: 999px;
            font-size: 0.82em;
        }

        .btn-generate {
            width: 100%;
            padding: 17px 30px;
            background: var(--gold-grad);
            color: var(--gold-ink);
            border: none;
            border-radius: 12px;
            font-family: var(--font-display);
            font-size: 0.95em;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            cursor: pointer;
            box-shadow: 0 10px 30px -12px rgba(212,175,106,0.65), inset 0 1px 0 rgba(255,255,255,0.3);
            transition: transform 0.2s, box-shadow 0.25s;
        }
        .btn-generate:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 38px -12px rgba(212,175,106,0.8), inset 0 1px 0 rgba(255,255,255,0.3);
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
            position: relative;
            background: linear-gradient(180deg, rgba(255,244,222,0.04), transparent 40%), var(--surface);
            border: 1px solid var(--line-strong);
            padding: 34px 30px 30px;
            border-radius: 18px;
            color: var(--ink);
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.05), 0 30px 60px -35px rgba(0,0,0,0.9);
            overflow: hidden;
        }
        .blueprint-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: var(--grad);
            box-shadow: 0 0 20px rgba(139,124,224,0.5);
        }
        .blueprint-card h2 {
            color: var(--ink);
            margin-bottom: 6px;
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 1.7em;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: linear-gradient(180deg, #fbf6e9 18%, #e8dcc0 52%, #bda87c 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .blueprint-oneliner { font-size: 1.05em; color: var(--gold); margin-bottom: 26px; }

        .blueprint-section { margin-bottom: 26px; }
        .blueprint-section h3 {
            font-family: var(--font-mono);
            font-size: 0.73em;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            font-weight: 500;
            color: var(--gold);
            margin-bottom: 11px;
        }
        .blueprint-section p { line-height: 1.7; color: var(--ink-2); }
        .blueprint-section ul { padding-left: 20px; color: var(--ink-2); }
        .blueprint-section li { margin-bottom: 6px; }

        .experience-card {
            background: var(--surface-2);
            border: 1px solid var(--line);
            padding: 15px 17px;
            border-radius: 10px;
            margin-bottom: 9px;
            box-shadow: inset 0 1px 0 rgba(255,244,222,0.04);
        }
        .experience-card h4 { margin-bottom: 5px; color: var(--ink); }
        .experience-card p { font-size: 0.9em; color: var(--ink-2); }

        .loading-indicator { text-align: center; padding: 44px; }
        .loading-indicator .spinner {
            width: 52px;
            height: 52px;
            border: 3px solid var(--line);
            border-top-color: var(--gold);
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
            margin: 0 auto 18px;
            box-shadow: 0 0 24px -6px rgba(212,175,106,0.4);
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .stage-item { color: var(--ink-2); padding: 4px 0; font-family: var(--font-mono); font-size: 0.85em; }
        .stage-icon { color: var(--gold); }

        .btn-actions {
            display: flex;
            gap: 10px;
            margin-top: 22px;
            flex-wrap: wrap;
        }
        .btn-secondary {
            flex: 1;
            min-width: 150px;
            padding: 13px;
            background: transparent;
            color: var(--ink-2);
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.92em;
            transition: all 0.22s;
        }
        .btn-secondary:hover { color: var(--ink); border-color: var(--gold); box-shadow: 0 0 16px -6px rgba(212,175,106,0.4); }

        .white-space-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: var(--surface-2);
            border: 1px solid var(--line);
            color: var(--ink-2);
            padding: 3px 10px;
            border-radius: 999px;
            font-family: var(--font-mono);
            font-size: 0.78em;
            margin-top: 8px;
        }

        /* Concept renders */
        .renders { margin-top: 22px; }
        .renders-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; flex-wrap: wrap; margin-bottom: 12px; }
        .renders-head p { color: var(--ink-2); font-size: 0.88em; max-width: 64ch; line-height: 1.5; }
        .renders-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
        @media (max-width: 760px) { .renders-grid { grid-template-columns: 1fr; } }
        .render-tile { position: relative; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; background: var(--surface-2); aspect-ratio: 3 / 2; }
        .render-tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .render-tile .cap { position: absolute; left: 0; right: 0; bottom: 0; padding: 10px 14px; background: linear-gradient(180deg, transparent, rgba(8,6,4,0.85)); color: var(--ink); font-family: var(--font-mono); font-size: 0.7em; letter-spacing: 0.14em; text-transform: uppercase; display: flex; justify-content: space-between; align-items: center; }
        .render-tile .cap button { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); color: var(--ink); font-family: var(--font-mono); font-size: 0.9em; letter-spacing: 0.1em; text-transform: uppercase; padding: 3px 8px; border-radius: 6px; cursor: pointer; }
        .render-tile.pending { display: flex; align-items: center; justify-content: center; color: var(--ink-3); font-family: var(--font-mono); font-size: 0.72em; letter-spacing: 0.14em; text-transform: uppercase; }
        .render-tile.pending::after { content: ''; position: absolute; inset: 0; background: linear-gradient(90deg, transparent, rgba(212,175,106,0.08), transparent); animation: shimmer 1.6s infinite; }
        @keyframes shimmer { from { transform: translateX(-100%); } to { transform: translateX(100%); } }
        .render-note { color: var(--ink-3); font-family: var(--font-mono); font-size: 0.66em; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 10px; }
        .render-prompt { color: var(--ink-3); font-size: 0.78em; line-height: 1.5; margin-top: 8px; display: none; }
        .render-tile:hover .render-prompt { display: block; }
        .pick-chip {
            background: var(--surface-2); border: 1px solid var(--line-strong); color: var(--ink); border-radius: 999px;
            padding: 8px 14px; font-size: 0.86em; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; transition: border-color 0.2s, background 0.2s;
        }
        .pick-chip:hover { border-color: var(--gold); background: var(--surface-3); }
        .pick-chip .ws { font-family: var(--font-mono); font-size: 0.72em; color: var(--gold); letter-spacing: 0.06em; }
        .card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
        .card-head h2 { margin-bottom: 6px; }
        .card-sub { color: var(--ink-3); font-size: 0.86em; max-width: 64ch; margin-bottom: 14px; line-height: 1.5; }
    </style>
</head>
<body>
    <div class="hero">
        <h1>Build a Brand</h1>
        <p>Transform market trends into unique hotel brand concepts</p>
        <a href="/api/monitoring/dashboard-v2" class="back-link">Back to Dashboard</a>
    </div>

    <div class="container">
        <div id="picks-card" class="card" style="display:none;">
            <div class="card-head">
                <div>
                    <h2>Start from your picks</h2>
                    <div class="card-sub">Trends you saved on the dashboard. Pick one and the brief below fills from it; the blueprint stays grounded in that signal's evidence.</div>
                </div>
            </div>
            <div id="picks-chips" style="display:flex;flex-wrap:wrap;gap:8px;"></div>
        </div>

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

                <div class="blueprint-section renders" id="bp-renders-section">
                    <div class="renders-head">
                        <div>
                            <h3>Concept renders</h3>
                            <p>Four spaces every concept has to answer for, visualised from this blueprint's own design direction, F&amp;B concept and brand feeling. Nothing is added that the concept did not specify.</p>
                        </div>
                        <div style="display:flex;gap:8px;align-items:center;">
                            <button class="btn-secondary" id="bp-render-btn" onclick="generateRenders()">Visualise the concept</button>
                        </div>
                    </div>
                    <div class="renders-grid" id="bp-renders"></div>
                    <div class="render-note" id="bp-renders-note"></div>
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

        // Shared auth token (set by the dashboard's sign-in); requests carry it
        // so generated blueprints are owned by the signed-in user.
        function authHeaders(extra) {
            var h = extra || {};
            try {
                var t = localStorage.getItem('brandclave_token');
                if (t) h['Authorization'] = 'Bearer ' + t;
            } catch (e) {}
            return h;
        }

        // Saved picks from the dashboard become one-click starting points.
        function renderPicks() {
            var chips = document.getElementById('picks-chips'), card = document.getElementById('picks-card');
            var saved = [];
            try { saved = JSON.parse(localStorage.getItem('brandclave_saved_trends') || '[]'); } catch (e) {}
            if (!Array.isArray(saved) || !saved.length) { card.style.display = 'none'; return; }
            chips.innerHTML = saved.slice(0, 12).map(function (t, i) {
                var ws = t.white_space_score != null ? Math.round(t.white_space_score * 100) + '% white space' : '';
                return '<button class="pick-chip" onclick="startFromPick(' + i + ')">' + (t.name || t.trend_name || 'Saved trend') + (ws ? ' <span class="ws">' + ws + '</span>' : '') + '</button>';
            }).join('');
            card.style.display = 'block';
        }
        function startFromPick(i) {
            var saved = [];
            try { saved = JSON.parse(localStorage.getItem('brandclave_saved_trends') || '[]'); } catch (e) {}
            var t = saved[i]; if (!t) return;
            sessionStorage.setItem('brandclave_brand_input', JSON.stringify({
                source_trend_id: t.id, source_trend_name: t.name || t.trend_name,
                initial_segment: t.audience_segment || 'lifestyle', initial_region: t.region || '',
                topics: t.topics || [], white_space_score: t.white_space_score,
                description: t.description, why_it_matters: t.why_it_matters
            }));
            sessionStorage.removeItem('brandclave_profile_data');
            loadSourceTrend();
            var card = document.getElementById('source-trend-card'); if (card) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

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
                    headers: authHeaders({ 'Content-Type': 'application/json' }),
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


        // ---------- Concept renders ----------
        var RENDER_SCENES = [
            { key: 'arrival', label: 'Arrival & facade' },
            { key: 'lobby', label: 'Lobby & social heart' },
            { key: 'room', label: 'Signature guest room' },
            { key: 'fnb', label: 'Food & beverage' }
        ];
        function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
        function renderTiles(manifest, pendingKeys) {
            var grid = document.getElementById('bp-renders');
            var byScene = {};
            ((manifest && manifest.renders) || []).forEach(function (r) { byScene[r.scene] = r; });
            grid.innerHTML = RENDER_SCENES.map(function (sc) {
                var r = byScene[sc.key];
                if (pendingKeys && pendingKeys.indexOf(sc.key) !== -1) {
                    return '<div class="render-tile pending">rendering ' + esc(sc.label) + '&hellip;</div>';
                }
                if (!r) {
                    return '<div class="render-tile pending" style="animation:none">' + esc(sc.label) + '</div>';
                }
                var bust = r.generated_at ? '?v=' + encodeURIComponent(r.generated_at) : '';
                return '<div class="render-tile"><img src="' + r.url + bust + '" alt="' + esc(sc.label) + '" loading="lazy">' +
                    '<div class="cap"><span>' + esc(sc.label) + '</span><button onclick="generateRenders(&#39;' + sc.key + '&#39;)">Redo</button></div></div>';
            }).join('');
            var note = document.getElementById('bp-renders-note');
            if (manifest && manifest.renders && manifest.renders.length) {
                note.textContent = manifest.renders.length + ' of 4 rendered · ' + (manifest.model || '') + ' · generated ' + parseUtcDate(manifest.generated_at);
            } else {
                note.textContent = '';
            }
        }
        function parseUtcDate(iso) {
            if (!iso) return '';
            var d = new Date(/[Zz]$|[+-][0-9]{2}:?[0-9]{2}$/.test(String(iso)) ? iso : iso + 'Z');
            return d.toLocaleString();
        }
        async function loadRenders(blueprintId) {
            if (!blueprintId) { renderTiles(null); return; }
            try {
                var res = await fetch('/api/brand-blueprint/' + blueprintId + '/renders');
                if (res.status === 404) { renderTiles(null); return; }
                var manifest = await res.json();
                renderTiles(manifest);
            } catch (e) { renderTiles(null); }
        }
        async function generateRenders(sceneKey) {
            var bp = currentBlueprint;
            var id = bp && (bp.id || bp.blueprint_id);
            if (!id) { alert('Save or reload the blueprint first so it has an id.'); return; }
            var btn = document.getElementById('bp-render-btn');
            var keys = sceneKey ? [sceneKey] : RENDER_SCENES.map(function (s) { return s.key; });
            btn.disabled = true; btn.textContent = sceneKey ? 'Rendering…' : 'Rendering 4 scenes…';
            var existing = null;
            try { var r0 = await fetch('/api/brand-blueprint/' + id + '/renders'); if (r0.ok) existing = await r0.json(); } catch (e) {}
            renderTiles(existing, keys);
            try {
                var res = await fetch('/api/brand-blueprint/' + id + '/renders', {
                    method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({ scenes: keys, quality: 'medium' })
                });
                var data = await res.json();
                if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
                renderTiles(data);
                if (data.failures && data.failures.length) {
                    document.getElementById('bp-renders-note').textContent += ' · ' + data.failures.length + ' scene(s) failed: ' + data.failures.map(function (f) { return f.error; }).join('; ');
                }
            } catch (e) {
                document.getElementById('bp-renders-note').textContent = 'Render failed: ' + e.message;
                renderTiles(existing);
            } finally {
                btn.disabled = false; btn.textContent = 'Visualise the concept';
            }
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

            loadRenders(blueprint.id || blueprint.blueprint_id);
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
                var response = await fetch('/api/brand-blueprint?limit=10', { headers: authHeaders() });
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
        loadSourceTrend(); renderPicks();
        loadSavedBlueprints();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
