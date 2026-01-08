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
    <title>BrandClave Intelligence</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #333;
        }
        .hero {
            background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }
        .hero h1 { font-size: 2em; margin-bottom: 8px; }
        .hero p { opacity: 0.9; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }

        .status-bar {
            background: rgba(255,255,255,0.1);
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: white;
        }
        .status-bar .icon { font-size: 1.3em; }
        .status-bar button {
            margin-left: auto;
            padding: 8px 16px;
            background: #e94560;
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
        }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
        }
        .tab:hover { background: rgba(255,255,255,0.2); }
        .tab.active { background: #e94560; }

        .section { display: none; }
        .section.active { display: block; }

        .card {
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h2 { color: #1a1a2e; margin-bottom: 15px; font-size: 1.3em; }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .metric-value { font-size: 1.8em; font-weight: bold; color: #0f3460; }
        .metric-label { color: #666; font-size: 0.8em; margin-top: 4px; }

        .trend-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .trend-card h3 { margin-bottom: 8px; }
        .trend-card p { opacity: 0.95; font-size: 0.9em; line-height: 1.4; }
        
        .trend-card:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        .move-card:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }

        .trend-meta { margin-top: 10px; font-size: 0.85em; opacity: 0.9; }

        .move-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .move-card h3 { margin-bottom: 5px; }
        .move-card .company { font-size: 0.9em; opacity: 0.9; margin-bottom: 8px; }
        .move-card p { font-size: 0.9em; line-height: 1.4; }
        .move-badges { display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
        .move-type-badge {
            background: rgba(255,255,255,0.25);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            text-transform: uppercase;
            font-weight: 600;
        }
        .market-badge {
            background: rgba(0,0,0,0.15);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
        }
        .move-actions {
            margin-top: 10px;
            display: flex;
            gap: 8px;
        }
        .move-action-btn {
            padding: 5px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }
        .move-action-btn.btn-save {
            background: rgba(255,255,255,0.2);
            color: white;
        }
        .move-action-btn.btn-save:hover { background: rgba(255,255,255,0.35); }
        .move-action-btn.btn-save.saved {
            background: rgba(255,255,255,0.9);
            color: #11998e;
        }

        .content-item {
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        .content-item:last-child { border-bottom: none; }
        .content-item h4 { color: #1a1a2e; margin-bottom: 5px; }
        .content-item p { color: #666; font-size: 0.9em; }
        .content-item .meta { font-size: 0.8em; color: #888; margin-top: 5px; }
        .content-item .source {
            background: #e94560;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75em;
        }

        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }

        .badge {
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }

        .empty { text-align: center; padding: 40px; color: #888; }
        .empty .icon { font-size: 3em; margin-bottom: 10px; }

        .error { background: #fee2e2; color: #991b1b; padding: 15px; border-radius: 8px; text-align: center; }

        /* White Space Badge */
        .white-space-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: rgba(255,255,255,0.2);
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }
        .white-space-high { background: rgba(34, 197, 94, 0.3); }
        .white-space-medium { background: rgba(251, 191, 36, 0.3); }
        .white-space-low { background: rgba(239, 68, 68, 0.2); }

        /* Filter Bar */
        .filter-bar {
            display: flex;
            gap: 12px;
            margin-bottom: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filter-select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            font-size: 0.9em;
            min-width: 140px;
            cursor: pointer;
        }
        .filter-select:focus { outline: none; border-color: #667eea; }
        .filter-reset {
            padding: 8px 16px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .filter-reset:hover { background: #e0e0e0; }
        .saved-count {
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-left: auto;
        }

        /* Trend Action Buttons */
        .trend-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }
        .trend-action-btn {
            padding: 6px 12px;
            border: none;
            border-radius: 6px;
            font-size: 0.85em;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: transform 0.1s, opacity 0.1s;
        }
        .trend-action-btn:hover { transform: translateY(-1px); }
        .btn-save { background: rgba(255,255,255,0.9); color: #333; }
        .btn-save.saved { background: #22c55e; color: white; }
        .btn-brand { background: #e94560; color: white; }

        /* Chat */
        .chat-message { margin-bottom: 15px; }
        .chat-message.user { text-align: right; }
        .chat-message.assistant { text-align: left; }
        .chat-bubble {
            display: inline-block;
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.4;
        }
        .chat-message.user .chat-bubble {
            background: #e94560;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .chat-message.assistant .chat-bubble {
            background: white;
            color: #333;
            border: 1px solid #eee;
            border-bottom-left-radius: 4px;
        }
        .chat-confidence {
            font-size: 0.75em;
            margin-top: 4px;
            opacity: 0.7;
        }
        .confidence-high { color: #22c55e; }
        .confidence-medium { color: #f59e0b; }
        .confidence-low { color: #ef4444; }
        .suggestion-chip {
            padding: 8px 16px;
            background: white;
            border: 1px solid #e94560;
            color: #e94560;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .suggestion-chip:hover { background: #e94560; color: white; }
        .chat-typing {
            display: flex;
            gap: 4px;
            padding: 10px 15px;
        }
        .chat-typing span {
            width: 8px;
            height: 8px;
            background: #999;
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        .profile-insights-card h3 { margin-bottom: 15px; }
        .profile-tag {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 15px;
            margin: 3px;
            font-size: 0.9em;
        }
        .profile-section { margin-bottom: 12px; }
        .profile-section-title { font-size: 0.85em; opacity: 0.9; margin-bottom: 6px; }
        .btn-primary {
            padding: 12px 24px;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1em;
        }
        .btn-primary:hover { background: #d63850; }
        .btn-primary:disabled { background: #ccc; cursor: not-allowed; }
        .btn-secondary {
            padding: 12px 24px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1em;
        }
        .btn-secondary:hover { background: #5a6268; }
        .saved-item-card {
            background: #f8f9fa;
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .saved-item-card h4 { margin-bottom: 4px; color: #1a1a2e; }
        .saved-item-meta { font-size: 0.85em; color: #666; }
        .saved-item-actions { display: flex; gap: 8px; }
        .btn-remove {
            padding: 5px 10px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }
        .btn-remove:hover { background: #c82333; }

        /* Modal */
        .modal-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7); z-index: 1000; overflow-y: auto; padding: 20px;
        }
        .modal-overlay.active { display: flex; justify-content: center; align-items: flex-start; }
        .modal-content { background: white; border-radius: 12px; max-width: 700px; width: 100%; margin: 40px auto; position: relative; }
        .modal-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px 12px 0 0; }
        .modal-header.move-header { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
        .modal-header h2 { margin: 0; font-size: 1.4em; line-height: 1.3; }
        .modal-header .meta { opacity: 0.9; margin-top: 8px; font-size: 0.9em; }
        .modal-body { padding: 20px; max-height: 60vh; overflow-y: auto; }
        .modal-section { margin-bottom: 20px; }
        .modal-section:last-child { margin-bottom: 0; }
        .modal-section h3 { color: #1a1a2e; margin-bottom: 10px; font-size: 1.1em; }
        .modal-section p { color: #444; line-height: 1.6; }
        .modal-close { position: absolute; top: 15px; right: 15px; background: rgba(255,255,255,0.2); border: none; color: white; width: 32px; height: 32px; border-radius: 50%; cursor: pointer; font-size: 1.2em; }
        .modal-close:hover { background: rgba(255,255,255,0.3); }
        .source-quote { background: #f8f9fa; border-left: 3px solid #667eea; padding: 12px 15px; margin-bottom: 10px; border-radius: 0 8px 8px 0; font-style: italic; color: #555; font-size: 0.9em; }
        .topic-tag { display: inline-block; background: #e0e7ff; color: #4338ca; padding: 4px 10px; border-radius: 15px; font-size: 0.85em; margin: 3px; }

        .quick-city {
            padding: 5px 12px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            margin: 2px;
        }
        .quick-city:hover { background: #e0e0e0; }

        .desire-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .desire-card h4 { margin-bottom: 8px; }
        .desire-card p { font-size: 0.9em; opacity: 0.95; }
        .desire-meta { margin-top: 10px; font-size: 0.85em; opacity: 0.9; }

        .opportunity-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 8px;
        }

        .concept-card {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: #333;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .concept-card h4 { margin-bottom: 8px; }

        /* Demand Scan */
        .property-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: all 0.2s;
        }
        .property-card:hover { box-shadow: 0 4px 15px rgba(0,0,0,0.12); }
        .property-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }
        .property-card h3 { margin: 0; color: #1a1a2e; font-size: 1.2em; }
        .property-card .location { color: #666; font-size: 0.9em; margin-top: 4px; }

        /* Demand Fit Score Badge */
        .demand-score {
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1em;
        }
        .demand-high { background: #d4edda; color: #155724; }
        .demand-medium { background: #fff3cd; color: #856404; }
        .demand-low { background: #f8d7da; color: #721c24; }

        /* Misalignment Flags */
        .misalignment-flag {
            display: inline-flex;
            align-items: center;
            background: #fee2e2;
            color: #991b1b;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            margin: 3px;
        }
        .misalignment-flag::before { content: "⚠️ "; }

        /* Property Sections */
        .property-section {
            margin-bottom: 15px;
        }
        .property-section-title {
            font-size: 0.85em;
            font-weight: 600;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .gap-item {
            display: inline-block;
            background: #fef3c7;
            color: #92400e;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            margin: 2px;
        }
        .opportunity-item {
            display: flex;
            align-items: center;
            background: #dbeafe;
            color: #1e40af;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9em;
            margin-bottom: 6px;
        }
        .opportunity-item::before { content: "→ "; font-weight: bold; margin-right: 6px; }

        .property-actions {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }
        .property-action-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: all 0.2s;
        }
        .property-action-btn.btn-brand {
            background: #e94560;
            color: white;
        }
        .property-action-btn.btn-brand:hover { background: #d63850; }
        .property-action-btn.btn-save {
            background: #f0f0f0;
            color: #333;
        }
        .property-action-btn.btn-save:hover { background: #e0e0e0; }
        .property-action-btn.btn-save.saved {
            background: #d4edda;
            color: #155724;
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
            <button class="tab" onclick="showTab('demandscan')">🔍 Demand Scan</button>
            <button class="tab" onclick="showTab('content')">Content</button>
            <button class="tab" onclick="showTab('scrapers')">Scrapers</button>
            <button class="tab" onclick="showTab('chat')">💬 Chat</button>
            <button class="tab" onclick="showTab('projects')" id="projects-tab">📁 My Projects</button>
        </div>

        <div id="overview" class="section active">
            <div class="card">
                <h2>📊 Metrics</h2>
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
                <h2>📈 Latest Trend</h2>
                <div id="latest-trend"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
            <div class="card">
                <h2>♟️ Latest Move</h2>
                <div id="latest-move"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="citydesires" class="section">
            <div class="card">
                <h2>🏙️ City Desires</h2>
                <p style="color:#666;margin-bottom:15px;">Discover what travelers are craving in specific destinations. Uncover unmet needs, frustrations, and white-space opportunities from social conversations.</p>
                <p style="color:#666;margin-bottom:15px;">Type a city to discover what travelers want but can't find.</p>
                <div style="display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;">
                    <input type="text" id="city-input" placeholder="City name (e.g., Lisbon)"
                           style="padding:10px 15px;border:1px solid #ddd;border-radius:6px;font-size:1em;flex:1;min-width:150px;">
                    <input type="text" id="country-input" placeholder="Country (optional)"
                           style="padding:10px 15px;border:1px solid #ddd;border-radius:6px;font-size:1em;width:150px;">
                    <button onclick="analyzeCity()" id="analyze-btn"
                            style="padding:10px 20px;background:#e94560;color:white;border:none;border-radius:6px;cursor:pointer;font-size:1em;">
                        Analyze
                    </button>
                </div>
                <div style="margin-bottom:15px;">
                    <span style="color:#888;font-size:0.9em;">Popular: </span>
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
                <h2>📈 Social Pulse Trends</h2>
                <p style="color:#666;margin-bottom:15px;">Track emerging hospitality trends from Reddit, industry news, and social conversations. Discover what's gaining momentum and find white-space opportunities before your competitors.</p>
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
                <h2>♟️ Hotelier Bets</h2>
                <p style="color:#666;margin-bottom:15px;">Monitor strategic moves by hotel companies worldwide. Track launches, acquisitions, repositionings, and partnerships to understand where the industry is heading and identify competitive signals.</p>
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
                    <button onclick="resetMoveFilters()" style="background:#6c757d;">Reset</button>
                    <span id="moves-saved-count" class="saved-count"></span>
                </div>
                <div id="moves-list"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="demandscan" class="section">
            <div class="card">
                <h2>🔍 Demand Scan</h2>
                <p style="color:#666;margin-bottom:15px;">Analyze any hotel website against current demand trends. Get fit scores, experience gaps, and opportunities.</p>

                <!-- URL Input Form -->
                <div style="display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;">
                    <input type="text" id="property-url-input" placeholder="Enter hotel website URL (e.g., https://www.example-hotel.com)"
                           style="padding:12px 15px;border:1px solid #ddd;border-radius:6px;font-size:1em;flex:1;min-width:250px;">
                    <button onclick="scanProperty()" id="scan-btn"
                            style="padding:12px 24px;background:#e94560;color:white;border:none;border-radius:6px;cursor:pointer;font-size:1em;font-weight:600;">
                        Scan Property
                    </button>
                </div>

                <!-- Scan Status -->
                <div id="scan-status" style="display:none;margin-bottom:20px;padding:15px;border-radius:8px;"></div>

                <!-- Previously Scanned Properties -->
                <h3 style="margin:20px 0 15px;color:#1a1a2e;">Previously Scanned Properties</h3>
                <div id="properties-list"><div class="empty"><div class="icon">🏨</div>No properties scanned yet. Enter a URL above to analyze a property.</div></div>
            </div>
        </div>

        <div id="content" class="section">
            <div class="card">
                <h2>📰 Recent Content</h2>
                <p style="color:#666;margin-bottom:15px;">Browse the latest scraped articles, social posts, and news from our 12+ hospitality sources. This raw content feeds our trend detection and move extraction engines.</p>
                <div id="content-list"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="scrapers" class="section">
            <div class="card">
                <h2>🔧 Scraper Status</h2>
                <p style="color:#666;margin-bottom:15px;">Monitor the health and activity of our data collection system. Our POMDP-driven scheduler intelligently prioritizes sources based on expected information gain.</p>
                <div id="scrapers-list"><div class="empty"><div class="icon">⏳</div>Loading...</div></div>
            </div>
        </div>

        <div id="chat" class="section">
            <div class="card">
                <h2>💬 BrandClave Chat</h2>
                <p style="margin-bottom:15px;color:#666;">Your AI-powered hospitality intelligence assistant. Ask about market trends, explore opportunities in specific cities, or get help ideating a brand concept with RAG-powered insights from our data.</p>

                <div id="chat-messages" style="min-height:300px;max-height:500px;overflow-y:auto;border:1px solid #eee;border-radius:8px;padding:15px;margin-bottom:15px;background:#fafafa;">
                    <div class="chat-welcome">
                        <div style="text-align:center;padding:40px 20px;">
                            <div style="font-size:3em;margin-bottom:15px;">🤖</div>
                            <h3 style="color:#1a1a2e;margin-bottom:10px;">Hello! I'm your hospitality intelligence assistant.</h3>
                            <p style="color:#666;">Try asking me about:</p>
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
                           style="flex:1;padding:12px 15px;border:1px solid #ddd;border-radius:8px;font-size:1em;"
                           onkeypress="if(event.key==='Enter')sendMessage()">
                    <button onclick="sendMessage()" style="padding:12px 25px;background:#e94560;color:white;border:none;border-radius:8px;cursor:pointer;font-weight:600;">Send</button>
                </div>

                <div id="chat-state" style="margin-top:10px;font-size:0.85em;color:#888;"></div>
            </div>
        </div>

        <div id="projects" class="section">
            <div class="card">
                <h2>📁 My Projects</h2>
                <p style="margin-bottom:15px;color:#666;">Save trends and strategic moves to build a research profile. Your saved items inform brand generation, helping BrandClave understand your interests and create more relevant concepts.</p>

                <!-- Profile Insights -->
                <div id="profile-insights" class="profile-insights-card">
                    <h3>🎯 Your Interest Profile</h3>
                    <div id="profile-content">
                        <div class="empty"><div class="icon">💡</div>Save trends and moves to build your profile</div>
                    </div>
                </div>

                <!-- Actions -->
                <div style="display:flex;gap:10px;margin:20px 0;">
                    <button onclick="buildBrandFromProfile()" class="btn-primary" id="build-from-profile-btn" disabled>
                        🚀 Build Brand from Profile
                    </button>
                    <button onclick="clearAllSaved()" class="btn-secondary">
                        🗑️ Clear All
                    </button>
                </div>

                <!-- Saved Trends -->
                <div style="margin-top:20px;">
                    <h3 style="margin-bottom:10px;">📊 Saved Trends <span id="saved-trends-count" style="font-weight:normal;color:#666;"></span></h3>
                    <div id="saved-trends-list">
                        <div class="empty"><div class="icon">📊</div>No saved trends yet</div>
                    </div>
                </div>

                <!-- Saved Moves -->
                <div style="margin-top:20px;">
                    <h3 style="margin-bottom:10px;">♟️ Saved Moves <span id="saved-moves-count" style="font-weight:normal;color:#666;"></span></h3>
                    <div id="saved-moves-list">
                        <div class="empty"><div class="icon">♟️</div>No saved moves yet</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Store data globally for modal access
        let allTrends = [];
        let allMoves = [];

        function openTrendModal(i) {
            const t = allTrends[i];
            if (!t) return;
            document.getElementById('modal-header').className = 'modal-header';
            document.getElementById('modal-title').textContent = t.name || t.trend_name || 'Unnamed Trend';
            const score = t.strength_score ? Math.round(t.strength_score * 100) + '%' : 'N/A';
            const whiteSpace = t.white_space_score ? Math.round(t.white_space_score * 100) : 0;
            document.getElementById('modal-meta').innerHTML = 'Strength: ' + score + ' | White Space: ' + whiteSpace + '% | ' + (t.volume || 0) + ' sources';
            let h = '';
            if (t.description) h += '<div class="modal-section"><h3>Description</h3><p>' + t.description + '</p></div>';
            if (t.why_it_matters) h += '<div class="modal-section"><h3>Why It Matters</h3><p>' + t.why_it_matters + '</p></div>';
            // White Space Analysis section
            if (t.white_space_score !== undefined) {
                const ws = Math.round((t.white_space_score || 0) * 100);
                const wsClass = ws >= 70 ? 'white-space-high' : ws >= 40 ? 'white-space-medium' : 'white-space-low';
                const wsLabel = ws >= 70 ? 'High Opportunity - underserved market' : ws >= 40 ? 'Moderate Opportunity' : 'Low - competitive market';
                h += '<div class="modal-section"><h3>🎯 White Space Analysis</h3>';
                h += '<p><span class="white-space-badge ' + wsClass + '" style="font-size:1em;padding:6px 12px;">' + ws + '% - ' + wsLabel + '</span></p>';
                if (t.region) h += '<p style="margin-top:10px;"><strong>Region:</strong> ' + t.region + '</p>';
                if (t.audience_segment) h += '<p><strong>Segment:</strong> ' + t.audience_segment + '</p>';
                h += '</div>';
            }
            if (t.topics && t.topics.length) h += '<div class="modal-section"><h3>Topics</h3><div>' + t.topics.map(x => '<span class="topic-tag">' + x + '</span>').join('') + '</div></div>';
            if (t.sample_quotes && t.sample_quotes.length) { h += '<div class="modal-section"><h3>Source Quotes</h3>'; t.sample_quotes.forEach(q => { h += '<div class="source-quote">"' + q + '"</div>'; }); h += '</div>'; }
            document.getElementById('modal-body').innerHTML = h || '<p>No additional details.</p>';
            document.getElementById('modal-overlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function openMoveModal(i) {
            const m = allMoves[i];
            if (!m) return;
            document.getElementById('modal-header').className = 'modal-header move-header';
            document.getElementById('modal-title').textContent = m.title || 'Untitled';

            // Enhanced meta with market and investment
            let metaParts = [m.company || 'Unknown', m.move_type ? m.move_type.replace('_', ' ') : 'Move'];
            if (m.market) metaParts.push('📍 ' + m.market);
            if (m.investment_amount) metaParts.push('💰 ' + m.investment_amount);
            document.getElementById('modal-meta').innerHTML = metaParts.join(' | ');

            let h = '';

            // Summary
            if (m.summary) h += '<div class="modal-section"><h3>Summary</h3><p>' + m.summary + '</p></div>';

            // Why It Matters
            if (m.why_it_matters) h += '<div class="modal-section"><h3>Why It Matters</h3><p>' + m.why_it_matters + '</p></div>';

            // Strategic Implications
            if (m.strategic_implications && m.strategic_implications.length > 0) {
                h += '<div class="modal-section"><h3>Strategic Implications</h3><ul style="margin:0;padding-left:20px;">';
                m.strategic_implications.forEach(imp => {
                    h += '<li style="margin-bottom:6px;">' + imp + '</li>';
                });
                h += '</ul></div>';
            }

            // Competitive Impact
            if (m.competitive_impact) {
                h += '<div class="modal-section"><h3>Competitive Impact</h3><p>' + m.competitive_impact + '</p></div>';
            }

            // Source
            if (m.source_url) h += '<div class="modal-section"><h3>Source</h3><p><a href="' + m.source_url + '" target="_blank" style="color:#667eea;">' + (m.source_name || 'View article') + '</a></p></div>';

            document.getElementById('modal-body').innerHTML = h || '<p>No additional details.</p>';
            document.getElementById('modal-overlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeModal(e) {
            if (e && e.target !== e.currentTarget) return;
            document.getElementById('modal-overlay').classList.remove('active');
            document.body.style.overflow = '';
        }
        document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });


        function showTab(tabId) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
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
                const [metricsRes, trendsRes, movesRes, contentRes, scrapersRes] = await Promise.all([
                    fetch('/api/monitoring/metrics'),
                    fetch('/api/social-pulse?limit=10'),
                    fetch('/api/hotelier-bets?limit=10'),
                    fetch('/api/monitoring/content?limit=30'),
                    fetch('/api/monitoring/scrapers')
                ]);

                const metrics = await metricsRes.json();
                const trendsData = await trendsRes.json();
                const movesData = await movesRes.json();
                const contentData = await contentRes.json();
                const scrapers = await scrapersRes.json();

                // Update metrics
                document.getElementById('m-content').textContent = metrics.total_content?.toLocaleString() || '0';
                document.getElementById('m-processed').textContent = metrics.processed_content?.toLocaleString() || '0';
                document.getElementById('m-trends').textContent = metrics.trends_count || '0';
                document.getElementById('m-moves').textContent = metrics.moves_count || '0';

                // Render trends
                allTrends = trendsData.trends || [];
                const trends = allTrends;
                if (trends.length > 0) {
                    document.getElementById('latest-trend').innerHTML = renderTrend(trends[0], 0);
                    document.getElementById('trends-list').innerHTML = trends.map((t, i) => renderTrend(t, i)).join('');
                } else {
                    document.getElementById('latest-trend').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends yet. Run POPULATE_DATA.bat</div>';
                    document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends yet</div>';
                }

                // Render moves
                allMoves = movesData.moves || [];
                const moves = allMoves;
                if (moves.length > 0) {
                    document.getElementById('latest-move').innerHTML = renderMove(moves[0], 0);
                    document.getElementById('moves-list').innerHTML = moves.map((m, i) => renderMove(m, i)).join('');
                } else {
                    document.getElementById('latest-move').innerHTML = '<div class="empty"><div class="icon">♟️</div>No moves yet. Run POPULATE_DATA.bat</div>';
                    document.getElementById('moves-list').innerHTML = '<div class="empty"><div class="icon">♟️</div>No moves yet</div>';
                }

                // Render content
                const content = contentData.items || [];
                if (content.length > 0) {
                    document.getElementById('content-list').innerHTML = content.map(renderContent).join('');
                } else {
                    document.getElementById('content-list').innerHTML = '<div class="empty"><div class="icon">📰</div>No content yet</div>';
                }

                // Render scrapers
                if (scrapers.length > 0) {
                    document.getElementById('scrapers-list').innerHTML = `
                        <table>
                            <thead>
                                <tr>
                                    <th>Source</th>
                                    <th>Total</th>
                                    <th>Last Run</th>
                                    <th>New Items</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${scrapers.map(s => `
                                    <tr>
                                        <td><strong>${s.source}</strong></td>
                                        <td>${s.total_items.toLocaleString()}</td>
                                        <td>${s.last_run_at ? new Date(s.last_run_at).toLocaleString() : 'Never'}</td>
                                        <td>${s.last_run_items || 0}</td>
                                        <td><span class="badge badge-${s.last_run_status === 'completed' ? 'success' : 'warning'}">${s.last_run_status || 'N/A'}</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
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
            const name = t.name || t.trend_name || 'Unnamed Trend';
            const score = t.strength_score ? Math.round(t.strength_score * 100) + '%' : 'N/A';
            const whiteSpace = t.white_space_score ? Math.round(t.white_space_score * 100) : 0;
            const wsClass = whiteSpace >= 70 ? 'white-space-high' : whiteSpace >= 40 ? 'white-space-medium' : 'white-space-low';
            const wsLabel = whiteSpace >= 70 ? 'High Opportunity' : whiteSpace >= 40 ? 'Moderate' : 'Low';
            const isSaved = isProjectSaved(t.id);

            return `
                <div class="trend-card" data-trend-id="${t.id}">
                    <div onclick="openTrendModal(${idx})" style="cursor:pointer;">
                        <h3>${truncate(name, 60)}</h3>
                        <p>${truncate(t.description || t.why_it_matters || '', 200)}</p>
                        <div class="trend-meta">
                            <span class="white-space-badge ${wsClass}">🎯 ${whiteSpace}% ${wsLabel}</span>
                            | Strength: ${score} | ${t.volume || 0} sources
                        </div>
                    </div>
                    <div class="trend-actions">
                        <button class="trend-action-btn btn-save ${isSaved ? 'saved' : ''}"
                                onclick="event.stopPropagation(); toggleSaveProject(${idx})">
                            ${isSaved ? '✓ Saved' : '💾 Save'}
                        </button>
                        <button class="trend-action-btn btn-brand"
                                onclick="event.stopPropagation(); turnIntoBrand(${idx})">
                            🚀 Build a Brand
                        </button>
                    </div>
                </div>
            `;
        }

        function renderMove(m, idx) {
            const isSaved = isMoveSaved(m.id);
            const moveTypeBadge = m.move_type ? `<span class="move-type-badge">${m.move_type.replace('_', ' ')}</span>` : '';
            const marketBadge = m.market ? `<span class="market-badge">📍 ${m.market}</span>` : '';

            return `
                <div class="move-card" onclick="openMoveModal(${idx})">
                    <div class="move-badges">
                        ${moveTypeBadge}
                        ${marketBadge}
                    </div>
                    <h3>${truncate(m.title || 'Untitled', 60)}</h3>
                    <div class="company">${m.company || 'Unknown'}</div>
                    <p>${truncate(m.summary || m.why_it_matters || '', 180)}</p>
                    <div class="move-actions">
                        <button class="move-action-btn btn-save ${isSaved ? 'saved' : ''}"
                                onclick="event.stopPropagation(); toggleSaveMove(${idx})">
                            ${isSaved ? '✓ Saved' : '💾 Save'}
                        </button>
                    </div>
                </div>
            `;
        }

        function renderContent(c) {
            return `
                <div class="content-item">
                    <h4>${truncate(c.title || 'Untitled', 70)}</h4>
                    <p>${truncate(c.content || '', 150)}</p>
                    <div class="meta">
                        <span class="source">${c.source || 'unknown'}</span>
                        ${c.published_at ? ' • ' + new Date(c.published_at).toLocaleDateString() : ''}
                    </div>
                </div>
            `;
        }

        // =============================================
        // Demand Scan Functions
        // =============================================

        let allProperties = [];

        async function scanProperty() {
            const urlInput = document.getElementById('property-url-input');
            const url = urlInput.value.trim();

            if (!url) {
                showScanStatus('error', 'Please enter a valid URL');
                return;
            }

            // Validate URL format
            try {
                new URL(url);
            } catch {
                showScanStatus('error', 'Invalid URL format. Please enter a complete URL including https://');
                return;
            }

            const scanBtn = document.getElementById('scan-btn');
            scanBtn.disabled = true;
            scanBtn.textContent = 'Scanning...';
            showScanStatus('info', 'Analyzing property website... This may take 30-60 seconds.');

            try {
                const response = await fetch('/api/demand-scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });

                const data = await response.json();

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
            const statusEl = document.getElementById('scan-status');
            const colors = {
                error: { bg: '#fee2e2', color: '#991b1b' },
                success: { bg: '#d4edda', color: '#155724' },
                warning: { bg: '#fff3cd', color: '#856404' },
                info: { bg: '#e0e7ff', color: '#4338ca' }
            };
            const style = colors[type] || colors.info;
            statusEl.style.display = 'block';
            statusEl.style.background = style.bg;
            statusEl.style.color = style.color;
            statusEl.textContent = message;

            if (type !== 'info') {
                setTimeout(() => { statusEl.style.display = 'none'; }, 5000);
            }
        }

        async function loadScannedProperties() {
            try {
                const response = await fetch('/api/demand-scan?limit=20');
                const data = await response.json();
                allProperties = data.properties || [];

                const container = document.getElementById('properties-list');
                if (allProperties.length > 0) {
                    container.innerHTML = allProperties.map((p, i) => renderPropertyCard(p, i)).join('');
                } else {
                    container.innerHTML = '<div class="empty"><div class="icon">🏨</div>No properties scanned yet. Enter a URL above to analyze a property.</div>';
                }
            } catch (err) {
                console.error('Load properties error:', err);
            }
        }

        function renderPropertyCard(p, idx) {
            // Calculate demand score as 0-100
            const score = p.demand_fit_score ? Math.round(p.demand_fit_score * 100) : 0;
            const scoreClass = score >= 70 ? 'demand-high' : score >= 40 ? 'demand-medium' : 'demand-low';
            const scoreLabel = score >= 70 ? 'High Fit' : score >= 40 ? 'Moderate Fit' : 'Low Fit';

            // Experience gaps (top 3)
            const gaps = (p.experience_gaps || []).slice(0, 3);
            const gapsHtml = gaps.length > 0
                ? gaps.map(g => `<span class="gap-item">${truncate(g.split(' (')[0], 30)}</span>`).join('')
                : '<span style="color:#888;font-size:0.85em;">No major gaps identified</span>';

            // Opportunity lanes (top 2)
            const opportunities = (p.opportunity_lanes || []).slice(0, 2);
            const oppsHtml = opportunities.length > 0
                ? opportunities.map(o => `<div class="opportunity-item">${truncate(o, 80)}</div>`).join('')
                : '<span style="color:#888;font-size:0.85em;">No opportunities identified</span>';

            // Misalignment flags
            const flags = p.positioning_misalignment_flags || [];
            const flagsHtml = flags.length > 0
                ? `<div class="property-section">
                    <div class="property-section-title">Positioning Issues</div>
                    ${flags.map(f => `<span class="misalignment-flag">${truncate(f.split(':')[1] || f, 50)}</span>`).join('')}
                   </div>`
                : '';

            // Property themes
            const themes = (p.themes || []).slice(0, 4);
            const themesHtml = themes.length > 0
                ? themes.map(t => `<span class="topic-tag">${t}</span>`).join('')
                : '';

            return `
                <div class="property-card" data-property-id="${p.id}">
                    <div class="property-card-header">
                        <div>
                            <h3>${p.name || 'Unnamed Property'}</h3>
                            <div class="location">${p.location || p.region || 'Location unknown'}</div>
                            <div style="margin-top:8px;">${themesHtml}</div>
                        </div>
                        <div class="demand-score ${scoreClass}">
                            ${score}% ${scoreLabel}
                        </div>
                    </div>

                    ${flagsHtml}

                    <div class="property-section">
                        <div class="property-section-title">Experience Gaps</div>
                        ${gapsHtml}
                    </div>

                    <div class="property-section">
                        <div class="property-section-title">Opportunity Lanes</div>
                        ${oppsHtml}
                    </div>

                    <div class="property-actions">
                        <button class="property-action-btn btn-brand" onclick="sendPropertyToBuildBrand(${idx})">
                            🚀 Build a Brand
                        </button>
                        <button class="property-action-btn btn-save" onclick="savePropertyToProject(${idx})">
                            💾 Save to Project
                        </button>
                        <a href="${p.url}" target="_blank" class="property-action-btn btn-save" style="text-decoration:none;">
                            🔗 Visit Site
                        </a>
                    </div>
                </div>
            `;
        }

        function sendPropertyToBuildBrand(idx) {
            const p = allProperties[idx];
            if (!p) return;

            // Store property data for Build a Brand page
            const brandData = {
                type: 'property',
                property_name: p.name,
                location: p.location || '',
                segment: p.price_segment || (p.themes && p.themes[0]) || '',
                context: `Property analysis of ${p.name}. Demand fit: ${Math.round((p.demand_fit_score || 0) * 100)}%`,
                gaps: p.experience_gaps || [],
                opportunities: p.opportunity_lanes || [],
                themes: p.themes || []
            };
            localStorage.setItem('brandclave_brand_prefill', JSON.stringify(brandData));

            // Navigate to Build a Brand page
            window.location.href = '/monitoring/build-a-brand';
        }

        function savePropertyToProject(idx) {
            const p = allProperties[idx];
            if (!p) return;

            // Get existing saved properties
            const saved = JSON.parse(localStorage.getItem('brandclave_saved_properties') || '[]');

            // Check if already saved
            const existingIdx = saved.findIndex(s => s.id === p.id);
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
        document.addEventListener('DOMContentLoaded', () => {
            const urlInput = document.getElementById('property-url-input');
            if (urlInput) {
                urlInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') scanProperty();
                });
            }
        });

        // =============================================
        // Filter Functions
        // =============================================
        let currentFilters = { region: '', audience: '', time: '' };

        async function loadFilterOptions() {
            try {
                const [regionsRes, audiencesRes] = await Promise.all([
                    fetch('/api/social-pulse/regions'),
                    fetch('/api/social-pulse/audiences')
                ]);

                const regionsData = await regionsRes.json();
                const audiencesData = await audiencesRes.json();

                // Populate region dropdown
                const regionSelect = document.getElementById('filter-region');
                regionSelect.innerHTML = '<option value="">All Regions</option>';
                (regionsData.regions || []).forEach(r => {
                    if (r.region) {
                        regionSelect.innerHTML += '<option value="' + r.region + '">' + r.region + ' (' + r.count + ')</option>';
                    }
                });

                // Populate audience dropdown
                const audienceSelect = document.getElementById('filter-audience');
                audienceSelect.innerHTML = '<option value="">All Segments</option>';
                (audiencesData.audiences || []).forEach(a => {
                    if (a.segment) {
                        audienceSelect.innerHTML += '<option value="' + a.segment + '">' + a.segment + ' (' + a.count + ')</option>';
                    }
                });
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
            const params = new URLSearchParams({ limit: '20' });
            if (currentFilters.region) params.append('region', currentFilters.region);
            if (currentFilters.audience) params.append('audience', currentFilters.audience);

            document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">⏳</div>Loading...</div>';

            try {
                const res = await fetch('/api/social-pulse?' + params.toString());
                const data = await res.json();

                let trends = data.trends || [];

                // Client-side time filtering
                if (currentFilters.time) {
                    const daysAgo = parseInt(currentFilters.time);
                    const cutoff = new Date();
                    cutoff.setDate(cutoff.getDate() - daysAgo);
                    trends = trends.filter(t => {
                        if (!t.first_seen) return true;
                        return new Date(t.first_seen) >= cutoff;
                    });
                }

                allTrends = trends;

                if (trends.length > 0) {
                    document.getElementById('trends-list').innerHTML = trends.map((t, i) => renderTrend(t, i)).join('');
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
        let moveFilters = { company: '', move_type: '', market: '' };

        async function loadMoveFilterOptions() {
            try {
                const [companiesRes, moveTypesRes, marketsRes] = await Promise.all([
                    fetch('/api/hotelier-bets/companies'),
                    fetch('/api/hotelier-bets/move-types'),
                    fetch('/api/hotelier-bets/markets')
                ]);

                const companiesData = await companiesRes.json();
                const moveTypesData = await moveTypesRes.json();
                const marketsData = await marketsRes.json();

                // Populate company dropdown
                const companySelect = document.getElementById('filter-company');
                companySelect.innerHTML = '<option value="">All Companies</option>';
                (companiesData.companies || []).forEach(c => {
                    companySelect.innerHTML += '<option value="' + c + '">' + c + '</option>';
                });

                // Populate move type dropdown
                const moveTypeSelect = document.getElementById('filter-move-type');
                moveTypeSelect.innerHTML = '<option value="">All Move Types</option>';
                (moveTypesData.move_types || []).forEach(mt => {
                    // Handle both object format {type, label} and string format
                    const value = typeof mt === 'object' ? mt.type : mt;
                    const display = typeof mt === 'object' ? mt.label : mt.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                    if (value) moveTypeSelect.innerHTML += '<option value="' + value + '">' + display + '</option>';
                });

                // Populate market dropdown
                const marketSelect = document.getElementById('filter-market');
                marketSelect.innerHTML = '<option value="">All Markets</option>';
                (marketsData.markets || []).forEach(m => {
                    if (m) marketSelect.innerHTML += '<option value="' + m + '">' + m + '</option>';
                });
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
            const params = new URLSearchParams({ limit: '20' });
            if (moveFilters.company) params.append('company', moveFilters.company);
            if (moveFilters.move_type) params.append('move_type', moveFilters.move_type);
            if (moveFilters.market) params.append('market', moveFilters.market);

            document.getElementById('moves-list').innerHTML = '<div class="empty"><div class="icon">⏳</div>Loading...</div>';

            try {
                const res = await fetch('/api/hotelier-bets?' + params.toString());
                const data = await res.json();

                allMoves = data.moves || [];

                if (allMoves.length > 0) {
                    document.getElementById('moves-list').innerHTML = allMoves.map((m, i) => renderMove(m, i)).join('');
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
        const STORAGE_KEY = 'brandclave_saved_trends';

        function getSavedProjects() {
            try {
                const data = localStorage.getItem(STORAGE_KEY);
                return data ? JSON.parse(data) : [];
            } catch (e) {
                console.error('Error reading saved projects:', e);
                return [];
            }
        }

        function saveProject(trend) {
            const saved = getSavedProjects();
            if (saved.some(s => s.id === trend.id)) return false;

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
            const saved = getSavedProjects();
            const filtered = saved.filter(s => s.id !== trendId);
            localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
        }

        function isProjectSaved(trendId) {
            return getSavedProjects().some(s => s.id === trendId);
        }

        function toggleSaveProject(idx) {
            const trend = allTrends[idx];
            if (!trend) return;

            if (isProjectSaved(trend.id)) {
                removeProject(trend.id);
            } else {
                saveProject(trend);
            }

            // Re-render the trends list
            document.getElementById('trends-list').innerHTML = allTrends.map((t, i) => renderTrend(t, i)).join('');
            updateSavedCount();
            renderMyProjects(); // Auto-update My Projects tab
        }

        function updateSavedCount() {
            const count = getSavedProjects().length;
            const countEl = document.getElementById('saved-count');
            if (countEl) {
                countEl.textContent = count > 0 ? count + ' saved' : '';
            }
        }

        // =============================================
        // LocalStorage Save Moves Functions
        // =============================================
        const MOVES_STORAGE_KEY = 'brandclave_saved_moves';

        function getSavedMoves() {
            try {
                const data = localStorage.getItem(MOVES_STORAGE_KEY);
                return data ? JSON.parse(data) : [];
            } catch (e) {
                console.error('Error reading saved moves:', e);
                return [];
            }
        }

        function saveMove(move) {
            const saved = getSavedMoves();
            if (saved.some(s => s.id === move.id)) return false;

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
            const saved = getSavedMoves();
            const filtered = saved.filter(s => s.id !== moveId);
            localStorage.setItem(MOVES_STORAGE_KEY, JSON.stringify(filtered));
        }

        function isMoveSaved(moveId) {
            return getSavedMoves().some(s => s.id === moveId);
        }

        function toggleSaveMove(idx) {
            const move = allMoves[idx];
            if (!move) return;

            if (isMoveSaved(move.id)) {
                removeMove(move.id);
            } else {
                saveMove(move);
            }

            // Re-render the moves list
            document.getElementById('moves-list').innerHTML = allMoves.map((m, i) => renderMove(m, i)).join('');
            updateMovesSavedCount();
            renderMyProjects(); // Auto-update My Projects tab
        }

        function updateMovesSavedCount() {
            const count = getSavedMoves().length;
            const countEl = document.getElementById('moves-saved-count');
            if (countEl) {
                countEl.textContent = count > 0 ? count + ' saved' : '';
            }
        }

        // =============================================
        // My Projects Functions
        // =============================================
        function renderMyProjects() {
            const savedTrends = getSavedProjects();
            const savedMoves = getSavedMoves();

            // Update counts
            document.getElementById('saved-trends-count').textContent = savedTrends.length > 0 ? `(${savedTrends.length})` : '';
            document.getElementById('saved-moves-count').textContent = savedMoves.length > 0 ? `(${savedMoves.length})` : '';

            // Render saved trends
            const trendsListEl = document.getElementById('saved-trends-list');
            if (savedTrends.length > 0) {
                trendsListEl.innerHTML = savedTrends.map(t => `
                    <div class="saved-item-card">
                        <div>
                            <h4>${t.name || 'Unnamed Trend'}</h4>
                            <div class="saved-item-meta">
                                ${t.region ? t.region + ' • ' : ''}${t.audience_segment || 'General'}
                                ${t.white_space_score ? ' • White Space: ' + (t.white_space_score * 100).toFixed(0) + '%' : ''}
                            </div>
                        </div>
                        <div class="saved-item-actions">
                            <button class="btn-remove" onclick="removeSavedTrend('${t.id}')">Remove</button>
                        </div>
                    </div>
                `).join('');
            } else {
                trendsListEl.innerHTML = '<div class="empty"><div class="icon">📊</div>No saved trends yet</div>';
            }

            // Render saved moves
            const movesListEl = document.getElementById('saved-moves-list');
            if (savedMoves.length > 0) {
                movesListEl.innerHTML = savedMoves.map(m => `
                    <div class="saved-item-card">
                        <div>
                            <h4>${m.title || 'Unnamed Move'}</h4>
                            <div class="saved-item-meta">
                                ${m.company || 'Unknown'} • ${m.move_type ? m.move_type.replace('_', ' ') : 'Move'}
                                ${m.market ? ' • ' + m.market : ''}
                            </div>
                        </div>
                        <div class="saved-item-actions">
                            <button class="btn-remove" onclick="removeSavedMove('${m.id}')">Remove</button>
                        </div>
                    </div>
                `).join('');
            } else {
                movesListEl.innerHTML = '<div class="empty"><div class="icon">♟️</div>No saved moves yet</div>';
            }

            // Update profile insights
            updateProfileInsights(savedTrends, savedMoves);

            // Enable/disable build button
            const buildBtn = document.getElementById('build-from-profile-btn');
            buildBtn.disabled = (savedTrends.length + savedMoves.length) === 0;
        }

        function updateProfileInsights(trends, moves) {
            const profileEl = document.getElementById('profile-content');

            if (trends.length === 0 && moves.length === 0) {
                profileEl.innerHTML = '<div class="empty" style="color:rgba(255,255,255,0.8);"><div class="icon">💡</div>Save trends and moves to build your profile</div>';
                return;
            }

            // Analyze patterns
            const regions = {};
            const segments = {};
            const topics = {};
            const companies = {};
            const moveTypes = {};
            const markets = {};

            // From trends
            trends.forEach(t => {
                if (t.region) regions[t.region] = (regions[t.region] || 0) + 1;
                if (t.audience_segment) segments[t.audience_segment] = (segments[t.audience_segment] || 0) + 1;
                (t.topics || []).forEach(topic => {
                    topics[topic] = (topics[topic] || 0) + 1;
                });
            });

            // From moves
            moves.forEach(m => {
                if (m.company) companies[m.company] = (companies[m.company] || 0) + 1;
                if (m.move_type) moveTypes[m.move_type] = (moveTypes[m.move_type] || 0) + 1;
                if (m.market) markets[m.market] = (markets[m.market] || 0) + 1;
            });

            // Sort by frequency and take top items
            const topRegions = Object.entries(regions).sort((a, b) => b[1] - a[1]).slice(0, 3);
            const topSegments = Object.entries(segments).sort((a, b) => b[1] - a[1]).slice(0, 3);
            const topTopics = Object.entries(topics).sort((a, b) => b[1] - a[1]).slice(0, 5);
            const topCompanies = Object.entries(companies).sort((a, b) => b[1] - a[1]).slice(0, 3);
            const topMoveTypes = Object.entries(moveTypes).sort((a, b) => b[1] - a[1]).slice(0, 3);
            const topMarkets = Object.entries(markets).sort((a, b) => b[1] - a[1]).slice(0, 3);

            let html = '';

            if (topRegions.length > 0 || topMarkets.length > 0) {
                const allLocations = [...topRegions, ...topMarkets].slice(0, 4);
                html += `<div class="profile-section">
                    <div class="profile-section-title">📍 Locations of Interest</div>
                    ${allLocations.map(([loc]) => `<span class="profile-tag">${loc}</span>`).join('')}
                </div>`;
            }

            if (topSegments.length > 0) {
                html += `<div class="profile-section">
                    <div class="profile-section-title">👥 Target Segments</div>
                    ${topSegments.map(([seg]) => `<span class="profile-tag">${seg}</span>`).join('')}
                </div>`;
            }

            if (topTopics.length > 0) {
                html += `<div class="profile-section">
                    <div class="profile-section-title">🔥 Key Themes</div>
                    ${topTopics.map(([topic]) => `<span class="profile-tag">${topic}</span>`).join('')}
                </div>`;
            }

            if (topCompanies.length > 0) {
                html += `<div class="profile-section">
                    <div class="profile-section-title">🏨 Companies Watched</div>
                    ${topCompanies.map(([co]) => `<span class="profile-tag">${co}</span>`).join('')}
                </div>`;
            }

            if (topMoveTypes.length > 0) {
                html += `<div class="profile-section">
                    <div class="profile-section-title">♟️ Move Types</div>
                    ${topMoveTypes.map(([mt]) => `<span class="profile-tag">${mt.replace('_', ' ')}</span>`).join('')}
                </div>`;
            }

            profileEl.innerHTML = html || '<div style="opacity:0.8;">Collecting insights...</div>';
        }

        function removeSavedTrend(trendId) {
            removeProject(trendId);
            renderMyProjects();
            updateSavedCount();
            // Re-render trends if visible
            if (allTrends.length > 0) {
                document.getElementById('trends-list').innerHTML = allTrends.map((t, i) => renderTrend(t, i)).join('');
            }
        }

        function removeSavedMove(moveId) {
            removeMove(moveId);
            renderMyProjects();
            updateMovesSavedCount();
            // Re-render moves if visible
            if (allMoves.length > 0) {
                document.getElementById('moves-list').innerHTML = allMoves.map((m, i) => renderMove(m, i)).join('');
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
                document.getElementById('trends-list').innerHTML = allTrends.map((t, i) => renderTrend(t, i)).join('');
            }
            if (allMoves.length > 0) {
                document.getElementById('moves-list').innerHTML = allMoves.map((m, i) => renderMove(m, i)).join('');
            }
        }

        function buildBrandFromProfile() {
            const savedTrends = getSavedProjects();
            const savedMoves = getSavedMoves();

            if (savedTrends.length === 0 && savedMoves.length === 0) {
                alert('Save some trends or moves first to build your profile.');
                return;
            }

            // Build profile data for brand generation
            const profileData = {
                trends: savedTrends,
                moves: savedMoves,
                // Extract key insights
                regions: [...new Set(savedTrends.map(t => t.region).filter(Boolean))],
                segments: [...new Set(savedTrends.map(t => t.audience_segment).filter(Boolean))],
                topics: [...new Set(savedTrends.flatMap(t => t.topics || []))],
                companies: [...new Set(savedMoves.map(m => m.company).filter(Boolean))],
                markets: [...new Set(savedMoves.map(m => m.market).filter(Boolean))],
                move_types: [...new Set(savedMoves.map(m => m.move_type).filter(Boolean))],
            };

            // Store for Build a Brand page
            sessionStorage.setItem('brandclave_profile_data', JSON.stringify(profileData));
            sessionStorage.setItem('brandclave_brand_input', JSON.stringify({
                from_profile: true,
                initial_region: profileData.regions[0] || '',
                initial_segment: profileData.segments[0] || 'lifestyle',
                topics: profileData.topics.slice(0, 5),
            }));

            // Navigate to Build a Brand
            window.location.href = '/api/monitoring/build-a-brand';
        }

        // =============================================
        // Turn Into Brand Function
        // =============================================
        function turnIntoBrand(idx) {
            const trend = allTrends[idx];
            if (!trend) return;

            // Store trend data for the Build a Brand page
            const brandInput = {
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

        async function analyzeCity() {
            const city = document.getElementById('city-input').value.trim();
            const country = document.getElementById('country-input').value.trim();

            if (!city) {
                alert('Please enter a city name');
                return;
            }

            const btn = document.getElementById('analyze-btn');
            const resultsDiv = document.getElementById('city-results');

            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            resultsDiv.innerHTML = '<div class="empty"><div class="icon">⏳</div>Analyzing ' + city + '... This may take 60-120 seconds (using semantic clustering).</div>';

            try {
                // Use adaptive endpoint with semantic clustering for better results
                const response = await fetch('/api/city-desires/adaptive', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ city, country })
                });

                if (!response.ok) {
                    throw new Error('Analysis failed: ' + response.status);
                }

                const data = await response.json();
                renderCityResults(data);

            } catch (err) {
                resultsDiv.innerHTML = '<div class="error">Analysis failed: ' + err.message + '</div>';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Analyze';
            }
        }

        function renderCityResults(data) {
            const resultsDiv = document.getElementById('city-results');

            // Format sources summary
            const sourcesSummary = data.sources_summary || {};
            const sourcesHtml = Object.entries(sourcesSummary)
                .sort((a, b) => b[1] - a[1])
                .map(([src, count]) => `<span style="background:#e8f4fd;padding:3px 8px;border-radius:12px;font-size:0.85em;margin-right:6px;">${src}: ${count}</span>`)
                .join('');

            let html = `
                <div style="background:#f8f9fa;padding:15px;border-radius:8px;margin-bottom:20px;">
                    <h3 style="margin-bottom:10px;">${data.city}, ${data.country}</h3>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:10px;">
                        <div><strong>${data.total_signals || 0}</strong> signals</div>
                        <div><strong>${data.num_learned_categories || 0}</strong> themes discovered</div>
                        <div>Confidence: <strong>${((data.model_confidence || 0) * 100).toFixed(0)}%</strong></div>
                    </div>
                    ${sourcesHtml ? `<div style="margin-top:10px;">Sources: ${sourcesHtml}</div>` : ''}
                </div>
            `;

            // Top Desires with source attribution
            if (data.top_desires && data.top_desires.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">🔥 Top Desires</h3>';
                html += data.top_desires.slice(0, 5).map(d => {
                    // Format per-theme sources
                    const themeSources = (d.sources || [])
                        .map(s => `${s.name} (${s.count})`)
                        .join(', ') || 'Unknown';

                    // Get example snippet if available
                    const example = d.example_snippets && d.example_snippets[0]
                        ? `<div style="margin-top:8px;padding:8px;background:rgba(0,0,0,0.03);border-radius:4px;font-size:0.85em;font-style:italic;">"${d.example_snippets[0].text.substring(0, 150)}..." <span style="color:#666;">— ${d.example_snippets[0].source}</span></div>`
                        : '';

                    return `
                        <div class="desire-card">
                            <h4>${d.theme_name || d.theme || 'Desire'}</h4>
                            <p>${d.description || ''}</p>
                            <div class="desire-meta">
                                Intensity: ${((d.intensity_score || 0) * 100).toFixed(0)}% |
                                ${d.frequency || 0} mentions |
                                Sources: ${themeSources}
                            </div>
                            ${example}
                        </div>
                    `;
                }).join('');
            }

            // White Space Opportunities
            if (data.white_space_opportunities && data.white_space_opportunities.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">💡 White Space Opportunities</h3>';
                html += data.white_space_opportunities.slice(0, 5).map(o => `
                    <div class="opportunity-card">${o}</div>
                `).join('');
            }

            // Concept Lanes
            if (data.concept_lanes && data.concept_lanes.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">🎯 Concept Lanes</h3>';
                html += data.concept_lanes.slice(0, 3).map(c => `
                    <div class="concept-card">
                        <h4>${c.concept || 'Concept'}</h4>
                        <p>${c.rationale || ''}</p>
                        ${c.key_features ? '<div style="margin-top:8px;font-size:0.9em;">Features: ' + c.key_features.slice(0,3).join(', ') + '</div>' : ''}
                    </div>
                `).join('');
            }

            // Underserved Segments
            if (data.underserved_segments && data.underserved_segments.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">👥 Underserved Segments</h3>';
                html += '<div style="display:flex;flex-wrap:wrap;gap:8px;">';
                html += data.underserved_segments.map(s => `
                    <span style="background:#e0e0e0;padding:5px 12px;border-radius:20px;font-size:0.9em;">${s}</span>
                `).join('');
                html += '</div>';
            }

            resultsDiv.innerHTML = html;
        }

        // Chat functionality
        let chatConversationId = null;

        function buildProfileContext() {
            const savedTrends = getSavedProjects();
            const savedMoves = getSavedMoves();

            if (savedTrends.length === 0 && savedMoves.length === 0) {
                return null;
            }

            let context = 'User research profile: ';

            // Trends summary
            if (savedTrends.length > 0) {
                const trendNames = savedTrends.slice(0, 3).map(t => t.name).filter(Boolean);
                const regions = [...new Set(savedTrends.map(t => t.region).filter(Boolean))];
                const segments = [...new Set(savedTrends.map(t => t.audience_segment).filter(Boolean))];
                const topics = [...new Set(savedTrends.flatMap(t => t.topics || []))].slice(0, 5);

                context += `Tracking ${savedTrends.length} trends`;
                if (trendNames.length) context += ` including "${trendNames.join('", "')}"`;
                if (regions.length) context += `. Interested in regions: ${regions.join(', ')}`;
                if (segments.length) context += `. Target segments: ${segments.join(', ')}`;
                if (topics.length) context += `. Key themes: ${topics.join(', ')}`;
                context += '. ';
            }

            // Moves summary
            if (savedMoves.length > 0) {
                const companies = [...new Set(savedMoves.map(m => m.company).filter(Boolean))].slice(0, 3);
                const markets = [...new Set(savedMoves.map(m => m.market).filter(Boolean))].slice(0, 3);
                const moveTypes = [...new Set(savedMoves.map(m => m.move_type).filter(Boolean))];

                context += `Watching ${savedMoves.length} strategic moves`;
                if (companies.length) context += ` by companies like ${companies.join(', ')}`;
                if (markets.length) context += ` in markets: ${markets.join(', ')}`;
                if (moveTypes.length) context += `. Move types: ${moveTypes.map(m => m.replace('_', ' ')).join(', ')}`;
                context += '.';
            }

            return context;
        }

        function sendSuggestion(text) {
            document.getElementById('chat-input').value = text;
            sendMessage();
        }

        function startBrandBuild() {
            const savedTrends = getSavedProjects();
            const savedMoves = getSavedMoves();

            let message;

            if (savedTrends.length > 0 || savedMoves.length > 0) {
                // Has profile - build from it
                const regions = [...new Set(savedTrends.map(t => t.region).filter(Boolean))];
                const segments = [...new Set(savedTrends.map(t => t.audience_segment).filter(Boolean))];
                const topics = [...new Set(savedTrends.flatMap(t => t.topics || []))].slice(0, 3);

                message = 'Help me build a hotel brand based on my research profile.';
                if (regions.length) message += ' I am interested in ' + regions.slice(0, 2).join(' and ') + '.';
                if (segments.length) message += ' Target segment: ' + segments[0] + '.';
                if (topics.length) message += ' Key themes I have been tracking: ' + topics.join(', ') + '.';
            } else {
                // No profile - ask for guidance
                message = 'I want to build a hotel brand but I am not sure where to start. Can you help me figure out what kind of brand would be right?';
            }

            document.getElementById('chat-input').value = message;
            sendMessage();
        }

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;

            const messagesDiv = document.getElementById('chat-messages');

            // Clear welcome message if present
            const welcome = messagesDiv.querySelector('.chat-welcome');
            if (welcome) welcome.remove();

            // Add user message
            messagesDiv.innerHTML += `
                <div class="chat-message user">
                    <div class="chat-bubble">${escapeHtml(message)}</div>
                </div>
            `;
            input.value = '';

            // Add typing indicator
            messagesDiv.innerHTML += `
                <div class="chat-message assistant" id="typing-indicator">
                    <div class="chat-bubble chat-typing">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            `;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Build profile context from saved items
            const profileContext = buildProfileContext();

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        conversation_id: chatConversationId,
                        user_context: profileContext
                    })
                });

                const data = await res.json();

                // Remove typing indicator
                const typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();

                if (res.ok) {
                    chatConversationId = data.conversation_id;

                    // Add assistant message
                    const confClass = data.confidence === 'High' ? 'confidence-high' :
                                      data.confidence === 'Medium' ? 'confidence-medium' : 'confidence-low';
                    messagesDiv.innerHTML += `
                        <div class="chat-message assistant">
                            <div class="chat-bubble">${formatResponse(data.response)}</div>
                            <div class="chat-confidence ${confClass}">
                                ${data.confidence} confidence | ${data.sources_used} sources | Mode: ${data.mode}
                            </div>
                        </div>
                    `;

                    // Show state info
                    if (data.state) {
                        document.getElementById('chat-state').innerHTML = `
                            Mode: ${data.mode} (${Math.round(data.state.mode_confidence * 100)}%) |
                            Location: ${data.state.slots?.location || '-'} |
                            Segment: ${data.state.slots?.segment || '-'}
                        `;
                    }

                    // Show suggested action
                    if (data.suggested_action) {
                        messagesDiv.innerHTML += `
                            <div class="chat-message assistant">
                                <button onclick="window.location.href='/api/monitoring/dashboard-v2#build'"
                                        class="suggestion-chip" style="margin-top:10px;">
                                    ➡️ Continue to Build a Brand
                                </button>
                            </div>
                        `;
                    }
                } else {
                    messagesDiv.innerHTML += `
                        <div class="chat-message assistant">
                            <div class="chat-bubble" style="background:#fee2e2;color:#991b1b;">
                                Error: ${data.detail || 'Something went wrong'}
                            </div>
                        </div>
                    `;
                }

            } catch (err) {
                const typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();
                messagesDiv.innerHTML += `
                    <div class="chat-message assistant">
                        <div class="chat-bubble" style="background:#fee2e2;color:#991b1b;">
                            Connection error: ${err.message}
                        </div>
                    </div>
                `;
            }

            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
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
    <title>Build a Brand | BrandClave</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #333;
        }
        .hero {
            background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }
        .hero h1 { font-size: 2em; margin-bottom: 8px; }
        .hero p { opacity: 0.9; }
        .back-link {
            display: inline-block;
            margin-top: 15px;
            color: white;
            text-decoration: none;
            opacity: 0.8;
        }
        .back-link:hover { opacity: 1; }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }

        .card {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .card h2 { color: #1a1a2e; margin-bottom: 20px; font-size: 1.4em; }

        .source-trend {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .source-trend h3 { margin-bottom: 8px; }
        .source-trend p { opacity: 0.9; font-size: 0.9em; }

        .profile-source-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .profile-source-card h3 { margin-bottom: 8px; }
        .profile-row { display: flex; gap: 8px; margin-bottom: 4px; }
        .profile-label { opacity: 0.85; }
        .profile-theme-tag {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            margin: 2px;
        }

        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            color: #1a1a2e;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
        }
        .form-group textarea { min-height: 100px; resize: vertical; }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }

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
            background: #e0e7ff;
            color: #4338ca;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
        }

        .btn-generate {
            width: 100%;
            padding: 15px 30px;
            background: linear-gradient(135deg, #e94560 0%, #f06292 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn-generate:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        }
        .btn-generate:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        #result-container { display: none; }

        .blueprint-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 12px;
            color: white;
        }
        .blueprint-card h2 { color: white; margin-bottom: 5px; }
        .blueprint-oneliner { font-size: 1.1em; opacity: 0.9; margin-bottom: 20px; }

        .blueprint-section { margin-bottom: 20px; }
        .blueprint-section h3 {
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.8;
            margin-bottom: 8px;
        }
        .blueprint-section p { line-height: 1.6; }
        .blueprint-section ul { padding-left: 20px; }
        .blueprint-section li { margin-bottom: 5px; }

        .experience-card {
            background: rgba(255,255,255,0.15);
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .experience-card h4 { margin-bottom: 5px; }
        .experience-card p { font-size: 0.9em; opacity: 0.9; }

        .loading-indicator {
            text-align: center;
            padding: 40px;
        }
        .loading-indicator .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(0,0,0,0.1);
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

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
            background: white;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
        }
        .btn-secondary:hover { background: #f0f0f0; }

        .white-space-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: rgba(255,255,255,0.2);
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.85em;
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
            <h3>🎯 Building from Your Profile</h3>
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
                    <div id="bp-alternates" style="font-size:0.9em;color:#666;margin-top:5px;"></div>
                </div>
                <p class="blueprint-oneliner" id="bp-oneliner">One-liner</p>

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
                    <p id="bp-investor" style="background:#f8f8f8;padding:15px;border-radius:8px;"></p>
                </div>

                <div id="bp-metadata" style="margin-top:20px;font-size:0.85em;color:#888;">
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
        let sourceTrend = null;
        let profileData = null;
        let currentBlueprint = null;

        // Load source trend or profile from sessionStorage
        function loadSourceTrend() {
            try {
                const data = sessionStorage.getItem('brandclave_brand_input');
                const profile = sessionStorage.getItem('brandclave_profile_data');

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
                        const topicsEl = document.getElementById('source-topics');
                        if (sourceTrend.topics && sourceTrend.topics.length) {
                            topicsEl.innerHTML = sourceTrend.topics.map(t =>
                                '<span class="topic-tag">' + t + '</span>'
                            ).join('');
                        }

                        // Show white space score
                        if (sourceTrend.white_space_score) {
                            const ws = Math.round(sourceTrend.white_space_score * 100);
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
            const themesEl = document.getElementById('profile-themes');
            const allThemes = [
                ...(profile.topics || []).slice(0, 3),
                ...(profile.segments || []).slice(0, 2),
                ...(profile.markets || []).slice(0, 2)
            ];

            if (allThemes.length > 0) {
                themesEl.innerHTML = '<div style="margin-top:8px;">' +
                    allThemes.map(t => '<span class="profile-theme-tag">' + t + '</span>').join('') +
                    '</div>';
            }
        }

        async function generateBrandConcept() {
            const btn = document.getElementById('generate-btn');
            const loadingEl = document.getElementById('loading-container');
            const resultEl = document.getElementById('result-container');

            // Validate inputs
            const location = document.getElementById('brand-location').value;
            const segment = document.getElementById('brand-segment').value;
            const adr = document.getElementById('brand-adr').value;
            const rooms = document.getElementById('brand-rooms').value || 100;
            const goal = document.getElementById('brand-goal').value;

            if (!location || !adr || !goal) {
                alert('Please fill in Location, Target ADR, and Developer Goal.');
                return;
            }

            btn.disabled = true;
            loadingEl.style.display = 'block';
            resultEl.style.display = 'none';

            // Reset stage indicators
            document.querySelectorAll('.stage-item').forEach(el => {
                el.querySelector('.stage-icon').innerHTML = '&#9675;';
                el.style.color = '#666';
            });

            try {
                // Call the new blueprint generation API
                const res = await fetch('/api/brand-blueprint/generate-simple', {
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

                const data = await res.json();

                if (res.ok && data.blueprint) {
                    // Update all stage indicators to complete
                    document.querySelectorAll('.stage-item').forEach(el => {
                        el.querySelector('.stage-icon').innerHTML = '&#10003;';
                        el.style.color = '#27ae60';
                    });

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
            let prompt = 'Help me create a detailed hotel brand concept';

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
                    const trendNames = inputs.profile.trends.slice(0, 3).map(t => t.name || t.description).filter(Boolean);
                    if (trendNames.length) prompt += trendNames.join(', ');
                }

                // Add topic themes
                if (inputs.profile.topics && inputs.profile.topics.length > 0) {
                    prompt += '. Key themes I am interested in: ' + inputs.profile.topics.slice(0, 5).join(', ');
                }

                // Add move insights
                if (inputs.profile.moves && inputs.profile.moves.length > 0) {
                    prompt += '. I have been watching ' + inputs.profile.moves.length + ' strategic moves by companies like: ';
                    const companies = [...new Set(inputs.profile.moves.map(m => m.company).filter(Boolean))].slice(0, 3);
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
            const names = blueprint.brand_names || {};
            document.getElementById('bp-name').textContent = names.primary || 'Brand Concept';
            if (names.alternate_1 || names.alternate_2) {
                document.getElementById('bp-alternates').textContent =
                    'Alternates: ' + [names.alternate_1, names.alternate_2].filter(Boolean).join(', ');
            }

            document.getElementById('bp-oneliner').textContent = blueprint.one_liner || '';
            document.getElementById('bp-thesis').textContent = blueprint.thesis || '';

            // Pillars
            const pillars = blueprint.pillars || [];
            document.getElementById('bp-pillars').innerHTML = pillars.map(p => '<li>' + p + '</li>').join('');

            // Positioning
            document.getElementById('bp-positioning').textContent = blueprint.positioning_statement || '';

            // Guest personas
            const personas = blueprint.guest_personas || [];
            document.getElementById('bp-personas').innerHTML = personas.map(p =>
                '<div class="experience-card">' +
                '<strong>' + (p.name || '') + '</strong>' +
                '<p>' + (p.description || '') + '</p>' +
                '<p style="font-size:0.9em;color:#666;">Spend: ' + (p.spend_behavior || '') + '</p>' +
                '</div>'
            ).join('');

            // Signature experiences
            const experiences = blueprint.signature_experiences || [];
            document.getElementById('bp-experiences').innerHTML = experiences.map(e =>
                '<div class="experience-card">' +
                '<strong>' + (e.name || '') + '</strong>' +
                '<p>' + (e.description || '') + '</p>' +
                '<p style="font-size:0.9em;color:#27ae60;">' + (e.why_it_matters || '') + '</p>' +
                '</div>'
            ).join('');

            // Guest journey
            const journey = blueprint.guest_journey || {};
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
            const fnb = blueprint.fnb_concepts || [];
            document.getElementById('bp-fnb').innerHTML = fnb.map(f =>
                '<div class="experience-card">' +
                '<strong>' + (f.name || '') + '</strong>' +
                '<p>' + (f.concept || '') + '</p>' +
                '<p style="font-size:0.9em;color:#666;">Vibe: ' + (f.vibe || '') + '</p>' +
                '</div>'
            ).join('');

            // Revenue logic
            document.getElementById('bp-revenue').textContent = blueprint.revenue_logic || '';

            // Investor summary
            document.getElementById('bp-investor').textContent = blueprint.investor_summary || '';

            // Metadata
            const confidence = Math.round((blueprint.confidence || 0) * 100);
            document.getElementById('bp-confidence').textContent = 'Confidence: ' + confidence + '%';

            const tokens = blueprint.token_usage || {};
            if (tokens.total_tokens) {
                document.getElementById('bp-tokens').textContent =
                    'Tokens: ' + tokens.total_tokens + ' (~$' + (tokens.estimated_cost_usd || 0).toFixed(3) + ')';
            }
        }

        function parseResponse(text) {
            const result = {
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
            const namePatterns = [
                /brand\\s*name[:\\s]*[""]([^""]+)[""]|brand\\s*name[:\\s]*["']([^"']+)["']/i,
                /[""]The\\s+([^""]+)[""]|["']The\\s+([^"']+)["']/i,
                /called\\s+[""]([^""]+)[""]|called\\s+["']([^"']+)["']/i,
                /\\*\\*[""]?([^"*]+)[""]?\\*\\*/
            ];
            for (const pattern of namePatterns) {
                const match = text.match(pattern);
                if (match) {
                    result.name = cleanMarkdown(match[1] || match[2] || '');
                    if (result.name) break;
                }
            }

            // Extract one-liner/essence
            const essenceMatch = text.match(/one[- ]liner[^:]*:[\\s]*[""]?([^""\\n]+)/i) ||
                                 text.match(/essence[^:]*:[\\s]*[""]?([^""\\n]+)/i);
            if (essenceMatch) result.oneliner = cleanMarkdown(essenceMatch[1]);

            // Split into sections by markdown headers
            const sections = text.split(/(?=#{2,3}\\s|\\*\\*\\d+\\.|\\*\\*[A-Z])/);

            for (const section of sections) {
                const lower = section.toLowerCase();
                const content = cleanMarkdown(section.replace(/^[#*\\d.\\s]+[^\\n]*\\n?/, ''));

                // Only match section headers, not content
                const isHeader = section.match(/^#{2,3}\\s|^\\*\\*\\d+\\.|^\\*\\*[A-Z]/);
                if (!isHeader) continue;

                if (lower.includes('thesis') || lower.includes('philosophy') || lower.includes('core concept')) {
                    result.thesis = content;
                } else if (lower.includes('pillar')) {
                    // Extract bullet points
                    const bullets = section.match(/[-*]\\s+\\*?\\*?([^\\n*]+)/g) || [];
                    result.pillars = bullets.map(b => cleanMarkdown(b)).filter(b => b.length > 3);
                } else if ((lower.includes('experience') || lower.includes('signature')) && !lower.includes('target')) {
                    const bullets = section.match(/[-*]\\s+\\*?\\*?([^\\n*]+)/g) ||
                                   section.match(/\\*\\*([^*]+)\\*\\*[^\\n]*/g) || [];
                    result.experiences = bullets.map(b => cleanMarkdown(b)).filter(b => b.length > 3);
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
                const expMatch = text.match(/signature\\s+experience[^:]*:([\\s\\S]*?)(?=###|\\*\\*\\d|$)/i);
                if (expMatch) {
                    const bullets = expMatch[1].match(/[-*]\\s+\\*?\\*?([^\\n]+)/g) || [];
                    result.experiences = bullets.map(b => cleanMarkdown(b)).filter(b => b.length > 5);
                }
            }

            // Try alternate extraction for personas if empty
            if (!result.personas) {
                const personaMatch = text.match(/target\\s+guest[^:]*:([\\s\\S]*?)(?=###|\\*\\*\\d|$)/i);
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
                const firstPara = text.split('\\n\\n')[0];
                result.thesis = cleanMarkdown(firstPara).substring(0, 400);
            }
            if (!result.success) {
                result.success = 'This concept fills a clear market gap by combining trending traveler preferences with authentic local experiences.';
            }

            return result;
        }

        function saveBlueprintToProject() {
            if (!currentBlueprint) return;

            const saved = JSON.parse(localStorage.getItem('brandclave_saved_blueprints') || '[]');
            saved.push(currentBlueprint);
            localStorage.setItem('brandclave_saved_blueprints', JSON.stringify(saved));

            alert('Blueprint saved!');
        }

        function regenerateConcept() {
            generateBrandConcept();
        }

        // Initialize
        loadSourceTrend();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
