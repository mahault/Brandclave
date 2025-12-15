"""Monitoring and metrics API routes."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

router = APIRouter()


class ScraperMetricsResponse(BaseModel):
    """Response model for scraper metrics."""

    source: str
    total_items: int
    items_last_24h: int
    items_last_7d: int
    last_run_at: Optional[str]
    last_run_status: Optional[str]
    last_run_items: int
    error_rate_24h: float


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics."""

    total_content: int
    content_by_source: dict
    content_by_type: dict
    processed_content: int
    unprocessed_content: int
    embeddings_count: int
    trends_count: int
    moves_count: int
    properties_count: int
    cache_stats: dict
    scheduler_stats: dict


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    database: str
    cache: str
    scheduler: str
    timestamp: str


@router.get("/monitoring/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    from db.database import engine
    from sqlalchemy import text

    # Database check
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)[:50]}"

    # Cache check
    try:
        from cache.redis_cache import get_cache
        cache = get_cache()
        cache_stats = cache.get_stats()
        cache_status = cache_stats.get("status", "unknown")
    except Exception:
        cache_status = "unavailable"

    # Scheduler check
    try:
        from scheduler.scheduler import get_scheduler
        scheduler = get_scheduler()
        if scheduler.is_running:
            scheduler_status = "running"
        elif scheduler.is_available:
            scheduler_status = "stopped"
        else:
            scheduler_status = "unavailable"
    except Exception:
        scheduler_status = "unavailable"

    # Overall status
    overall = "healthy" if db_status == "healthy" else "degraded"

    return HealthResponse(
        status=overall,
        version="0.5.0",
        database=db_status,
        cache=cache_status,
        scheduler=scheduler_status,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/monitoring/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get overall system metrics."""
    with MetricsCollector() as collector:
        metrics = collector.get_system_metrics()

    return SystemMetricsResponse(
        total_content=metrics.total_content,
        content_by_source=metrics.content_by_source,
        content_by_type=metrics.content_by_type,
        processed_content=metrics.processed_content,
        unprocessed_content=metrics.unprocessed_content,
        embeddings_count=metrics.embeddings_count,
        trends_count=metrics.trends_count,
        moves_count=metrics.moves_count,
        properties_count=metrics.properties_count,
        cache_stats=metrics.cache_stats,
        scheduler_stats=metrics.scheduler_stats,
    )


@router.get("/monitoring/scrapers", response_model=list[ScraperMetricsResponse])
async def get_scraper_metrics():
    """Get metrics for all scrapers."""
    with MetricsCollector() as collector:
        metrics = collector.get_all_scraper_metrics()

    return [
        ScraperMetricsResponse(
            source=m.source,
            total_items=m.total_items,
            items_last_24h=m.items_last_24h,
            items_last_7d=m.items_last_7d,
            last_run_at=m.last_run_at,
            last_run_status=m.last_run_status,
            last_run_items=m.last_run_items,
            error_rate_24h=m.error_rate_24h,
        )
        for m in metrics
    ]


@router.get("/monitoring/scrapers/{source}", response_model=ScraperMetricsResponse)
async def get_single_scraper_metrics(source: str):
    """Get metrics for a specific scraper."""
    with MetricsCollector() as collector:
        m = collector.get_scraper_metrics(source)

    return ScraperMetricsResponse(
        source=m.source,
        total_items=m.total_items,
        items_last_24h=m.items_last_24h,
        items_last_7d=m.items_last_7d,
        last_run_at=m.last_run_at,
        last_run_status=m.last_run_status,
        last_run_items=m.last_run_items,
        error_rate_24h=m.error_rate_24h,
    )


@router.get("/monitoring/errors")
async def get_recent_errors(limit: int = Query(20, ge=1, le=100)):
    """Get recent job errors."""
    with MetricsCollector() as collector:
        errors = collector.get_recent_errors(limit)

    return {"errors": errors, "total": len(errors)}


@router.get("/monitoring/activity")
async def get_recent_activity(hours: int = Query(24, ge=1, le=168)):
    """Get recent activity summary."""
    with MetricsCollector() as collector:
        activity = collector.get_recent_activity(hours)

    return activity


@router.get("/monitoring/content")
async def get_recent_content(limit: int = Query(20, ge=1, le=100)):
    """Get recent scraped content for display."""
    from db.database import SessionLocal
    from db.models import RawContentModel

    db = SessionLocal()
    try:
        items = db.query(RawContentModel).order_by(
            RawContentModel.scraped_at.desc()
        ).limit(limit).all()

        return {
            "items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "content": item.content[:500] if item.content else None,
                    "source": item.source,
                    "source_type": item.source_type,
                    "author": item.author,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                    "scraped_at": item.scraped_at.isoformat() if item.scraped_at else None,
                    "url": item.url,
                }
                for item in items
            ],
            "total": len(items),
        }
    finally:
        db.close()


@router.get("/monitoring/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Interactive dashboard showcasing BrandClave intelligence."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>BrandClave Intelligence Dashboard</title>
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
            padding: 40px 20px;
            text-align: center;
        }
        .hero h1 { font-size: 2.5em; margin-bottom: 10px; }
        .hero p { opacity: 0.9; font-size: 1.1em; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        /* Navigation Tabs */
        .tabs {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .tab {
            padding: 12px 24px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        .tab:hover { background: rgba(255,255,255,0.2); }
        .tab.active { background: #e94560; }

        /* Cards */
        .card {
            background: white;
            padding: 25px;
            margin: 15px 0;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .card h2 {
            color: #1a1a2e;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card h2 .icon { font-size: 1.3em; }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric {
            text-align: center;
            padding: 20px 15px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
        }
        .metric-value {
            font-size: 2.2em;
            font-weight: bold;
            color: #0f3460;
        }
        .metric-label {
            color: #666;
            font-size: 0.85em;
            margin-top: 5px;
        }

        /* Trend Cards */
        .trend-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .trend-card h3 { margin-bottom: 10px; font-size: 1.3em; }
        .trend-card p { opacity: 0.95; line-height: 1.5; }
        .trend-meta {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .trend-meta span {
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85em;
        }

        /* Move Cards */
        .move-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .move-card h3 { margin-bottom: 10px; font-size: 1.2em; }
        .move-card .company {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        .move-card p { opacity: 0.95; line-height: 1.5; }

        /* Content Items */
        .content-item {
            border-left: 4px solid #e94560;
            padding: 15px 20px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 0 8px 8px 0;
        }
        .content-item h4 {
            color: #1a1a2e;
            margin-bottom: 8px;
        }
        .content-item p {
            color: #555;
            font-size: 0.95em;
            line-height: 1.5;
        }
        .content-item .meta {
            margin-top: 10px;
            font-size: 0.85em;
            color: #888;
        }
        .content-item .source {
            display: inline-block;
            background: #e94560;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-right: 10px;
        }

        /* Property Card */
        .property-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .property-card h3 { margin-bottom: 5px; font-size: 1.4em; }
        .property-card .type { opacity: 0.9; margin-bottom: 15px; }
        .property-score {
            display: inline-block;
            background: rgba(255,255,255,0.25);
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        .property-themes {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .property-themes span {
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85em;
        }

        /* Status badges */
        .status-healthy { color: #22c55e; }
        .status-degraded { color: #f59e0b; }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: 500;
        }
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .badge-error { background: #fee2e2; color: #991b1b; }
        .badge-info { background: #dbeafe; color: #1e40af; }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #888;
        }
        .empty-state .icon { font-size: 3em; margin-bottom: 15px; }
        .empty-state p { margin-bottom: 15px; }
        .empty-state .hint {
            background: #f0f9ff;
            padding: 15px;
            border-radius: 8px;
            font-size: 0.9em;
            color: #0369a1;
        }

        /* Section visibility */
        .section { display: none; }
        .section.active { display: block; }

        /* Buttons */
        .btn {
            background: #e94560;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95em;
            transition: all 0.3s;
        }
        .btn:hover { background: #d63850; transform: translateY(-2px); }
        .btn-outline {
            background: transparent;
            border: 2px solid #e94560;
            color: #e94560;
        }
        .btn-outline:hover { background: #e94560; color: white; }

        /* City Desires */
        .desire-theme {
            background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .desire-theme h3 { margin-bottom: 8px; font-size: 1.2em; }
        .desire-theme p { opacity: 0.95; line-height: 1.5; margin-bottom: 12px; }
        .desire-scores {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .desire-scores span {
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85em;
        }
        .opportunity-card {
            background: linear-gradient(135deg, #5f27cd 0%, #341f97 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .opportunity-card h3 { margin-bottom: 10px; }
        .concept-card {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .concept-card h3 { margin-bottom: 8px; font-size: 1.1em; }
        .concept-card .rationale { opacity: 0.95; font-size: 0.95em; line-height: 1.5; }
        .segment-badge {
            display: inline-block;
            background: rgba(0,0,0,0.2);
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            margin: 3px;
        }
        .city-summary {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .city-summary h2 { margin-bottom: 15px; }
        .city-summary .summary-metrics {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .city-summary .summary-metric {
            text-align: center;
            background: rgba(255,255,255,0.1);
            padding: 15px 25px;
            border-radius: 10px;
        }
        .city-summary .summary-metric .value {
            font-size: 2em;
            font-weight: bold;
        }
        .city-summary .summary-metric .label {
            font-size: 0.85em;
            opacity: 0.8;
            margin-top: 5px;
        }

        /* Loading */
        .loading {
            text-align: center;
            padding: 40px;
            color: #888;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 { font-size: 1.8em; }
            .tabs { justify-content: center; }
            .tab { padding: 10px 16px; font-size: 0.9em; }
        }
    </style>
</head>
<body>
    <div class="hero">
        <h1>BrandClave Intelligence</h1>
        <p>Hospitality Demand Intelligence, Trend Signals & Strategic Moves</p>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="tab active" onclick="showSection('overview')">Overview</button>
            <button class="tab" onclick="showSection('citydesires')">City Desires</button>
            <button class="tab" onclick="showSection('trends')">Social Pulse</button>
            <button class="tab" onclick="showSection('moves')">Hotelier Bets</button>
            <button class="tab" onclick="showSection('properties')">Demand Scan</button>
            <button class="tab" onclick="showSection('content')">Raw Content</button>
            <button class="tab" onclick="showSection('system')">System Status</button>
        </div>

        <!-- Overview Section -->
        <div id="overview" class="section active">
            <div class="card">
                <h2><span class="icon">📊</span> Intelligence Overview</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="total-content">-</div>
                        <div class="metric-label">Content Scraped</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="processed">-</div>
                        <div class="metric-label">Processed</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="trends-count">-</div>
                        <div class="metric-label">Trends Detected</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="moves-count">-</div>
                        <div class="metric-label">Hotelier Moves</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="properties-count">-</div>
                        <div class="metric-label">Properties Scanned</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">🔥</span> Latest Trend</h2>
                <div id="latest-trend">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">🎯</span> Latest Hotelier Move</h2>
                <div id="latest-move">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </div>
            </div>
        </div>

        <!-- City Desires Section -->
        <div id="citydesires" class="section">
            <div class="card">
                <h2><span class="icon">🌆</span> City Desires - What Travelers Want</h2>
                <p style="color:#666;margin-bottom:20px;">Type a city to discover unmet traveler needs and white space opportunities</p>

                <div style="display:flex;gap:10px;margin-bottom:25px;flex-wrap:wrap;">
                    <input type="text" id="city-input" placeholder="Enter city name..."
                           style="flex:1;min-width:200px;padding:12px 15px;border:2px solid #e0e0e0;border-radius:8px;font-size:1em;outline:none;"
                           onkeypress="if(event.key==='Enter')analyzeCity()">
                    <input type="text" id="country-input" placeholder="Country (optional)"
                           style="width:150px;padding:12px 15px;border:2px solid #e0e0e0;border-radius:8px;font-size:1em;outline:none;"
                           onkeypress="if(event.key==='Enter')analyzeCity()">
                    <button class="btn" onclick="analyzeCity()">Analyze City</button>
                </div>

                <div style="margin-bottom:20px;">
                    <span style="color:#888;font-size:0.9em;margin-right:10px;">Popular:</span>
                    <button onclick="analyzeCity('Lisbon','Portugal')" class="btn-outline btn" style="padding:6px 12px;font-size:0.85em;margin:3px;">Lisbon</button>
                    <button onclick="analyzeCity('Barcelona','Spain')" class="btn-outline btn" style="padding:6px 12px;font-size:0.85em;margin:3px;">Barcelona</button>
                    <button onclick="analyzeCity('Tokyo','Japan')" class="btn-outline btn" style="padding:6px 12px;font-size:0.85em;margin:3px;">Tokyo</button>
                    <button onclick="analyzeCity('Bali','Indonesia')" class="btn-outline btn" style="padding:6px 12px;font-size:0.85em;margin:3px;">Bali</button>
                    <button onclick="analyzeCity('Paris','France')" class="btn-outline btn" style="padding:6px 12px;font-size:0.85em;margin:3px;">Paris</button>
                </div>

                <div id="city-results">
                    <div class="empty-state">
                        <div class="icon">🔍</div>
                        <p>Enter a city name to discover what travelers want but can't find</p>
                        <div class="hint">
                            This scrapes Reddit, YouTube, and travel forums to identify:<br>
                            • Top unmet desires and frustrations<br>
                            • Underserved traveler segments<br>
                            • White space opportunities for hotels
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Social Pulse Section -->
        <div id="trends" class="section">
            <div class="card">
                <h2><span class="icon">📈</span> Social Pulse - Trend Signals</h2>
                <p style="color:#666;margin-bottom:20px;">AI-detected travel trends from Reddit, YouTube, and social conversations</p>
                <div id="trends-list">
                    <div class="loading"><div class="spinner"></div>Loading trends...</div>
                </div>
            </div>
        </div>

        <!-- Hotelier Bets Section -->
        <div id="moves" class="section">
            <div class="card">
                <h2><span class="icon">♟️</span> Hotelier Bets - Strategic Moves</h2>
                <p style="color:#666;margin-bottom:20px;">AI-extracted strategic moves from hospitality news and press releases</p>
                <div id="moves-list">
                    <div class="loading"><div class="spinner"></div>Loading moves...</div>
                </div>
            </div>
        </div>

        <!-- Demand Scan Section -->
        <div id="properties" class="section">
            <div class="card">
                <h2><span class="icon">🏨</span> Demand Scan - Property Analysis</h2>
                <p style="color:#666;margin-bottom:20px;">AI-powered property analysis with demand fit scoring</p>
                <div id="properties-list">
                    <div class="loading"><div class="spinner"></div>Loading properties...</div>
                </div>
            </div>
        </div>

        <!-- Raw Content Section -->
        <div id="content" class="section">
            <div class="card">
                <h2><span class="icon">📰</span> Recent Content Scraped</h2>
                <p style="color:#666;margin-bottom:20px;">Latest content collected from all sources</p>
                <div id="content-list">
                    <div class="loading"><div class="spinner"></div>Loading content...</div>
                </div>
            </div>
        </div>

        <!-- System Status Section -->
        <div id="system" class="section">
            <div class="card">
                <h2><span class="icon">💚</span> System Health</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="health-status">-</div>
                        <div class="metric-label">Overall</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="db-status">-</div>
                        <div class="metric-label">Database</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="cache-status">-</div>
                        <div class="metric-label">Cache</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="scheduler-status">-</div>
                        <div class="metric-label">Scheduler</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">🔄</span> Scraper Status</h2>
                <div id="scrapers-list">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab navigation
        function showSection(sectionId) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(sectionId).classList.add('active');
            event.target.classList.add('active');
        }

        function getStatusClass(status) {
            const s = (status || '').toLowerCase();
            if (s === 'healthy' || s === 'running' || s === 'connected' || s === 'completed')
                return 'status-healthy';
            if (s === 'degraded' || s === 'stopped' || s === 'disconnected')
                return 'status-degraded';
            return '';
        }

        function truncate(text, maxLength) {
            if (!text) return '';
            return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
        }

        async function loadData() {
            try {
                // Load all data in parallel
                const [health, metrics, trends, moves, properties, content, scrapers] = await Promise.all([
                    fetch('/api/monitoring/health').then(r => r.json()).catch(() => null),
                    fetch('/api/monitoring/metrics').then(r => r.json()).catch(() => null),
                    fetch('/api/social-pulse?limit=10').then(r => r.json()).catch(() => ({trends: []})),
                    fetch('/api/hotelier-bets?limit=10').then(r => r.json()).catch(() => ({moves: []})),
                    fetch('/api/demand-scan?limit=5').then(r => r.json()).catch(() => ({properties: []})),
                    fetch('/api/monitoring/content?limit=50').then(r => r.json()).catch(() => ({items: []})),
                    fetch('/api/monitoring/scrapers').then(r => r.json()).catch(() => []),
                ]);

                // Update metrics
                if (metrics) {
                    document.getElementById('total-content').textContent = metrics.total_content?.toLocaleString() || '0';
                    document.getElementById('processed').textContent = metrics.processed_content?.toLocaleString() || '0';
                    document.getElementById('trends-count').textContent = metrics.trends_count || '0';
                    document.getElementById('moves-count').textContent = metrics.moves_count || '0';
                    document.getElementById('properties-count').textContent = metrics.properties_count || '0';
                }

                // Update health
                if (health) {
                    document.getElementById('health-status').textContent = health.status;
                    document.getElementById('health-status').className = 'metric-value ' + getStatusClass(health.status);
                    document.getElementById('db-status').textContent = health.database;
                    document.getElementById('db-status').className = 'metric-value ' + getStatusClass(health.database);
                    document.getElementById('cache-status').textContent = health.cache;
                    document.getElementById('scheduler-status').textContent = health.scheduler;
                }

                // Render trends
                const trendsList = trends.trends || [];
                if (trendsList.length > 0) {
                    document.getElementById('latest-trend').innerHTML = renderTrend(trendsList[0]);
                    document.getElementById('trends-list').innerHTML = trendsList.map(renderTrend).join('');
                } else {
                    const emptyTrend = renderEmptyState('trends', 'No trends detected yet',
                        'Run: python scripts/run_crawlers.py --all --process --trends');
                    document.getElementById('latest-trend').innerHTML = emptyTrend;
                    document.getElementById('trends-list').innerHTML = emptyTrend;
                }

                // Render moves
                const movesList = moves.moves || [];
                if (movesList.length > 0) {
                    document.getElementById('latest-move').innerHTML = renderMove(movesList[0]);
                    document.getElementById('moves-list').innerHTML = movesList.map(renderMove).join('');
                } else {
                    const emptyMove = renderEmptyState('moves', 'No hotelier moves extracted yet',
                        'Run: python scripts/run_crawlers.py --source skift --process --moves');
                    document.getElementById('latest-move').innerHTML = emptyMove;
                    document.getElementById('moves-list').innerHTML = emptyMove;
                }

                // Render properties
                const propsList = properties.properties || [];
                if (propsList.length > 0) {
                    document.getElementById('properties-list').innerHTML = propsList.map(renderProperty).join('');
                } else {
                    document.getElementById('properties-list').innerHTML = renderEmptyState('properties',
                        'No properties scanned yet',
                        'Run: python scripts/run_crawlers.py --scan "https://acehotel.com/new-york/"');
                }

                // Render content
                const contentItems = content.items || [];
                if (contentItems.length > 0) {
                    document.getElementById('content-list').innerHTML = contentItems.map(renderContent).join('');
                } else {
                    document.getElementById('content-list').innerHTML = renderEmptyState('content',
                        'No content scraped yet',
                        'Run: python scripts/run_crawlers.py --all');
                }

                // Render scrapers
                if (scrapers.length > 0) {
                    document.getElementById('scrapers-list').innerHTML = `
                        <table style="width:100%;border-collapse:collapse;">
                            <thead>
                                <tr style="background:#f8f9fa;">
                                    <th style="padding:12px;text-align:left;">Source</th>
                                    <th style="padding:12px;text-align:left;">Total</th>
                                    <th style="padding:12px;text-align:left;">Last 24h</th>
                                    <th style="padding:12px;text-align:left;">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${scrapers.map(s => `
                                    <tr style="border-bottom:1px solid #eee;">
                                        <td style="padding:12px;"><strong>${s.source}</strong></td>
                                        <td style="padding:12px;">${s.total_items.toLocaleString()}</td>
                                        <td style="padding:12px;">${s.items_last_24h}</td>
                                        <td style="padding:12px;"><span class="badge badge-${s.last_run_status === 'completed' ? 'success' : 'warning'}">${s.last_run_status || 'N/A'}</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
                } else {
                    document.getElementById('scrapers-list').innerHTML = '<p style="color:#888;text-align:center;padding:20px;">No scraper data available</p>';
                }

            } catch (error) {
                console.error('Failed to load data:', error);
            }
        }

        function renderTrend(t) {
            // Ensure we have a meaningful name
            let trendName = t.name || t.trend_name || '';
            if (!trendName || trendName.toLowerCase().includes('unnamed') || trendName.toLowerCase().includes('untitled')) {
                // Try to generate from topics or description
                if (t.topics && t.topics.length > 0) {
                    trendName = t.topics.slice(0, 3).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') + ' Trend';
                } else if (t.description) {
                    trendName = truncate(t.description, 40) + ' Trend';
                } else {
                    trendName = 'Hospitality Trend #' + (Math.floor(Math.random() * 1000));
                }
            }

            const sourceCount = t.volume || (t.source_content_ids ? t.source_content_ids.length : 0);
            const hasClickableSources = t.id && t.source_content_ids && t.source_content_ids.length > 0;

            return `
                <div class="trend-card">
                    <h3>${trendName}</h3>
                    <p>${t.description || t.why_it_matters || 'Emerging trend in the hospitality space.'}</p>
                    <div class="trend-meta">
                        <span>Strength: ${t.strength_label || (t.strength_score ? Math.round(t.strength_score * 100) + '%' : 'N/A')}</span>
                        ${t.region ? `<span>Region: ${t.region}</span>` : ''}
                        ${hasClickableSources ?
                            `<span class="sources-link" onclick="showTrendSources('${t.id}')" style="cursor:pointer;text-decoration:underline;">${sourceCount} sources (click to view)</span>` :
                            (sourceCount ? `<span>${sourceCount} sources</span>` : '')}
                    </div>
                </div>
            `;
        }

        // Show sources modal for a trend
        async function showTrendSources(trendId) {
            try {
                const response = await fetch(`/api/social-pulse/${trendId}/sources`);
                if (!response.ok) throw new Error('Failed to load sources');

                const data = await response.json();

                // Create modal
                const modal = document.createElement('div');
                modal.id = 'sources-modal';
                modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:1000;display:flex;justify-content:center;align-items:center;';
                modal.onclick = (e) => { if (e.target === modal) modal.remove(); };

                const content = document.createElement('div');
                content.style.cssText = 'background:white;padding:30px;border-radius:12px;max-width:800px;max-height:80vh;overflow-y:auto;width:90%;';

                let sourcesHtml = `
                    <h2 style="margin-bottom:20px;">Sources for: ${data.trend_name}</h2>
                    <p style="color:#666;margin-bottom:20px;">${data.total} source(s) contributed to this trend</p>
                `;

                if (data.sources && data.sources.length > 0) {
                    sourcesHtml += data.sources.map(s => `
                        <div style="border-left:4px solid #667eea;padding:15px;margin:10px 0;background:#f8f9fa;border-radius:0 8px 8px 0;">
                            <h4 style="margin-bottom:8px;">
                                ${s.url ? `<a href="${s.url}" target="_blank" style="color:#667eea;text-decoration:none;">${truncate(s.title || 'Untitled', 60)}</a>` : truncate(s.title || 'Untitled', 60)}
                            </h4>
                            <p style="color:#555;font-size:0.9em;margin-bottom:8px;">${truncate(s.content_preview || '', 200)}</p>
                            <div style="font-size:0.8em;color:#888;">
                                <span style="background:#e94560;color:white;padding:2px 8px;border-radius:4px;margin-right:10px;">${s.source}</span>
                                ${s.author ? `by ${s.author}` : ''}
                                ${s.published_at ? ` • ${new Date(s.published_at).toLocaleDateString()}` : ''}
                            </div>
                        </div>
                    `).join('');
                } else {
                    sourcesHtml += '<p style="text-align:center;color:#888;padding:20px;">No sources available</p>';
                }

                sourcesHtml += '<button onclick="this.closest(\'#sources-modal\').remove()" style="margin-top:20px;padding:10px 20px;background:#e94560;color:white;border:none;border-radius:6px;cursor:pointer;">Close</button>';

                content.innerHTML = sourcesHtml;
                modal.appendChild(content);
                document.body.appendChild(modal);

            } catch (error) {
                alert('Failed to load sources: ' + error.message);
            }
        }

        function renderMove(m) {
            return `
                <div class="move-card">
                    <h3>${m.title || 'Untitled Move'}</h3>
                    <div class="company">${m.company || 'Unknown Company'} ${m.move_type ? '• ' + m.move_type : ''}</div>
                    <p>${m.summary || m.why_it_matters || 'No summary available'}</p>
                    <div class="trend-meta" style="margin-top:15px;">
                        ${m.market ? `<span>${m.market}</span>` : ''}
                        ${m.confidence ? `<span>Confidence: ${(m.confidence * 100).toFixed(0)}%</span>` : ''}
                    </div>
                </div>
            `;
        }

        function renderProperty(p) {
            // Ensure we have a meaningful property name
            let propertyName = p.name || p.property_name || '';
            if (!propertyName || propertyName.toLowerCase().includes('unknown') || propertyName.toLowerCase().includes('untitled')) {
                // Try to extract from URL
                if (p.url) {
                    try {
                        const url = new URL(p.url);
                        let domain = url.hostname.replace('www.', '');
                        let name = domain.split('.')[0];
                        name = name.replace(/-/g, ' ').replace(/_/g, ' ');
                        propertyName = name.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') + ' Property';
                    } catch (e) {
                        propertyName = 'Scanned Property';
                    }
                } else {
                    propertyName = 'Analyzed Property';
                }
            }

            return `
                <div class="property-card">
                    <h3>${propertyName}</h3>
                    <div class="type">${p.property_type || 'Hotel'} ${p.price_segment && p.price_segment !== 'unknown' ? '• ' + p.price_segment : ''}</div>
                    ${p.demand_fit_score ? `<div class="property-score">Demand Fit: ${(p.demand_fit_score * 100).toFixed(0)}%</div>` : ''}
                    ${p.brand_positioning ? `<p>${truncate(p.brand_positioning, 200)}</p>` : (p.positioning_statement ? `<p>${truncate(p.positioning_statement, 200)}</p>` : '')}
                    ${p.themes && p.themes.length > 0 ? `
                        <div class="property-themes">
                            ${p.themes.slice(0, 5).map(t => `<span>${t}</span>`).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        }

        function renderContent(c) {
            return `
                <div class="content-item">
                    <h4>${truncate(c.title || 'Untitled', 80)}</h4>
                    <p>${truncate(c.content || c.summary || '', 200)}</p>
                    <div class="meta">
                        <span class="source">${c.source || 'unknown'}</span>
                        ${c.author ? `by ${c.author}` : ''}
                        ${c.published_at ? ` • ${new Date(c.published_at).toLocaleDateString()}` : ''}
                    </div>
                </div>
            `;
        }

        function renderEmptyState(type, message, hint) {
            const icons = {trends: '📈', moves: '♟️', properties: '🏨', content: '📰'};
            return `
                <div class="empty-state">
                    <div class="icon">${icons[type] || '📊'}</div>
                    <p>${message}</p>
                    <div class="hint">
                        <strong>To generate data:</strong><br>
                        <code>${hint}</code>
                    </div>
                </div>
            `;
        }

        // City Desires functions
        async function analyzeCity(city, country) {
            // Get values from inputs if not provided
            if (!city) {
                city = document.getElementById('city-input').value.trim();
                country = document.getElementById('country-input').value.trim();
            }

            if (!city) {
                alert('Please enter a city name');
                return;
            }

            const resultsDiv = document.getElementById('city-results');
            resultsDiv.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing ${city}${country ? ', ' + country : ''}...</p>
                    <p style="font-size:0.85em;margin-top:10px;color:#888;">This may take 30-60 seconds as we scrape Reddit, YouTube, and travel forums</p>
                </div>
            `;

            try {
                const url = `/api/city-desires/quick?city=${encodeURIComponent(city)}${country ? '&country=' + encodeURIComponent(country) : ''}`;
                const response = await fetch(url);

                if (!response.ok) {
                    throw new Error(`Analysis failed: ${response.status}`);
                }

                const data = await response.json();
                renderCityResults(data);
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div class="empty-state">
                        <div class="icon">❌</div>
                        <p>Failed to analyze city: ${error.message}</p>
                        <div class="hint">
                            Make sure the API server is running and try again.
                        </div>
                    </div>
                `;
            }
        }

        function renderCityResults(data) {
            const resultsDiv = document.getElementById('city-results');

            // Build HTML
            let html = `
                <div class="city-summary">
                    <h2>🌆 ${data.city}${data.country ? ', ' + data.country : ''}</h2>
                    <div class="summary-metrics">
                        <div class="summary-metric">
                            <div class="value">${data.total_signals || 0}</div>
                            <div class="label">Signals Found</div>
                        </div>
                        <div class="summary-metric">
                            <div class="value">${data.total_sources || 0}</div>
                            <div class="label">Sources</div>
                        </div>
                        <div class="summary-metric">
                            <div class="value">${((data.avg_frustration || 0) * 100).toFixed(0)}%</div>
                            <div class="label">Avg Frustration</div>
                        </div>
                        <div class="summary-metric">
                            <div class="value">${(data.top_desires || []).length}</div>
                            <div class="label">Desire Themes</div>
                        </div>
                    </div>
                </div>
            `;

            // White Space Opportunities
            const whiteSpace = data.white_space_opportunities || [];
            if (whiteSpace.length > 0) {
                html += `
                    <h3 style="color:white;margin:20px 0 10px;">🎯 White Space Opportunities</h3>
                `;
                whiteSpace.forEach(opp => {
                    html += `<div class="opportunity-card"><h3>${opp}</h3></div>`;
                });
            }

            // Top Desires
            const desires = data.top_desires || [];
            if (desires.length > 0) {
                html += `
                    <h3 style="color:white;margin:20px 0 10px;">🔥 Top Unmet Desires</h3>
                `;
                desires.slice(0, 5).forEach(d => {
                    html += `
                        <div class="desire-theme">
                            <h3>${d.theme_name || 'Unnamed Theme'}</h3>
                            <p>${d.description || ''}</p>
                            <div class="desire-scores">
                                <span>Intensity: ${((d.intensity_score || 0) * 100).toFixed(0)}%</span>
                                <span>Frustration: ${((d.frustration_score || 0) * 100).toFixed(0)}%</span>
                                <span>Opportunity: ${((d.opportunity_score || 0) * 100).toFixed(0)}%</span>
                                <span>${d.frequency || 0} mentions</span>
                            </div>
                            ${d.segments && d.segments.length > 0 ? `
                                <div style="margin-top:10px;">
                                    ${d.segments.map(s => `<span class="segment-badge">${s}</span>`).join('')}
                                </div>
                            ` : ''}
                        </div>
                    `;
                });
            }

            // Underserved Segments
            const underserved = data.underserved_segments || [];
            if (underserved.length > 0) {
                html += `
                    <h3 style="color:white;margin:20px 0 10px;">👥 Underserved Segments</h3>
                    <div class="card" style="padding:15px;">
                        ${underserved.map(s => `<span class="badge badge-warning" style="margin:3px;">${s}</span>`).join('')}
                    </div>
                `;
            }

            // Concept Lanes
            const concepts = data.concept_lanes || [];
            if (concepts.length > 0) {
                html += `
                    <h3 style="color:white;margin:20px 0 10px;">💡 Recommended Hotel Concepts</h3>
                `;
                concepts.forEach(c => {
                    html += `
                        <div class="concept-card">
                            <h3>${c.concept || 'New Concept'}</h3>
                            <p class="rationale">${c.rationale || ''}</p>
                            ${c.target_segments && c.target_segments.length > 0 ? `
                                <div style="margin-top:10px;">
                                    <strong style="font-size:0.85em;">Target:</strong>
                                    ${c.target_segments.map(s => `<span class="segment-badge">${s}</span>`).join('')}
                                </div>
                            ` : ''}
                            ${c.key_features && c.key_features.length > 0 ? `
                                <div style="margin-top:8px;">
                                    <strong style="font-size:0.85em;">Features:</strong>
                                    ${c.key_features.map(f => `<span class="segment-badge">${f}</span>`).join('')}
                                </div>
                            ` : ''}
                            ${c.opportunity_score ? `
                                <div style="margin-top:10px;">
                                    <span style="background:rgba(255,255,255,0.3);padding:5px 12px;border-radius:15px;font-size:0.9em;">
                                        Opportunity Score: ${(c.opportunity_score * 100).toFixed(0)}%
                                    </span>
                                </div>
                            ` : ''}
                        </div>
                    `;
                });
            }

            // Empty state if no results
            if (desires.length === 0 && whiteSpace.length === 0) {
                html += `
                    <div class="empty-state" style="margin-top:20px;">
                        <div class="icon">🤷</div>
                        <p>No desire signals found for this city</p>
                        <div class="hint">
                            Try a more popular tourist destination or check the spelling
                        </div>
                    </div>
                `;
            }

            resultsDiv.innerHTML = html;
        }

        // Load data on page load
        loadData();

        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
