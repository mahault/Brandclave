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
                    fetch('/api/monitoring/content?limit=10').then(r => r.json()).catch(() => ({items: []})),
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
            return `
                <div class="trend-card">
                    <h3>${t.trend_name || 'Unnamed Trend'}</h3>
                    <p>${t.description || t.why_it_matters || 'No description available'}</p>
                    <div class="trend-meta">
                        <span>Strength: ${t.strength_label || t.trend_strength || 'N/A'}</span>
                        ${t.region ? `<span>Region: ${t.region}</span>` : ''}
                        ${t.content_count ? `<span>${t.content_count} sources</span>` : ''}
                    </div>
                </div>
            `;
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
            return `
                <div class="property-card">
                    <h3>${p.property_name || 'Unknown Property'}</h3>
                    <div class="type">${p.property_type || 'Hotel'} ${p.price_segment ? '• ' + p.price_segment : ''}</div>
                    ${p.demand_fit_score ? `<div class="property-score">Demand Fit: ${(p.demand_fit_score * 100).toFixed(0)}%</div>` : ''}
                    ${p.positioning_statement ? `<p>${truncate(p.positioning_statement, 200)}</p>` : ''}
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

        // Load data on page load
        loadData();

        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
