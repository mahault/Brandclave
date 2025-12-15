# BrandClave Aggregator

**Data Engine for Hospitality Demand Intelligence, Trend Signals, and Hotelier Moves**

---

## Quick Start for Demo (Sarah's Guide)

### First Time Setup (One Time Only)

1. **Install Miniconda** (if not already installed)
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Choose "Miniconda3 Windows 64-bit"
   - Run installer with default settings
   - Restart your computer

2. **Get the Project**
   - Download/clone this project to your computer
   - Remember where you put it!

3. **Run First-Time Setup**
   - Open the project folder
   - Double-click **`SETUP_FIRST_TIME.bat`**
   - Follow the prompts (takes ~5-10 minutes)
   - When Notepad opens, paste your Mistral API key and save

### Running the Demo

Just double-click **`START_DEMO.bat`**

That's it! The browser will open automatically to the API docs.

### What to Show Investors

The browser opens to the **Monitoring Dashboard** - a real-time view of the system.

**Demo Flow:**

1. **Dashboard** (opens automatically) - http://localhost:8000/api/monitoring/dashboard
   - Shows system health at a glance
   - Content metrics, scraper status, recent activity
   - Auto-refreshes every 30 seconds

2. **Social Pulse** - Go to http://localhost:8000/docs → "GET /api/social-pulse" → "Try it out" → "Execute"
   - Shows AI-detected travel trends from Reddit & YouTube
   - Real data: trend names, descriptions, "why it matters"
   - Example: "Digital Nomad Coliving Spaces" trend

3. **Hotelier Bets** - Click "GET /api/hotelier-bets" → "Try it out" → "Execute"
   - Shows strategic moves extracted from hospitality news
   - AI identifies: company, move type, market impact
   - Example: "Marriott Acquires Boutique Hotel Group"

4. **Demand Scan** - Click "POST /api/demand-scan" → "Try it out" → Enter a hotel URL → "Execute"
   - Analyzes any hotel website instantly
   - Extracts: property type, themes, amenities, positioning
   - Shows: demand fit score, experience gaps, recommendations
   - Example URL: `https://acehotel.com/new-york/`

**Talking Points:**
- "This is real data scraped from Reddit, YouTube, TripAdvisor, Booking.com, and Skift news"
- "AI automatically clusters conversations into trends"
- "Each trend includes 'why it matters' for hotel strategists"
- "We extract strategic moves from news articles automatically"
- "Demand Scan analyzes any hotel website in seconds - instant competitive intelligence"
- "The dashboard shows real-time system health and scraper status"
- "Automated scheduling keeps data fresh without manual intervention"

### To Stop

Press `Ctrl+C` in the black terminal window, or just close it.

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "Conda not found" | Install Miniconda and restart computer |
| Script closes immediately | Right-click → Run as Administrator |
| "Environment not found" | Run SETUP_FIRST_TIME.bat again |
| Browser doesn't open | Manually go to http://localhost:8000/docs |

---

## Overview

BrandClave Aggregator is the backend data intelligence system powering:

### What's Hot in Hospitality

- **Social Pulse**: emerging guest desires extracted from social media, reviews, and UGC
- **Hotelier Bets**: strategic moves scraped from hospitality news, deals, and press releases

### Demand Scan → Build-a-Brand Flow

- Turn property URLs into structured analyses of positioning, gaps, demand fit, and opportunity lanes
- Provide region-specific demand signals, white-space detection, and competitor moves

The Aggregator collects, normalizes, enriches, and serves structured insights through a clean API layer used by the BrandClave platform. It is designed as a modular, extensible data engine capable of incorporating new sources, updating models, and scaling in volume and scope.

## Project Definition

BrandClave Aggregator implements four major capabilities:

### 1. Data Ingestion Layer: Scrapers & Connectors

Pull data from:

- Social and UGC platforms (posts, hashtags, reviews, discussions)
- Hospitality news feeds (RSS, press releases, M&A announcements)
- OTA listings (supply and competitive density)
- Property websites (copy + images for Demand Scan)

Each source produces a standardized `RawContent` object that is persisted for downstream processing.

### 2. Content Normalization & Enrichment

A unified NLP pipeline processes raw data to generate:

- Cleaned text, extracted entities, sentiment scores
- Embeddings for clustering and semantic search
- Trend clusters over time windows
- Topic assignment and region/segment mapping

These are used to derive higher-level constructs like trend strength, engagement shifts, and white-space scores.

### 3. Intelligence Layer: Signals & Moves

The Aggregator outputs two structured knowledge streams:

**Trend Signals (Social Pulse)**

- Trend name
- Strength indicator (volume, engagement, sentiment deltas)
- Why it matters
- White-space score (demand vs supply imbalance)
- Region, audience segment

**Hotelier Moves (Hotelier Bets)**

- Action title
- Company + market
- Move type (launch, acquisition, repositioning, reflag, concept)
- Why it matters
- Strategic implications

These two streams form the core industry map used throughout BrandClave.

### 4. Demand Scan & Opportunity Modeling

Given a hotel/property URL, the system:

- Scrapes and parses website content
- Extracts positioning, themes, amenities, pricing signals
- Compares features to real-time demand (Trend Signals)
- Identifies experience gaps & misalignments
- Outputs Demand Fit Score, Opportunity Lanes, and key recommendations

This module is the analytical bridge between market reality and brand concepting.

## Repository Structure

A clean, modular layout supporting incremental development:

```
brandclave-aggregator/
│
├── README.md
│
├── configs/
│   ├── sources.yaml          # Registry of all data sources
│   ├── scraping.yaml         # Frequency, rules, user-agents, delays
│   └── nlp.yaml              # Model config, thresholds, clustering params
│
├── data_models/
│   ├── raw_content.py        # RawContent schema
│   ├── trend_signal.py       # TrendSignal schema
│   ├── hotelier_move.py      # HotelierMove schema
│   ├── property_features.py  # Property feature extraction schema
│   └── embeddings.py         # Embedding store & utilities
│
├── ingestion/
│   ├── base_scraper.py       # Abstract scraper class
│   ├── social/
│   │   ├── reddit_scraper.py
│   │   ├── tiktok_scraper.py
│   │   └── instagram_scraper.py
│   │
│   ├── reviews/
│   │   ├── tripadvisor_scraper.py
│   │   ├── booking_scraper.py
│   │   └── google_reviews_scraper.py
│   │
│   ├── news/
│   │   ├── hospitalitynet_rss.py
│   │   ├── skift_scraper.py
│   │   └── hotelnewsnow_scraper.py
│   │
│   └── properties/
│       └── property_page_scraper.py
│
├── processing/
│   ├── cleaning.py           # HTML cleanup, normalization
│   ├── nlp_pipeline.py       # Sentiment + embeddings + NER
│   ├── clustering.py         # Trend clustering by region/time
│   ├── scoring.py            # Trend strength + white-space scoring
│   ├── move_extraction.py    # Extract structured Hotelier Bets
│   └── property_analysis.py  # Demand Scan feature extraction + scoring
│
├── services/
│   ├── social_pulse.py       # Build social pulse cards
│   ├── hotelier_bets.py      # Build hotelier move cards
│   ├── demand_scan.py        # Property analysis endpoint
│   └── orchestrator.py       # High-level workflow orchestration
│
├── api/
│   ├── main.py               # FastAPI / NestJS root server
│   ├── routes/
│   │   ├── social_pulse.py
│   │   ├── hotelier_bets.py
│   │   ├── demand_scan.py
│   │   └── healthcheck.py
│   └── utils/
│       └── exceptions.py
│
├── db/
│   ├── migrations/
│   ├── models.py
│   └── vector_index.py
│
├── tests/
│   ├── ingestion/
│   ├── processing/
│   ├── api/
│   └── fixtures/
│
└── scripts/
    ├── run_crawlers.py
    ├── rebuild_embeddings.py
    └── backfill_sources.py
```

This structure cleanly separates data, logic, APIs, and ops, and is easy to extend for new markets, new scrapers, or new intelligence layers.

## Roadmap

A structured development plan from MVP to full system.

### Phase 1 — Core Foundations (COMPLETE)

**Goal**: System skeleton + minimal ingestion + schemas

**Tech Stack:**
- Database: SQLite + ChromaDB (local development)
- Embeddings: Mistral Embed API with local sentence-transformers fallback
- First scraper: Skift RSS (HospitalityNet deprecated)

**Deliverables:**

1. **Dependencies & Environment**
   - `requirements.txt` with all dependencies
   - `.env.example` for configuration
   - Updated `environment.yml` for conda

2. **Data Models** (`data_models/`)
   - `raw_content.py` - RawContent schema (id, source, url, title, content, author, published_at, scraped_at, metadata, embedding_id)
   - `trend_signal.py` - TrendSignal schema
   - `hotelier_move.py` - HotelierMove schema
   - `property_features.py` - Property extraction schema
   - `embeddings.py` - Embedding utilities

3. **Database Layer** (`db/`)
   - `database.py` - SQLite connection setup, session management
   - `models.py` - SQLAlchemy models (raw_content, trend_signals, hotelier_moves, processing_jobs)
   - `vector_store.py` - ChromaDB wrapper for vector search

4. **NLP Pipeline** (`processing/`)
   - `cleaning.py` - HTML cleanup, text normalization
   - `nlp_pipeline.py` - Orchestrator: language detection (langdetect), sentiment (TextBlob), embeddings (Mistral/local)
   - `configs/nlp.yaml` - Model configuration

5. **Scraper Infrastructure** (`ingestion/`)
   - `base_scraper.py` - Abstract base class with rate limiting, retry logic, robots.txt checking
   - `news/skift_rss.py` - Hospitality news RSS scraper

6. **Scripts** (`scripts/`)
   - `init_db.py` - Initialize SQLite + ChromaDB
   - `run_crawlers.py` - CLI to run scrapers
   - `check_db.py` - Database statistics

**Output**: Database populated with raw items + simple processing pipeline.

### Phase 2 — Social Pulse MVP (COMPLETE)

**Goal**: first working trend signal engine

**Implemented:**
- Reddit scraper (old.reddit.com JSON endpoints, no API key required)
- YouTube search scraper (no API key required)
- HDBSCAN clustering of embeddings by semantic similarity
- Trend strength metrics (volume, engagement, sentiment delta)
- White-space scoring (demand vs supply imbalance)
- Mistral LLM for trend names and "Why it matters" generation
- `/api/social-pulse` endpoint with filtering
- Semantic search endpoint

**Output**: Working Social Pulse API returning real trend data.

### Phase 3 — Hotelier Bets MVP (COMPLETE)

**Goal**: structured competitor intelligence

**Implemented:**
- LLM-powered move extraction from news content
- Extracts: company, company type, move type, market, investment amount
- Generates: title, summary, why it matters, strategic implications
- Competitive impact analysis
- Confidence scoring with threshold filtering
- `/api/hotelier-bets` endpoint with filtering
- CLI command for move extraction

**Output**: Live competitor moves extracted from hospitality news.

### Phase 4 — Demand Scan MVP (COMPLETE)

**Goal**: property-level analysis + opportunity mapping

**Implemented:**
- Property page scraper (BeautifulSoup + httpx)
- LLM-powered feature extraction (name, type, positioning, amenities, themes)
- Region detection from content
- Price segment classification
- Demand fit scoring against regional trends
- Experience gap identification
- Opportunity lane generation
- Strategic recommendations
- `/api/demand-scan` endpoint with scan, refresh, and list
- CLI command for property scanning

**Output**: Full pipeline from property URL → insights → recommendations.

### Phase 5 — Scaling, Reliability & Expansion (COMPLETE)

**Goal**: robustness, accuracy, geographic expansion

**Implemented:**
- **Redis Caching Layer**: HTTP response caching, embedding caching, API response caching with graceful degradation
- **Review Scrapers**: TripAdvisor and Booking.com scrapers with conservative rate limiting
- **APScheduler Integration**: Background scheduling with SQLite job persistence, configurable intervals per source
- **Monitoring Dashboard**: Real-time system metrics, scraper status, error tracking, HTML dashboard with auto-refresh
- **Scheduler API**: Job management endpoints (list, add, remove, pause, resume, run now)
- **Health Monitoring**: Comprehensive health checks for database, cache, and scheduler

**Output**: A production-grade intelligence engine with automated scheduling and real-time monitoring.

## Current Status

### Phase 1 - Core Foundations (COMPLETE)
- SQLite + ChromaDB database layer
- Mistral Embed API for embeddings
- Skift RSS scraper (HospitalityNet deprecated)
- NLP pipeline with sentiment analysis and language detection
- Base scraper infrastructure with rate limiting

### Phase 2 - Social Pulse MVP (COMPLETE)
- Reddit scraper (old.reddit.com JSON endpoints, no API key)
- YouTube search scraper (no API key)
- HDBSCAN clustering for trend detection
- Trend scoring (strength, white-space, engagement)
- Mistral LLM for trend names and "Why it matters" generation
- FastAPI endpoints for Social Pulse

### Phase 3 - Hotelier Bets MVP (COMPLETE)
- LLM-powered move extraction from news articles
- Extracts: company, move type, market, strategic implications
- Generates "Why it matters" and competitive impact analysis
- FastAPI endpoints for Hotelier Bets
- CLI command for move extraction

### Phase 4 - Demand Scan MVP (COMPLETE)
- Property page scraping and analysis
- LLM-powered feature extraction
- Demand fit scoring against regional trends
- Experience gap and opportunity identification
- Strategic recommendations generation
- FastAPI endpoints for Demand Scan
- CLI command for property scanning

### Phase 5 - Scaling & Expansion (COMPLETE)
- Redis caching with graceful degradation (works without Redis)
- TripAdvisor scraper (conservative rate limiting)
- Booking.com scraper (conservative rate limiting)
- APScheduler with SQLite job persistence
- Monitoring dashboard with auto-refresh
- Scheduler API for job management
- Health monitoring endpoints

## Getting Started

### Prerequisites

- Python 3.11+
- Conda (recommended) or virtualenv
- Mistral API key (for embeddings and LLM)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Brandclave

# Create conda environment
conda env create -f environment.yml
conda activate brandclave

# Or use pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your MISTRAL_API_KEY

# Initialize database
python scripts/init_db.py

# Run scrapers
python scripts/run_crawlers.py --all

# Process content with NLP pipeline
python scripts/run_crawlers.py --process

# Generate trends
python scripts/run_crawlers.py --trends

# Start the API server
uvicorn api.main:app --reload
```

### API Endpoints

```
GET  /                           - API info
GET  /health                     - Health check

# Social Pulse (Trend Signals)
GET  /api/social-pulse           - List trend signals
GET  /api/social-pulse/{id}      - Single trend detail
POST /api/social-pulse/search/semantic - Semantic search
POST /api/social-pulse/generate  - Trigger trend generation

# Hotelier Bets (Strategic Moves)
GET  /api/hotelier-bets          - List strategic moves
GET  /api/hotelier-bets/{id}     - Single move detail
GET  /api/hotelier-bets/companies - List companies
GET  /api/hotelier-bets/move-types - List move types
POST /api/hotelier-bets/generate - Extract moves from news

# Demand Scan (Property Analysis)
POST /api/demand-scan            - Scan a property URL
POST /api/demand-scan/refresh    - Rescan a property
GET  /api/demand-scan            - List scanned properties
GET  /api/demand-scan/{id}       - Single property detail
GET  /api/demand-scan/property-types - List property types
GET  /api/demand-scan/price-segments - List price segments

# Monitoring (System Status)
GET  /api/monitoring/health      - Comprehensive health check
GET  /api/monitoring/metrics     - System metrics
GET  /api/monitoring/scrapers    - All scraper metrics
GET  /api/monitoring/scrapers/{source} - Single scraper metrics
GET  /api/monitoring/errors      - Recent job errors
GET  /api/monitoring/activity    - Recent activity summary
GET  /api/monitoring/dashboard   - HTML monitoring dashboard

# Scheduler (Job Management)
GET  /api/scheduler/status       - Scheduler status
GET  /api/scheduler/jobs         - List all scheduled jobs
GET  /api/scheduler/jobs/{id}    - Single job detail
POST /api/scheduler/jobs         - Add a new job
DELETE /api/scheduler/jobs/{id}  - Remove a job
POST /api/scheduler/jobs/{id}/pause  - Pause a job
POST /api/scheduler/jobs/{id}/resume - Resume a job
POST /api/scheduler/jobs/{id}/run    - Run job immediately
POST /api/scheduler/start        - Start the scheduler
POST /api/scheduler/stop         - Stop the scheduler
```

### CLI Commands

```bash
# List available scrapers
python scripts/run_crawlers.py --list

# Run specific scraper
python scripts/run_crawlers.py --source reddit
python scripts/run_crawlers.py --source youtube
python scripts/run_crawlers.py --source skift
python scripts/run_crawlers.py --source tripadvisor
python scripts/run_crawlers.py --source booking

# Run all scrapers
python scripts/run_crawlers.py --all

# Process unprocessed content
python scripts/run_crawlers.py --process --limit 100

# Generate Social Pulse trends
python scripts/run_crawlers.py --trends --days 30

# Extract Hotelier Bets moves
python scripts/run_crawlers.py --moves --days 30

# Scan a property URL with Demand Scan
python scripts/run_crawlers.py --scan "https://acehotel.com/new-york/"

# Check database stats
python scripts/check_db.py
```

### Running Tests

```bash
pytest tests/
```

## Contributing

PRs welcome!

- Please annotate new scrapers with ToS-compliance notes
- Add new models to `configs/nlp.yaml` before use
- Run tests with `pytest` before submitting

## License

TBD (commercial, private to BrandClave)
