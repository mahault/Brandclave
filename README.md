# BrandClave Aggregator

**Data Engine for Hospitality Demand Intelligence, Trend Signals, and Hotelier Moves**

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

### Phase 3 — Hotelier Bets MVP (Weeks 5–7)

**Goal**: structured competitor intelligence

- Add 2–3 hospitality news sources
- Implement move extraction (company, market, move type)
- Auto-generate "Why it matters" and strategic implications
- Build `/api/hotelier-bets` endpoint
- Integrate "Turn Into Brand Concept" pathway

**Output**: Live competitor moves powering BrandClave's inspiration flow.

### Phase 4 — Demand Scan MVP (Weeks 7–10)

**Goal**: property-level analysis + opportunity mapping

- Implement property-page scraper
- Extract brand positioning & features (themes, amenities, tone)
- Compute Demand Fit Score vs regional trends
- Identify Experience Gaps and Opportunity Lanes
- Build `/api/demand-scan` endpoint
- Connect to Build-a-Brand module

**Output**: First full pipeline from property URL → insights → brand concepting.

### Phase 5 — Scaling, Reliability & Expansion (Ongoing)

**Goal**: robustness, accuracy, geographic expansion

- Add more markets, more scrapers, more languages
- Improve trend scoring (e.g., engagement time derivatives, anomaly detection)
- Build supply-side database (OTA listings, hotel categories, density)
- Add A/B evaluation on accuracy of trend and move extraction
- Introduce caching + rate limiting + error resilience
- Implement unified scheduler + monitoring dashboard

**Output**: A production-grade intelligence engine supporting global hospitality markets.

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

**Current Data:**
- 552 content items scraped
- 70 items processed with embeddings
- 3 trend clusters identified

### Phase 3 - Hotelier Bets MVP (IN PROGRESS)
- Next: News source expansion and move extraction

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
GET  /api/social-pulse           - List trend signals
GET  /api/social-pulse/{id}      - Single trend detail
POST /api/social-pulse/search/semantic - Semantic search
POST /api/social-pulse/generate  - Trigger trend generation
```

### CLI Commands

```bash
# List available scrapers
python scripts/run_crawlers.py --list

# Run specific scraper
python scripts/run_crawlers.py --source reddit
python scripts/run_crawlers.py --source youtube
python scripts/run_crawlers.py --source skift

# Run all scrapers
python scripts/run_crawlers.py --all

# Process unprocessed content
python scripts/run_crawlers.py --process --limit 100

# Generate trends
python scripts/run_crawlers.py --trends --days 30

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

TBD (likely commercial, private to BrandClave)
