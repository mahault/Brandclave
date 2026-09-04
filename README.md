# BrandClave Aggregator

**AI-Powered Hospitality Intelligence Platform with Active Inference**

An intelligent system that scrapes hospitality news and social media, uses **PyMDP active inference** to adaptively decide what to collect, and generates actionable trend insights.

---

## Quick Start (Non-Technical Users)

### First Time Setup (10 minutes)

1. **Double-click `SETUP_FIRST_TIME.bat`**
   - Installs Python/Conda automatically if needed
   - Creates the environment with all dependencies
   - When Notepad opens, paste your **Mistral API key** and save
   - Wait for "Setup complete!" message

2. **Double-click `POPULATE_DATA.bat`**
   - Scrapes content from 12 reliable hospitality sources
   - Processes content with AI (embeddings, clustering)
   - Generates trends and extracts strategic moves
   - Takes about 10-15 minutes

### Running the Dashboard

**Double-click `START_DEMO.bat`**

- Opens the dashboard at http://localhost:8000/api/monitoring/dashboard-v2
- Shows existing data immediately (no waiting)
- Background scheduler keeps data fresh automatically

### What You'll See

| Tab | What It Shows |
|-----|---------------|
| **Overview (Signal Room)** | KPIs with 7-day deltas, demand curves per city (Wikimedia pageviews, indexed), trend movers, latest operator bets, the active-inference attention model, source coverage; stake a prediction from any trend |
| **City Desires** | Type a city to discover what travelers want but can't find |
| **Social Pulse** | AI-detected travel trends from social conversations |
| **Hotelier Bets** | Strategic moves extracted from hospitality news |
| **Demand Scan** | Analyze any hotel website against current demand trends: semantic fit (mistral-embed), alignment per trend, gaps, adjacent white space, and an LLM-written brief |
| **Signal Ledger** | Sealed, hash-stamped predictions with forecasts, falsifiers, evidence trail and accuracy KPIs |
| **Content** | Latest scraped articles and posts |
| **Scrapers** | Status of each data source |
| **Chat** | RAG-powered assistant over the platform's data |
| **My Projects** | Saved trends/moves, interest profile, and brand blueprints |

The dashboard follows the SENTIENT design language: warm near-black surfaces, champagne-gold accents, Archivo display type and monospace labels, with a colorblind-safe accent palette (validated in OKLCH for CVD separation and contrast).

---

## How It Works

### Architecture Overview

```
                    +------------------+
                    |   PyMDP/JAX      |
                    | Active Inference |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v----+  +------v------+  +---v--------+
     | Scraping    |  | Clustering  |  | Extraction |
     | POMDP       |  | POMDP       |  | POMDP      |
     +--------+----+  +------+------+  +---+--------+
              |              |              |
     +--------v----+  +------v------+  +---v--------+
     | 12 Sources  |  | Embeddings  |  | LLM/NER    |
     | Reddit/News |  | HDBSCAN     |  | Analysis   |
     +--------+----+  +------+------+  +---+--------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |    Dashboard     |
                    | Trends & Moves   |
                    +------------------+
```

### The Intelligence Layer: PyMDP Active Inference

BrandClave uses **PyMDP** (a JAX-based active inference library) to make adaptive decisions. Instead of fixed rules, the system learns which actions yield the best information.

#### What is Active Inference?

Active inference is a framework where an agent:
1. **Has beliefs** about hidden states (e.g., "Is Reddit productive right now?")
2. **Makes observations** (e.g., "I scraped 50 items with 2 errors")
3. **Updates beliefs** based on observations (Bayesian inference)
4. **Selects actions** that minimize **Expected Free Energy** (EFE)

EFE balances two goals:
- **Pragmatic value**: Actions that lead to preferred outcomes
- **Epistemic value**: Actions that reduce uncertainty (exploration)

#### POMDPs in BrandClave

| POMDP | Decision | How It Works |
|-------|----------|--------------|
| **ScrapingPOMDP** | Which source to scrape next | Tracks productivity/freshness of each source. Prefers sources with high expected yield but also explores uncertain ones. |
| **ClusteringPOMDP** | Which clustering parameters to use | Adapts min_cluster_size, min_samples based on data characteristics. Learns which settings produce best clusters. |
| **MoveExtractionPOMDP** | Which extraction method to use | Chooses between LLM, NER, or keyword extraction based on content type and past success. |
| **CoordinatorPOMDP** | Which task to prioritize | Balances scraping vs processing vs analysis based on system state. |

#### Example: ScrapingPOMDP

```python
# Hidden states: [high_productivity, medium, low, stale]
# Observations: [items_scraped, freshness, error_rate]
# Actions: [scrape_reddit, scrape_skift, ..., wait]

# The A matrix defines P(observation | state)
A_productivity = [
    [0.7, 0.5, 0.2, 0.1],  # P(high_obs | state)
    [0.2, 0.4, 0.5, 0.3],  # P(med_obs | state)
    [0.1, 0.1, 0.3, 0.6],  # P(low_obs | state)
]

# PyMDP Agent selects actions via EFE minimization
q_pi, G = agent.infer_policies(beliefs)
best_action = argmin(G)  # Lowest EFE = best action
```

When you run the system, you'll see logs like:
```
Scraping POMDP enabled for adaptive source selection
Initialized Scraping POMDP with 12 sources (JAX/JIT enabled)
Clustering POMDP enabled for adaptive parameter selection
```

### Data Sources (12 Reliable)

**Social Media:**
- Reddit (r/hotels, r/travel, r/digitalnomad, etc.)
- YouTube (hotel reviews, travel vlogs)

**Hospitality News:**
- Skift, Hotel Dive, Hotel Management
- Top Hotel News, SiteMinder, EHL Insights
- eHotelier, Lodging Magazine, Luxury Hospitality, Hotel Business

### Processing Pipeline

1. **Scraping** - Adaptive source selection via ScrapingPOMDP
2. **Embeddings** - Mistral AI converts text to vectors
3. **Clustering** - HDBSCAN groups similar content (params via ClusteringPOMDP)
4. **Trend Detection** - LLM generates trend names and descriptions
5. **Move Extraction** - Extracts strategic moves from news (method via MoveExtractionPOMDP)
6. **Quality Filtering** - Removes low-quality trends automatically

---

## Features

### Social Pulse (Trends)
- Clusters social conversations into trend signals
- Each trend has: name, description, strength score, region
- **Click any trend card** to expand and see full details, topics, and source quotes
- Quality filtering removes garbage like "Tour Advice There Trend"

### Hotelier Bets (Strategic Moves)
- Extracts company moves from news: expansions, acquisitions, launches
- Shows company, move type, market, strategic implications
- **Click any move card** to expand and see full summary and source links

### City Desires
- Type any city (Lisbon, Tokyo, Barcelona, etc.)
- Scrapes Reddit/YouTube for that city in real-time
- Shows: unmet traveler needs, frustration points, white space opportunities
- Recommends hotel concepts based on gaps

### Brand Blueprints (Build a Brand)
- Turns a trend, a saved-research profile, or manual inputs into a full hotel brand concept
- Five-stage pipeline: foundation, strategic, experience, atmosphere, investor summary
- Blueprints are persisted and browsable from **My Projects**

### Signal Ledger
- BrandClave's longitudinal prediction record: every demand hypothesis is captured
  **timestamped and hash-sealed before the outcome is known**
- Forecasts must be measurable: metric, predicted range, horizon date, stated confidence, falsifier
- Evidence accumulates through stages (awareness → engagement → declared intent →
  willingness to pay → deposit → contract → operating revenue)
- Outcomes are scored against the sealed forecast; the ledger reports hit rate,
  mean forecast error and calibration gap as corporate KPIs
- REST API under `/api/signal-ledger` — this is the dataset that makes demand
  financeable over time (see the BrandClave Future Strategy doc)

---

## Installation (Technical)

### Requirements
- Python 3.11+
- **Mistral API key** (required for trend naming and embeddings)
- Optional: Redis (for caching)
- Optional: VERSES Genius API (for cloud-based active inference)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MISTRAL_API_KEY` | **Yes** | For LLM trend naming and embeddings |
| `BLUESKY_HANDLE` | For Bluesky | Full handle (e.g. `name.bsky.social`); public search is WAF-blocked, so an authenticated session is required |
| `BLUESKY_APP_PASSWORD` | For Bluesky | An app password from Settings → Privacy and security → App passwords, never the account password |
| `GENIUS_API_URL` | No | VERSES Genius agent URL |
| `GENIUS_API_KEY` | No | VERSES Genius API key |
| `REDIS_URL` | No | Redis for caching (defaults to in-memory) |
| `SCHEDULER_ENABLED` | No | Background scraping scheduler (default `true`) |
| `LOG_FORMAT` | No | `json` (default, production) or `text` (local dev) |
| `LOG_LEVEL` | No | Logging level (default `INFO`) |
| `SENTRY_DSN` | No | Enables Sentry error tracking when set |

Configuration is validated at boot by a typed settings object (`config/settings.py`):
malformed values fail fast with a clear message, and startup logs which integrations
are configured — never the secret values.

### Manual Setup

```bash
# Create environment
conda create -n brandclave python=3.11 -y
conda activate brandclave

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY

# Initialize database (local bootstrap; deployments run `python -m alembic upgrade head`)
python -c "from db.database import init_db; init_db()"

# Run the server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Cloud Deployment (Render)

The app can be deployed to Render's free tier:

1. Push to GitHub
2. Create new Web Service on Render, connect repo
3. Add environment variables in Render dashboard:
   - `MISTRAL_API_KEY` - your Mistral API key
   - `SCHEDULER_ENABLED` - set to `true` for auto-scraping
4. Deploy

**Note:** Free tier has cold starts (~30s) and ephemeral storage. For persistent data, use Render PostgreSQL or an external database.

### CLI Commands

```bash
# Activate environment
conda activate brandclave

# List available scrapers
python scripts/run_crawlers.py --list

# Run specific scraper
python scripts/run_crawlers.py --source reddit

# Run all scrapers
python scripts/run_crawlers.py --all

# Process content (embeddings + analysis)
python scripts/run_crawlers.py --process --limit 300

# Generate trends
python scripts/regenerate_trends.py

# Extract moves
python scripts/run_crawlers.py --moves --days 30

# Scan a property
python scripts/run_crawlers.py --scan "https://acehotel.com/new-york/"
```

---

## API Endpoints

```
Dashboard:       http://localhost:8000/api/monitoring/dashboard-v2
Debug:           http://localhost:8000/api/monitoring/debug
API Docs:        http://localhost:8000/docs

Auth:            POST /api/auth/register | /api/auth/login | GET /api/auth/me
Saved research:  GET/POST /api/projects/saved (Bearer token)
Social Pulse:    GET  /api/social-pulse
Trend Sources:   GET  /api/social-pulse/{id}/sources
Hotelier Bets:   GET  /api/hotelier-bets
City Desires:    POST /api/city-desires
                 GET  /api/city-desires/quick?city=Lisbon
Demand Scan:     POST /api/demand-scan
Brand Blueprint: POST /api/brand-blueprint/generate-simple
Signal Ledger:   POST /api/signal-ledger/predictions
                 GET  /api/signal-ledger/predictions
                 POST /api/signal-ledger/predictions/{id}/events
                 GET  /api/signal-ledger/metrics
System Health:   GET  /api/monitoring/health
Metrics:         GET  /api/monitoring/metrics
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Active Inference** | PyMDP (JAX-based), Expected Free Energy minimization |
| **Backend** | Python 3.11, FastAPI, Pydantic |
| **Database** | SQLite (data), ChromaDB (vectors) |
| **AI/ML** | Mistral AI (embeddings + LLM), HDBSCAN (clustering) |
| **Scheduling** | APScheduler (background jobs) |
| **Caching** | Redis (optional) |

---

## Project Structure

```
brandclave/
├── api/                    # FastAPI routes & dashboard
│   └── routes/             # Endpoint handlers
├── services/
│   └── active_inference/   # PyMDP POMDP controllers
│       ├── scraping_pomdp.py
│       ├── clustering_pomdp.py
│       ├── move_extraction_pomdp.py
│       ├── coordinator_pomdp.py
│       └── pymdp_learner.py
├── ingestion/              # Scrapers (news, social, reviews)
├── processing/             # NLP pipeline (embeddings, clustering)
├── services/               # Business logic (trends, moves)
├── monitoring/             # Metrics collection
├── scheduler/              # Automated job scheduling
├── cache/                  # Redis caching layer
├── db/                     # Database models & migrations
├── scripts/                # CLI tools & batch files
└── configs/                # YAML configurations
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Dashboard shows loading forever | Use the debug page: `/api/monitoring/debug` |
| "Environment not found" | Run `SETUP_FIRST_TIME.bat` again |
| No trends showing | Run `POPULATE_DATA.bat` to generate data |
| LLM rate limited | The system auto-retries with backoff |
| Poor trend names like "Tour Advice There Trend" | Add `MISTRAL_API_KEY` to `.env` - LLM generates proper names |
| Trends show old data | Trends auto-filter to last 7 days; regenerate with `python scripts/regenerate_trends.py` |

### To Stop the Server

Press `Ctrl+C` in the terminal window, or close it.

---

## Production Engineering

- **Typed settings** — `config/settings.py` validates all configuration at boot and fails fast
- **Structured logging** — JSON log lines by default (`LOG_FORMAT=text` for local dev); optional Sentry via `SENTRY_DSN`
- **Resilient HTTP** — `ingestion/http_client.py` gives every scraper and external call explicit timeouts, retries with exponential backoff and jitter, and `Retry-After` handling; a dead source logs and continues — a scrape run never aborts because one source failed
- **CI** — GitHub Actions runs lint (ruff error gate), the fast test suite, and an app boot check on every push and PR

See `REVIVAL-AND-COMMERCIAL-PLAN.md` §5 for the full production roadmap (next up: managed Postgres + pgvector, Alembic migrations, scheduler as a separate worker).

---

## Development

### Running Tests

```bash
# Fast unit tests (same as CI)
python -m pytest tests/ingestion -q

# Test API endpoints
python scripts/test_api_endpoints.py

# Test PyMDP integration
python test_pymdp.py
```

### Adding a New Source

1. Add an entry to `configs/sources.yaml` (status, kind, priority, config)
2. Create the scraper class (`BaseScraper` for content, `MetricScraper` for time series)
3. That's it — the CLI, the POMDP action space and the scheduler all derive
   from the registry (`ingestion/registry.py`), and the POMDP learns the new
   source's characteristics automatically

See `docs/ARCHITECTURE.md` for the full platform architecture.

---

## License

Private - BrandClave
