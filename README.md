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
| **Overview** | Quick stats: content count, trends found, moves extracted |
| **City Desires** | Type a city to discover what travelers want but can't find |
| **Social Pulse** | AI-detected travel trends from social conversations |
| **Hotelier Bets** | Strategic moves extracted from hospitality news |
| **Content** | Latest scraped articles and posts |
| **Scrapers** | Status of each data source |

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
- Quality filtering removes garbage like "Tour Advice There Trend"
- Click sources to see original posts

### Hotelier Bets (Strategic Moves)
- Extracts company moves from news: expansions, acquisitions, launches
- Shows company, move type, market, strategic implications
- Links to original articles

### City Desires
- Type any city (Lisbon, Tokyo, Barcelona, etc.)
- Scrapes Reddit/YouTube for that city in real-time
- Shows: unmet traveler needs, frustration points, white space opportunities
- Recommends hotel concepts based on gaps

---

## Installation (Technical)

### Requirements
- Python 3.11+
- Mistral API key (for embeddings and LLM)
- Optional: Redis (for caching)

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

# Initialize database
python -c "from db.database import init_db; init_db()"

# Run the server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

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

Social Pulse:    GET  /api/social-pulse
Trend Sources:   GET  /api/social-pulse/{id}/sources
Hotelier Bets:   GET  /api/hotelier-bets
City Desires:    POST /api/city-desires
                 GET  /api/city-desires/quick?city=Lisbon
Demand Scan:     POST /api/demand-scan
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
| Poor trend names | Trends are now quality-filtered automatically |

### To Stop the Server

Press `Ctrl+C` in the terminal window, or close it.

---

## Development

### Running Tests

```bash
# Test API endpoints
python scripts/test_api_endpoints.py

# Test PyMDP integration
python test_pymdp.py
```

### Adding a New Scraper

1. Create scraper in `ingestion/scrapers/`
2. Register in `ingestion/factory.py`
3. Add to `ScrapingPOMDP.SOURCES` list
4. The POMDP will automatically learn its characteristics

---

## License

Private - BrandClave
