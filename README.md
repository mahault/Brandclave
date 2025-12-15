# BrandClave Aggregator

**AI-Powered Hospitality Intelligence Platform**

---

## For Sarah: Quick Start Guide

### First Time Setup (One Time Only - 10 minutes)

1. **Open the project folder** on your computer

2. **Double-click `SETUP_FIRST_TIME.bat`**
   - This installs everything automatically (including Conda if needed)
   - When Notepad opens, paste your Mistral API key and save
   - Wait for it to finish

That's it for setup!

### Running the Demo

**Double-click `START_DEMO.bat`**

The script will:
1. Scrape fresh data from 24 hospitality sources (15-20 min)
2. Analyze content with AI to detect trends and moves
3. Open the dashboard in your browser

### What You'll See

The **Dashboard** has 7 tabs:

| Tab | What It Shows |
|-----|---------------|
| **Overview** | Quick stats: content scraped, trends found, moves extracted |
| **City Desires** | Type a city to see what travelers want but can't find |
| **Social Pulse** | AI-detected travel trends from social conversations |
| **Hotelier Bets** | Strategic moves extracted from hospitality news |
| **Demand Scan** | Hotel property analysis with demand fit scoring |
| **Raw Content** | Latest scraped articles and posts from all sources |
| **System Status** | Health of database, cache, and scrapers |

---

## How the System Works

### The Big Picture

```
[17 Sources] --> [Scrape] --> [AI Analysis] --> [Dashboard]
     |               |              |               |
  News sites     Raw content    Clustering      Trends
  Reddit         stored in      & LLM calls     Moves
  YouTube        database       for insights    Opportunities
```

### Step 1: Data Collection (Scraping)

The system automatically scrapes content from 24 sources:

**News Sites (13):**
- Skift, Hotel Dive, PhocusWire, Hotel Management
- Travel Weekly, Hospitality Net, Hotel News Resource
- Travel Daily News, Business Travel News
- Boutique Hotelier, Hotel-Online, Hotel Tech Report, Top Hotel News

**Research & Insights (6):**
- SiteMinder (distribution insights)
- EHL Hospitality Insights (academic research)
- CBRE Hotels (market research)
- Cushman & Wakefield (property insights)
- CoStar (hospitality data)
- Travel Daily (industry news)

**Social Media (3):**
- Reddit (r/hotels, r/travel, r/solotravel, r/digitalnomad, etc.)
- YouTube (hotel reviews, travel guides, accommodation tips)
- Quora (hospitality Q&A)

**Reviews (2):**
- TripAdvisor hotel reviews
- Booking.com guest reviews

### Step 2: Processing (AI Analysis)

Once content is scraped, the AI pipeline:

1. **Generates embeddings** - Converts text to vectors using Mistral AI
2. **Clusters similar content** - Groups related discussions using HDBSCAN algorithm
3. **Detects trends** - Identifies patterns from clusters (Social Pulse)
4. **Extracts moves** - Finds strategic announcements from news (Hotelier Bets)

### Step 3: Presentation (Dashboard)

The dashboard displays everything in an easy-to-understand format:

#### Social Pulse (Trends)
- Shows trends detected from social conversations
- Each trend has: name, description, strength score, region
- **Click "X sources" to see the actual posts/articles that formed the trend**
- Sources open in a popup with links to original content

#### Hotelier Bets (Strategic Moves)
- Shows what hotel companies are doing
- Extracted from news: expansions, acquisitions, brand launches
- Includes company name, move type, and strategic analysis

#### City Desires (NEW!)
- Type any city name (e.g., "Lisbon", "Tokyo", "Barcelona")
- System scrapes Reddit, YouTube, and travel forums for that city
- Shows: what travelers want, frustration points, underserved segments
- Recommends hotel concepts based on gaps in the market

---

## Showing to Investors

### Demo Flow (5 minutes)

1. **Open the dashboard** - "This is our intelligence platform"

2. **Overview tab** - "We've scraped X articles and detected Y trends"

3. **Social Pulse tab** - "Here's what travelers are talking about"
   - Click a trend to show the sources
   - "Each trend is backed by real social media posts"

4. **Hotelier Bets tab** - "We track what competitors are doing"
   - "This is extracted automatically from news"

5. **City Desires tab** - "Type a city, see unmet demand"
   - Type "Lisbon" or "Barcelona"
   - "This shows white space opportunities for hotel development"

### Key Talking Points

- "We aggregate 24 hospitality data sources automatically"
- "AI clusters conversations into actionable trend signals"
- "Every insight links back to original sources for verification"
- "City Desires identifies unmet traveler needs by location"
- "This runs continuously - always fresh data"

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Script closes immediately | Right-click the .bat file, Run as Administrator |
| "Environment not found" | Run SETUP_FIRST_TIME.bat again |
| Browser doesn't open | Go to http://localhost:8000/api/monitoring/dashboard |
| No data showing | Wait for data collection to finish (10-15 min) |
| "travel not recognized" error | This is fixed - just run again |
| Trends say "Unnamed" | Re-run --trends to regenerate with better names |

### To Stop the Server

Press `Ctrl+C` in the black terminal window, or just close it.

---

## Technical Details

### API Endpoints

```
Dashboard:       http://localhost:8000/api/monitoring/dashboard
API Docs:        http://localhost:8000/docs

Social Pulse:    GET  /api/social-pulse
Trend Sources:   GET  /api/social-pulse/{id}/sources
Hotelier Bets:   GET  /api/hotelier-bets
City Desires:    POST /api/city-desires
                 GET  /api/city-desires/quick?city=Lisbon
Demand Scan:     POST /api/demand-scan
System Health:   GET  /api/monitoring/health
```

### CLI Commands

```bash
# Activate environment first
conda activate brandclave

# List all scrapers
python scripts/run_crawlers.py --list

# Run specific scraper
python scripts/run_crawlers.py --source skift

# Run all scrapers
python scripts/run_crawlers.py --all

# Process content with AI
python scripts/run_crawlers.py --process --limit 100

# Generate trends
python scripts/run_crawlers.py --trends

# Extract hotelier moves
python scripts/run_crawlers.py --moves

# Scan a property
python scripts/run_crawlers.py --scan "https://acehotel.com/new-york/"
```

### Tech Stack

- **Python 3.11+** with FastAPI
- **SQLite** for data storage
- **ChromaDB** for vector embeddings
- **HDBSCAN** for trend clustering
- **Mistral AI** for LLM analysis
- **APScheduler** for automated scraping
- **Redis** (optional) for caching

### Project Structure

```
brandclave/
├── api/                 # FastAPI routes & dashboard
├── ingestion/           # Scrapers (news, social, reviews)
├── processing/          # NLP pipeline (embeddings, clustering)
├── services/            # Business logic (trends, moves, city desires)
├── monitoring/          # Metrics collection
├── scheduler/           # Automated job scheduling
├── cache/               # Redis caching layer
├── db/                  # Database models & vector store
├── data_models/         # Pydantic models
├── configs/             # YAML configurations
└── scripts/             # CLI tools
```

---

## What's New (Latest Updates)

### City Desires Feature
- Type a city to discover unmet traveler needs
- Scrapes Reddit, YouTube, and travel forums
- Shows frustration scores, underserved segments
- Recommends hotel concepts for market gaps

### Clickable Trend Sources
- Every trend now shows "X sources (click to view)"
- Opens a popup with all source articles/posts
- Each source links to the original URL

### Improved Trend Quality
- Filters out "list" articles (e.g., "Top 10 trends for 2025")
- Better trend naming from actual content
- No more "Unnamed Trend" labels

### 24 Data Sources
- 13 hospitality news sites
- 6 research & insights sources (CBRE, CoStar, EHL, etc.)
- 3 social platforms (Reddit, YouTube, Quora)
- 2 review sites (TripAdvisor, Booking.com)

---

## License

Private - BrandClave
