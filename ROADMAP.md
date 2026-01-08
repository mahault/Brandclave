# BrandClave AI — Development Roadmap

## Current Status: ~42% Complete
**Backend intelligence layer exists** — scraping, trends, moves, city desires, demand scan, chat with RAG.
**Recently completed** — Phase 2 complete + POMDP adaptive scraping improvements.
**Not yet built** — Full brand blueprint output, project management, breakthrough features, React frontend.

---

## Phase 1: Foundation (Current)
*Core data engine and basic dashboard*

### ✅ Completed
- [x] Social Pulse — trend detection with clustering + LLM descriptions
- [x] Hotelier Bets — strategic move extraction from news
- [x] City Desires — location-based desire analysis
- [x] Demand Scan — basic property URL scanning
- [x] Dashboard v2 — tabs, metrics, click-to-expand modals
- [x] 12 data sources (Reddit, Skift, Hotel Dive, etc.)
- [x] PyMDP active inference for adaptive scraping
- [x] Quality filtering for trend descriptions

### ✅ POMDP Scraping Improvements (Jan 2026)
- [x] Staleness-based exploration — freshness beliefs decay over time
- [x] Increased exploration bonus — better coverage of all sources
- [x] Debug endpoints — `/api/scheduler/pomdp/beliefs` to inspect POMDP state
- [x] Manual scrape endpoint — `/api/scheduler/scrape-all` for bulk refresh
- [x] EFE logging — shows top candidates and stale sources needing attention
- [x] Keep-alive for Render — `/health` endpoint with scheduler status

---

## Phase 2: What's Hot Enhancements
*Make Social Pulse & Hotelier Bets production-ready*

### Social Pulse Improvements ✅
- [x] Add **White Space Score** to each trend card
- [x] Add **Filters** — by region, segment, time period
- [x] Add **"Save to Project"** button on each trend (localStorage)
- [x] Add **"Turn Into Brand Concept"** button → pre-fills Build a Brand

### Hotelier Bets Improvements ✅
- [x] Add **Market** field to move cards
- [x] Add **Strategic Implications** section
- [x] Add **Filters** — by company, move type, region
- [x] Add **"Save to Project"** button

### Demand Scan Enhancements ✅

**Dashboard Tab UI** ✅
- [x] Add Demand Scan tab to dashboard navigation
- [x] Property URL input form with scan button
- [x] Previously scanned properties list with cards
- [ ] Property detail modal with full analysis (future)

**Demand Fit Score (0-100)** ✅
- [x] Convert existing 0-1 float to 0-100 integer display
- [x] Color-coded badge: High (70+, green), Medium (40-69, yellow), Low (<40, red)
- [ ] Visual gauge/meter component (future)

**Positioning Misalignment Flags** ✅
- [x] Detect when property claims don't match offerings (e.g., "luxury" with budget amenities)
- [x] Detect price-tier mismatch (luxury price, midscale experience)
- [x] Detect theme confusion (conflicting positioning signals)
- [x] Display as warning badges on property cards

**Experience Gap Snapshot** ✅
- [x] Show top 2-3 trending experiences property is missing
- [x] Link gaps to specific trend data with strength %
- [x] Priority ordering by trend strength

**Opportunity Lanes** ✅
- [x] Format as strategic trajectory cards
- [x] Show demand driver + positioning recommendation
- [ ] Include competitive landscape insight (future)

**Actions** ✅
- [x] Add **"Send to Build a Brand"** CTA button
- [x] Pre-fill Build a Brand form with property context
- [x] Add **"Save to Project"** button

**Tests** ✅
- [x] `test_demand_fit_score_0_to_100`: Verify score conversion and badge colors
- [x] `test_positioning_misalignment_detection`: Test luxury/budget mismatch detection
- [x] `test_experience_gap_snapshot`: Verify top 2-3 gaps are returned
- [x] `test_opportunity_lanes_format`: Check strategic trajectory format
- [x] `test_send_to_build_brand`: Verify form pre-fill works
- [x] `test_demand_scan_api`: End-to-end API test

---

## Phase 3: Build a Brand (MVP) 🚧 In Progress
*The core product — automated brand creation*

### Input Form ✅
- [x] City & location type selector
- [x] Target ADR input
- [x] Segment selector (lifestyle, luxury, boutique, etc.)
- [x] Developer goal input
- [x] Attach What's Hot signals option (via "Turn Into Brand" from trends)

### Architecture: Multi-Step Pipeline
*5-stage LLM pipeline with RAG context enrichment (~12,700 tokens/blueprint, ~$0.002)*

```
Input Form → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Blueprint
              ↓         ↓         ↓         ↓         ↓
            RAG       RAG       RAG      RAG(lite)  (none)
```

**Stage 1: Foundation** — brand names (primary + 2 alts), one-liner, thesis
**Stage 2: Strategic** — pillars, positioning, unmet desires solved
**Stage 3: Experience** — guest personas, signature experiences, guest journey
**Stage 4: Atmosphere** — design direction, F&B concepts, revenue logic
**Stage 5: Summary** — investor summary (synthesizes all stages)

### Implementation Steps
- [x] `db/models.py` — Add BrandBlueprintModel
- [x] `services/brand_blueprint/schemas.py` — Pydantic models
- [x] `services/brand_blueprint/prompts.py` — Stage prompt templates
- [x] `services/brand_blueprint/stages/` — 5 stage implementations
- [x] `services/brand_blueprint/pipeline.py` — Orchestrator
- [x] `services/brand_blueprint/repository.py` — Database CRUD
- [x] `api/routes/brand_blueprint.py` — API endpoints
- [x] Dashboard frontend update — progress UI, structured display

### Brand Blueprint Output (MVP)
- [ ] **Brand name** — primary + 2 alternates
- [ ] **One-liner** — single sentence essence
- [ ] **Thesis** — core brand philosophy
- [ ] **Pillars** — 3-5 brand pillars
- [ ] **Positioning statement**
- [ ] **Signature experiences** — 3-5 unique experiences
- [ ] **Guest journey** — arrival to departure flow
- [ ] **Design/atmosphere direction** — visual language description
- [ ] **Revenue logic (lite)** — how the brand drives ADR
- [ ] **Guest persona sets** — 2-3 target personas
- [ ] **Unmet guest desires solved** — linked to demand data
- [ ] **F&B micro-concepts** — restaurant/bar ideas
- [ ] **Investor summary** — one-page pitch

### Storage
- [ ] SQLite database persistence (BrandBlueprintModel)
- [ ] localStorage sync for offline access

---

## Phase 4: My Brands & Projects
*Storage, organization, collaboration*

### User Accounts
- [ ] Authentication (email/password, OAuth)
- [ ] User profiles
- [ ] Account settings

### Project Management
- [ ] Create/save brand projects
- [ ] Project listing & organization
- [ ] Version history for brands
- [ ] Duplicate/fork projects

### Export Tools
- [ ] Export to PDF
- [ ] Export to PowerPoint
- [ ] Share link generation

---

## Phase 5: BrandClave Chat ✅ MVP Complete
*Intelligence assistant interface*

> **Technical Spec:** [docs/CHAT_MODULE_SPEC.md](docs/CHAT_MODULE_SPEC.md)

### Architecture: Bayesian RAG + POMDP-lite
- [x] Mode Router — Bayesian intent inference (insight/brand_build/demand_scan)
- [x] RAG Layer — vector + keyword retrieval with Bayesian fusion scoring
- [x] Belief Manager — POMDP-lite dialogue control (slots, confidence, stage)
- [x] Mistral LLM integration for response generation
- [x] Structured Artifacts — JSON outputs with sources + confidence scores

### Chat Interface
- [x] Conversational UI (single screen in dashboard)
- [x] Suggested prompts carousel
- [x] Confidence panel (High/Medium/Low + sources)
- [ ] Chat history persistence (in-memory only, needs DB)

### Three Modes
- [x] **Insight Mode** — trend forecasting, market opportunity analysis
- [x] **Brand Build Mode** — interactive brand concept creation
- [x] **Demand Scan Mode** — property analysis with prefill to Build a Brand

### Capabilities
- [x] Reference What's Hot data (trends, moves) in answers via RAG
- [x] Context-aware follow-up questions
- [x] 90+ city location detection
- [x] Segment/ADR/URL slot extraction
- [ ] "Send to Build a Brand" action with prefilled inputs
- [ ] "Save to Project" for any artifact
- [ ] Read and enhance existing projects

### Data Model (In Progress)
- [ ] `projects` — user brand projects (not yet)
- [ ] `messages` — conversation history (in-memory only)
- [x] `artifacts` — structured JSON outputs with provenance
- [x] `knowledge_chunks` — ChromaDB vector store for RAG

---

## Phase 6: Breakthrough Features (Part 1)

### A. Feasibility Forecaster
- [ ] Auto-generate ADR projections
- [ ] Occupancy forecasts
- [ ] RevPAR estimates
- [ ] 5-year performance curve
- [ ] Best/average/stressed scenarios
- [ ] Risk score
- [ ] Demand-driver mapping

### B. Guest Persona Generator
- [ ] Emotional drivers
- [ ] Spend behavior profiles
- [ ] Preferred amenities
- [ ] Aesthetic taste indicators
- [ ] Content behaviors
- [ ] Travel motivations

### C. Design Direction Generator
- [ ] Lobby moodboards
- [ ] Room direction concepts
- [ ] Exterior vibe
- [ ] F&B themes
- [ ] Color & material palettes
- [ ] Scent & sensory direction

---

## Phase 7: Breakthrough Features (Part 2)

### D. Geo-Opportunity Radar
- [ ] What locals + travelers are craving
- [ ] What doesn't exist yet
- [ ] Which brands are oversaturated
- [ ] Where demand spikes are forming
- [ ] What concept would dominate

### E. Pre-Launch Demand Builder
- [ ] Teaser campaign templates
- [ ] Viral content suggestions
- [ ] Influencer targeting lists
- [ ] Waitlist flow templates
- [ ] Social scripts
- [ ] Press angles

### F. Development Alignment Mode
- [ ] Shareable project links
- [ ] Role-based views (investor, architect, operator, lender)
- [ ] Comment/feedback system
- [ ] Approval workflows

---

## Phase 8: Breakthrough Features (Part 3)

### G. Risk Radar
- [ ] Overplayed trend warnings
- [ ] Zoning consideration flags
- [ ] Sustainability requirements
- [ ] Inflation and construction risk indicators
- [ ] Guest behavior volatility analysis

### H. Experience Layer Generator
- [ ] Tech layer recommendations
- [ ] Signature rituals
- [ ] Wellness micro-experiences
- [ ] In-room differentiators
- [ ] Community modules

---

## Phase 9: Implementation Blueprint (Brand-in-a-Box)
*Turn brand blueprints into execution-ready roadmaps*

### Inputs
- [ ] Property parameters (keys, public area, room types)
- [ ] Region selection (affects pricing)
- [ ] Quality tier (Good/Better/Best)
- [ ] Timeline (fast-track vs standard)

### Outputs
- [ ] **Implementation Cost Range** — FF&E, OS&E, signage, tech, artwork
- [ ] **Cost Bands per Space** — lobby, guestrooms, corridors, F&B, wellness
- [ ] **Good/Better/Best Scenarios** — impact on experience integrity
- [ ] **Brand Build Kit Spec List:**
  - Materials (stone, woods, metals)
  - Lighting frameworks
  - Furniture types + specs
  - Scent + audio + sensory elements
  - Technology layers
- [ ] **Architect & Designer Hand-off Pack:**
  - Narrative intent
  - Non-negotiable brand elements
  - Flexible zones for interpretation
  - Experience-driven spatial guidance

### Innovations
- [ ] Dynamic Cost Indexing (regional pricing)
- [ ] Value-Engineered Alternates
- [ ] Future: Shoppable Spec Packs (vendor shortlists)

---

## Phase 10: Platform & Scale

### React Frontend
- [ ] Migrate from HTML-in-Python to React
- [ ] Component library
- [ ] Responsive design
- [ ] Left sidebar navigation

### Performance & Scale
- [ ] < 5 second response times
- [ ] PostgreSQL for production
- [ ] Redis caching
- [ ] CDN for assets

### Business Features
- [ ] Subscription tiers
- [ ] Usage analytics
- [ ] Admin dashboard

---

## Technical Debt & Improvements
- [ ] Move dashboard HTML to separate frontend repo
- [ ] API versioning
- [ ] Comprehensive test coverage
- [ ] CI/CD pipeline
- [ ] Documentation site
- [ ] **Migrate to shared cloud database** — currently using SQLite committed to git; switch to Neon/Supabase PostgreSQL + pgvector for production (avoids repo bloat, enables proper sync between local/Render)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Brand blueprint generation | < 3 minutes |
| Trend accuracy | Expert-level insights |
| User retention | Daily intelligence tool usage |
| Brand quality | Outperforms agency first drafts |

---

*Last updated: January 2026*
