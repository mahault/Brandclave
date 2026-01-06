# BrandClave AI — Development Roadmap

## Current Status: ~20% Complete
**Backend intelligence layer exists** — scraping, trends, moves, city desires, basic demand scan.
**Not yet built** — user-facing product shell, brand builder, project management, breakthrough features.

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

---

## Phase 2: What's Hot Enhancements
*Make Social Pulse & Hotelier Bets production-ready*

### Social Pulse Improvements
- [ ] Add **White Space Score** to each trend card
- [ ] Add **Filters** — by region, segment, time period
- [ ] Add **"Save to Project"** button on each trend
- [ ] Add **"Turn Into Brand Concept"** button → pre-fills Build a Brand

### Hotelier Bets Improvements
- [ ] Add **Market** field to move cards
- [ ] Add **Strategic Implications** section
- [ ] Add **Filters** — by company, move type, region
- [ ] Add **"Save to Project"** button

### Demand Scan Enhancements
- [ ] Generate **Demand Fit Score (0-100)**
- [ ] Show **Positioning Misalignment Flags**
- [ ] Add **Experience Gap Snapshot** (2-3 themes)
- [ ] Add **Opportunity Lanes** (strategic trajectories)
- [ ] Add **"Send to Build a Brand"** CTA

---

## Phase 3: Build a Brand (MVP)
*The core product — automated brand creation*

### Input Form
- [ ] City & location type selector
- [ ] Target ADR input
- [ ] Segment selector (lifestyle, luxury, boutique, etc.)
- [ ] Developer goal input
- [ ] Attach What's Hot signals option

### Brand Blueprint Output (MVP)
- [ ] **Brand name** — AI-generated unique name
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

## Phase 5: BrandClave Chat
*Intelligence assistant interface*

> **Technical Spec:** [docs/CHAT_MODULE_SPEC.md](docs/CHAT_MODULE_SPEC.md)

### Architecture: Bayesian RAG + POMDP-lite
- [ ] Mode Router — Bayesian intent inference (insight/brand_build/demand_scan)
- [ ] RAG Layer — vector + BM25 retrieval with Bayesian fusion scoring
- [ ] Belief Manager — POMDP-lite dialogue control (slots, confidence, stage)
- [ ] Structured Artifacts — JSON outputs with sources + confidence scores

### Chat Interface
- [ ] Conversational UI (single screen)
- [ ] Chat history persistence
- [ ] Suggested prompts carousel
- [ ] Confidence panel (High/Medium/Low + sources)

### Three Modes
- [ ] **Insight Mode** — trend forecasting, market opportunity analysis
- [ ] **Brand Build Mode** — interactive brand concept creation
- [ ] **Demand Scan Mode** — property analysis with prefill to Build a Brand

### Capabilities
- [ ] Reference What's Hot data (trends, moves) in answers
- [ ] Read and enhance existing projects
- [ ] Context-aware follow-up questions
- [ ] "Send to Build a Brand" action with prefilled inputs
- [ ] "Save to Project" for any artifact

### Data Model (Supabase)
- [ ] `projects` — user brand projects
- [ ] `messages` — conversation history
- [ ] `artifacts` — structured JSON outputs with provenance
- [ ] `knowledge_chunks` — vector store for RAG

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
