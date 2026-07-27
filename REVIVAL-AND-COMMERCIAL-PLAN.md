# BrandClave, Revival, Sources and Production Plan

Internal engineering doc. Written 2026-07-21, substantially updated 2026-07-22 after actually reviving the codebase and researching data sources.

Three separate jobs, kept separate on purpose:

1. **Revival** — get it running and filmable. Days. Milestone 1 in the co-founder agreement.
2. **Source rebuild** — the current corpus cannot support the product thesis. Weeks. Part of Milestone 2.
3. **Production engineering** — make it code you can sell work from without it falling over. Weeks, ongoing.

---

## 1. Verified state (as of 2026-07-22)

- Roughly 60% built per ROADMAP.md. Last substantive commit `ea40204`, 14 Jan 2026.
- Stack: FastAPI + SQLAlchemy + SQLite, ChromaDB vectors, Mistral embeddings/LLM, HDBSCAN clustering, PyMDP active inference, APScheduler, 12 sources.
- Render deploy is dead. `brandclave-demo.onrender.com` 404s on both health routes.

### What six months of drift actually broke

Only one thing, and it is fixed:

- **pymdp git tag deleted.** `requirements.txt` pinned `git+...@v1.0.0_alpha`; upstream removed that tag once 1.0.0 shipped. Now pinned to `inferactively-pymdp>=1.0.0,<2.0` from PyPI. Verified API-identical to what the code calls: `infer_policies(qs)`, `sample_action(q_pi, rng_key=...)`, `infer_states(observations, empirical_prior, ...)`. No code changes needed. `environment.yml` had a conflicting `>=0.0.7.1` pin; reconciled.

Everything else imports clean. For a codebase untouched since January that is a good result.

### The environment problem, diagnosed

Every stall observed on 2026-07-22 traced to one cause. A `faulthandler` stack dump during a hung boot showed Python blocked in `importlib._bootstrap_external.get_data`, i.e. **reading a source file off disk**, not network and not JAX.

Symptoms it explains:
- `scraping_pomdp` importing in 422s cold, then 3.1s warm
- Repeated, near-identical ~180s stalls on `city_desires` and `api.routes.monitoring`
- `sqlite3.OperationalError: disk I/O error` when querying `data/brandclave.db` in place
- Two server boots hanging at the same import

Cause: OneDrive stores files as cloud-only placeholders and hydrates them lazily, one at a time, as something touches them. A Python import touches hundreds of small files, so it is close to the worst case. `attrib +P -U` (keep on this device) marks files but OneDrive still downloads on its own schedule and was not fast enough.

---

## 2. Decision: stop working out of OneDrive

**The working copy is `C:\dev\brandclave`. OneDrive is not a development location.**

Rationale:
- Lazy hydration makes imports, test runs and scrapes unpredictably slow or hanging.
- SQLite over OneDrive throws I/O errors because of sync locking, and risks corrupting the DB through sync conflicts.
- Sync conflicts on `.py` files can silently produce `file (1).py` style duplicates.
- Git already gives version history and offsite backup via the GitHub remote, so OneDrive adds nothing here.

Applies to any conda work too: `Push-Location C:\` before creating environments, since installers extracting into a OneDrive path fail.

The OneDrive copy can stay as an archive, but it should not be the one that runs. Ideally delete it later to remove ambiguity about which copy is real.

---

## 3. Revival (Milestone 1)

Goal: runs locally, data is fresh, dashboard works, redeployed off the dead free tier.

**Done already:**
- pymdp pin fixed in both dependency files, git dependency dropped
- conda env `brandclave` created (python 3.11), all dependencies installed
- pymdp API verified against actual call sites
- root cause of the boot hang identified
- working copy relocated to local disk

**Remaining:**
- Boot from `C:\dev\brandclave`, confirm dashboard and API docs load
- Verify Mistral client calls still match the installed 1.x SDK
- Run a full scrape, see how many of the 12 sources still return anything
- Regenerate trends and moves, check quality
- Redeploy somewhere that can actually run it (see section 6)

---

## 4. Data sources

### 4.1 Why the current corpus fails

`ScrapingPOMDP.SOURCES` has 12 entries. **Ten are hospitality trade publications** (Skift, Hotel Dive, Hotel Management, SiteMinder, Top Hotel News, EHL Insights, eHotelier, Lodging Magazine, Luxury Hospitality, Hotel Business). They cover the same press releases and quote the same executives, so they are effectively one source. Only Reddit and YouTube carry consumer voice, and Reddit is dead (below).

Two consequences:

- The build plan promises "what people actually want, **not what the industry assumes**". Ten of twelve sources are the industry assuming.
- Trade press produces **no city-level signal**, so the signal-to-city matcher, the flagship feature, cannot be built on this corpus at all.

### 4.2 Verified status of current and candidate sources (July 2026)

| Source | Status | Notes |
| :-- | :-- | :-- |
| Reddit public JSON | **Dead, 403** | Tested live; blocked on both project and browser user agents |
| Reddit official API | **Not usable free** | 100 QPM free tier is non-commercial by terms. Commercial ≈ $0.24/1k on a reviewed contract. RSS workaround throttled to ~1 req/min/feed and is a terms problem, not just technical |
| TripAdvisor / Booking scrapers in repo | **Remove** | `ingestion/reviews/`, plus a live TripAdvisor call at `services/city_desires.py:271-292`. Contradicts build plan 2.3 and 4 |
| TripAdvisor Content API | **Sunsets 2026-08-31** | Replaced by Terra. Quote-only, usage rights vary by tier. Start the BD conversation now if brand review data matters |
| Google Places | Constrained | Reviews only in Enterprise+Atmosphere, $25/1k, 1,000 free/mo. **Caching prohibited**, so it cannot be a time-series backbone |
| Google Trends | **Unreliable** | pytrends archived April 2025; official API still alpha-gated. Do not build on it |
| Yelp Fusion | No perpetual free tier | $229-643/mo, review excerpts only (3 on Enhanced, 7 on Premium) |
| News RSS (existing 10) | Working | Keep, but demote. They are one voice, not ten |

### 4.3 Target source catalogue

Free and legally clean. This is the backbone.

**Geo-resolvable demand — this is what makes signal-to-city computable**

| Source | Access | Notes |
| :-- | :-- | :-- |
| Wikimedia Pageviews API | Free, no key | Daily interest per destination back to 2015. Best available substitute for Google Trends. Requires descriptive User-Agent with contact info |
| Eurostat `tour_occ_*` | Free, no key | Arrivals and occupancy by NUTS region, monthly and annual. Includes short-term-rental occupancy booked via Airbnb/Booking/Expedia from 2018 |
| Eurostat `tour_cap_*` | Free, no key | Accommodation capacity, the supply side |
| US NTTO I-94 / I-92 | Free | Monthly international arrivals to the US, filterable by country and period |
| National tourism boards | Free, varies | e.g. Spain Dataestur, Austria BMWET. No common API, coverage uneven |
| UN Tourism (UNWTO) | Mixed | Dashboard and Barometer headline data free; full database is subscription. Free for university researchers on written request |
| OSM Overpass API | Free, no key | Supply-side density (hotels, restaurants, attractions per area). ~10k requests/day fair use |
| Inside Airbnb | Free | Quarterly listings and reviews per city; review velocity is a known occupancy proxy. Check licence per city before commercial use |

**Consumer voice — currently one working source, needs to be many**

| Source | Access | Notes |
| :-- | :-- | :-- |
| Bluesky AT Protocol | Free, unauthenticated | 3,000 req/5min by IP. No commercial restriction anywhere in the docs. Cleanest legal position found. Firehose gives unmetered real-time ingest |
| Mastodon | Free per instance | 300 req/5min per account and per IP. Fragmented, budget per instance |
| YouTube Data v3 | Free, commercial OK | 10,000 units/day, but only **100 search calls/day**. Search is the scarce resource; video lookups and comments cost 1 unit each. Cache video IDs, treat discovery as the budget. Comments must not be stored beyond 30 days |
| Quora | Scraper already in repo | Verify it still works and that terms permit |
| Travel Substacks and blogs | Free, RSS | Long-form consumer voice, easy to add |
| Podcast transcripts | Free/varies | Travel and lifestyle podcasts, richer signal than headlines |

**Societal-shift signal — currently zero sources, and it is the actual thesis**

This is where "AI reduces human interaction so people crave connection", the SENTIENT premise, would come from. Right now that claim has no data behind it.

| Source | Access | Notes |
| :-- | :-- | :-- |
| Pew Research Center | Free | Consumer and social attitude research |
| World Economic Forum reports | Free | Future-of-travel and future-of-work themes |
| Edelman Trust Barometer | Free | Annual, trust and institutional sentiment |
| McKinsey / Deloitte / PwC / Accenture | Free reports | Consumer and travel outlooks, published openly |
| Gallup, Ipsos | Free summaries | Wellbeing and behaviour indices |
| Remote-work indices | Free/varies | Where and how people work now, drives extended-stay demand |

**Adjacent culture — where shifts surface before they reach hospitality**

Sarah's own examples (techno and electronic music, places to think and create, wellness) live here, not in Hotel Business.

| Source | Access | Notes |
| :-- | :-- | :-- |
| Dezeen, ArchDaily | Free RSS | Design and architecture direction |
| Wallpaper, Business of Fashion | Free/partial | Aesthetic and luxury signals |
| Resident Advisor | Free | Nightlife, electronic music, event geography |
| Eater, World's 50 Best, Michelin | Free | Food culture by city |
| Global Wellness Institute | Free | Wellness trend research |

Paid, only if justified:

- **TripAdvisor Terra** for brand-level review data. Quote-only, and the legacy API dies 2026-08-31
- **Reddit commercial contract** if Reddit is genuinely needed
- **Google Places** for spot-checks only, never as a backbone

### 4.4 On scraping vendors

Apify, Bright Data, ScraperAPI and similar all push ToS liability to the customer. Apify's terms require the customer to indemnify them and hold the customer responsible for extracting from unauthorised sources. The only genuine liability transfer found is SerpApi's US Legal Shield, which exists only at $150/mo and above, covers collection but not use, and excludes copyright, IP and privacy claims.

So paying a vendor to fetch OTA data does not cure the terms problem, it adds a bill to it.

Legal position in short: logged-out scraping of public pages is defensible against CFAA claims (hiQ, Meta v Bright Data), but hiQ still lost on breach of contract, and EU privacy regulators are hostile to scraping personal data. Review text with usernames attached is personal data. Get counsel before shipping anything that stores it.

### 4.5 Side benefit

The active-inference layer is currently wasted. Adaptive source selection across 12 near-identical trade feeds has almost nothing to learn. Across 40+ heterogeneous sources with genuinely different yields, freshness and error rates, the POMDP starts doing real work, and becomes a defensible technical story rather than decoration.

---

## 5. Production engineering

What "production level" means here, in priority order. These apply regardless of business model.

### 5.1 Data layer

- **Move off SQLite committed to git.** Concurrent writes corrupt it, Render's disk is ephemeral so data vanishes on redeploy, and committing a binary DB bloats the repo every scrape. Target managed Postgres + pgvector (Neon or Supabase).
- **Consolidate the two stores.** SQLite for relational and ChromaDB for vectors is two things to back up, migrate and keep consistent. pgvector collapses them into one.
- **Add migrations.** Alembic. There is no migration story today, which makes schema change risky.

### 5.2 Configuration and secrets

- `.env` currently holds live keys and is gitignored, which is right, but there is no validation. A missing or malformed key fails deep inside a request instead of at boot.
- Add a typed settings object (pydantic-settings) that validates on startup and fails fast with a clear message.
- Secrets belong in the host's secret store in production, not a file.

### 5.3 Reliability

- **Retries and backoff** on every external call. LLM APIs, scrapers, and the vector store all currently fail in ad hoc ways.
- **Rate-limit protection** on Mistral and any source API, with a token budget per run.
- **Timeouts everywhere.** Several scrapers use bare `httpx.get` with default behaviour.
- **Graceful degradation.** A dead source should log and continue, never abort a run.

### 5.4 Testing and CI

- Tests exist but there is no pipeline. Add GitHub Actions: lint, type check, test on push.
- The POMDP layer needs deterministic tests with a fixed `rng_seed`, since it is the most novel and least obvious part of the system.
- Contract tests per scraper that assert shape, so an upstream layout change fails loudly instead of silently returning zero rows.

### 5.5 Observability

- Structured logging (JSON) rather than free-text, so failures are searchable.
- Error tracking. `SENTRY_DSN` already exists in `.env` as a placeholder; wire it up.
- Health and metrics endpoints exist under `/api/monitoring`, but nothing alerts. Add external uptime checking.

### 5.6 Code structure

- **Scheduler must run as its own worker process**, not inside the web process. Today a restart or scale event disrupts scraping.
- **No network or heavy work at import time.** Nothing found so far, but the boot investigation showed how expensive import-time surprises are.
- Dashboard is HTML-in-Python. Fine to film, not fine to maintain. React migration is Phase 10 in ROADMAP.
- Type hints and `mypy` on the service layer, which is where the logic lives.

### 5.7 Hosting

- Free tier is 512MB, 30s cold starts, sleeps when idle. It cannot hold JAX, which is why `PREWARM_SERVICES=false` is set and the active-inference layer is effectively throttled off in production.
- Needs a real always-on instance sized for JAX, plus a separate worker for the scheduler.

---

## 6. Product and commercial features

Deferred deliberately. The revenue model is **design services**, not SaaS: BrandClave delivers concepts to clients, and the platform is an internal accelerator plus lead-gen demo. So the usual product scaffolding is not the early priority.

**Not needed early:**
- User accounts and multi-tenancy. The team are the users
- Stripe and billing. Clients pay for design work, invoiced
- Self-serve onboarding

**Needed instead:**
- Output quality good enough to put in front of a paying client
- Export to a real deliverable (PDF, deck)
- Reliability, because a failed run in front of a client is the actual failure mode

Revisit if the model changes to self-serve.

---

## 7. Rough effort and cost

| Track | Effort | Recurring cost |
| :-- | :-- | :-- |
| Revival (Milestone 1) | 1-2 days, mostly done | small paid host |
| Source rebuild | 2-4 weeks | mostly free sources; paid only if TripAdvisor Terra or Reddit contract are needed |
| Production engineering | several weeks, incremental | managed Postgres, real host, error tracking, LLM API spend |
| Design-services product polish | weeks | as above |

Milestone dates for the agreement should be set after revival completes, per the agreed wording in section 4 of the memorandum.

---

## 8. Open decisions

- Delete the OneDrive copy once `C:\dev\brandclave` is confirmed good, to avoid two working copies.
- Remove the TripAdvisor and Booking scrapers, and the TripAdvisor call inside City Desires. Needs a decision because it slightly degrades City Desires output.
- Which candidate city set to pre-scan for the signal-to-city matcher.
- Whether to open the TripAdvisor Terra conversation before the 2026-08-31 sunset.
- Whether Reddit is worth a commercial contract or should be dropped.
