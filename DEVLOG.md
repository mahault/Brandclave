# Devlog

## 2026-07-27 (night)
- Postgres-ready: Alembic wired to app settings (baseline `5cc92be0ef43`, all 11 tables, regression test vs model metadata), psycopg3 driver, provider-URL normalization (postgres:// → postgresql+psycopg://), pool_pre_ping + pooling on PG engines. Local DB stamped at baseline.
- Scheduler extracted to a standalone worker entrypoint (`scripts/run_worker.py`) per plan §5.6.
- `render.yaml` is now a full blueprint: managed Postgres + web (health check, migrations in preDeploy) + worker; only MISTRAL_API_KEY is manual. Actual provisioning blocked on a provider account (no CLIs/Docker on this machine).
- 51 tests green, lint + boot clean.

## 2026-07-27 (evening)
- Wrote `docs/ARCHITECTURE.md` — the platform orientation doc (ingestion registry, data layer, multi-tenancy, Signal Ledger, frontend, production engineering, deployment state, how-tos). README updated to match (auth endpoints, registry-driven "add a source" instructions).
- Dashboard sign-in shipped: account button in the status bar, SENTIENT-styled register/sign-in modal, JWT stored client-side. Saved trends/moves now sync to `/api/projects/saved` when signed in — localStorage stays the fast render cache, the API is the durable store; existing anonymous saves migrate up automatically on first login; clear-all also clears server copies. Blueprint generate/list calls carry the Bearer token on both pages so blueprints are user-owned.
- Generated a real JWT_SECRET into local `.env` (sessions now survive restarts).
- Verified end to end against the running server: register 201 → me 200 → save 201 → list → re-login 200 → dashboard serves the auth UI. Bluesky + scheduler already pushed the corpus from 2,896 to 3,294 items.

## 2026-07-27 (later)
- Made `configs/sources.yaml` load-bearing: new `ingestion/registry.py` is the single source of truth for all 40+ sources (active/planned/blocked + priority + per-source config). `run_crawlers.SCRAPERS`, the ScrapingPOMDP action space, scheduler fallbacks and API defaults all derive from it. Reddit and OTA scrapers marked blocked and excluded from the rotation.
- New source kind: metric. `demand_metrics` table (unique per source/city/metric/date) + `MetricScraper` base that reuses BaseScraper.run() job tracking with an upsert persistence path.
- **Bluesky live** (388 posts first run): AT Protocol search via `api.bsky.app` — `public.api.bsky.app`'s searchPosts 403s scripted clients behind a WAF. **Wikimedia pageviews live** (837 daily points, 27 cities, 30-day window): the geo-resolvable demand backbone.
- Fixed a real POMDP bug: the scheduler fed `result.get("items_count")` but scrapers return `items_scraped` — the active-inference layer had been observing constant zeros. Novelty now = items_saved/items_scraped (dedup ratio). Also: POMDP "wait" action is respected instead of being coerced into a skift scrape.
- Multi-tenant foundation (direction change: platform becomes self-serve): `users` + `saved_items` tables, register/login/me endpoints with JWT (pyjwt) and pbkdf2 password hashing, per-user blueprint ownership (anonymous flows unchanged), authenticated saved-research CRUD under `/api/projects`. 11 new auth tests; JWT_SECRET documented in .env.example.
- 20 ingestion tests + 11 auth tests green; boot and ruff clean.
- Added the Signal Ledger (strategy-doc CTO ask): sealed, hash-stamped prediction records with append-only evidence/outcome events and accuracy KPIs. New `data_models/signal_ledger.py`, `services/signal_ledger.py`, `api/routes/signal_ledger.py`, two new tables; REST under `/api/signal-ledger`. Full lifecycle smoke-tested against the real DB.
- Redesigned both dashboard pages to the SENTIENT design language: warm near-black surfaces, champagne-gold accents, Archivo display type, JetBrains Mono labels, signature four-color gradient bars. All 70+ ad-hoc hex colors replaced with a CSS design-token system; card-type accent palette validated for colorblind safety (OKLCH dark-band + CVD separation checks). Verified live with headless-browser screenshots.
- Production engineering (plan §5): typed settings validated at boot (`config/settings.py`, fail-fast with clear messages, integrations summary logged without secrets); JSON structured logging with optional Sentry (`monitoring/logging_config.py`); shared resilient HTTP client (`ingestion/http_client.py` — timeouts everywhere, retry/backoff with jitter, Retry-After honored, dead sources log-and-continue) applied across all scrapers and live service callsites, with 10 new tests; GitHub Actions CI (ruff error-gate + fast tests + boot check). Fixed one real F821 in `brand_blueprint/stages/base.py`.

## 2026-06-26
- brandclave.db updated and assorted temp working dirs created; no substantive source change (uncommitted, work in progress)

## 2026-06-27
- (uncommitted, work in progress) Updated the brandclave.db database; no substantive source change

## 2026-06-29
- Updated the brandclave.db data store. (uncommitted, work in progress)

## 2026-06-30
- Updated the brandclave.db data store. (uncommitted, work in progress)

## 2026-07-21
- Added revival and commercial-grade plan plus devlog (committed); local DB updated.

## 2026-07-22
- Revived the environment. Created conda env `brandclave` (python 3.11) and installed all dependencies.
- Fixed the one real breakage from six months of drift: the pinned pymdp git tag `v1.0.0_alpha` was deleted upstream once 1.0.0 shipped. Repinned to `inferactively-pymdp>=1.0.0,<2.0` from PyPI in both `requirements.txt` and `environment.yml`, dropping the git dependency. Verified the installed API matches every call site: `infer_policies(qs)`, `sample_action(q_pi, rng_key=...)`, `infer_states(observations, empirical_prior, ...)`. No source changes required.
- `environment.yml` had previously disagreed with `requirements.txt` (`>=0.0.7.1` vs the JAX 1.x line); reconciled.
- Diagnosed repeated boot hangs. A faulthandler stack dump showed Python blocked in `importlib._bootstrap_external.get_data`, i.e. reading source files off disk. Cause is OneDrive lazy hydration of cloud-only placeholders, which also explains the 422s cold import, the repeated ~180s stalls, and the `sqlite3.OperationalError: disk I/O error` on the database. Marking files "keep on this device" was not sufficient.
- Decision recorded: development moves off OneDrive to `C:\dev\brandclave`.
- Verified Reddit's public JSON endpoint now returns 403, so the primary consumer-voice source is dead. Reddit's official free API tier is non-commercial by terms, so it is not a drop-in replacement.
- Found OTA review scrapers in `ingestion/reviews/` and a live TripAdvisor call at `services/city_desires.py:271-292`, which contradict the stated data-sourcing position. Flagged for removal.
- Researched replacement sources and rewrote the plan doc: full source catalogue by category, production engineering standards, and the OneDrive decision.
