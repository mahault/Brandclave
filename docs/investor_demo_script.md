# BrandClave demo script

Driftwood Capital (Pranav, Carlos Jr). Tuesday 8 September 2026, 12:00 EDT, 15 minutes of demo inside a 90 minute call.
Sarah presents. Mau shares the screen and takes technical questions.

Live site: https://brandclave.onrender.com (opens on the dashboard). Build a Brand: https://brandclave.onrender.com/api/monitoring/build-a-brand

Before the call: open both pages at 11:45 and click through every tab once. The free server sleeps after 15 idle minutes, so keep a tab active. Open the Cities tab and type Lisbon at 11:55 so the analysis is already on screen when you get there.

## Timing

| Minute | Where | What happens |
|---|---|---|
| 0:00 | Signal Room | Opener, then Opportunity Map and City Matrix |
| 2:30 | Signal Room | Demand Curves, Attention Model |
| 4:00 | Trends | Two clusters, save one |
| 5:30 | Market Moves | Who is moving, SEC filings |
| 7:00 | Cities | Lisbon fact sheet, then start a second city and let it run |
| 8:30 | Demand Scan | The Hoxton card |
| 10:00 | Signal Ledger | Sealed forecasts, no hit rate yet |
| 11:00 | Build a Brand | Quinta do Alto and the renders; start a live generation |
| 13:30 | Cities, then Build a Brand | Show the results of the two things you started |
| 14:30 | Close | |

The two live waits (city analysis 60 to 90 s, concept generation 60 to 90 s) are started early and collected at the end, so nobody watches a spinner.

## Opener (30 seconds)

Say: "Every hotel brand starts with someone's hunch about what guests want. BrandClave replaces the hunch with evidence. It reads 26 sources every day, from official tourism statistics to what travellers post about a city this week, finds where demand is moving and where supply is thin, and turns that into a brand concept a developer can build. I'll show you the four steps: see the signal, pick a market, test a property, make a brand."

On screen: the dashboard, Signal Room tab. Point at the four steps written across the top bar.

## 1. Signal Room

The overview. Four figures at the top, then eight panels.

Say: "This is what the platform is watching right now. About 1,900 items collected in the last week, 26 sources live, 99 demand clusters tracked, 8 forecasts on the record. Everything else on this page is built from those."

### Opportunity Map

Say: "Every bubble is a demand cluster we found across consumer posts, trade press and news. Right means stronger demand. Up means less supply answering it. The upper right is where a concept should be built. Click any bubble and it opens, or you can build a brand from it directly."

Point at: the upper right of the plot. Hover one bubble to show the tooltip. The Table toggle shows the same numbers as rows.

If asked how the clusters are made: content is embedded with Mistral's embedding model (1,024 dimensions, stored in ChromaDB, about 4,300 vectors). HDBSCAN clusters the vectors. An active inference agent (PyMDP on JAX) chooses the clustering parameters by watching cluster quality over time, instead of a fixed setting. Strength is cluster cohesion and volume. White space is how little supply answers the cluster, from the language in the cluster set against supply data.

Limitation: white space is estimated from text and supply counts, not from bookings. It ranks, it does not price.

### City Matrix

Say: "Same logic for cities. Right is rising attention this week, from Wikipedia pageviews. Up is how many hotels already exist on the map. Bubble size is Airbnb listings. Lower right is thin supply meeting rising interest, which is where you want to be early."

Point at: Lisbon, then a large upper bubble to contrast (a saturated market).

If asked: attention is daily Wikipedia pageviews for each destination article, indexed to its own 30 day average. Hotel counts come from OpenStreetMap through the Overpass API (tourism=hotel within the city boundary). Airbnb listings come from Inside Airbnb's latest published snapshot per city.

Limitation: Wikipedia attention is a proxy for interest, and it spikes on news. We read the trend, not one day.

### Demand Curves

Say: "Here you can switch what demand means. Attention is daily. Official nights spent come from Eurostat, monthly by country. Airbnb review velocity and median price are quarterly. Hotel supply is the OpenStreetMap count. The three strongest risers this week are highlighted; everyone else is the grey field behind them."

Point at: the metric selector. Switch to Nights spent to show the official series, then back to Attention.

If asked: each city is indexed to its own 30 day average (100), so the chart compares movement, not size. Sources: Wikimedia pageviews API, Eurostat tourism tables, Inside Airbnb snapshots, OpenStreetMap. Where a source publishes one observation per period, the panel shows ranked levels instead of a curve, and says so.

Limitation: Eurostat is by country and monthly. City level official data varies by country and lags.

### Where Capital Is Moving, Operator Bets, Trend Movers

Say: "The right hand column is what operators are doing, not what travellers are saying. Deals, launches, renovations, reflags, week by week, with SEC filings counted alongside the press."

Point at: the weekly bars, then one card in Operator Bets.

If asked: moves are extracted from trade press, the GDELT global news index and SEC EDGAR filings (8-K and 10-K for 14 listed hotel companies, read directly from the SEC) by a language model with a fixed schema (company, move type, market, confidence). Only moves above a confidence threshold are kept.

### Attention Model

Say: "This panel is how the platform decides what to read next. Each source carries a belief about how productive it is. The scheduler picks the source that best balances reading known good sources and exploring the ones it is unsure about. That's what lets us add sources without the system drowning in noise."

Point at: the belief bars, the source marked next to read, and the free energy figure.

If asked what active inference buys: the scheduler minimises expected free energy, which combines expected yield (pragmatic value) and expected information gain (epistemic value). Concretely, a source that has not been read recently or whose yield is uncertain gets attention even if its last run was poor, so the corpus does not collapse onto the three loudest feeds. Same machinery chooses clustering parameters and drives the city analysis. It runs on PyMDP with JAX.

Limitation: the beliefs shown are from the last scheduler run. On the demo server the scheduler runs on the worker, not the web process, so the panel shows a snapshot with its timestamp.

### Coverage

Say (only if time): "Every registered source and when it last delivered. Fresh, ageing, silent, blocked."

If asked about blocked: sources whose terms or robots rules forbid scraping are registered as blocked and never read. Reddit is retired for that reason.

## 2. Trends

Say: "These are the demand clusters as a list, with filters by region, segment and time. Open one and you get the description, why it matters, the white space analysis and the actual quotes it was built from. Save the ones that matter to a project. Everything you save can become a brand later."

On screen: open Hyperlocal Urban Retreats (strength about 0.80). Show Source Quotes. Save it. Then point at Hotel Equity Renaissance as an example of high white space.

If asked: each cluster gets a name, description and white space analysis written by a language model from the cluster's own content, with the source quotes kept so a reader can check the claim. Names are regenerated when the cluster changes.

Limitation: cluster names are generated and can be awkward. The quotes are the evidence; the name is a label.

## 3. Market Moves

Say: "Who Is Moving ranks operators, REITs and platforms by what they did in the window and what kind of thing it was. Below that, every move on file. Filings come straight from the SEC, so when Sunstone sells the Hyatt Regency San Francisco or Park Hotels disposes of non core assets, it shows up here the day it is filed."

On screen: the league table, then filter Market Moves to acquisitions.

If asked: 14 listed companies are tracked by CIK on EDGAR. Each new 8-K and 10-K is fetched, the primary document is read, and moves are extracted. Press coverage fills in the private operators.

Limitation: private company moves depend on press coverage. Filings give us the listed ones with certainty.

## 4. Cities

Type Lisbon. The fact sheet is immediate. The analysis takes 60 to 90 seconds.

Say: "Type a city. The fact sheet is what the metric sources measure there: attention this week, the short term rental market, the built supply and the official country demand. Lisbon has 743 hotels on the map and 24,912 Airbnb listings. Below it, the analysis reads what travellers say they want in Lisbon and cannot find, and proposes concept lanes."

On screen: the fact sheet tiles with their sparklines. Then the desire themes (name, unmet need, intensity, frustration) and Concept Lanes at the bottom.

While it runs: "It's searching Bluesky, Mastodon and YouTube live for Lisbon right now, plus everything we already hold about the city, and grouping what it finds. That's why it takes a minute."

If asked: the analyzer is a structure learner. It embeds each relevant post and either joins it to an existing theme or opens a new one when the fit is below 0.63 (a mix of cosine similarity and keyword overlap). It decides what to search next from which themes are uncertain. Posts must be about staying in or visiting the place, and deal bots and emoji walls are filtered out. Sources: authenticated Bluesky search, Mastodon hashtag timelines, YouTube results, and the platform corpus.

Limitation: consumer voice volume varies by city. Lisbon and Barcelona are rich. A small city can return two themes.

Move on after the fact sheet. Type a second city (Barcelona) and leave it running. Come back at 13:30.

## 5. Demand Scan

Say: "Give it any hotel website and it scores the property against current demand. This is The Hoxton. It reads the site, builds a profile of what the property actually offers, and compares that to every demand cluster we track. You get the clusters it already speaks to, the gaps, and a brief written from that evidence."

On screen: open The Hoxton card. Point at the fit score, the alignment bars (Boutique Parisian Hideaways, Neighborhood Niche Stays, Lisbon Neighborhood Nomadism), the gaps, and the brief.

If asked: the property profile is embedded with the same model as the corpus and scored by cosine similarity against each trend's embedding. Alignment above 0.77 counts as speaking to the trend, below 0.745 as a gap. The brief is written by a language model from the aligned and unaligned clusters only.

Limitation: a scan reads the public website. Properties with thin websites get thin profiles.

## 6. Signal Ledger

Say: "This is the part nobody else does. Every forecast the platform makes is sealed the moment it is written, with a hash of its content, and it can never be edited. Evidence and outcomes are appended afterwards. There are 8 forecasts on the record. The hit rate is blank on purpose, because the first ones come due in December. We would rather show you an empty scorecard now than a good one we made up."

On screen: the Horizons strip. Point at the seal dates and today's marker. Open one prediction and expand its events to show the hash verification.

If asked: each record stores the hypothesis, the product implication, the forecasts and the uncertainty notes, and a SHA-256 hash of that content. The page recomputes the hash from the stored content and shows whether it matches. Outcomes are appended as events with their own timestamps. Nothing resolves by hand.

Limitation: eight forecasts is a start, not a track record. The value is that the track record will be auditable.

## 7. Build a Brand

Open the Build a Brand page. Open the saved blueprint Quinta do Alto.

Say: "Everything so far is seeing, picking and testing. This is making. Quinta do Alto is a concept for a 70 room lifestyle hotel in Alfama at a 260 dollar rate, for people who work from Lisbon for a few weeks at a time. Name and alternates, thesis, pillars, positioning, the unmet desires it answers, guest personas, signature experiences, the guest journey, design direction, food and drink, and an investor summary. Then four renders of the concept: arrival, lobby, room, and the bar."

On screen: scroll steadily. Stop at Unmet Desires Solved (each with a demand strength), then Design Direction, then the four renders.

If asked: the writer is a five stage pipeline (foundation, strategic, experience, atmosphere, summary), each stage a structured call to Mistral with a house style contract: short plain sentences, concrete nouns, no invented statistics. The renders come from OpenAI's image model, gpt-image-1.5, and each prompt is composed only from the blueprint's own fields (design direction, the first food and drink concept, the first signature experience, the property inputs). Nothing is added that the concept did not specify. A concept costs a fraction of a cent in tokens; a set of four renders about 30 cents.

Limitation: today the writer works from the inputs and the saved picks. Grounding it in the retrieved corpus, so the thesis cites the clusters directly, is the next build.

Live generation: at 11:00 on the clock, fill the form and click Generate. Rooms 7 and a goal like "a small hotel where guests can feed the neighbourhood cats" is fine and gets a laugh. Say: "It takes about a minute. We'll come back to it." Then go to Cities for the Barcelona result, then return here.

Do not click Visualise the concept on the new one unless you have 80 seconds to fill. It renders four images.

## Collect the two live results (13:30)

Cities: the Barcelona analysis is done. Read the first theme's unmet need aloud and one concept lane.

Build a Brand: the cat concept is done. Read the name and the one liner. Point out that it is saved in the list.

## Close (30 seconds)

Say: "So that's the loop. Signals from 26 sources, a place to pick, a property to test, and a brand to make, with every forecast on the record so we can be judged on it. What we have built in the last week is the data layer and the honesty layer. What we want to build next with you is the ground truth: bookings and performance data from partner hotels, so the forecasts resolve against revenue, not proxies."

## Questions they will ask

**How valid is the data?** Three tiers. Official: Eurostat, US NTTO, SEC filings. Structural: OpenStreetMap, Inside Airbnb, Wikimedia pageviews. Voice: Bluesky, Mastodon, YouTube, trade press (Skift, Hotel Dive, Hotel Management, TopHotelNews, Lodging Magazine, Hospitality Net, Hotel Business, eHotelier, EHL Insights, SiteMinder), culture and news (Dezeen, ArchDaily, Eater, Global Wellness Institute, Pew, GDELT). Every item keeps its source and URL, and the trend cards show the quotes. What we do not have yet is booking data. That is the partnership ask.

**Reddit?** Retired. Reddit's API terms do not allow this use, so its content was archived and removed from every count. Consumer voice now comes from Bluesky (authenticated search), Mastodon and YouTube. It also turns out Reddit's share of citations in AI answers fell sharply in August, so the industry is moving off it anyway.

**What does active inference buy you over a cron job?** Three things. The scheduler spends its reading budget where information is, so adding a source does not add noise. Clustering parameters adapt as the corpus changes instead of being tuned by hand. And the city analysis decides what to search next from its own uncertainty, which is why it finds themes a keyword search would miss. It is the same framework across all three, PyMDP on JAX.

**Defensibility?** The sealed ledger. Anyone can call an LLM and write a brand deck. Nobody else stakes the forecast first and lets you audit it. Over time the ledger is the asset: a dated, unalterable record of what the platform said before the market moved.

**Cost per blueprint?** About 0.2 cents in tokens for the text and about 30 cents for four renders. A full city analysis is a few cents. Hosting for the demo is a small cloud instance; a production deployment is a few hundred dollars a month.

**Why Mistral?** European model, good multilingual coverage for European markets, cheap, and the pipeline is model agnostic. Images are OpenAI because that is the best image model today.

**What is fake on this page?** Nothing is fabricated. Some things are thin: eight forecasts, a corpus one week deep after the Reddit removal, city analyses that vary in richness. It gets sharper every day it runs.

## If something breaks

The page loads but a panel says loading: wait ten seconds, it is a cold start.
A generation errors: check rooms is a number and the goal is a sentence. The form now says what is wrong.
The site does not answer: it is asleep. Refresh once and wait a minute. Keep talking over it.
