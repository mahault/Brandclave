# Active Inference Structure Learning

This module implements adaptive category discovery using principles from active inference and Bayesian nonparametrics.

## The Problem

Traditional approach (what we had before):
```python
class DesireCategory(str, Enum):
    ACCOMMODATION = "accommodation"
    EXPERIENCE = "experience"
    AMENITY = "amenity"
    # ... fixed categories decided upfront
```

**Problems with fixed categories:**
- We might miss important categories that emerge from data
- Categories might be too broad or too narrow
- We can't adapt to different cities having different desire patterns
- No uncertainty quantification

## The Solution: Structure Learning

Instead of fixed categories, we **learn them from data**:

```
Observation 1: "Looking for boutique hotel with wifi for remote work"
→ Creates Category: "Boutique + Wifi + Remote" (confidence: low)

Observation 2: "Need coworking space in my hotel"
→ Updates Category: "Remote Work + Coworking" (confidence: medium)

Observation 3: "Best digital nomad friendly hotels"
→ Updates Category: "Digital Nomad Accommodation" (confidence: higher)

Observation 4: "Cheap hostels with party atmosphere"
→ Doesn't fit! Creates new Category: "Budget + Social + Party"
```

## Key Concepts

### 1. Dirichlet Process Prior

We use a Chinese Restaurant Process-like mechanism:
- **New observations can join existing categories** (probability ∝ category size)
- **Or create new categories** (probability ∝ α / (n + α))
- This naturally balances parsimony (fewer categories) vs expressiveness (more categories)

### 2. Active Inference

The analyzer doesn't just passively observe - it **actively chooses what to search for**:

```
Current belief: High uncertainty about "luxury" category
→ Agent decides: Search for "luxury hotel {city}"
→ Observes results
→ Updates beliefs
→ Uncertainty reduced
→ Now uncertain about "budget" category
→ Agent decides: Search for "cheap hostel {city}"
→ ...
```

This is **expected free energy minimization**:
- **Epistemic value**: Reduce uncertainty (information gain)
- **Pragmatic value**: Get useful results (goal achievement)

### 3. Structure Adaptation

Categories can:
- **Expand**: When observations don't fit well, create new category
- **Merge**: When two categories become too similar, combine them
- **Refine**: Keywords and centroids update with each observation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AdaptiveCityAnalyzer                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Active Inference Loop                   │   │
│  │                                                      │   │
│  │  1. Compute confidence in current understanding      │   │
│  │  2. If confident enough → Stop                       │   │
│  │  3. Otherwise → Select next query (minimize EFE)     │   │
│  │  4. Execute query (scrape Reddit, YouTube, etc.)     │   │
│  │  5. Feed observations to StructureLearner            │   │
│  │  6. Go to 1                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              StructureLearner                        │   │
│  │                                                      │   │
│  │  - Maintains learned categories                      │   │
│  │  - Computes observation → category fits              │   │
│  │  - Decides when to create/merge categories           │   │
│  │  - Tracks assignment probabilities                   │   │
│  │  - Computes free energy (model quality)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Learned Categories                      │   │
│  │                                                      │   │
│  │  Category 1: "Digital Nomad + Coworking + Wifi"     │   │
│  │    - 15 observations, avg_fit: 0.72                 │   │
│  │                                                      │   │
│  │  Category 2: "Budget + Social + Hostel"             │   │
│  │    - 8 observations, avg_fit: 0.65                  │   │
│  │                                                      │   │
│  │  Category 3: "Luxury + Spa + Romantic"              │   │
│  │    - 12 observations, avg_fit: 0.78                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Free Energy

The model tracks **variational free energy**:

```
F = -Accuracy + Complexity

Accuracy: How well do observations fit their assigned categories?
Complexity: How many categories do we have vs expected?
```

**Lower free energy = better model**

This is used to:
1. Decide when to stop exploring (confidence)
2. Balance model complexity vs fit
3. Detect when structure needs to change

## API Usage

### Standard Analysis (fixed categories)
```
POST /api/city-desires
{"city": "Lisbon", "country": "Portugal"}
```

### Adaptive Analysis (learned categories)
```
POST /api/city-desires/adaptive
{"city": "Lisbon", "country": "Portugal"}
```

The adaptive endpoint returns additional fields:
- `num_learned_categories`: How many categories were discovered
- `model_confidence`: How confident we are in the analysis
- `free_energy`: Model quality metric
- `search_history`: What queries were executed
- `method`: "active_inference_structure_learning"

## VERSES Genius Integration

For production use, we integrate with the **VERSES Genius Active Inference API**,
which provides proper Bayesian inference instead of our local approximations.

### What Genius Provides

1. **Proper POMDP Action Selection**: Uses Expected Free Energy (EFE) minimization
   to choose what to search next, balancing exploration and exploitation.

2. **Bayesian Parameter Learning**: Updates model parameters from observed data
   using variational inference.

3. **Variable Factor Graphs (VFG)**: Native format for probabilistic models with
   proper uncertainty quantification.

4. **Real Variational Free Energy**: Proper computation of model quality metric.

### API Endpoints

```
# Check Genius connection status
GET /api/city-desires/genius/status

# Run Genius-powered analysis
POST /api/city-desires/genius
{"city": "Lisbon", "country": "Portugal"}
```

### Configuration

Add to `.env`:
```env
GENIUS_API_URL=https://your-agent-url.agents.genius.verses.ai
GENIUS_API_KEY=your_api_key_here
GENIUS_AGENT_ID=your_agent_id_here
GENIUS_LICENSE_KEY=your_license_key_here
```

### Comparison

| Feature | Local (adaptive) | Genius |
|---------|------------------|--------|
| Category inference | Cosine similarity | Bayesian posterior |
| Action selection | Heuristic | EFE minimization |
| Parameter learning | Running average | Variational inference |
| Free energy | Approximation | Proper VFE |
| Offline use | Yes | No |
| API required | No | Yes |

## Future Improvements

1. **Hierarchical categories**: Categories within categories
2. **Temporal dynamics**: Track how categories evolve over time
3. **Multi-modal**: Incorporate images, ratings, prices
4. **Counterfactual reasoning**: "What if this category didn't exist?"
5. **Transfer learning**: Share learned structure across cities

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Teh, Y. W. (2010). Dirichlet Process (Encyclopedia of Machine Learning)
- Fountas, Z. et al. (2020). Deep active inference agents using Monte-Carlo methods
- VERSES AI. (2024). Genius Active Inference Platform
