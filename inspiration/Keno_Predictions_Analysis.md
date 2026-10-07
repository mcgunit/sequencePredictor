# Keno Prediction Improvements - Analysis & Recommendations

## Current Architecture

### How it Works Now
```python
# Line 1354-1357 in Predictor.py
1. DL model predicts 20 numbers → per-position softmax (shape: 20 positions × 70 classes)
2. score_numbers_from_prediction() aggregates to {number: max_probability} dict
3. generate_subset_from_scores() uses softmax sampling to pick N numbers from 20 for each subset size (5,6,7,8,9,10)

Single model trains on 70-class output, then post-hoc slices into subsets.
```

### Key Components
- **`score_numbers_from_prediction()`** (Helpers.py:2124): Collapses position-wise softmax to per-number scores via max-aggregation across positions
- **`generate_subset_from_scores()`** (Helpers.py:2911): Selects N numbers using either "top" mode or temperature-scaled softmax sampling

## The Problem Identified

Currently, **one model learns to optimize for 70-class output**, regardless of whether actual payout depends on hitting 5 matches vs 10 matches.

### Why This Matters for Profitability:
- Keno payouts scale nonlinearly by subset size:
  - 5/20 numbers drawn → modest payout
  - (8)/20 numbers drawn → significantly better payout
  - 10/20 numbers drawn → highest payout

- The loss gradient from a single model is **averaged** across all subselections during training, which dilutes the optimization signal for any one subset size.

## Proposed Improvement: Separate Per-Subset Models

### Architecture Change
Instead of predicting 20 numbers then slicing:
```
[Current] 1 Model → [20-class output] → (slice) → {subset_5, subset_6, ... subset_10}

[Proposed] SubModel5 → SubModel6 → SubModel7 → SubModel8 → SubModel9 → SubModel10
```

### Benefits

#### 1. **Loss Alignment with Payout Tables**
Each model optimizes against its actual payout schedule:

```python
# Current (all models share same objective)
def loss(prediction_20, real_20):
    # Averages over all subset sizes equally
    total_score = sum(subset_score(p[:n], real_20) for n in range(5,11)) / 6

# Proposed - true per-strategy optimization
def loss_5(predictions_5, real_20):
    matches = len(set(pred) & set(real_20))
    return -payout_table['played=5']['matches=matches']  # Optimized for 5-matches!
```

#### 2. **Specialized Feature Engineering**

| Subset Size | Optimal Features |
|------------|-----------------|
| pick 5 (harder to hit) | High confidence, tight clustering needed |
| pick 6 | Moderate variance tolerance |
| pick 7+ | Can afford dispersion; ensemble voting becomes more valuable |

Example - **Markov transition model** could be configured differently:
```python
# For subset_5: Use stricter minOccurrences (14+), higher alpha, lower temperature
# For subset_8: Use looser minOccurrences (9-10), incorporate pair-scoring weights
```

#### 3. **Stronger Hyperopt Signals**

Current setup (from TuningGate.py:63):
```python
GAMES = ("euromillions", "lotto", "eurodreams", "keno", ...)
SKIPPED_STRATEGIES = ("KenoSubsetTuning",)  # Can't tune independently!
```

Problem: `use_5`, `use_6`, etc. only control which predictions are **extracted after training**, not how models train.

With separate models, each subset size becomes first-class citizen for hyperopt tuning with real profit gradients.

#### 4. **Better Ensemble Voting**

vote-ensemble rows already use softmax temperature as tie-breaker (line 1289):
```python
# Current: Both main tickets and subsets come from same model → deterministic slicing
for subset_size in range(5, 11):
    subset = generate_subset_from_scores(number_scores, ..., subset_size)

# Proposed: Each subset model has independent "voice" in the vote row. 
# Can apply different ensemble weights based on which strategy worked best historically.
```

---

## Implementation Options

### Option A: Full Decomposition (Maximum Improvement)

Create 6 independent models per draw, each outputting its dedicated subset size:

**Pros:**
- True specialization to payout structure
- Strongest hyperopt signals for subset-tuning
- Clean separation of concerns

**Cons:**
- 6× training compute cost
- Model instance proliferation (current: ~10 models; proposed: ~60 models)
- More complex ensemble coordination

**Code Structure:**
```python
# New Models per file or unified model class with strategy dimension

class KenoStatisticalModelV2:
    # Configured via prefix key, e.g., "markov_5" means subset_size=5
    
    def train(self, data, target_size=5):  # Fixed output size
        # Loss function weighted for 5 matches specifically
        pass
        
    def predict(self):
        return self._sample_from_distributions()[:self.target_size]

# Usage in statisticalMethod():
submodels = {n: {} for n in [5,6,7,8,9,10]}

for game_type in games:
    # Build dedicated submodel dicts per subset size
    submodels[5]["markov"] = KenoStatisticalModelV2(target_size=5, ...)
    submodels[6]["poisson_mc"] = PoissonMCModel(target_size=6, ...)
    ...

# Train all at once (parallel), stack by subset:
for size in range(5, 11):
    row_predictions[size] = [model.predict() for model in submodels[size].values()]
```

---

### Option B: Hybrid Approach (Recommended) ⭐

Keep main 20-class prediction, but use per-subset features and scoring internally, plus separate hyperopt space per subset size:

**Pros:**
- Minimal training overhead (still ~10 models total)  
- Stronger profit gradients via loss weighting
- Cleaner feature engineering per strategy class

**Cons:**
- Still one model type, so generalization constraints remain

**Implementation:**
```python
# In each Statistical Method (Markov.py, PoissonMC.py, etc.):

def compute_loss_scores(self, prediction_20, real_result):
    """Override default behavior to weight subset sizes appropriately."""
    scores = {}  # Already computed in Helpers
    # Apply payout-weighted aggregation instead of uniform averaging:
    
    weighted_score = 0.0
    weighting_factors = {5: 1.0, 6: 1.2, 7: 1.5, 8: 1.8, 9: 2.0, 10: 2.2}  # Example
    for n in range(5, 11):
        matches = len(set(pred[:n]) & set(real_result))
        weighted_score += weighting_factors[n] * score_table.get(n, matches)
    
    return scores_by_match_weighted=weighted_score

# This way each base model remains simple but its objective aligns better with profit.
```

---

### Option C: Soft Architecture (Minimal Changes)

Keep current prediction pipeline, improve only the extraction and scoring logic:

**Changes:**
1. Replace `softmax/sampling` mode with `"top-k"` deterministic selection
2. Use ensemble voting for each subset separately
3. Adjust temperature per hyperopt trial to find best extraction strategy

**Tradeoff:** Quick win without model proliferation, but doesn't address the core loss misalignment issue.

---

## Feature Engineering Per Subset Size

### Markov Chain Model: Subset-Specific Parameters

| Parameter | Pick 5 | Pick 6-7 | Pick 8+ |
|-----------|--------|----------|---------|
| `minOccurrences` | Higher (rare numbers needed) | Moderate | Lower (can hit on common) |
| `softmaxTemperature` | Lower (~0.2) - more deterministic | Medium (~0.35) | Higher (~0.5+) |
| `pairScoringWeight` | Zero or negative | Neutral positive | Strong positive contribution |
| `recencyWeight` | Emphasize recent patterns | Standard decay | Incorporate longer context |

**Rationale:** A strategy targeting 10/20 hits must tolerate numbers appearing frequently; a "5-number" strategy bets on rarer, more concentrated configurations.

---

## Statistical Model-Specific Recommendations

### Markov Chain
```python
# Current single run
markov.setSoftMaxTemperature(temp)  # All subsets use same temp
markov.generate_subset_from_scores(scores, [5,6,7,8,9,10], mode="softmax")

# Improvement - subset-aware
class MarkovForKenoSubset:
    def __init__(self, target_size):
        self.target_size = target_size  # Configures softmax temp via hyperopt
        
    def train(self, data, **hyperparams_5_through_10_for_this_size):
        temp = self.hyperopt_tuned_temp_by_subset_size[self.target_size]
        
    def generate_submission(self):
        return TopKSelection(self.probs_dict, k=self.target_size)  # Deterministic

```

### Poisson Monte Carlo
```python
# Current: Single simulation run samples all 20, then slices for subsets
pmc.setNumSimulations(n)  # Same n for everything

# Improvement - calibrate sims per target size
class PoissonMCForKenoSubset:
    def __init__(self, target_size):
        self.simPerConfig = hyperopt_tuned_sims_by_subset[self.target_size]
        # Example mapping (tunable!):
        self.base_sim_map = {5: 500,   # Fewer sims for harder-to-hit configs
                             7: 600,   # Sweet spot where bet most
                             10: 200}  # More sims for easier subset

```

### LightGBM / XGBoost
```python
# Current: Trains on [number, markov_prob/poisson_prob] per row, outputs 70-class logits per draw

# Improvement - stratified training with per-strategy loss weighting
def build_keno_training_dataset_v2(self, games_data):
    rows = []
    for game_row in games_data:
        # For each subset size, create its own row variant with transformed target:
        for size in [5,6,7,8,9,10]:
            target_label = self._encode_subset_size_from_payout_table(size)
            feature_transform = self._feature_transform_for_target_size(size)  # e.g.
            
            rows.append({
                "markov_prob_1": markov.get_prob(1),
                "transformed_for_target_5": feature_transform(game_row, target=5),
                "draw_size_encoded": game_row['size'],
                ...
            })

```

---

## Ensemble Voting Per Subselection Size

### Current Implementation (Line 1289)
Single softmax-temperature determines slicing mode for both main ticket and subselections:
```python
subsetMode = bestParams_json_object["markov_mcSubsetSelectionMode"]  # "softmax" or "top-k"
for size in range(5, 11):
    subsets[size] = generate_subset_from_scores(score_dict, ..., mode=subsetMode, temperature=subsetTemp)
```

### Proposed Per-Size Ensemble Mode

```python
ENSET_SUBSET_MODES = hyperopt_tuned  # Could be: ["top-k", "softmax_0.3", "weighted_vote"]

# Example: vote ensemble votes independently per subset size
def weighted_vote_for_subset_size(ensemble_predictions, real_draw, target_size=5):
    """Each ensemble row votes on the subset of its own size, then aggregated."""
    predictions = [model.predict(target=target_size) for model in self.models]
    # Vote only counts against matches at this specific size:
    combined = weighted_vote(predictions)
    score = keno_ticket_profit(combined, real_draw)  # Correctly calculated now!
    
# Hyperopt space becomes clean:
# - Which models contribute to subset-5 vote?
# - Weight ensemble_1_keno_5 vs ensemble_2_keno_5 
# (previously hyperopt had no way to tune this because subsets were post-hoc slices)
```

---

## Payout Table Alignment Check

Current profit calculation (Helpers.py:360):
```python
def keno_ticket_profit(self, prediction, real_result):
    """Net profit for a single Keno subset ticket."""
    played = len(prediction)  # This matches the actual number of tickets we play
    if played < 5 or played > 10:  # Only subsets get tracked correctly
    
    table = self.PAYOUT_TABLE_KENO
    
    stake = -table["lost"]  # Already adjusted for 1 EUR ticket cost
    matches = len(set(map(int, prediction)) & set(map(int, real_result)))  # Correct!
    
    payout = table.get(played, {}).get(matches)  # This lookup is good!
    return payout - stake
```

✅ **Good**: Profit calculation already accounts for subset size and match count.

❌ **Bad**: The model never learned this alignment because gradient averaged the same loss function across all six extraction strategies.

---

## Testing & Validation Strategy

### Unit Tests to Add

```python
# Test suite: KenoSubsetAlignmentTests  # NEW FILE NEEDED

class TestPerSizeOptimization:
    def test_5_subset_hardest_optimal_strategy(self):
        """Pick-5 should prefer concentrated, high-probability configurations."""
        model_config_5 = {
            "minOccurrences": 14,
            "temperature": 0.25,  # Deterministic selection
            "softmax_mode": "top-k"
        }
        expected_behavior = "Should select numbers from frequent sequences only"
    
    def test_8_subset_benefit_from_dispersion(self):
        """Pick-8 should incorporate broader number spread."""
        model_config_8 = {
            "minOccurrences": 9,
            "temperature": 0.65,  # More diversity
            "pairScoringFactor": 0.4  // Use pair patterns in selection
        }
    
    def test_profit_gradient_aligns_with_loss_weighting(self):
        """Verify that higher weighting of target size produces better profit scores."""
        for size in range(5, 12):
            weights = self._get_weights_for_size(size)
            predictions, profit = model.predict_and_score(real_draw=size)
            gradient_direction = (profit > self.reference_baseline).astype(float)
            assert np.isclose(gradient_direction, weighted_loss_gradient, rtol=0.1)
```

---

## Alternative: Keep Current for Now, Add Loss Weighting Future-Proofing

If to ship quickly and iterate later, minimum viable change is to add loss-weighted scoring in the existing models:

### Quick Patch (Insert in StatisticalMethod or each Model's predict method):

```python
# In Predictor.py::statisticalMethod() OR inline per-model prediction path
# NEW - Per subset size loss weighting:
def compute_profit_aligned_scores(number_scores_dict, real_draw, use_per_size_weights=True):
    """Override raw max-aggregation with payout-weighted scoring."""
    
    # Current (line 1354):
    #     number_scores = helpers.score_numbers_from_prediction(newPredictionRaw, unique_labels)
    #     subset_5 = helpers.generate_subset_from_scores(number_scores, ..., mode="softmax", temp=0.35)
    
    # NEW - Compute all per-size scores, weight them:
    scored_by_size = {}  # size: [scores] where scores = {number: prob}
    weighted_aggregations = {}  # size: float
    
    for model_prediction in predictions:
        # Keep existing extraction (great!)
        number_scores = helpers.score_numbers_from_prediction(
            model_prediction['raw'], unique_labels
        )
        
        # New - Compute AND WEIGHT separately per subset size:
        for n, candidates in enumerate(range(5, 11)):
            probs = extract_n_probs(number_scores, candidates)
            subset_candidates = helpers.generate_subset_from_scores(
                number_scores, real_draw_candidates, candidates,
                mode="softmax", temperature=TEMP_PER_SIZE[candidates]
            )
            
            scored_by_size[candidates].append(subset_candidates)  # Keep all
            
    if use_per_size_weights:
        # Compute weighted ensemble score by size:
        weights_keno = [0.35, 0.38, 0.42, 0.46, 0.51, 0.55]  # Example hyperopt-tuned
        final_score = sum(w * score_for_size(model_probs) for w,s in zip(weights_keno, scored_by_size.values()))
    
    return weighted_aggregations, model_predictions_with_all_subsets_stored
```

---

## Recommended Action Plan

### Option 1: Full Rewrite (`statisticalMethod_KenoV2()`)
**Timeline:** 2-3 hours development, thorough testing  
**Pros:** Clean architecture, maximum future-proofing  
**Cons:** High risk; needs extensive regression testing

### Option 2: Loss Weighting Patch (Recommended) ⭐
**Timeline:** 1-2 hours, minimal refactoring  
**Pros:** Immediate payout alignment benefit without architectural overhaul  
**Cons:** Still one model type per game, but better signal-to-noise now  

### Option 3: Conservative Ensemble Split
**Timeline:** 0.5 hour  
**Pros:** Safe change using existing prediction infrastructure  
**Cons:** Doesn't solve core loss misalignment issue  

---

## Key Decision Points to Consult on

1. **Willing to refactor all statistical methods** (Markov, PMC, Laplace, Hybrid) for this change, or want incremental patch?

2. **For Poisson Monte Carlo specifically**: Can we configure different Monte Carlo simulation counts per subset size without breaking the base model?

3. **Ensemble coordination**: Want each subselection (5/6/7/8/9/10) to have its own dedicated ensemble row, or do prefeingr a single weighted vote that varies by target size?

4. **Immediate need vs experimental**: Is this for production use right now (pick option 2) or exploring research directions (any of the three)?

