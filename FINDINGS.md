# Data Loading & Temporal Integrity Audit

## 🚨 Critical Finding: Temporal Inversion in Raw Data
The primary source CSV files (e.g., `lotto-gamedata-NL-1996.csv`) are stored in **Descending Order (Newest $\to$ Oldest)**.

**Evidence from Probe 1:**
- First Row Entry: `1996-12-28`
- Last Row Entry: `1996-01-03`
- **Status:** The raw data is "backwards" relative to standard time-series progression.

## ⚠️ The Conflict Points
The codebase currently uses three different sorting strategies, creating a high risk of temporal mismatch:

| Module | Strategy | Impact |
| :--- | :--- | :--- |
| `Helpers.load_data` | **Ascending** (Fixed) | ✅ **Safe**. It explicitly re-sorts data to Oldest $\to$ Newest for training. |
| `DataFetcher.py` | **Descending** (Raw) | ❌ **Dangerous**. Maintains the raw "Newest $\to$ Oldest" state. |
| `getLatestPrediction` | **Descending** (Targeted) | ✅ **Safe**. Specifically designed to find the most recent draw. |

## 📉 Example of Failure Scenario
If a developer creates a new feature using `DataFetcher` logic without implementing an explicit `.sort(reverse=False)`, the sliding window will regress in time.

**The "Time-Travel" Error:**
1. **Correct Sequence (Training):** 
   $[Draw_{Jan 1} \to Draw_{Jan 5}] \implies Target: [Draw_{Jan 8}]$
2. **Broken Sequence (using `Data_Fetcher` raw order):**
   $[Draw_{Jan 5} \to Draw_{Jan 1}] \implies Target: [Dec\ 30]$ 

**Result:** The model is being trained to predict the **past** based on the **future**, rendering all learned weights mathematically invalid for real-world forecasting.

## ✅ Recommendations
1.  **Standardize `DataFetcher.py`**: Update it to use `reverse=False` (Ascending) to match `Helpers.load_data`.
2.  **Unified Interface**: Ensure all data-loading utilities return a consistent **Chronological (Oldest $\to$ Newest)** array.
3.  **Regression Test**: Add a unit test in the CI pipeline that asserts `timestamps[0] < timestamps[-1]` for any loaded dataset.
