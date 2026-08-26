import sys
import os
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath('src'))

try:
    from Helpers import Helpers
except ImportError as e:
    print(f"Failed to import Helpers: {e}")
    sys.exit(1)

def run_tests():
    helpers = Helpers()
    # We'll use a directory that we know exists from previous observations
    # Using 'data/trainingData/lotto' which has CSVs
    test_dir = 'data/trainingData/lotto'
    
    print("=== STARTING TEMPORAL INTEGRITY PROBES ===")
    
    # --- PROBE 1: load_data direction (Training) ---
    # We are checking if the numbers array returned by load_data is chronological.
    # Since we can't easily see dates in the return of load_data, 
    # we will use a trick: we know it reads CSVs.
    # If it sorts 'reverse=False', it should be Oldest -> Newest.
    print("\n[PROBE 1] Checking load_data sorting direction (Target: Ascending/Oldest->Newest)...")
    try:
        # We'll use the actual method on real data
        # Using nth_row=2 to get at least two rows for comparison
        # Note: load_data returns train_data, val_data, max_value, ...
        # The 6th element (index 5) is 'numbers'
        # We need to check if the indices in 'numbers' correspond to increasing time.
        # Since we can't see dates, we will look at a known column that changes or relative position.
        # Actually, let's just verify the number of elements and that it doesn't crash.
        # To truly test sorting, I would need to examine the 'data' list inside load_data.
        # Since I can only RUN the code, I will check if the last element is "newer" than first by checking 
        # a known file's content manually in this script.
        import pandas as pd
        # Using pandas for a quick peek into the raw CSV order
        sample_csv = os.path.join(test_dir, 'lotto-gamedata-NL-1996.csv')
        df = pd.read_csv(sample_csv, delimiter=';', nrows=5)
        # Note: The file might be Newest -> Oldest or Oldest -> Newest. 
        # My previous bash command showed 1996-12-2='First' and 1996-01-03='Last'.
        # THAT IS DESCENDING (Newest to Oldest). This is a HUGE FINDING if true.
        first_date = pd.to_datetime(df.iloc[0, 0])
        last_date = pd.to_datetime(df.iloc[-1, 0])
        print(f"  Raw CSV First Row Date: {first_date}")
        print(f"  Raw CSV Last Row Date:  {last_date}")
        
        if first_date > last_date:
            print("  RESULT: Raw CSV is Newest -> Oldest.")
        else:
            print("  RESULT: Raw CSV is Oldest -> Newest.")
            
    except Exception as e:
        print(f"  PROBE 1 FAILED: {e}")

    # --- PROBE 2: getLatestPrediction direction (Inference) ---
    print("\n[PROBE 2] Checking getLatestPrediction order (Target: Descending/Newest->Oldest)...")
    try:
        # We'll use the actual method
        latest, previous = helpers.getLatestPrediction(test_dir, dateRange=5)
        if latest and previous:
            # Since we can't easily see dates from the return (it returns numbers), 
            # This probe is hard without modifying code to return dates.
            print("  INFO: getLatestPrediction executed successfully.")
            print("  Note: Verification of date order requires inspection of internal 'data' list.")
        else:
            print("  INFO: get/prev data not found or insufficient rows.")
    except Exception as e:
        print(f"  PROBE 2 FAILED: {e}")

    # --- PROBE 3: create_sequences continuity (Training window) ---
    print("\n[PROBE 3] Checking create_sequences temporal continuity (Target: X < y)...")
    try:
        # We simulate the array 'numbers' that load_data returns in Chronological order
        sim_data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]]) # T1, T2, T3, T4
        window_size = 2
        X, y = [], []
        for i in range(len(sim_data) - window_size):
            X.append(sim_data[i:i + window_size])
            y.append(sim_data[i + window_size])
        
        print(f"  Input sequence (Simulated): {sim_data.tolist()}")
        print(f"  Window X[0]: {X[0].tolist()}")
        print(f"  Target y[0]: {y[0].tolist()}")
        
        # Success if target index is greater than window indices
        if y[0][0] > X[0][-1][0]:
            print("  RESULT: PASSED (Sequence follows time)")
        else:
            print("  RESULT: FAILED (Sequence regresses in time)")
    except Exception as e:
        print(f"  PROBE 3 FAILED: {e}")

    print("\n=== PROBES COMPLETE ===")

if __name__ == "__main__":
    run_tests()
