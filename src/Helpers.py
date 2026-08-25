import os, json, subprocess, optuna, argparse
import numpy as np
import scipy.special
import asciichartpy

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder


class Helpers():

    PAYOUT_TABLE_KENO = {
        10: { 0: 3, 5: 1, 6: 4, 7: 10, 8: 200, 9: 2000, 10: 250000 },
        9: { 0: 3, 5: 2, 6: 5, 7: 50, 8: 500, 9: 50000 },
        8: { 0: 3, 5: 4, 6: 10, 7: 100, 8: 10000 },
        7: { 0: 3, 5: 3, 6: 30, 7: 3000 },
        6: { 3: 1, 4: 4, 5: 20, 6: 200 },
        5: { 3: 2, 4: 5, 5: 150 },
        4: { 2: 1, 3: 2, 4: 30 },
        3: { 2: 1, 3: 16 },
        2: { 2: 6.50 },
        "lost": -1  # Because it cost 1 euro
    }

    PAYOUT_TABLE_PICK3 = {
        "straight": 500,
        "box_with_doubles": 160,
        "box_no_doubles": 80,
        "front_pair": 50,
        "back_pair": 50,
        "last_number": 1,
        "lost": -4  # Because it cost 1 euro but for each prediction and we need to choose all win orders like: straight, etc...
    }

    # The trailing special/bonus column(s)' own number range - distinct from
    # the main numbers' range (Euromillions stars 1-12, EuroDreams dream
    # number 1-5, VikingLotto viking number 1-8; observed directly from the
    # training CSVs). Lotto's bonus column isn't included here because it's
    # dropped entirely via skipLastColumns rather than modeled.
    SPECIAL_UNIQUE_LABELS = {
        "euromillions": np.arange(1, 13),
        "eurodreams": np.arange(1, 6),
        "vikinglotto": np.arange(1, 9),
    }

    def get_unique_labels(self, dataPath, special=False):
        """
        Fixed one-hot label range for a game - kept constant regardless of
        which data window a given call loads, so a model's Dense/Embedding
        output shapes never change between training calls and saved weights
        can always be reloaded (see e.g. the `Numclasses:` prints staying
        identical across HyperoptDeepLearning.py's history-rebuild loop).

        special=True returns the trailing special/bonus column(s)' own
        (smaller) range instead of the main-number range - see
        SPECIAL_UNIQUE_LABELS. Falls back to the main range if dataPath
        doesn't match a known special-column game (shouldn't happen in
        practice - callers only pass special=True for those games).
        """
        if special:
            for game, labels in self.SPECIAL_UNIQUE_LABELS.items():
                if game in dataPath:
                    return labels
            return np.arange(1, 51)

        # Euromillions are 50 numbers, Lotto are 45 numbers
        unique_labels = np.arange(1, 51)  # This should create an array [1, 2, ..., 50]
        if "lotto" in dataPath:
            unique_labels = np.arange(1, 46)  # This should create an array [1, 2, ..., 45]
        if "keno" in dataPath:
            unique_labels = np.arange(1, 81)  # This should create an array [1, 2, ..., 80]
        if "vikinglotto" in dataPath:
            unique_labels = np.arange(1, 49)  # This should create an array [1, 2, ..., 49]
        if "pick3" in dataPath:
            unique_labels = np.arange(0, 10)  # This should create an array [0, 2, ..., 9]
        if "jokerplus" in dataPath:
            unique_labels = np.arange(0, 10).tolist()
            unique_labels.append("Boogschutter")
            unique_labels.append("Kreeft")
            unique_labels.append("Weegschaal")
            unique_labels.append("Schorpioen")
            unique_labels.append("Stier")
            unique_labels.append("Leeuw")
            unique_labels.append("Maagd")
            unique_labels.append("Ram")
            unique_labels.append("Waterman")
            unique_labels.append("Vissen")
            unique_labels.append("Steenbok")
            unique_labels.append("Tweeling")
        return unique_labels

    def run_model_with_special_column(self, model, generateSubsets=None, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Runs a statistical model's run() for games with trailing special/bonus
        columns (Euromillions has 2 star columns, EuroDreams 1 dream number,
        VikingLotto 1 super viking). Those columns have their own, smaller
        number range, so they're modeled completely independently from the
        main numbers (own transition matrix/frequencies/range) as a single
        group and simply appended to the main-number prediction afterwards -
        never merged or re-sorted together with it.

        For games without a special column (specialColumnCount=0), this is
        equivalent to a plain model.run(...) call, using skipLastColumns as-is
        (e.g. Lotto passes skipLastColumns=1 to drop its unplayed bonus number).
        """
        if generateSubsets is None:
            generateSubsets = []

        main_prediction, subsets = model.run(
            generateSubsets=generateSubsets,
            skipRows=skipRows,
            skipLastColumns=specialColumnCount if specialColumnCount > 0 else skipLastColumns
        )

        if specialColumnCount <= 0:
            return main_prediction, subsets

        special_prediction, _ = model.run(
            generateSubsets=[],
            skipRows=skipRows,
            skipLastColumns=0,
            specialColumnCount=specialColumnCount
        )

        combined_prediction = list(main_prediction) + list(special_prediction)

        return combined_prediction, subsets

    def keno_ticket_profit(self, prediction, real_result):
        """
        Net profit (in euro) for a single Keno subset ticket.
        Only subsets of 5-10 numbers have real payouts; below that, profit is not tracked.
        """
        played = len(prediction)
        if played < 5 or played > 10:
            return None

        table = self.PAYOUT_TABLE_KENO
        matches = len(set(map(int, prediction)) & set(map(int, real_result)))
        profit = table.get(played, {}).get(matches, table["lost"])

        if profit != table["lost"]:
            return profit
        else:
            return profit + table["lost"]

    def pick3_ticket_profit(self, prediction, real_result):
        """
        Net profit (in euro) for a single Pick3 ticket. Order matters (straight/pair payouts).
        """
        if len(prediction) != 3 or len(real_result) != 3:
            return None

        table = self.PAYOUT_TABLE_PICK3
        pred = [int(p) for p in prediction]
        actual = [int(a) for a in real_result]

        if pred == actual:
            return table["straight"] + table["lost"]

        if sorted(pred) == sorted(actual):
            pred_counts = {x: pred.count(x) for x in pred}
            if 2 in pred_counts.values():
                return table["box_with_doubles"] + table["lost"]
            return table["box_no_doubles"] + table["lost"]

        if pred[0:2] == actual[0:2]:
            return table["front_pair"] + table["lost"]

        if pred[1:3] == actual[1:3]:
            return table["back_pair"] + table["lost"]

        if pred[2] == actual[2]:
            return table["last_number"] + table["lost"]

        return table["lost"]

    def generate_model_performance_report(self, databaseDir, outputFileName="modelPerformance.json"):
        """
        Scans every game folder under databaseDir and writes a per-game,
        per-model performance summary over ALL scored history (each file's
        currentPrediction vs its realResult), for the web UI's History page.

        Ranking metric per game:
        - keno/pick3 (real payout tables exist): average profit per bet -
          per-bet rather than total, because models joined at different times
          (the boosting rows are weeks old, the statistical rows months) and
          bet different numbers of Keno subsets, so totals aren't comparable.
        - every other game: average hits of the model's main ticket.

        The full ranking is stored per game (not just the winner) so the UI
        can grow without regenerating anything; "draws" is included so small
        sample sizes are visible instead of masquerading as strong averages.
        """
        report = {
            "generatedAt": datetime.now().isoformat(timespec="seconds"),
            "games": {},
        }

        for game in sorted(os.listdir(databaseDir)):
            gameDir = os.path.join(databaseDir, game)
            if not os.path.isdir(gameDir):
                continue

            hasPayout = "keno" in game or "pick3" in game
            stats = {}

            for fileName in os.listdir(gameDir):
                if not fileName.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(gameDir, fileName), "r") as infile:
                        dayData = json.load(infile)
                except Exception:
                    continue

                realResult = dayData.get("realResult")
                scoredModels = dayData.get("currentPrediction") or []
                if not realResult or not scoredModels:
                    continue

                for model in scoredModels:
                    name = model.get("name")
                    predictions = model.get("predictions") or []
                    if not name or not predictions or not predictions[0]:
                        continue

                    entry = stats.setdefault(name, {
                        "draws": 0, "hits_total": 0, "best_hits": 0,
                        "profit_total": 0.0, "bets": 0,
                    })

                    mainTicket = predictions[0]
                    hits = len(set(map(int, mainTicket)) & set(map(int, realResult)))
                    entry["draws"] += 1
                    entry["hits_total"] += hits
                    entry["best_hits"] = max(entry["best_hits"], hits)

                    if "keno" in game:
                        # Profit exists only for playable 5-10-number subsets,
                        # not the full 20-number ticket.
                        for ticket in predictions:
                            profit = self.keno_ticket_profit(ticket, realResult)
                            if profit is not None:
                                entry["profit_total"] += profit
                                entry["bets"] += 1
                    elif "pick3" in game:
                        profit = self.pick3_ticket_profit(mainTicket, realResult)
                        if profit is not None:
                            entry["profit_total"] += profit
                            entry["bets"] += 1

            models = []
            for name, entry in stats.items():
                avgHits = entry["hits_total"] / entry["draws"] if entry["draws"] else 0.0
                profitPerBet = entry["profit_total"] / entry["bets"] if entry["bets"] else None
                models.append({
                    "name": name,
                    "draws": entry["draws"],
                    "avg_hits": round(avgHits, 3),
                    "best_hits": entry["best_hits"],
                    "profit_total": round(entry["profit_total"], 2) if entry["bets"] else None,
                    "profit_per_bet": round(profitPerBet, 3) if profitPerBet is not None else None,
                    "bets": entry["bets"],
                })

            if not models:
                continue

            if hasPayout:
                sortKey = lambda m: (m["profit_per_bet"] if m["profit_per_bet"] is not None else float("-inf"))
                metric = "profit_per_bet"
            else:
                sortKey = lambda m: m["avg_hits"]
                metric = "avg_hits"

            # A model scored on 1-2 draws can top an average-based ranking on
            # a single lucky day (models join at different times - boosting
            # rows are much younger than the statistical ones). Rank models
            # with a reasonable sample first; the rest keep their stats but
            # sort below, and the UI shows the draw count either way.
            maxDraws = max(m["draws"] for m in models)
            minDraws = min(10, maxDraws)
            established = sorted([m for m in models if m["draws"] >= minDraws], key=sortKey, reverse=True)
            young = sorted([m for m in models if m["draws"] < minDraws], key=sortKey, reverse=True)
            models = established + young

            report["games"][game] = {
                "metric": metric,
                "minDrawsForRanking": minDraws,
                "bestModel": models[0]["name"],
                "models": models,
            }

        # ------------------------------------------------------------------
        # Phase-shift (lag) analysis: score each file's newPrediction not only
        # against the draw it was made for (lag 1) but against the K draws
        # after that. A model whose signal is real but time-shifted would
        # peak at some lag > 1; an aligned model peaks at lag 1; a model with
        # no temporal information shows a flat line across all lags (its hits
        # are draw-independent frequency structure). Uses the main ticket
        # only. Pick3 is scored positionally (digit-in-right-place count) -
        # set intersection is the wrong question for a positional game with
        # repeated digits.
        # ------------------------------------------------------------------
        MAX_LAG = 30

        for game in list(report["games"].keys()):
            gameDir = os.path.join(databaseDir, game)
            isPick3 = "pick3" in game

            def dayHits(prediction, realResult):
                if isPick3:
                    return sum(1 for p, r in zip(prediction, realResult) if int(p) == int(r))
                return len(set(map(int, prediction)) & set(map(int, realResult)))

            # Chronologically ordered (date, realResult, {model: mainTicket})
            days = []
            for fileName in os.listdir(gameDir):
                if not fileName.endswith(".json"):
                    continue
                try:
                    fileDate = datetime.strptime(fileName.replace(".json", ""), "%Y-%m-%d")
                except ValueError:
                    continue
                try:
                    with open(os.path.join(gameDir, fileName), "r") as infile:
                        dayData = json.load(infile)
                except Exception:
                    continue
                tickets = {
                    model.get("name"): model["predictions"][0]
                    for model in (dayData.get("newPrediction") or [])
                    if model.get("name") and model.get("predictions") and model["predictions"][0]
                }
                days.append((fileDate, dayData.get("realResult") or [], tickets))
            days.sort(key=lambda d: d[0])

            lagStats = {}  # model -> lag -> [hits_total, count]
            for i, (_, _, tickets) in enumerate(days):
                for lag in range(1, MAX_LAG + 1):
                    j = i + lag - 1  # file i's newPrediction targets file i+1's draw = lag 1
                    if j + 1 > len(days) - 1:
                        break
                    realResult = days[j + 1][1]
                    if not realResult:
                        continue
                    for name, ticket in tickets.items():
                        perLag = lagStats.setdefault(name, {})
                        bucket = perLag.setdefault(lag, [0, 0])
                        bucket[0] += dayHits(ticket, realResult)
                        bucket[1] += 1

            lagAnalysis = {}
            for name, perLag in lagStats.items():
                lags = []
                for lag in range(1, MAX_LAG + 1):
                    total, count = perLag.get(lag, (0, 0))
                    lags.append({"lag": lag, "avg_hits": round(total / count, 3) if count else None, "n": count})
                scored = [l for l in lags if l["avg_hits"] is not None and l["n"] >= 10]
                bestLag = max(scored, key=lambda l: l["avg_hits"])["lag"] if scored else None
                lagAnalysis[name] = {"lags": lags, "best_lag": bestLag}

            report["games"][game]["lagAnalysis"] = lagAnalysis

        outputPath = os.path.join(databaseDir, outputFileName)
        with open(outputPath, "w") as outfile:
            json.dump(report, outfile, indent=2)
        print(f"Model performance report written to {outputPath}")
        return report

    def load_weights_if_fingerprint_matches(self, model, model_path, fingerprint):
        """
        Loads saved .weights.h5 into `model` only if the fingerprint stored
        next to it matches `fingerprint` (the architecture-shaping parameters
        the weights were trained under). Returns True if weights were loaded.

        Why: warm-starting from saved weights saves real training time, but
        after a hyperopt run changes an architecture parameter (units, layers,
        window size, ...) the saved weights either fail to load (shape
        mismatch, killing that model's prediction row until someone manually
        deletes the file) or - worse - load silently when shapes happen to
        match, so the newly tuned parameters never actually take effect.
        Weights without a fingerprint file (saved before this existed) are
        treated as mismatched: one fresh retrain, after which the fingerprint
        is written by save_weights_with_fingerprint.

        Only include parameters that change weight SHAPES or input semantics
        in the fingerprint (units, layers, window size, heads, class counts) -
        not dropout rate/learning rate/l2 etc., for which warm-starting stays
        valid and desirable.
        """
        weights_path = f"{model_path}.weights.h5"
        fingerprint_path = f"{model_path}.fingerprint.json"

        if not os.path.exists(weights_path):
            return False

        stored = None
        if os.path.exists(fingerprint_path):
            try:
                with open(fingerprint_path, "r") as infile:
                    stored = json.load(infile)
            except Exception:
                stored = None

        if stored != fingerprint:
            print(f"Saved weights at {weights_path} were trained with different "
                  f"architecture parameters (or predate fingerprinting) - training fresh")
            return False

        try:
            model.load_weights(weights_path)
            print(f"Loading weights from {weights_path}")
            return True
        except Exception as e:
            print(f"Failed to load weights from {weights_path}, training fresh: {e}")
            return False

    def save_weights_with_fingerprint(self, model, model_path, fingerprint):
        """Counterpart of load_weights_if_fingerprint_matches: persists the
        weights together with the architecture fingerprint they were trained
        under."""
        model.save_weights(f"{model_path}.weights.h5")
        try:
            with open(f"{model_path}.fingerprint.json", "w") as outfile:
                json.dump(fingerprint, outfile, indent=2)
        except Exception as e:
            print(f"Failed to write weights fingerprint for {model_path}: {e}")

    def getLatestPrediction(self, csvPath, dateRange=None):
        """
            Get latest result from csv file.
            If dateRange is provided it will return a list containing multiple results
            If dateRange is None (default) it will return the current and previous result

            @param csvFile: The csv file to get the latest prediction from
            @param dateRange: The numbers of days to get the predictions from
            
        """
        # Initialize an empty list to hold the data
        data = []

        print(f"Getting latest prediction from folder: {csvPath}")
        try:
            for csvFile in os.listdir(csvPath):
                if csvFile.endswith(".csv"):
                    # Construct full file path
                    file_path = os.path.join(csvPath, csvFile)
                    
                    # Load data from the file
                    csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1)

                    if not isinstance(csvData[0], (list, np.ndarray)):
                        print("Need to reform loaded latest prediction data")
                        csvData = [csvData.tolist()]

                    #print("Reading CSV: ", csvFile)
                    
                    # Append each entry to the data list
                    for entry in csvData:
                        # Attempt to parse the date
                        date_str = entry[0]
                        try:
                            # Use dateutil.parser to parse the date
                            date = parse(date_str)
                        except Exception as e:
                            print(f"Date parsing error for entry '{date_str}': {e}")
                            continue  # Skip this entry if date parsing fails
                        
                        # Convert the rest to integers
                        try:
                            numbers = list(map(int, entry[1:]))  # Convert the rest to integers
                        except ValueError as ve:
                            print(f"Number conversion error for entry '{entry[1:]}': {ve}")
                            continue  # Skip this entry if number conversion fails
                        
                        data.append((date, numbers))  # Store as a tuple (date, number1, number2, ...)

        except Exception as e:
            print(f"Error processing file {csvFile}: {e}")

        #print("Data: ", data)

        # If data is not empty, find the most recent entry
        if data:
            if dateRange is None:
                if len(data) == 1:
                    # Sort data by date (the first element of the tuple)
                    data.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order
                    previous_entry = None
                    latest_entry = data[0]  # Get the most recent entry
                    return (latest_entry, previous_entry)  # Return the most recent entry
                else:
                    # Sort data by date (the first element of the tuple)
                    data.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order
                    previous_entry = data[1] # needed to find the previous prediction to compare with the latest entry
                    latest_entry = data[0]  # Get the most recent entry
                    return (latest_entry, previous_entry)  # Return the most recent entry
            else:
                data.sort(key=lambda x: x[0], reverse=False)  # Sort in ascending order
                # dateRange counts actual draw rows, not calendar days - a
                # calendar cutoff (datetime.now() - relativedelta(days=...))
                # silently starves games with fewer than 7 draws/week (e.g.
                # Euromillions draws twice a week, so dateRange=8 calendar
                # days would only ever resolve to ~2 real draws). Matches
                # HyperoptStatistics.py's row-index-based day-to-rebuild
                # semantics (start_index = total_rows - days_to_rebuild).
                filtered_data = data[-dateRange:] if dateRange > 0 else []

                return filtered_data
        else:
            print("No data found.")
            return None  # Return None if no data was found
        

    def find_best_matching_prediction(self, sequence, predictions_dict):
        sequence_set = set(sequence)  # Convert the sequence to a set for fast lookup

        best_match = {
            "model": None,
            "prediction": None,
            "matching_numbers": [],
            "match_count": 0
        }

        for model in predictions_dict:
            model_name = model["name"]
            for predicted_list in model["predictions"]:
                matching_numbers = list(sequence_set.intersection(predicted_list))
                match_count = len(matching_numbers)

                # If this prediction has more matches, update the best match
                if match_count > best_match["match_count"]:
                    best_match["model"] = model_name
                    best_match["prediction"] = predicted_list
                    best_match["matching_numbers"] = matching_numbers
                    best_match["match_count"] = match_count

        return best_match  # Return full details of the best matching prediction
    
    def decode_predictions(self, raw_predictions, labels, nHighestProb=0, remove_duplicates=True):
        """
        Decode the prediction based on probability and match with corresponding labels.
        Ensures distinct selections across probability ranks if remove_duplicates=True.

        Parameters
        ----------
        raw_predictions : np.ndarray
            Array of shape (num_samples, num_classes) containing probabilities.
        labels : list or np.ndarray
            List of labels corresponding to the classes.
        nHighestProb : int
            Rank of probability to consider (0 = highest, 1 = second-highest, etc.).
        remove_duplicates : bool, optional
            If True, prevents selecting the same number across different ranks.

        Returns
        -------
        list
            Decoded predictions as per the provided labels.
        """

        raw_predictions = np.array(raw_predictions)  # Shape (numbersLength, num_classes)
        labels = np.array(labels)  # Shape (num_classes,)

        sorted_indices = np.argsort(raw_predictions, axis=1)[:, ::-1]  # Sort descending (highest first)

        if remove_duplicates:
            selected_indices = []
            for i, row in enumerate(sorted_indices):
                # Get the nth highest index, skipping previously selected ones
                unique_indices = [idx for idx in row if idx not in selected_indices]
                selected_indices.append(unique_indices[nHighestProb])  # Pick the nth highest

        else:
            selected_indices = sorted_indices[:, nHighestProb]  # Direct selection without duplicate filtering

        return labels[selected_indices].tolist()
    
    def model_weights_are_finite(self, model):
        """
        True if every weight tensor in the model is free of NaN/Inf.

        Exploding gradients (mid-training loss->nan, most common with
        certain hyperopt-sampled learning rate/dropout/l2 combos, especially
        on Keno's large 20-position/80-class output) leave the model's live
        weights corrupted even when EarlyStopping's restore_best_weights
        can't help - if the very first epoch of a run is already NaN, there
        is no earlier "best" epoch to restore to. Call this right before
        model.save_weights()/model.save() so a corrupted run doesn't get
        persisted to disk and then reloaded (and re-corrupt) every
        subsequent retrain step - see HyperoptDeepLearning.py's
        process_single_history_entry, which reloads this same weights file
        on every history day.
        """
        return all(np.all(np.isfinite(w)) for w in model.get_weights())

    def predict_numbers(self, model, numbers, window_size=10):
        # Take the last `window_size` draws
        input_seq = numbers[-window_size:]  # shape (10, 3)

        # Reshape to model input shape: (1 sample, 10 time steps, 3 features)
        input_seq = np.expand_dims(input_seq, axis=0)  # shape (1, 10, 3)

        # Predict
        raw_predictions = model.predict(input_seq)

        # Dual-head models (main numbers + special/bonus column, see
        # LSTM.py/TCN.py/UnifiedLstmTcn.py/UnifiedLstmGruTcn.py create_model)
        # return a list of two batched outputs instead of one - unwrap both.
        if isinstance(raw_predictions, (list, tuple)):
            return [output[0] for output in raw_predictions]

        return raw_predictions[0]

    def combine_special_prediction(self, main_prediction, special_prediction, main_labels, special_labels):
        """
        Recombines a dual-head DL model's separate main/special predictions
        (see LSTM.py/TCN.py/UnifiedLstmTcn.py/UnifiedLstmGruTcn.py create_model)
        back into the single flat (digitsPerDraw, num_classes) array + shared
        labels list that deepLearningMethod's decode (np.argmax + labels[i])
        expects, so nothing downstream of run() needs to know two heads exist.

        Each head's class axis is zero-padded to sit in its own disjoint slice
        of a combined (main_num_classes + special_num_classes)-wide axis
        (main first, special appended) - so per-position argmax still only
        ever picks a class index within that position's own valid range, and
        one shared `combined_labels[class_index]` correctly decodes any
        position regardless of which head it came from.
        """
        main_prediction = np.asarray(main_prediction)
        special_prediction = np.asarray(special_prediction)

        main_num_classes = main_prediction.shape[-1]
        special_num_classes = special_prediction.shape[-1]

        main_padded = np.pad(main_prediction, ((0, 0), (0, special_num_classes)))
        special_padded = np.pad(special_prediction, ((0, 0), (main_num_classes, 0)))

        combined_prediction = np.concatenate([main_padded, special_padded], axis=0)
        combined_labels = [int(v) for v in sorted(main_labels)] + [int(v) for v in sorted(special_labels)]

        return combined_prediction, combined_labels

    def score_numbers_from_prediction(self, raw_predictions, unique_labels):
        """
        Turns a DL model's per-position softmax prediction (shape
        (digitsPerDraw, num_classes)) into a single {number: score} dict, by
        taking - for each number - the highest probability it received across
        every draw position that could produce it. Mirrors what the
        statistical models' own score_numbers() provides, so Keno subset
        generation (generate_subset_from_scores) works the same way for both:
        no DL model tracks per-position "which slot is this" meaning (Keno's
        20 positions are unordered), so max-across-positions is the right
        aggregation rather than e.g. summing (which would double-count a
        number the model is confident about in multiple positions).
        """
        raw_predictions = np.asarray(raw_predictions)
        scores = {}
        for position_probs in raw_predictions:
            for class_index, probability in enumerate(position_probs):
                number = int(unique_labels[class_index])
                scores[number] = max(scores.get(number, 0.0), float(probability))
        return scores

    # Function to print the predicted numbers
    def print_predicted_numbers(self, predicted_numbers):

        #print("Predicted Numbers Shape:", predicted_numbers.shape)
        #print("Predicted Numbers Type:", type(predicted_numbers))
        """
        print("============================================================")
        for i in range(len(predicted_numbers)):
            print(f"Predicted Numbers {i}: {', '.join(map(str, predicted_numbers[i]))}")
        print("============================================================")
        """

        for i, sublist in enumerate(predicted_numbers):
            chart = asciichartpy.plot(sublist, {'height': 10})
            print(f"Graph for Sublist {i+1}:\n{chart}\n")


        
        
    def load_data(self, dataPath, skipLastColumns=0, nth_row=5, maxRows=0, skipRows=0, years_back=None, specialColumnCount=0):
        # Initialize an empty list to hold the data
        data = []

        #print("skip last column: ", skipLastColumns)

        for csvFile in os.listdir(dataPath):
            if csvFile.endswith(".csv"):
                try:
                    # Construct full file path
                    file_path = os.path.join(dataPath, csvFile)

                    # Load data from the file
                    if maxRows > 0:
                        csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1, max_rows=maxRows)
                    else:
                        csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1)

                    if not isinstance(csvData[0], (list, np.ndarray)):
                        print("Need to reform loaded csv data")
                        csvData = [csvData.tolist()]
                        
                    # Skip last number of columns by slicing (if required)
                    if skipLastColumns > 0:
                        csvData = csvData[:, :-skipLastColumns]

                    # Append each entry to the data list
                    for entry in csvData:
                        date_str = entry[0]
                        try:
                            date = parse(date_str)
                        except Exception as e:
                            print(f"Date parsing error for entry '{date_str}': {e}")
                            continue

                        try:
                            numbers = list(map(int, entry[1:]))
                        except ValueError as ve:
                            print(f"Number conversion error for entry '{entry[1:]}': {ve}")
                            continue

                        data.append((date, *numbers))

                except Exception as e:
                    print(f"Error processing file {csvFile}: {e}")

        #print("data: ", data[len(data)-1])

        # Sort the data by date
        data.sort(key=lambda x: x[0], reverse=False)  # Oldest to newest

        # Convert to NumPy array
        sorted_data = np.array(data)

        # Filter data for a relative range of years
        if years_back is not None:
            most_recent_date = sorted_data[-1, 0]  # Most recent date in the array
            cutoff_date = most_recent_date.replace(year=most_recent_date.year - years_back)
            filtered_data = [entry for entry in sorted_data if entry[0] >= cutoff_date]
            sorted_data = np.array(filtered_data)

        # Continue processing
        dates = sorted_data[:, 0]
        numbers = sorted_data[:, 1:].astype(int)

        # Replace all -1 values with 0
        numbers[numbers == -1] = 0

        # Remove the last n elements in case of history building
        if skipRows > 0:
            #print("Skipping Rows: ", skipRows)
            #print("Length of data before skipping rows: ", len(numbers))
            numbers = numbers[:-skipRows]
            #print("Length after skipping rows: ", len(numbers))
            #print("last entry: ", numbers[len(numbers)-1])

        # Isolate the trailing special/bonus column(s) (e.g. Euromillions's 2
        # star columns, EuroDreams's 1 dream number, VikingLotto's 1 super
        # viking) so they can be modeled with their own range, independently
        # from the main numbers. Requires skipLastColumns=0 so they're still
        # present in `numbers`.
        if specialColumnCount > 0:
            numbers = numbers[:, -specialColumnCount:]

        # Unique labels for one-hot encoding
        unique_labels = self.get_unique_labels(dataPath, special=specialColumnCount > 0)

        #print("unique_labels: ", unique_labels)

        encoder = OneHotEncoder(categories=[unique_labels], sparse_output=False)

        # Reshape numbers array to a single column for encoding, then reshape back
        one_hot_labels = encoder.fit_transform(numbers.flatten().reshape(-1, 1))

        one_hot_labels = one_hot_labels.reshape(numbers.shape[0], numbers.shape[1], -1)

        # Number of classes (the unique labels we have after one-hot encoding)
        num_classes = one_hot_labels.shape[2] 

        #print("Num classes: ", num_classes)

        # Prepare training and validation sets
        train_indices = [i for i in range(len(numbers)) if i % nth_row != 0]  # Indices for training data
        val_indices = [i for i in range(len(numbers)) if i % nth_row == 0]    # Indices for validation data

        train_data = numbers[train_indices]
        val_data = numbers[val_indices]

        #print("length of train data: ", len(train_data))
        #print("length of val_data: ", len(val_data))

        train_labels = one_hot_labels[train_indices]
        val_labels = one_hot_labels[val_indices]

        #print("Train data shape: ", train_data.shape)       # (samples, sequence_length) -> (n_samples, 7)
        #print("Train labels shape: ", train_labels.shape)   # (samples, sequence

        #print("Train Labels Example:", train_labels[:5])  # Corresponding one-hot encoded labels

        # Get the maximum value in the data (for scaling purposes, if needed)
        max_value = np.max(numbers)

        #print("Length of data: ", len(numbers))
        #print("Numbers: ", numbers)
        

        return train_data, val_data, max_value, train_labels, val_labels, numbers, num_classes, unique_labels



    def load_prediction_data(self, dataPath, skipLastColumns=0, maxRows=0):
        # Initialize an empty list to hold the data
        data = []

        for csvFile in os.listdir(dataPath):
            if csvFile.endswith(".csv"):
                try:
                    # Construct full file path
                    file_path = os.path.join(dataPath, csvFile)

                    # Load data from the file
                    if maxRows > 0:
                        csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1, max_rows=maxRows)
                    else:
                        csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1)

                
                    if not isinstance(csvData[0], (list, np.ndarray)):
                        print("Need to reform loaded csv data")
                        csvData = [csvData.tolist()]
                        
                    
                    # Skip last number of columns by slicing (if required)
                    if skipLastColumns > 0:
                        csvData = csvData[:, :-skipLastColumns]

                    #print("csv data: ", csvData)

                    # Append each entry to the data list
                    for entry in csvData:
                        # Attempt to parse the date
                        date_str = entry[0]
                        #print("Date: ", date_str)
                        try:
                            # Use dateutil.parser to parse the date
                            date = parse(date_str)
                        except Exception as e:
                            print(f"Date parsing error for entry '{date_str}': {e}")
                            continue  # Skip this entry if date parsing fails

                        # Convert the rest to integers
                        try:
                            numbers = list(map(int, entry[1:]))  # Convert the rest to integers
                        except ValueError as ve:
                            print(f"Number conversion error for entry '{entry[1:]}': {ve}")
                            continue  # Skip this entry if number conversion fails

                        data.append((date, *numbers))  # Store as a tuple (date, number1, number2, ...)

                except Exception as e:
                    print(f"Error processing file {csvFile}: {e}")

        # Sort the data by date
        data.sort(key=lambda x: x[0], reverse=False)  # Sort by the date (first element of the tuple)

        #print("Data: ", data)

        # Convert the sorted data into a NumPy array
        sorted_data = np.array(data)

        #print("Sorted data: ", sorted_data)

        # If you want to separate the date and numbers into different arrays
        dates = sorted_data[:, 0]  # Dates
        numbers = sorted_data[:, 1:].astype(int)  # Numbers as integers (multi-label data)

        # Replace all -1 values with 0 (or you can remove them if it's not needed)
        numbers[numbers == -1] = 0

        #print("length of data: ", len(numbers))
        #print("shape of data: ", numbers.shape)


        return numbers

    def create_sequences(self, data, window_size=10):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])       # window of draws
            y.append(data[i + window_size])         # next draw
        return np.array(X), np.array(y)

    
    def generatePredictionTextFile(self, path):
        print("Generating text file with latest predictions")
        latestPredictionFile = os.path.join(os.getcwd(), "latestPrediction.txt")

        if os.path.exists(latestPredictionFile):
            os.remove(latestPredictionFile)

        for folder in os.listdir(path):
            #print(folder)
            folder_path = os.path.join(path, folder)
            # Get all JSON files in the folder
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            
            # Parse dates from the filenames and find the latest
            latest_file = None
            latest_date = None
            
            for file in files:
                try:
                    # Extract the date from the filename and parse it
                    date_part = file.split('.')[0]  # Assuming format: YYYY-MM-DD.json
                    file_date = datetime.strptime(date_part, "%Y-%m-%d")
                    
                    # Update the latest file if this one is more recent
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = file
                except ValueError:
                    # Ignore files with invalid date formats
                    continue

            if latest_file is not None:
                # Opening JSON file
                with open(os.path.join(path, folder, latest_file), 'r') as openfile:
                
                    # Reading from json file
                    predictionFObject = json.load(openfile)

                with open(latestPredictionFile, "a+") as myfile:
                    myfile.write("{}:\n".format(folder))
                    myfile.write("{}\n".format(predictionFObject["newPrediction"]))
                    myfile.write("\n")
                    myfile.write("\n")

    
    def git_push(self, commit_message="saving last predictions"):
        try:
            # Stage all changes
            subprocess.run(["git", "add", "-A"], check=True)

            # Commit changes
            subprocess.run(["git", "commit", "-m", f"{commit_message}"], check=True)

            # Push changes
            subprocess.run(["git", "push"], check=True)

            print("Changes have been pushed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing Git commands: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def git_pull(self):
        try:
            subprocess.run(["git", "fetch"], check=True)

            subprocess.run(["git", "pull"], check=True)

            print("Got latest changes")
        except Exception as e:
            print("Failed to get latest changes")

    def extractFeaturesFromJsonForRefinement(self, jsonFileOrDir, num_classes=80, numbersLength=20):
        """
            Function to extract features for training a refinement model
            Can be a folder containing json files or a single file
        """
        X = []
        y = []
        if os.path.isdir(jsonFileOrDir):
            for file in sorted(os.listdir(jsonFileOrDir)):
                if file.endswith(".json"):
                    with open(os.path.join(jsonFileOrDir, file), "r") as f:
                        data = json.load(f)

                    if "currentPredictionRaw" not in data or not data["currentPredictionRaw"]:
                        print(f"Skipping {file} - No valid 'currentPredictionRaw' data.")
                        continue
                    
                    raw_probs = np.array(data["currentPredictionRaw"])
                    
                    """
                    # Ensure correct shape before reshaping
                    if raw_probs.size != numbersLength * num_classes:
                        print(f"Skipping {file} - Unexpected shape {raw_probs.shape}")
                        continue
                    """
                    
                    raw_probs = raw_probs.reshape(numbersLength, num_classes)

                    if "realResult" not in data or not data["realResult"]:
                        print(f"Skipping {file} - No valid 'realResult' data.")
                        continue

                    actual_result = data["realResult"]
                    real_result_vector = np.zeros((numbersLength, num_classes))

                    for i, num in enumerate(actual_result):
                        if 1 <= num <= num_classes:  # Ensure number is within valid range
                            real_result_vector[i, num - 1] = 1  # One-hot encode
                    
                    X.append(raw_probs)
                    y.append(real_result_vector)


        elif os.path.isfile(jsonFileOrDir):
            if jsonFileOrDir.endswith(".json"):
                with open(os.path.join(jsonFileOrDir), "r") as f:
                    data = json.load(f)

                if "currentPredictionRaw" not in data or not data["currentPredictionRaw"]:
                    print(f"Skipping {jsonFileOrDir} - No valid 'currentPredictionRaw' data.")
                    return
                
                raw_probs = np.array(data["currentPredictionRaw"])
                
                # Ensure correct shape before reshaping
                """
                if raw_probs.size != numbersLength * num_classes:
                    print(f"Skipping {jsonFileOrDir} - Unexpected shape {raw_probs.shape}")
                    return
                """
                
                raw_probs = raw_probs.reshape(numbersLength, num_classes)

                if "realResult" not in data or not data["realResult"]:
                    print(f"Skipping {jsonFileOrDir} - No valid 'realResult' data.")
                    return

                actual_result = data["realResult"]
                real_result_vector = np.zeros((numbersLength, num_classes))

                for i, num in enumerate(actual_result):
                    if 1 <= num <= num_classes:  # Ensure number is within valid range
                        real_result_vector[i, num - 1] = 1  # One-hot encode
                
                X.append(raw_probs)
                y.append(real_result_vector)

        return np.array(X), np.array(y)  # Both X and y now have compatible shapes
    
    def extractFeaturesFromJsonForDetermineTopPrediction(self, jsonFileOrDir, num_classes=80, numbersLength=20):
        """
            Function to extract features for training a refinement model
            Can be a folder containing json files or a single file
        """
        X = []
        y = []
        if os.path.isdir(jsonFileOrDir):
            for file in sorted(os.listdir(jsonFileOrDir)):
                if file.endswith(".json"):
                    with open(os.path.join(jsonFileOrDir, file), "r") as f:
                        data = json.load(f)

                    if "currentPredictionRaw" not in data:
                        print(f"⚠ Warning: No 'currentPredictionRaw' in {file}")
                        continue
                    
                    raw_probs = np.array(data["currentPredictionRaw"])
                    
                    if raw_probs.size == 0:
                        print(f"⚠ Warning: Empty probability array in {file}")
                        continue

                    # Debug: Print shape of `raw_probs`
                    #print(f"Processing {file}, shape: {raw_probs.shape}")

                    # Ensure raw_probs has expected shape (numbersLength, num_classes)
                    if raw_probs.shape[0] != numbersLength or raw_probs.shape[1] != num_classes:
                        print(f"⚠ Unexpected shape {raw_probs.shape} in {file}")
                        continue

                    # Feature Extraction
                    mean_probs = np.mean(raw_probs, axis=0)  # Average probability per number
                    max_probs = np.max(raw_probs, axis=0)    # Maximum probability per number
                    sum_probs = np.sum(raw_probs, axis=0)    # Sum of probabilities per number

                    # Combine features
                    features = np.concatenate([mean_probs, max_probs, sum_probs])

                    # Ensure actual result exists
                    if "realResult" not in data or len(data["realResult"]) == 0:
                        print(f"⚠ Warning: No realResult in {file}")
                        continue
                    
                    actual_result = data["realResult"]  # This is a list of actual drawn numbers

                    # Convert realResult into a one-hot encoded vector (shape: (num_classes,))
                    real_result_vector = np.zeros(num_classes)  # num_classes possible numbers
                    for num in actual_result:
                        real_result_vector[num - 1] = 1  # Convert numbers (1 - num_classes) to index (0 - (num_classes-1))

                    X.append(features)
                    y.append(real_result_vector)  # Now y is a probability-like distribution
        elif os.path.isfile(jsonFileOrDir):
            if jsonFileOrDir.endswith(".json"):
                with open(os.path.join(jsonFileOrDir), "r") as f:
                    data = json.load(f)

                if "currentPredictionRaw" not in data:
                    print(f"⚠ Warning: No 'currentPredictionRaw' in {jsonFileOrDir}")

                
                raw_probs = np.array(data["currentPredictionRaw"])
                
                if raw_probs.size == 0:
                    print(f"⚠ Warning: Empty probability array in {jsonFileOrDir}")


                # Debug: Print shape of `raw_probs`
                #print(f"Processing {jsonFileOrDir}, shape: {raw_probs.shape}")

                # Ensure raw_probs has expected shape (numbersLength, num_classes)
                if raw_probs.shape[0] != numbersLength or raw_probs.shape[1] != num_classes:
                    print(f"⚠ Unexpected shape {raw_probs.shape} in {jsonFileOrDir}")


                # Feature Extraction
                mean_probs = np.mean(raw_probs, axis=0)  # Average probability per number
                max_probs = np.max(raw_probs, axis=0)    # Maximum probability per number
                sum_probs = np.sum(raw_probs, axis=0)    # Sum of probabilities per number

                # Combine features
                features = np.concatenate([mean_probs, max_probs, sum_probs])

                # Ensure actual result exists
                if "realResult" not in data or len(data["realResult"]) == 0:
                    print(f"⚠ Warning: No realResult in {jsonFileOrDir}")
                
                actual_result = data["realResult"]  # This is a list of actual drawn numbers

                # Convert realResult into a one-hot encoded vector (shape: (num_classes,))
                real_result_vector = np.zeros(num_classes)  # num_classes possible numbers
                for num in actual_result:
                    real_result_vector[num - 1] = 1  # Convert numbers (1 - num_classes) to index (0 - (num_classes-1))

                X.append(features)
                y.append(real_result_vector)  # Now y is a probability-like distribution

        return np.array(X), np.array(y)  # Both X and y now have compatible shapes
    
    def getTopPredictions(self, predictions, labels, num_top=20):
        """
        Extracts the top N most probable numbers from the model output.

        :param predictions: Model output (probability distribution of shape (batch_size, 80)).
        :param labels: The corresponding numbers (1-80 for Keno).
        :param num_top: Number of top predictions to extract.
        :return: List of top N numbers for each prediction.
        """
        top_numbers = []
        
        for prediction in predictions:
            # Get indices of top N probabilities
            top_indices = np.argsort(prediction)[-num_top:]  # Indices of top 20 numbers
            top_numbers.append([labels[i] for i in top_indices])  # Convert indices to numbers

        return top_numbers


    def count_number_frequencies(self, dataPath):
        """
        Count the frequency of each number in all CSV files within the specified folder and normalize the frequencies.

        Parameters
        ----------
        dataPath : str
            Path to the folder containing CSV files with historical data.

        Returns
        -------
        dict
            A dictionary where keys are numbers and values are their normalized frequencies.
        """
        # Initialize a dictionary to store number frequencies
        number_frequencies = {}

        # Iterate over all CSV files in the folder
        for csvFile in os.listdir(dataPath):
            if csvFile.endswith(".csv"):
                try:
                    # Construct full file path
                    file_path = os.path.join(dataPath, csvFile)

                    # Load data from the file
                    csvData = np.genfromtxt(file_path, delimiter=';', dtype=str, skip_header=1)

                    # Ensure the data is in the correct format
                    if not isinstance(csvData[0], (list, np.ndarray)):
                        csvData = [csvData.tolist()]

                    # Process each entry in the CSV data
                    for entry in csvData:
                        # Skip the date and convert the rest to integers
                        try:
                            numbers = list(map(int, entry[1:]))
                        except ValueError as ve:
                            print(f"Number conversion error for entry '{entry[1:]}': {ve}")
                            continue

                        # Update the frequency count for each number
                        for number in numbers:
                            if number in number_frequencies:
                                number_frequencies[number] += 1
                            else:
                                number_frequencies[number] = 1

                except Exception as e:
                    print(f"Error processing file {csvFile}: {e}")

        # Normalize the frequencies
        total_counts = sum(number_frequencies.values())
        normalized_frequencies = {number: count / total_counts for number, count in number_frequencies.items()}

        return normalized_frequencies


    def count_number_frequencies_from_new_prediction(self, json_data, model_scores=None):
        """
        Count normalized frequencies of numbers in 'newPrediction' field from the given JSON structure.

        Parameters
        ----------
        json_data : dict
            Dictionary parsed from a JSON containing 'newPrediction' with model predictions.
        model_scores : dict, optional
            Maps a model's "name" (as used in each newPrediction entry) to its
            own Hyperopt/Backtester score (e.g. hits_avg or profit_total), so
            better-performing models get more weight in the combined vote
            instead of every model counting equally. This only affects the
            combined numberFrequency view - it does not filter, reorder, or
            otherwise change any individual model's own prediction. Missing or
            absent scores (e.g. LSTM/xgboost, or when model_scores is None)
            fall back to a neutral weight of 1, i.e. today's unweighted count.

        Returns
        -------
        dict
            A dictionary where keys are numbers and values are their normalized weighted frequencies.
        """
        model_scores = model_scores or {}
        weight_for = self._build_model_weight_lookup(model_scores)

        number_frequencies = {}

        # Iterate through each model's predictions in 'newPrediction'
        for model in json_data.get("newPrediction", []):
            weight = weight_for(model.get("name"))
            predictions = model.get("predictions", [])
            for pred_set in predictions:
                for number in pred_set:
                    number_frequencies[number] = number_frequencies.get(number, 0) + weight

        # Normalize frequencies
        total_counts = sum(number_frequencies.values())
        normalized_frequencies = {
            number: count / total_counts for number, count in number_frequencies.items()
        }

        return normalized_frequencies


    def _build_model_weight_lookup(self, model_scores):
        """
        Shared by count_number_frequencies_from_new_prediction and
        count_number_frequencies_by_position: min-max scales scores to a
        [1, 2] weight range so every model still contributes at least the old
        neutral weight of 1 - a poorly scoring model isn't zeroed out, just
        outweighted by better ones. Unscored models (not in model_scores) get
        the same neutral weight of 1.
        """
        scored_values = list(model_scores.values())
        score_min = min(scored_values) if scored_values else 0
        score_max = max(scored_values) if scored_values else 0
        score_range = score_max - score_min

        def weight_for(model_name):
            score = model_scores.get(model_name)
            if score is None or score_range == 0:
                return 1.0
            return 1.0 + (score - score_min) / score_range

        return weight_for


    def count_number_frequencies_by_position(self, json_data, main_count, model_scores=None):
        """
        Like count_number_frequencies_from_new_prediction, but splits each
        model's main prediction row (predictions[0]) into its first
        main_count numbers vs whatever comes after (a game's special
        column(s) - Euromillions star numbers, EuroDreams dream number,
        VikingLotto super viking - see Predictor.py's SPECIAL_COLUMN_COUNTS),
        counting weighted votes separately for each range. Without this, a
        combined ensemble ticket built from one flat vote lets main-range
        numbers crowd out the special slot(s), producing an out-of-range
        special number - unlike every individual model, which already keeps
        the two separate via Helpers.run_model_with_special_column.

        Subset rows (predictions[1:], only ever populated for Keno, which has
        no special columns) are ignored here - only the main ticket carries
        positional meaning.

        Returns
        -------
        (main_frequencies, special_frequencies) : tuple of dict
            special_frequencies is empty if main_count covers the whole row
            (no special columns for this game).
        """
        model_scores = model_scores or {}
        weight_for = self._build_model_weight_lookup(model_scores)

        main_frequencies = {}
        special_frequencies = {}

        for model in json_data.get("newPrediction", []):
            predictions = model.get("predictions", [])
            if not predictions:
                continue

            weight = weight_for(model.get("name"))
            main_row = predictions[0][:main_count]
            special_row = predictions[0][main_count:]

            for number in main_row:
                main_frequencies[number] = main_frequencies.get(number, 0) + weight
            for number in special_row:
                special_frequencies[number] = special_frequencies.get(number, 0) + weight

        return main_frequencies, special_frequencies


    def build_weighted_ensemble_prediction(self, number_frequencies, ticket_size, sorted_prediction=True, name="WeightedEnsemble Model"):
        """
        Turns the (optionally score-weighted) numberFrequency vote into an
        actual ticket - the ticket_size numbers with the highest combined
        vote - so it can be shown as its own row in the Model table
        (newPrediction), next to every individual model's own prediction,
        instead of only existing as a separate chart.

        Note: this only makes sense for non-positional games. Pick3 predicts
        digits in drawn order, and a frequency vote across models has no
        notion of position, so callers should skip this for Pick3.

        Parameters
        ----------
        number_frequencies : dict
            Output of count_number_frequencies_from_new_prediction.
        ticket_size : int
            How many numbers the ticket should contain (e.g. the draw size).
        sorted_prediction : bool
            Whether to sort the resulting ticket ascending (matches how the
            other non-positional models format their predictions).

        Returns
        -------
        dict or None
            {"name": name, "predictions": [[...]]}, or None if there isn't
            enough data to build a ticket_size-length ticket.
        """
        if not number_frequencies or ticket_size <= 0:
            return None

        top_numbers = sorted(number_frequencies, key=number_frequencies.get, reverse=True)[:ticket_size]

        if len(top_numbers) < ticket_size:
            return None

        top_numbers = [int(n) for n in top_numbers]
        if sorted_prediction:
            top_numbers = sorted(top_numbers)

        return {"name": name, "predictions": [top_numbers]}


    def generate_subset_from_scores(self, number_scores, ticket_numbers, subset_size, mode="softmax", temperature=0.5):
        """
        Shared subset generator: picks subset_size numbers out of a scored
        ticket, for callers (WeightedEnsemble Model, MetaLearner Model, or any
        future model/method) that already have a {number: score} dict rather
        than the raw vote/probability arrays each individual model's own
        generate_best_subset works from. Currently only used for Keno, the
        only game with sub-selections (5-10 out of 20).

        Parameters
        ----------
        number_scores : dict
            {number: score}, higher = more likely. Only numbers already in
            ticket_numbers are considered (a subset is a subset of the full
            ticket, not a re-ranking over the whole game range).
        ticket_numbers : list
            The full ticket (e.g. the 20 Keno numbers) to select from.
        subset_size : int
        mode : str
            "top" - deterministic top-N by score (always the same numbers for
            a given score dict - maximizes expected hits, zero diversity).
            "softmax" (default) - probability-weighted sample without
            replacement using softmax(score/temperature) as weights, the same
            idea as Markov's own softmax-temperature subset selection: still
            favors higher-scored numbers but keeps some variation between
            runs instead of always producing the identical subset.
        temperature : float
            Only used for mode="softmax". Lower = closer to deterministic
            top-N; higher = closer to a uniform random subset.

        Returns
        -------
        list
            sorted list of subset_size ints, or the full ticket_numbers list
            if there aren't enough numbers to form a subset.
        """
        candidates = [int(n) for n in ticket_numbers if n in number_scores]
        if len(candidates) < subset_size:
            candidates = [int(n) for n in ticket_numbers]
        if len(candidates) <= subset_size:
            return sorted(candidates)

        if mode == "top":
            chosen = sorted(candidates, key=lambda n: number_scores.get(n, 0), reverse=True)[:subset_size]
            return sorted(int(n) for n in chosen)

        scores = np.array([number_scores.get(n, 0) for n in candidates], dtype=float)
        temperature = max(temperature, 1e-3)
        probabilities = scipy.special.softmax(scores / temperature)
        chosen = np.random.choice(candidates, size=subset_size, replace=False, p=probabilities)
        return sorted(int(n) for n in chosen)


    def calculate_profit(self, name, path):
        """
        Used for Hyperopt to calculate the profit of a given model.
        """

        json_dir = os.path.join(path, "data", "hyperOptCache", name)
        if not os.path.exists(json_dir):
            print(f"Directory does not exist: {json_dir}")
            return

        total_profit = 0

        for filename in os.listdir(json_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(json_dir, filename)
                with open(filepath, "r") as file:
                    data = json.load(file)

                    real_result = data.get("realResult", [])
                    real_result_set = set(real_result)
                    model_predictions = data.get("currentPrediction", [])

                    # We need to calculate the lost even at winning. Because the ticket costs needs to be deducted

                    for model in model_predictions:
                        for prediction in model["predictions"]:
                            # For keno and pick3 the profits can be calculated. For others we check the matches
                            if "keno" in name:
                                profit = self.keno_ticket_profit(prediction, real_result_set)
                                if profit is not None:
                                    total_profit += profit
                            elif "pick3" in name:
                                profit = self.pick3_ticket_profit(prediction, real_result)
                                if profit is not None:
                                    total_profit += profit
                            else:
                                matches = len(set(prediction) & real_result_set)
                                total_profit += matches
        #print("Total profit: ", total_profit)
        return total_profit
    
    def cleanupDatabase(self, storage_url):

        # Connect to the existing storage
        storage = optuna.storages.RDBStorage(url=storage_url)

        # Load all studies
        studies = optuna.get_all_study_summaries(storage=storage)

        if not studies:
            print("No studies found. Nothing to delete.")
            return

        print(f"Found {len(studies)} studies. Deleting all...")

        # Delete all studies
        for s in studies:
            print(f"Deleting study: {s.study_name}")
            optuna.delete_study(study_name=s.study_name, storage=storage)

        print("All studies deleted successfully.")

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')