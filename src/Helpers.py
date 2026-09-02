import os, json, math, subprocess, optuna, argparse
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

    # Official Belgian Pick-3 payouts per 1 EUR bet (Reglement Pick-3, juli
    # 2024, Art. 6/7/14). The tracked ticket plays every bet type at once
    # (straight + box + front pair + back pair, 1 EUR each) and the bets are
    # evaluated independently, so prizes cumulate - e.g. an exact match also
    # collects the box and both pair prizes on top of the straight prize.
    PAYOUT_TABLE_PICK3 = {
        "straight": 500,             # all 3 digits in exact order
        "straight_consolation": 1,   # straight bet's consolation: units digit (position 3) matches
        "box_with_doubles": 160,     # any order, prediction contains a doubled digit (3 ways to win)
        "box_no_doubles": 80,        # any order, 3 distinct digits (6 ways to win)
        "front_pair": 50,            # positions 1-2 in exact order
        "back_pair": 50,             # positions 2-3 in exact order
        "bet_cost": 1,               # stake per bet type
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

    # Mirror of Predictor.py's / HyperoptDeepLearning.py's SPECIAL_COLUMN_COUNTS:
    # how many trailing values of a row belong to the special pool. Needed here
    # because scoring functions below (performance report, lag analysis,
    # randomness watch, hyperopt profit) must never pool special hits with main
    # hits - the specials are drawn from their own, much smaller range, so a
    # pooled set intersection systematically inflates hit counts.
    SPECIAL_COLUMN_COUNTS = {"euromillions": 2, "eurodreams": 1, "vikinglotto": 1}

    def main_special_split(self, game, real_result):
        """
        (realMainCount, specialColumnCount) for a game name/folder: how many
        leading values of real_result count as the main-pool hit targets, and
        how many trailing special columns the game models. Lotto follows the
        real game's tier rules: a play is 6 numbers compared against the 6
        drawn mains, and the 7th (bonus) value only ever supplements a
        partial match ("5 (1)" is a high tier, "6 (0)" the jackpot) - so
        lotto's realMainCount is 6 and the bonus is handled as a separate
        supplement pool by the consumers, never as a 7th main
        ("vikinglotto" contains "lotto" but its viking is a dedicated
        special column, hence the exclusion).
        """
        specialColumnCount = next(
            (count for g, count in self.SPECIAL_COLUMN_COUNTS.items() if g in game), 0)
        if "lotto" in game and "vikinglotto" not in game:
            return len(real_result) - 1, 0
        return len(real_result) - specialColumnCount, specialColumnCount

    def split_ticket(self, ticket, realMainCount, specialColumnCount):
        """
        (mains, specials) of one prediction row. A row longer than the real
        main count carries its specials appended at the end (the normal
        statistical/DL/boosting rows); a row of main-count length or shorter
        is mains-only (RL Ticket Model rows, Keno subset tickets) and has no
        specials to slice off.
        """
        if specialColumnCount > 0 and len(ticket) > realMainCount:
            return ticket[:-specialColumnCount], ticket[-specialColumnCount:]
        return list(ticket), []

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
        Net profit (in euro) for a single Keno subset ticket: official payout
        minus the 1 EUR stake on a win, -1 on a loss. Net convention matches
        pick3_ticket_profit so profit_per_bet is comparable across games (the
        old version returned the gross payout on wins and double-counted the
        stake on losses).
        Only subsets of 5-10 numbers have real payouts; below that, profit is not tracked.
        """
        played = len(prediction)
        if played < 5 or played > 10:
            return None

        table = self.PAYOUT_TABLE_KENO
        # The table's payout entries are gross winnings for a 1 EUR ticket;
        # "lost" is already the NET result of that losing ticket (-stake).
        stake = -table["lost"]
        matches = len(set(map(int, prediction)) & set(map(int, real_result)))
        payout = table.get(played, {}).get(matches)

        if payout is None:
            return table["lost"]
        return payout - stake

    def pick3_ticket_profit(self, prediction, real_result):
        """
        Net profit (in euro) for a single Pick3 ticket under the official
        cumulative model (Reglement Pick-3, juli 2024, Art. 6/7/14): the
        tracked ticket plays all bet types at 1 EUR each and each bet is
        evaluated independently, so prizes cumulate. Order matters
        (straight/pair payouts). A triple prediction [x,x,x] cannot play box
        (it has no distinct orderings), so its stake is 3 instead of 4.
        """
        if len(prediction) != 3 or len(real_result) != 3:
            return None

        table = self.PAYOUT_TABLE_PICK3
        pred = [int(p) for p in prediction]
        actual = [int(a) for a in real_result]
        is_triple = len(set(pred)) == 1

        stake = (3 if is_triple else 4) * table["bet_cost"]
        payout = 0

        # Straight bet: exact order wins the full prize; otherwise a matching
        # units digit (position 3) alone pays the 1 EUR consolation.
        if pred == actual:
            payout += table["straight"]
        elif pred[2] == actual[2]:
            payout += table["straight_consolation"]

        # Box bet (non-triples only): sorted equality. It is an independent
        # bet, so it also pays on an exact-order match - no elif with straight.
        if not is_triple and sorted(pred) == sorted(actual):
            if len(set(pred)) == 2:
                payout += table["box_with_doubles"]
            else:
                payout += table["box_no_doubles"]

        # Pair bets: two positions in exact order, each its own 1 EUR bet.
        if pred[0:2] == actual[0:2]:
            payout += table["front_pair"]
        if pred[1:3] == actual[1:3]:
            payout += table["back_pair"]

        return payout - stake

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
        - every other game: average MAIN-number hits of the model's main
          ticket. avg_hits/best_hits/hits_total count main hits only - the
          special columns (stars/dream/viking) live in their own smaller range
          and lotto's bonus isn't played at all, so pooling them inflated the
          averages. Games with special columns additionally get a per-model
          "avg_special_hits" (None for the other games).

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
            # Games with a modeled special pool get their own special-hit
            # average; for every other game the field stays None.
            gameSpecialCount = next(
                (count for g, count in self.SPECIAL_COLUMN_COUNTS.items() if g in game), 0)
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
                        "special_hits_total": 0,
                        "profit_total": 0.0, "bets": 0,
                    })

                    mainTicket = predictions[0]
                    # Main hits only: slicing both ticket and realResult keeps
                    # star/dream/viking hits from inflating the average -
                    # those come from a much smaller special range and aren't
                    # comparable. Lotto's bonus follows the real tier rules:
                    # it supplements a partial match ("5 (1)"), so it counts
                    # into special_hits by matching the ticket itself, never
                    # into the main average.
                    realMainCount, specialCount = self.main_special_split(game, realResult)
                    ticketMains, ticketSpecials = self.split_ticket(mainTicket, realMainCount, specialCount)
                    ticketMainSet = set(map(int, ticketMains))
                    hits = len(ticketMainSet & set(map(int, realResult[:realMainCount])))
                    entry["draws"] += 1
                    entry["hits_total"] += hits
                    entry["best_hits"] = max(entry["best_hits"], hits)
                    if specialCount > 0:
                        realSpecials = set(map(int, realResult[realMainCount:realMainCount + specialCount]))
                        entry["special_hits_total"] += len(set(map(int, ticketSpecials)) & realSpecials)
                    realBonus = set(map(int, realResult[realMainCount + specialCount:]))
                    if realBonus:
                        entry["special_hits_total"] += len(ticketMainSet & realBonus)

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
                avgSpecialHits = None
                # Lotto counts here too: its bonus supplements land in
                # special_hits_total, so the "(M)" average exists for it.
                hasSupplement = gameSpecialCount > 0 or ("lotto" in game and "vikinglotto" not in game)
                if hasSupplement and entry["draws"]:
                    avgSpecialHits = round(entry["special_hits_total"] / entry["draws"], 3)
                models.append({
                    "name": name,
                    "draws": entry["draws"],
                    "avg_hits": round(avgHits, 3),
                    "avg_special_hits": avgSpecialHits,
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
        #
        # Only the single best peak of this run is kept per model - the full
        # 30-column table was mostly noise, and one run's table says nothing
        # about whether a lag is real. Instead each run's peak is appended to
        # a persisted history (lagPeakHistory.json) keyed by game/model, so a
        # lag that keeps winning across runs (e.g. pick3 peaking near 30 on
        # run after run) becomes visible as a recurring peak, while a peak
        # that wanders every run is exposed as noise.
        # ------------------------------------------------------------------
        MAX_LAG = 30
        HISTORY_RUNS = 60  # keep roughly two months of daily runs per model

        historyPath = os.path.join(databaseDir, "lagPeakHistory.json")
        try:
            with open(historyPath, "r") as infile:
                peakHistory = json.load(infile)
        except Exception:
            peakHistory = {}

        # One-time self-migration: run entries persisted before the
        # mains/specials split measured POOLED hits (special columns and
        # lotto's bonus included), an incompatible metric - mixing them into
        # the consensus trails would poison every lag vote with numbers
        # counted under different rules. Entries carrying the "main_hits"
        # marker written below are the only comparable ones; everything else
        # is dropped.
        for gameHistory in peakHistory.values():
            for modelName in list(gameHistory.keys()):
                gameHistory[modelName] = [
                    r for r in gameHistory[modelName] if r.get("metric") == "main_hits"]
                if not gameHistory[modelName]:
                    del gameHistory[modelName]
        runDate = report["generatedAt"][:10]

        for game in list(report["games"].keys()):
            gameDir = os.path.join(databaseDir, game)
            isPick3 = "pick3" in game

            def dayHits(prediction, realResult):
                if isPick3:
                    return sum(1 for p, r in zip(prediction, realResult) if int(p) == int(r))
                # Main columns only, both sides: a star/dream/viking (or lotto
                # bonus) "hit" comes from a different number range and would
                # blur the lag signal the analysis is hunting for.
                realMainCount, specialCount = self.main_special_split(game, realResult)
                mains, _ = self.split_ticket(prediction, realMainCount, specialCount)
                return len(set(map(int, mains)) & set(map(int, realResult[:realMainCount])))

            # Chronologically ordered (date, realResult, {model: mainTicket})
            days = []
            latestAnomaly = None  # newest day's autoencoder anomalyWatch
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
                if dayData.get("anomalyWatch") and (latestAnomaly is None or fileDate > latestAnomaly[0]):
                    latestAnomaly = (fileDate, dayData["anomalyWatch"])
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
            gameHistory = peakHistory.setdefault(game, {})

            for name, perLag in lagStats.items():
                lags = []
                for lag in range(1, MAX_LAG + 1):
                    total, count = perLag.get(lag, (0, 0))
                    lags.append({"lag": lag, "avg_hits": round(total / count, 3) if count else None, "n": count})

                scored = [l for l in lags if l["avg_hits"] is not None and l["n"] >= 10]
                if not scored:
                    continue

                peak = max(scored, key=lambda l: l["avg_hits"])
                values = [l["avg_hits"] for l in scored]
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = variance ** 0.5
                # How far the peak sticks out of its own lag profile. A flat
                # profile (no timing information) gives z ~ 1 whatever lag
                # happens to win; a genuinely shifted signal gives a large z.
                zScore = round((peak["avg_hits"] - mean) / std, 2) if std > 0 else None

                thisRun = {
                    "run": runDate,
                    "lag": peak["lag"],
                    "avg_hits": peak["avg_hits"],
                    "n": peak["n"],
                    "z": zScore,
                    "profile_mean": round(mean, 3),
                    # Marks which counting rule produced this entry - the
                    # migration on load drops anything without it.
                    "metric": "main_hits",
                }

                # One entry per run date: a second run on the same day replaces
                # the first (it scored strictly more history) instead of
                # double-counting that day's vote.
                runs = [r for r in gameHistory.get(name, []) if r.get("run") != runDate]
                # Only record a run whose peak sample size differs from the
                # last recorded entry: consecutive daily runs re-scan 97-100%
                # identical history (measured over 5 runs), so an unchanged n
                # means the run added zero new draws for this model - counting
                # it as a fresh "vote" would certify stable noise as consensus.
                if not runs or runs[-1].get("n") != thisRun["n"]:
                    runs.append(thisRun)
                runs.sort(key=lambda r: r.get("run") or "")
                gameHistory[name] = runs[-HISTORY_RUNS:]

                lagCounts = {}
                for r in gameHistory[name]:
                    key = str(r.get("lag"))
                    lagCounts[key] = lagCounts.get(key, 0) + 1
                consensusLag, consensusRuns = max(
                    lagCounts.items(), key=lambda kv: (kv[1], -abs(int(kv[0]) - peak["lag"]))
                )

                lagAnalysis[name] = {
                    "peak": thisRun,
                    "history": gameHistory[name],
                    "runs": len(gameHistory[name]),
                    "lag_counts": lagCounts,
                    # The lag that has peaked most often across runs, and how
                    # many of the kept runs voted for it - that ratio, not any
                    # single run, is the evidence a shift is real.
                    "consensus_lag": int(consensusLag),
                    "consensus_runs": consensusRuns,
                    "best_lag": peak["lag"],
                }

            report["games"][game]["lagAnalysis"] = lagAnalysis

            # Autoencoder security layer (README "Unsupervised Anomaly
            # Detection"): surface the newest day's reconstruction-NLL watch.
            # Summary only - the full score series stays in that day's json.
            if latestAnomaly is not None:
                anomalyDate, watch = latestAnomaly
                report["games"][game]["anomalyWatch"] = {
                    "date": anomalyDate.strftime("%Y-%m-%d"),
                    "latest_z": watch.get("latest_z"),
                    "min_z_recent": watch.get("min_z_recent"),
                    "alert": bool(watch.get("alert")),
                }

            # --------------------------------------------------------------
            # Randomness watch (README "Entropy & Divergence Analysis"): the
            # drawing process itself should be stationary and near-uniform.
            # Monitored per game over the scored history collected above:
            # - KL(recent window || full history): a drift in which numbers
            #   come up. Near 0 = stationary.
            # - KL(recent window || uniform) and normalized entropy: how far
            #   the recent process is from a fair draw. Entropy near 1 =
            #   healthy randomness; a sustained drop = structure appearing.
            # - a trend series (checkpoints every 10 draws) so the UI can
            #   show movement instead of a single point.
            # - per model: KL(predicted numbers || real numbers) over the
            #   same recent days - a model whose output distribution drifts
            #   far from the real process is betting on structure that is
            #   not there (or has found some).
            # Pick3 is positional: distributions are computed per digit
            # position (10 classes each) and averaged - pooling positions
            # would mask a single-position anomaly.
            # All distributions use add-0.5 smoothing so unseen numbers do
            # not blow KL up to infinity on small windows.
            # --------------------------------------------------------------
            WINDOW = 60
            MIN_WINDOW = 30

            def drawDistributions(results):
                """List of per-position (pick3) or single pooled count dicts."""
                if isPick3:
                    dists = [{} for _ in range(3)]
                    for res in results:
                        for pos, v in enumerate(res[:3]):
                            dists[pos][int(v)] = dists[pos].get(int(v), 0) + 1
                    return dists
                dist = {}
                for res in results:
                    for v in res:
                        dist[int(v)] = dist.get(int(v), 0) + 1
                return [dist]

            def klAndEntropy(recentDists, baseDists, labelSets):
                """Mean over positions of (KL(recent||base), KL(recent||uniform),
                normalized entropy of recent). Add-0.5 smoothed."""
                kls, klUs, ents = [], [], []
                for recent, base, labels in zip(recentDists, baseDists, labelSets):
                    k = len(labels)
                    if k < 2:
                        continue
                    rTot = sum(recent.values()) + 0.5 * k
                    bTot = sum(base.values()) + 0.5 * k
                    kl = klU = ent = 0.0
                    for lbl in labels:
                        pr = (recent.get(lbl, 0) + 0.5) / rTot
                        pb = (base.get(lbl, 0) + 0.5) / bTot
                        kl += pr * math.log(pr / pb)
                        klU += pr * math.log(pr * k)
                        ent -= pr * math.log(pr)
                    kls.append(kl)
                    klUs.append(klU)
                    ents.append(ent / math.log(k))
                if not kls:
                    return None, None, None
                mean = lambda xs: sum(xs) / len(xs)
                return mean(kls), mean(klUs), mean(ents)

            # Main columns only for every distribution: the special columns
            # (and lotto's bonus) come from their own, smaller ranges, so
            # pooling them skews every KL/entropy figure and - for lotto -
            # leaked bonus values into a label set that should be the played
            # 1-45 mains. Pick3 passes through untouched (no special columns).
            def resultMains(res):
                realMainCount, _ = self.main_special_split(game, res)
                return [int(v) for v in res[:realMainCount]]

            def ticketMains(ticket, res):
                realMainCount, specialCount = self.main_special_split(game, res)
                mains, _ = self.split_ticket(ticket, realMainCount, specialCount)
                return [int(v) for v in mains]

            realDays = [(d, res) for d, res, _ in days if res]
            if len(realDays) >= MIN_WINDOW * 2:
                allResults = [resultMains(res) for _, res in realDays]
                labelSets = [sorted({int(v) for res in allResults for v in (res[:3] if isPick3 else res)[pos:pos+1]})
                             for pos in range(3)] if isPick3 else                             [sorted({int(v) for res in allResults for v in res})]
                baseDists = drawDistributions(allResults)

                recentResults = allResults[-WINDOW:]
                kl, klU, ent = klAndEntropy(drawDistributions(recentResults), baseDists, labelSets)

                # Trend: same stats for windows ending 10, 20, ... draws ago,
                # up to 12 checkpoints, oldest first.
                trend = []
                for back in range(110, -1, -10):
                    end = len(allResults) - back
                    if end < WINDOW:
                        continue
                    windowResults = allResults[end - WINDOW:end]
                    tKl, tKlU, tEnt = klAndEntropy(drawDistributions(windowResults), baseDists, labelSets)
                    trend.append({
                        "end_date": realDays[end - 1][0].strftime("%Y-%m-%d"),
                        "kl_vs_history": round(tKl, 4),
                        "kl_vs_uniform": round(tKlU, 4),
                        "entropy_norm": round(tEnt, 4),
                    })

                # Per model: predicted-number distribution over the recent
                # days it actually predicted, against the real draws of those
                # same days.
                modelKl = {}
                recentDays = [d for d in days if d[1]][-WINDOW:]
                for _, res, tickets in recentDays:
                    resMains = resultMains(res)
                    for modelName, ticket in tickets.items():
                        entry = modelKl.setdefault(modelName, {"pred": [], "real": []})
                        entry["pred"].append(ticketMains(ticket, res))
                        entry["real"].append(resMains)
                perModel = {}
                for modelName, entry in modelKl.items():
                    if len(entry["pred"]) < MIN_WINDOW:
                        continue
                    mKl, _, _ = klAndEntropy(drawDistributions(entry["pred"]),
                                             drawDistributions(entry["real"]), labelSets)
                    if mKl is not None:
                        perModel[modelName] = round(mKl, 4)

                report["games"][game]["randomnessWatch"] = {
                    "window": WINDOW,
                    "draws_total": len(allResults),
                    "kl_vs_history": round(kl, 4) if kl is not None else None,
                    "kl_vs_uniform": round(klU, 4) if klU is not None else None,
                    "entropy_norm": round(ent, 4) if ent is not None else None,
                    "trend": trend,
                    "model_kl_vs_real": perModel,
                }

        try:
            with open(historyPath, "w") as outfile:
                json.dump(peakHistory, outfile, indent=2)
        except Exception as e:
            print(f"Failed to write lag peak history to {historyPath}: {e}")

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
        

    def find_best_matching_prediction(self, real_result, predictions_dict, specialColumnCount=0, realMainCount=None):
        """
        Best-scoring prediction row against a real result, scored per pool:
        main numbers only count against the real mains and special numbers
        (stars/dream/viking) only against the real specials - a pooled set
        intersection let a predicted star "hit" a main number (they share
        numeric values but not a drawing), inflating match counts. Rows are
        ranked by (main hits, special hits), so mains outrank specials.

        realMainCount defaults to len(real_result) - specialColumnCount.
        Lotto passes (specialColumnCount=0, realMainCount=6): its 7th value
        is the BONUS ball, which follows the real game's tier rules - a
        played ticket has no bonus slot, so the bonus is matched against the
        ticket's own 6 numbers and lands in the special count ("5 (1)" = 5
        mains + bonus, a high tier but not the jackpot; at "6 (0)" all six
        played numbers are used as mains, and indeed a full main match makes
        a bonus match impossible since the bonus differs from every drawn
        main). Games without special columns (keno/pick3) pass through
        unchanged.

        Returned keys keep their historical names with mains-only semantics
        ("matching_numbers"/"match_count" = matched MAIN numbers), plus the
        mirrored "special_matching_numbers"/"special_match_count" (dedicated
        special columns and the lotto bonus both land there).
        """
        if realMainCount is None:
            realMainCount = len(real_result) - specialColumnCount
        real_mains = set(map(int, real_result[:realMainCount]))
        real_specials = set(map(int, real_result[realMainCount:realMainCount + specialColumnCount]))
        # Trailing values beyond mains + dedicated specials (lotto's bonus):
        # matched against the ticket's MAIN numbers, counted as special hits.
        real_bonus = set(map(int, real_result[realMainCount + specialColumnCount:]))

        best_match = {
            "model": None,
            "prediction": None,
            "matching_numbers": [],
            "match_count": 0,
            "special_matching_numbers": [],
            "special_match_count": 0,
        }

        best_score = (0, 0)
        for model in predictions_dict:
            model_name = model["name"]
            for predicted_list in model["predictions"]:
                ticket_mains, ticket_specials = self.split_ticket(
                    predicted_list, realMainCount, specialColumnCount)
                ticket_main_set = set(map(int, ticket_mains))
                matching_numbers = sorted(real_mains.intersection(ticket_main_set))
                special_matching_numbers = sorted(
                    real_specials.intersection(map(int, ticket_specials))
                    | real_bonus.intersection(ticket_main_set))

                # Strictly-greater keeps the historical behavior of leaving
                # model/prediction at None on an all-miss day.
                score = (len(matching_numbers), len(special_matching_numbers))
                if score > best_score:
                    best_score = score
                    best_match["model"] = model_name
                    best_match["prediction"] = predicted_list
                    best_match["matching_numbers"] = matching_numbers
                    best_match["match_count"] = len(matching_numbers)
                    best_match["special_matching_numbers"] = special_matching_numbers
                    best_match["special_match_count"] = len(special_matching_numbers)

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
                                # Main-pool hits only: a pooled intersection let
                                # special-column values (and lotto's unplayed
                                # bonus) count as hits, rewarding hyperopt for
                                # the wrong thing.
                                realMainCount, specialCount = self.main_special_split(name, real_result)
                                mains, _ = self.split_ticket(prediction, realMainCount, specialCount)
                                matches = len(set(map(int, mains)) & set(map(int, real_result[:realMainCount])))
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