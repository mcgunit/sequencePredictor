import os, sys, json, time
import numpy as np
from datetime import datetime

# Dynamically adjust the import path for Helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

# Ensure Helpers can be imported
if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Helpers import Helpers

helpers = Helpers()


class RLTicketModel():
    """
    Reinforcement-learned TICKET CONSTRUCTION, not number prediction.

    Every other model in the pipeline tries to guess which numbers will be
    drawn. This one instead learns how to ASSEMBLE a ticket out of the signals
    that already exist for a day - the other models' stored predictions plus
    the game's own draw history - so that the expected value under the real
    payout structure is maximized (Keno and Pick3 have actual payout tables in
    Helpers; games without one fall back to main-ticket hit count as reward).
    That distinction matters because payout tables are wildly non-linear
    (a Keno 10/10 pays 250000x, a near miss pays almost nothing), so the
    profit-optimal ticket is not necessarily the hit-optimal one.

    Policy: a linear score s(n) = theta . phi(n) per candidate number, with
    tickets sampled WITHOUT replacement via Plackett-Luce (iterative softmax
    over the remaining candidates). Trained with plain REINFORCE and a
    mean-reward baseline over the day JSONs the pipeline has already produced
    (each stores the model rows AND the real draw). Kept pure numpy on purpose:
    it must be cheap enough to (re)train per day on CPU next to the heavy DL
    models, and a 5-dimensional linear policy simply doesn't need TF/torch.

    Pick3 is positional (3 independent digit positions 0-9 where order decides
    straight/box/pair payouts), so it gets one theta-scored softmax PER
    POSITION with the same feature template, sampling one digit per position -
    digits may repeat across positions, exactly like the real game.
    """

    def __init__(self):
        # Default matches the repo layout (data/models/<model folder>); the
        # Predictor overrides this via setModelPath like it does for every
        # other model.
        self.modelPath = os.path.join(parent_dir, "data", "models", "rl_model")
        self.learningRate = 0.05
        self.epochs = 30
        self.samplesPerDay = 32
        self.trainDays = 120
        self._mainCount = None
        self.seed = None
        # Hard wall-clock cap: this runs inside the per-day pipeline where a
        # runaway training loop would stall every other model, so we check the
        # clock between epochs and simply keep whatever theta we have.
        self.maxTrainSeconds = 60
        # Payout tables have jackpot outliers (Keno 10/10 pays 250000, Pick3
        # straight pays ~500 on a 1-euro-scale stake). One lucky sample times
        # learningRate would otherwise catapult theta into a regime where the
        # softmax saturates and the policy can never recover, so gradients get
        # L2-norm-clipped before every update.
        self.maxGradNorm = 10.0
        # 4 informative features + 1 bias, see _normalizeFeatures for why the
        # bias is appended after z-scoring instead of being z-scored itself.
        self.featureNames = ["voteShare", "meanRank", "drawFrequency", "drawsSinceSeen", "bias"]

    # --- SETTERS (repo idiom: one-liners, configured by the Predictor) ---
    def setModelPath(self, modelPath): self.modelPath = modelPath
    def setLearningRate(self, alpha): self.learningRate = float(alpha)
    def setEpochs(self, epochs): self.epochs = max(1, int(epochs))
    def setSamplesPerDay(self, samples): self.samplesPerDay = max(1, int(samples))
    def setTrainDays(self, days): self.trainDays = max(1, int(days))
    def setSeed(self, seed): self.seed = seed
    def setMaxTrainSeconds(self, seconds): self.maxTrainSeconds = float(seconds)

    # ------------------------------------------------------------------ #
    # History loading                                                     #
    # ------------------------------------------------------------------ #

    def _loadHistory(self, historyDir, cutoffDate=None, mainCount=None):
        """
        Reads the game's day JSONs (the pipeline's own database) into a
        date-sorted list of (date, rows, realResult). Only "currentPrediction"
        rows are used - they are the scored/final rows FOR that day's draw.
        (newPrediction is deliberately not a fallback: it targets the NEXT
        day's draw, so pairing it with this day's realResult would train the
        policy on a one-draw-misaligned sample.) Days missing a realResult
        are skipped entirely since they can neither be scored for training
        nor contribute draw statistics.

        cutoffDate: during a history rebuild, days >= the day being rebuilt
        already exist on disk WITH their realResults (step 1 writes them all
        before step 2 loops) - training on them would be look-ahead bias, so
        they are dropped, mirroring the skipRows discipline every other model
        gets. None (the daily fresh-prediction path) keeps everything.

        mainCount: realResult is truncated to the main numbers - trailing
        special columns (stars/dream/viking) and lotto's bonus share value
        ranges with the mains, and folding them into the draw statistics or
        the reward would systematically inflate the low numbers.
        """
        entries = []
        if not historyDir or not os.path.isdir(historyDir):
            return entries

        for fileName in os.listdir(historyDir):
            if not fileName.endswith(".json"):
                continue
            try:
                # File names are "YYYY-M-D.json" (not zero padded), which
                # strptime accepts fine; anything unparsable isn't a day file.
                fileDate = datetime.strptime(fileName[:-5], "%Y-%m-%d")
            except ValueError:
                continue
            try:
                with open(os.path.join(historyDir, fileName), "r") as infile:
                    dayData = json.load(infile)
            except Exception:
                continue

            if cutoffDate is not None and fileDate >= cutoffDate:
                continue

            realResult = dayData.get("realResult")
            if not realResult:
                continue
            if mainCount:
                realResult = realResult[:mainCount]
            rows = dayData.get("currentPrediction") or []
            entries.append((fileDate, rows, realResult))

        entries.sort(key=lambda entry: entry[0])
        return entries

    # ------------------------------------------------------------------ #
    # Feature construction                                                #
    # ------------------------------------------------------------------ #

    def _rawFeatures(self, rows, freqCounts, lastSeen, historyCount, candidates):
        """
        Raw (unnormalized) per-number features for a set-based game. Only each
        row's FIRST ticket is counted (predictions[0] is every model's main
        ticket; the rest are Keno subsets, and counting those would double
        weight whatever the subsets already favor). Tickets longer than
        self._mainCount are truncated to it: the trailing entries are special
        columns (stars/dream/viking) whose smaller value range sits INSIDE
        the main range, so without the cut they would count as phantom votes
        for the low main numbers. Numbers outside the candidate range are
        still ignored rather than crashing the indexer.
        """
        candidateCount = len(candidates)
        indexOf = {int(n): i for i, n in enumerate(candidates)}

        votes = np.zeros(candidateCount)
        rankSum = np.zeros(candidateCount)
        rankCount = np.zeros(candidateCount)

        for row in rows or []:
            predictions = row.get("predictions") or []
            if not predictions or not predictions[0]:
                continue
            ticket = predictions[0]
            if self._mainCount and len(ticket) > self._mainCount:
                ticket = ticket[:self._mainCount]
            ticketLength = len(ticket)
            for position, number in enumerate(ticket):
                try:
                    i = indexOf.get(int(number))
                except (TypeError, ValueError):
                    continue
                if i is None:
                    continue
                votes[i] += 1
                # Rank normalized to [0,1] so 20-number Keno tickets and
                # 7-number Lotto tickets land on the same scale.
                rankSum[i] += position / max(1, ticketLength - 1)
                rankCount[i] += 1

        totalVotes = votes.sum()
        voteShare = votes / totalVotes if totalVotes > 0 else votes
        # Numbers no model picked get the neutral mid-rank 0.5 instead of 0 -
        # 0 would falsely mark them as "always first in the ticket".
        meanRank = np.where(rankCount > 0, rankSum / np.maximum(rankCount, 1), 0.5)

        safeHistory = max(1, historyCount)
        drawFrequency = np.array(
            [freqCounts.get(int(n), 0) for n in candidates], dtype=float) / safeHistory
        # "Never seen" gets historyCount + 1: strictly staler than anything
        # actually observed, without inventing an infinite value.
        drawsSince = np.array(
            [historyCount - lastSeen[int(n)] if int(n) in lastSeen else historyCount + 1
             for n in candidates], dtype=float)

        return np.stack([voteShare, meanRank, drawFrequency, drawsSince], axis=1)

    def _rawFeaturesPick3(self, rows, positionFreq, positionLastSeen, historyCount, positions, classes):
        """
        Same feature template but per (position, digit), because Pick3 payouts
        are positional. Vote share and draw statistics are computed per
        position; mean rank is computed per digit across all positions it
        appears in (within a single position the rank would be the constant
        position index, i.e. no signal at all).
        """
        votes = np.zeros((positions, classes))
        rankSum = np.zeros(classes)
        rankCount = np.zeros(classes)

        for row in rows or []:
            predictions = row.get("predictions") or []
            if not predictions or not predictions[0]:
                continue
            ticket = predictions[0]
            if len(ticket) != positions:
                continue
            for position, digit in enumerate(ticket):
                try:
                    digit = int(digit)
                except (TypeError, ValueError):
                    continue
                if digit < 0 or digit >= classes:
                    continue
                votes[position, digit] += 1
                rankSum[digit] += position / max(1, positions - 1)
                rankCount[digit] += 1

        positionTotals = votes.sum(axis=1, keepdims=True)
        voteShare = np.divide(votes, positionTotals, out=np.zeros_like(votes),
                              where=positionTotals > 0)
        meanRank = np.where(rankCount > 0, rankSum / np.maximum(rankCount, 1), 0.5)
        meanRank = np.tile(meanRank, (positions, 1))

        safeHistory = max(1, historyCount)
        drawFrequency = positionFreq / safeHistory
        drawsSince = np.where(positionLastSeen >= 0,
                              historyCount - positionLastSeen,
                              historyCount + 1).astype(float)

        return np.stack([voteShare, meanRank, drawFrequency, drawsSince], axis=2)

    def _normalizeFeatures(self, raw):
        """
        Z-scores each feature column within the day's candidate set, then
        appends the bias column. Per-day normalization is deliberate: vote
        shares shrink as more models join the pipeline and drawsSinceSeen
        grows with history length, so a single global normalization would
        silently drift out from under a warm-started theta. Columns with ~zero
        variance are zeroed (they carry no signal that day, and dividing by
        their std would blow up) - the bias term is appended AFTER z-scoring
        precisely so it can't be destroyed by that same rule.
        """
        axis = -2  # candidates axis, works for both (N,4) and (P,C,4)
        mean = raw.mean(axis=axis, keepdims=True)
        std = raw.std(axis=axis, keepdims=True)
        safeStd = np.where(std > 1e-12, std, 1.0)
        z = np.where(std > 1e-12, (raw - mean) / safeStd, 0.0)
        bias = np.ones(z.shape[:-1] + (1,))
        return np.concatenate([z, bias], axis=-1), mean.reshape(-1).tolist(), std.reshape(-1).tolist()

    def _sweepHistory(self, entries, candidates, isPick3, positions, classes):
        """
        Single chronological pass over the history: for each day that will be
        trained on, features are built from THAT day's stored rows plus the
        draw statistics accumulated strictly BEFORE it (no peeking at its own
        result), and only afterwards is the day's realResult folded into the
        running statistics. The final statistics (covering the whole history)
        are returned as well - they're exactly what today's prediction
        features need.
        """
        trainableIndexes = [i for i, (_, rows, _) in enumerate(entries) if rows]
        windowIndexes = set(trainableIndexes[-self.trainDays:])

        if isPick3:
            positionFreq = np.zeros((positions, classes))
            positionLastSeen = np.full((positions, classes), -1.0)
            freqCounts, lastSeen = None, None
        else:
            freqCounts, lastSeen = {}, {}
            positionFreq, positionLastSeen = None, None

        historyCount = 0
        trainingDays = []

        for i, (_, rows, realResult) in enumerate(entries):
            if i in windowIndexes:
                if isPick3:
                    raw = self._rawFeaturesPick3(rows, positionFreq, positionLastSeen,
                                                 historyCount, positions, classes)
                else:
                    raw = self._rawFeatures(rows, freqCounts, lastSeen, historyCount, candidates)
                phi, _, _ = self._normalizeFeatures(raw)
                trainingDays.append({"phi": phi, "real": realResult})

            # Fold this day's draw into the running stats AFTER any feature
            # build, so day i never sees its own outcome.
            if isPick3:
                for position, digit in enumerate(realResult[:positions]):
                    try:
                        digit = int(digit)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= digit < classes:
                        positionFreq[position, digit] += 1
                        positionLastSeen[position, digit] = historyCount
            else:
                for number in realResult:
                    try:
                        number = int(number)
                    except (TypeError, ValueError):
                        continue
                    freqCounts[number] = freqCounts.get(number, 0) + 1
                    lastSeen[number] = historyCount
            historyCount += 1

        finalStats = {
            "freqCounts": freqCounts, "lastSeen": lastSeen,
            "positionFreq": positionFreq, "positionLastSeen": positionLastSeen,
            "historyCount": historyCount,
        }
        return trainingDays, finalStats

    # ------------------------------------------------------------------ #
    # REINFORCE training                                                  #
    # ------------------------------------------------------------------ #

    def _rewardFunction(self, isPick3, isKeno):
        """
        Keno and Pick3 have real payout tables, so the reward IS the euro
        profit - that's the whole point of this model. Games without a payout
        table in Helpers fall back to main-ticket hit count, which at least
        points the policy at the drawn numbers.
        """
        if isPick3:
            def reward(ticket, realResult):
                profit = helpers.pick3_ticket_profit(list(ticket), realResult)
                return float(profit) if profit is not None else 0.0
            return reward
        if isKeno:
            def reward(ticket, realResult):
                profit = helpers.keno_ticket_profit(list(ticket), realResult)
                return float(profit) if profit is not None else 0.0
            return reward

        def reward(ticket, realResult):
            realSet = set(int(n) for n in realResult)
            return float(len(set(int(n) for n in ticket) & realSet))
        return reward

    def _clipGradient(self, gradient):
        norm = float(np.linalg.norm(gradient))
        if norm > self.maxGradNorm and norm > 0:
            return gradient * (self.maxGradNorm / norm)
        return gradient

    def _trainSetGame(self, trainingDays, theta, candidates, ticketSize, rewardFunc, startTime, rng):
        """
        REINFORCE for the set-based games. Ticket sampling uses the Gumbel
        top-k trick: adding independent Gumbel noise to the scores and taking
        the top-k is EXACTLY Plackett-Luce sampling without replacement, but
        it vectorizes over all samplesPerDay tickets at once instead of
        looping softmax-draw-remove per slot in Python. The gradient of the
        log-probability still follows the sequential formulation
        (sum over slots of phi(chosen) - E_softmax[phi | remaining]), which the
        reverse cumulative sums below compute for every sample in one shot.
        """
        epochMeanRewards = []
        candidateArray = np.asarray(candidates)
        sampleCount = self.samplesPerDay

        for epoch in range(self.epochs):
            # Clock check between epochs: keep whatever theta we already have
            # rather than risk stalling the whole per-day pipeline.
            if time.time() - startTime > self.maxTrainSeconds:
                break
            epochRewardSum = 0.0
            for day in trainingDays:
                phi = day["phi"]
                scores = phi @ theta
                scores = scores - scores.max()  # softmax shift-invariance, avoids exp overflow
                weights = np.exp(scores)

                gumbel = rng.gumbel(size=(sampleCount, len(candidateArray)))
                order = np.argsort(-(scores + gumbel), axis=1)

                orderedWeights = weights[order]
                orderedPhi = phi[order]
                # Reverse cumulative sums give, for every slot t, the softmax
                # normalizer and expected feature vector over the numbers
                # still available at that slot.
                suffixWeight = np.cumsum(orderedWeights[:, ::-1], axis=1)[:, ::-1]
                suffixPhi = np.cumsum((orderedWeights[:, :, None] * orderedPhi)[:, ::-1, :],
                                      axis=1)[:, ::-1, :]

                expectedPhi = suffixPhi[:, :ticketSize, :] / np.maximum(
                    suffixWeight[:, :ticketSize, None], 1e-30)
                gradLogProb = np.sum(orderedPhi[:, :ticketSize, :] - expectedPhi, axis=1)

                tickets = candidateArray[order[:, :ticketSize]]
                rewards = np.array([rewardFunc(ticket, day["real"]) for ticket in tickets])

                # Mean-reward baseline per day: with jackpot-style rewards the
                # variance is brutal, and the baseline is what keeps the
                # estimator from just pushing theta wherever the last lucky
                # sample pointed.
                advantage = rewards - rewards.mean()
                gradient = (advantage[:, None] * gradLogProb).mean(axis=0)
                theta = theta + self.learningRate * self._clipGradient(gradient)
                epochRewardSum += rewards.mean()

            epochMeanRewards.append(epochRewardSum / max(1, len(trainingDays)))

        return theta, epochMeanRewards

    def _trainPick3(self, trainingDays, theta, positions, classes, rewardFunc, startTime, rng):
        """
        REINFORCE for Pick3: one independent softmax per position (digits may
        repeat across positions, like the real game), so the log-prob gradient
        is simply summed over the three positional categorical draws.
        """
        epochMeanRewards = []
        sampleCount = self.samplesPerDay

        for epoch in range(self.epochs):
            if time.time() - startTime > self.maxTrainSeconds:
                break
            epochRewardSum = 0.0
            for day in trainingDays:
                phi = day["phi"]  # (positions, classes, features)
                scores = np.einsum('pcf,pf->pc', phi, theta)
                scores = scores - scores.max(axis=1, keepdims=True)
                probs = np.exp(scores)
                probs = probs / probs.sum(axis=1, keepdims=True)

                digits = np.stack(
                    [rng.choice(classes, size=sampleCount, p=probs[p]) for p in range(positions)],
                    axis=1)  # (samples, positions)

                expectedPhi = np.einsum('pc,pcf->pf', probs, phi)
                chosenPhi = phi[np.arange(positions)[None, :], digits]  # (samples, positions, features)
                gradLogProb = chosenPhi - expectedPhi[None, :, :]

                rewards = np.array([rewardFunc(ticket, day["real"]) for ticket in digits])
                advantage = rewards - rewards.mean()
                gradient = (advantage[:, None, None] * gradLogProb).mean(axis=0)
                theta = theta + self.learningRate * self._clipGradient(gradient)
                epochRewardSum += rewards.mean()

            epochMeanRewards.append(epochRewardSum / max(1, len(trainingDays)))

        return theta, epochMeanRewards

    # ------------------------------------------------------------------ #
    # Policy persistence                                                  #
    # ------------------------------------------------------------------ #

    def _policyPath(self, name):
        return os.path.join(self.modelPath, f"{name}_policy.json")

    def _loadPolicy(self, name, expectedShape):
        """
        Warm start: yesterday's theta is the best available initialization,
        since the feature template and per-day normalization keep its scale
        meaningful across days. A shape mismatch (feature template changed, or
        a different game variant) silently falls back to zeros - a zero theta
        is a uniform Plackett-Luce policy, i.e. a safe blank slate.
        """
        policyPath = self._policyPath(name)
        if not os.path.exists(policyPath):
            return None
        try:
            with open(policyPath, "r") as infile:
                stored = json.load(infile)
            theta = np.array(stored.get("theta"), dtype=float)
            if theta.shape != expectedShape or not np.all(np.isfinite(theta)):
                print(f"RLTicketModel: stored policy for {name} has shape {theta.shape}, "
                      f"expected {expectedShape} - starting fresh")
                return None
            return theta
        except Exception as e:
            print(f"RLTicketModel: could not load stored policy for {name}: {e}")
            return None

    def _savePolicy(self, name, isPick3, theta, featureMean, featureStd, report):
        try:
            os.makedirs(self.modelPath, exist_ok=True)
            policy = {
                "game": name,
                "isPick3": isPick3,
                "featureNames": self.featureNames,
                "theta": theta.tolist(),
                # The z-scoring itself is per-day (see _normalizeFeatures);
                # these are the most recent day's raw stats, persisted so the
                # policy file documents the feature scale theta was trained
                # against and a future run can sanity-check against drift.
                "featureStats": {"mean": featureMean, "std": featureStd},
                "report": report,
            }
            with open(self._policyPath(name), "w") as outfile:
                json.dump(policy, outfile, indent=2)
        except Exception as e:
            print(f"RLTicketModel: failed to persist policy for {name}: {e}")

    # ------------------------------------------------------------------ #
    # Fallback                                                            #
    # ------------------------------------------------------------------ #

    def _voteShareFallback(self, rows, candidates, drawSize, isPick3, positions, classes, kenoSubsetSizes):
        """
        Degraded mode when there is nothing to train on (or training blew up):
        rank purely by today's vote share across the other models' main
        tickets. It's the strongest single feature the policy uses anyway, so
        it is the honest approximation of an untrained policy's intent.
        """
        if isPick3:
            votes = np.zeros((positions, classes))
            for row in rows or []:
                predictions = row.get("predictions") or []
                if not predictions or len(predictions[0]) != positions:
                    continue
                for position, digit in enumerate(predictions[0]):
                    try:
                        digit = int(digit)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= digit < classes:
                        votes[position, digit] += 1
            # argmax ties resolve to the smallest digit, keeping the fallback
            # deterministic without a seed.
            return [[int(np.argmax(votes[p])) for p in range(positions)]]

        votes = {}
        for row in rows or []:
            predictions = row.get("predictions") or []
            if not predictions:
                continue
            ticket = predictions[0]
            if self._mainCount and len(ticket) > self._mainCount:
                # Same special-column cut as _rawFeatures - see there.
                ticket = ticket[:self._mainCount]
            for number in ticket:
                try:
                    number = int(number)
                except (TypeError, ValueError):
                    continue
                votes[number] = votes.get(number, 0) + 1
        # Sort by (-votes, number): deterministic, and unvoted numbers only
        # ever appear if fewer than drawSize numbers were voted on at all.
        ranked = sorted((int(n) for n in candidates), key=lambda n: (-votes.get(n, 0), n))
        mainTicket = sorted(ranked[:drawSize])
        predictions = [mainTicket]
        for subsetSize in kenoSubsetSizes or []:
            subset = sorted(ranked[:min(subsetSize, len(ranked))])
            predictions.append(subset)
        return predictions

    # ------------------------------------------------------------------ #
    # Public prediction API                                               #
    # ------------------------------------------------------------------ #

    def run(self, name, listOfDecodedPredictions, historyDir, gameConfig):
        """
        (Re)trains the policy on the game's stored day JSONs (warm-started
        from the persisted policy) and returns this model's row for today:

            {"name": "RL Ticket Model", "predictions": [mainTicket, subsets...]}

        The output is deterministic: sampling only happens during training,
        the emitted ticket is the top-drawSize numbers by policy score (Pick3:
        argmax per position). Never raises - any failure degrades to the
        vote-share fallback so one broken day can't take the pipeline down.
        """
        startTime = time.time()
        gameConfig = gameConfig or {}
        isPick3 = bool(gameConfig.get("isPick3"))
        kenoSubsetSizes = gameConfig.get("kenoSubsetSizes") or []
        drawSize = int(gameConfig.get("drawSize") or (3 if isPick3 else 20))
        # Everything this model reads (stored tickets, realResults) is cut to
        # the main numbers: drawSize IS the main count (specials/bonus have
        # their own ranges and are not constructed by this model).
        self._mainCount = drawSize
        cutoffDate = gameConfig.get("cutoffDate")

        if isPick3:
            classes = int(gameConfig.get("perPositionClasses") or 10)
            positions = drawSize
            candidates = np.arange(classes)
        else:
            numberRange = gameConfig.get("numberRange") or (1, 80)
            if isinstance(numberRange, (list, tuple)):
                low, high = int(numberRange[0]), int(numberRange[-1])
            else:
                low, high = 1, int(numberRange)
            candidates = np.arange(low, high + 1)
            classes, positions = 0, 0

        try:
            rng = np.random.default_rng(self.seed)
            isKeno = "keno" in str(name)
            featureCount = len(self.featureNames)
            expectedShape = (positions, featureCount) if isPick3 else (featureCount,)

            theta = self._loadPolicy(name, expectedShape)
            warmStart = theta is not None
            if theta is None:
                theta = np.zeros(expectedShape)

            entries = self._loadHistory(historyDir, cutoffDate=cutoffDate, mainCount=drawSize if not isPick3 else None)
            trainingDays, finalStats = self._sweepHistory(
                entries, candidates, isPick3, positions, classes)

            epochMeanRewards = []
            if trainingDays:
                rewardFunc = self._rewardFunction(isPick3, isKeno)
                if isPick3:
                    theta, epochMeanRewards = self._trainPick3(
                        trainingDays, theta, positions, classes, rewardFunc, startTime, rng)
                else:
                    if isKeno:
                        # The 20-number Keno main ticket itself has no payout
                        # (only 5-10-number subsets are playable), so training
                        # samples tickets of the largest configured playable
                        # size - the profit gradient then shapes the same
                        # score ranking the emitted 20-number ticket and its
                        # subsets are read from.
                        playable = [s for s in kenoSubsetSizes if 5 <= int(s) <= 10]
                        ticketSize = max(playable) if playable else 10
                    else:
                        ticketSize = drawSize
                    ticketSize = min(int(ticketSize), len(candidates))
                    theta, epochMeanRewards = self._trainSetGame(
                        trainingDays, theta, candidates, ticketSize, rewardFunc, startTime, rng)
            else:
                if not warmStart:
                    # Nothing to train on AND no previously learned policy: a
                    # zero theta would rank every number equally and just emit
                    # the lowest candidates, so the vote-share fallback is the
                    # more honest answer here.
                    print(f"RLTicketModel: no trainable history for {name} in {historyDir} "
                          f"and no stored policy - falling back to vote-share ranking")
                    return {"name": "RL Ticket Model",
                            "predictions": self._voteShareFallback(
                                listOfDecodedPredictions, candidates, drawSize,
                                isPick3, positions, classes, kenoSubsetSizes)}
                print(f"RLTicketModel: no trainable history for {name} in {historyDir} - "
                      f"skipping training, decoding with the stored policy")

            if not np.all(np.isfinite(theta)):
                # A diverged policy is worse than no policy: reset instead of
                # persisting garbage that would poison every future warm start.
                print(f"RLTicketModel: non-finite theta after training for {name} - resetting")
                theta = np.zeros(expectedShape)

            # Today's features come from the passed-in rows plus the FULL
            # history statistics the sweep ended on.
            if isPick3:
                todayRaw = self._rawFeaturesPick3(
                    listOfDecodedPredictions, finalStats["positionFreq"],
                    finalStats["positionLastSeen"], finalStats["historyCount"],
                    positions, classes)
            else:
                todayRaw = self._rawFeatures(
                    listOfDecodedPredictions, finalStats["freqCounts"],
                    finalStats["lastSeen"], finalStats["historyCount"], candidates)
            todayPhi, featureMean, featureStd = self._normalizeFeatures(todayRaw)

            elapsed = time.time() - startTime
            report = {
                "trainedAt": datetime.now().isoformat(timespec="seconds"),
                "warmStart": warmStart,
                "trainingDays": len(trainingDays),
                "epochsRun": len(epochMeanRewards),
                "epochsRequested": self.epochs,
                "timeCapped": len(epochMeanRewards) < self.epochs and bool(trainingDays),
                "elapsedSeconds": round(elapsed, 2),
                "firstEpochMeanReward": epochMeanRewards[0] if epochMeanRewards else None,
                "lastEpochMeanReward": epochMeanRewards[-1] if epochMeanRewards else None,
                "epochMeanRewards": epochMeanRewards,
            }
            if trainingDays:
                self._savePolicy(name, isPick3, theta, featureMean, featureStd, report)
                if epochMeanRewards:
                    rewardSummary = (f"mean reward {report['firstEpochMeanReward']:.4f} -> "
                                     f"{report['lastEpochMeanReward']:.4f}")
                else:
                    # Time cap hit before even one epoch finished - theta is
                    # just the warm start (or zeros), which is still valid.
                    rewardSummary = "no epoch completed inside the time cap"
                print(f"RLTicketModel [{name}]: {len(trainingDays)} days, "
                      f"{len(epochMeanRewards)} epochs in {elapsed:.1f}s, "
                      f"{rewardSummary} (warmStart={warmStart})")

            # Deterministic decode: no sampling at prediction time, so the
            # emitted row is reproducible from the persisted policy alone.
            if isPick3:
                scores = np.einsum('pcf,pf->pc', todayPhi, theta)
                # Drawn order = positional order; ties resolve to the lowest
                # digit via argmax, keeping the output deterministic.
                mainTicket = [int(np.argmax(scores[p])) for p in range(positions)]
                return {"name": "RL Ticket Model", "predictions": [mainTicket]}

            scores = todayPhi @ theta
            # Stable sort on -scores: equal scores keep candidate (ascending
            # number) order, so ties never depend on rng state.
            ranked = candidates[np.argsort(-scores, kind="stable")]
            mainTicket = sorted(int(n) for n in ranked[:drawSize])
            predictions = [mainTicket]
            # Keno subsets mirror how the other rows emit theirs: a subset of
            # the main ticket, here just the greedy top scorers among it.
            rankedMain = [int(n) for n in ranked[:drawSize]]
            for subsetSize in kenoSubsetSizes:
                subsetSize = min(int(subsetSize), len(rankedMain))
                predictions.append(sorted(rankedMain[:subsetSize]))
            return {"name": "RL Ticket Model", "predictions": predictions}

        except Exception as e:
            print(f"RLTicketModel: training/decoding failed for {name} ({e}) - "
                  f"falling back to vote-share ranking")
            try:
                predictions = self._voteShareFallback(
                    listOfDecodedPredictions, candidates, drawSize,
                    isPick3, positions, classes, kenoSubsetSizes)
            except Exception as fallbackError:
                # Absolute last resort: a syntactically valid ticket so the
                # day JSON stays well-formed even on a completely broken day.
                print(f"RLTicketModel: fallback failed too ({fallbackError})")
                if isPick3:
                    predictions = [[0] * drawSize]
                else:
                    predictions = [sorted(int(n) for n in candidates[:drawSize])]
            return {"name": "RL Ticket Model", "predictions": predictions}
