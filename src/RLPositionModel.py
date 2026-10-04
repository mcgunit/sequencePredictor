"""
Position sizing for the market games (README roadmap item 4, phase M4): the
RL row of crypto and shares.

The RL Ticket Model learns how to ASSEMBLE a lottery ticket against a payout
table. A market game has no payout table and nothing to assemble - every row
already names one bin per instrument - but it has the one lever a lottery
player never gets: how MUCH to put on a call. This row learns that. Per
instrument and day it chooses a position size from SIZES (0 = sit out, 1 =
the paper stake, up to 2 = twice the stake) from what the other rows said
about that day, and is paid the paper money the position made under the
daily rule (MarketSettle.cash_pnl: the stake bought at the previous close,
sold at the day's close, a fee on each leg), scaled by the size.

Features, all in BIN space so they mean the same for every instrument and
need no edges (one bin is one tenth of the instrument's own history):
    upShare     share of the rows whose bin is in the upper half (5-9)
    consensus   mean of (bin - 4.5) / 4.5 over the rows: -1 .. +1
    garch       the GARCH Model's bin on the same scale (0 when absent)
    regime      the Regime HMM Model's bin on the same scale
    vote        the WeightedEnsemble Model's bin on the same scale, else the consensus
    yesterday   the instrument's realised bin of the newest settled day, same scale
    recentAbs   mean |bin - 4.5| / 4.5 of its last RECENT realised bins (how wild the days were)
    bias        1

Policy: a linear softmax over the sizes, theta of shape (features, sizes),
trained by REINFORCE with a per-state baseline (the mean reward of the
samples drawn for the same instrument-day) on the stored day JSONs the
pipeline has already produced - each holds the rows made FOR a day
(currentPrediction) and the day's real bins; the day's real returns come
from the market's returns file (MarketModels.MarketHistory). The reward is
not the money itself but a concave utility of it, money - riskAversion x
money^2: expected money is linear in the size, so a risk-neutral learner
would only ever choose 0 or the maximum and "sizing" would be a coin flip
on the sign; the quadratic term is what makes an intermediate size the
right answer for an uncertain call. Pure numpy, warm-started from the
persisted policy, wall-clock capped: it runs inside the daily pipeline like
the RL Ticket Model.

The row's TICKET - the bins the hit rates are scored on - is the per-slot
weighted vote over the other rows (Helpers.build_positional_vote_predictions,
the WeightedEnsemble recipe), because this row predicts sizes, not bins; its
own contribution is "positions": the size per instrument, which the
settlement stores with the prediction and the paper books apply
(MarketSettle.position_of). Without a trainable history and without a stored
policy it falls back to the plain rule: size 1 where the vote is up, 0 where
it is not - a row no better and no worse than the vote.

    python3 -m src.RLPositionModel      # self-check on a synthetic history
"""

import json
import math
import os
import time
from datetime import datetime

import numpy as np

try:
    from src.Helpers import Helpers, MARKET_BINS
    from src.MarketModels import MarketHistory
    from src.MarketSettle import cash_pnl, STAKE, FEE_PER_LEG
except ImportError:  # imported from inside src/
    from Helpers import Helpers, MARKET_BINS
    from MarketModels import MarketHistory
    from MarketSettle import cash_pnl, STAKE, FEE_PER_LEG

helpers = Helpers()

ROW_NAME = "RL Position Model"
SIZES = (0.0, 0.5, 1.0, 1.5, 2.0)
FEATURES = ["upShare", "consensus", "garch", "regime", "vote", "yesterday", "recentAbs", "bias"]
RECENT = 5
GARCH_ROW = "GARCH Model"
REGIME_ROW = "Regime HMM Model"
VOTE_ROW = "WeightedEnsemble Model"


def _scaled(bin_value, k):
    """A bin as a number in -1 .. +1 around the middle of the range."""
    half = (k - 1) / 2.0
    return (float(bin_value) - half) / half


class RLPositionModel:
    def __init__(self):
        self.modelPath = os.path.join("data", "models", "rl_position")
        self.learningRate = 0.05
        self.epochs = 30
        self.samplesPerDay = 16
        self.trainDays = 120
        self.riskAversion = 0.1
        self.maxTrainSeconds = 60
        self.maxGradNorm = 10.0
        self.stake = STAKE
        self.fee = FEE_PER_LEG
        self.seed = None
        self.k = MARKET_BINS
        self.lastReport = None

    # --- setters (the Predictor configures the row from bestParams_<market>.json) ---
    def setModelPath(self, modelPath): self.modelPath = modelPath
    def setLearningRate(self, alpha): self.learningRate = float(alpha)
    def setEpochs(self, epochs): self.epochs = max(1, int(epochs))
    def setSamplesPerDay(self, samples): self.samplesPerDay = max(1, int(samples))
    def setTrainDays(self, days): self.trainDays = max(1, int(days))
    def setRiskAversion(self, value): self.riskAversion = max(0.0, float(value))
    def setMaxTrainSeconds(self, seconds): self.maxTrainSeconds = float(seconds)
    def setSeed(self, seed): self.seed = seed

    # --- history ---------------------------------------------------------------
    def _loadHistory(self, historyDir, cutoffDate=None):
        """
        The market's day JSONs as a date-sorted list of (date 'YYYY-MM-DD',
        rows, realResult): the rows made FOR the day (currentPrediction) and
        its real bins. Days without a real result are skipped; with a
        cutoffDate (a history rebuild) days on or after it are dropped, so
        the policy never trains on a day that is still to be predicted.
        """
        entries = []
        if not historyDir or not os.path.isdir(historyDir):
            return entries
        for fileName in os.listdir(historyDir):
            if not fileName.endswith(".json"):
                continue
            try:
                fileDate = datetime.strptime(fileName[:-5], "%Y-%m-%d")
            except ValueError:
                continue
            if cutoffDate is not None and fileDate >= cutoffDate:
                continue
            try:
                with open(os.path.join(historyDir, fileName), "r") as infile:
                    dayData = json.load(infile)
            except Exception:
                continue
            realResult = dayData.get("realResult")
            rows = dayData.get("currentPrediction") or []
            if not realResult or not rows:
                continue
            entries.append((fileDate.strftime("%Y-%m-%d"), rows, [int(v) for v in realResult]))
        entries.sort(key=lambda entry: entry[0])
        return entries

    def _loadReturns(self, dataPath):
        """{date: [log return per slot]} from the market's returns file."""
        days, matrix = MarketHistory(dataPath).visible(0)
        return {str(d): [float(v) for v in row] for d, row in zip(days, matrix)}

    # --- features --------------------------------------------------------------
    def _features(self, rows, slots, yesterday, recent):
        """(slots, len(FEATURES)) raw features from the day's rows and the realised bins before it."""
        k = self.k
        out = np.zeros((slots, len(FEATURES)))
        named = {}
        tickets = []
        for row in rows:
            name = row.get("name")
            preds = row.get("predictions") or []
            if name == ROW_NAME or not preds or not preds[0] or len(preds[0]) < slots:
                continue
            try:
                ticket = [int(v) for v in preds[0][:slots]]
            except (TypeError, ValueError):
                continue
            tickets.append(ticket)
            named[name] = ticket
        for slot in range(slots):
            bins = [t[slot] for t in tickets]
            up = float(np.mean([b >= k / 2 for b in bins])) if bins else 0.5
            consensus = float(np.mean([_scaled(b, k) for b in bins])) if bins else 0.0
            garch = _scaled(named[GARCH_ROW][slot], k) if GARCH_ROW in named else 0.0
            regime = _scaled(named[REGIME_ROW][slot], k) if REGIME_ROW in named else 0.0
            vote = _scaled(named[VOTE_ROW][slot], k) if VOTE_ROW in named else consensus
            yday = _scaled(yesterday[slot], k) if yesterday is not None and slot < len(yesterday) else 0.0
            past = [abs(_scaled(r[slot], k)) for r in recent if slot < len(r)]
            wild = float(np.mean(past)) if past else 0.5
            out[slot] = [up, consensus, garch, regime, vote, yday, wild, 1.0]
        return out

    def _normalise(self, raw, mean, std):
        phi = (raw - mean) / std
        phi[..., -1] = 1.0   # the bias is never z-scored
        return phi

    # --- reward ------------------------------------------------------------------
    def _utility(self, size, ret):
        money = float(size) * cash_pnl(ret, self.stake, self.fee)
        return money - self.riskAversion * money * money

    # --- training ------------------------------------------------------------------
    def _train(self, theta, phis, rets, startTime, rng):
        """
        REINFORCE over every instrument-day of the window: phis (days, slots,
        features) already normalised, rets (days, slots) the real log returns.
        Returns (theta, mean utility per epoch); stops at the wall-clock cap.
        """
        sizes = np.asarray(SIZES)
        n_actions = len(sizes)
        epochMeans = []
        states = phis.reshape(-1, phis.shape[-1])
        returns = rets.reshape(-1)
        keep = np.isfinite(returns)
        states, returns = states[keep], returns[keep]
        if not len(states):
            return theta, epochMeans
        money = np.array([[s * cash_pnl(r, self.stake, self.fee) for s in sizes] for r in returns])   # (states, actions)
        utility = money - self.riskAversion * money * money
        for epoch in range(self.epochs):
            if time.time() - startTime > self.maxTrainSeconds:
                break
            logits = states @ theta
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            grad = np.zeros_like(theta)
            total = 0.0
            for _ in range(self.samplesPerDay):
                actions = np.array([rng.choice(n_actions, p=p) for p in probs])
                rewards = utility[np.arange(len(states)), actions]
                # per-state baseline: the expected utility under the current policy
                baseline = (probs * utility).sum(axis=1)
                advantage = rewards - baseline
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(states)), actions] = 1.0
                grad += states.T @ ((onehot - probs) * advantage[:, None])
                total += rewards.mean()
            grad /= (self.samplesPerDay * len(states))
            norm = float(np.linalg.norm(grad))
            if norm > self.maxGradNorm:
                grad *= self.maxGradNorm / norm
            theta = theta + self.learningRate * grad
            epochMeans.append(float(total / self.samplesPerDay))
        return theta, epochMeans

    # --- persistence ------------------------------------------------------------
    def _policyPath(self, name):
        return os.path.join(self.modelPath, f"{name}_position_policy.json")

    def _loadPolicy(self, name):
        path = self._policyPath(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as infile:
                stored = json.load(infile)
            theta = np.array(stored.get("theta"), dtype=float)
            mean = np.array(stored.get("featureMean"), dtype=float)
            std = np.array(stored.get("featureStd"), dtype=float)
            if theta.shape != (len(FEATURES), len(SIZES)) or mean.shape != (len(FEATURES),) or std.shape != (len(FEATURES),) \
                    or not np.all(np.isfinite(theta)) or stored.get("sizes") != list(SIZES):
                print(f"RLPositionModel: stored policy for {name} does not fit this feature template - starting fresh")
                return None
            return {"theta": theta, "mean": mean, "std": std}
        except Exception as e:
            print(f"RLPositionModel: could not load the stored policy for {name}: {e}")
            return None

    def _savePolicy(self, name, theta, mean, std, report):
        try:
            os.makedirs(self.modelPath, exist_ok=True)
            with open(self._policyPath(name), "w") as outfile:
                json.dump({"game": name, "row": ROW_NAME, "features": FEATURES, "sizes": list(SIZES), "theta": theta.tolist(),
                           "featureMean": [float(v) for v in mean], "featureStd": [float(v) for v in std], "report": report}, outfile, indent=2)
        except Exception as e:
            print(f"RLPositionModel: failed to persist the policy for {name}: {e}")

    # --- the row -----------------------------------------------------------------
    def _voteTicket(self, rows, slots, modelScores):
        vote = helpers.build_positional_vote_predictions([r for r in rows if r.get("name") != ROW_NAME], slots, model_scores=modelScores)
        return [int(v) for v in vote["ticket"]] if vote and vote.get("ticket") else None

    def run(self, name, listOfDecodedPredictions, historyDir, dataPath, config=None):
        """
        (Re)trains the policy on the market's stored day JSONs, warm-started
        from the persisted one, and returns this row for the day the other
        rows predict:

            {"name": "RL Position Model", "predictions": [ticket], "positions": [size per instrument]}

        config: cutoffDate (datetime, a history rebuild), modelScores (the
        vote weights), slots (defaults to the rows' ticket length), returns
        ({date: [log return per slot]} - tests pass it; the pipeline reads
        the market's returns file). Never raises: any failure degrades to the
        plain rule (size 1 where the vote is up).
        """
        config = config or {}
        startTime = time.time()
        rows = [r for r in listOfDecodedPredictions if r.get("predictions") and r["predictions"][0]]
        slots = int(config.get("slots") or (len(rows[0]["predictions"][0]) if rows else 0))
        if not rows or slots <= 0:
            print(f"RLPositionModel: no rows with a ticket for {name} - no row")
            return None
        ticket = self._voteTicket(rows, slots, config.get("modelScores"))
        if ticket is None:
            print(f"RLPositionModel: the rows cannot vote a ticket for {name} - no row")
            return None
        plain = [1.0 if b >= self.k / 2 else 0.0 for b in ticket]
        try:
            rng = np.random.default_rng(self.seed)
            entries = self._loadHistory(historyDir, cutoffDate=config.get("cutoffDate"))
            returns = config.get("returns")
            if returns is None and entries:
                try:
                    returns = self._loadReturns(dataPath)
                except Exception as e:
                    print(f"RLPositionModel: no returns for {name} ({e}) - the stored policy decides, or the plain rule")
                    returns = {}
            # the training set: the newest trainDays scored days with a return
            trainable = [(d, r, real) for d, r, real in entries if returns and d in returns and len(returns[d]) >= slots]
            trainable = trainable[-self.trainDays:]
            stored = self._loadPolicy(name)
            theta = stored["theta"] if stored else np.zeros((len(FEATURES), len(SIZES)))
            raws, rets = [], []
            history = {d: real for d, _, real in entries}
            ordered = [d for d, _, _ in entries]
            for d, r, real in trainable:
                before = [history[x] for x in ordered if x < d][-RECENT:]
                yesterday = before[-1] if before else None
                raws.append(self._features(r, slots, yesterday, before))
                rets.append(returns[d][:slots])
            epochMeans = []
            if raws:
                raws = np.asarray(raws)
                mean = raws.reshape(-1, raws.shape[-1]).mean(axis=0)
                std = raws.reshape(-1, raws.shape[-1]).std(axis=0)
                std = np.where(std > 1e-9, std, 1.0)
                mean[-1], std[-1] = 0.0, 1.0
                phis = self._normalise(raws, mean, std)
                theta, epochMeans = self._train(theta, phis, np.asarray(rets, dtype=float), startTime, rng)
                if not np.all(np.isfinite(theta)):
                    print(f"RLPositionModel: non-finite theta after training for {name} - resetting")
                    theta = np.zeros_like(theta)
            elif stored:
                mean, std = stored["mean"], stored["std"]
                print(f"RLPositionModel: no trainable history for {name} - decoding with the stored policy")
            else:
                print(f"RLPositionModel: no trainable history and no stored policy for {name} - the plain rule (size 1 where the vote is up)")
                self.lastReport = {"fallback": "plain rule"}
                return {"name": ROW_NAME, "predictions": [ticket], "positions": plain}
            # today's state: the rows made for the coming day, the newest settled bins before it
            recent = [real for _, _, real in entries][-RECENT:]
            today = self._normalise(self._features(rows, slots, recent[-1] if recent else None, recent), mean, std)
            logits = today @ theta
            sizes = [float(SIZES[int(np.argmax(row))]) for row in logits]
            report = {"trainedAt": datetime.now().isoformat(timespec="seconds"), "warmStart": stored is not None, "trainingDays": len(raws),
                      "epochsRun": len(epochMeans), "epochsRequested": self.epochs, "elapsedSeconds": round(time.time() - startTime, 2),
                      "firstEpochMeanUtility": epochMeans[0] if epochMeans else None, "lastEpochMeanUtility": epochMeans[-1] if epochMeans else None,
                      "riskAversion": self.riskAversion, "sizesToday": sizes}
            self.lastReport = report
            if len(raws):
                self._savePolicy(name, theta, mean, std, report)
                if epochMeans:
                    print(f"RLPositionModel [{name}]: {len(raws)} days, {len(epochMeans)} epochs in {report['elapsedSeconds']}s, "
                          f"mean utility {epochMeans[0]:.4f} -> {epochMeans[-1]:.4f}, sizes today {sizes}")
            return {"name": ROW_NAME, "predictions": [ticket], "positions": sizes}
        except Exception as e:
            print(f"RLPositionModel: training/decoding failed for {name} ({e}) - the plain rule")
            self.lastReport = {"fallback": f"plain rule after error: {e}"}
            return {"name": ROW_NAME, "predictions": [ticket], "positions": plain}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check():
    import tempfile
    rng = np.random.default_rng(7)
    slots, days, k = 4, 160, MARKET_BINS
    dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(days)]
    returns, dayfiles = {}, {}
    for i, d in enumerate(dates):
        ret = rng.normal(0.0, 0.02, size=slots)
        returns[d] = [float(v) for v in ret]
        # a day's rows: six "informed" rows that lean the right way, four noise rows, GARCH, HMM and the vote
        rows = []
        for m in range(6):
            rows.append({"name": f"Informed {m}", "predictions": [[int(7 if r > 0 else 2) if rng.random() < 0.8 else int(rng.integers(0, k)) for r in ret]]})
        for m in range(4):
            rows.append({"name": f"Noise {m}", "predictions": [[int(v) for v in rng.integers(0, k, size=slots)]]})
        rows.append({"name": GARCH_ROW, "predictions": [[int(5 if r > 0 else 4) for r in ret]]})
        rows.append({"name": REGIME_ROW, "predictions": [[int(6 if r > 0 else 3) for r in ret]]})
        vote = helpers.build_positional_vote_predictions(rows, slots)
        rows.append({"name": VOTE_ROW, "predictions": [vote["ticket"]]})
        real = [min(k - 1, max(0, int(round(4.5 + r / 0.02 * 2.5)))) for r in ret]
        dayfiles[d] = {"realResult": real, "currentPrediction": rows}

    with tempfile.TemporaryDirectory() as root:
        historyDir = os.path.join(root, "database", "crypto")
        os.makedirs(historyDir)
        for d, data in dayfiles.items():
            y, m, dd = d.split("-")
            with open(os.path.join(historyDir, f"{int(y)}-{int(m)}-{int(dd)}.json"), "w") as handle:
                json.dump(data, handle)
        model = RLPositionModel()
        model.setModelPath(os.path.join(root, "models", "rl_position"))
        model.setSeed(3)
        model.setEpochs(60)
        model.setSamplesPerDay(16)
        model.setMaxTrainSeconds(120)

        # 1. features: an all-up day reads as up, yesterday and the wildness are read from the realised bins
        raw = model._features(dayfiles[dates[10]]["currentPrediction"], slots, dayfiles[dates[9]]["realResult"], [dayfiles[x]["realResult"] for x in dates[5:10]])
        assert raw.shape == (slots, len(FEATURES)) and np.all(raw[:, -1] == 1.0) and np.all((0 <= raw[:, 0]) & (raw[:, 0] <= 1))
        # 2. the plain rule without history or policy
        today = dayfiles[dates[-1]]["currentPrediction"]
        plain = model.run("crypto", today, os.path.join(root, "nowhere"), root, {"returns": returns})
        assert plain["name"] == ROW_NAME and len(plain["predictions"][0]) == slots and set(plain["positions"]) <= {0.0, 1.0}
        assert plain["positions"] == [1.0 if b >= k / 2 else 0.0 for b in plain["predictions"][0]]
        # 3. training: the newest day is predicted from the days before it (cutoff keeps it out of the training set)
        cutoff = datetime.strptime(dates[-1], "%Y-%m-%d")
        row = model.run("crypto", today, historyDir, root, {"returns": returns, "cutoffDate": cutoff})
        report = model.lastReport
        assert report["trainingDays"] == min(120, days - 1) and report["epochsRun"] > 0, report
        assert report["lastEpochMeanUtility"] > report["firstEpochMeanUtility"], (report["firstEpochMeanUtility"], report["lastEpochMeanUtility"])
        assert all(s in SIZES for s in row["positions"]) and len(row["positions"]) == slots
        # 4. the learned policy sizes up strong-consensus up days and sits out down days
        stored = model._loadPolicy("crypto")
        assert stored is not None
        up_sizes, down_sizes = [], []
        for d in dates[-40:-1]:
            prev = [dayfiles[x]["realResult"] for x in dates if x < d][-RECENT:]
            phi = model._normalise(model._features(dayfiles[d]["currentPrediction"], slots, prev[-1], prev), stored["mean"], stored["std"])
            chosen = [SIZES[int(np.argmax(l))] for l in phi @ stored["theta"]]
            for slot in range(slots):
                (up_sizes if returns[d][slot] > 0 else down_sizes).append(chosen[slot])
        assert np.mean(up_sizes) > np.mean(down_sizes) + 0.5, (np.mean(up_sizes), np.mean(down_sizes))
        assert np.mean(down_sizes) < 0.5, np.mean(down_sizes)
        # 5. decode is deterministic and the second run warm-starts
        again = model.run("crypto", today, historyDir, root, {"returns": returns, "cutoffDate": cutoff})
        assert model.lastReport["warmStart"] is True and len(again["positions"]) == slots
        # 6. the utility is concave in the size: for an uncertain call a middle size can win
        model.setRiskAversion(0.5)
        assert model._utility(2.0, 0.02) < 2 * model._utility(1.0, 0.02), "the quadratic term must bite"
        print(f"RLPositionModel self-check OK: {report['trainingDays']} synthetic days, utility {report['firstEpochMeanUtility']:.3f} -> "
              f"{report['lastEpochMeanUtility']:.3f} in {report['epochsRun']} epochs; mean size {np.mean(up_sizes):.2f} on up days, "
              f"{np.mean(down_sizes):.2f} on down days; plain rule without history; cutoff respected; warm start")


if __name__ == "__main__":
    _self_check()
