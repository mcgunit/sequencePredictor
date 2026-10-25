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
    upShare     share of the voters (below) whose bin is in the upper half (5-9)
    consensus   mean of (bin - 4.5) / 4.5 over the voters: -1 .. +1
                (both count XGBoost and LightGBM as one voter, half each, like the vote)
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

The row's TICKET - the bins the hit rates are scored on - is the
WeightedEnsemble Model's ticket, because this row predicts sizes, not bins:
the per-slot weighted vote over the same voters (POSITIONAL_VOTE_DEFAULT_ROWS
in src/VoteRows.py) by the same rule (VoteRows.positional_vote: the market
tie-break, XGBoost and LightGBM as one voter); its own contribution is
"positions": the size per instrument, which the settlement stores with the
prediction and the paper books apply (MarketSettle.position_of). Without a
trainable history and without a stored policy it falls back to the plain
rule: size 1 where the vote is up, 0 where it is not - a row no better and
no worse than the vote. Until 25 Oct 2026 the vote, upShare and consensus
ran over every other row: the stuck deep-learning tickets, the constant
OrderStatistics Baseline, the three near-identical Regime HMM rows and the
aggregates outvoted the voters (on sharesweek the row replayed GNN's 34-run
constant [9,9,0,9] on 9 runs), and the week games' sizes were taken from
those polluted features. Those rows keep their own rows; they just do not
vote here. XGBoost and LightGBM, one learner fitted twice on the markets,
count as one voter in the ticket and in upShare and consensus alike. A
policy saved under the old features is not loaded (FEATURE_VERSION): the
row starts a fresh one.

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
    from src.VoteRows import POSITIONAL_VOTE_DEFAULT_ROWS, positional_vote, vote_shares
except ImportError:  # imported from inside src/
    from Helpers import Helpers, MARKET_BINS
    from MarketModels import MarketHistory
    from MarketSettle import cash_pnl, STAKE, FEE_PER_LEG
    from VoteRows import POSITIONAL_VOTE_DEFAULT_ROWS, positional_vote, vote_shares

helpers = Helpers()

ROW_NAME = "RL Position Model"
SIZES = (0.0, 0.5, 1.0, 1.5, 2.0)
FEATURES = ["upShare", "consensus", "garch", "regime", "vote", "yesterday", "recentAbs", "bias"]
# What the features MEAN, stored with the policy: a policy saved under another
# version was fitted on other inputs and is not warm-started from. 1 (the
# unversioned files): upShare and consensus over every other row; 2 (25 Oct
# 2026): over the voters only, XGBoost and LightGBM as one voter (half each).
FEATURE_VERSION = 2
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
        # the gate's floor (HyperoptRLTicket.tune_position): when no sizing it scored beats a policy that never
        # bets - exactly 0 under the bound - the row is served sitting out, size 0 everywhere (13 Oct 2026)
        self.sitOut = False

    # --- setters (the Predictor configures the row from bestParams_<market>.json) ---
    def setModelPath(self, modelPath): self.modelPath = modelPath
    def setLearningRate(self, alpha): self.learningRate = float(alpha)
    def setEpochs(self, epochs): self.epochs = max(1, int(epochs))
    def setSamplesPerDay(self, samples): self.samplesPerDay = max(1, int(samples))
    def setTrainDays(self, days): self.trainDays = max(1, int(days))
    def setRiskAversion(self, value): self.riskAversion = max(0.0, float(value))
    def setMaxTrainSeconds(self, seconds): self.maxTrainSeconds = float(seconds)
    def setSeed(self, seed): self.seed = seed
    def setSitOut(self, sitOut): self.sitOut = bool(sitOut)

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
    def _features(self, rows, slots, yesterday, recent, game=None):
        """
        (slots, len(FEATURES)) raw features from the day's rows and the
        realised bins before it. upShare and consensus read the voters only
        (POSITIONAL_VOTE_DEFAULT_ROWS, like the ticket), each voter with its
        share of the game's vote (VoteRows.vote_shares: XGBoost and LightGBM
        half each on a market, 25 Oct 2026); garch, regime and vote read their
        own named rows.
        """
        k = self.k
        out = np.zeros((slots, len(FEATURES)))
        named = {}
        tickets = {}
        for row in rows:
            name = row.get("name")
            preds = row.get("predictions") or []
            if name == ROW_NAME or not preds or not preds[0] or len(preds[0]) < slots:
                continue
            try:
                ticket = [int(v) for v in preds[0][:slots]]
            except (TypeError, ValueError):
                continue
            if name in POSITIONAL_VOTE_DEFAULT_ROWS:
                tickets[name] = ticket
            named[name] = ticket
        shares = vote_shares(list(tickets), game)
        for slot in range(slots):
            bins = [t[slot] for t in tickets.values()]
            share = [shares[voter] for voter in tickets]
            up = float(np.average([b >= k / 2 for b in bins], weights=share)) if bins else 0.5
            consensus = float(np.average([_scaled(b, k) for b in bins], weights=share)) if bins else 0.0
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
            if stored.get("featureVersion") != FEATURE_VERSION:
                print(f"RLPositionModel: stored policy for {name} was fitted on feature version "
                      f"{stored.get('featureVersion', 1)}, not {FEATURE_VERSION} - starting fresh")
                return None
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
                json.dump({"game": name, "row": ROW_NAME, "features": FEATURES, "featureVersion": FEATURE_VERSION,
                           "sizes": list(SIZES), "theta": theta.tolist(),
                           "featureMean": [float(v) for v in mean], "featureStd": [float(v) for v in std], "report": report}, outfile, indent=2)
        except Exception as e:
            print(f"RLPositionModel: failed to persist the policy for {name}: {e}")

    # --- the row -----------------------------------------------------------------
    def _voteTicket(self, rows, slots, modelScores, game):
        """
        The WeightedEnsemble Model's ticket: the per-slot vote over the
        voters only, with the market tie-break and XGBoost and LightGBM as one
        voter (VoteRows.positional_vote, the call the served row makes) - the
        same rule wherever the model runs (Predictor.py, HyperoptRLTicket.py),
        so tuning and serving cannot drift apart. None when no voter has a
        ticket.
        """
        voters = [r for r in rows if r.get("name") in POSITIONAL_VOTE_DEFAULT_ROWS]
        vote = positional_vote(voters, slots, model_scores=modelScores, game=game)
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
        ticket = self._voteTicket(rows, slots, config.get("modelScores"), name)
        if ticket is None:
            print(f"RLPositionModel: the voters cannot vote a ticket for {name} - no row")
            return None
        plain = [1.0 if b >= self.k / 2 else 0.0 for b in ticket]
        if self.sitOut:
            # the Saturday gate found no sizing that beats not betting: the row still shows the vote's ticket (its
            # bins are scored like every row's), but takes no position; the stored policy is left as it is
            print(f"RLPositionModel [{name}]: sitting out - the tuning gate found no sizing above the 0 of never betting (rlPositionSitOut)")
            self.lastReport = {"sitOut": True}
            return {"name": ROW_NAME, "predictions": [ticket], "positions": [0.0] * len(ticket)}
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
                raws.append(self._features(r, slots, yesterday, before, name))
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
            today = self._normalise(self._features(rows, slots, recent[-1] if recent else None, recent, name), mean, std)
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
    voters = sorted(POSITIONAL_VOTE_DEFAULT_ROWS)
    # rows that are not voters: a bloc of five identical constant tickets (the stuck DL rows, the baseline, an
    # aggregate) that out-numbers the voters' agreement - an all-rows vote replayed such a bloc until 25 Oct 2026
    stuck = ["GNN Model", "Autoencoder Model", "LSTM Base Model", "OrderStatistics Baseline", "MetaLearner Model"]
    for i, d in enumerate(dates):
        ret = rng.normal(0.0, 0.02, size=slots)
        returns[d] = [float(v) for v in ret]
        # a day's rows: five "informed" voters that lean the right way, two noise voters, the stuck bloc, GARCH, HMM and the vote
        rows = []
        for name in voters[:5]:
            rows.append({"name": name, "predictions": [[int(7 if r > 0 else 2) if rng.random() < 0.8 else int(rng.integers(0, k)) for r in ret]]})
        for name in voters[5:]:
            rows.append({"name": name, "predictions": [[int(v) for v in rng.integers(0, k, size=slots)]]})
        for name in stuck:
            rows.append({"name": name, "predictions": [[0, 9, 0, 9]]})
        rows.append({"name": GARCH_ROW, "predictions": [[int(5 if r > 0 else 4) for r in ret]]})
        rows.append({"name": REGIME_ROW, "predictions": [[int(6 if r > 0 else 3) for r in ret]]})
        # the WeightedEnsemble Model as Predictor.py serves it: the voters, the market tie-break, XGBoost + LightGBM as one
        vote = positional_vote([r for r in rows if r["name"] in POSITIONAL_VOTE_DEFAULT_ROWS], slots, game="crypto")
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
        # 1b. only the voters vote (25 Oct 2026): the ticket is the WeightedEnsemble Model's on every day, the stuck bloc
        # moves neither the ticket nor upShare/consensus, and an all-rows vote would have replayed the bloc
        replayed = 0
        for d in dates:
            dayRows = dayfiles[d]["currentPrediction"]
            weTicket = next(r for r in dayRows if r["name"] == VOTE_ROW)["predictions"][0]
            assert model._voteTicket(dayRows, slots, None, "crypto") == weTicket, d
            allRows = helpers.build_positional_vote_predictions([r for r in dayRows if r["name"] != VOTE_ROW], slots, game="crypto")
            replayed += sum(a == b for a, b in zip(allRows["ticket"], [0, 9, 0, 9]))
        assert replayed > days * slots // 2, f"the stuck bloc should own most slots of an all-rows vote, it owned {replayed} of {days * slots}"
        dayRows = dayfiles[dates[10]]["currentPrediction"]
        context = (dayfiles[dates[9]]["realResult"], [dayfiles[x]["realResult"] for x in dates[5:10]])
        assert np.array_equal(model._features(dayRows, slots, *context, "crypto"),
                              model._features([r for r in dayRows if r["name"] not in stuck], slots, *context, "crypto"))
        # the market tie-break: 1 against 6 goes to 6 (nearer the middle), 2 against 7 to the lower; GNN does not vote
        tied = [{"name": "Markov Model", "predictions": [[1, 2]]}, {"name": "XGBoost Model", "predictions": [[6, 7]]}, {"name": "GNN Model", "predictions": [[1, 2]]}]
        assert model._voteTicket(tied, 2, None, "crypto") == [6, 2], model._voteTicket(tied, 2, None, "crypto")
        assert model._voteTicket([{"name": "GNN Model", "predictions": [[1, 2]]}], 2, None, "crypto") is None, "no voter, no ticket"
        # 1c. XGBoost and LightGBM are one voter on a market (25 Oct 2026): their [1, 7] against Markov's [6, 2] is a tie
        # per slot (half + half against one) that the tie-break settles, where it was 2 votes to 1 for their bins;
        # upShare counts them as one voter too - (0 + 0) x 0.5 + 1 out of 2, not 1 out of 3
        twins = [{"name": "XGBoost Model", "predictions": [[1, 7]]}, {"name": "LightGBM Model", "predictions": [[1, 7]]},
                 {"name": "Markov Model", "predictions": [[6, 2]]}]
        assert model._voteTicket(twins, 2, None, "crypto") == [6, 2], model._voteTicket(twins, 2, None, "crypto")
        assert helpers.build_positional_vote_predictions(twins, 2, game="crypto")["ticket"] == [1, 7]
        assert abs(model._features(twins, 2, None, [], "crypto")[0][0] - 0.5) < 1e-12
        assert abs(model._features(twins, 2, None, [], "pick3")[0][0] - 1 / 3) < 1e-12
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
            phi = model._normalise(model._features(dayfiles[d]["currentPrediction"], slots, prev[-1], prev, "crypto"), stored["mean"], stored["std"])
            chosen = [SIZES[int(np.argmax(l))] for l in phi @ stored["theta"]]
            for slot in range(slots):
                (up_sizes if returns[d][slot] > 0 else down_sizes).append(chosen[slot])
        assert np.mean(up_sizes) > np.mean(down_sizes) + 0.5, (np.mean(up_sizes), np.mean(down_sizes))
        assert np.mean(down_sizes) < 0.5, np.mean(down_sizes)
        # 5. decode is deterministic and the second run warm-starts
        again = model.run("crypto", today, historyDir, root, {"returns": returns, "cutoffDate": cutoff})
        assert model.lastReport["warmStart"] is True and len(again["positions"]) == slots
        # 5b. the gate's floor: a sit-out row keeps the vote's ticket and takes no position, and leaves the policy alone
        stamp = os.path.getmtime(os.path.join(model.modelPath, os.listdir(model.modelPath)[0]))
        model.setSitOut(True)
        idle = model.run("crypto", today, historyDir, root, {"returns": returns, "cutoffDate": cutoff})
        assert idle["positions"] == [0.0] * slots and idle["predictions"] == again["predictions"] and model.lastReport == {"sitOut": True}, idle
        assert os.path.getmtime(os.path.join(model.modelPath, os.listdir(model.modelPath)[0])) == stamp, "a sit-out run does not retrain"
        model.setSitOut(False)
        # 5c. a policy saved under the old features (no featureVersion: the files before 25 Oct 2026) is not
        # warm-started from - the run trains a fresh one and saves it under FEATURE_VERSION, nobody deletes a file
        path = model._policyPath("crypto")
        with open(path, "r") as handle:
            old = json.load(handle)
        assert old.get("featureVersion") == FEATURE_VERSION, old.keys()
        old.pop("featureVersion")
        with open(path, "w") as handle:
            json.dump(old, handle)
        assert model._loadPolicy("crypto") is None
        model.setEpochs(5)
        model.run("crypto", today, historyDir, root, {"returns": returns, "cutoffDate": cutoff})
        assert model.lastReport["warmStart"] is False and model._loadPolicy("crypto") is not None, model.lastReport
        # 6. the utility is concave in the size: for an uncertain call a middle size can win
        model.setRiskAversion(0.5)
        assert model._utility(2.0, 0.02) < 2 * model._utility(1.0, 0.02), "the quadratic term must bite"
        print(f"RLPositionModel self-check OK: {report['trainingDays']} synthetic days, utility {report['firstEpochMeanUtility']:.3f} -> "
              f"{report['lastEpochMeanUtility']:.3f} in {report['epochsRun']} epochs; mean size {np.mean(up_sizes):.2f} on up days, "
              f"{np.mean(down_sizes):.2f} on down days; plain rule without history; cutoff respected; warm start; "
              f"only the voters vote (an all-rows vote took the stuck bloc's bin in {replayed} of {days * slots} slots); "
              f"XGBoost and LightGBM are one voter; an old-feature policy starts fresh")


if __name__ == "__main__":
    _self_check()
