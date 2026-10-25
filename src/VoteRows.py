"""
Who votes in the positional vote rows, and who still counts in a weighted
vote (25 Oct 2026).

POSITIONAL_VOTE_DEFAULT_ROWS lived in Predictor.py, where only the
WeightedEnsemble Model read it. The RL Position Model (src/RLPositionModel.py)
promised "the WeightedEnsemble recipe" for its ticket but voted over every
other row - the stuck deep-learning tickets, the constant OrderStatistics
Baseline, three near-identical Regime HMM rows and the aggregates - and fed
the same rows to its sizing features (upShare, consensus). The constant lives
here so the model itself, and with it HyperoptRLTicket.py, which runs the
model directly, votes over the same rows as the served vote: importing
Predictor from src/ is not an option.

effective_members() answers "is this weighted vote an ensemble at all": a
member whose weight is STRICTLY greater than the others' together outvotes
them on every slot (and every number) they disagree on, so the vote is that
member's ticket under the ensemble's name - pick3's SubsetEnsemble Model
(LaplaceMonteCarlo + Markov) served Markov's ticket on 35 of 35 days, the
eurodreams one (LSTM Base + LaplaceMonteCarlo) Laplace's. At exact equality
the tie-break still decides, so that is not a copy. The vote weights are
modelScores min-max mapped to [1, 2] (Helpers._build_model_weight_lookup), so
in practice only a two-member weighted subset can collapse - on the market
games also a trio holding XGBoost and LightGBM, which vote with half their
weight each there (below).

Two market rules live here too (owner, 25 Oct 2026), so every place that
votes or scores a market vote applies them alike: XGBoost and LightGBM count
as one voter (MARKET_SHARED_VOTE_ROWS, positional_vote), and the
SubsetEnsemble Model weighs its members by HyperoptEnsemble's own per-row
scores (SUBSET_WEIGHTS_KEY, subset_vote_scores).

    python3 -m src.VoteRows         # self-check
"""

try:
    from src.Helpers import Helpers, is_market_game
except ImportError:  # imported from inside src/
    from Helpers import Helpers, is_market_game

helpers = Helpers()

# The untuned positional WeightedEnsemble Model (and the RL Position Model's
# ticket and features) vote per slot over the rows that own a model of their
# own only: the meta rows are aggregates of these and the DL research rows
# emit identical tickets day after day, so an all-rows vote double-counts.
# The tuned SubsetEnsemble Model may contain any row HyperoptEnsemble.py
# selected. Rows outside this set keep their own rows and their tracking -
# they just do not vote here.
POSITIONAL_VOTE_DEFAULT_ROWS = frozenset({"Markov Model", "MarkovMonteCarlo Model", "PoissonMonteCarlo Model",
                                          "LaplaceMonteCarlo Model", "XGBoost Model", "LightGBM Model", "CatBoost Model"})

# Rows that are one voter on the market games (owner, 25 Oct 2026): where two
# or more of a group vote together, each votes with 1/n of its weight, so the
# group weighs as one row. XGBoost and LightGBM run untuned on identical
# defaults there (HyperoptBoost.GAME_CONFIG covers the lotteries only; the
# market bestParams have no xgBoost*/lightGbm* key) and LightGBM's leaf cap is
# tied to 2**depth: the same depth-3 learner fitted twice on one feature
# matrix. They agree on 67-77% of market slots against 12-17% for independent
# rows, so every market vote counted one call twice - the WeightedEnsemble
# Model took their shared bin in 67-74% of the slots where they agreed. The
# lotteries tune each library on its own and keep two voters. Applied by
# positional_vote, member_weights and the RL Position Model's features.
MARKET_SHARED_VOTE_ROWS = (("XGBoost Model", "LightGBM Model"),)

# bestParams key of the SubsetEnsemble Model's own vote scores on the market
# games (owner, 25 Oct 2026), written by HyperoptEnsemble.py: each candidate
# row's own ticket scored by the tuner's objective over its evaluation window,
# mapped to [1, 2] like modelScores. On the markets modelScores hold only the
# four statistical rows - every other member weighed 1.0 and "weighted" was
# nearly the flat vote.
SUBSET_WEIGHTS_KEY = "subsetEnsembleWeights"


def vote_shares(names, game):
    """
    {name: the share of its weight the row votes with}: 1/n for each of n
    rows of one MARKET_SHARED_VOTE_ROWS group voting together on a market
    game, 1.0 for every other row and on every other game.
    """
    shares = {name: 1.0 for name in names}
    if not is_market_game(game):
        return shares
    for group in MARKET_SHARED_VOTE_ROWS:
        voting = [name for name in group if name in shares]
        if len(voting) > 1:
            for name in voting:
                shares[name] = 1.0 / len(voting)
    return shares


def positional_vote(rows, slot_count, model_scores=None, game=None):
    """
    The per-slot vote (Helpers.build_positional_vote_predictions) with the
    shares above - the one call through which the WeightedEnsemble and
    SubsetEnsemble rows (Predictor.py), the RL Position Model and
    HyperoptEnsemble.py vote, so serving and tuning count the same voters.
    """
    rows = list(rows or [])
    voting = [row.get("name") for row in rows if row.get("predictions") and row["predictions"][0]]
    return helpers.build_positional_vote_predictions(rows, slot_count, model_scores=model_scores, game=game,
                                                     shares=vote_shares(voting, game))


def voter_count(members, game=None):
    """How many voters the members are: a MARKET_SHARED_VOTE_ROWS group counts once on a market game."""
    members = list(dict.fromkeys(members or []))
    if not is_market_game(game):
        return len(members)
    grouped = sum(max(0, sum(name in members for name in group) - 1) for group in MARKET_SHARED_VOTE_ROWS)
    return len(members) - grouped


def subset_vote_scores(game, best_params, model_scores):
    """
    What the SubsetEnsemble Model's weighted vote is weighted with: on a
    market game the tuner's SUBSET_WEIGHTS_KEY once written, otherwise
    model_scores - the lotteries always, a market until the Saturday tuning
    after 25 Oct 2026 writes its weights.
    """
    stored = (best_params or {}).get(SUBSET_WEIGHTS_KEY)
    # a hand-edited or half-written value that is not a {row: weight} map is no weights: modelScores, not a crash
    return stored if is_market_game(game) and isinstance(stored, dict) and stored else model_scores


def member_weights(members, model_scores, game=None):
    """
    {member: vote weight} as the vote itself weighs them: the score lookup
    (model_scores None, a flat vote = 1.0 each) times the member's share
    (vote_shares - half each for XGBoost and LightGBM together on a market).
    """
    weight_for = helpers._build_model_weight_lookup(model_scores or {})
    shares = vote_shares(list(members), game)
    return {name: float(weight_for(name)) * shares[name] for name in members}


def effective_members(members, model_scores, game=None):
    """
    The members that can still change the vote's ticket: [the one member]
    when its weight is strictly greater than the sum of the others' weights,
    otherwise every member as given. model_scores is what the vote is
    weighted with - None for a flat vote, in which no member can dominate
    (the halved XGBoost and LightGBM only ever vote together, so they weigh
    1.0 between them). The weights are the vote's own (member_weights, with
    the market shares): together the pair weighs the mean of its two
    weights, so a market trio of XGBoost, LightGBM and a row weighted above
    that mean is that row's ticket, though the same trio on a lottery is
    not.

    Keno keeps every member: its 20-number ticket is the dominant member's,
    but the played 5-10 number sub-selections are drawn from the vote's own
    scores (Helpers.generate_subset_from_scores), where the numbers the other
    members agree on outrank the rest - an ensemble where the money is.
    """
    members = list(members or [])
    if len(members) < 2 or "keno" in str(game or "").lower():
        return members
    weights = member_weights(members, model_scores, game)
    total = sum(weights.values())
    for name in members:
        if weights[name] > total - weights[name]:
            return [name]
    return members


def weighted_is_flat(members, model_scores, game=None):
    """
    True when the weighted vote weighs every member as the flat vote does
    (every score weight 1.0: unscored rows, or scored at the minimum), so
    "weighted" is the flat vote; the market shares apply to both alike.
    """
    if not members:
        return False
    return member_weights(members, model_scores, game) == member_weights(members, None, game)


def _self_check():
    scores = {"Markov Model": 0.30, "LaplaceMonteCarlo Model": 0.10, "PoissonMonteCarlo Model": 0.20}
    pair = ["LaplaceMonteCarlo Model", "Markov Model"]
    # pick3's served pair: Markov 2.0 against Laplace 1.0 - Markov's ticket under the ensemble's name
    assert effective_members(pair, scores, "pick3") == ["Markov Model"]
    assert effective_members(pair, None, "pick3") == pair, "a flat vote has no dominant member"
    assert effective_members(pair, scores, "keno") == pair, "keno's played sub-selections still rank by agreement"
    # three members within [1, 2]: no one outweighs the other two
    trio = pair + ["PoissonMonteCarlo Model"]
    assert effective_members(trio, scores, "pick3") == trio
    # exact equality is not a copy: where the other two agree against it, the tie-break decides
    assert effective_members(["A Model", "B Model", "C Model"], {"A Model": 1.0, "B Model": 0.0, "C Model": 0.0}, "lotto") \
        == ["A Model", "B Model", "C Model"]
    assert effective_members(["Markov Model"], scores, "pick3") == ["Markov Model"] and effective_members([], scores) == []
    # the markets' served subsets: the members are unscored or scored at the minimum - weighted is flat
    market = {"PoissonMonteCarlo Model": 0.674, "Markov Model": 0.9}
    assert weighted_is_flat(["GNN Model", "PoissonMonteCarlo Model", "Transformer Model"], market)
    assert not weighted_is_flat(["GNN Model", "Markov Model"], market) and not weighted_is_flat([], market)
    assert "Markov Model" in POSITIONAL_VOTE_DEFAULT_ROWS and "OrderStatistics Baseline" not in POSITIONAL_VOTE_DEFAULT_ROWS
    # the vote's tie-break (Helpers.build_positional_vote_predictions): lottery ties go to the lowest value, market
    # ties to the tied bin nearest the middle of the bins, then the lower one
    rows = [{"name": "A Model", "predictions": [[1, 2, 4]]}, {"name": "B Model", "predictions": [[6, 7, 5]]}]
    assert helpers.build_positional_vote_predictions(rows, 3)["ticket"] == [1, 2, 4]
    assert helpers.build_positional_vote_predictions(rows, 3, game="pick3")["ticket"] == [1, 2, 4]
    assert helpers.build_positional_vote_predictions(rows, 3, game="cryptoweek")["ticket"] == [6, 2, 4], \
        helpers.build_positional_vote_predictions(rows, 3, game="cryptoweek")
    weighted = helpers.build_positional_vote_predictions(rows, 3, model_scores={"A Model": 1.0, "B Model": 0.0}, game="shares")
    assert weighted["ticket"] == [1, 2, 4], "a vote that is not tied keeps its winner"
    # XGBoost and LightGBM are one voter on a market (25 Oct 2026): half each when both vote, in full alone or elsewhere
    twins = ["XGBoost Model", "LightGBM Model"]
    assert vote_shares(twins + ["Markov Model"], "cryptoweek") == {"XGBoost Model": 0.5, "LightGBM Model": 0.5, "Markov Model": 1.0}
    assert set(vote_shares(twins, "pick3").values()) == {1.0} and vote_shares(["XGBoost Model", "Markov Model"], "crypto")["XGBoost Model"] == 1.0
    assert voter_count(twins, "shares") == 1 and voter_count(twins + ["Markov Model"], "shares") == 2 and voter_count(twins, "lotto") == 2
    # their [1, 7, 3] against Markov's [6, 2, 3] and Poisson's [8, 2, 0]: on pick3 the pair's two votes take slot 1;
    # on a market the pair weighs 1.0 there, a three-way tie that goes to 6, and on slot 3 its 3 with Markov's is 2.0
    # against 1.0 - the shares weigh in the confidence and the frequency too. Together the pair weighs what either
    # one alone does: the same vote without LightGBM.
    rows = [{"name": "XGBoost Model", "predictions": [[1, 7, 3]]}, {"name": "LightGBM Model", "predictions": [[1, 7, 3]]},
            {"name": "Markov Model", "predictions": [[6, 2, 3]]}, {"name": "PoissonMonteCarlo Model", "predictions": [[8, 2, 0]]}]
    assert positional_vote(rows, 3, game="pick3")["ticket"] == [1, 2, 3]
    market = positional_vote(rows, 3, game="crypto")
    assert market["ticket"] == [6, 2, 3] and market["frequency"][0] == {1: 1.0, 6: 1.0, 8: 1.0} and market["confidence"][2] == round(2 / 3, 3), market
    assert positional_vote(rows[:1] + rows[2:], 3, game="crypto") == market, "XGBoost alone votes in full"
    # ...and the weights effective_members reasons with are the vote's: under the same scores a market trio of the
    # twins and a row scored above both is that row's ticket (2.0 > 0.5 + 0.5), the lottery trio is not (2.0 < 1 + 1)
    trio_scores = {"Markov Model": 1.0, "XGBoost Model": 0.0, "LightGBM Model": 0.0}
    assert effective_members(twins + ["Markov Model"], trio_scores, "crypto") == ["Markov Model"]
    assert effective_members(twins + ["Markov Model"], trio_scores, "pick3") == twins + ["Markov Model"]
    assert effective_members(twins + ["Markov Model"], None, "crypto") == twins + ["Markov Model"], "a flat vote: 1.0 against 1.0"
    assert weighted_is_flat(twins + ["GNN Model"], {"Markov Model": 0.3}, "crypto"), "halved alike in both votes"
    # the SubsetEnsemble Model's weights: the tuner's on a market once written, modelScores otherwise
    stored = {"GNN Model": 2.0, "Markov Model": 1.0}
    assert subset_vote_scores("crypto", {SUBSET_WEIGHTS_KEY: stored}, scores) is stored
    assert subset_vote_scores("crypto", {}, scores) is scores and subset_vote_scores("lotto", {SUBSET_WEIGHTS_KEY: stored}, scores) is scores
    assert subset_vote_scores("crypto", {SUBSET_WEIGHTS_KEY: [2.0, 1.0]}, scores) is scores, "not a {row: weight} map"
    print("VoteRows self-check OK: a dominant member is a copy, flat and three-member votes are not, keno keeps its "
          "members, weighted = flat when every weight is 1.0, market ties go to the middle; on a market XGBoost and "
          "LightGBM are one voter and the SubsetEnsemble weights are the tuner's")


if __name__ == "__main__":
    _self_check()
