"""
The 'no better than the frequencies' label of a deep learning row (25 Oct
2026).

On a draw with nothing to learn, the network that minimises cross-entropy
ignores its input and outputs each position's class frequencies of its
training targets, and the argmax decode then serves the training split's
most common value per position: pick3's 604/504 from the Transformer, GNN,
Autoencoder and UnifiedLstmTcn rows, keno GNN = TCN Base, sharesweek GNN on
9909 every week (the 24 Oct investigation). Rows that memorise instead (pick3
LSTM Base, UnifiedLstmGruTcn) vary from day to day and do worse than the
frequencies on held-out draws. On the pages both look like a model at work.

check_predictions() reads one row on the validation windows it was early-
stopped on, head by head (the special head before combine_special_prediction
pads it into the main axis), summed over the heads the way Keras sums their
losses:

  valLoss       the row's cross-entropy: the data term of its val_loss,
                without the L2 penalty, so it compares with the next one
  baselineLoss  the cross-entropy of predicting each position's class
                frequencies of the training targets, label-smoothed the way
                the row's loss smooths its targets
  margin        how far below baselineLoss the validation windows' OWN
                frequencies (smoothed the same way) get on those windows
  verdict       'learns' when valLoss < baselineLoss - margin, else
                'counts-only' (also a row that does worse than the
                frequencies: it is no better than them either)
  sameTicket    the share of the validation windows whose argmax ticket is
                the most common one; inputBlind when that is 95% or more
  windows       the number of validation windows

Why that margin, and not zero or a fixed number: the frequencies drift
between the training years and the validation years (pick3's 2016-2024
frequencies score 2.3035 on its validation split, worse than uniform at
2.3026; the market bins drift with volatility), so an output that ignores
its input and only shrinks or tilts the frequencies already gets below the
baseline. The validation windows' own frequencies are the lowest loss any
input-blind output can reach on them - they are fitted to those very windows
- so a row below them has to be reading its input. How far below the
baseline they get depends on the game: on a fair draw about (K - 1) / 2n nats
per position with K classes and n windows for the hindsight fit, plus what
the training frequencies lose out of sample. On the production histories
(25 Oct 2026, window 20, smoothing 0.05) that is 0.004 for pick3, 0.014
jokerplus, 0.020 crypto, 0.025 keno, 0.029 shares, but 0.11-0.16 for lotto,
euromillions, cryptoweek and sharesweek (25-190 validation windows) - so one
fixed number would be too strict for one game and too loose for another.
On those small splits the bar is conservative: a row with a weak real signal
can still read 'counts-only' there, which is the side this label should err
on.

It is a label: nothing is dropped, hidden or reweighted on it (owner's
decision, 25 Oct 2026). Predictor.py stores it on the row as learningCheck,
and Helpers.generate_model_performance_report carries each row's newest one
into modelPerformance.json for the admin pages.

    python3 -m src.LearningCheck     # self-check on synthetic data
"""

import numpy as np

# Keras' backend epsilon: categorical_crossentropy clips the probabilities to
# [EPS, 1 - EPS] before the log, so a class the training frequencies never
# saw costs the baseline what it would cost the row's own loss.
EPS = 1e-7
# share of the validation windows on one argmax ticket from which a row
# counts as ignoring its input (owner's "~95%", 25 Oct 2026)
INPUT_BLIND_SHARE = 0.95


def class_frequencies(targets, num_classes):
    """(positions, num_classes): each position's share of every class in the
    integer targets (windows, positions)."""
    targets = np.asarray(targets, dtype=int)
    counts = [np.bincount(targets[:, p], minlength=num_classes)[:num_classes] for p in range(targets.shape[1])]
    return np.stack(counts) / float(len(targets))


def smoothed_frequencies(targets, num_classes, smoothing):
    """The best prediction that ignores the input under label-smoothed
    targets: the frequencies smoothed the way Keras smooths a one-hot
    target, y * (1 - s) + s / K."""
    return class_frequencies(targets, num_classes) * (1.0 - smoothing) + smoothing / num_classes


def cross_entropy(probs, targets, smoothing=0.0):
    """
    Keras' categorical_crossentropy for integer targets (windows, positions):
    the probabilities renormalised and clipped, the one-hot targets smoothed,
    the mean over windows and positions (Keras' sum_over_batch_size). probs
    is (windows, positions, K), or (positions, K) for one prediction served
    to every window.
    """
    targets = np.asarray(targets, dtype=int)
    probs = np.asarray(probs, dtype=float)
    num_classes = probs.shape[-1]
    probs = probs / probs.sum(axis=-1, keepdims=True)
    log_probs = np.log(np.clip(probs, EPS, 1.0 - EPS))
    smoothed = np.eye(num_classes)[targets] * (1.0 - smoothing) + smoothing / num_classes
    return float(-(smoothed * log_probs).sum(axis=-1).mean())


def check_predictions(heads):
    """
    The label from the row's validation predictions. heads holds one
    (probs, train_targets, val_targets, smoothing) per output head: probs the
    row's softmax on the validation windows (windows, positions, K), the
    targets as 0-based class indices (windows, positions) - run()'s y and
    y_val sliced to the head's positions - and the head's label smoothing.
    """
    val_loss = baseline = floor = 0.0
    tickets = []
    windows = 0
    for probs, train_targets, val_targets, smoothing in heads:
        probs = np.asarray(probs, dtype=float)
        val_targets = np.asarray(val_targets, dtype=int)
        smoothing = float(smoothing or 0.0)
        # finite weights can still overflow to a NaN/Inf output; its loss would be NaN, which json.dump writes as
        # a bare NaN into the day file and modelPerformance.json - invalid JSON for every Node reader of them
        if not np.all(np.isfinite(probs)):
            raise ValueError("the predictions hold NaN/Inf values")
        if probs.shape[:2] != val_targets.shape:
            raise ValueError(f"predictions {probs.shape} do not match the validation targets {val_targets.shape}")
        num_classes = probs.shape[-1]
        val_loss += cross_entropy(probs, val_targets, smoothing)
        baseline += cross_entropy(smoothed_frequencies(train_targets, num_classes, smoothing), val_targets, smoothing)
        floor += cross_entropy(smoothed_frequencies(val_targets, num_classes, smoothing), val_targets, smoothing)
        tickets.append(probs.argmax(axis=-1))
        windows = len(val_targets)
    if not windows:
        raise ValueError("no validation windows")

    _, counts = np.unique(np.concatenate(tickets, axis=1), axis=0, return_counts=True)
    same_ticket = counts.max() / float(windows)
    # never negative: the validation frequencies are the best input-blind
    # output on those windows, the training frequencies one of the others
    margin = max(0.0, baseline - floor)
    return {
        "valLoss": round(val_loss, 4),
        "baselineLoss": round(baseline, 4),
        "margin": round(margin, 4),
        "verdict": "learns" if val_loss < baseline - margin else "counts-only",
        "sameTicket": round(float(same_ticket), 3),
        "inputBlind": bool(same_ticket >= INPUT_BLIND_SHARE),
        "windows": int(windows),
    }


def _loss_smoothing(model, output_name):
    """The label smoothing of the row's compiled loss for one output (a dict
    of losses per output for the two-head rows, one loss otherwise)."""
    loss = getattr(model, "loss", None)
    if hasattr(loss, "keys"):
        loss = loss.get(output_name)
    return float(getattr(loss, "label_smoothing", 0.0) or 0.0)


def learning_check(model, X_val, train_targets, val_targets, main_digits, label=""):
    """
    The label for a trained Keras row: predicts its validation windows and
    scores them per head with check_predictions. train_targets/val_targets
    are the 0-based next draws the row was trained and validated on (run()'s
    y and y_val), main_digits the main head's positions; the smoothing comes
    from the row's compiled loss, so the baseline is smoothed exactly as the
    row's targets are. Returns None, with one log line, when it cannot be
    computed: the label must never cost the row its prediction.
    """
    try:
        train_targets = np.asarray(train_targets, dtype=int)
        val_targets = np.asarray(val_targets, dtype=int)
        if len(val_targets) == 0:
            raise ValueError("no validation windows")
        outputs = model.predict(X_val, verbose=0)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        heads = [(outputs[0], train_targets[:, :main_digits], val_targets[:, :main_digits],
                  _loss_smoothing(model, "main_output"))]
        if len(outputs) > 1:
            heads.append((outputs[1], train_targets[:, main_digits:], val_targets[:, main_digits:],
                          _loss_smoothing(model, "special_output")))
        result = check_predictions(heads)
        print(f"Learning check ({label}): val_loss {result['valLoss']:.4f} against {result['baselineLoss']:.4f} for the "
              f"training frequencies (margin {result['margin']:.4f}) - {result['verdict']}; the same argmax ticket on "
              f"{result['sameTicket']:.0%} of {result['windows']} validation windows"
              f"{' (input-blind)' if result['inputBlind'] else ''}")
        return result
    except Exception as e:
        print(f"Learning check ({label}) skipped: {e}")
        return None


if __name__ == "__main__":
    import json

    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    rng = np.random.default_rng(7)
    K, positions, train_n, val_n, s = 10, 3, 2800, 700, 0.05
    # a fair 3-digit draw: nothing in the window says anything about the next draw
    train = rng.integers(0, K, size=(train_n, positions))
    val = rng.integers(0, K, size=(val_n, positions))

    # 1. the collapsed row: the smoothed training frequencies for every window
    prior = smoothed_frequencies(train, K, s)
    collapsed = check_predictions([(np.broadcast_to(prior, (val_n, positions, K)), train, val, s)])
    check(abs(collapsed["valLoss"] - collapsed["baselineLoss"]) < 1e-6, f"the frequencies score the baseline: {collapsed}")
    check(collapsed["verdict"] == "counts-only" and collapsed["inputBlind"] and collapsed["sameTicket"] == 1.0,
          f"the collapsed row is counts-only and input-blind: {collapsed}")
    check(0 < collapsed["margin"] < 0.05, f"the hindsight margin is small on a fair draw: {collapsed['margin']}")

    # 2. an input-blind row that beats the training frequencies without reading anything: uniform, when the
    #    frequencies drifted (the pick3 case) - below the baseline, still counts-only
    drift_train = np.concatenate([np.zeros((400, positions), dtype=int), train])  # digit 0 over-represented in training
    uniform = np.full((val_n, positions, K), 1.0 / K)
    drifted = check_predictions([(uniform, drift_train, val, s)])
    check(drifted["valLoss"] < drifted["baselineLoss"] and drifted["verdict"] == "counts-only",
          f"beating drifted frequencies without the input is not learning: {drifted}")

    # 3. a row that reads its input: here the targets are visible to it (a learnable game), most of the mass on
    #    the drawn value - well below every input-blind output
    informative = np.full((val_n, positions, K), 0.3 / (K - 1))
    np.put_along_axis(informative, val[..., None], 0.7, axis=-1)
    learner = check_predictions([(informative, train, val, s)])
    check(learner["verdict"] == "learns" and not learner["inputBlind"], f"an informative row learns: {learner}")

    # 4. the memorising row: confident and wrong as often as chance - worse than the frequencies, counts-only
    noisy = rng.dirichlet(np.full(K, 0.2), size=(val_n, positions))
    memoriser = check_predictions([(noisy, train, val, s)])
    check(memoriser["valLoss"] > memoriser["baselineLoss"] and memoriser["verdict"] == "counts-only"
          and not memoriser["inputBlind"], f"a memorising row is counts-only and not input-blind: {memoriser}")

    # 5. two heads (Joker+: six digits + the 12-sign zodiac head) sum like Keras' two losses, each head against
    #    its own frequencies; the ticket covers both heads
    zodiac_train = rng.integers(0, 12, size=(train_n, 1))
    zodiac_val = rng.integers(0, 12, size=(val_n, 1))
    zprior = smoothed_frequencies(zodiac_train, 12, s)
    both = check_predictions([(np.broadcast_to(prior, (val_n, positions, K)), train, val, s),
                              (np.broadcast_to(zprior, (val_n, 1, 12)), zodiac_train, zodiac_val, s)])
    check(abs(both["baselineLoss"] - collapsed["baselineLoss"] - cross_entropy(zprior, zodiac_val, s)) < 1e-3,
          f"the heads' losses add up: {both} vs {collapsed}")
    check(both["inputBlind"] and both["verdict"] == "counts-only", f"a collapsed two-head row: {both}")
    flipped = np.broadcast_to(zprior, (val_n, 1, 12)).copy()
    flipped[::2, 0, :] = flipped[::2, 0, ::-1]  # the sign head changes its mind on every other window
    check(not check_predictions([(np.broadcast_to(prior, (val_n, positions, K)), train, val, s),
                                 (flipped, zodiac_train, zodiac_val, s)])["inputBlind"],
          "a head that follows the input breaks the shared ticket")

    # 6. Keras parity: smoothing and clipping as categorical_crossentropy does them (worked by hand)
    p = np.array([[[0.5, 0.5, 0.0]]])
    hand = -((1 - 0.1 + 0.1 / 3) * np.log(0.5) + 0.1 / 3 * np.log(0.5) + 0.1 / 3 * np.log(EPS))
    check(abs(cross_entropy(p, [[0]], 0.1) - hand) < 1e-9, f"smoothing and clipping: {cross_entropy(p, [[0]], 0.1)} vs {hand}")
    check(abs(cross_entropy(np.full((1, 1, 10), 0.1), [[3]]) - np.log(10)) < 1e-9, "uniform scores ln K")

    # 7. the compiled loss decides the smoothing; nothing computable -> None, never an exception
    class Loss:
        def __init__(self, label_smoothing):
            self.label_smoothing = label_smoothing

    class Model:
        def __init__(self, loss, outputs):
            self.loss, self.outputs = loss, outputs

        def predict(self, X, verbose=0):
            return self.outputs

    check(_loss_smoothing(Model({"main_output": Loss(0.06), "special_output": Loss(0.0)}, None), "main_output") == 0.06
          and _loss_smoothing(Model({"main_output": Loss(0.06), "special_output": Loss(0.0)}, None), "special_output") == 0.0
          and _loss_smoothing(Model(Loss(0.02), None), "main_output") == 0.02
          and _loss_smoothing(Model("categorical_crossentropy", None), "main_output") == 0.0,
          "the smoothing is read per output from the compiled loss")
    one_head = learning_check(Model(Loss(s), np.broadcast_to(prior, (val_n, positions, K))), None, train, val, positions, "synthetic")
    check(one_head == collapsed, f"learning_check on one head equals check_predictions: {one_head} vs {collapsed}")
    two_head = learning_check(Model({"main_output": Loss(s), "special_output": Loss(s)},
                                    [np.broadcast_to(prior, (val_n, positions, K)), np.broadcast_to(zprior, (val_n, 1, 12))]),
                              None, np.concatenate([train, zodiac_train], axis=1), np.concatenate([val, zodiac_val], axis=1),
                              positions, "synthetic two-head")
    check(two_head == both, f"learning_check splits the heads at main_digits: {two_head} vs {both}")
    check(learning_check(Model(Loss(s), np.zeros((0, positions, K))), None, train, val[:0], positions, "empty") is None,
          "no validation windows -> None")
    check(learning_check(Model(Loss(s), np.zeros((5, positions, K))), None, train, val, positions, "mismatch") is None,
          "a shape mismatch -> None")
    # a NaN output (finite weights that overflow) is no label: a NaN loss would make the day file invalid JSON
    overflow = np.broadcast_to(prior, (val_n, positions, K)).copy()
    overflow[3, 1, :] = np.nan
    check(learning_check(Model(Loss(s), overflow), None, train, val, positions, "non-finite") is None,
          "non-finite predictions -> None")
    json.dumps(collapsed)

    for message in failures:
        print("FAIL:", message)
    print(f"LearningCheck self-check: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    for title, result in (("collapsed", collapsed), ("drifted+uniform", drifted), ("informative", learner),
                          ("memorising", memoriser), ("two heads", both)):
        print(f"   {title}: {result}")
    raise SystemExit(1 if failures else 0)
