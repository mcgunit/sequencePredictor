"""
The market-specific rows (README roadmap item 4, phase M3): a GARCH row and
a Regime HMM row with its two ablations. They model the RETURNS a market
game's bins were cut from - <market>-returns.tsv next to the yearly game
files (src/MarketGame.write_returns_file) - and hand the game what every
other row hands it: one bin per instrument (run) and one {bin: probability}
per instrument (score_positions), so the Backtester, the meta-learners, the
controls and Predictor.py treat them like any statistical model.

  GARCH Model              a GARCH(1,1) per instrument, constant (or zero, or
                           AR(1)) mean, Student-t innovations by default; the
                           predictive distribution of the next return
                           integrated over the K quantile bins. Predicts
                           volatility, not direction: whatever bin accuracy
                           it shows above 1/K is volatility clustering, the
                           known predictable component of returns - the
                           reference every other row is read against.
  Regime HMM Model         a Gaussian hidden Markov model over the market's
                           instruments jointly (arXiv 2603.04441's machinery,
                           see the README): features per instrument - the
                           day's log return, the 60-day rolling volatility and
                           the 20-day mean return, both built from returns
                           strictly before the day - z-scored on the training
                           window, full covariances, refit on every prediction
                           day on an expanding window, the number of regimes
                           reselected on a cadence by a complexity-penalised
                           one-step-ahead predictive log-likelihood on a
                           validation slice. The next day's return per
                           instrument is the regime mixture of the return
                           coordinate's Gaussian, conditioned on the next
                           day's known features (yesterday's volatility and
                           momentum), integrated bin by bin - the mixture
                           shape is where the regime information lives.
  Regime HMM ZeroMean Model  the same model with the return means fixed at
                           zero and the return-feature covariance block at
                           zero: regimes carry variance only. If the full row
                           is not better than this one under a proper score,
                           the regimes add nothing beyond volatility.
  Regime HMM Single Model  one regime (K = 1): the homoskedastic model. With
                           the default features it is a linear regression of
                           the next return on volatility and momentum; with
                           returns only, the random walk with drift.

Tickets: GARCH names the most likely bin per instrument; the three Regime HMM
rows, since 25 Oct 2026, the bin that holds their mixture's predictive median
(_median_ticket says why). The probabilities - what the proper score judges -
are the same either way.

Templates: across daily refits a regime's label is arbitrary (label switching),
so when a state directory is set (Predictor.py's live run) each fitted regime
is matched to a persistent template by the closed-form 2-Wasserstein distance
between Gaussians (whitened by the market's return covariance), one-to-one
assignment, a distance threshold spawning a new template; the reading - which
template the model believes the market is in - is appended to a regime log
for the market page. The probabilities a row emits never depend on the
labels, so the Backtester needs no state.

No third-party model library: numpy and scipy only, the GARCH recursion as a
linear filter, the HMM's EM written out (`arch` and `hmmlearn` are not on
the pipeline interpreter, and hmmlearn's defaults - diagonal covariances,
ten EM iterations - would have changed the model silently).

    python3 -m src.MarketModels        # self-check (part of npm test)
"""

from __future__ import annotations

import bisect
import json
import math
import os
import time

import numpy as np
from scipy import linalg, optimize, signal, stats

try:
    from src.MarketGame import K_BINS, quantile_edges, read_game_csv, read_returns_file, representative_return
except ImportError:  # imported from within src/
    from MarketGame import K_BINS, quantile_edges, read_game_csv, read_returns_file, representative_return

GARCH_NAME = "GARCH Model"
HMM_NAME = "Regime HMM Model"
HMM_ZERO_MEAN_NAME = "Regime HMM ZeroMean Model"
HMM_SINGLE_NAME = "Regime HMM Single Model"
MARKET_MODEL_NAMES = [GARCH_NAME, HMM_NAME, HMM_ZERO_MEAN_NAME, HMM_SINGLE_NAME]
HMM_ROWS = {HMM_NAME: "full", HMM_ZERO_MEAN_NAME: "zero_mean", HMM_SINGLE_NAME: "single"}

TEMPLATE_FILE = "regime_templates.json"
REGIME_LOG_KEEP = 250
LOG_FLOOR = 1e-4     # a bin probability is never reported below this (a proper score needs a finite log)


# ---------------------------------------------------------------------------
# The history a row may see
# ---------------------------------------------------------------------------

class MarketHistory:
    """
    The returns behind the visible game days. The game files say which days
    exist (and skipRows hides the newest ones, as for every model); the
    returns file gives the aligned log returns up to and including the last
    visible day - the warm-up before the first game day included, since a
    volatility model wants all of it.
    """

    def __init__(self, dataPath):
        self.dataPath = dataPath
        self._stamp = None
        self._rows = None
        self._days = None
        self._matrix = None
        self._symbols = None

    def stamp(self):
        try:
            names = sorted(n for n in os.listdir(self.dataPath) if n.endswith(".csv") or n.endswith("-returns.tsv"))
        except OSError:
            return None
        return tuple((n, os.path.getmtime(os.path.join(self.dataPath, n))) for n in names)

    def load(self):
        stamp = self.stamp()
        if stamp is None:
            raise FileNotFoundError(f"{self.dataPath}: no such game folder")
        if stamp == self._stamp and self._rows is not None:
            return
        rows = read_game_csv(self.dataPath)
        found = read_returns_file(self.dataPath)
        if found is None:
            raise FileNotFoundError(f"{self.dataPath}: no *-returns.tsv next to the game files - the market rows model returns; "
                                    "MarketsDaily.py --no-fetch writes it from the store")
        days, matrix, symbols = found
        if matrix.ndim != 2 or len(days) != len(matrix):
            raise ValueError(f"{self.dataPath}: the returns file is malformed")
        self._rows, self._days, self._matrix, self._symbols, self._stamp = rows, list(days), np.asarray(matrix, dtype=float), list(symbols), stamp

    @property
    def symbols(self):
        self.load()
        return list(self._symbols)

    def visible(self, skipRows=0):
        """(days, returns) up to the last visible game day, oldest first."""
        self.load()
        rows = self._rows[:len(self._rows) - int(skipRows)] if skipRows and skipRows > 0 else self._rows
        if not rows:
            raise ValueError(f"{self.dataPath}: no game day left to predict from (skipRows={skipRows})")
        last = rows[-1][0]
        n = bisect.bisect_right(self._days, last)
        if n == 0 or self._days[n - 1] != last:
            raise ValueError(f"{self.dataPath}: the returns file has no row for game day {last}")
        return self._days[:n], self._matrix[:n]


def edges_for_next_day(history, k=K_BINS):
    """The edges the next game day will be cut with: the quantiles of every return so far (MarketGame.cut_game's rule)."""
    return [quantile_edges(history[:, pos], k) for pos in range(history.shape[1])]


# ---------------------------------------------------------------------------
# From a predictive distribution to bin probabilities
# ---------------------------------------------------------------------------

def mixture_bin_probabilities(edges, weights, locs, scales, df=None):
    """
    Probabilities of the K bins under a mixture of Gaussians (df None) or of
    scaled Student-t laws (df per component or a scalar). `edges` are the
    K-1 inner edges; component c has weight, location and scale.
    """
    edges = np.asarray(edges, dtype=float)
    weights = np.asarray(weights, dtype=float)
    locs = np.asarray(locs, dtype=float)
    scales = np.maximum(np.asarray(scales, dtype=float), 1e-12)
    z = (edges[None, :] - locs[:, None]) / scales[:, None]
    if df is None:
        cdf = stats.norm.cdf(z)
    else:
        dfs = np.broadcast_to(np.asarray(df, dtype=float), locs.shape)
        cdf = stats.t.cdf(z, dfs[:, None])
    cdf = np.concatenate([np.zeros((len(locs), 1)), cdf, np.ones((len(locs), 1))], axis=1)
    per_component = np.diff(cdf, axis=1)
    probs = (weights[:, None] * per_component).sum(axis=0)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    return probs / total if total > 0 else np.full(len(edges) + 1, 1.0 / (len(edges) + 1))


def _ticket(probabilities):
    return [int(np.argmax(row)) for row in probabilities]


def _median_ticket(probabilities):
    """
    Per instrument the bin that holds the predictive median: the first bin
    whose cumulative probability reaches one half. The Regime HMM rows'
    ticket since 25 Oct 2026 (the owner's item 2); GARCH keeps _ticket. The
    bins are quantiles of fat-tailed returns - narrow in the middle, wide
    outside - so a forecast near zero mean puts its most mass in a bin away
    from the middle (mostly 1 or 8, 0-2 or 7-9 as its spread varies), and
    which side won was set by the asymmetry of the edges, not by the model:
    the ZeroMean row, whose mean is exactly zero, went long on BTC, ETH and
    XRP every crypto day and never in shares. The median moves with the forecast's location, so the
    upper-half call the fixed rule trades is the model's own: its predictive
    median above the instrument's historical median return.
    """
    out = []
    for row in probabilities:
        cumulative = np.cumsum(np.asarray(row, dtype=float))
        out.append(int(min(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"), len(cumulative) - 1)))
    return out


def _slots(probabilities):
    return [{int(b): float(p) for b, p in enumerate(row)} for row in probabilities]


def _pooled(probabilities):
    """
    The flat {bin: score} every statistical row also offers (score_numbers):
    the mean probability of each bin over the instruments. A market is
    positional, so nothing trains on this - but the Backtester collects
    per-slot scores only from models that have score_numbers at all.
    """
    pooled = np.asarray(probabilities, dtype=float).mean(axis=0)
    return {int(b): float(p) for b, p in enumerate(pooled)}


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

def garch_variances(e2, omega, alpha, beta, h0):
    """
    h[t] = omega + alpha * e2[t-1] + beta * h[t-1] for t = 1..T as one linear
    filter: returns T values, the variance the model assigns to the return
    AFTER each e2 entry - h[0] follows e2[0] (with h0 before it), h[T-1] is
    the one-step-ahead forecast after the last return.
    """
    x = omega + alpha * np.asarray(e2, dtype=float)
    zi = signal.lfiltic([1.0], [1.0, -beta], y=[h0], x=[0.0])
    h, _ = signal.lfilter([1.0], [1.0, -beta], x, zi=zi)
    return h


def fit_garch(returns, mean="constant", dist="t"):
    """
    Maximum likelihood GARCH(1,1) on one return series (in percent - the
    optimiser likes that scale). mean: "zero" | "constant" | "ar1"; dist:
    "normal" | "t" (standardised Student-t with estimated degrees of
    freedom). Returns a dict with the parameters, the in-sample log
    likelihood, and the one-step-ahead forecast (loc, scale, df) in the same
    units; falls back to the unconditional law when the optimiser fails, and
    says so in "converged".
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 30:
        raise ValueError("GARCH needs at least 30 returns")
    if mean not in ("zero", "constant", "ar1") or dist not in ("normal", "t"):
        raise ValueError(f"unknown mean {mean!r} or dist {dist!r}")
    lag = r[:-1] if mean == "ar1" else None
    y = r[1:] if mean == "ar1" else r
    var = float(np.var(y)) or 1e-8

    def unpack(theta):
        i = 0
        mu = phi = 0.0
        if mean in ("constant", "ar1"):
            mu = theta[i]; i += 1
        if mean == "ar1":
            phi = theta[i]; i += 1
        omega, alpha, beta = theta[i], theta[i + 1], theta[i + 2]
        nu = theta[i + 3] if dist == "t" else None
        return mu, phi, omega, alpha, beta, nu

    def residuals(mu, phi):
        return y - mu - (phi * lag if mean == "ar1" else 0.0)

    def nll(theta):
        mu, phi, omega, alpha, beta, nu = unpack(theta)
        if alpha + beta >= 0.9995:
            return 1e12 * (alpha + beta)
        e = residuals(mu, phi)
        e2 = e * e
        h = garch_variances(e2, omega, alpha, beta, h0=e2.mean())
        h_in, e_in = h[:-1], e[1:]
        if not np.all(np.isfinite(h_in)) or h_in.min() <= 0:
            return 1e12
        if dist == "t":
            s = np.sqrt(h_in * (nu - 2.0) / nu)
            ll = stats.t.logpdf(e_in / s, nu) - np.log(s)
        else:
            ll = -0.5 * (np.log(2 * np.pi * h_in) + e_in * e_in / h_in)
        return -float(ll.sum())

    bounds, starts = [], []
    base = []
    if mean in ("constant", "ar1"):
        bounds.append((-10.0, 10.0)); base.append(float(np.mean(y)))
    if mean == "ar1":
        bounds.append((-0.99, 0.99)); base.append(0.0)
    bounds += [(1e-8, 10.0 * var), (0.0, 0.999), (0.0, 0.999)]
    if dist == "t":
        bounds.append((2.1, 200.0))
    for alpha0, beta0 in ((0.05, 0.90), (0.10, 0.80)):
        theta = list(base) + [var * (1 - alpha0 - beta0), alpha0, beta0] + ([6.0] if dist == "t" else [])
        starts.append(np.asarray(theta, dtype=float))

    best = None
    for x0 in starts:
        try:
            res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        except (ValueError, FloatingPointError):
            continue
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res
    converged = best is not None and best.fun < 1e11
    if not converged:
        mu = float(np.mean(y)) if mean != "zero" else 0.0
        return {"mean": mean, "dist": dist, "mu": mu, "phi": 0.0, "omega": var, "alpha": 0.0, "beta": 0.0,
                "nu": None, "loglik": None, "converged": False, "n": len(y),
                "forecast": {"loc": mu, "scale": math.sqrt(var), "df": None}}
    mu, phi, omega, alpha, beta, nu = unpack(best.x)
    e = residuals(mu, phi)
    e2 = e * e
    h = garch_variances(e2, omega, alpha, beta, h0=e2.mean())
    h_next = float(h[-1])
    loc = mu + (phi * r[-1] if mean == "ar1" else 0.0)
    if dist == "t":
        scale, df = math.sqrt(h_next * (nu - 2.0) / nu), float(nu)
    else:
        scale, df = math.sqrt(h_next), None
    return {"mean": mean, "dist": dist, "mu": float(mu), "phi": float(phi), "omega": float(omega), "alpha": float(alpha),
            "beta": float(beta), "nu": None if nu is None else float(nu), "loglik": -float(best.fun), "converged": True,
            "n": len(y), "persistence": float(alpha + beta), "h_next": h_next,
            "unconditional_variance": float(omega / (1 - alpha - beta)) if alpha + beta < 1 else None,
            "forecast": {"loc": float(loc), "scale": float(scale), "df": df}}


class GarchModel:
    """The GARCH Model row: one GARCH(1,1) per instrument on the returns file, bins from the predictive law."""

    def __init__(self, mean="constant", dist="t", window=2000, k=K_BINS):
        self.mean = mean
        self.dist = dist
        self.window = int(window)
        self.k = int(k)
        self.dataPath = ""
        self.history = None
        self._last = None
        self.last_fits = None

    def setDataPath(self, dataPath):
        self.dataPath = dataPath
        self.history = MarketHistory(dataPath)
        self._last = None

    def setSortedPrediction(self, use):
        pass   # a market ticket is positional by nature

    def setMean(self, mean): self.mean = str(mean)
    def setDist(self, dist): self.dist = str(dist)
    def setWindow(self, window): self.window = int(window)
    def clear(self): self._last = None

    def forecast(self, history):
        """Per instrument the fitted model and its one-step-ahead law, in log-return units."""
        fits = []
        for pos in range(history.shape[1]):
            series = history[:, pos] * 100.0
            if self.window > 0 and len(series) > self.window:
                series = series[-self.window:]
            fit = fit_garch(series, mean=self.mean, dist=self.dist)
            fit["forecast"] = {"loc": fit["forecast"]["loc"] / 100.0, "scale": fit["forecast"]["scale"] / 100.0, "df": fit["forecast"]["df"]}
            fits.append(fit)
        return fits

    def bin_probabilities(self, history, edges=None):
        edges = edges or edges_for_next_day(history, self.k)
        fits = self.forecast(history)
        self.last_fits = fits
        probs = []
        for pos, fit in enumerate(fits):
            f = fit["forecast"]
            probs.append(mixture_bin_probabilities(edges[pos], [1.0], [f["loc"]], [f["scale"]], df=None if f["df"] is None else [f["df"]]))
        return np.asarray(probs)

    def _probabilities(self, skipRows):
        if self.history is None:
            raise ValueError("setDataPath first")
        key = (self.history.stamp(), int(skipRows), self.mean, self.dist, self.window)
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        _, history = self.history.visible(skipRows)
        probs = self.bin_probabilities(history)
        self._last = (key, probs)
        return probs

    def run(self, generateSubsets=None, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        return _ticket(self._probabilities(skipRows)), {}

    def score_positions(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        return _slots(self._probabilities(skipRows))

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        return _pooled(self._probabilities(skipRows))


# ---------------------------------------------------------------------------
# A Gaussian hidden Markov model, EM written out
# ---------------------------------------------------------------------------

# The smallest normal double. A scaled forward/backward normaliser below it has
# lost the step to underflow (0, or a denormal with too few bits left), and the
# pass is redone in log space - see GaussianHMM._forward.
_TINY = np.finfo(float).tiny


def _logsumexp(a, axis=None, keepdims=False):
    """
    log(sum(exp(a))) along `axis`, shifted by the slice maximum so nothing
    over- or underflows; a slice that is all -inf (no state can get there)
    gives -inf, not NaN. numpy only: scipy.special.logsumexp costs ~90 us a
    call, five times this, and the log-space passes call it twice a step.
    """
    m = np.max(a, axis=axis, keepdims=True)
    m[~np.isfinite(m)] = 0.0
    with np.errstate(divide="ignore"):
        out = np.log(np.exp(a - m).sum(axis=axis, keepdims=True)) + m
    if keepdims:
        return out
    return float(out.ravel()[0]) if axis is None else out.squeeze(axis=axis)


def _log_gaussian(X, mean, cov):
    """log N(x; mean, cov) for every row of X; jitter added until the Cholesky factor exists."""
    d = X.shape[1]
    jitter = 0.0
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(d))
            break
        except np.linalg.LinAlgError:
            jitter = 1e-8 if jitter == 0.0 else jitter * 100
    else:
        raise np.linalg.LinAlgError("covariance not positive definite")
    z = linalg.solve_triangular(L, (X - mean).T, lower=True)
    maha = (z * z).sum(axis=0)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * math.log(2 * math.pi) + logdet + maha)


# A Dirichlet prior of one pseudo-transition per cell of the transition matrix
# (30 Oct 2026, owner's decision on the 26 Oct reading): EM's maximum
# likelihood keeps a transition it never saw at exactly 0, so K=4-5 fits on
# the ~100 rows of a week game ended with absorbing regimes - a regime the
# chain can enter but never leave, the root of the 0/0 the log-space passes
# now survive. The M-step takes the posterior mode instead: (expected
# transitions + 1) / (row total + K). On a daily game's ~1000 rows it moves a
# probability by about 1/rows; on a week game's 100 it keeps every exit open.
TRANSITION_PSEUDOCOUNT = 1.0


class GaussianHMM:
    """
    Full-covariance Gaussian HMM fitted by EM (Baum-Welch) with scaled
    forward-backward passes. Optional constraints, applied exactly in the
    M-step: `zero_mean_dims` keeps those coordinates' means at zero and
    `block` = (dims_a, dims_b) zeroes the covariance between the two groups.

    A scaled pass that underflows is redone in log space (26 Oct 2026). The
    emissions are scaled by the row's best state before exp, and a scaled
    posterior holds nothing below the smallest normal double (~1e-308): it
    becomes exactly 0. On a short window in many dimensions EM can end with
    an absorbing regime while the other regimes' filtered probabilities sit
    hundreds of nats below it - in the 10 Oct cryptoweek tuning study (trial
    6, K=4) at -891 to -1058 nats when the training slice ended - so
    select_regimes' validation pass started one-hot on the absorbing regime,
    and the first week 750-1215 nats less likely under it than under the
    best regime summed the scaled step to 0: 0/0, then math.log(0), and the
    ValueError dropped K=4 from the comparison instead of scoring it. Two of
    the 18 Oct week-game null controls hit the same; in a forecast's own fit
    it would have errored the day out of the tuner's objective. In log space
    the step's normaliser is always finite. _forward (the filter and the
    validation score) also redoes a pass that flushed a state the chain can
    reach (_flushed), and select_regimes scores the validation slice inside
    one pass, so nothing is cut off at the boundary (review, 26 Oct 2026):
    trial 6's K=4 scores -4047 nats there, not -79627. Ordinary inputs never
    reach the log-space path and stay bit-identical; the passes that do on
    the served fits agree with the scaled numbers to ~1e-12.
    """

    def __init__(self, n_states, n_iter=100, tol=1e-5, min_covar=1e-4, shrinkage=0.1, zero_mean_dims=None, block=None, seed=0,
                 transition_prior=TRANSITION_PSEUDOCOUNT):
        self.K = int(n_states)
        self.transition_prior = float(transition_prior)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.min_covar = float(min_covar)
        self.shrinkage = float(shrinkage)
        self.zero_mean_dims = list(zero_mean_dims) if zero_mean_dims else []
        self.block = block
        self.seed = int(seed)
        self.startprob = self.transmat = self.means = self.covars = None
        self.loglik = None
        self.iterations = 0
        self.converged = False

    # --- pieces --------------------------------------------------------------
    def _emission(self, X):
        return np.column_stack([_log_gaussian(X, self.means[k], self.covars[k]) for k in range(self.K)])

    def _forward(self, logB, start=None):
        """Forward pass: (filtered posteriors T x K, per-step log normalisers) - scaled, or in log space where scaling underflows or flushed a reachable state."""
        scaled = self._forward_scaled(logB, start)
        if scaled is not None and not self._flushed(scaled[0], self.startprob if start is None else start):
            return scaled
        log_alpha, logc = self._forward_log(logB, start)
        return np.exp(log_alpha), logc

    def _flushed(self, alpha, start):
        """
        True when a scaled forward pass flushed a state the chain can reach:
        one the start's support leads to through positive transitions - its
        filtered probability is positive however small - came out below the
        smallest normal double. The pass has then lost it for good, and rows
        that later favour it go unnoticed while every normaliser stays a
        normal double: the 10 Oct trial 6 window (skipRows 35-54, K=4) scored
        its validation slice 4539 nats too low that way, a regime at ~1e-385
        after the 1e-144 transition into it, then favoured by ~120 nats a
        week (review, 26 Oct 2026). Only _forward checks it: in EM's E-step
        the same flush fires on 2-5% of the served fits' E-steps (63% on
        trial 6), was harmless in every one sampled, and would send them all
        to the slower log-space pass.
        """
        small = alpha < _TINY
        if not small.any():
            return False
        step = self.transmat > 0
        support = np.asarray(start, dtype=float) > 0
        for t in range(len(alpha)):
            if t:
                support = support @ step
            if (small[t] & support).any():
                return True
        return False

    def _forward_scaled(self, logB, start=None):
        """The scaled forward pass; None as soon as a step's normaliser underflows (the caller redoes the pass in log space)."""
        T, K = logB.shape
        shift = logB.max(axis=1)
        B = np.exp(logB - shift[:, None])
        alpha = np.empty((T, K))
        logc = np.empty(T)
        a = (self.startprob if start is None else start) * B[0]
        c = a.sum()
        if not c >= _TINY:       # also catches NaN
            return None
        alpha[0] = a / c
        logc[0] = math.log(c) + shift[0]
        A = self.transmat
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c = a.sum()
            if not c >= _TINY:
                return None
            alpha[t] = a / c
            logc[t] = math.log(c) + shift[t]
        return alpha, logc

    def _forward_log(self, logB, start=None):
        """
        The forward pass in log space: (log filtered posteriors T x K, per-step
        log normalisers), the normaliser logsumexp(log(alpha[t-1] @ A) + logB[t])
        done per state so a regime with a posterior far below double range
        keeps it. Finite whenever the emissions are: the predicted state
        distribution sums to one, so some state always carries mass. A
        non-finite emission or an empty start raises ValueError (what a
        failed fit raises), never NaN.
        """
        T, K = logB.shape
        with np.errstate(divide="ignore"):
            logA = np.log(self.transmat)
            log_pred = np.log(np.asarray(self.startprob if start is None else start, dtype=float))
        log_alpha = np.empty((T, K))
        logc = np.empty(T)
        for t in range(T):
            if t:
                log_pred = _logsumexp(log_alpha[t - 1][:, None] + logA, axis=0)
            la = log_pred + logB[t]
            logc[t] = _logsumexp(la)
            if not math.isfinite(logc[t]):
                raise ValueError(f"HMM forward pass: no state can emit row {t} (non-finite emission or start)")
            log_alpha[t] = la - logc[t]
        return log_alpha, logc

    def _backward_scaled(self, logB):
        """The scaled backward pass: (beta T x K, rows normalised; the shifted emissions B); beta None when a step underflows."""
        T, K = logB.shape
        B = np.exp(logB - logB.max(axis=1)[:, None])
        beta = np.empty((T, K))
        beta[-1] = 1.0
        A = self.transmat
        for t in range(T - 2, -1, -1):
            b = A @ (B[t + 1] * beta[t + 1])
            s = b.sum()
            if not s >= _TINY:
                return None, B
            beta[t] = b / s
        return beta, B

    def _backward_log(self, logB):
        """
        The backward pass in log space, each row normalised like the scaled
        one's. Finite everywhere whenever the emissions are: the last row is
        log 1 = 0 and every row of the transition matrix sums to one, so
        each state reaches some state with a finite log beta at every step.
        """
        T, K = logB.shape
        with np.errstate(divide="ignore"):
            logA = np.log(self.transmat)
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            lb = _logsumexp(logA + (logB[t + 1] + log_beta[t + 1])[None, :], axis=1)
            log_beta[t] = lb - _logsumexp(lb)
        return log_beta

    def _expectations(self, logB):
        """
        The E-step: (state posteriors T x K, expected transition counts K x K -
        None for one state -, log likelihood). The scaled passes, exactly as
        before 26 Oct 2026, whenever every normaliser they divide by is a
        normal double and the counts come out finite; otherwise the same
        three quantities from the log-space passes (_expectations_log).
        """
        scaled = self._forward_scaled(logB)
        if scaled is None:
            return self._expectations_log(logB)
        alpha, logc = scaled
        beta, B = self._backward_scaled(logB)
        if beta is None:
            return self._expectations_log(logB)
        gamma = alpha * beta
        occupancy = gamma.sum(axis=1, keepdims=True)
        if not occupancy.min() >= _TINY:
            return self._expectations_log(logB)
        gamma /= occupancy
        xi = None
        if self.K > 1:
            w = B[1:] * beta[1:]
            norm = np.einsum("ti,ij,tj->t", alpha[:-1], self.transmat, w)
            if not norm.min() >= _TINY:
                return self._expectations_log(logB)
            with np.errstate(over="ignore", invalid="ignore"):
                xi = self.transmat * ((alpha[:-1] / norm[:, None]).T @ w)
            if not np.isfinite(xi).all():     # T terms of up to 1/norm each can still overflow
                return self._expectations_log(logB)
        return gamma, xi, float(logc.sum())

    def _expectations_log(self, logB):
        """The E-step from the log-space passes: posteriors and transition counts normalised per step by logsumexp."""
        log_alpha, logc = self._forward_log(logB)
        log_beta = self._backward_log(logB)
        log_gamma = log_alpha + log_beta
        gamma = np.exp(log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True))
        xi = None
        if self.K > 1:
            with np.errstate(divide="ignore"):
                logA = np.log(self.transmat)
            log_xi = log_alpha[:-1, :, None] + logA[None, :, :] + (logB[1:] + log_beta[1:])[:, None, :]
            log_xi -= _logsumexp(log_xi, axis=(1, 2), keepdims=True)
            xi = np.exp(log_xi).sum(axis=0)
        return gamma, xi, float(logc.sum())

    def _constrain(self):
        if self.zero_mean_dims:
            self.means[:, self.zero_mean_dims] = 0.0
        if self.block:
            a, b = self.block
            for k in range(self.K):
                self.covars[k][np.ix_(a, b)] = 0.0
                self.covars[k][np.ix_(b, a)] = 0.0

    def _variance_only(self, d):
        """Every coordinate's mean held at zero: K > 1 states can then differ in their covariances only."""
        return self.K > 1 and len(set(self.zero_mean_dims)) >= d

    def _spread(self, rng, size):
        """Covariance scale factors calm-to-turbulent around 1 (0.5 .. 2), jittered by this restart's rng."""
        return np.geomspace(0.5, 2.0, size) * np.exp(rng.uniform(-0.2, 0.2, size))

    def _init(self, X, rng):
        T, d = X.shape
        idx = rng.choice(T, size=self.K, replace=False) if T >= self.K else rng.integers(0, T, size=self.K)
        self.means = X[idx].astype(float).copy()
        global_cov = np.cov(X, rowvar=False) if T > 1 else np.eye(d)
        global_cov = np.atleast_2d(global_cov) + self.min_covar * np.eye(d)
        self.covars = np.array([global_cov.copy() for _ in range(self.K)])
        if self._variance_only(d):
            # With every mean held at zero - the ZeroMean row once
            # regimeHmmFeatures is 'returns' only (tuned in on 10 Oct 2026) -
            # the random starting means are zeroed below and every state
            # would start as the same N(0, global_cov): the E-step splits each
            # row evenly, the M-step rebuilds identical states, and EM never
            # leaves that symmetric point (K=2 at exactly [0.5, 0.5], the
            # ablation reduced to Single without drift). Variance is all such
            # states may differ in, so they start spread around the global
            # covariance, drawn from this restart's own rng so restarts
            # explore different starts (25 Oct 2026). Only here: the full row
            # and ZeroMean with vol/mom features separate their states through
            # the free means and start exactly as before.
            self.covars *= self._spread(rng, self.K)[:, None, None]
        self.startprob = np.full(self.K, 1.0 / self.K)
        if self.K == 1:
            self.transmat = np.ones((1, 1))
        else:
            self.transmat = np.full((self.K, self.K), 0.1 / (self.K - 1))
            np.fill_diagonal(self.transmat, 0.9)
        self._constrain()
        return global_cov

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < max(10, 3 * self.K):
            raise ValueError("not enough rows to fit the HMM")
        rng = np.random.default_rng(self.seed)
        global_cov = self._init(X, rng)
        T, d = X.shape
        previous = -np.inf
        self.converged = False
        for iteration in range(1, self.n_iter + 1):
            logB = self._emission(X)
            gamma, xi, loglik = self._expectations(logB)
            # M-step
            if self.K > 1:
                counts = xi + self.transition_prior          # the Dirichlet prior's mode (TRANSITION_PSEUDOCOUNT)
                rows = counts.sum(axis=1, keepdims=True)
                self.transmat = np.where(rows > 0, counts / np.where(rows > 0, rows, 1.0), 1.0 / self.K)
            self.startprob = np.clip(gamma[0], 1e-12, None)
            self.startprob /= self.startprob.sum()
            Nk = gamma.sum(axis=0)
            for k in range(self.K):
                if Nk[k] < 1e-3:
                    # a regime that lost every observation: restart it somewhere
                    self.means[k] = X[rng.integers(0, T)]
                    self.covars[k] = global_cov.copy()
                    if self._variance_only(d):
                        # its mean is zeroed again below, so at global_cov it
                        # would copy any other regime restarted this step (see _init)
                        self.covars[k] *= 2.0 ** rng.uniform(-1.0, 1.0)
                    continue
                self.means[k] = (gamma[:, k] @ X) / Nk[k]
                if self.zero_mean_dims:
                    self.means[k, self.zero_mean_dims] = 0.0
                diff = X - self.means[k]
                cov = (gamma[:, k, None] * diff).T @ diff / Nk[k]
                if self.shrinkage > 0:
                    cov = (1 - self.shrinkage) * cov + self.shrinkage * np.diag(np.diag(cov))
                self.covars[k] = cov + self.min_covar * np.eye(d)
            self._constrain()
            self.iterations = iteration
            self.loglik = loglik
            if abs(loglik - previous) < self.tol * max(1.0, abs(loglik)):
                self.converged = True
                break
            previous = loglik
        return self

    def filter(self, X):
        """(last filtered state posterior, total log likelihood)."""
        alpha, logc = self._forward(self._emission(np.asarray(X, dtype=float)))
        return alpha[-1], float(logc.sum())

    def predictive_loglik(self, X_new, start):
        """
        One-step-ahead predictive log likelihood of X_new, continuing from a
        state distribution `start` (already propagated). A start built from
        filter()'s posterior has lost every state below ~1e-308; that is why
        select_regimes scores its validation slice inside one pass instead.
        """
        _, logc = self._forward(self._emission(np.asarray(X_new, dtype=float)), start=start)
        return float(logc.sum())

    def n_params(self):
        d = self.means.shape[1]
        free_means = d - len(self.zero_mean_dims)
        cov_params = d * (d + 1) // 2
        if self.block:
            cov_params -= len(self.block[0]) * len(self.block[1])
        return (self.K - 1) + self.K * (self.K - 1) + self.K * free_means + self.K * cov_params


# ---------------------------------------------------------------------------
# Templates: stable regime identities across refits
# ---------------------------------------------------------------------------

def _sqrtm_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T


def wasserstein2(mean_a, cov_a, mean_b, cov_b):
    """The 2-Wasserstein distance between two Gaussians (closed form)."""
    mean_a, mean_b = np.asarray(mean_a, dtype=float), np.asarray(mean_b, dtype=float)
    cov_a, cov_b = np.asarray(cov_a, dtype=float), np.asarray(cov_b, dtype=float)
    root_b = _sqrtm_psd(cov_b)
    cross = _sqrtm_psd(root_b @ cov_a @ root_b)
    value = float(((mean_a - mean_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * cross))
    return math.sqrt(max(value, 0.0))


def whiten(mean, cov, whitener):
    return whitener @ np.asarray(mean, dtype=float), whitener @ np.asarray(cov, dtype=float) @ whitener.T


def match_templates(regimes, templates, whitener, threshold=1.0, rate=0.1):
    """
    regimes: [(mean, cov)] in return units; templates: [{"id", "mean", "cov", "count"}] (mutated).
    One-to-one assignment by whitened W2 distance; a regime farther than
    `threshold` from every free template spawns one. Matched templates move
    towards the regime by `rate`. Returns [(template id, distance)] per regime.
    """
    assignment = [None] * len(regimes)
    if templates and regimes:
        D = np.array([[wasserstein2(*whiten(m, c, whitener), *whiten(t["mean"], t["cov"], whitener)) for t in templates]
                      for m, c in regimes])
        rows, cols = optimize.linear_sum_assignment(D)
        for r, c in zip(rows, cols):
            if D[r, c] <= threshold:
                assignment[r] = (templates[c]["id"], float(D[r, c]))
                t = templates[c]
                t["mean"] = ((1 - rate) * np.asarray(t["mean"]) + rate * np.asarray(regimes[r][0])).tolist()
                t["cov"] = ((1 - rate) * np.asarray(t["cov"]) + rate * np.asarray(regimes[r][1])).tolist()
                t["count"] = int(t.get("count", 0)) + 1
    next_id = max([int(t["id"]) for t in templates], default=0) + 1
    for r, (mean, cov) in enumerate(regimes):
        if assignment[r] is None:
            templates.append({"id": next_id, "mean": np.asarray(mean, dtype=float).tolist(), "cov": np.asarray(cov, dtype=float).tolist(), "count": 1})
            assignment[r] = (next_id, 0.0)
            next_id += 1
    return assignment


def volatility_label(regime_sd, market_sd):
    """calm / normal / turbulent by the regime's mean return volatility against the market's."""
    ratio = float(np.mean(regime_sd)) / max(float(np.mean(market_sd)), 1e-12)
    return "calm" if ratio < 0.8 else ("turbulent" if ratio > 1.25 else "normal")


# ---------------------------------------------------------------------------
# The Regime HMM rows
# ---------------------------------------------------------------------------

def rolling_mean(r, L):
    """value at t = mean of r[t-L:t] (strictly before t); NaN where fewer than L returns precede t."""
    r = np.asarray(r, dtype=float)
    out = np.full(len(r), np.nan)
    if len(r) >= L:
        c = np.concatenate([[0.0], np.cumsum(r)])
        out[L:] = (c[L:len(r)] - c[:len(r) - L]) / L
    return out


def rolling_std(r, L):
    r = np.asarray(r, dtype=float)
    out = np.full(len(r), np.nan)
    if len(r) >= L:
        c1 = np.concatenate([[0.0], np.cumsum(r)])
        c2 = np.concatenate([[0.0], np.cumsum(r * r)])
        m = (c1[L:len(r)] - c1[:len(r) - L]) / L
        v = (c2[L:len(r)] - c2[:len(r) - L]) / L - m * m
        out[L:] = np.sqrt(np.clip(v, 0.0, None))
    return out


def build_features(history, features=("returns", "vol", "mom"), vol_lookback=60, mom_lookback=20):
    """
    Feature rows for the HMM from the aligned returns (T x n_pos): per
    instrument the day's return, then (if asked) the rolling volatility and
    the rolling mean of the returns strictly before the day. Returns (X, the
    next day's known features (the non-return columns), first row index
    used, return dims, feature dims).
    """
    history = np.asarray(history, dtype=float)
    T, n_pos = history.shape
    blocks, next_blocks = [history], []
    lookbacks = []
    if "vol" in features:
        lookbacks.append(vol_lookback)
        blocks.append(np.column_stack([rolling_std(history[:, p], vol_lookback) for p in range(n_pos)]))
        next_blocks.append(history[-vol_lookback:].std(axis=0))
    if "mom" in features:
        lookbacks.append(mom_lookback)
        blocks.append(np.column_stack([rolling_mean(history[:, p], mom_lookback) for p in range(n_pos)]))
        next_blocks.append(history[-mom_lookback:].mean(axis=0))
    first = max(lookbacks) if lookbacks else 0
    if T - first < 30:
        raise ValueError("not enough history for the HMM features")
    X = np.column_stack(blocks)[first:]
    next_features = np.concatenate(next_blocks) if next_blocks else np.zeros(0)
    return X, next_features, first, list(range(n_pos)), list(range(n_pos, X.shape[1]))


class RegimeHmmModel:
    """
    The Regime HMM Model row and its two ablations (variant "full" |
    "zero_mean" | "single"). Refit from scratch on every call - each backtest
    day is independent, as for every other row - with the number of regimes
    reselected every `reselect_every` game days (the selection at a day uses
    the history up to the most recent multiple of that cadence, so it is
    causal and stable between two reselections).
    """

    def __init__(self, variant="full", regimes=(2, 5), penalty=1.0, validation=250, reselect_every=20, window=0,
                 features=("returns", "vol", "mom"), vol_lookback=60, mom_lookback=20, shrinkage=0.1, min_covar=1e-4,
                 restarts=2, max_iter=100, tol=1e-5, template_rate=0.1, template_threshold=1.0, seed=0, k=K_BINS):
        if variant not in ("full", "zero_mean", "single"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.regimes = (int(regimes[0]), int(regimes[1]))
        self.penalty = float(penalty)
        self.validation = int(validation)
        self.reselect_every = max(1, int(reselect_every))
        self.window = int(window)
        self.features = tuple(features) if not isinstance(features, str) else tuple(f.strip() for f in features.split(",") if f.strip())
        self.vol_lookback = int(vol_lookback)
        self.mom_lookback = int(mom_lookback)
        self.shrinkage = float(shrinkage)
        self.min_covar = float(min_covar)
        self.restarts = max(1, int(restarts))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.template_rate = float(template_rate)
        self.template_threshold = float(template_threshold)
        self.seed = int(seed)
        self.k = int(k)
        self.dataPath = ""
        self.history = None
        self.state_dir = None
        self.regime_log = None
        self._last = None
        self._selection_cache = {}
        self.last_forecast = None

    @property
    def name(self):
        return {"full": HMM_NAME, "zero_mean": HMM_ZERO_MEAN_NAME, "single": HMM_SINGLE_NAME}[self.variant]

    def setDataPath(self, dataPath):
        self.dataPath = dataPath
        self.history = MarketHistory(dataPath)
        self._last = None
        self._selection_cache = {}

    def setSortedPrediction(self, use):
        pass

    def setStateDir(self, directory):
        """Where the regime templates persist between daily refits (Predictor.py's live run only)."""
        self.state_dir = directory

    def setRegimeLog(self, path):
        """Where each day's reading (dominant template, probabilities) is appended."""
        self.regime_log = path

    def clear(self):
        self._last = None

    # --- fitting -------------------------------------------------------------
    def _standardise(self, X, return_dims):
        centre = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        if self.variant == "zero_mean":
            centre = centre.copy()
            centre[return_dims] = 0.0        # a zero mean must mean zero return, not the window's mean return
        return (X - centre) / scale, centre, scale

    def _make(self, K, return_dims, feature_dims, seed):
        zero = return_dims if self.variant == "zero_mean" else None
        block = (return_dims, feature_dims) if (self.variant == "zero_mean" and feature_dims) else None
        return GaussianHMM(K, n_iter=self.max_iter, tol=self.tol, min_covar=self.min_covar, shrinkage=self.shrinkage,
                           zero_mean_dims=zero, block=block, seed=seed)

    def _fit(self, Z, K, return_dims, feature_dims, restarts):
        best = None
        for restart in range(restarts):
            hmm = self._make(K, return_dims, feature_dims, seed=self.seed + 1000 * restart + K).fit(Z)
            if best is None or hmm.loglik > best.loglik:
                best = hmm
        return best

    def select_regimes(self, Z, return_dims, feature_dims):
        """K by the penalised one-step-ahead predictive log likelihood of a validation slice."""
        lo, hi = self.regimes
        T = len(Z)
        V = min(self.validation, T // 3)
        if lo >= hi or T - V < 60 or V < 20:
            return lo, {}
        train = Z[:T - V]
        scores = {}
        for K in range(lo, hi + 1):
            try:
                hmm = self._fit(train, K, return_dims, feature_dims, restarts=1)
                # The validation rows' one-step-ahead log likelihoods are their
                # normalisers in one forward pass over training and validation
                # rows together (review, 26 Oct 2026). Filtering the training
                # slice and continuing from last @ A, as until then, cut every
                # regime below ~1e-308 to exactly 0 at the boundary: trial 6's
                # K=4 had three at -891 to -1058 nats there and scored -79627
                # nats instead of -4047. The same numbers otherwise - the
                # emissions are row by row and the recursion is the same; a
                # pass _flushed sends to log space moves them by ~1e-12.
                _, logc = hmm._forward(hmm._emission(Z))
                predictive = float(logc[T - V:].sum())
                scores[K] = {"predictive": predictive, "params": hmm.n_params(), "score": predictive - self.penalty * hmm.n_params()}
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not scores:
            return lo, {}
        best = max(sorted(scores), key=lambda K: scores[K]["score"])   # ties go to fewer regimes
        return best, scores

    def forecast(self, history):
        """Fit on the history and return the next day's return law per instrument as a regime mixture."""
        history = np.asarray(history, dtype=float)
        X, next_features, first, return_dims, feature_dims = build_features(history, self.features, self.vol_lookback, self.mom_lookback)
        if self.window > 0 and len(X) > self.window:
            X = X[-self.window:]
        Z, centre, scale = self._standardise(X, return_dims)
        if self.variant == "single":
            K, selection = 1, {}
        else:
            anchor = (len(Z) // self.reselect_every) * self.reselect_every
            key = (anchor, float(Z[0, 0]) if len(Z) else 0.0, float(Z[max(anchor - 1, 0), 0]) if len(Z) else 0.0, self.regimes, self.penalty, self.validation)
            if key in self._selection_cache:
                K, selection = self._selection_cache[key]
            else:
                K, selection = self.select_regimes(Z[:anchor] if anchor > 0 else Z, return_dims, feature_dims)
                self._selection_cache[key] = (K, selection)
                if len(self._selection_cache) > 64:
                    self._selection_cache.pop(next(iter(self._selection_cache)))
        hmm = self._fit(Z, K, return_dims, feature_dims, restarts=self.restarts)
        last, loglik = hmm.filter(Z)
        weights = last @ hmm.transmat
        f = (next_features - centre[feature_dims]) / scale[feature_dims] if feature_dims else np.zeros(0)
        # The exact log of the predicted regime weights (26 Oct 2026): a regime
        # the fitted chain cannot reach tomorrow (no positive transition into
        # it from a regime it may be in today) has weight 0 and stays at -inf.
        # The 1e-300 floor this replaces put it at -691 nats, so next-day
        # features 700+ nats likelier under it than under the reachable
        # regimes would have handed the mixture to a regime the model calls
        # impossible. A weight also reads 0 when it is merely below ~1e-308
        # (filter() cuts its posterior off there): -inf then understates it
        # where the floor overstated it, and only next-day features ~700+
        # nats apart tell the two apart. The weights sum to one, so one is
        # positive and the shift below stays finite.
        with np.errstate(divide="ignore"):
            log_w = np.log(weights)
        locs, sds = [], []
        for k in range(K):
            mu, cov = hmm.means[k], hmm.covars[k]
            if feature_dims:
                mu_r, mu_f = mu[return_dims], mu[feature_dims]
                S_rr = cov[np.ix_(return_dims, return_dims)]
                S_rf = cov[np.ix_(return_dims, feature_dims)]
                S_ff = cov[np.ix_(feature_dims, feature_dims)]
                solve = np.linalg.solve(S_ff, np.eye(len(feature_dims)))
                gain = S_rf @ solve
                cond_mu = mu_r + gain @ (f - mu_f)
                cond_cov = S_rr - gain @ S_rf.T
                log_w[k] += _log_gaussian(f[None, :], mu_f, S_ff)[0]
            else:
                cond_mu, cond_cov = mu[return_dims], cov[np.ix_(return_dims, return_dims)]
            locs.append(cond_mu * scale[return_dims] + centre[return_dims])
            sds.append(np.sqrt(np.clip(np.diag(cond_cov), 1e-12, None)) * scale[return_dims])
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        regimes_in_return_units = []
        for k in range(K):
            mu_r = hmm.means[k][return_dims] * scale[return_dims] + centre[return_dims]
            S = hmm.covars[k][np.ix_(return_dims, return_dims)] * np.outer(scale[return_dims], scale[return_dims])
            regimes_in_return_units.append((mu_r, S))
        market_sd = history.std(axis=0)
        out = {"K": K, "weights": w, "locs": np.asarray(locs), "sds": np.asarray(sds), "selection": selection,
               "loglik": loglik, "iterations": hmm.iterations, "converged": hmm.converged,
               "regimes": regimes_in_return_units, "market_sd": market_sd, "rows": len(Z), "features": list(self.features)}
        self.last_forecast = out
        return out

    def bin_probabilities(self, history, edges=None):
        edges = edges or edges_for_next_day(history, self.k)
        fc = self.forecast(history)
        probs = [mixture_bin_probabilities(edges[pos], fc["weights"], fc["locs"][:, pos], fc["sds"][:, pos]) for pos in range(len(edges))]
        return np.asarray(probs)

    # --- the reading (templates) ---------------------------------------------
    def reading(self, forecast, date):
        """Which regime the model believes the market is in, with a persistent identity when a state directory is set."""
        K = forecast["K"]
        order = np.argsort([float(np.mean(np.sqrt(np.diag(S)))) for _, S in forecast["regimes"]])   # calmest first
        rank = {int(k): int(i) + 1 for i, k in enumerate(order)}
        dominant = int(np.argmax(forecast["weights"]))
        labels = {k: volatility_label(np.sqrt(np.diag(forecast["regimes"][k][1])), forecast["market_sd"]) for k in range(K)}
        record = {"date": date, "row": self.name, "regimes": K, "features": list(self.features), "dominant": dominant, "volatility_rank": rank[dominant],
                  "label": labels[dominant], "probability": float(forecast["weights"][dominant]),
                  "weights": [float(w) for w in forecast["weights"]],
                  "expected_return": [float(v) for v in (forecast["weights"] @ forecast["locs"])],
                  "volatility": [float(v) for v in np.sqrt(forecast["weights"] @ (forecast["sds"] ** 2 + forecast["locs"] ** 2) - (forecast["weights"] @ forecast["locs"]) ** 2)],
                  "template": None}
        if self.state_dir:
            os.makedirs(self.state_dir, exist_ok=True)
            store = os.path.join(self.state_dir, f"{self.variant}_{TEMPLATE_FILE}")
            try:
                with open(store) as handle:
                    templates = json.load(handle).get("templates", [])
            except (OSError, json.JSONDecodeError):
                templates = []
            whitener = self._whitener(forecast)
            assignment = match_templates(forecast["regimes"], templates, whitener, self.template_threshold, self.template_rate)
            record["template"] = int(assignment[dominant][0])
            record["template_distance"] = assignment[dominant][1]
            record["templates"] = [int(a[0]) for a in assignment]
            with open(store, "w") as handle:
                json.dump({"row": self.name, "updated": date, "templates": templates}, handle)
        return record

    def _whitener(self, forecast):
        """G^-1/2 for the market's return covariance (the mixture's), so template distances are dimensionless."""
        w, locs, sds = forecast["weights"], forecast["locs"], forecast["sds"]
        G = sum(w[k] * (forecast["regimes"][k][1] + np.outer(forecast["regimes"][k][0], forecast["regimes"][k][0])) for k in range(forecast["K"]))
        mean = sum(w[k] * forecast["regimes"][k][0] for k in range(forecast["K"]))
        G = G - np.outer(mean, mean) + 1e-12 * np.eye(len(mean))
        values, vectors = np.linalg.eigh((G + G.T) / 2)
        return (vectors / np.sqrt(np.clip(values, 1e-18, None))) @ vectors.T

    def log_reading(self, record):
        if not self.regime_log:
            return
        os.makedirs(os.path.dirname(self.regime_log) or ".", exist_ok=True)
        try:
            with open(self.regime_log) as handle:
                log = json.load(handle)
        except (OSError, json.JSONDecodeError):
            log = {}
        entries = [e for e in log.get(self.name, []) if e.get("date") != record["date"]]
        entries.append(record)
        entries.sort(key=lambda e: e["date"])
        log[self.name] = entries[-REGIME_LOG_KEEP:]
        log["updated"] = record["date"]
        with open(self.regime_log, "w") as handle:
            json.dump(log, handle, indent=1)

    # --- the row interface -----------------------------------------------------
    def _probabilities(self, skipRows):
        if self.history is None:
            raise ValueError("setDataPath first")
        key = (self.history.stamp(), int(skipRows), self.variant, self.regimes, self.penalty, self.window, self.features)
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        days, history = self.history.visible(skipRows)
        probs = self.bin_probabilities(history)
        if self.state_dir or self.regime_log:
            try:
                self.log_reading(self.reading(self.last_forecast, days[-1]))
            except Exception as exc:   # the reading is a by-product; never the prediction's problem
                print(f"{self.name}: regime reading not recorded ({exc})")
        self._last = (key, probs)
        return probs

    def run(self, generateSubsets=None, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        # the median bin, not the most likely one (_median_ticket); the probabilities below are unchanged
        return _median_ticket(self._probabilities(skipRows)), {}

    def score_positions(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        return _slots(self._probabilities(skipRows))

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        return _pooled(self._probabilities(skipRows))


# ---------------------------------------------------------------------------
# Building the rows from a game's parameters
# ---------------------------------------------------------------------------

# The knobs of the market rows as bestParams_<market>.json keys, with the
# values served when a key is missing - the ONE place they are defined:
# build_market_models reads them, and HyperoptStatistics.py's SERVED_DEFAULTS
# imports them, so the champion/challenger gate's "untuned default" reference
# is this table by construction and cannot drift from what Predictor.py
# serves. The three Regime HMM rows share the regimeHmm* keys on purpose:
# the two ablation rows differ from the full row by one constraint each,
# and would stop being ablations if they were tuned apart. The template
# knobs only label the regimes (they never touch a probability), so the
# tuner leaves them alone.
MARKET_DEFAULTS = {
    "garchMean": "constant",            # constant | zero | ar1
    "garchDist": "t",                   # t | normal
    "garchWindow": 2000,                # newest returns the GARCH is fitted on
    "regimeHmmMin": 2,                  # regime counts tried, inclusive
    "regimeHmmMax": 5,
    "regimeHmmPenalty": 1.0,            # nats per parameter in the count selection
    "regimeHmmValidation": 250,         # days of the one-step-ahead validation slice
    "regimeHmmReselectEvery": 20,       # game days between two count selections
    "regimeHmmWindow": 0,               # 0 = expanding window, else the newest N days
    "regimeHmmFeatures": "returns,vol,mom",
    "regimeHmmVolLookback": 60,
    "regimeHmmMomLookback": 20,
    "regimeHmmShrinkage": 0.1,          # covariance shrinkage towards the diagonal
    "regimeHmmMinCovar": 1e-4,
    "regimeHmmRestarts": 2,
    "regimeHmmMaxIter": 100,
    "regimeHmmTemplateRate": 0.1,
    "regimeHmmTemplateThreshold": 1.0,
}


def build_market_models(dataPath, bestParams=None, state_dir=None, regime_log=None):
    """
    {row name: model} for a market game, configured from bestParams_<game>.json
    (MARKET_DEFAULTS when a key is missing). state_dir/regime_log make the HMM
    rows record their reading (Predictor.py's live run); the Backtester leaves
    them unset. Same call for the trainer, the tuners, the controls and the
    report.
    """
    p = dict(MARKET_DEFAULTS)
    p.update({k: v for k, v in (bestParams or {}).items() if k in MARKET_DEFAULTS})
    garch = GarchModel(mean=p["garchMean"], dist=p["garchDist"], window=p["garchWindow"])
    garch.setDataPath(dataPath)
    models = {GARCH_NAME: garch}
    common = dict(regimes=(p["regimeHmmMin"], p["regimeHmmMax"]), penalty=p["regimeHmmPenalty"],
                  validation=p["regimeHmmValidation"], reselect_every=p["regimeHmmReselectEvery"],
                  window=p["regimeHmmWindow"], features=p["regimeHmmFeatures"],
                  vol_lookback=p["regimeHmmVolLookback"], mom_lookback=p["regimeHmmMomLookback"],
                  shrinkage=p["regimeHmmShrinkage"], min_covar=p["regimeHmmMinCovar"],
                  restarts=p["regimeHmmRestarts"], max_iter=p["regimeHmmMaxIter"],
                  template_rate=p["regimeHmmTemplateRate"], template_threshold=p["regimeHmmTemplateThreshold"])
    for name, variant in HMM_ROWS.items():
        row = RegimeHmmModel(variant=variant, **common)
        row.setDataPath(dataPath)
        if state_dir:
            row.setStateDir(state_dir)
        if regime_log:
            row.setRegimeLog(regime_log)
        models[name] = row
    return models


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check():
    import tempfile
    import warnings
    try:
        from src.MarketGame import cut_game, write_game_csv, write_returns_file
    except ImportError:
        from MarketGame import cut_game, write_game_csv, write_returns_file

    rng = np.random.default_rng(3)

    # 1. The GARCH recursion as a filter equals the loop.
    e2 = rng.standard_normal(50) ** 2
    h = garch_variances(e2, 0.1, 0.1, 0.8, h0=1.3)
    loop, prev = [], 1.3
    for value in e2:
        prev = 0.1 + 0.1 * value + 0.8 * prev
        loop.append(prev)
    assert np.allclose(h, loop), "lfilter recursion differs from the loop"

    # 2. GARCH recovers a simulated GARCH(1,1) and reacts to a shock.
    T = 4000
    omega, alpha, beta, nu = 0.05, 0.08, 0.90, 5.0
    z = rng.standard_t(nu, size=T) * math.sqrt((nu - 2) / nu)
    r, hv = np.empty(T), omega / (1 - alpha - beta)
    for t in range(T):
        if t:
            hv = omega + alpha * r[t - 1] ** 2 + beta * hv
        r[t] = math.sqrt(hv) * z[t]
    started = time.time()
    fit = fit_garch(r, mean="constant", dist="t")
    took = time.time() - started
    assert fit["converged"] and abs(fit["persistence"] - (alpha + beta)) < 0.05, fit
    assert 3.0 < fit["nu"] < 9.0, fit["nu"]
    calm = fit_garch(np.concatenate([r[:-5], np.full(5, 0.01)]))["forecast"]["scale"]
    shocked = fit_garch(np.concatenate([r[:-1], [8.0 * math.sqrt(hv)]]))["forecast"]["scale"]
    assert shocked > calm, (shocked, calm)
    normal = fit_garch(r, mean="zero", dist="normal")
    ar1 = fit_garch(r, mean="ar1", dist="t")
    assert normal["converged"] and ar1["converged"] and abs(ar1["phi"]) < 0.15
    print(f"GARCH: persistence {fit['persistence']:.3f} (true {alpha + beta}), nu {fit['nu']:.1f} (true {nu}), "
          f"forecast scale {shocked:.3f} after a shock vs {calm:.3f} after calm days, fit {took:.2f}s")

    # 3. Bins: probabilities sum to one; a wide law puts more than 1/K on the outer bins.
    edges = quantile_edges(rng.normal(0, 0.02, 5000), 10)
    probs = mixture_bin_probabilities(edges, [1.0], [0.0], [0.04])
    assert abs(probs.sum() - 1) < 1e-9 and probs[0] > 0.1 and probs[-1] > 0.1 and probs[4] < 0.1
    mix = mixture_bin_probabilities(edges, [0.5, 0.5], [0.0, 0.0], [0.01, 0.04], df=[5.0, 5.0])
    assert abs(mix.sum() - 1) < 1e-9 and mix.min() > 0

    # 3b. The HMM rows' ticket is the median bin (25 Oct 2026): on fat-tailed
    # edges a zero-mean Gaussian with the history's spread has its most mass
    # in an outer bin - the old ticket - while its median sits in the middle;
    # moving the location moves the median bin, and with it the side.
    fat = np.random.default_rng(25).standard_t(3, size=5000) * 0.02   # own stream: the sections below keep their data
    fat_edges = quantile_edges(fat, 10)
    flat_law = mixture_bin_probabilities(fat_edges, [1.0], [0.0], [fat.std()])
    assert _ticket([flat_law])[0] in (0, 1, 8, 9) and _median_ticket([flat_law])[0] in (4, 5), (_ticket([flat_law]), flat_law.round(3))
    centre = 0.5 * (fat_edges[3] + fat_edges[4])    # inside bin 4
    assert _median_ticket([mixture_bin_probabilities(fat_edges, [1.0], [centre], [0.03])]) == [4]
    assert _median_ticket([mixture_bin_probabilities(fat_edges, [1.0], [fat_edges[6] + 1e-4], [0.03])]) == [7]
    assert _median_ticket([[0.25, 0.25, 0.0, 0.5]]) == [1] and _median_ticket([[0.2, 0.25, 0.1, 0.45]]) == [2], "the first bin reaching one half"
    print(f"tickets: zero-mean law on t(3) edges - most likely bin {_ticket([flat_law])[0]}, median bin {_median_ticket([flat_law])[0]}; "
          "the median bin follows the location")

    # 4. The HMM finds two simulated volatility regimes, K is selected, constraints hold.
    K_true, T2, d = 2, 1500, 3
    A = np.array([[0.97, 0.03], [0.05, 0.95]])
    states = np.empty(T2, dtype=int)
    states[0] = 0
    for t in range(1, T2):
        states[t] = rng.choice(2, p=A[states[t - 1]])
    sds = np.array([0.5, 2.0])
    X = rng.standard_normal((T2, d)) * sds[states][:, None] + np.array([0.3, 0.0])[states][:, None]
    started = time.time()
    hmm = GaussianHMM(2, seed=1).fit(X)
    took = time.time() - started
    alpha_f, _ = hmm._forward(hmm._emission(X))
    decoded = alpha_f.argmax(axis=1)
    accuracy = max((decoded == states).mean(), (decoded != states).mean())
    assert accuracy > 0.9, accuracy
    assert hmm.converged and np.allclose(hmm.transmat.sum(axis=1), 1)
    # the transition prior (30 Oct 2026): a fit on a short history whose second regime is never left keeps every
    # transition open - no exact 0 in the fitted matrix; without the prior EM keeps the unseen exit at 0
    absorb = np.concatenate([rng.standard_normal((60, d)) * 0.5, rng.standard_normal((40, d)) * 2.0 + 0.3])
    open_fit = GaussianHMM(2, seed=1).fit(absorb)
    mle_fit = GaussianHMM(2, seed=1, transition_prior=0.0).fit(absorb)
    assert open_fit.transmat.min() > 0.005, open_fit.transmat
    assert mle_fit.transmat.min() < open_fit.transmat.min(), (mle_fit.transmat, open_fit.transmat)
    row = RegimeHmmModel(variant="full", regimes=(1, 4), features=("returns",), penalty=1.0, validation=300, restarts=1)
    K_sel, scores = row.select_regimes(X, list(range(d)), [])
    assert K_sel == 2, (K_sel, {k: round(v["score"], 1) for k, v in scores.items()})
    zero = GaussianHMM(2, zero_mean_dims=[0, 1], block=([0, 1], [2]), seed=1).fit(X)
    assert np.all(zero.means[:, [0, 1]] == 0) and np.all(zero.covars[:, :2, 2] == 0) and np.all(zero.covars[:, 2, :2] == 0)
    one = GaussianHMM(1, shrinkage=0.0, min_covar=0.0).fit(X)
    assert np.allclose(one.means[0], X.mean(axis=0)) and np.allclose(one.covars[0], np.cov(X, rowvar=False, bias=True), atol=1e-6)
    score_text = ", ".join("K%d:%.0f" % (k, v["score"]) for k, v in scores.items())
    print(f"HMM: two volatility regimes decoded at {accuracy:.1%}, K=2 selected from 1..4 ({score_text}), "
          f"zero-mean and block constraints hold, K=1 is the sample Gaussian; fit {took:.2f}s for {T2} rows")

    # 4b. Every mean held at zero (the ZeroMean row on regimeHmmFeatures='returns'):
    # the states must still separate by variance instead of sitting on the
    # identical start, which served exactly [0.5, 0.5] from 10 to 25 Oct 2026.
    var_only = GaussianHMM(2, zero_mean_dims=list(range(d)), seed=1).fit(X)
    assert np.all(var_only.means == 0) and not np.allclose(var_only.covars[0], var_only.covars[1]), "zero-mean states never separated"
    decoded_v = var_only._forward(var_only._emission(X))[0].argmax(axis=1)
    accuracy_v = max((decoded_v == states).mean(), (decoded_v != states).mean())
    assert accuracy_v > 0.9, accuracy_v
    variance_fc = RegimeHmmModel(variant="zero_mean", regimes=(2, 3), features="returns", validation=300, restarts=2).forecast(X)
    assert not np.allclose(variance_fc["weights"], 1.0 / variance_fc["K"]), variance_fc["weights"]
    assert not all(np.allclose(variance_fc["regimes"][0][1], S) for _, S in variance_fc["regimes"][1:]), "identical zero-mean regimes"
    free = GaussianHMM(2, seed=1)
    free._init(X, np.random.default_rng(1))
    assert np.array_equal(free.covars[0], free.covars[1]), "a start with free means must stay the shared global covariance"
    print(f"HMM zero-mean on returns only: variance regimes decoded at {accuracy_v:.1%}, row K={variance_fc['K']} "
          f"weights {np.round(variance_fc['weights'], 3).tolist()} (not uniform); free-mean starts unchanged")

    # 4c. The 0/0 of the 10 Oct tuning study and the 18 Oct week-game controls
    # (26 Oct 2026): a fitted chain sitting in an absorbing regime meets a row
    # ~1250 nats likelier under a regime it can no longer reach (there the
    # start was one-hot because the other regimes had been cut off at double
    # range, see GaussianHMM). The scaled step summed to 0 (RuntimeWarning),
    # math.log(0) raised, and select_regimes dropped that K instead of
    # scoring it. The step now runs in log space and gives the closed form,
    # without a warning.
    trap = GaussianHMM(2)
    trap.startprob = np.array([0.5, 0.5])
    trap.transmat = np.array([[0.9, 0.1], [0.0, 1.0]])     # regime 1 absorbing: EM keeps a 0 at 0
    trap.means = np.array([[0.0], [50.0]])
    trap.covars = np.array([[[1.0]], [[1.0]]])
    sitting = np.array([0.0, 1.0])
    assert trap._forward_scaled(trap._emission(np.array([[50.0], [0.0]])), start=sitting) is None, "the trap must underflow the scaled step"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predictive = trap.predictive_loglik(np.array([[50.0], [0.0], [50.0]]), start=sitting)
        filtered, _ = trap.filter(np.array([[0.0], [0.0], [50.0], [50.0], [0.0]]))
    exact = -1.5 * math.log(2 * math.pi) - 0.5 * 50.0 ** 2       # three rows in regime 1, one of them 50 sd out
    assert abs(predictive - exact) < 1e-9, (predictive, exact)
    assert np.isfinite(filtered).all() and abs(filtered.sum() - 1) < 1e-12 and filtered[1] > 0.999, filtered
    # The E-step on the same trap against brute force over all 2^5 state paths:
    # posteriors, expected transition counts and the log likelihood exact.
    X_trap = np.array([[0.0], [0.3], [50.0], [49.0], [0.0]])
    logB_trap = trap._emission(X_trap)
    assert trap._forward_scaled(logB_trap) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gamma_t, xi_t, loglik_t = trap._expectations(logB_trap)
    with np.errstate(divide="ignore"):
        log_start, log_A = np.log(trap.startprob), np.log(trap.transmat)
    paths = np.array(np.meshgrid(*[[0, 1]] * len(X_trap), indexing="ij")).reshape(len(X_trap), -1).T
    log_paths = np.array([log_start[p[0]] + sum(log_A[p[t - 1], p[t]] for t in range(1, len(p)))
                          + sum(logB_trap[t, p[t]] for t in range(len(p))) for p in paths])
    log_total = _logsumexp(log_paths)
    weight = np.exp(log_paths - log_total)
    gamma_bf = np.array([[weight[paths[:, t] == k].sum() for k in range(2)] for t in range(len(X_trap))])
    xi_bf = np.array([[sum(weight[(paths[:, t] == i) & (paths[:, t + 1] == j)].sum() for t in range(len(X_trap) - 1))
                       for j in range(2)] for i in range(2)])
    assert abs(loglik_t - log_total) < 1e-9 and np.allclose(gamma_t, gamma_bf, atol=1e-12) and np.allclose(xi_t, xi_bf, atol=1e-12), \
        (loglik_t, log_total, gamma_t, gamma_bf, xi_t, xi_bf)
    # A pass that loses a regime it can reach with no normaliser underflowing
    # (review, 26 Oct 2026): regimes 0 and 2 absorbing, a row 2000 nats
    # against regime 2, then rows 200 nats a row in its favour. Every scaled
    # step's normaliser is e^-200, a normal double, yet the scaled pass has
    # flushed regime 2 and scores only regime 0's path; _forward redoes it
    # in log space and gives the two-path closed form.
    flush = GaussianHMM(3)
    flush.startprob = np.array([0.5, 0.0, 0.5])
    flush.transmat = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.0, 0.0, 1.0]])
    flush.means = np.array([[0.0], [40.0], [-40.0]])
    flush.covars = np.ones((3, 1, 1))
    logB_flush = flush._emission(np.array([[30.0]] + [[-25.0]] * 14))
    scaled_flush = flush._forward_scaled(logB_flush)
    assert scaled_flush is not None and flush._flushed(scaled_flush[0], flush.startprob), "the scaled pass must flush regime 2 without underflowing"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, logc_flush = flush._forward(logB_flush)
    two_paths = _logsumexp(math.log(0.5) + logB_flush[:, [0, 2]].sum(axis=0))
    assert abs(logc_flush.sum() - two_paths) < 1e-9 and two_paths - scaled_flush[1].sum() > 700, (logc_flush.sum(), two_paths, scaled_flush[1].sum())
    # ...and select_regimes scores its validation slice inside one pass: 59
    # rows neutral between regimes 0 and 2, the 2000-nat row ending the
    # training slice, 20 validation rows for regime 2. Filtering the training
    # rows and continuing from last @ A (until 26 Oct) cut regime 2 off at
    # the boundary and scored the slice ~2000 nats too low.
    picker = RegimeHmmModel(variant="full", regimes=(2, 3), features=("returns",), validation=20, penalty=0.0)
    picker._fit = lambda *args, **kwargs: flush
    Z_pick = np.array([[-20.0]] * 59 + [[30.0]] + [[-25.0]] * 20)
    logB_pick = flush._emission(Z_pick)
    exact_pick = (_logsumexp(math.log(0.5) + logB_pick[:, [0, 2]].sum(axis=0))
                  - _logsumexp(math.log(0.5) + logB_pick[:60][:, [0, 2]].sum(axis=0)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, pick_scores = picker.select_regimes(Z_pick, [0], [])
        handoff = flush.predictive_loglik(Z_pick[60:], start=flush.filter(Z_pick[:60])[0] @ flush.transmat)
    assert abs(pick_scores[2]["predictive"] - exact_pick) < 1e-9 and exact_pick - handoff > 1500, (pick_scores, exact_pick, handoff)
    # The row's mixture keeps a regime the chain cannot reach at weight 0: the
    # fit is stuck in absorbing regime 1, and tomorrow's known momentum sits
    # ~1000 nats out under it - the old 1e-300 floor handed the whole mixture
    # to regime 0. The filter's pass over the history underflows too.
    stuck = RegimeHmmModel(variant="full", regimes=(2, 2), features=("returns", "mom"), mom_lookback=5)
    hist = np.random.default_rng(26).standard_normal((200, 1)) * 0.02     # own stream, like 3b
    X_s, next_s, _, rd_s, fd_s = build_features(hist, stuck.features, stuck.vol_lookback, stuck.mom_lookback)
    _, centre_s, scale_s = stuck._standardise(X_s, rd_s)
    f_s = float((next_s[0] - centre_s[fd_s[0]]) / scale_s[fd_s[0]])
    stuck_hmm = GaussianHMM(2)
    stuck_hmm.startprob, stuck_hmm.transmat = np.array([0.0, 1.0]), np.array([[0.5, 0.5], [0.0, 1.0]])
    stuck_hmm.means = np.array([[0.0, f_s], [0.0, f_s + 1.0]])
    stuck_hmm.covars = np.array([np.eye(2), np.diag([1.0, 5e-4])])
    stuck._fit = lambda *args, **kwargs: stuck_hmm
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stuck_fc = stuck.forecast(hist)
        stuck_probs = stuck.bin_probabilities(hist)
    assert stuck_fc["weights"].tolist() == [0.0, 1.0], stuck_fc["weights"]
    assert np.isfinite(stuck_probs).all() and abs(stuck_probs.sum() - 1) < 1e-9, stuck_probs
    # Ordinary inputs never reach the log-space path (their numbers are the
    # scaled passes', as before), and the two paths agree on them.
    logB_ok = hmm._emission(X)
    assert hmm._forward_scaled(logB_ok) is not None and hmm._backward_scaled(logB_ok)[0] is not None
    assert not hmm._flushed(hmm._forward_scaled(logB_ok)[0], hmm.startprob)
    gamma_s, xi_s, loglik_s = hmm._expectations(logB_ok)
    gamma_l, xi_l, loglik_l = hmm._expectations_log(logB_ok)
    alpha_s, logc_s = hmm._forward(logB_ok)
    log_alpha_l, logc_l = hmm._forward_log(logB_ok)
    gaps = (np.abs(gamma_s - gamma_l).max(), np.abs(xi_s - xi_l).max() / xi_s.max(), abs(loglik_s - loglik_l) / abs(loglik_s),
            np.abs(alpha_s - np.exp(log_alpha_l)).max(), np.abs(logc_s - logc_l).max())
    assert max(gaps) < 1e-10, gaps
    print(f"HMM underflow: an absorbing regime meeting a row 50 sd out scores {predictive:.1f} nats (closed form {exact:.1f}) "
          f"instead of 0/0; the E-step matches brute force over 32 paths; a pass that flushed a reachable regime and the "
          f"validation score are exact ({pick_scores[2]['predictive']:.0f} nats, the old handoff {handoff:.0f}); an unreachable "
          f"regime keeps weight 0 in the mixture; on ordinary rows the log-space passes agree with the scaled ones to {max(gaps):.0e}")

    # 5. Conditional Gaussian against a brute-force sample.
    Sigma = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.1], [0.2, 0.1, 1.0]])
    sample = rng.multivariate_normal([0, 0, 0], Sigma, size=200000)
    near = sample[np.all(np.abs(sample[:, 1:] - [1.0, -0.5]) < 0.1, axis=1)]
    gain = Sigma[:1, 1:] @ np.linalg.inv(Sigma[1:, 1:])
    cond_mu = (gain @ np.array([1.0, -0.5]))[0]
    cond_var = (Sigma[:1, :1] - gain @ Sigma[1:, :1])[0, 0]
    assert abs(near[:, 0].mean() - cond_mu) < 0.05 and abs(near[:, 0].var() - cond_var) < 0.05, (near[:, 0].mean(), cond_mu)

    # 6. Templates: W2 is a metric on Gaussians; matching is stable under relabelling and spawns beyond the threshold.
    m1, C1 = np.array([0.0, 0.0]), np.diag([1.0, 1.0])
    m2, C2 = np.array([0.0, 0.0]), np.diag([4.0, 4.0])
    assert wasserstein2(m1, C1, m1, C1) < 1e-9 and abs(wasserstein2(m1, C1, m2, C2) - math.sqrt(2 * (1 - 2) ** 2)) < 1e-9
    templates = []
    W = np.eye(2)
    first = match_templates([(m1, C1), (m2, C2)], templates, W, threshold=1.0, rate=0.5)
    again = match_templates([(m2, C2), (m1, C1)], templates, W, threshold=1.0, rate=0.5)
    assert [a[0] for a in first] == [1, 2] and [a[0] for a in again] == [2, 1] and len(templates) == 2
    spawned = match_templates([(m1 + 10, C1)], templates, W, threshold=1.0, rate=0.5)
    assert spawned[0][0] == 3 and len(templates) == 3
    print("templates: W2 exact on isotropic Gaussians, identities survive a relabelled refit, a far regime spawns a template")

    # 7. The row interface on a market-shaped folder: tickets, probabilities, causality, the reading.
    with tempfile.TemporaryDirectory() as folder:
        n_days, n_pos = 700, 3
        vol_state = np.where(np.sin(np.arange(n_days) / 40.0) > 0, 0.01, 0.03)
        matrix = rng.standard_normal((n_days, n_pos)) * vol_state[:, None]
        days = [f"{2024 + i // 360}-{1 + (i % 360) // 30:02d}-{1 + i % 30:02d}" for i in range(n_days)]
        game = cut_game(days, matrix, k=10, min_history=250)
        write_game_csv(game, folder, "crypto")
        write_returns_file(days, matrix, ["A", "B", "C"], folder, "crypto")
        params = {"regimeHmmMax": 3, "regimeHmmValidation": 100, "regimeHmmRestarts": 1}
        rows = build_market_models(folder, params, state_dir=os.path.join(folder, "state"), regime_log=os.path.join(folder, "regimes.json"))
        assert list(rows) == MARKET_MODEL_NAMES
        timings = {}
        for name, model in rows.items():
            started = time.time()
            ticket, subsets = model.run(skipRows=0)
            slots = model.score_positions(skipRows=0)
            timings[name] = time.time() - started
            assert subsets == {} and len(ticket) == n_pos and all(0 <= b <= 9 for b in ticket)
            assert len(slots) == n_pos and all(abs(sum(s.values()) - 1) < 1e-9 for s in slots) and all(len(s) == 10 for s in slots)
            if name == GARCH_NAME:
                assert ticket == [max(s, key=s.get) for s in slots], "GARCH names the most likely bin"
            else:
                # the median bin of the same probabilities: below it less than half the mass, up to it at least half
                for b, s in zip(ticket, slots):
                    below = sum(s[j] for j in range(b))
                    assert below < 0.5 <= below + s[b], (name, b, s)
            pooled = model.score_numbers(skipRows=0)
            assert hasattr(model, "score_numbers") and abs(sum(pooled.values()) - 1) < 1e-9 and len(pooled) == 10, "score_numbers is what makes the Backtester collect the per-slot scores"
        # causal: hiding the newest five days gives the same answer whatever those five days hold
        garch = rows[GARCH_NAME]
        before = garch.score_positions(skipRows=5)
        perturbed = matrix.copy()
        perturbed[-5:] *= 10
        write_returns_file(days, perturbed, ["A", "B", "C"], folder, "crypto")
        garch.clear()
        after = garch.score_positions(skipRows=5)
        assert all(abs(a[b] - c[b]) < 1e-12 for a, c in zip(before, after) for b in range(10)), "a hidden day leaked"
        # ...and the history a row sees ends at the last visible game day
        seen_days, seen = rows[GARCH_NAME].history.visible(skipRows=5)
        assert seen_days[-1] == game[-6]["date"] and len(seen) == n_days - 5
        # the reading was recorded with a template
        with open(os.path.join(folder, "regimes.json")) as handle:
            log = json.load(handle)
        entry = log[HMM_NAME][-1]
        assert entry["date"] == days[-1] and entry["template"] is not None and entry["label"] in ("calm", "normal", "turbulent")
        assert abs(sum(entry["weights"]) - 1) < 1e-9 and len(entry["expected_return"]) == n_pos
        assert os.path.exists(os.path.join(folder, "state", f"full_{TEMPLATE_FILE}"))
        zero_fc = rows[HMM_ZERO_MEAN_NAME].last_forecast
        assert np.allclose(zero_fc["locs"], 0.0), "the zero-mean row must predict a zero return"
        # a folder without the returns file says what is missing
        os.remove(os.path.join(folder, "crypto-returns.tsv"))
        bare = GarchModel()
        bare.setDataPath(folder)
        try:
            bare.run()
            raise AssertionError("a missing returns file must be reported")
        except FileNotFoundError as exc:
            assert "returns.tsv" in str(exc)
    print("rows: " + ", ".join(f"{n.replace(' Model', '')} {t:.1f}s" for n, t in timings.items())
          + " on 700 days x 3 instruments; GARCH tickets are the most likely bin and the HMM rows' the median bin, "
          "probabilities sum to one, hidden days never leak, the reading is logged")
    print("MarketModels self-check OK")


if __name__ == "__main__":
    _self_check()
