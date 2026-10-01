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


class GaussianHMM:
    """
    Full-covariance Gaussian HMM fitted by EM (Baum-Welch) with scaled
    forward-backward passes. Optional constraints, applied exactly in the
    M-step: `zero_mean_dims` keeps those coordinates' means at zero and
    `block` = (dims_a, dims_b) zeroes the covariance between the two groups.
    """

    def __init__(self, n_states, n_iter=100, tol=1e-5, min_covar=1e-4, shrinkage=0.1, zero_mean_dims=None, block=None, seed=0):
        self.K = int(n_states)
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
        """Scaled forward pass: (filtered posteriors T x K, per-step log normalisers)."""
        T, K = logB.shape
        shift = logB.max(axis=1)
        B = np.exp(logB - shift[:, None])
        alpha = np.empty((T, K))
        logc = np.empty(T)
        a = (self.startprob if start is None else start) * B[0]
        c = a.sum()
        alpha[0] = a / c
        logc[0] = math.log(c) + shift[0]
        A = self.transmat
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c = a.sum()
            alpha[t] = a / c
            logc[t] = math.log(c) + shift[t]
        return alpha, logc

    def _backward(self, logB):
        T, K = logB.shape
        B = np.exp(logB - logB.max(axis=1)[:, None])
        beta = np.empty((T, K))
        beta[-1] = 1.0
        A = self.transmat
        for t in range(T - 2, -1, -1):
            b = A @ (B[t + 1] * beta[t + 1])
            beta[t] = b / b.sum()
        return beta, B

    def _constrain(self):
        if self.zero_mean_dims:
            self.means[:, self.zero_mean_dims] = 0.0
        if self.block:
            a, b = self.block
            for k in range(self.K):
                self.covars[k][np.ix_(a, b)] = 0.0
                self.covars[k][np.ix_(b, a)] = 0.0

    def _init(self, X, rng):
        T, d = X.shape
        idx = rng.choice(T, size=self.K, replace=False) if T >= self.K else rng.integers(0, T, size=self.K)
        self.means = X[idx].astype(float).copy()
        global_cov = np.cov(X, rowvar=False) if T > 1 else np.eye(d)
        global_cov = np.atleast_2d(global_cov) + self.min_covar * np.eye(d)
        self.covars = np.array([global_cov.copy() for _ in range(self.K)])
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
            alpha, logc = self._forward(logB)
            beta, B = self._backward(logB)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            loglik = float(logc.sum())
            # M-step
            if self.K > 1:
                w = B[1:] * beta[1:]
                norm = np.einsum("ti,ij,tj->t", alpha[:-1], self.transmat, w)
                xi = self.transmat * ((alpha[:-1] / norm[:, None]).T @ w)
                rows = xi.sum(axis=1, keepdims=True)
                self.transmat = np.where(rows > 0, xi / np.where(rows > 0, rows, 1.0), 1.0 / self.K)
            self.startprob = np.clip(gamma[0], 1e-12, None)
            self.startprob /= self.startprob.sum()
            Nk = gamma.sum(axis=0)
            for k in range(self.K):
                if Nk[k] < 1e-3:
                    # a regime that lost every observation: restart it somewhere
                    self.means[k] = X[rng.integers(0, T)]
                    self.covars[k] = global_cov.copy()
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
        """One-step-ahead predictive log likelihood of X_new, continuing from a state distribution `start` (already propagated)."""
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
        train, valid = Z[:T - V], Z[T - V:]
        scores = {}
        for K in range(lo, hi + 1):
            try:
                hmm = self._fit(train, K, return_dims, feature_dims, restarts=1)
                last, _ = hmm.filter(train)
                predictive = hmm.predictive_loglik(valid, start=last @ hmm.transmat)
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
        locs, sds, log_w = [], [], np.log(np.clip(weights, 1e-300, None))
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
        record = {"date": date, "row": self.name, "regimes": K, "dominant": dominant, "volatility_rank": rank[dominant],
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
        return _ticket(self._probabilities(skipRows)), {}

    def score_positions(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        return _slots(self._probabilities(skipRows))

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        return _pooled(self._probabilities(skipRows))


# ---------------------------------------------------------------------------
# Building the rows from a game's parameters
# ---------------------------------------------------------------------------

def build_market_models(dataPath, bestParams=None, state_dir=None, regime_log=None):
    """
    {row name: model} for a market game, configured from bestParams_<game>.json
    (defaults when a key is missing). state_dir/regime_log make the HMM rows
    record their reading (Predictor.py's live run); the Backtester leaves them
    unset. Same call for the trainer, the tuners, the controls and the report.
    """
    p = bestParams or {}
    garch = GarchModel(mean=p.get("garchMean", "constant"), dist=p.get("garchDist", "t"), window=p.get("garchWindow", 2000))
    garch.setDataPath(dataPath)
    models = {GARCH_NAME: garch}
    common = dict(regimes=(p.get("regimeHmmMin", 2), p.get("regimeHmmMax", 5)), penalty=p.get("regimeHmmPenalty", 1.0),
                  validation=p.get("regimeHmmValidation", 250), reselect_every=p.get("regimeHmmReselectEvery", 20),
                  window=p.get("regimeHmmWindow", 0), features=p.get("regimeHmmFeatures", "returns,vol,mom"),
                  vol_lookback=p.get("regimeHmmVolLookback", 60), mom_lookback=p.get("regimeHmmMomLookback", 20),
                  shrinkage=p.get("regimeHmmShrinkage", 0.1), min_covar=p.get("regimeHmmMinCovar", 1e-4),
                  restarts=p.get("regimeHmmRestarts", 2), max_iter=p.get("regimeHmmMaxIter", 100),
                  template_rate=p.get("regimeHmmTemplateRate", 0.1), template_threshold=p.get("regimeHmmTemplateThreshold", 1.0))
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
            assert ticket == [max(s, key=s.get) for s in slots]
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
          + " on 700 days x 3 instruments; tickets are argmax bins, probabilities sum to one, hidden days never leak, the reading is logged")
    print("MarketModels self-check OK")


if __name__ == "__main__":
    _self_check()
