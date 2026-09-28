import os, sys
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Dynamically adjust the import path (same bootstrap as the other src
# modules, so this file imports cleanly both as src.QuantumModels from the
# repo root and as QuantumModels from within src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')

# Ensure sibling src modules can be imported
if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)


# ---------------------------------------------------------------------------
# Batched statevector simulator
#
# The README's quantum research track deliberately starts at 4 qubits
# (2^4 = 16 amplitudes), so a quantum framework dependency would be pure
# overhead - plain numpy simulates these circuits exactly. Everything below
# is vectorized over the SAMPLE batch (states has shape (batch, 2**n_qubits))
# because the meta-learner training tables have thousands of rows and a
# per-row Python loop would dominate the runtime. Qubit q owns bit weight
# 2**(n_qubits-1-q) of the basis index; that convention is what lets the
# reshape in the gate helpers put the target qubit on its own axis.
# ---------------------------------------------------------------------------

def _apply_ry(states, angles, qubit, n_qubits):
    """
    In-place RY(angle) on one qubit for the whole batch. `angles` is either a
    per-sample vector (feature encoding: every row rotates by its own feature
    value) or a scalar (variational layers: one trainable angle shared by the
    batch) - the broadcasting below covers both without separate code paths.
    """
    half = 0.5 * np.asarray(angles, dtype=float)
    cos = np.cos(half)
    sin = np.sin(half)
    if cos.ndim == 1:
        # per-sample angles: broadcast over the two basis-index axes the
        # reshape exposes on either side of the target qubit
        cos = cos[:, None, None]
        sin = sin[:, None, None]
    view = states.reshape(states.shape[0], 2 ** qubit, 2, 2 ** (n_qubits - qubit - 1))
    zero = view[:, :, 0, :]
    one = view[:, :, 1, :]
    new_zero = cos * zero - sin * one
    new_one = sin * zero + cos * one
    view[:, :, 0, :] = new_zero
    view[:, :, 1, :] = new_one


def _apply_rz(states, angles, qubit, n_qubits):
    """
    In-place RZ(angle) on one qubit for the whole batch - RZ is diagonal, so
    it is just two phase multiplies. Same scalar/per-sample angle handling
    as _apply_ry.
    """
    half = 0.5 * np.asarray(angles, dtype=float)
    phase = np.exp(1j * half)
    if phase.ndim == 1:
        phase = phase[:, None, None]
    view = states.reshape(states.shape[0], 2 ** qubit, 2, 2 ** (n_qubits - qubit - 1))
    view[:, :, 0, :] *= np.conj(phase)
    view[:, :, 1, :] *= phase


def _qubit_bits(n_qubits):
    # bit value of every qubit for every basis index, columns ordered by
    # qubit number so bits[:, q] matches the gate helpers' axis convention
    idx = np.arange(2 ** n_qubits)
    shifts = n_qubits - 1 - np.arange(n_qubits)
    return (idx[:, None] >> shifts[None, :]) & 1


def _ring_phases(n_qubits):
    """
    The CZ entangling ring collapsed into one diagonal. CZ only flips the
    sign of basis states where both of its qubits are 1, so the entire ring
    of CZs is a single precomputable vector of +-1 phases applied with one
    elementwise multiply per layer - far cheaper than gate-by-gate and
    mathematically identical (all CZs commute with each other).
    """
    bits = _qubit_bits(n_qubits)
    if n_qubits >= 3:
        pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    elif n_qubits == 2:
        # a 2-qubit "ring" would apply CZ twice on the same pair and cancel
        # itself out (CZ^2 = I), so keep a single CZ instead
        pairs = [(0, 1)]
    else:
        pairs = []
    phases = np.ones(2 ** n_qubits)
    for a, b in pairs:
        phases = phases * (1.0 - 2.0 * (bits[:, a] & bits[:, b]))
    return phases


def _readout_signs(n_qubits):
    """
    <Z> readout on the LAST qubit: +1 for basis states where its bit is 0,
    -1 where it is 1; <Z> = sum(|amplitude|^2 * sign). The last qubit and not
    qubit 0 because of the CNOT ring that precedes the readout
    (_cnot_ring_source_indices): its final CNOT has qubit 0 as TARGET, and Z
    on a CNOT's target pulls back through the ring to Z_1 Z_2 ... Z_{n-1} -
    an observable without qubit 0 - so with a qubit-0 readout the last
    layer's two rotations on qubit 0 could never change the output (their
    gradient is exactly zero; the module self-check shows it). The last
    qubit is the ring's final CONTROL, and Z on it pulls back to
    Z_0 Z_1 ... Z_{n-1}, which every qubit's rotations move. Until 28 Sept
    2026 the readout was qubit 0; artifacts carry their own sign vector, so
    models trained before then keep serving exactly as trained.
    """
    return 1.0 - 2.0 * _qubit_bits(n_qubits)[:, n_qubits - 1]


def _cnot_ring_source_indices(n_qubits):
    """
    The CNOT entangling ring collapsed into one basis-state permutation:
    every CNOT just relabels basis states (|c,t> -> |c, t xor c>), so the
    whole ring composes into a single index array applied as one gather,
    states[:, src]. The variational ansatz uses CNOTs instead of CZs on
    purpose: CZ (and RZ) are diagonal, so they never change |amplitude|^2 -
    a trailing CZ ring would be completely invisible to the Z-basis readout
    and its layer's parameters would train against a dead gradient. A CNOT
    ring moves probability between basis states and keeps every layer live -
    provided the readout qubit is the ring's final control, not its final
    target, whose Z pulls back past the last layer's gates on it (see
    _readout_signs).
    """
    dim = 2 ** n_qubits
    src = np.arange(dim)
    if n_qubits >= 3:
        pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    elif n_qubits == 2:
        # a 2-qubit "ring" would apply the same CNOT twice and cancel out
        pairs = [(0, 1)]
    else:
        pairs = []
    for control, target in pairs:
        j = np.arange(dim)
        control_bit = (j >> (n_qubits - 1 - control)) & 1
        # each CNOT is its own inverse, so amp_new[j] = amp_old[cnot(j)]
        src = src[j ^ (control_bit << (n_qubits - 1 - target))]
    return src


def _quantum_kernel(states_a, states_b):
    """
    Quantum kernel k(x, y) = |<phi(x)|phi(y)>|^2 (the state-overlap /
    fidelity kernel from the README's quantum-kernel candidate). For the
    train-vs-train case this is the Schur product of a PSD Gram matrix with
    its own conjugate, hence itself PSD - exactly what
    SVC(kernel='precomputed') requires to behave like a proper kernel method.
    """
    return np.abs(states_a @ states_b.conj().T) ** 2


def _sigmoid(x):
    # split by sign so neither tail overflows exp()
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------

def _fit_feature_reduction(X, n_qubits, random_state):
    """
    StandardScaler + PCA down to n_qubits features, fitted ON THE GIVEN
    TRAINING DATA ONLY - the README's leakage rule: fitting preprocessing on
    the full dataset would leak information from the holdout period into
    training. TrainMetaLearner's fit_meta_model only ever passes the
    walk-forward training partition to fit(), so the rule holds as long as
    the transforms are (re)fitted inside fit() and nowhere else.
    """
    scaler = StandardScaler().fit(X)
    # PCA cannot produce more components than features or samples; when the
    # input is narrower than the circuit, _reduce_features zero-pads instead
    n_components = min(n_qubits, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state).fit(scaler.transform(X))
    return scaler, pca


def _reduce_features(scaler, pca, X, n_qubits):
    feats = pca.transform(scaler.transform(np.asarray(X, dtype=float)))
    if feats.shape[1] < n_qubits:
        # RY(0) is the identity, so zero-padding keeps the circuit width
        # fixed without injecting information
        feats = np.hstack([feats, np.zeros((feats.shape[0], n_qubits - feats.shape[1]))])
    return feats


def _balanced_subsample_indices(y, max_samples, rng):
    """
    Class-balanced subsample of at most max_samples rows. The meta-learner
    labels are ~10% positive (draw_size hits out of the whole number range),
    so a uniform random subsample would starve the positive class; instead
    aim for a 50/50 split and let whichever class is scarcer keep all of its
    rows while the other class fills the remaining budget.
    """
    y = np.asarray(y)
    if len(y) <= max_samples:
        return np.arange(len(y))
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y != 1)
    half = max_samples // 2
    n_positive = min(len(positive), half)
    n_negative = min(len(negative), max_samples - n_positive)
    # backfill from positives if the negatives could not fill their share
    n_positive = min(len(positive), max_samples - n_negative)
    keep = np.concatenate([
        rng.choice(positive, size=n_positive, replace=False),
        rng.choice(negative, size=n_negative, replace=False),
    ])
    rng.shuffle(keep)
    return keep


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

class QuantumFeatureMap:
    """
    Angle-encoding feature map from the README's quantum section: each
    (already scaled/PCA-reduced) feature enters as an RY rotation angle times
    encoding_scale, followed by a CZ entangling ring, with the SAME features
    re-uploaded encoding_layers times. The re-uploading is what buys
    expressivity - a single RY+CZ layer produces states whose overlaps are
    still close to a simple trigonometric function of the features. Only
    plain numpy/python attributes, so instances pickle cleanly inside the
    joblib artifacts TrainMetaLearner persists.
    """

    def __init__(self, n_qubits=4, encoding_layers=2, encoding_scale=1.0):
        self.n_qubits = int(n_qubits)
        self.encoding_layers = int(encoding_layers)
        self.encoding_scale = float(encoding_scale)
        self.ring_phases = _ring_phases(self.n_qubits)

    def encode(self, features):
        """Returns the batch of encoded statevectors, shape (batch, 2**n_qubits)."""
        feats = np.asarray(features, dtype=float)
        states = np.zeros((feats.shape[0], 2 ** self.n_qubits), dtype=complex)
        states[:, 0] = 1.0  # every sample starts in |0...0>
        for _ in range(self.encoding_layers):
            for qubit in range(self.n_qubits):
                _apply_ry(states, self.encoding_scale * feats[:, qubit], qubit, self.n_qubits)
            states *= self.ring_phases
        return states


# ---------------------------------------------------------------------------
# Quantum-kernel classifier (README candidate 1)
# ---------------------------------------------------------------------------

class QuantumKernelClassifier:
    """
    Quantum-kernel classifier: encodes every row into a quantum state,
    measures similarity as the state-overlap kernel, and hands the
    precomputed kernel to a classical SVC - the README prefers this as the
    first prototype because it compares cleanly against a classical RBF SVM.
    sklearn-style fit/predict_proba API so Predictor.rankByModel can serve it
    exactly like the existing meta-learners; only numpy/sklearn attributes,
    so joblib round-trips work.
    """

    def __init__(self, n_qubits=4, encoding_layers=2, encoding_scale=1.0, C=1.0,
                 max_train_samples=2000, random_state=0):
        self.n_qubits = int(n_qubits)
        self.encoding_layers = int(encoding_layers)
        self.encoding_scale = float(encoding_scale)
        self.C = float(C)
        self.max_train_samples = int(max_train_samples)
        self.random_state = int(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # scaler+PCA fitted on the given training data ONLY (README leakage
        # rule - see _fit_feature_reduction)
        self._scaler, self._pca = _fit_feature_reduction(X, self.n_qubits, self.random_state)
        feats = _reduce_features(self._scaler, self._pca, X, self.n_qubits)

        # kernel SVMs are O(n^2) in memory and worse in time, so cap the
        # training rows - class-balanced, because the labels are ~10% positive
        rng = np.random.default_rng(self.random_state)
        keep = _balanced_subsample_indices(y, self.max_train_samples, rng)

        self._feature_map = QuantumFeatureMap(self.n_qubits, self.encoding_layers, self.encoding_scale)
        # the retained training encodings are what inference computes kernel
        # columns against, so they are part of the fitted model
        self._train_states = self._feature_map.encode(feats[keep])

        kernel = _quantum_kernel(self._train_states, self._train_states)
        # probability=True because Predictor's rankByModel consumes
        # predict_proba(...)[:, 1]; class_weight='balanced' mirrors the
        # imbalance handling of the existing classical meta-learners
        self._svc = SVC(kernel="precomputed", C=self.C, probability=True,
                        class_weight="balanced", random_state=self.random_state)
        self._svc.fit(kernel, y[keep])
        self.classes_ = self._svc.classes_
        return self

    def _encode(self, X):
        # np.asarray also accepts the plain nested list rankByModel passes
        feats = _reduce_features(self._scaler, self._pca, np.asarray(X, dtype=float), self.n_qubits)
        return self._feature_map.encode(feats)

    def predict_proba(self, X):
        kernel = _quantum_kernel(self._encode(X), self._train_states)
        return self._svc.predict_proba(kernel)

    def predict(self, X):
        # argmax of the Platt-scaled probabilities rather than SVC.predict's
        # raw decision function, so predict() always agrees with the
        # probabilities Predictor actually ranks by
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# Classical control for the quantum kernel (README Q1: "compare with ...
# classical-kernel SVM")
# ---------------------------------------------------------------------------

class RbfKernelClassifier:
    """
    The quantum-kernel classifier with the one quantum piece replaced: the
    same StandardScaler + PCA to the same width (n_features, the control's
    n_qubits), the same class-balanced subsampling, the same probability-
    calibrated, class-balanced SVC - and a classical RBF kernel
    exp(-gamma |x - y|^2) where the quantum row measures the state overlap
    |<phi(x)|phi(y)>|^2. Same pipeline, same features, same evaluation; the
    kernel differs, and so do the SVC settings each row is tuned to (C, the
    sample cap, here the RBF width), because HyperoptQuantum tunes each row
    by its own study - a C chosen for the fidelity kernel is not the right C
    for an RBF. Same picklable sklearn-style API.
    """

    def __init__(self, n_features=4, gamma="scale", C=1.0, max_train_samples=2000, random_state=0):
        self.n_features = int(n_features)
        # "scale" is sklearn's 1 / (n_features * X.var()); a number is the
        # tuned width
        self.gamma = gamma if isinstance(gamma, str) else float(gamma)
        self.C = float(C)
        self.max_train_samples = int(max_train_samples)
        self.random_state = int(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # fitted on the given training data ONLY - the same leakage rule as
        # the quantum classifiers (_fit_feature_reduction)
        self._scaler, self._pca = _fit_feature_reduction(X, self.n_features, self.random_state)
        feats = _reduce_features(self._scaler, self._pca, X, self.n_features)

        rng = np.random.default_rng(self.random_state)
        keep = _balanced_subsample_indices(y, self.max_train_samples, rng)

        self._svc = SVC(kernel="rbf", gamma=self.gamma, C=self.C, probability=True,
                        class_weight="balanced", random_state=self.random_state)
        self._svc.fit(feats[keep], y[keep])
        self.classes_ = self._svc.classes_
        return self

    def predict_proba(self, X):
        feats = _reduce_features(self._scaler, self._pca, np.asarray(X, dtype=float), self.n_features)
        return self._svc.predict_proba(feats)

    def predict(self, X):
        # argmax of the Platt-scaled probabilities, like QuantumKernelClassifier
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# Variational quantum classifier (README candidate 2)
# ---------------------------------------------------------------------------

class VariationalQuantumClassifier:
    """
    Variational quantum classifier: the same angle-encoding feature map,
    followed by num_layers of [trainable RZ+RY on every qubit, CNOT
    entangling ring]. RZ comes BEFORE RY inside each layer and the entangler
    is a CNOT ring rather than CZ, both for the same reason: diagonal gates
    (RZ, CZ) don't change |amplitude|^2, so anything diagonal at the tail of
    the circuit is invisible to the Z-basis readout and its parameters would
    sit on an exactly-zero gradient - see _cnot_ring_source_indices. Readout
    is P(class 1) = (1 + <Z on the last qubit>) / 2 (the ring's final control;
    _readout_signs says why not qubit 0), squashed through a trainable
    affine + sigmoid so the raw expectation can calibrate itself to the class
    balance. Trained with class-weighted binary cross-entropy and Adam in
    numpy; circuit gradients come from the exact parameter-shift rule,
    evaluated over whole minibatches at once. Same picklable sklearn-style
    API as QuantumKernelClassifier.
    """

    def __init__(self, n_qubits=4, num_layers=2, encoding_scale=1.0,
                 learning_rate=0.05, epochs=80, batch_size=128, random_state=0):
        self.n_qubits = int(n_qubits)
        self.num_layers = int(num_layers)
        self.encoding_scale = float(encoding_scale)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        # re-upload depth fixed to mirror QuantumKernelClassifier's default
        # feature map, so both quantum variants see the same encoding
        self.encoding_layers = 2

    def _expectation(self, encoded_states, thetas):
        """
        Runs the variational part of the circuit on already-encoded states
        and returns the raw P(class 1) readout per sample. Split from the
        encoding because the encoding does not depend on the trainable
        parameters - the training loop encodes each minibatch once and reuses
        it for every parameter-shift evaluation.
        """
        states = encoded_states.copy()
        for layer in range(self.num_layers):
            # RZ first, RY second: with RY last, every layer's rotations stay
            # visible to the Z readout (a trailing RZ would be diagonal = dead)
            for qubit in range(self.n_qubits):
                _apply_rz(states, thetas[layer, qubit, 0], qubit, self.n_qubits)
            for qubit in range(self.n_qubits):
                _apply_ry(states, thetas[layer, qubit, 1], qubit, self.n_qubits)
            states = states[:, self._ring_perm]
        return self._readout(states)

    def _readout(self, states):
        # raw P(class 1) = (1 + <Z on the last qubit>) / 2, before the trainable
        # affine + sigmoid calibration
        z = (np.abs(states) ** 2) @ self._z_signs
        return 0.5 * (1.0 + z)

    def _tail_readout(self, states, thetas, layer, kind, qubit):
        """
        Applies the variational circuit from just AFTER rotation (layer,
        kind, qubit) to the end, then reads out - the suffix walk of the
        prefix-cached parameter-shift evaluation in fit(). kind 0 is the RZ
        pass, kind 1 the RY pass, matching _expectation's gate order exactly.
        """
        if kind == 0:
            for q in range(qubit + 1, self.n_qubits):
                _apply_rz(states, thetas[layer, q, 0], q, self.n_qubits)
            for q in range(self.n_qubits):
                _apply_ry(states, thetas[layer, q, 1], q, self.n_qubits)
        else:
            for q in range(qubit + 1, self.n_qubits):
                _apply_ry(states, thetas[layer, q, 1], q, self.n_qubits)
        states = states[:, self._ring_perm]
        for l in range(layer + 1, self.num_layers):
            for q in range(self.n_qubits):
                _apply_rz(states, thetas[l, q, 0], q, self.n_qubits)
            for q in range(self.n_qubits):
                _apply_ry(states, thetas[l, q, 1], q, self.n_qubits)
            states = states[:, self._ring_perm]
        return self._readout(states)

    def _loss_and_gradient(self, params, encoded, y, pos_weight):
        """
        Class-weighted binary cross-entropy of the calibrated readout on one
        already-encoded minibatch, and its exact gradient with respect to
        every parameter: the rotation angles by the parameter-shift rule, the
        two readout parameters analytically. fit() calls this once per
        minibatch; the module self-check (python3 -m src.QuantumModels) calls
        it against central finite differences of the same loss, which is what
        makes the shift rule testable gate by gate instead of through a
        training curve. `params` is the flat vector fit() optimises: the
        (num_layers, n_qubits, 2) angles, then readout weight and bias.
        """
        n_theta = self.num_layers * self.n_qubits * 2
        thetas = params[:n_theta].reshape(self.num_layers, self.n_qubits, 2)
        readout_w = params[n_theta]
        readout_b = params[n_theta + 1]
        n_batch = len(y)

        # Forward pass in circuit order, caching the state right BEFORE every
        # rotation: each parameter-shift evaluation can then restart from its
        # own gate instead of re-running the whole circuit - roughly halves
        # the gate applications per optimizer step without changing the math.
        prefix_states = []
        states = encoded.copy()
        for layer in range(self.num_layers):
            for kind in (0, 1):  # RZ pass, then RY pass
                for qubit in range(self.n_qubits):
                    prefix_states.append((layer, kind, qubit, states.copy()))
                    if kind == 0:
                        _apply_rz(states, thetas[layer, qubit, 0], qubit, self.n_qubits)
                    else:
                        _apply_ry(states, thetas[layer, qubit, 1], qubit, self.n_qubits)
            states = states[:, self._ring_perm]
        praw = self._readout(states)
        prob = _sigmoid(readout_w * praw + readout_b)

        weights = np.where(y == 1, pos_weight, 1.0)
        # the loss itself is only reported (and differentiated numerically by
        # the self-check); the optimiser needs the gradient below
        clipped = np.clip(prob, 1e-12, 1.0 - 1e-12)
        loss = -np.sum(weights * (y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped))) / np.sum(weights)
        # d(weighted BCE)/d(logit) for a sigmoid output collapses to
        # weight * (p - y); normalizing by the weight sum keeps step sizes
        # comparable across batches with different class mixes
        dlogit = weights * (prob - y) / np.sum(weights)

        grad = np.empty_like(params)
        for layer, kind, qubit, before in prefix_states:
            # parameter-shift rule: gates generated by a Pauli/2 have the
            # exact derivative (f(t + pi/2) - f(t - pi/2)) / 2. Both shifted
            # circuits are evaluated over the whole minibatch at once, stacked
            # into one array so the shared suffix is walked a single time.
            angle = thetas[layer, qubit, kind]
            both = np.concatenate([before, before])
            shifted_angles = np.concatenate([
                np.full(n_batch, angle + np.pi / 2),
                np.full(n_batch, angle - np.pi / 2),
            ])
            if kind == 0:
                _apply_rz(both, shifted_angles, qubit, self.n_qubits)
            else:
                _apply_ry(both, shifted_angles, qubit, self.n_qubits)
            praw_both = self._tail_readout(both, thetas, layer, kind, qubit)
            dpraw = 0.5 * (praw_both[:n_batch] - praw_both[n_batch:])
            grad[(layer * self.n_qubits + qubit) * 2 + kind] = readout_w * np.dot(dlogit, dpraw)
        grad[n_theta] = np.dot(dlogit, praw)
        grad[n_theta + 1] = np.sum(dlogit)
        return loss, grad

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # scaler+PCA fitted on the given training data ONLY (README leakage
        # rule - see _fit_feature_reduction)
        self._scaler, self._pca = _fit_feature_reduction(X, self.n_qubits, self.random_state)
        feats = _reduce_features(self._scaler, self._pca, X, self.n_qubits)

        self._feature_map = QuantumFeatureMap(self.n_qubits, self.encoding_layers, self.encoding_scale)
        self._ring_perm = _cnot_ring_source_indices(self.n_qubits)
        self._z_signs = _readout_signs(self.n_qubits)

        rng = np.random.default_rng(self.random_state)
        # small init keeps the ansatz near the identity so early training is
        # driven by the encoded features instead of random circuit noise
        thetas = rng.normal(0.0, 0.1, size=(self.num_layers, self.n_qubits, 2))
        # w=4, b=-2 maps the raw readout's midpoint 0.5 to probability 0.5
        # with unit slope after the sigmoid - a neutral starting calibration
        n_theta = thetas.size
        params = np.concatenate([thetas.ravel(), [4.0, -2.0]])

        positives = int(np.sum(y == 1))
        negatives = len(y) - positives
        # class-weighted BCE: weight positives by the neg/pos ratio so the
        # ~10% positive labels aren't drowned out by the negatives
        pos_weight = (negatives / positives) if positives > 0 else 1.0

        # Adam state
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        # Cost check: parameter-shift needs two evaluations per parameter per
        # step, but with the prefix cache below each one only re-runs the
        # circuit from its own gate onward, so a step costs about one forward
        # pass plus n_theta suffix walks over the (batch, 2**n_qubits) array -
        # milliseconds at the 4-qubit defaults, keeping epochs * batches
        # tractable on CPU.
        n_samples = len(feats)
        batch_size = min(self.batch_size, n_samples)
        for _ in range(self.epochs):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = order[start:start + batch_size]
                yb = y[batch]

                encoded = self._feature_map.encode(feats[batch])
                _, grad = self._loss_and_gradient(params, encoded, yb, pos_weight)

                step += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self._thetas = params[:n_theta].reshape(self.num_layers, self.n_qubits, 2)
        self._readout_w = float(params[n_theta])
        self._readout_b = float(params[n_theta + 1])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        # np.asarray also accepts the plain nested list rankByModel passes
        feats = _reduce_features(self._scaler, self._pca, np.asarray(X, dtype=float), self.n_qubits)
        praw = self._expectation(self._feature_map.encode(feats), self._thetas)
        prob = _sigmoid(self._readout_w * praw + self._readout_b)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Factory functions for TrainMetaLearner
# ---------------------------------------------------------------------------

def fit_quantum_kernel(X, y, params=None):
    """
    Factory matching the fit_logistic_regression(X, y) call convention -
    TrainMetaLearner's fit_meta_model only ever calls fit_func(X, y), so bind
    tuned hyperparameters with functools.partial(fit_quantum_kernel,
    params=bestParams) (params is keyword-friendly and last on purpose).

    bestParams keys (all optional, defaults in parentheses):
        quantumKernel_nQubits          -> n_qubits          (4)
        quantumKernel_encodingLayers   -> encoding_layers   (2)
        quantumKernel_encodingScale    -> encoding_scale    (1.0)
        quantumKernel_C                -> C                 (1.0)
        quantumKernel_maxTrainSamples  -> max_train_samples (2000)
    """
    params = params or {}
    model = QuantumKernelClassifier(
        n_qubits=int(params.get("quantumKernel_nQubits", 4)),
        encoding_layers=int(params.get("quantumKernel_encodingLayers", 2)),
        encoding_scale=float(params.get("quantumKernel_encodingScale", 1.0)),
        C=float(params.get("quantumKernel_C", 1.0)),
        max_train_samples=int(params.get("quantumKernel_maxTrainSamples", 2000)),
    )
    return model.fit(X, y)


def fit_quantum_vqc(X, y, params=None):
    """
    Factory matching the fit_logistic_regression(X, y) call convention - see
    fit_quantum_kernel for the functools.partial binding pattern.

    bestParams keys (all optional, defaults in parentheses):
        quantumVqc_nQubits       -> n_qubits      (4)
        quantumVqc_numLayers     -> num_layers    (2)
        quantumVqc_encodingScale -> encoding_scale (1.0)
        quantumVqc_learningRate  -> learning_rate (0.05)
        quantumVqc_epochs        -> epochs        (80)
        quantumVqc_batchSize     -> batch_size    (128)
    """
    params = params or {}
    model = VariationalQuantumClassifier(
        n_qubits=int(params.get("quantumVqc_nQubits", 4)),
        num_layers=int(params.get("quantumVqc_numLayers", 2)),
        encoding_scale=float(params.get("quantumVqc_encodingScale", 1.0)),
        learning_rate=float(params.get("quantumVqc_learningRate", 0.05)),
        epochs=int(params.get("quantumVqc_epochs", 80)),
        batch_size=int(params.get("quantumVqc_batchSize", 128)),
    )
    return model.fit(X, y)


def fit_classical_svm(X, y, params=None):
    """
    Factory for the classical RBF control row, same call convention as
    fit_quantum_kernel (bind tuned params with functools.partial).

    bestParams keys (all optional, defaults in parentheses):
        classicalSvm_nFeatures        -> n_features        (4, the quantum rows' qubit count)
        classicalSvm_gamma            -> gamma             ("scale"; a number when tuned)
        classicalSvm_C                -> C                 (1.0)
        classicalSvm_maxTrainSamples  -> max_train_samples (2000)
    """
    params = params or {}
    model = RbfKernelClassifier(
        n_features=int(params.get("classicalSvm_nFeatures", 4)),
        gamma=params.get("classicalSvm_gamma", "scale"),
        C=float(params.get("classicalSvm_C", 1.0)),
        max_train_samples=int(params.get("classicalSvm_maxTrainSamples", 2000)),
    )
    return model.fit(X, y)


# ---------------------------------------------------------------------------
# Self-check: python3 -m src.QuantumModels (part of npm test)
# ---------------------------------------------------------------------------

def _self_check():
    import pickle

    rng = np.random.default_rng(3)

    # 1. The parameter-shift gradient against central finite differences of
    #    the very loss fit() minimises, for every parameter: each RZ and RY
    #    angle in every layer (the shift rule is exact only for gates
    #    generated by a Pauli/2, so a wrong gate convention or a wrong suffix
    #    walk in _tail_readout shows up here as a mismatch) and the two
    #    readout parameters. Angles are drawn wide so the check does not sit
    #    in the near-identity region where every gradient is small.
    X = rng.normal(size=(48, 6))
    y = (rng.random(48) < 0.3).astype(int)
    vqc = VariationalQuantumClassifier(n_qubits=4, num_layers=2, encoding_scale=0.7)
    vqc._scaler, vqc._pca = _fit_feature_reduction(X, vqc.n_qubits, 0)
    feats = _reduce_features(vqc._scaler, vqc._pca, X, vqc.n_qubits)
    vqc._feature_map = QuantumFeatureMap(vqc.n_qubits, vqc.encoding_layers, vqc.encoding_scale)
    vqc._ring_perm = _cnot_ring_source_indices(vqc.n_qubits)
    vqc._z_signs = _readout_signs(vqc.n_qubits)
    encoded = vqc._feature_map.encode(feats)
    n_theta = vqc.num_layers * vqc.n_qubits * 2
    params = np.concatenate([rng.normal(0.0, 0.8, size=n_theta), [3.0, -1.0]])
    pos_weight = (len(y) - y.sum()) / y.sum()

    loss, grad = vqc._loss_and_gradient(params, encoded, y, pos_weight)
    eps = 1e-6
    numeric = np.empty_like(params)
    for i in range(len(params)):
        up, down = params.copy(), params.copy()
        up[i] += eps
        down[i] -= eps
        numeric[i] = (vqc._loss_and_gradient(up, encoded, y, pos_weight)[0]
                      - vqc._loss_and_gradient(down, encoded, y, pos_weight)[0]) / (2 * eps)
    worst = float(np.max(np.abs(numeric - grad)))
    largest = float(np.max(np.abs(grad)))
    smallest = float(np.min(np.abs(grad[:n_theta])))
    assert np.isfinite(loss) and largest > 1e-3, f"degenerate check point: loss={loss}, max|grad|={largest}"
    # every angle must be live, or the comparison below is 0 == 0 for it and
    # proves nothing about that gate
    assert smallest > 1e-5, f"an angle is dead at the check point: min |grad| = {smallest:.1e}"
    assert worst < 1e-6, f"parameter-shift gradient disagrees with finite differences by {worst:.2e}"
    kinds = ["RZ", "RY"]
    per_kind = {k: float(np.max(np.abs((numeric - grad)[[i for i in range(n_theta) if i % 2 == kinds.index(k)]])))
                for k in kinds}
    print(f"parameter-shift vs finite differences over {len(params)} parameters: worst |diff| = {worst:.1e} "
          f"(RZ {per_kind['RZ']:.1e}, RY {per_kind['RY']:.1e}, readout "
          f"{float(np.max(np.abs((numeric - grad)[n_theta:]))):.1e}); max |grad| = {largest:.3f}")

    # 1b. Why the readout is the last qubit: with Z on qubit 0 - the ring's
    #     final target - the last layer's two rotations on qubit 0 are inert
    #     for every input, and the comparison above would pass them as
    #     0 == 0. Shown here so the reason stays checked, not just written.
    vqc._z_signs = 1.0 - 2.0 * _qubit_bits(vqc.n_qubits)[:, 0]
    _, grad_qubit0 = vqc._loss_and_gradient(params, encoded, y, pos_weight)
    vqc._z_signs = _readout_signs(vqc.n_qubits)
    dead = [((vqc.num_layers - 1) * vqc.n_qubits + 0) * 2 + kind for kind in (0, 1)]
    assert np.max(np.abs(grad_qubit0[dead])) < 1e-12, "expected a qubit-0 readout to leave the last layer's qubit-0 angles dead"
    print(f"readout on qubit 0 would leave the last layer's two qubit-0 angles dead (|grad| < 1e-12); "
          f"on the last qubit every angle is live (min |grad| = {smallest:.1e})")

    # 1c. The training loop end to end on the refactored gradient step.
    trained = fit_quantum_vqc(X, y, {"quantumVqc_epochs": 3, "quantumVqc_batchSize": 16})
    proba_vqc = trained.predict_proba(X)
    assert proba_vqc.shape == (len(X), 2) and np.allclose(proba_vqc.sum(axis=1), 1.0)
    print("VQC fits end to end and returns calibrated two-column probabilities")

    # 2. A shift rule with the wrong shift would pass nothing above; make sure
    #    the check itself has teeth by breaking it on purpose.
    broken = grad.copy()
    broken[0] *= 1.01
    assert np.max(np.abs(numeric - broken)) > 1e-6 or abs(grad[0]) < 1e-4, "the check cannot tell a 1% error"
    print("a 1% error in one angle's gradient is detected")

    # 3. The fidelity kernel is a kernel: symmetric, unit diagonal, PSD.
    states = QuantumFeatureMap(4, 2, 0.7).encode(feats)
    kernel = _quantum_kernel(states, states)
    assert np.allclose(kernel, kernel.T) and np.allclose(np.diag(kernel), 1.0)
    assert np.min(np.linalg.eigvalsh(kernel)) > -1e-9, "quantum kernel is not PSD"
    print(f"quantum kernel on {len(feats)} rows: symmetric, unit diagonal, smallest eigenvalue "
          f"{np.min(np.linalg.eigvalsh(kernel)):.1e}")

    # 4. The classical control: fits, gives calibrated two-column
    #    probabilities, learns a separable toy problem, survives pickling
    #    (the joblib artifact), and takes a tuned numeric gamma.
    #    Four columns, so the PCA to four features is a rotation and keeps the
    #    two the label depends on (on wider isotropic noise it would not).
    X_toy = rng.normal(size=(400, 4))
    y_toy = (X_toy[:, 0] * X_toy[:, 1] > 0).astype(int)       # XOR-like: not linearly separable
    control = fit_classical_svm(X_toy[:300], y_toy[:300])
    proba = control.predict_proba(X_toy[300:])
    assert proba.shape == (100, 2) and np.allclose(proba.sum(axis=1), 1.0)
    accuracy = float(np.mean(control.predict(X_toy[300:]) == y_toy[300:]))
    assert accuracy > 0.75, f"RBF control only {accuracy:.2f} accurate on a separable toy problem"
    again = pickle.loads(pickle.dumps(control))
    assert np.array_equal(again.predict_proba(X_toy[300:]), proba)
    tuned = fit_classical_svm(X_toy[:300], y_toy[:300],
                              params={"classicalSvm_gamma": 0.5, "classicalSvm_C": 10.0,
                                      "classicalSvm_maxTrainSamples": 200})
    assert tuned._svc.gamma == 0.5 and len(tuned._svc.support_) <= 200
    print(f"RBF control: {accuracy:.2f} accurate on the XOR-like toy holdout, pickle round-trip exact, "
          f"tuned gamma/C/sample cap applied")

    # 5. Both kernel rows and the VQC share one preprocessing: the same
    #    scaler + PCA reduce to the same width on the same rows.
    quantum = QuantumKernelClassifier(max_train_samples=200).fit(X_toy[:300], y_toy[:300])
    assert np.allclose(quantum._pca.components_, control._pca.components_)
    print("quantum kernel row and its classical control reduce features identically")
    print("QuantumModels self-check OK")


if __name__ == "__main__":
    _self_check()
