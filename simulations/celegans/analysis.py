"""
Analysis toolkit for C. elegans predictive-horizon simulations.

Functions
---------
d_eff           Effective dimensionality via participation ratio
d_eff_timeseries    D_eff in sliding (non-overlapping) windows
gaussian_mi     Mutual information via Gaussian covariance
mi_profile      Future MI at multiple time lags
motor_correlation   Pearson correlation between motor neuron groups
spectral_overlap    Spectral overlap between internal and env signals
classify_trajectory D_eff trajectory classification
"""

import numpy as np
from scipy.stats import pearsonr


# ──────────────────────────────────────────────────────────────────
# Effective dimensionality
# ──────────────────────────────────────────────────────────────────

def d_eff(states):
    """Effective dimensionality via participation ratio.

    Parameters
    ----------
    states : ndarray, shape (T, D)
        Time-series array (T samples, D dimensions).

    Returns
    -------
    float
        Participation ratio: (sum lambda)^2 / sum(lambda^2),
        where lambda are the eigenvalues of the covariance matrix
        (obtained via SVD).
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
        return 0.0

    X = states - states.mean(axis=0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s ** 2 / len(X)
    ev = ev[ev > 1e-10]

    if len(ev) == 0:
        return 0.0
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def d_eff_timeseries(states, window):
    """D_eff computed in non-overlapping windows.

    Parameters
    ----------
    states : ndarray, shape (T, D)
        Time-series array.
    window : int
        Window size in samples.

    Returns
    -------
    ndarray, shape (n_windows,)
        D_eff for each window.
    """
    states = np.asarray(states, dtype=float)
    T = states.shape[0]
    n_windows = T // window
    if n_windows == 0:
        return np.array([])

    result = np.zeros(n_windows)
    for i in range(n_windows):
        chunk = states[i * window : (i + 1) * window]
        result[i] = d_eff(chunk)
    return result


# ──────────────────────────────────────────────────────────────────
# Mutual information (Gaussian approximation)
# ──────────────────────────────────────────────────────────────────

def gaussian_mi(X, Y, reg=0.01):
    """Mutual information via Gaussian covariance estimate.

    Parameters
    ----------
    X : ndarray, shape (n, dX)
    Y : ndarray, shape (n, dY)
    reg : float
        Regularisation coefficient; ``reg * I`` is added to each
        covariance matrix to ensure positive-definiteness.

    Returns
    -------
    float
        MI in nats (non-negative).  Returns 0.0 if any log-det
        sign is non-positive.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]
    dX, dY = X.shape[1], Y.shape[1]

    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    C_XX = Xc.T @ Xc / n + reg * np.eye(dX)
    C_YY = Yc.T @ Yc / n + reg * np.eye(dY)
    XY = np.hstack([Xc, Yc])
    C_XY = XY.T @ XY / n + reg * np.eye(dX + dY)

    s1, l1 = np.linalg.slogdet(C_XX)
    s2, l2 = np.linalg.slogdet(C_YY)
    s3, l3 = np.linalg.slogdet(C_XY)

    if s1 <= 0 or s2 <= 0 or s3 <= 0:
        return 0.0
    return float(max(0.0, 0.5 * (l1 + l2 - l3)))


def mi_profile(states, env_signal, delta_ts, dt_stored,
               n_pcs=50, max_samples=3000):
    """Future mutual information at multiple time lags.

    Parameters
    ----------
    states : ndarray, shape (T, D)
        Internal state time series.
    env_signal : ndarray, shape (T, D_env)
        Environment signal (same time base as *states*).
    delta_ts : array-like
        Prediction horizons (in the same time units as *dt_stored*).
    dt_stored : float
        Sampling interval of *states* and *env_signal*.
    n_pcs : int
        Number of PCs for internal states (if D > n_pcs).
    max_samples : int
        Maximum number of samples for MI estimation.

    Returns
    -------
    ndarray, shape (len(delta_ts),)
        MI (nats) at each lag.
    """
    states = np.asarray(states, dtype=float)
    env_signal = np.asarray(env_signal, dtype=float)
    n_t = states.shape[0]

    # PCA-reduce internal states
    X = states - states.mean(axis=0)
    if n_pcs < X.shape[1]:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        X = X @ Vt[:n_pcs].T

    # PCA-reduce environment
    E = env_signal - env_signal.mean(axis=0)
    if E.ndim == 1:
        E = E[:, None]
    if E.shape[1] > 20:
        _, _, Ve = np.linalg.svd(E, full_matrices=False)
        E = E @ Ve[:20].T

    I = np.zeros(len(delta_ts))
    for i, lag in enumerate(delta_ts):
        ls = max(1, int(lag / dt_stored))
        if ls >= n_t - 50:
            continue
        Xn = X[:-ls]
        Ef = E[ls:]
        ml = min(len(Xn), len(Ef))
        if ml > max_samples:
            idx = np.random.choice(ml, max_samples, replace=False)
            Xn, Ef = Xn[idx], Ef[idx]
        else:
            Xn, Ef = Xn[:ml], Ef[:ml]
        I[i] = gaussian_mi(Xn, Ef)
    return I


# ──────────────────────────────────────────────────────────────────
# Motor coordination
# ──────────────────────────────────────────────────────────────────

def motor_correlation(theta_hist, group_a, group_b):
    """Pearson correlation between group-averaged cos(theta).

    Parameters
    ----------
    theta_hist : ndarray, shape (T, N)
        Phase history for all N neurons.
    group_a, group_b : array-like of int
        Neuron indices for each motor group.

    Returns
    -------
    float
        Pearson r between the two group-averaged signals.
        Returns 0.0 if either group is empty or has zero variance.
    """
    theta_hist = np.asarray(theta_hist, dtype=float)
    group_a = list(group_a)
    group_b = list(group_b)

    if not group_a or not group_b:
        return 0.0

    a = np.mean(np.cos(theta_hist[:, group_a]), axis=1)
    b = np.mean(np.cos(theta_hist[:, group_b]), axis=1)

    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0

    r, _ = pearsonr(a, b)
    return float(r)


# ──────────────────────────────────────────────────────────────────
# Spectral overlap
# ──────────────────────────────────────────────────────────────────

def spectral_overlap(internal_signal, env_signal, dt_stored):
    """Spectral overlap between internal dynamics and environment.

    Parameters
    ----------
    internal_signal : ndarray, shape (T, D1)
        Internal state time series.
    env_signal : ndarray, shape (T, D2)
        Environment signal.
    dt_stored : float
        Sampling interval (seconds).

    Returns
    -------
    dict
        freqs           : 1-D array of frequency bins (Hz)
        psd_internal    : mean PSD of internal signal
        psd_env         : mean PSD of environment signal
        overlap         : min(norm_internal, norm_env) at each freq
        total_overlap   : weighted sum (env-weighted)
    """
    internal_signal = np.asarray(internal_signal, dtype=float)
    env_signal = np.asarray(env_signal, dtype=float)

    if internal_signal.ndim == 1:
        internal_signal = internal_signal[:, None]
    if env_signal.ndim == 1:
        env_signal = env_signal[:, None]

    T = internal_signal.shape[0]

    # FFT and mean PSD across dimensions
    fft_int = np.fft.rfft(internal_signal, axis=0)
    psd_int = np.mean(np.abs(fft_int) ** 2, axis=1)

    fft_env = np.fft.rfft(env_signal, axis=0)
    psd_env = np.mean(np.abs(fft_env) ** 2, axis=1)

    freqs = np.fft.rfftfreq(T, d=dt_stored)

    # Normalise each PSD to [0, 1]
    max_int = psd_int.max()
    max_env = psd_env.max()
    norm_int = psd_int / max_int if max_int > 0 else psd_int
    norm_env = psd_env / max_env if max_env > 0 else psd_env

    # Overlap = min of normalised PSDs
    overlap = np.minimum(norm_int, norm_env)

    # Env-weighted total
    env_weight_sum = psd_env.sum()
    if env_weight_sum > 0:
        env_weight = psd_env / env_weight_sum
        total_overlap = float(np.sum(overlap * env_weight))
    else:
        total_overlap = 0.0

    return dict(
        freqs=freqs,
        psd_internal=psd_int,
        psd_env=psd_env,
        overlap=overlap,
        total_overlap=total_overlap,
    )


# ──────────────────────────────────────────────────────────────────
# Trajectory classification
# ──────────────────────────────────────────────────────────────────

def classify_trajectory(d_eff_series, recovery_threshold=0.8):
    """Classify a D_eff trajectory as resilient, fragile, or terminal.

    Parameters
    ----------
    d_eff_series : array-like
        Sequence of D_eff values over time.
    recovery_threshold : float
        Ratio of last-quarter mean to first-quarter mean above which
        the trajectory is classified as 'resilient'.

    Returns
    -------
    str
        One of 'resilient', 'fragile', 'terminal', or 'unknown'.
    """
    s = np.asarray(d_eff_series, dtype=float)
    if len(s) < 4:
        return 'unknown'

    q = len(s) // 4
    first_q = s[:q].mean()
    last_q = s[-q:].mean()

    if first_q < 1e-10:
        return 'unknown'

    ratio = last_q / first_q

    if ratio >= recovery_threshold:
        return 'resilient'

    if ratio >= 0.4:
        # Check that the second half isn't mostly declining
        half = len(s) // 2
        second_half = s[half:]
        diffs = np.diff(second_half)
        if np.sum(diffs < 0) <= len(diffs) / 2:
            return 'fragile'

    return 'terminal'
