"""
Multi-mode oscillatory environment generator for C. elegans simulations.

Generates environmental signals as superpositions of Ornstein-Uhlenbeck
processes at different timescales, with 1/f spectral scaling so that slow
environmental modes (food regime, temperature drift) carry larger amplitude
than fast fluctuations (momentary vibration).

Each sensory channel receives the same temporal modes but with independent
noise realisations, so channels share spectral structure but not exact
waveforms.
"""

import numpy as np


def default_taus(K=10):
    """Return K timescales log-spaced from 0.1 s to 3000 s.

    Parameters
    ----------
    K : int
        Number of timescales.

    Returns
    -------
    taus : ndarray, shape (K,)
        Autocorrelation times in seconds, ascending.
    """
    return np.logspace(np.log10(0.1), np.log10(3000.0), K)


def default_amplitudes(taus):
    """Return 1/f-like amplitudes for an ordered timescale spectrum.

    The timescales are assumed to represent progressively slower
    environmental modes. We keep the amplitudes O(1) while ensuring that
    slower modes have larger stationary variance than faster modes.

    Parameters
    ----------
    taus : array-like, shape (K,)
        Autocorrelation times in seconds.

    Returns
    -------
    amplitudes : ndarray, shape (K,)
        Monotonically increasing amplitudes when ``taus`` is ascending.
    """
    taus = np.asarray(taus, dtype=float)
    if taus.ndim != 1 or len(taus) == 0:
        return np.array([], dtype=float)

    # Rank modes by timescale so the slowest mode gets the largest amplitude
    # while preserving the modest O(1) scale used by the original runs.
    tau_rank = np.argsort(np.argsort(taus))
    return 1.0 / np.sqrt(len(taus) - tau_rank)


def generate_multimode_env(n_steps, dt, n_channels, taus, amplitudes=None,
                           seed=42):
    """Generate a multi-mode OU environment signal.

    The environment is the sum of K independent Ornstein-Uhlenbeck processes,
    each with its own autocorrelation time and amplitude.  Every sensory
    channel gets independent noise draws, so channels share temporal
    statistics but not exact trajectories.

    Parameters
    ----------
    n_steps : int
        Number of simulation time steps.
    dt : float
        Time step in seconds.
    n_channels : int
        Number of independent sensory channels.
    taus : array-like, shape (K,)
        Autocorrelation time for each OU mode (seconds).
    amplitudes : array-like, shape (K,), optional
        Stationary standard deviation for each mode. If None, uses a
        rank-ordered 1/f-like profile so slower modes have larger
        amplitude than faster modes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    env_total : ndarray, shape (n_steps, n_channels)
        Summed environment signal across all modes.
    env_modes : list of ndarray, each shape (n_steps, n_channels)
        Per-mode OU time series.
    taus : ndarray, shape (K,)
        The timescales used (echoed back for convenience).
    """
    rng = np.random.RandomState(seed)
    taus = np.asarray(taus, dtype=float)
    K = len(taus)

    if amplitudes is None:
        amplitudes = default_amplitudes(taus)
    else:
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != taus.shape:
            raise ValueError("amplitudes must have the same shape as taus")

    env_total = np.zeros((n_steps, n_channels))
    env_modes = []

    for k in range(K):
        tau_k = taus[k]
        amp_k = amplitudes[k]
        decay = np.exp(-dt / tau_k)
        noise_scale = amp_k * np.sqrt(2.0 * dt / tau_k)

        mode = np.zeros((n_steps, n_channels))
        # Warm-start from the stationary distribution so slow modes are
        # represented correctly even in finite-time simulations.
        mode[0] = amp_k * rng.randn(n_channels)
        for t in range(1, n_steps):
            mode[t] = mode[t - 1] * decay + noise_scale * rng.randn(n_channels)

        env_total += mode
        env_modes.append(mode)

    return env_total, env_modes, taus


def generate_simple_env(n_steps, dt, n_channels, tau=2.0, sigma=0.5,
                        seed=42):
    """Generate a single-timescale OU environment (backward-compatible).

    This reproduces the simple OU environment used in earlier scripts,
    providing a drop-in replacement with explicit seed control.

    Parameters
    ----------
    n_steps : int
        Number of simulation time steps.
    dt : float
        Time step in seconds.
    n_channels : int
        Number of independent sensory channels.
    tau : float
        Autocorrelation time in seconds.
    sigma : float
        Amplitude (standard deviation scale) of the OU process.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    env : ndarray, shape (n_steps, n_channels)
        The environment signal.
    """
    rng = np.random.RandomState(seed)
    decay = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(2.0 * dt / tau)

    env = np.zeros((n_steps, n_channels))
    env[0] = sigma * rng.randn(n_channels)
    for t in range(1, n_steps):
        env[t] = env[t - 1] * decay + noise_scale * rng.randn(n_channels)

    return env
