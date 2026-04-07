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
        Amplitude for each mode.  If None, uses 1/f-like scaling:
        ``A_k = 1.0 / sqrt(k + 1)``.
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
        amplitudes = np.array([1.0 / np.sqrt(k + 1) for k in range(K)])
    else:
        amplitudes = np.asarray(amplitudes, dtype=float)

    env_total = np.zeros((n_steps, n_channels))
    env_modes = []

    for k in range(K):
        tau_k = taus[k]
        amp_k = amplitudes[k]
        decay = np.exp(-dt / tau_k)
        noise_scale = amp_k * np.sqrt(2.0 * dt / tau_k)

        mode = np.zeros((n_steps, n_channels))
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
    for t in range(1, n_steps):
        env[t] = env[t - 1] * decay + noise_scale * rng.randn(n_channels)

    return env
