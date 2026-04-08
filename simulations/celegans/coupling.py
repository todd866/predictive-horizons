"""
Coupling matrix construction and body-axis grid mapping for C. elegans.

Normalizes raw connectome matrices from data.py into simulation-ready
coupling weights, and builds the 1D body-axis grid used by the coherence
field model.

Reference: celegans_trilayer.py lines 137-234.
"""

import numpy as np
from collections import namedtuple

# ── Named result types ─────────────────────────────────────────────

CouplingMatrices = namedtuple('CouplingMatrices', [
    'W_chem_norm',   # (N, N) chemical synapses, column-normalized
    'W_gap_norm',    # (N, N) gap junctions, column-normalized
    'W_eph_norm',    # (N, N) ephaptic (distance-based Gaussian), row-normalized
    'W_npp_norm',    # (N, N) neuropeptide, max-normalized
    'W_mono_norm',   # (N, N) monoamine, max-normalized
])


# ── Coupling matrix construction ───────────────────────────────────

def build_coupling_matrices(worm_data, eph_sigma=0.8):
    """
    Normalize raw adjacency matrices into simulation-ready coupling weights.

    Parameters
    ----------
    worm_data : WormData namedtuple (from data.py)
        Must have fields: W_chem, W_gap, W_npp, W_mono, dist_3d.
    eph_sigma : float
        Gaussian width for ephaptic coupling, in atlas units
        (roughly 1 body width). Default 0.8.

    Returns
    -------
    CouplingMatrices namedtuple with normalized weight matrices.
    """
    # Chemical synapses: normalize by column sum (incoming connections)
    W_chem = worm_data.W_chem
    W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)

    # Gap junctions: normalize by column sum
    W_gap = worm_data.W_gap
    W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

    # Ephaptic coupling: Gaussian kernel on 3D distances
    dist_3d = worm_data.dist_3d
    W_eph = np.exp(-dist_3d**2 / (2 * eph_sigma**2))
    np.fill_diagonal(W_eph, 0)
    # Row-normalize (each neuron's total ephaptic input sums to 1)
    W_eph_norm = W_eph / np.maximum(W_eph.sum(axis=1, keepdims=True), 1)

    # Neuropeptide network: normalize by max value (preserve relative strengths)
    W_npp = worm_data.W_npp
    if W_npp.max() > 0:
        W_npp_norm = W_npp / W_npp.max()
    else:
        W_npp_norm = W_npp.copy()

    # Monoamine network: normalize by max value
    W_mono = worm_data.W_mono
    if W_mono.max() > 0:
        W_mono_norm = W_mono / W_mono.max()
    else:
        W_mono_norm = W_mono.copy()

    return CouplingMatrices(
        W_chem_norm=W_chem_norm,
        W_gap_norm=W_gap_norm,
        W_eph_norm=W_eph_norm,
        W_npp_norm=W_npp_norm,
        W_mono_norm=W_mono_norm,
    )


# ── Body-axis grid mapping ─────────────────────────────────────────

def build_grid_mapping(pos_1d, Nx, sigma_sp=0.08):
    """
    Map neurons to a 1D body-axis grid for the coherence field.

    Parameters
    ----------
    pos_1d : (N,) array
        Body-axis position of each neuron, in [0, 1].
    Nx : int
        Number of grid points along the body axis.
    sigma_sp : float
        Gaussian width for neuron-to-grid weighting. Default 0.08.

    Returns
    -------
    dict with keys:
        grid_x          : (Nx,)    grid positions in [0, 1]
        neuron_body_idx : (N,)     nearest grid index per neuron
        neuron_w_grid   : (Nx, N)  Gaussian-weighted neuron-to-grid mapping
                                   (normalized per grid point)
        n_left          : (N,)     left grid index for linear interpolation
        n_frac          : (N,)     fractional position for interpolation
        dx              : float    grid spacing (1.0 / Nx)
    """
    N = len(pos_1d)
    dx = 1.0 / Nx

    # Uniform grid along body axis
    grid_x = np.linspace(0, 1, Nx)

    # Nearest grid index per neuron
    neuron_body_idx = np.array([
        np.argmin(np.abs(grid_x - p)) for p in pos_1d
    ])

    # Gaussian-weighted neuron-to-grid mapping
    # neuron_w_grid[gi, ni] = how much neuron ni contributes to grid point gi
    neuron_w_grid = np.zeros((Nx, N))
    for gi in range(Nx):
        for ni in range(N):
            neuron_w_grid[gi, ni] = np.exp(
                -(grid_x[gi] - pos_1d[ni])**2 / (2 * sigma_sp**2)
            )
        s = neuron_w_grid[gi].sum()
        if s > 0:
            neuron_w_grid[gi] /= s

    # Linear interpolation indices
    n_left = np.clip(np.floor(pos_1d * (Nx - 1)).astype(int), 0, Nx - 2)
    n_frac = pos_1d * (Nx - 1) - n_left

    return {
        'grid_x': grid_x,
        'neuron_body_idx': neuron_body_idx,
        'neuron_w_grid': neuron_w_grid,
        'n_left': n_left,
        'n_frac': n_frac,
        'dx': dx,
    }


def build_frustration_matrix(pos_1d, motor_indices, alpha=1.0,
                              W_chem=None, W_gap=None):
    """Build Sakaguchi-Kuramoto frustration matrix for motor connections.

    Returns an (N, N) matrix where entry [i, j] = alpha * (pos_j - pos_i)
    for motor-motor pairs connected by chemical or gap junctions.
    Zero elsewhere, so non-motor and unconnected pairs are unaffected.

    When used in coupling: sin(theta_i - theta_j - frustration[i,j]),
    this creates a preferred head-leads-tail phase gradient in the
    motor neuron chain, enabling traveling wave propagation.

    Parameters
    ----------
    pos_1d : ndarray, shape (N,)
        Body-axis position of each neuron in [0, 1].
    motor_indices : list of int
        Indices of motor neurons (VA, VB, DA, DB, etc.).
    alpha : float
        Frustration strength (radians per body-length of separation).
    W_chem : ndarray, shape (N, N), optional
        Chemical synapse weight matrix.  If given, motor-motor chemical
        connections also receive frustration.
    W_gap : ndarray, shape (N, N), optional
        Gap junction weight matrix.  If given, motor-motor gap junctions
        also receive frustration.

    Returns
    -------
    frustration : ndarray, shape (N, N)
        Frustration offset matrix.
    """
    N = len(pos_1d)
    motor_mask = np.zeros(N, dtype=bool)
    motor_mask[motor_indices] = True
    mm = motor_mask[:, None] & motor_mask[None, :]
    # Only frustrate connected pairs
    connected = np.zeros((N, N), dtype=bool)
    if W_gap is not None:
        connected |= (W_gap > 0)
    if W_chem is not None:
        connected |= (W_chem > 0)
    if W_gap is None and W_chem is None:
        # Fallback: all motor-motor pairs
        connected = np.ones((N, N), dtype=bool)
    mask = mm & connected
    pos_offset = pos_1d[None, :] - pos_1d[:, None]
    return np.where(mask, alpha * pos_offset, 0.0)
