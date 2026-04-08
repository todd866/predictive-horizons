"""
Phase oscillator dynamics and coherence field evolution for C. elegans.

Extracts the core simulation stepping from celegans_trilayer.py into a
reusable module: phase oscillator update, coherence field Landau-Ginzburg
dynamics, synaptic depression, and body mechanics.

Reference: celegans_trilayer.py lines 280-502.
"""

import numpy as np

from .muscle import compute_dv_drive


# ── Helper functions ──────────────────────────────────────────────────

def laplacian_1d(f, dx):
    """1D Laplacian with Neumann (zero-flux) boundary conditions.

    Parameters
    ----------
    f : ndarray, shape (Nx,)
        Field values on a uniform 1D grid.
    dx : float
        Grid spacing.

    Returns
    -------
    lap : ndarray, shape (Nx,)
        Discrete Laplacian of *f*.
    """
    lap = np.zeros_like(f)
    lap[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
    lap[0] = (f[1] - f[0]) / dx**2
    lap[-1] = (f[-2] - f[-1]) / dx**2
    return lap


def interp_field(field, n_left, n_frac):
    """Linear interpolation of a 1D field at neuron positions.

    Parameters
    ----------
    field : ndarray, shape (Nx,)
        Values on the body-axis grid.
    n_left : ndarray, shape (N,), int
        Left grid index for each neuron (from ``build_grid_mapping``).
    n_frac : ndarray, shape (N,)
        Fractional position between ``n_left`` and ``n_left + 1``.

    Returns
    -------
    vals : ndarray, shape (N,)
        Interpolated field value at each neuron's body-axis position.
    """
    return field[n_left] * (1 - n_frac) + field[n_left + 1] * n_frac


# ── Parameter container ──────────────────────────────────────────────

class WormParams:
    """All coupling strengths and timescales for the trilayer model.

    Attributes mirror the coupling parameters in celegans_trilayer.py
    lines 253-283.  Every attribute has a sensible default; override by
    passing keyword arguments to ``__init__``.
    """

    def __init__(self, **kwargs):
        # Layer 1: Connectome (fast, always on)
        self.K_chem = 0.4
        self.K_gap = 0.8

        # Layer 2: Ephaptic (fast, distance-dependent, gated by coherence)
        self.K_eph = 1.0

        # Layer 3: Neuropeptide (slow)
        self.K_npp = 0.5
        self.tau_npp = 2.0

        # Sensory drive
        self.K_drive = 1.0
        self.sigma_noise = 0.12

        # Adaptation
        self.tau_adapt = 3.0
        self.g_adapt = 0.4

        # Short-term depression (all layers)
        self.tau_rec_gap = 2.0
        self.tau_rec_eph = 1.5
        self.tau_rec_npp = 5.0
        self.tau_dep = 0.3
        self.U_init = 0.8

        # Coherence field (Landau-Ginzburg)
        self.a_LG = 0.5
        self.b_LG = 2.0
        self.kappa_phi = 0.015
        self.sigma_phi = 0.3
        self.gamma_pump = 0.7
        self.tau_phi = 0.5

        # Body mechanics
        self.epsilon_body = 3.0
        self.gamma_body = 1.5
        self.K_muscle = 0.8
        self.sigma_body = 0.01
        self.tau_muscle = 0.05    # muscle activation time constant (50 ms)
        self.K_muscle_inh = 0.6   # inhibitory gain (VD/DD, relative to excitatory)

        # Proprioception
        self.proprio_gain = 1.5      # proprioceptive coupling strength
        self.proprio_delta_s = 0.08  # anterior offset in body-lengths

        # Override any defaults with caller-supplied values
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"WormParams has no parameter '{key}'")
            setattr(self, key, val)


# ── Mutable simulation state ─────────────────────────────────────────

class WormState:
    """All mutable simulation state for one worm.

    Parameters
    ----------
    N : int
        Number of neurons.
    Nx : int
        Number of body-axis grid points.
    seed : int
        Base random seed.  Three separate RandomState streams are created
        at ``seed + 100``, ``seed + 200``, ``seed + 300`` for noise,
        coherence field, and body mechanics respectively.
    """

    def __init__(self, N, Nx, seed=77):
        rng_init = np.random.RandomState(seed)

        # Neural state
        self.theta = 2 * np.pi * rng_init.rand(N)
        self.adapt = np.zeros(N)
        self.npp_mod = np.zeros(N)

        # Coherence field
        self.phi = np.ones(Nx) * 0.15

        # Body mechanics
        self.kappa = np.zeros(Nx)
        self.kappa_dot = np.zeros(Nx)
        self.m_dorsal = np.zeros(Nx)   # dorsal muscle activation
        self.m_ventral = np.zeros(Nx)  # ventral muscle activation

        # Synaptic resources (per layer, NxN)
        self.u_gap = np.ones((N, N)) * 0.8
        self.u_eph = np.ones((N, N)) * 0.8
        self.u_npp = np.ones((N, N)) * 0.8

        # Separate RNG streams
        self.rng_noise = np.random.RandomState(seed + 100)
        self.rng_phi = np.random.RandomState(seed + 200)
        self.rng_body = np.random.RandomState(seed + 300)


# ── Frequency assignment ──────────────────────────────────────────────

def assign_frequencies(N, sensory, motor, inter, seed=99):
    """Assign intrinsic oscillation frequencies by neuron class.

    Interneurons are slowest, sensory intermediate, motor fastest —
    reflecting the functional gradient from integration to action.

    Parameters
    ----------
    N : int
        Total number of neurons.
    sensory, motor, inter : list of int
        Neuron indices for each functional class.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    omegas : ndarray, shape (N,)
        Intrinsic angular frequencies (rad/s).
    """
    rng = np.random.RandomState(seed)
    omegas = np.zeros(N)

    for i in inter:
        omegas[i] = 2 * np.pi * (0.7 + 0.3 * rng.rand())
    for i in sensory:
        omegas[i] = 2 * np.pi * (0.9 + 0.3 * rng.rand())
    for i in motor:
        omegas[i] = 2 * np.pi * (1.0 + 0.3 * rng.rand())

    return omegas


# ── Main simulation step ─────────────────────────────────────────────

def step(state, params, omegas, coupling_matrices, grid_mapping,
         sensory_indices, motor_indices, n_sensors,
         env_signal, dt,
         use_eph=True, use_npp=True, use_phi=True, use_body=True,
         budget_scale=1.0,
         external_drive=None,
         frustration_matrix=None,
         motor_classes=None,
         pos_1d=None):
    """Advance the simulation by one time step.

    This is the hot loop extracted from celegans_trilayer.py lines 375-471,
    cleaned up and parameterised for reuse.

    Parameters
    ----------
    state : WormState
        Mutable simulation state (modified in place).
    params : WormParams
        Coupling strengths and timescales.
    omegas : ndarray, shape (N,)
        Intrinsic angular frequencies.
    coupling_matrices : CouplingMatrices namedtuple
        Normalized weight matrices (W_chem_norm, W_gap_norm, W_eph_norm,
        W_npp_norm).
    grid_mapping : dict
        Body-axis grid mapping from ``build_grid_mapping``.
    sensory_indices : list of int
        Indices of sensory neurons.
    motor_indices : list of int
        Indices of motor neurons.
    n_sensors : int
        Number of sensory channels receiving environmental input.
    env_signal : ndarray, shape (n_sensors,)
        Current environmental signal for this time step.
    dt : float
        Integration time step (seconds).
    use_eph : bool
        Enable Layer 2 (ephaptic coupling).
    use_npp : bool
        Enable Layer 3 (neuropeptide modulation).
    use_phi : bool
        Enable coherence field dynamics.
    use_body : bool
        Enable body mechanics.
    budget_scale : float in [0, 1]
        Metabolic budget multiplier on all coupling strengths.
    frustration_matrix : ndarray, shape (N, N), optional
        Sakaguchi-Kuramoto phase frustration offsets for gap junctions.
        Entry [i,j] is subtracted from (theta_i - theta_j) in gap coupling.
        Use ``build_frustration_matrix`` to construct.
    motor_classes : dict, optional
        Dict with keys 'VA','VB','DA','DB','VD','DD' mapping to index lists.
        When provided and ``use_body`` is True, uses dorsal-ventral muscle
        model instead of the default single-channel body mechanics.

    Returns
    -------
    diagnostics : dict
        ``{conn_rms, eph_rms, npp_std, order_r, mean_phi}``
    """
    p = params
    theta = state.theta
    N = len(theta)
    Nx = len(state.phi)

    n_left = grid_mapping['n_left']
    n_frac = grid_mapping['n_frac']
    neuron_w_grid = grid_mapping['neuron_w_grid']
    neuron_body_idx = grid_mapping['neuron_body_idx']
    dx = grid_mapping['dx']

    W_chem_norm = coupling_matrices.W_chem_norm
    W_gap_norm = coupling_matrices.W_gap_norm
    W_eph_norm = coupling_matrices.W_eph_norm
    W_npp_norm = coupling_matrices.W_npp_norm

    # ── 1. Interpolate coherence field at neuron positions ─────────
    phi_n = interp_field(state.phi, n_left, n_frac)
    phi_clipped = np.clip(phi_n, 0, 1)

    # ── 2. Sensory drive from environment ──────────────────────────
    if external_drive is not None:
        drive = external_drive.copy()
    else:
        drive = np.zeros(N)
        si = np.array(sensory_indices[:n_sensors])
        drive[si] = p.K_drive * env_signal[:len(si)]

    # ── 3. Phase differences (vectorised NxN) ─────────────────────
    phase_diff = theta[:, None] - theta[None, :]
    sin_diff = np.sin(phase_diff)
    # Frustrated sin_diff: applies to motor-motor pairs (zero elsewhere)
    if frustration_matrix is not None:
        sin_diff_f = np.sin(phase_diff - frustration_matrix)
    else:
        sin_diff_f = sin_diff

    # ── 4. Layer 1: Connectome coupling (chemical + gap w/ depression)
    gap_eff = state.u_gap * W_gap_norm
    conn_coupling = budget_scale * (
        p.K_chem * np.sum(W_chem_norm * sin_diff_f, axis=0)
        + p.K_gap * np.sum(gap_eff * sin_diff_f, axis=0)
    )

    # ── 5. Layer 2: Ephaptic coupling (gated by phi, w/ depression)
    eph_coupling = np.zeros(N)
    if use_eph:
        phi_gate = phi_clipped[:, None] * phi_clipped[None, :]
        eph_eff = state.u_eph * W_eph_norm * phi_gate
        eph_coupling = budget_scale * p.K_eph * np.sum(eph_eff * sin_diff_f, axis=0)

    # ── 6. Layer 3: Neuropeptide modulation (slow frequency mod) ──
    if use_npp:
        activity_npp = 0.5 + 0.5 * np.cos(theta)
        # Vectorized: npp_input[j] = sum_i W_npp[i,j] * activity[i] * u_npp[i,j]
        npp_input = np.sum(W_npp_norm * (activity_npp[:, None] * state.u_npp), axis=0)

        dnpp = (-state.npp_mod + budget_scale * p.K_npp * npp_input) / p.tau_npp
        state.npp_mod += dnpp * dt

    # ── 7. Effective frequency ─────────────────────────────────────
    omega_eff = omegas - p.g_adapt * state.adapt + state.npp_mod

    # ── 8. Proprioception ─────────────────────────────────────────
    proprio = np.zeros(N)
    if use_body and len(motor_indices) > 0:
        if pos_1d is not None:
            # Anterior-delayed: read curvature from position ahead
            if motor_classes is not None:
                # Restrict to body-wall motor neurons only
                bw = []
                for cls in ('DA', 'DB', 'VA', 'VB', 'VD', 'DD'):
                    bw.extend(motor_classes[cls])
                mi = np.array(bw) if bw else np.array([], dtype=int)
            else:
                mi = np.array(motor_indices)
            if len(mi) > 0:
                anterior_pos = np.clip(pos_1d[mi] - p.proprio_delta_s, 0.0, 1.0)
                kappa_ant = np.interp(anterior_pos, grid_mapping['grid_x'], state.kappa)
                proprio[mi] = p.proprio_gain * np.sin(kappa_ant - theta[mi])
        else:
            # Legacy: local proprioception
            mi = np.array(motor_indices)
            kappa_n = interp_field(state.kappa, n_left, n_frac)
            proprio[mi] = 0.3 * np.sin(kappa_n[mi] - theta[mi])

    # ── 9. Phase update ────────────────────────────────────────────
    dtheta = omega_eff + drive + conn_coupling + eph_coupling + proprio
    dtheta += p.sigma_noise * state.rng_noise.randn(N) / np.sqrt(dt)
    theta += dtheta * dt
    # (theta modified in place via state.theta)

    # ── 10. Adaptation update ──────────────────────────────────────
    activity = 0.5 * (1 + np.cos(theta))
    state.adapt += (-state.adapt + activity) / p.tau_adapt * dt

    # ── 11. Synaptic depression update ─────────────────────────────
    abs_sin = np.abs(sin_diff)

    du_gap = (p.U_init - state.u_gap) / p.tau_rec_gap - state.u_gap * abs_sin / p.tau_dep
    state.u_gap = np.clip(state.u_gap + du_gap * dt, 0.01, 1.0)

    if use_eph:
        du_eph = (p.U_init - state.u_eph) / p.tau_rec_eph - state.u_eph * abs_sin / p.tau_dep
        state.u_eph = np.clip(state.u_eph + du_eph * dt, 0.01, 1.0)

    if use_npp:
        du_npp = (p.U_init - state.u_npp) / p.tau_rec_npp - state.u_npp * abs_sin / (p.tau_dep * 3)
        state.u_npp = np.clip(state.u_npp + du_npp * dt, 0.01, 1.0)

    # ── 12. Coherence field Landau-Ginzburg update ─────────────────
    if use_phi:
        phi = state.phi
        dVdphi = -p.a_LG * phi + p.b_LG * phi**3
        lap_phi = laplacian_1d(phi, dx)

        # Vectorized coherence source: local Kuramoto order parameter
        phases = np.exp(1j * theta)
        weighted = neuron_w_grid * phases[None, :]  # (Nx, N)
        w_sums = neuron_w_grid.sum(axis=1)
        r_local = np.abs(weighted.sum(axis=1)) / np.maximum(w_sums, 1e-10)
        source = p.gamma_pump * r_local

        dphi = (-dVdphi + p.kappa_phi * lap_phi + source) / p.tau_phi
        dphi = np.clip(dphi, -10, 10)
        state.phi += dphi * dt + (
            (p.sigma_phi / np.sqrt(p.tau_phi))
            * state.rng_phi.randn(Nx)
            * np.sqrt(dt)
        )
        state.phi = np.clip(state.phi, 0, 3.0)

    # ── 13. Body mechanics update ──────────────────────────────────
    if use_body and motor_classes is not None:
        # ── 13a. Dorsal-ventral muscle model ──────────────────────
        activity = 0.5 * (1 + np.cos(theta))  # [0, 1]

        F_d, F_v = compute_dv_drive(
            activity, neuron_body_idx, motor_classes, p.K_muscle_inh, Nx)

        # Low-pass muscle filter
        alpha_m = dt / (p.tau_muscle + dt)
        state.m_dorsal += alpha_m * (F_d - state.m_dorsal)
        state.m_ventral += alpha_m * (F_v - state.m_ventral)

        # Curvature from D-V difference
        state.kappa = p.K_muscle * (state.m_dorsal - state.m_ventral)
        state.kappa = np.clip(state.kappa, -3, 3)

    elif use_body:
        # ── 13b. Legacy single-channel body mechanics ─────────────
        F_muscle = np.zeros(Nx)
        if len(motor_indices) > 0:
            mi = np.array(motor_indices)
            np.add.at(F_muscle, neuron_body_idx[mi], p.K_muscle * np.sin(theta[mi]))

        dkdot = (-p.epsilon_body * state.kappa
                 - p.gamma_body * state.kappa_dot
                 + F_muscle)
        dkdot += p.sigma_body * state.rng_body.randn(Nx) / np.sqrt(dt)
        state.kappa_dot += dkdot * dt
        state.kappa += state.kappa_dot * dt
        state.kappa = np.clip(state.kappa, -3, 3)

    # ── Diagnostics ────────────────────────────────────────────────
    order_r = float(np.abs(np.mean(np.exp(1j * theta))))
    mean_phi = float(state.phi.mean())

    return {
        'conn_rms': float(np.sqrt(np.mean(conn_coupling**2))),
        'eph_rms': float(np.sqrt(np.mean(eph_coupling**2))),
        'npp_std': float(np.std(state.npp_mod)),
        'order_r': order_r,
        'mean_phi': mean_phi,
    }
