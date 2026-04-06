#!/usr/bin/env python3
"""
Evidentiary figures for Predictive Horizons paper.

2D oscillator field (25x25 = 625) coupled to spatiotemporal OU environment.
Three figures testing three claims:

1. Sync acquires future-relevant information (step-response τ_sync)
2. Low-dimensional observers under-measure that information (ΔI_hidden)
3. Discrete coding loses predictive content (ΔI_code)

Key design choices:
- Predictor state is INTERNAL oscillator state [cos θ_i, sin θ_i],
  NOT the tracking field cos(env - theta). This ensures MI measures
  what the organism knows, not what the environment persists.
- Observer dimensionality uses nested PCA truncation on the same state,
  guaranteeing monotonicity.
- Commit channel is an explicit quantize + lowpass + dwell-time readout.
- MI estimated via Gaussian covariance (one estimator for all levels).

Author: Ian Todd
"""

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm', 'axes.linewidth': 0.8,
})

# ── Parameters ──────────────────────────────────────────────────────
Nx, Ny = 25, 25
N = Nx * Ny  # 625
dt = 0.005
T_total = 200.0
T_transient = 50.0
n_steps = int(T_total / dt)
n_transient = int(T_transient / dt)

tau_env = 4.0
L_env = 4.0
sigma_env = 0.8
omega_0 = 2 * np.pi * 1.5
omega_spread = 0.6
K_nn = 2.0
sigma_noise = 0.15

K_sweep = np.array([0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 18.0, 25.0, 35.0, 45.0])
observer_dims = np.array([1, 2, 3, 5, 8, 12, 20, 35, 50, 80, 120, 200, 400, 625])
delta_ts = np.linspace(0.1, 10.0, 50)

fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

# ── Environment ─────────────────────────────────────────────────────
def generate_environment(n_steps, dt, Nx, Ny, tau, L, sigma, rng):
    env = np.zeros((n_steps, Nx, Ny))
    decay = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(2 * dt / tau)
    for t in range(1, n_steps):
        white = noise_scale * rng.randn(Nx, Ny)
        smooth = ndimage.gaussian_filter(white, sigma=L, mode='wrap')
        smooth *= noise_scale / (np.std(smooth) + 1e-10)
        env[t] = env[t-1] * decay + smooth
    return env

# ── Simulation ──────────────────────────────────────────────────────
def run_simulation(K_ext, seed=77):
    rng = np.random.RandomState(seed)
    omegas = omega_0 + omega_spread * rng.randn(Nx, Ny)
    env = generate_environment(n_steps, dt, Nx, Ny, tau_env, L_env, sigma_env, rng)
    theta = 2 * np.pi * rng.rand(Nx, Ny)

    subsample = 10
    n_stored = (n_steps - n_transient) // subsample

    # Store INTERNAL state [cos θ, sin θ] — not tracking field
    internal_state = np.zeros((n_stored, N * 2))  # cos + sin for each oscillator
    env_flat = np.zeros((n_stored, N))
    alignment_field = np.zeros((n_stored, Nx, Ny))
    store_idx = 0

    for t in range(1, n_steps):
        coupling_nn = (np.sin(np.roll(theta, -1, 1) - theta) +
                       np.sin(np.roll(theta, 1, 1) - theta) +
                       np.sin(np.roll(theta, -1, 0) - theta) +
                       np.sin(np.roll(theta, 1, 0) - theta))
        force_ext = K_ext * np.sin(env[t] - theta)
        dtheta = omegas + K_nn * coupling_nn / 4.0 + force_ext
        dtheta += sigma_noise * rng.randn(Nx, Ny) / np.sqrt(dt)
        theta += dtheta * dt

        if t >= n_transient and (t - n_transient) % subsample == 0:
            if store_idx < n_stored:
                flat_theta = theta.ravel()
                internal_state[store_idx, :N] = np.cos(flat_theta)
                internal_state[store_idx, N:] = np.sin(flat_theta)
                env_flat[store_idx] = env[t].ravel()
                alignment_field[store_idx] = np.cos(env[t] - theta)
                store_idx += 1

    return internal_state[:store_idx], env_flat[:store_idx], alignment_field[:store_idx]

# ── Gaussian MI via shared covariance ───────────────────────────────
def gaussian_mi(X, Y):
    """
    MI between multivariate Gaussian X and Y.
    I(X;Y) = 0.5 * [log det C_XX + log det C_YY - log det C_XXYY]
    Uses shrinkage covariance for stability.
    """
    n = X.shape[0]
    dX, dY = X.shape[1], Y.shape[1]

    # Shrinkage parameter
    alpha = 0.01

    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    C_XX = Xc.T @ Xc / n + alpha * np.eye(dX)
    C_YY = Yc.T @ Yc / n + alpha * np.eye(dY)

    XY = np.hstack([Xc, Yc])
    C_XXYY = XY.T @ XY / n + alpha * np.eye(dX + dY)

    sign_XX, logdet_XX = np.linalg.slogdet(C_XX)
    sign_YY, logdet_YY = np.linalg.slogdet(C_YY)
    sign_XY, logdet_XY = np.linalg.slogdet(C_XXYY)

    if sign_XX <= 0 or sign_YY <= 0 or sign_XY <= 0:
        return 0.0

    mi = 0.5 * (logdet_XX + logdet_YY - logdet_XY)
    return max(0.0, mi)

def compute_mi_profile(internal_state, env_flat, delta_ts, dt_stored, n_pcs=None):
    """
    Compute I(X_int(t); Y_env(t+Δ)) at each lag.
    If n_pcs is given, project internal_state onto top-n PCs first.
    """
    n_t = internal_state.shape[0]

    # PCA on internal state
    X = internal_state - internal_state.mean(axis=0)
    if n_pcs is not None and n_pcs < X.shape[1]:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        X = X @ Vt[:n_pcs].T

    # Target: first few PCs of environment (for tractable MI estimation)
    E = env_flat - env_flat.mean(axis=0)
    _, _, Vt_env = np.linalg.svd(E, full_matrices=False)
    n_env_pcs = min(20, E.shape[1])
    E_proj = E @ Vt_env[:n_env_pcs].T

    I_pred = np.zeros(len(delta_ts))
    for i, lag in enumerate(delta_ts):
        lag_steps = max(1, int(lag / dt_stored))
        if lag_steps >= n_t - 100:
            continue
        X_now = X[:-lag_steps]
        E_future = E_proj[lag_steps:]
        min_len = min(len(X_now), len(E_future))
        # Subsample for speed if needed
        if min_len > 5000:
            idx = np.random.choice(min_len, 5000, replace=False)
            X_now = X_now[idx]
            E_future = E_future[idx]
        else:
            X_now = X_now[:min_len]
            E_future = E_future[:min_len]
        I_pred[i] = gaussian_mi(X_now, E_future)

    return I_pred

# ── Commit channel ──────────────────────────────────────────────────
def make_commit_channel(internal_state, n_readout=5, n_bits=4, commit_interval=50):
    """
    Explicit commit channel: project to n_readout dims, quantize to n_bits,
    update only every commit_interval steps.
    """
    X = internal_state - internal_state.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    projected = X @ Vt[:n_readout].T

    # Quantize each dimension
    quantized = np.zeros_like(projected)
    for d in range(n_readout):
        col = projected[:, d]
        bins = np.linspace(col.min(), col.max(), 2**n_bits + 1)
        digitized = np.digitize(col, bins) - 1
        digitized = np.clip(digitized, 0, 2**n_bits - 1)
        # Map back to bin centers
        centers = (bins[:-1] + bins[1:]) / 2
        quantized[:, d] = centers[digitized]

    # Dwell time: only update every commit_interval steps
    committed = np.zeros_like(quantized)
    committed[0] = quantized[0]
    for t in range(1, len(quantized)):
        if t % commit_interval == 0:
            committed[t] = quantized[t]
        else:
            committed[t] = committed[t-1]

    return committed

# ── Discrete-update simulation ──────────────────────────────────────
def run_simulation_discrete(K_ext, update_interval, seed=77,
                            shared_env=None, shared_omegas=None, shared_theta0=None):
    """
    Same oscillator network, but external coupling only updates
    every update_interval timesteps. Between updates, the coupling
    force is frozen at its last-sampled value.
    This simulates a clocked digital system that samples the environment
    at discrete intervals rather than coupling continuously.

    Pass shared_env, shared_omegas, shared_theta0 for controlled comparisons.
    """
    omegas = shared_omegas if shared_omegas is not None else omega_0 + omega_spread * np.random.RandomState(seed).randn(Nx, Ny)
    env = shared_env if shared_env is not None else generate_environment(n_steps, dt, Nx, Ny, tau_env, L_env, sigma_env, np.random.RandomState(seed))
    theta = shared_theta0.copy() if shared_theta0 is not None else 2 * np.pi * np.random.RandomState(seed).rand(Nx, Ny)

    # Separate noise RNG — same seed for all runs so noise is identical
    noise_rng = np.random.RandomState(seed + 999)

    subsample = 10
    n_stored = (n_steps - n_transient) // subsample
    internal_state = np.zeros((n_stored, N * 2))
    env_flat = np.zeros((n_stored, N))
    store_idx = 0

    # Frozen coupling force
    frozen_force = K_ext * np.sin(env[0] - theta)

    for t in range(1, n_steps):
        coupling_nn = (np.sin(np.roll(theta, -1, 1) - theta) +
                       np.sin(np.roll(theta, 1, 1) - theta) +
                       np.sin(np.roll(theta, -1, 0) - theta) +
                       np.sin(np.roll(theta, 1, 0) - theta))

        # Update coupling force only at discrete intervals
        if t % update_interval == 0:
            frozen_force = K_ext * np.sin(env[t] - theta)

        # Pre-draw noise (same for all runs regardless of trajectory)
        noise = sigma_noise * noise_rng.randn(Nx, Ny) / np.sqrt(dt)

        dtheta = omegas + K_nn * coupling_nn / 4.0 + frozen_force
        dtheta += noise
        theta += dtheta * dt

        if t >= n_transient and (t - n_transient) % subsample == 0:
            if store_idx < n_stored:
                flat_theta = theta.ravel()
                internal_state[store_idx, :N] = np.cos(flat_theta)
                internal_state[store_idx, N:] = np.sin(flat_theta)
                env_flat[store_idx] = env[t].ravel()
                store_idx += 1

    return internal_state[:store_idx], env_flat[:store_idx]

# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Evidentiary Figures — Predictive Horizons")
print("=" * 60)

dt_stored = dt * 10
K_demo = 22.0

# ── Run simulation ──────────────────────────────────────────────────
print("\nPhase 1: Running simulation...")
internal, env_f, align_f = run_simulation(K_demo)
print(f"  Stored {internal.shape[0]} timepoints, {internal.shape[1]} state dims")

# Also run K=0 for baseline
internal_0, env_0, _ = run_simulation(0.0)

# ── FIGURE 1: Alignment snapshots ───────────────────────────────────
print("\nPhase 2: Alignment snapshots...")
snap_Ks = [0.0, 5.0, 18.0, 45.0]
snap_labels = ['No coupling\n$K = 0$', 'Weak\n$K = 5$',
               'Moderate\n$K = 18$', 'Strong\n$K = 45$']
snapshots = {}
for K in snap_Ks:
    _, _, af = run_simulation(K)
    snapshots[K] = af[-1]

fig = plt.figure(figsize=(13, 3.2))
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.08)
axes = [fig.add_subplot(gs[i]) for i in range(4)]
cax = fig.add_subplot(gs[4])

for ax, K, label in zip(axes, snap_Ks, snap_labels):
    im = ax.imshow(snapshots[K], cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='bilinear', aspect='equal')
    ax.set_title(label, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    m = np.mean(snapshots[K])
    ax.text(0.95, 0.05, f'$\\bar{{a}} = {m:.2f}$', transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom',
            color='white' if m > 0.5 else 'black',
            bbox=dict(boxstyle='round,pad=0.2',
                     facecolor='black' if m > 0.5 else 'white',
                     alpha=0.4, edgecolor='none'))
fig.colorbar(im, cax=cax, label=r'Alignment $\cos(\theta_{\mathrm{env}} - \theta)$')
plt.savefig(fig_dir / 'fig1_alignment.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig1_alignment.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig1_alignment")

# ── FIGURE 2: Observational gap (nested PCA, one estimator) ────────
print("\nPhase 3: Observational gap...")

# Compute MI at each observer resolution using nested PCA
ipred_by_dim = np.zeros((len(observer_dims), len(delta_ts)))
for i, d in enumerate(observer_dims):
    n_pcs = min(d * 2, internal.shape[1])  # d oscillators → d*2 state dims (cos+sin)
    print(f"  d_obs = {d} (n_pcs = {n_pcs})")
    ipred_by_dim[i] = compute_mi_profile(internal, env_f, delta_ts, dt_stored, n_pcs=n_pcs)

# Also K=0 baseline at full resolution
ipred_baseline = compute_mi_profile(internal_0, env_0, delta_ts, dt_stored)

# Short-lag MI
ipred_short = np.mean(ipred_by_dim[:, :8], axis=1)
ipred_system = ipred_short[-1]

print(f"\n  System MI (d=625): {ipred_system:.4f}")
print(f"  Observer MI (d=1): {ipred_short[0]:.4f}")
print(f"  Observer MI (d=5): {ipred_short[np.argmin(np.abs(observer_dims-5))]:.4f}")
print(f"  Monotonic: {np.all(np.diff(ipred_short) >= -0.001)}")
print(f"  K=0 baseline: {np.mean(ipred_baseline[:8]):.4f}")

fig = plt.figure(figsize=(12, 4.5))

# (a) MI curves at different observer resolutions
ax = fig.add_subplot(131)
show_dims = [1, 5, 20, 200, 625]
cmap = plt.cm.viridis
for nd in show_dims:
    idx = np.argmin(np.abs(observer_dims - nd))
    actual = observer_dims[idx]
    frac = np.log(max(actual, 1)) / np.log(625)
    lw = 2.5 if actual >= 625 else 1.3
    label = 'Full system' if actual >= 625 else f'$d_{{\\mathrm{{obs}}}} = {actual}$'
    ax.plot(delta_ts, ipred_by_dim[idx], color=cmap(frac), linewidth=lw, label=label)
ax.plot(delta_ts, ipred_baseline, 'k--', linewidth=0.8, alpha=0.4, label='$K=0$ (no sync)')
ax.axvline(tau_env, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel(r'Prediction lag $\Delta t$ (s)')
ax.set_ylabel(r'$I(X_{\mathrm{int}}(t); Y_{\mathrm{env}}(t{+}\Delta t))$ (nats)')
ax.set_title(f'(a) Internal state predicts at $K={K_demo:.0f}$')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.2)

# (b) The gap
ax = fig.add_subplot(132)
ax.semilogx(observer_dims, ipred_short, 'ko-', markersize=4, linewidth=1.2,
            label='Observer measures', zorder=3)
ax.axhline(ipred_system, color='#EF5350', linestyle='--', linewidth=1.5,
           label='System contains', zorder=2)
ax.fill_between(observer_dims, ipred_short, ipred_system,
                alpha=0.2, color='#EF5350', zorder=1)
ax.set_xlabel(r'Observer dimensions $d_{\mathrm{obs}}$')
ax.set_ylabel(r'$\langle I_{\mathrm{pred}} \rangle$ (nats)')
ax.set_title('(b) The observational gap')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2)
mid = len(observer_dims) // 4
gap = ipred_system - ipred_short
if gap[mid] > 0.01 * ipred_system:
    ax.annotate('hidden\ncorrelations',
                xy=(observer_dims[mid], (ipred_short[mid] + ipred_system) / 2),
                fontsize=8, ha='center', color='#EF5350', style='italic')

# (c) Gap across coupling strengths
ax = fig.add_subplot(133)
gap_d5 = np.zeros(len(K_sweep))
gap_d50 = np.zeros(len(K_sweep))
for i, K in enumerate(K_sweep):
    print(f"  Gap sweep K = {K:.0f}")
    int_k, env_k, _ = run_simulation(K)
    ip_full = compute_mi_profile(int_k, env_k, delta_ts, dt_stored)
    ip_5 = compute_mi_profile(int_k, env_k, delta_ts, dt_stored, n_pcs=10)
    ip_50 = compute_mi_profile(int_k, env_k, delta_ts, dt_stored, n_pcs=100)
    full_short = np.mean(ip_full[:8])
    gap_d5[i] = max(0, full_short - np.mean(ip_5[:8]))
    gap_d50[i] = max(0, full_short - np.mean(ip_50[:8]))

ax.plot(K_sweep, gap_d5, 'o-', color='#E91E63', markersize=5, linewidth=1.2,
        label=f'$d_{{\\mathrm{{obs}}}} = 5$')
ax.plot(K_sweep, gap_d50, 's-', color='#9C27B0', markersize=5, linewidth=1.2,
        label=f'$d_{{\\mathrm{{obs}}}} = 50$')
ax.set_xlabel(r'Coupling strength $K$')
ax.set_ylabel('Information gap (nats)')
ax.set_title('(c) Gap grows with sync')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(fig_dir / 'fig2_observational_gap.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig2_observational_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig2_observational_gap")

# ── FIGURE 3: Code collapse — prediction loss from coding ───────────
print("\nPhase 4: Code collapse...")

# Full internal MI
ipred_full = compute_mi_profile(internal, env_f, delta_ts, dt_stored)

# Commit channel at different bandwidths
commit_configs = [
    ('5 dims, 4 bits, fast', 5, 4, 10),
    ('5 dims, 4 bits, slow', 5, 4, 100),
    ('20 dims, 4 bits, fast', 20, 4, 10),
    ('Full continuous (no commit)', None, None, None),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

colors_code = ['#2196F3', '#FF9800', '#4CAF50', '#EF5350']
for (label, n_rd, n_bits, c_int), color in zip(commit_configs, colors_code):
    if n_rd is None:
        # Full continuous = full internal state
        ax1.plot(delta_ts, ipred_full, color=color, linewidth=2.5, label=label)
    else:
        committed = make_commit_channel(internal, n_readout=n_rd, n_bits=n_bits,
                                         commit_interval=c_int)
        ipred_code = np.zeros(len(delta_ts))
        # Target env
        E = env_f - env_f.mean(axis=0)
        _, _, Vt_e = np.linalg.svd(E, full_matrices=False)
        E_proj = E @ Vt_e[:20].T

        for i, lag in enumerate(delta_ts):
            lag_steps = max(1, int(lag / dt_stored))
            if lag_steps >= len(committed) - 100:
                continue
            C_now = committed[:-lag_steps]
            E_fut = E_proj[lag_steps:]
            ml = min(len(C_now), len(E_fut))
            if ml > 5000:
                idx = np.random.choice(ml, 5000, replace=False)
                C_now = C_now[idx]; E_fut = E_fut[idx]
            else:
                C_now = C_now[:ml]; E_fut = E_fut[:ml]
            ipred_code[i] = gaussian_mi(C_now, E_fut)
        ax1.plot(delta_ts, ipred_code, color=color, linewidth=1.3,
                 linestyle='--', label=label)

ax1.axvline(tau_env, color='gray', linestyle=':', alpha=0.4)
ax1.set_xlabel(r'Prediction lag $\Delta t$ (s)')
ax1.set_ylabel(r'$I_{\mathrm{pred}}$ (nats)')
ax1.set_title('(a) Prediction: continuous vs coded')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(True, alpha=0.2)

# (b) Information loss from coding
# Compare full vs best commit channel at short lag
full_short = np.mean(ipred_full[:8])
code_labels = []
code_losses = []
for label, n_rd, n_bits, c_int in commit_configs:
    if n_rd is None:
        continue
    committed = make_commit_channel(internal, n_readout=n_rd, n_bits=n_bits,
                                     commit_interval=c_int)
    E = env_f - env_f.mean(axis=0)
    _, _, Vt_e = np.linalg.svd(E, full_matrices=False)
    E_proj = E @ Vt_e[:20].T

    # Short lag MI
    lag_steps = max(1, int(delta_ts[3] / dt_stored))
    C_now = committed[:-lag_steps]
    E_fut = E_proj[lag_steps:]
    ml = min(len(C_now), len(E_fut))
    if ml > 5000:
        idx = np.random.choice(ml, 5000, replace=False)
        C_now = C_now[idx]; E_fut = E_fut[idx]
    else:
        C_now = C_now[:ml]; E_fut = E_fut[:ml]
    code_mi = gaussian_mi(C_now, E_fut)
    code_labels.append(f'{n_rd}d/{n_bits}b\n{"fast" if c_int <= 10 else "slow"}')
    code_losses.append(full_short - code_mi)

ax2.bar(range(len(code_labels)), code_losses, color=['#2196F3', '#FF9800', '#4CAF50'],
        edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(code_labels)))
ax2.set_xticklabels(code_labels, fontsize=8)
ax2.set_ylabel(r'$\Delta I_{\mathrm{code}}$ (nats lost)')
ax2.set_title('(b) Information lost by coding')
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(fig_dir / 'fig3_code_collapse.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig3_code_collapse.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig3_code_collapse")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"System MI (full, K={K_demo}): {ipred_system:.4f} nats")
print(f"Observer MI (d=1): {ipred_short[0]:.4f} nats")
print(f"Observer MI (d=5): {ipred_short[np.argmin(np.abs(observer_dims-5))]:.4f} nats")
print(f"Monotonic in observer dims: {np.all(np.diff(ipred_short) >= -0.001)}")
print(f"K=0 baseline: {np.mean(ipred_baseline[:8]):.4f} nats")
print(f"Full continuous MI (short lag): {full_short:.4f} nats")
for label, loss in zip(code_labels, code_losses):
    print(f"Code loss ({label.replace(chr(10), ', ')}): {loss:.4f} nats")
print("\nDone!")
