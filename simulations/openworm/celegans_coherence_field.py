#!/usr/bin/env python3
"""
C. elegans as a coherence field.

The primary dynamical variable is φ(x,t) — a coherence order parameter
on the worm's body axis. Neurons are embedded in this field. Their coupling
is mediated by φ: neurons in coherent regions (high φ) coordinate through
the field; neurons in decoherent regions (low φ) rely only on the connectome.

The feedback loop:
  Neural synchrony → pumps local φ → increases coupling → more synchrony

This is self-sustaining when coupling > noise (coherence bubble persists)
and collapses when noise > coupling (bubble shrinks below critical radius).

The coherence field evolves by Landau-Ginzburg dynamics:
  ∂φ/∂t = -dV/dφ + κ∇²φ + S(neural sync) + noise
  V(φ) = -(a/2)φ² + (b/4)φ⁴   (double-well: stable at φ=0 and φ=√(a/b))

What we calculate:
  - φ(x,t) kymograph: travelling coherence waves = locomotion
  - Free energy F[φ] over time
  - Local D_eff(x) varying with coherence
  - Surface tension at coherence boundaries
  - Motor coordination emerging from the field dynamics
  - Phase diagram: coherence vs noise

Usage:
  python3 celegans_coherence_field.py [--fine]

  Default: coarse run (50 grid, T=100s, ~10 min)
  --fine:  publication run (200 grid, T=200s, overnight)

Author: Ian Todd
"""

import sys
import numpy as np
import csv
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm', 'axes.linewidth': 0.8,
})

fig_dir = Path(__file__).parent.parent / "figures"
fig_dir.mkdir(exist_ok=True)
data_dir = Path(__file__).parent

FINE = '--fine' in sys.argv

# ════════════════════════════════════════════════════════════════════
# RESOLUTION SETTINGS
# ════════════════════════════════════════════════════════════════════
if FINE:
    Nx_grid = 200       # spatial grid points on body axis
    dt = 0.0002          # timestep (must satisfy CFL: dt < dx²/(2κ))
    T_total = 200.0      # total simulation time
    T_transient = 30.0   # transient to discard
    print("=== FINE resolution (publication quality) ===")
else:
    Nx_grid = 50
    dt = 0.001
    T_total = 100.0
    T_transient = 20.0
    print("=== COARSE resolution (verification run) ===")

n_steps = int(T_total / dt)
n_transient = int(T_transient / dt)
subsample = max(1, int(0.01 / dt))  # store every ~10ms
dt_stored = dt * subsample
dx = 1.0 / Nx_grid

print(f"  Grid: {Nx_grid} points, dt={dt}, T={T_total}s")
print(f"  Steps: {n_steps}, transient: {n_transient}")

# ════════════════════════════════════════════════════════════════════
# LOAD CONNECTOME
# ════════════════════════════════════════════════════════════════════
print("\nLoading connectome...")

neurons_set = set()
chemical_edges = []
gap_edges = []

with open(data_dir / "herm_full_edgelist.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row['Source'].strip()
        tgt = row['Target'].strip()
        w = int(row['Weight'].strip())
        typ = row['Type'].strip()
        neurons_set.add(src)
        neurons_set.add(tgt)
        if typ == 'chemical':
            chemical_edges.append((src, tgt, w))
        elif typ == 'electrical':
            gap_edges.append((src, tgt, w))

positions = {}
with open(data_dir / "neuron_positions.csv") as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            name = parts[0].strip()
            try:
                x, y = float(parts[1]), float(parts[2])
                positions[name] = (x, y)
            except ValueError:
                continue

neurons = sorted(neurons_set)
N = len(neurons)
neuron_idx = {n: i for i, n in enumerate(neurons)}

# Neuron positions along body axis [0, 1]
rng_pos = np.random.RandomState(0)
pos_1d = np.zeros(N)  # 1D position along body axis
pos_2d = np.zeros((N, 2))
has_position = 0
for i, n in enumerate(neurons):
    if n in positions:
        pos_2d[i] = positions[n]
        has_position += 1
    else:
        pos_2d[i] = rng_pos.randn(2) * 0.3

# Map 2D positions to 1D body axis (use first principal axis)
# The body axis is approximately the x-coordinate for C. elegans
x_coords = pos_2d[:, 0]
pos_1d = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)

# Map each neuron to its nearest grid point
grid_x = np.linspace(0, 1, Nx_grid)
neuron_grid_idx = np.array([np.argmin(np.abs(grid_x - p)) for p in pos_1d])

# Adjacency matrices
W_chem = np.zeros((N, N))
W_gap = np.zeros((N, N))
for src, tgt, w in chemical_edges:
    W_chem[neuron_idx[src], neuron_idx[tgt]] += w
for src, tgt, w in gap_edges:
    i, j = neuron_idx[src], neuron_idx[tgt]
    W_gap[i, j] += w
    W_gap[j, i] += w
W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

# Neuron classification
sensory_prefixes = ('ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB',
                    'AWC','AFD','BAG','IL1','IL2','OLL','OLQ','PHA','PHB','PLM',
                    'ALM','AVM','PVD','FLP')
motor_prefixes = ('VA','VB','VD','DA','DB','DD','AS','RMD','RME','SMD','SMB')

sensory = [i for i, n in enumerate(neurons) if n.startswith(sensory_prefixes)]
motor = [i for i, n in enumerate(neurons) if n.startswith(motor_prefixes)]
inter = [i for i in range(N) if i not in sensory and i not in motor]

VA = [i for i, n in enumerate(neurons) if n.startswith('VA')]
VB = [i for i, n in enumerate(neurons) if n.startswith('VB')]
DA = [i for i, n in enumerate(neurons) if n.startswith('DA')]
DB = [i for i, n in enumerate(neurons) if n.startswith('DB')]

print(f"  {N} neurons ({has_position} with positions)")
print(f"  {len(chemical_edges)} chemical, {len(gap_edges)} gap junctions")
print(f"  Sensory: {len(sensory)}, Inter: {len(inter)}, Motor: {len(motor)}")

# ════════════════════════════════════════════════════════════════════
# COHERENCE FIELD PARAMETERS
# ════════════════════════════════════════════════════════════════════

# Landau-Ginzburg potential: V(φ) = A·φ²·(1-φ)²
# Genuinely bistable: stable minima at φ=0 (decoherent) AND φ=1 (coherent)
# Barrier at φ=0.5, height A/16.
# Neural sync TILTS the landscape toward coherent well via source term S(x,t).
# Without pump: both wells equally stable. With pump: coherent well deeper.
# Coherence potential: V(φ) = -(a/2)φ² + (b/4)φ⁴
# Graded coherence model. The potential has a single minimum at
# φ* = √(a/b) and an unstable fixed point at φ=0. Decoherence
# (φ ≈ 0) is maintained by noise competing with the neural pump;
# coherence (φ ≈ φ*) is maintained by the pump overcoming noise.
# The transition between these regimes is graded, not a sharp
# phase boundary — see the paper for discussion of this choice.
a_LG = 0.5       # determines coherent equilibrium φ* = √(a/b) = 0.5
b_LG = 2.0       # quartic stabilization
kappa = 0.02     # diffusion / surface tension of coherence gradients
sigma_phi = 0.4  # noise (competes with pump to keep φ low)
gamma_pump = 0.8 # neural sync → coherence pump
tau_phi = 0.5    # coherence timescale

# Neural coupling
K_chem = 0.3     # connectome chemical (always on)
K_gap = 0.5      # gap junction (weaker — system not auto-synced)
K_field = 1.5    # field-mediated coupling (gated by φ — the key variable)
K_drive = 1.0    # sensory drive (slightly weaker)
sigma_noise = 0.15  # neural phase noise (slightly higher)

# Environment
n_sensors = min(len(sensory), 50)
tau_env = 2.0
sigma_env = 0.5

np.random.seed(42)

print(f"\n  Coherence field: a={a_LG}, b={b_LG}, κ={kappa}, σ_φ={sigma_phi}")
print(f"  Neural: K_chem={K_chem}, K_gap={K_gap}, K_field={K_field}")
print(f"  Feedback: γ_pump={gamma_pump}, τ_φ={tau_phi}")

# ════════════════════════════════════════════════════════════════════
# PRECOMPUTE
# ════════════════════════════════════════════════════════════════════

# For each grid point, which neurons are nearby (within σ=0.1 body units)
sigma_spatial = 0.1
neuron_weight_at_grid = np.zeros((Nx_grid, N))
for gi in range(Nx_grid):
    for ni in range(N):
        d = abs(grid_x[gi] - pos_1d[ni])
        neuron_weight_at_grid[gi, ni] = np.exp(-d**2 / (2 * sigma_spatial**2))
    s = neuron_weight_at_grid[gi].sum()
    if s > 0:
        neuron_weight_at_grid[gi] /= s

# For each neuron, interpolate φ from the grid
# (precompute interpolation weights — linear between nearest grid points)
neuron_grid_left = np.clip(np.floor(pos_1d * (Nx_grid - 1)).astype(int), 0, Nx_grid - 2)
neuron_grid_frac = pos_1d * (Nx_grid - 1) - neuron_grid_left

# Distance-based field coupling (neurons couple through the field if close)
dist_1d = np.abs(pos_1d[:, None] - pos_1d[None, :])
W_field_spatial = np.exp(-dist_1d**2 / (2 * sigma_spatial**2))
np.fill_diagonal(W_field_spatial, 0)
W_field_norm = W_field_spatial / np.maximum(W_field_spatial.sum(axis=1, keepdims=True), 1)

# Frequency assignment
rng_freq = np.random.RandomState(99)
omegas = np.zeros(N)
for i in inter:
    omegas[i] = 2 * np.pi * (0.7 + 0.3 * rng_freq.rand())
for i in sensory:
    omegas[i] = 2 * np.pi * (0.9 + 0.3 * rng_freq.rand())
for i in motor:
    omegas[i] = 2 * np.pi * (1.0 + 0.3 * rng_freq.rand())

# Environment
def gen_env(n_steps, dt, n_s, tau, sigma, seed=42):
    rng = np.random.RandomState(seed)
    env = np.zeros((n_steps, n_s))
    decay = np.exp(-dt / tau)
    ns = sigma * np.sqrt(2 * dt / tau)
    for t in range(1, n_steps):
        env[t] = env[t-1] * decay + ns * rng.randn(n_s)
    return env

sensory_input = gen_env(n_steps, dt, n_sensors, tau_env, sigma_env)

# ════════════════════════════════════════════════════════════════════
# HELPER: interpolate φ at neuron positions
# ════════════════════════════════════════════════════════════════════
def phi_at_neurons(phi):
    """Interpolate coherence field to neuron positions."""
    left = neuron_grid_left
    frac = neuron_grid_frac
    return phi[left] * (1 - frac) + phi[left + 1] * frac

def local_order_parameter(theta, gi):
    """Local Kuramoto order parameter at grid point gi."""
    weights = neuron_weight_at_grid[gi]
    mask = weights > 1e-6
    if mask.sum() < 2:
        return 0.0
    w = weights[mask]
    th = theta[mask]
    r = np.abs(np.sum(w * np.exp(1j * th)) / w.sum())
    return r

# ════════════════════════════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════════════════════════════
print("\nRunning simulation...")

# Initial conditions
theta = 2 * np.pi * np.random.RandomState(77).rand(N)
phi = np.ones(Nx_grid) * 0.1  # start mostly decoherent

rng_noise = np.random.RandomState(123)
rng_phi_noise = np.random.RandomState(456)

# Storage
n_stored = (n_steps - n_transient) // subsample
phi_history = np.zeros((n_stored, Nx_grid))
theta_history = np.zeros((n_stored, N))
free_energy_history = np.zeros(n_stored)
order_param_history = np.zeros(n_stored)
si = 0

# Laplacian operator (periodic or Neumann boundary)
def laplacian_1d(f, dx):
    """1D Laplacian with Neumann (zero-flux) boundaries."""
    lap = np.zeros_like(f)
    lap[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    lap[0] = (f[1] - f[0]) / dx**2      # one-sided
    lap[-1] = (f[-2] - f[-1]) / dx**2    # one-sided
    return lap

def free_energy(phi, dx, kappa, a, b):
    """Landau-Ginzburg free energy: V(φ) = -(a/2)φ² + (b/4)φ⁴."""
    V = -a/2 * phi**2 + b/4 * phi**4
    grad_phi = np.gradient(phi, dx)
    F = np.sum(V + 0.5 * kappa * grad_phi**2) * dx
    return F

report_interval = max(1, n_steps // 20)

for t in range(1, n_steps):
    # ── φ at each neuron (for gating field coupling) ────────────
    phi_n = phi_at_neurons(phi)
    phi_n_clipped = np.clip(phi_n, 0, 1)

    # ── Sensory drive ───────────────────────────────────────────
    drive = np.zeros(N)
    for k, idx in enumerate(sensory[:n_sensors]):
        drive[idx] = K_drive * sensory_input[t, k]

    # ── Connectome coupling (always on) ─────────────────────────
    sin_diff = np.sin(theta[:, None] - theta[None, :])
    chem = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)
    gap = K_gap * np.sum(W_gap_norm * sin_diff, axis=0)

    # ── Field-mediated coupling (gated by local φ) ──────────────
    # Coupling strength between neurons i,j is:
    #   K_field * φ(x_i) * φ(x_j) * spatial_weight(i,j)
    phi_gate = phi_n_clipped[:, None] * phi_n_clipped[None, :]
    field_coupling = K_field * np.sum(W_field_norm * phi_gate * sin_diff, axis=0)

    # ── Neural phase update ─────────────────────────────────────
    dtheta = omegas + drive + chem + gap + field_coupling
    dtheta += sigma_noise * rng_noise.randn(N) / np.sqrt(dt)
    theta += dtheta * dt

    # ── Coherence field update (Landau-Ginzburg + neural pump) ──
    # dV/dφ = -aφ + bφ³
    # dV/dφ for V(φ) = A·φ²·(1-φ)² → -dV/dφ = -2A·φ(1-φ)(1-2φ)
    # dV/dφ for V(φ) = -(a/2)φ² + (b/4)φ⁴
    dVdphi = -a_LG * phi + b_LG * phi**3
    lap_phi = laplacian_1d(phi, dx)

    # Neural synchrony source: local order parameter
    source = np.zeros(Nx_grid)
    for gi in range(Nx_grid):
        source[gi] = gamma_pump * local_order_parameter(theta, gi)

    # Coherence dynamics (Euler-Maruyama)
    dphi_det = (-dVdphi + kappa * lap_phi + source) / tau_phi
    dphi_stoch = (sigma_phi / np.sqrt(tau_phi)) * rng_phi_noise.randn(Nx_grid)
    phi += dphi_det * dt + dphi_stoch * np.sqrt(dt)

    # Clamp φ ≥ 0
    phi = np.clip(phi, 0, 3.0)

    # ── Store ───────────────────────────────────────────────────
    if t >= n_transient and (t - n_transient) % subsample == 0 and si < n_stored:
        phi_history[si] = phi.copy()
        theta_history[si] = theta.copy()
        free_energy_history[si] = free_energy(phi, dx, kappa, a_LG, b_LG)
        order_param_history[si] = np.abs(np.mean(np.exp(1j * theta)))
        si += 1

    if t % report_interval == 0:
        mean_phi = phi.mean()
        max_phi = phi.max()
        r = np.abs(np.mean(np.exp(1j * theta)))
        F = free_energy(phi, dx, kappa, a_LG, b_LG)
        pct = 100 * t / n_steps
        print(f"  [{pct:5.1f}%] t={t*dt:.1f}s  <φ>={mean_phi:.3f}  max(φ)={max_phi:.3f}  r={r:.3f}  F={F:.2f}")

n_stored_actual = si
phi_history = phi_history[:n_stored_actual]
theta_history = theta_history[:n_stored_actual]
free_energy_history = free_energy_history[:n_stored_actual]
order_param_history = order_param_history[:n_stored_actual]

print(f"\n  Stored {n_stored_actual} timepoints")

# ════════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════════
print("\nAnalysis...")

# Time axis
t_stored = np.arange(n_stored_actual) * dt_stored

# Motor coordination
def motor_coord(theta_hist, class_A, class_B):
    if len(class_A) == 0 or len(class_B) == 0:
        return 0.0
    a = np.mean(np.cos(theta_hist[:, class_A]), axis=1)
    b = np.mean(np.cos(theta_hist[:, class_B]), axis=1)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    rho, _ = pearsonr(a, b)
    return rho

va_vb = motor_coord(theta_history, VA, VB)
va_da = motor_coord(theta_history, VA, DA)
da_db = motor_coord(theta_history, DA, DB)

# Local D_eff as function of body position (at final timestep)
def local_deff(theta_hist, pos_1d, grid_x, window=0.1):
    """Compute D_eff in spatial windows along body axis."""
    n_windows = len(grid_x)
    deff = np.zeros(n_windows)
    for gi in range(n_windows):
        mask = np.abs(pos_1d - grid_x[gi]) < window
        if mask.sum() < 3:
            deff[gi] = np.nan
            continue
        local_states = np.column_stack([
            np.cos(theta_hist[:, mask]),
            np.sin(theta_hist[:, mask])
        ])
        X = local_states - local_states.mean(0)
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        ev = s**2 / len(X)
        ev = ev[ev > 1e-10]
        if len(ev) == 0:
            deff[gi] = np.nan
        else:
            deff[gi] = (ev.sum())**2 / (ev**2).sum()
    return deff

local_deff_vals = local_deff(theta_history, pos_1d, grid_x)

# Surface tension: |∇φ|² at each point, averaged over time
grad_phi_sq = np.zeros(Nx_grid)
for si in range(n_stored_actual):
    gp = np.gradient(phi_history[si], dx)
    grad_phi_sq += gp**2
grad_phi_sq /= n_stored_actual

# Mean coherence profile
mean_phi = phi_history.mean(axis=0)
std_phi = phi_history.std(axis=0)

# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Mean coherence <φ>: {mean_phi.mean():.3f} ± {mean_phi.std():.3f}")
print(f"Max coherence: {phi_history.max():.3f}")
print(f"Global order parameter <r>: {order_param_history.mean():.3f}")
print(f"Free energy (final): {free_energy_history[-1]:.2f}")
print(f"Free energy (mean): {free_energy_history.mean():.2f}")
print(f"Surface tension (total): {np.sum(grad_phi_sq) * dx:.4f}")
print(f"Motor coordination:")
print(f"  VA-VB = {va_vb:.3f}")
print(f"  VA-DA = {va_da:.3f}")
print(f"  DA-DB = {da_db:.3f}")

# ════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

fig = plt.figure(figsize=(16, 11))
gs = GridSpec(3, 3, hspace=0.4, wspace=0.4)

# (a) Coherence kymograph: φ(x,t)
ax = fig.add_subplot(gs[0, :2])
extent = [0, t_stored[-1], 0, 1]
im = ax.imshow(phi_history.T, aspect='auto', origin='lower', extent=extent,
               cmap='inferno', interpolation='bilinear')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Body axis position')
ax.set_title('(a) Coherence field φ(x, t)', fontweight='bold')
fig.colorbar(im, ax=ax, label='φ', shrink=0.8)

# Neuron positions as dots on the right edge
for cls, color, label in [(sensory, '#4CAF50', 'S'), (inter, '#2196F3', 'I'), (motor, '#EF5350', 'M')]:
    ax.scatter([t_stored[-1]*1.01]*len(cls), [pos_1d[i] for i in cls],
               c=color, s=2, alpha=0.5, clip_on=False)

# (b) Free energy over time
ax = fig.add_subplot(gs[0, 2])
ax.plot(t_stored, free_energy_history, color='#673AB7', linewidth=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('F[φ]')
ax.set_title('(b) Free energy')
ax.grid(True, alpha=0.2)

# (c) Mean coherence profile along body
ax = fig.add_subplot(gs[1, 0])
ax.fill_between(grid_x, mean_phi - std_phi, mean_phi + std_phi,
                alpha=0.3, color='#FF9800')
ax.plot(grid_x, mean_phi, color='#FF9800', linewidth=2)
ax.set_xlabel('Body axis position')
ax.set_ylabel('⟨φ⟩')
ax.set_title('(c) Mean coherence profile')
ax.grid(True, alpha=0.2)

# Mark neuron density
neuron_density = np.zeros(Nx_grid)
for ni in range(N):
    gi = neuron_grid_idx[ni]
    neuron_density[gi] += 1
ax2 = ax.twinx()
ax2.bar(grid_x, neuron_density, width=dx, alpha=0.15, color='gray')
ax2.set_ylabel('Neuron count', color='gray', alpha=0.5)

# (d) Local D_eff vs position (with coherence overlay)
ax = fig.add_subplot(gs[1, 1])
valid = ~np.isnan(local_deff_vals)
ax.plot(grid_x[valid], local_deff_vals[valid], 'o-', color='#2196F3',
        markersize=3, linewidth=1, label=r'$D_{\mathrm{eff}}$')
ax.set_xlabel('Body axis position')
ax.set_ylabel(r'Local $D_{\mathrm{eff}}$', color='#2196F3')
ax.set_title('(d) Local dimensionality vs coherence')
ax.grid(True, alpha=0.2)

ax2 = ax.twinx()
ax2.plot(grid_x, mean_phi, color='#FF9800', linewidth=1.5, alpha=0.7, label='φ')
ax2.set_ylabel('⟨φ⟩', color='#FF9800')

# (e) Surface tension along body
ax = fig.add_subplot(gs[1, 2])
ax.fill_between(grid_x, 0, grad_phi_sq, alpha=0.4, color='#E91E63')
ax.plot(grid_x, grad_phi_sq, color='#E91E63', linewidth=1.5)
ax.set_xlabel('Body axis position')
ax.set_ylabel(r'$|\nabla\phi|^2$')
ax.set_title('(e) Surface tension (coherence boundaries)')
ax.grid(True, alpha=0.2)

# (f) Motor coordination time series
ax = fig.add_subplot(gs[2, 0])
# Sliding window motor coordination
window = min(200, n_stored_actual // 5)
if window > 10 and len(VA) > 0 and len(VB) > 0:
    va_sig = np.mean(np.cos(theta_history[:, VA]), axis=1)
    vb_sig = np.mean(np.cos(theta_history[:, VB]), axis=1)
    rolling_corr = np.zeros(n_stored_actual - window)
    for i in range(len(rolling_corr)):
        a = va_sig[i:i+window]
        b = vb_sig[i:i+window]
        if np.std(a) > 1e-10 and np.std(b) > 1e-10:
            rolling_corr[i], _ = pearsonr(a, b)
    t_roll = t_stored[window//2:window//2+len(rolling_corr)]
    ax.plot(t_roll, rolling_corr, color='#EF5350', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('VA-VB correlation')
ax.set_title('(f) Motor coordination over time')
ax.grid(True, alpha=0.2)

# (g) Global order parameter and mean φ over time
ax = fig.add_subplot(gs[2, 1])
ax.plot(t_stored, order_param_history, color='#2196F3', linewidth=0.8, label='r (neural sync)')
mean_phi_t = phi_history.mean(axis=1)
ax.plot(t_stored, mean_phi_t, color='#FF9800', linewidth=0.8, label='⟨φ⟩ (coherence)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Order parameter')
ax.set_title('(g) Neural sync tracks coherence')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (h) Phase portrait: φ vs D_eff
ax = fig.add_subplot(gs[2, 2])
# Sample timepoints for scatter
n_samples = min(20, n_stored_actual)
sample_idx = np.linspace(0, n_stored_actual-1, n_samples, dtype=int)
colors_t = plt.cm.viridis(np.linspace(0, 1, n_samples))
for si_idx, color in zip(sample_idx, colors_t):
    th = theta_history[si_idx]
    ph = phi_history[si_idx]
    # Compute D_eff in windows
    for gi in range(0, Nx_grid, max(1, Nx_grid//10)):
        mask = np.abs(pos_1d - grid_x[gi]) < 0.1
        if mask.sum() < 3:
            continue
        local_r = np.abs(np.mean(np.exp(1j * th[mask])))
        ax.scatter(ph[gi], local_r, c=[color], s=8, alpha=0.3, edgecolors='none')

ax.set_xlabel('Local φ')
ax.set_ylabel('Local neural sync r')
ax.set_title('(h) Coherence predicts synchrony')
ax.grid(True, alpha=0.2)
# Add diagonal guide
ax.plot([0, 1.5], [0, 1], '--', color='gray', alpha=0.3, linewidth=0.5)

suffix = '_fine' if FINE else '_coarse'
plt.savefig(fig_dir / f'fig_coherence_field{suffix}.pdf', bbox_inches='tight')
plt.savefig(fig_dir / f'fig_coherence_field{suffix}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved fig_coherence_field{suffix}")

print("\nDone!")
