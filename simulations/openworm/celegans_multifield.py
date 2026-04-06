#!/usr/bin/env python3
"""
C. elegans: Multi-field coherence model with information geometry.

Extends the single-φ Landau-Ginzburg model with:

1. VECTOR coherence field φ(x,t) ∈ R^K — K competing coherence modes
   at each body position (locomotion, sensory integration, feeding).
   Modes compete via winnerless competition for neural resources.

2. Fisher information metric g(x,t) — computed from local neural
   activity, measures the manifold's local curvature / information
   density. High g = sharp distinctions between states (informative).
   Low g = flat, noisy, uninformative.

3. Metabolic field m(x,t) — local energy budget constraining coherence.
   Maintaining coherence costs energy proportional to |φ|².
   m depletes with coherence, recovers with a slow timescale.
   Aging = slower recovery → smaller coherence bubbles.

4. Proprioceptive wave — motor output creates a mechanical wave
   along the body that feeds back into the coherence field,
   creating travelling coherence waves = locomotion.

The full dynamics:
  ∂φ_k/∂t = -δF/δφ_k + κ∇²φ_k - competition + m(x)·pump_k + noise
  ∂m/∂t = (m_0 - m)/τ_m - c·Σ|φ_k|²  (depletion + recovery)
  g_ij(x) = Fisher metric from local neural covariance

Usage:
  python3 celegans_multifield.py [--fine]

Author: Ian Todd
"""

import sys
import numpy as np
import csv
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
# RESOLUTION
# ════════════════════════════════════════════════════════════════════
if FINE:
    Nx = 150; dt = 0.0005; T_total = 200.0; T_trans = 30.0
    print("=== FINE ===")
else:
    Nx = 60; dt = 0.001; T_total = 120.0; T_trans = 20.0
    print("=== COARSE ===")

n_steps = int(T_total / dt)
n_trans = int(T_trans / dt)
subsample = max(1, int(0.01 / dt))
dt_stored = dt * subsample
dx = 1.0 / Nx

print(f"  Grid: {Nx}, dt={dt}, T={T_total}s, steps={n_steps}")

# ════════════════════════════════════════════════════════════════════
# CONNECTOME (same loading as other sims)
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
        neurons_set.add(src); neurons_set.add(tgt)
        if typ == 'chemical': chemical_edges.append((src, tgt, w))
        elif typ == 'electrical': gap_edges.append((src, tgt, w))

positions = {}
with open(data_dir / "neuron_positions.csv") as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            name = parts[0].strip()
            try: positions[name] = (float(parts[1]), float(parts[2]))
            except ValueError: continue

neurons = sorted(neurons_set)
N = len(neurons)
neuron_idx = {n: i for i, n in enumerate(neurons)}

rng_pos = np.random.RandomState(0)
pos_2d = np.zeros((N, 2))
for i, n in enumerate(neurons):
    pos_2d[i] = positions.get(n, rng_pos.randn(2) * 0.3)

# 1D body axis
x_coords = pos_2d[:, 0]
pos_1d = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)
grid_x = np.linspace(0, 1, Nx)
neuron_grid_idx = np.array([np.argmin(np.abs(grid_x - p)) for p in pos_1d])

# Adjacency
W_chem = np.zeros((N, N))
W_gap = np.zeros((N, N))
for src, tgt, w in chemical_edges:
    W_chem[neuron_idx[src], neuron_idx[tgt]] += w
for src, tgt, w in gap_edges:
    i, j = neuron_idx[src], neuron_idx[tgt]
    W_gap[i, j] += w; W_gap[j, i] += w
W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

# Classification
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

# Spatial coupling weights (for field-mediated coupling)
sigma_sp = 0.1
dist_1d = np.abs(pos_1d[:, None] - pos_1d[None, :])
W_field = np.exp(-dist_1d**2 / (2 * sigma_sp**2))
np.fill_diagonal(W_field, 0)
W_field /= np.maximum(W_field.sum(axis=1, keepdims=True), 1)

# Neuron-to-grid weights (for computing local order params)
neuron_w_grid = np.zeros((Nx, N))
for gi in range(Nx):
    for ni in range(N):
        neuron_w_grid[gi, ni] = np.exp(-(grid_x[gi] - pos_1d[ni])**2 / (2*sigma_sp**2))
    s = neuron_w_grid[gi].sum()
    if s > 0: neuron_w_grid[gi] /= s

# Interpolation for φ at neuron positions
n_grid_left = np.clip(np.floor(pos_1d * (Nx - 1)).astype(int), 0, Nx - 2)
n_grid_frac = pos_1d * (Nx - 1) - n_grid_left

print(f"  {N} neurons, {len(chemical_edges)} chem, {len(gap_edges)} gap")

# ════════════════════════════════════════════════════════════════════
# MULTI-FIELD PARAMETERS
# ════════════════════════════════════════════════════════════════════
K_modes = 3  # number of competing coherence modes
mode_names = ['Locomotion', 'Sensory', 'Feeding']

# Landau-Ginzburg per mode
a_LG = 0.5       # coherent well depth
b_LG = 2.0       # quartic (φ* = √(a/b) = 0.5)
kappa = 0.015     # diffusion / surface tension
sigma_phi = 0.35  # noise on φ

# Competition between modes (winnerless)
alpha_compete = 0.8  # cross-inhibition strength

# Metabolic field
m_0 = 1.0        # resting metabolic level
tau_m = 5.0       # metabolic recovery timescale (slow — seconds)
c_depletion = 0.3 # coherence depletes metabolism
sigma_m = 0.05    # metabolic noise (small)

# Neural coupling
K_chem = 0.3
K_gap = 0.5
K_field = 1.5     # field coupling (gated by dominant φ mode)
K_drive = 1.0
sigma_noise = 0.15

# Mode-specific pump strengths: which neuron classes pump which mode
# Locomotion: pumped by motor neuron synchrony
# Sensory: pumped by sensory neuron synchrony
# Feeding: pumped by pharyngeal/head interneuron synchrony
pump_strength = np.zeros((K_modes, N))
gamma_base = 0.6

# Mode 0 (locomotion): motor neurons pump it
for i in motor:
    pump_strength[0, i] = gamma_base
# Mode 1 (sensory): sensory neurons pump it
for i in sensory:
    pump_strength[1, i] = gamma_base
# Mode 2 (feeding): head interneurons pump it
head_inter = [i for i in inter if pos_1d[i] < 0.2]
for i in head_inter:
    pump_strength[2, i] = gamma_base

# Proprioceptive wave parameters
K_proprio_wave = 0.3  # motor output → mechanical wave → coherence feedback
tau_wave = 0.2         # wave propagation timescale
v_wave = 2.0           # wave speed along body (body lengths / second)

# Environment
n_sensors = min(len(sensory), 50)
tau_env = 2.0
sigma_env = 0.5

np.random.seed(42)

print(f"\n  Modes: {K_modes} ({', '.join(mode_names)})")
print(f"  LG: a={a_LG}, b={b_LG}, κ={kappa}")
print(f"  Competition: α={alpha_compete}")
print(f"  Metabolism: m₀={m_0}, τ_m={tau_m}, c={c_depletion}")

# ════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ════════════════════════════════════════════════════════════════════
theta = 2 * np.pi * np.random.RandomState(77).rand(N)
phi = np.ones((K_modes, Nx)) * 0.05  # start near decoherent
phi[0, Nx//3:2*Nx//3] = 0.2  # seed locomotion mode in mid-body
phi[1, :Nx//4] = 0.2          # seed sensory mode in head
m_field = np.ones(Nx) * m_0   # metabolic field starts full
mech_wave = np.zeros(Nx)       # mechanical wave (proprioceptive)

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

# Frequency assignment
rng_freq = np.random.RandomState(99)
omegas = np.zeros(N)
for i in inter: omegas[i] = 2 * np.pi * (0.7 + 0.3 * rng_freq.rand())
for i in sensory: omegas[i] = 2 * np.pi * (0.9 + 0.3 * rng_freq.rand())
for i in motor: omegas[i] = 2 * np.pi * (1.0 + 0.3 * rng_freq.rand())

rng_n = np.random.RandomState(123)
rng_phi = np.random.RandomState(456)
rng_m = np.random.RandomState(789)

# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def phi_at_neurons(phi_mode):
    """Interpolate one φ mode to neuron positions."""
    return phi_mode[n_grid_left] * (1 - n_grid_frac) + phi_mode[n_grid_left + 1] * n_grid_frac

def local_order_param(theta, gi, weights):
    """Weighted local Kuramoto order parameter."""
    w = weights[gi]
    mask = w > 1e-6
    if mask.sum() < 2: return 0.0
    return float(np.abs(np.sum(w[mask] * np.exp(1j * theta[mask])) / w[mask].sum()))

def laplacian(f, dx):
    lap = np.zeros_like(f)
    lap[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    lap[0] = (f[1] - f[0]) / dx**2
    lap[-1] = (f[-2] - f[-1]) / dx**2
    return lap

def free_energy_mode(phi_k, dx, kappa, a, b):
    V = -a/2 * phi_k**2 + b/4 * phi_k**4
    gp = np.gradient(phi_k, dx)
    return np.sum(V + 0.5 * kappa * gp**2) * dx

# ════════════════════════════════════════════════════════════════════
# STORAGE
# ════════════════════════════════════════════════════════════════════
n_stored = (n_steps - n_trans) // subsample
phi_hist = np.zeros((n_stored, K_modes, Nx))
m_hist = np.zeros((n_stored, Nx))
theta_hist = np.zeros((n_stored, N))
fisher_hist = np.zeros((n_stored, Nx))
fe_hist = np.zeros((n_stored, K_modes))
wave_hist = np.zeros((n_stored, Nx))
si = 0

report_interval = max(1, n_steps // 20)

# ════════════════════════════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════════════════════════════
print("\nRunning simulation...")

for t in range(1, n_steps):
    # ── Dominant mode at each neuron position ───────────────────
    phi_at_n = np.array([phi_at_neurons(phi[k]) for k in range(K_modes)])  # (K, N)
    phi_dominant = phi_at_n.max(axis=0)  # dominant mode amplitude at each neuron
    phi_dominant_clipped = np.clip(phi_dominant, 0, 1)

    # ── Sensory drive ───────────────────────────────────────────
    drive = np.zeros(N)
    for k, idx in enumerate(sensory[:n_sensors]):
        drive[idx] = K_drive * sensory_input[t, k]

    # ── Connectome coupling (always on) ─────────────────────────
    sin_diff = np.sin(theta[:, None] - theta[None, :])
    chem = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)
    gap = K_gap * np.sum(W_gap_norm * sin_diff, axis=0)

    # ── Field coupling (gated by dominant φ) ────────────────────
    phi_gate = phi_dominant_clipped[:, None] * phi_dominant_clipped[None, :]
    field_coup = K_field * np.sum(W_field * phi_gate * sin_diff, axis=0)

    # ── Proprioceptive wave feedback ────────────────────────────
    wave_at_n = mech_wave[neuron_grid_idx]
    proprio = K_proprio_wave * np.sin(wave_at_n - theta) * phi_dominant_clipped

    # ── Neural update ───────────────────────────────────────────
    dtheta = omegas + drive + chem + gap + field_coup + proprio
    dtheta += sigma_noise * rng_n.randn(N) / np.sqrt(dt)
    theta += dtheta * dt

    # ── Mode-specific neural pump ───────────────────────────────
    # For each mode, compute how much local synchrony pumps it
    source = np.zeros((K_modes, Nx))
    for k in range(K_modes):
        for gi in range(Nx):
            # Weighted sync of neurons that pump this mode
            w = neuron_w_grid[gi] * pump_strength[k]
            mask = w > 1e-8
            if mask.sum() < 2:
                source[k, gi] = 0.0
                continue
            wm = w[mask]
            r_local = np.abs(np.sum(wm * np.exp(1j * theta[mask])) / wm.sum())
            source[k, gi] = r_local

    # ── Coherence field update (per mode) ───────────────────────
    for k in range(K_modes):
        dVdphi = -a_LG * phi[k] + b_LG * phi[k]**3
        lap = laplacian(phi[k], dx)

        # Cross-inhibition from other modes
        competition = np.zeros(Nx)
        for j in range(K_modes):
            if j != k:
                competition += alpha_compete * phi[j]**2 * phi[k]

        # Metabolic gating: coherence pump scaled by local metabolism
        pump = m_field * source[k, :]

        dphi_det = (-dVdphi + kappa * lap - competition + pump)
        # Clamp deterministic update to prevent blowup
        dphi_det = np.clip(dphi_det, -10, 10)
        phi[k] += dphi_det * dt + sigma_phi * rng_phi.randn(Nx) * np.sqrt(dt)
        phi[k] = np.clip(phi[k], 0, 1.5)

    # ── Metabolic field update ──────────────────────────────────
    total_coherence = np.sum(phi**2, axis=0)  # sum of |φ_k|² across modes
    dm = (m_0 - m_field) / tau_m - c_depletion * total_coherence
    m_field += dm * dt + sigma_m * rng_m.randn(Nx) * np.sqrt(dt)
    m_field = np.clip(m_field, 0.01, m_0 * 1.5)

    # ── Mechanical wave (proprioceptive) ────────────────────────
    # Motor neuron activity creates a wave that propagates posteriorly
    motor_activity = np.zeros(Nx)
    for i in motor:
        gi = neuron_grid_idx[i]
        motor_activity[gi] += np.cos(theta[i])
    # Advection-diffusion: wave propagates along body
    wave_lap = laplacian(mech_wave, dx)
    wave_grad = np.zeros(Nx)
    wave_grad[1:] = (mech_wave[1:] - mech_wave[:-1]) / dx  # upwind
    dmw = -v_wave * wave_grad + 0.5 * wave_lap + motor_activity / tau_wave - mech_wave / tau_wave
    dmw = np.clip(dmw, -50, 50)
    mech_wave += dmw * dt
    mech_wave = np.clip(mech_wave, -5, 5)

    # ── Store ───────────────────────────────────────────────────
    if t >= n_trans and (t - n_trans) % subsample == 0 and si < n_stored:
        phi_hist[si] = phi.copy()
        m_hist[si] = m_field.copy()
        theta_hist[si] = theta.copy()
        wave_hist[si] = mech_wave.copy()
        for k in range(K_modes):
            fe_hist[si, k] = free_energy_mode(phi[k], dx, kappa, a_LG, b_LG)

        # Local Fisher information: tr(local covariance inverse)
        # Approximated by inverse local variance of neural phases
        for gi in range(Nx):
            w = neuron_w_grid[gi]
            mask = w > 1e-6
            if mask.sum() < 3:
                fisher_hist[si, gi] = 0
                continue
            phases = theta[mask]
            # Circular variance → Fisher info proxy
            r_loc = np.abs(np.mean(np.exp(1j * phases)))
            # Fisher info ∝ 1/circular_variance = 1/(1-r²) for von Mises
            fisher_hist[si, gi] = r_loc**2 / max(1 - r_loc**2, 0.01)
        si += 1

    if t % report_interval == 0:
        pct = 100 * t / n_steps
        mp = phi.mean(axis=1)
        r = np.abs(np.mean(np.exp(1j * theta)))
        mm = m_field.mean()
        print(f"  [{pct:5.1f}%] t={t*dt:.1f}s  φ=[{mp[0]:.2f},{mp[1]:.2f},{mp[2]:.2f}]  r={r:.3f}  m={mm:.3f}")

n_stored = si
phi_hist = phi_hist[:n_stored]
m_hist = m_hist[:n_stored]
theta_hist = theta_hist[:n_stored]
fisher_hist = fisher_hist[:n_stored]
fe_hist = fe_hist[:n_stored]
wave_hist = wave_hist[:n_stored]
t_stored = np.arange(n_stored) * dt_stored

# ════════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════════
print("\nAnalysis...")

# Motor coordination
def mcorr(th, A, B):
    if not A or not B: return 0.0
    a = np.mean(np.cos(th[:, A]), axis=1)
    b = np.mean(np.cos(th[:, B]), axis=1)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10: return 0.0
    return pearsonr(a, b)[0]

va_vb = mcorr(theta_hist, VA, VB)
va_da = mcorr(theta_hist, VA, DA)

# Mode dominance over time
mode_means = phi_hist.mean(axis=2)  # (n_stored, K_modes)

# Metabolic depletion profile
mean_m = m_hist.mean(axis=0)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for k in range(K_modes):
    mp = phi_hist[:, k, :].mean()
    print(f"  {mode_names[k]}: <φ_{k}> = {mp:.3f}, F = {fe_hist[:, k].mean():.3f}")
print(f"  Metabolism: <m> = {mean_m.mean():.3f} ± {mean_m.std():.3f}")
print(f"  Global r: {np.abs(np.mean(np.exp(1j * theta_hist[-1]))):.3f}")
print(f"  Fisher info (mean): {fisher_hist.mean():.3f}")
print(f"  Motor: VA-VB={va_vb:.3f}, VA-DA={va_da:.3f}")

# ════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

mode_colors = ['#EF5350', '#4CAF50', '#2196F3']

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 4, hspace=0.4, wspace=0.4)

# (a-c) Kymographs for each mode
for k in range(K_modes):
    ax = fig.add_subplot(gs[0, k])
    extent = [0, t_stored[-1], 0, 1]
    im = ax.imshow(phi_hist[:, k, :].T, aspect='auto', origin='lower',
                   extent=extent, cmap='inferno', vmin=0,
                   vmax=max(0.1, phi_hist[:, k, :].max()),
                   interpolation='bilinear')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Body axis')
    ax.set_title(f'({"abc"[k]}) {mode_names[k]} φ_{k}(x,t)', fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.7)

# (d) Metabolic field kymograph
ax = fig.add_subplot(gs[0, 3])
im = ax.imshow(m_hist.T, aspect='auto', origin='lower',
               extent=[0, t_stored[-1], 0, 1], cmap='YlGn',
               interpolation='bilinear')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Body axis')
ax.set_title('(d) Metabolic field m(x,t)')
fig.colorbar(im, ax=ax, shrink=0.7)

# (e) Mode competition over time
ax = fig.add_subplot(gs[1, 0])
for k in range(K_modes):
    ax.plot(t_stored, mode_means[:, k], color=mode_colors[k],
            linewidth=1, label=mode_names[k], alpha=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('⟨φ_k⟩')
ax.set_title('(e) Mode competition')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (f) Fisher information along body
ax = fig.add_subplot(gs[1, 1])
mean_fisher = fisher_hist.mean(axis=0)
ax.fill_between(grid_x, 0, mean_fisher, alpha=0.3, color='#9C27B0')
ax.plot(grid_x, mean_fisher, color='#9C27B0', linewidth=1.5)
ax.set_xlabel('Body axis')
ax.set_ylabel('Fisher information')
ax.set_title('(f) Information density')
ax.grid(True, alpha=0.2)

# (g) Metabolic profile (mean over time)
ax = fig.add_subplot(gs[1, 2])
ax.fill_between(grid_x, 0, mean_m, alpha=0.3, color='#4CAF50')
ax.plot(grid_x, mean_m, color='#4CAF50', linewidth=1.5)
ax.axhline(m_0, color='gray', linestyle=':', alpha=0.5, label=f'$m_0 = {m_0}$')
ax.set_xlabel('Body axis')
ax.set_ylabel('⟨m⟩')
ax.set_title('(g) Metabolic depletion')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (h) Free energy per mode
ax = fig.add_subplot(gs[1, 3])
for k in range(K_modes):
    ax.plot(t_stored, fe_hist[:, k], color=mode_colors[k],
            linewidth=0.8, label=mode_names[k], alpha=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('F[φ_k]')
ax.set_title('(h) Free energy per mode')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (i) Mechanical wave kymograph
ax = fig.add_subplot(gs[2, 0])
wmax = max(0.01, np.abs(wave_hist).max())
im = ax.imshow(wave_hist.T, aspect='auto', origin='lower',
               extent=[0, t_stored[-1], 0, 1], cmap='RdBu_r',
               vmin=-wmax, vmax=wmax, interpolation='bilinear')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Body axis')
ax.set_title('(i) Proprioceptive wave')
fig.colorbar(im, ax=ax, shrink=0.7)

# (j) Motor coordination over time
ax = fig.add_subplot(gs[2, 1])
window = min(200, n_stored // 5)
if window > 10 and VA and VB:
    va_sig = np.mean(np.cos(theta_hist[:, VA]), axis=1)
    vb_sig = np.mean(np.cos(theta_hist[:, VB]), axis=1)
    rc = np.zeros(n_stored - window)
    for i in range(len(rc)):
        a, b = va_sig[i:i+window], vb_sig[i:i+window]
        if np.std(a) > 1e-10 and np.std(b) > 1e-10:
            rc[i] = pearsonr(a, b)[0]
    ax.plot(t_stored[window//2:window//2+len(rc)], rc, color='#EF5350', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('VA-VB correlation')
ax.set_title('(j) Motor coordination')
ax.grid(True, alpha=0.2)

# (k) Coherence vs metabolism scatter
ax = fig.add_subplot(gs[2, 2])
# Sample timepoints
samp = np.linspace(0, n_stored-1, min(30, n_stored), dtype=int)
colors_s = plt.cm.viridis(np.linspace(0, 1, len(samp)))
for idx, col in zip(samp, colors_s):
    total_phi = np.sum(phi_hist[idx]**2, axis=0)
    ax.scatter(m_hist[idx], total_phi, c=[col], s=5, alpha=0.3, edgecolors='none')
ax.set_xlabel('Local metabolism m')
ax.set_ylabel('Total coherence Σ|φ_k|²')
ax.set_title('(k) Coherence costs metabolism')
ax.grid(True, alpha=0.2)

# (l) Dominant mode map
ax = fig.add_subplot(gs[2, 3])
dominant = phi_hist[-1].argmax(axis=0)  # which mode dominates at final t
for k in range(K_modes):
    mask = dominant == k
    ax.bar(grid_x[mask], phi_hist[-1, k, mask], width=dx,
           color=mode_colors[k], alpha=0.7, label=mode_names[k])
ax.set_xlabel('Body axis')
ax.set_ylabel('φ (dominant mode)')
ax.set_title('(l) Mode dominance map')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

suffix = '_fine' if FINE else '_coarse'
plt.savefig(fig_dir / f'fig_multifield{suffix}.pdf', bbox_inches='tight')
plt.savefig(fig_dir / f'fig_multifield{suffix}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved fig_multifield{suffix}")
print("\nDone!")
