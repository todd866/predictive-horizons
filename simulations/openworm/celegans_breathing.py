#!/usr/bin/env python3
"""
C. elegans: Breathing coherence bubbles.

The missing physics from the previous models: NEGATIVE FEEDBACK.
Three mechanisms prevent lock-in:

1. SHORT-TERM SYNAPTIC DEPRESSION (Tsodyks-Markram)
   Each coupling channel has a resource variable u ∈ [0,1].
   Usage depletes u; recovery is slow. Effective coupling = K·u·W.
   Sustained coherence erodes its own foundation.

2. SPIKE-FREQUENCY ADAPTATION
   Each neuron has an adaptation variable that accumulates with
   activity and reduces the effective frequency. Synchronized
   neurons adapt together → collective refractory period.

3. DYNAMIC BODY GEOMETRY
   The body curvature field κ(x,t) modulates the distance matrix.
   When the worm bends, neurons move relative to each other,
   reshuffling the ephaptic coupling landscape every locomotor cycle.

Together these create BREATHING COHERENCE BUBBLES:
   Form → deplete coupling → collapse → recover → reform

The breathing frequency, spatial extent, and anterior-posterior
gradient emerge from the interplay of these mechanisms with the
connectome topology and neuron positions.

Expected: genuinely graded, spatially varying, temporally
fluctuating coherence — not locked into either basin.

Author: Ian Todd
"""

import numpy as np
import csv
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
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

wall_start = time.time()

# ════════════════════════════════════════════════════════════════════
# RESOLUTION
# ════════════════════════════════════════════════════════════════════
dt = 0.0005
T_total = 300.0      # 5 minutes — long enough for multiple breath cycles
T_trans = 30.0
n_steps = int(T_total / dt)
n_trans = int(T_trans / dt)
subsample = max(1, int(0.02 / dt))  # store every 20ms
dt_stored = dt * subsample
Nx = 100             # body axis grid

dx = 1.0 / Nx

print("="*70)
print("C. elegans: Breathing Coherence Bubbles")
print("="*70)
print(f"  dt={dt}, T={T_total}s, steps={n_steps:,}, grid={Nx}")

# ════════════════════════════════════════════════════════════════════
# CONNECTOME
# ════════════════════════════════════════════════════════════════════
print("\nLoading connectome...")

neurons_set = set()
chemical_edges = []
gap_edges = []

with open(data_dir / "herm_full_edgelist.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row['Source'].strip(); tgt = row['Target'].strip()
        w = int(row['Weight'].strip()); typ = row['Type'].strip()
        neurons_set.add(src); neurons_set.add(tgt)
        if typ == 'chemical': chemical_edges.append((src, tgt, w))
        elif typ == 'electrical': gap_edges.append((src, tgt, w))

positions = {}
with open(data_dir / "neuron_positions.csv") as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            try: positions[parts[0].strip()] = (float(parts[1]), float(parts[2]))
            except ValueError: pass

neurons = sorted(neurons_set)
N = len(neurons)
neuron_idx = {n: i for i, n in enumerate(neurons)}

rng_pos = np.random.RandomState(0)
pos_2d = np.zeros((N, 2))
for i, n in enumerate(neurons):
    pos_2d[i] = positions.get(n, rng_pos.randn(2) * 0.3)

x_coords = pos_2d[:, 0]
pos_1d = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)

grid_x = np.linspace(0, 1, Nx)
neuron_body_idx = np.array([np.argmin(np.abs(grid_x - p)) for p in pos_1d])

# Adjacency
W_chem = np.zeros((N, N))
W_gap = np.zeros((N, N))
for s, t, w in chemical_edges:
    W_chem[neuron_idx[s], neuron_idx[t]] += w
for s, t, w in gap_edges:
    i, j = neuron_idx[s], neuron_idx[t]
    W_gap[i, j] += w; W_gap[j, i] += w

# Row-normalize
W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

# Classification
sensory_pfx = ('ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB',
               'AWC','AFD','BAG','IL1','IL2','OLL','OLQ','PHA','PHB','PLM',
               'ALM','AVM','PVD','FLP')
motor_pfx = ('VA','VB','VD','DA','DB','DD','AS','RMD','RME','SMD','SMB')
sensory = [i for i, n in enumerate(neurons) if n.startswith(sensory_pfx)]
motor = [i for i, n in enumerate(neurons) if n.startswith(motor_pfx)]
inter = [i for i in range(N) if i not in sensory and i not in motor]
VA = [i for i, n in enumerate(neurons) if n.startswith('VA')]
VB = [i for i, n in enumerate(neurons) if n.startswith('VB')]
DA = [i for i, n in enumerate(neurons) if n.startswith('DA')]

# Base distance matrix (will be modulated by body curvature)
base_dist = np.abs(pos_1d[:, None] - pos_1d[None, :])

# Neuron-to-grid weights
sigma_sp = 0.08
neuron_w_grid = np.zeros((Nx, N))
for gi in range(Nx):
    for ni in range(N):
        neuron_w_grid[gi, ni] = np.exp(-(grid_x[gi] - pos_1d[ni])**2 / (2*sigma_sp**2))
    s = neuron_w_grid[gi].sum()
    if s > 0: neuron_w_grid[gi] /= s

# Interpolation
n_left = np.clip(np.floor(pos_1d * (Nx - 1)).astype(int), 0, Nx - 2)
n_frac = pos_1d * (Nx - 1) - n_left

print(f"  {N} neurons, {len(chemical_edges)} chem, {len(gap_edges)} gap")
print(f"  Sensory: {len(sensory)}, Inter: {len(inter)}, Motor: {len(motor)}")

# ════════════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════════════

# Neural oscillators
K_chem = 0.4       # chemical synapse coupling
K_gap = 0.8        # gap junction coupling
K_ephaptic = 1.2   # ephaptic field coupling (distance-dependent, dynamic)
K_drive = 1.0      # sensory drive
sigma_noise = 0.12 # phase noise

# Adaptation (spike-frequency adaptation)
tau_adapt = 3.0    # adaptation timescale (slow — seconds)
g_adapt = 0.4      # adaptation → frequency reduction

# Short-term synaptic depression (Tsodyks-Markram style)
# Resource variables for each coupling channel
tau_rec_gap = 2.0      # gap junction recovery (seconds)
tau_rec_eph = 1.5      # ephaptic recovery
tau_dep = 0.3          # depletion rate constant
U_init = 0.8           # initial resource level

# Coherence field (graded, not bistable)
a_LG = 0.5
b_LG = 2.0
kappa_phi = 0.015
sigma_phi = 0.3
gamma_pump = 0.7
tau_phi = 0.5

# Body mechanics
epsilon_body = 3.0     # elastic restoring
gamma_body = 1.5       # viscous damping
K_muscle = 0.8         # motor → curvature
sigma_body = 0.01
# Body curvature modulates distance: when worm bends, nearby neurons
# on opposite sides of the bend move apart, neurons on same side come closer
body_coupling_mod = 0.3  # how much curvature modulates distances

# Environment
n_sensors = min(len(sensory), 50)
tau_env = 2.0
sigma_env = 0.5

np.random.seed(42)

print(f"\n  Adaptation: τ={tau_adapt}s, g={g_adapt}")
print(f"  Depression: τ_rec={tau_rec_gap}/{tau_rec_eph}s, τ_dep={tau_dep}s")
print(f"  Body coupling modulation: {body_coupling_mod}")

# ════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ════════════════════════════════════════════════════════════════════
rng = np.random.RandomState(77)

theta = 2 * np.pi * rng.rand(N)
adapt = np.zeros(N)                # adaptation variable

# Synaptic resources: one per coupling channel type
# Gap junction resources (N×N symmetric, but we only use upper triangle logic)
u_gap = np.ones((N, N)) * U_init
u_eph = np.ones((N, N)) * U_init

# Coherence field
phi = np.ones(Nx) * 0.15

# Body
kappa_body = np.zeros(Nx)
kappa_dot = np.zeros(Nx)

# Frequency assignment
rng_freq = np.random.RandomState(99)
omegas = np.zeros(N)
for i in inter: omegas[i] = 2*np.pi*(0.7 + 0.3*rng_freq.rand())
for i in sensory: omegas[i] = 2*np.pi*(0.9 + 0.3*rng_freq.rand())
for i in motor: omegas[i] = 2*np.pi*(1.0 + 0.3*rng_freq.rand())

# Environment
def gen_env(n_steps, dt, n_s, tau, sigma, seed=42):
    rng = np.random.RandomState(seed)
    env = np.zeros((n_steps, n_s))
    decay = np.exp(-dt / tau)
    ns = sigma * np.sqrt(2 * dt / tau)
    for t in range(1, n_steps):
        env[t] = env[t-1] * decay + ns * rng.randn(n_s)
    return env

print("  Generating environment...")
sensory_input = gen_env(n_steps, dt, n_sensors, tau_env, sigma_env)

rng_n = np.random.RandomState(123)
rng_phi = np.random.RandomState(456)
rng_body = np.random.RandomState(789)

# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def interp(field, left, frac):
    return field[left] * (1 - frac) + field[left + 1] * frac

def laplacian(f, dx):
    lap = np.zeros_like(f)
    lap[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    lap[0] = (f[1] - f[0]) / dx**2
    lap[-1] = (f[-2] - f[-1]) / dx**2
    return lap

# ════════════════════════════════════════════════════════════════════
# STORAGE
# ════════════════════════════════════════════════════════════════════
n_stored = (n_steps - n_trans) // subsample
phi_hist = np.zeros((n_stored, Nx))
kappa_hist = np.zeros((n_stored, Nx))
adapt_hist = np.zeros((n_stored, N))
theta_hist = np.zeros((n_stored, N))
u_gap_mean_hist = np.zeros(n_stored)    # mean gap resource
u_eph_mean_hist = np.zeros(n_stored)    # mean ephaptic resource
order_r_hist = np.zeros(n_stored)
si = 0

report_interval = max(1, n_steps // 40)
print(f"\n  Will store {n_stored:,} timepoints")
print(f"  Expected runtime: 1-3 hours\n")

# ════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════
print("Running simulation...")

for t in range(1, n_steps):
    # ── Fields at neuron positions ──────────────────────────────
    phi_n = interp(phi, n_left, n_frac)
    phi_clipped = np.clip(phi_n, 0, 1)
    kappa_n = interp(kappa_body, n_left, n_frac)

    # ── Dynamic distance matrix (body curvature modulates distances)
    # Curvature at each neuron's position changes effective distance
    # Simple model: curvature adds a perturbation to the base distance
    kappa_diff = np.abs(kappa_n[:, None] - kappa_n[None, :])
    eff_dist = base_dist + body_coupling_mod * kappa_diff
    eph_sigma = 0.12
    W_eph_dynamic = np.exp(-eff_dist**2 / (2 * eph_sigma**2))
    np.fill_diagonal(W_eph_dynamic, 0)
    W_eph_row_sum = W_eph_dynamic.sum(axis=1, keepdims=True)
    W_eph_dynamic /= np.maximum(W_eph_row_sum, 1)

    # ── Sensory drive ───────────────────────────────────────────
    drive = np.zeros(N)
    for k, idx in enumerate(sensory[:n_sensors]):
        drive[idx] = K_drive * sensory_input[t, k]

    # ── Phase differences (used by all coupling terms) ──────────
    sin_diff = np.sin(theta[:, None] - theta[None, :])

    # ── Chemical synapse coupling (no depression — fast, specific)
    chem = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)

    # ── Gap junction coupling WITH depression ───────────────────
    gap_effective = u_gap * W_gap_norm
    gap = K_gap * np.sum(gap_effective * sin_diff, axis=0)

    # ── Ephaptic coupling WITH depression AND coherence gating ──
    phi_gate = phi_clipped[:, None] * phi_clipped[None, :]
    eph_effective = u_eph * W_eph_dynamic * phi_gate
    eph = K_ephaptic * np.sum(eph_effective * sin_diff, axis=0)

    # ── Proprioceptive feedback (motor neurons sense curvature) ─
    proprio = np.zeros(N)
    for i in motor:
        proprio[i] = 0.3 * np.sin(kappa_n[i] - theta[i])

    # ── Adaptation-modulated frequency ──────────────────────────
    omega_eff = omegas - g_adapt * adapt

    # ── Neural update ───────────────────────────────────────────
    dtheta = omega_eff + drive + chem + gap + eph + proprio
    dtheta += sigma_noise * rng_n.randn(N) / np.sqrt(dt)
    theta += dtheta * dt

    # ── Adaptation update (slow, activity-dependent) ────────────
    # Accumulates when neuron is active (|cos θ| high), decays slowly
    activity = 0.5 * (1 + np.cos(theta))  # ∈ [0, 1]
    dadapt = (-adapt + activity) / tau_adapt
    adapt += dadapt * dt

    # ── Synaptic depression update ──────────────────────────────
    # Activity measure for depression: |sin(θ_i - θ_j)| = coupling usage
    abs_sin_diff = np.abs(sin_diff)

    # Gap junction resources: deplete with usage, recover slowly
    du_gap = (U_init - u_gap) / tau_rec_gap - u_gap * abs_sin_diff / tau_dep
    u_gap += du_gap * dt
    u_gap = np.clip(u_gap, 0.01, 1.0)

    # Ephaptic resources: deplete with usage, recover
    du_eph = (U_init - u_eph) / tau_rec_eph - u_eph * abs_sin_diff / tau_dep
    u_eph += du_eph * dt
    u_eph = np.clip(u_eph, 0.01, 1.0)

    # ── Coherence field update ──────────────────────────────────
    dVdphi = -a_LG * phi + b_LG * phi**3
    lap_phi = laplacian(phi, dx)

    source = np.zeros(Nx)
    for gi in range(Nx):
        w = neuron_w_grid[gi]
        mask = w > 1e-6
        if mask.sum() < 2: continue
        wm = w[mask]
        r_local = np.abs(np.sum(wm * np.exp(1j * theta[mask])) / wm.sum())
        source[gi] = gamma_pump * r_local

    dphi = (-dVdphi + kappa_phi * lap_phi + source) / tau_phi
    dphi = np.clip(dphi, -10, 10)
    phi += dphi * dt + (sigma_phi / np.sqrt(tau_phi)) * rng_phi.randn(Nx) * np.sqrt(dt)
    phi = np.clip(phi, 0, 1.5)

    # ── Body mechanics ──────────────────────────────────────────
    F_muscle = np.zeros(Nx)
    for i in motor:
        gi = neuron_body_idx[i]
        F_muscle[gi] += K_muscle * np.sin(theta[i])

    dkdot = -epsilon_body * kappa_body - gamma_body * kappa_dot + F_muscle
    dkdot += sigma_body * rng_body.randn(Nx) / np.sqrt(dt)
    kappa_dot += dkdot * dt
    kappa_body += kappa_dot * dt
    kappa_body = np.clip(kappa_body, -3, 3)

    # ── Store ───────────────────────────────────────────────────
    if t >= n_trans and (t - n_trans) % subsample == 0 and si < n_stored:
        phi_hist[si] = phi.copy()
        kappa_hist[si] = kappa_body.copy()
        adapt_hist[si] = adapt.copy()
        theta_hist[si] = theta.copy()
        # Mean resource levels (diagonal excluded)
        mask_upper = np.triu_indices(N, k=1)
        u_gap_mean_hist[si] = u_gap[mask_upper].mean()
        u_eph_mean_hist[si] = u_eph[mask_upper].mean()
        order_r_hist[si] = np.abs(np.mean(np.exp(1j * theta)))
        si += 1

    if t % report_interval == 0:
        pct = 100 * t / n_steps
        elapsed = time.time() - wall_start
        eta = (elapsed / (t / n_steps) - elapsed) / 60 if t > 0 else 0
        mp = phi.mean()
        r = np.abs(np.mean(np.exp(1j * theta)))
        ug = u_gap[mask_upper].mean()
        ue = u_eph[mask_upper].mean()
        ad = adapt.mean()
        print(f"  [{pct:5.1f}%] t={t*dt:.0f}s  φ={mp:.3f}  r={r:.3f}  "
              f"u_gap={ug:.3f}  u_eph={ue:.3f}  adapt={ad:.3f}  "
              f"wall={elapsed/60:.1f}m  ETA={eta:.0f}m")

n_stored = si
t_stored = np.arange(n_stored) * dt_stored
wall_total = time.time() - wall_start

# ════════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

def mcorr(phases, A, B):
    if not A or not B: return 0.0
    a = np.mean(np.cos(phases[:, A]), axis=1)
    b = np.mean(np.cos(phases[:, B]), axis=1)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10: return 0.0
    return pearsonr(a, b)[0]

va_vb = mcorr(theta_hist[:n_stored], VA, VB)
va_da = mcorr(theta_hist[:n_stored], VA, DA)

mean_phi = phi_hist[:n_stored].mean(axis=0)

# Breathing frequency: spectral analysis of mean coherence
phi_mean_t = phi_hist[:n_stored].mean(axis=1)
if n_stored > 100:
    from scipy.signal import welch
    freqs, psd = welch(phi_mean_t - phi_mean_t.mean(), fs=1/dt_stored, nperseg=min(512, n_stored//2))
    peak_freq = freqs[np.argmax(psd[1:])+1]  # skip DC
    breath_period = 1/peak_freq if peak_freq > 0 else np.inf
else:
    breath_period = np.inf

print(f"\n  Wall time: {wall_total/3600:.2f} hours ({wall_total/60:.1f} min)")
print(f"  Stored: {n_stored:,} timepoints")
print(f"  Mean φ: {mean_phi.mean():.3f} ± {mean_phi.std():.3f}")
print(f"  Mean r: {order_r_hist[:n_stored].mean():.3f}")
print(f"  Mean u_gap: {u_gap_mean_hist[:n_stored].mean():.3f}")
print(f"  Mean u_eph: {u_eph_mean_hist[:n_stored].mean():.3f}")
print(f"  Mean adapt: {adapt_hist[:n_stored].mean():.3f}")
print(f"  Breath period: {breath_period:.1f}s")
print(f"  Motor: VA-VB={va_vb:.3f}, VA-DA={va_da:.3f}")

# ════════════════════════════════════════════════════════════════════
# FIGURE (16-panel)
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 4, hspace=0.45, wspace=0.4)

ext = [0, t_stored[-1] if n_stored > 1 else 1, 0, 1]

# Row 1: Kymographs
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(phi_hist[:n_stored].T, aspect='auto', origin='lower',
               extent=ext, cmap='inferno', interpolation='bilinear')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
ax.set_title('(a) Coherence φ(x,t)', fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.7)

ax = fig.add_subplot(gs[0, 1])
kmax = max(0.01, np.abs(kappa_hist[:n_stored]).max() * 0.8)
im = ax.imshow(kappa_hist[:n_stored].T, aspect='auto', origin='lower',
               extent=ext, cmap='RdBu_r', vmin=-kmax, vmax=kmax,
               interpolation='bilinear')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
ax.set_title('(b) Body curvature κ(x,t)')
fig.colorbar(im, ax=ax, shrink=0.7)

ax = fig.add_subplot(gs[0, 2])
adapt_mean_body = np.zeros((n_stored, Nx))
for si_i in range(n_stored):
    for gi in range(Nx):
        w = neuron_w_grid[gi]
        adapt_mean_body[si_i, gi] = np.sum(w * adapt_hist[si_i])
im = ax.imshow(adapt_mean_body.T, aspect='auto', origin='lower',
               extent=ext, cmap='YlOrRd', interpolation='bilinear')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
ax.set_title('(c) Adaptation a(x,t)')
fig.colorbar(im, ax=ax, shrink=0.7)

ax = fig.add_subplot(gs[0, 3])
ax.plot(t_stored[:n_stored], u_gap_mean_hist[:n_stored], color='#2196F3',
        linewidth=0.5, label='Gap junctions', alpha=0.8)
ax.plot(t_stored[:n_stored], u_eph_mean_hist[:n_stored], color='#FF9800',
        linewidth=0.5, label='Ephaptic', alpha=0.8)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean resource u')
ax.set_title('(d) Synaptic depression')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# Row 2: Time series
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_stored[:n_stored], phi_mean_t, color='#FF9800', linewidth=0.8, label='⟨φ⟩')
ax.plot(t_stored[:n_stored], order_r_hist[:n_stored], color='#2196F3',
        linewidth=0.5, alpha=0.7, label='r')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Value')
ax.set_title('(e) Coherence + sync over time')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[1, 1])
window = min(200, n_stored // 5)
if window > 10 and VA and VB:
    va_sig = np.mean(np.cos(theta_hist[:n_stored, VA]), axis=1)
    vb_sig = np.mean(np.cos(theta_hist[:n_stored, VB]), axis=1)
    rc = np.zeros(n_stored - window)
    for i in range(len(rc)):
        a, b = va_sig[i:i+window], vb_sig[i:i+window]
        if np.std(a) > 1e-10 and np.std(b) > 1e-10:
            rc[i] = pearsonr(a, b)[0]
    ax.plot(t_stored[window//2:window//2+len(rc)], rc, color='#EF5350', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Time (s)'); ax.set_ylabel('VA-VB corr')
ax.set_title('(f) Motor coordination'); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[1, 2])
adapt_mean_t = adapt_hist[:n_stored].mean(axis=1)
ax.plot(t_stored[:n_stored], adapt_mean_t, color='#9C27B0', linewidth=0.8)
ax.set_xlabel('Time (s)'); ax.set_ylabel('⟨adaptation⟩')
ax.set_title('(g) Neural adaptation'); ax.grid(True, alpha=0.2)

# Breathing spectrum
ax = fig.add_subplot(gs[1, 3])
if n_stored > 100:
    ax.semilogy(freqs[1:], psd[1:], color='#FF9800', linewidth=1)
    ax.axvline(peak_freq, color='gray', linestyle=':', alpha=0.5,
               label=f'Peak: {breath_period:.1f}s')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD')
    ax.set_title(f'(h) Coherence spectrum (T≈{breath_period:.0f}s)')
    ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Row 3: Spatial profiles
ax = fig.add_subplot(gs[2, 0])
std_phi = phi_hist[:n_stored].std(axis=0)
ax.fill_between(grid_x, mean_phi - std_phi, mean_phi + std_phi,
                alpha=0.3, color='#FF9800')
ax.plot(grid_x, mean_phi, color='#FF9800', linewidth=2)
ax.set_xlabel('Body axis'); ax.set_ylabel('⟨φ⟩')
ax.set_title('(i) Coherence profile'); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[2, 1])
# Local D_eff vs position
for gi in range(0, Nx, Nx // 10):
    mask = np.abs(pos_1d - grid_x[gi]) < 0.08
    if mask.sum() < 3: continue
    local = np.column_stack([np.cos(theta_hist[:n_stored, mask]),
                             np.sin(theta_hist[:n_stored, mask])])
    X = local - local.mean(0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s**2/len(X); ev = ev[ev > 1e-10]
    deff = (ev.sum())**2 / (ev**2).sum() if len(ev) > 0 else 0
    ax.scatter(mean_phi[gi], deff, c='#2196F3', s=30, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Local ⟨φ⟩'); ax.set_ylabel('Local D_eff')
ax.set_title('(j) Coherence → low D_eff'); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[2, 2])
grad_phi_sq = np.zeros(Nx)
for s_i in range(n_stored):
    gp = np.gradient(phi_hist[s_i], dx)
    grad_phi_sq += gp**2
grad_phi_sq /= n_stored
ax.fill_between(grid_x, 0, grad_phi_sq, alpha=0.4, color='#E91E63')
ax.plot(grid_x, grad_phi_sq, color='#E91E63', linewidth=1.5)
ax.set_xlabel('Body axis'); ax.set_ylabel('|∇φ|²')
ax.set_title('(k) Surface tension'); ax.grid(True, alpha=0.2)

# Cross-correlation of coherence and resource depletion
ax = fig.add_subplot(gs[2, 3])
ax.scatter(phi_mean_t[::5], u_eph_mean_hist[:n_stored:5],
           c=t_stored[:n_stored:5], cmap='viridis', s=5, alpha=0.5)
ax.set_xlabel('⟨φ⟩'); ax.set_ylabel('⟨u_eph⟩')
ax.set_title('(l) φ depletes resources'); ax.grid(True, alpha=0.2)

# Row 4: Phase relationships
ax = fig.add_subplot(gs[3, 0])
# Coherence at head vs tail
phi_head = phi_hist[:n_stored, :Nx//4].mean(axis=1)
phi_tail = phi_hist[:n_stored, 3*Nx//4:].mean(axis=1)
ax.plot(t_stored[:n_stored], phi_head, color='#EF5350', linewidth=0.8, label='Head')
ax.plot(t_stored[:n_stored], phi_tail, color='#2196F3', linewidth=0.8, label='Tail')
ax.set_xlabel('Time (s)'); ax.set_ylabel('φ')
ax.set_title('(m) Head vs tail coherence')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[3, 1])
# Adaptation at head vs tail
adapt_head = np.zeros(n_stored)
adapt_tail = np.zeros(n_stored)
head_neurons = [i for i in range(N) if pos_1d[i] < 0.25]
tail_neurons = [i for i in range(N) if pos_1d[i] > 0.75]
if head_neurons:
    adapt_head = adapt_hist[:n_stored, head_neurons].mean(axis=1)
if tail_neurons:
    adapt_tail = adapt_hist[:n_stored, tail_neurons].mean(axis=1)
ax.plot(t_stored[:n_stored], adapt_head, color='#EF5350', linewidth=0.8, label='Head')
ax.plot(t_stored[:n_stored], adapt_tail, color='#2196F3', linewidth=0.8, label='Tail')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Adaptation')
ax.set_title('(n) Head vs tail adaptation')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[3, 2])
# Phase portrait: φ vs u_eph (should show limit cycle if breathing)
ax.plot(phi_mean_t, u_eph_mean_hist[:n_stored], color='#673AB7',
        linewidth=0.3, alpha=0.5)
ax.scatter(phi_mean_t[0], u_eph_mean_hist[0], c='green', s=50, zorder=5, label='Start')
ax.scatter(phi_mean_t[-1], u_eph_mean_hist[n_stored-1], c='red', s=50, zorder=5, label='End')
ax.set_xlabel('⟨φ⟩'); ax.set_ylabel('⟨u_eph⟩')
ax.set_title('(o) Phase portrait (φ vs resource)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# Summary text
ax = fig.add_subplot(gs[3, 3])
ax.axis('off')
summary = (
    f"Wall time: {wall_total/3600:.2f} hrs\n"
    f"Sim time: {T_total}s\n\n"
    f"⟨φ⟩ = {mean_phi.mean():.3f} ± {mean_phi.std():.3f}\n"
    f"⟨r⟩ = {order_r_hist[:n_stored].mean():.3f}\n"
    f"⟨u_gap⟩ = {u_gap_mean_hist[:n_stored].mean():.3f}\n"
    f"⟨u_eph⟩ = {u_eph_mean_hist[:n_stored].mean():.3f}\n"
    f"⟨adapt⟩ = {adapt_mean_t.mean():.3f}\n"
    f"Breath T ≈ {breath_period:.1f}s\n\n"
    f"VA-VB = {va_vb:.3f}\n"
    f"VA-DA = {va_da:.3f}"
)
ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_title('(p) Summary')

plt.savefig(fig_dir / 'fig_breathing.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_breathing.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved fig_breathing")

print(f"\nTotal wall time: {wall_total/3600:.2f} hours")
print("Done!")
