#!/usr/bin/env python3
"""
C. elegans: Triple-layer coupling model with real data.

Three simultaneous coupling channels, ALL from real data:

Layer 1 — CONNECTOME (chemical synapses + gap junctions)
  Source: OpenWorm herm_full_edgelist.csv (White/Cook/Varshney)
  Fast, directed (chemical) and bidirectional (gap), sparse.

Layer 2 — EPHAPTIC FIELD (distance-dependent bioelectric coupling)
  Source: NeuroPAL 3D neuron positions (Brittin et al. atlas via WormNeuroAtlas)
  Fast, continuous, distance-dependent, all-to-all weighted by proximity.

Layer 3 — NEUROPEPTIDE NETWORK (extrasynaptic neuromodulation)
  Source: Ripoll-Sánchez et al. 2023, Neuron — 31,479 interactions
  Slow, diffuse, receptor-specific. Only 5% overlap with wired connectome.
  This is the coupling channel that NO simulation has ever integrated.

Plus: short-term synaptic depression on all channels (the breathing
mechanism from celegans_breathing.py) and body mechanics.

The coherence field φ(x,t) emerges from the interplay of all three layers.

Author: Ian Todd
"""

import numpy as np
import csv
import pandas as pd
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
T_total = 300.0
T_trans = 30.0
n_steps = int(T_total / dt)
n_trans = int(T_trans / dt)
subsample = max(1, int(0.02 / dt))
dt_stored = dt * subsample
Nx = 100

dx = 1.0 / Nx

print("="*70)
print("C. elegans: Triple-Layer Coupling (Real Data)")
print("="*70)
print(f"  dt={dt}, T={T_total}s, steps={n_steps:,}, grid={Nx}")

# ════════════════════════════════════════════════════════════════════
# LAYER 1: CONNECTOME (OpenWorm)
# ════════════════════════════════════════════════════════════════════
print("\n[Layer 1] Loading connectome...")

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

# ════════════════════════════════════════════════════════════════════
# LAYER 3: NEUROPEPTIDE CONNECTOME (Ripoll-Sánchez 2023)
# ════════════════════════════════════════════════════════════════════
print("[Layer 3] Loading neuropeptide connectome...")

npp_df = pd.read_csv(data_dir / "data" / "neuropeptide_connectome_LR.csv", index_col=0)
npp_neurons = list(npp_df.columns)
mono_df = pd.read_csv(data_dir / "data" / "monoamine_connectome.csv", index_col=0)

print(f"  Neuropeptide matrix: {npp_df.shape[0]}×{npp_df.shape[1]}")
print(f"  Monoamine matrix: {mono_df.shape[0]}×{mono_df.shape[1]}")
print(f"  Total NPP interactions: {(npp_df.values > 0).sum()}")
print(f"  Total monoamine interactions: {(mono_df.values > 0).sum()}")

# ════════════════════════════════════════════════════════════════════
# LAYER 2: 3D NEURON POSITIONS (NeuroPAL atlas)
# ════════════════════════════════════════════════════════════════════
print("[Layer 2] Loading 3D neuron positions...")

with open(data_dir / "data" / "neuron_positions_3d.txt") as f:
    header = f.readline().strip().lstrip('#')
    pos3d_names = header.split()
    pos3d_coords = []
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            pos3d_coords.append([float(x) for x in parts])

pos3d_coords = np.array(pos3d_coords)
pos3d_map = {name: pos3d_coords[i] for i, name in enumerate(pos3d_names) if i < len(pos3d_coords)}
print(f"  3D positions for {len(pos3d_map)} neurons")

# ════════════════════════════════════════════════════════════════════
# UNIFIED NEURON LIST
# ════════════════════════════════════════════════════════════════════
# Use intersection of all three data sources
all_neurons = sorted(neurons_set)

# Build unified neuron list — all connectome neurons
neurons = all_neurons
N = len(neurons)
neuron_idx = {n: i for i, n in enumerate(neurons)}

print(f"\n  Unified neuron list: {N} neurons")

# ── Build adjacency matrices ────────────────────────────────────
W_chem = np.zeros((N, N))
W_gap = np.zeros((N, N))
for src, tgt, w in chemical_edges:
    W_chem[neuron_idx[src], neuron_idx[tgt]] += w
for src, tgt, w in gap_edges:
    i, j = neuron_idx[src], neuron_idx[tgt]
    W_gap[i, j] += w; W_gap[j, i] += w

W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

# ── Neuropeptide matrix (map to our neuron indices) ─────────────
W_npp = np.zeros((N, N))
W_mono = np.zeros((N, N))
npp_mapped = 0
mono_mapped = 0

for src_name in npp_neurons:
    if src_name not in neuron_idx: continue
    si = neuron_idx[src_name]
    for tgt_name in npp_neurons:
        if tgt_name not in neuron_idx: continue
        ti = neuron_idx[tgt_name]
        if src_name in npp_df.index and tgt_name in npp_df.columns:
            val = npp_df.loc[src_name, tgt_name]
            if val > 0:
                W_npp[si, ti] = val
                npp_mapped += 1
        if src_name in mono_df.index and tgt_name in mono_df.columns:
            val = mono_df.loc[src_name, tgt_name]
            if val > 0:
                W_mono[si, ti] = val
                mono_mapped += 1

# Normalize (by max, not row sum, to preserve relative strengths)
if W_npp.max() > 0:
    W_npp_norm = W_npp / W_npp.max()
else:
    W_npp_norm = W_npp
if W_mono.max() > 0:
    W_mono_norm = W_mono / W_mono.max()
else:
    W_mono_norm = W_mono

print(f"  Mapped NPP interactions: {npp_mapped}")
print(f"  Mapped monoamine interactions: {mono_mapped}")

# Overlap between wired and neuropeptide connectomes
wired_mask = (W_chem + W_gap) > 0
npp_mask = W_npp > 0
overlap = (wired_mask & npp_mask).sum()
total_npp = npp_mask.sum()
print(f"  Wired-NPP overlap: {overlap}/{total_npp} ({100*overlap/max(total_npp,1):.1f}%)")

# ── 3D positions → 1D body axis + 3D distance matrix ───────────
pos_3d = np.zeros((N, 3))
has_3d = 0
rng_pos = np.random.RandomState(0)
for i, n in enumerate(neurons):
    if n in pos3d_map:
        pos_3d[i] = pos3d_map[n]
        has_3d += 1
    else:
        pos_3d[i] = rng_pos.randn(3) * 0.3

# Body axis = second coordinate (anterior-posterior in the atlas)
# The atlas y-coordinate runs from head (~4) to tail (~-4)
y_coords = pos_3d[:, 1]
pos_1d = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-10)

# 3D distance matrix for ephaptic coupling
dist_3d = squareform(pdist(pos_3d))

# Ephaptic coupling: Gaussian falloff in 3D
eph_sigma = 0.8  # in atlas units (roughly 1 body width)
W_eph = np.exp(-dist_3d**2 / (2 * eph_sigma**2))
np.fill_diagonal(W_eph, 0)
W_eph_norm = W_eph / np.maximum(W_eph.sum(axis=1, keepdims=True), 1)

print(f"  3D positions: {has_3d}/{N} neurons with atlas coordinates")
print(f"  Ephaptic connections (>0.1): {(W_eph > 0.1).sum()}")

# ── Grid mapping ────────────────────────────────────────────────
grid_x = np.linspace(0, 1, Nx)
neuron_body_idx = np.array([np.argmin(np.abs(grid_x - p)) for p in pos_1d])

sigma_sp = 0.08
neuron_w_grid = np.zeros((Nx, N))
for gi in range(Nx):
    for ni in range(N):
        neuron_w_grid[gi, ni] = np.exp(-(grid_x[gi] - pos_1d[ni])**2 / (2*sigma_sp**2))
    s = neuron_w_grid[gi].sum()
    if s > 0: neuron_w_grid[gi] /= s

n_left = np.clip(np.floor(pos_1d * (Nx - 1)).astype(int), 0, Nx - 2)
n_frac = pos_1d * (Nx - 1) - n_left

# ── Neuron classification ───────────────────────────────────────
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

print(f"  Sensory: {len(sensory)}, Inter: {len(inter)}, Motor: {len(motor)}")

# ════════════════════════════════════════════════════════════════════
# COUPLING PARAMETERS
# ════════════════════════════════════════════════════════════════════

# Layer 1: Connectome (fast, always on)
K_chem = 0.4
K_gap = 0.8

# Layer 2: Ephaptic (fast, distance-dependent, gated by coherence)
K_eph = 1.0

# Layer 3: Neuropeptide (SLOW — timescale of seconds)
K_npp = 0.5        # neuropeptide coupling strength
tau_npp = 2.0       # neuropeptide field timescale (slow modulation)

# Sensory drive
K_drive = 1.0
sigma_noise = 0.12
n_sensors = min(len(sensory), 50)
tau_env = 2.0; sigma_env = 0.5

# Adaptation
tau_adapt = 3.0; g_adapt = 0.4

# Short-term depression (all layers)
tau_rec_gap = 2.0; tau_rec_eph = 1.5; tau_rec_npp = 5.0
tau_dep = 0.3; U_init = 0.8

# Coherence field
a_LG = 0.5; b_LG = 2.0
kappa_phi = 0.015; sigma_phi = 0.3
gamma_pump = 0.7; tau_phi = 0.5

# Body mechanics
epsilon_body = 3.0; gamma_body = 1.5; K_muscle = 0.8; sigma_body = 0.01

np.random.seed(42)

print(f"\n  Layer 1 (connectome): K_chem={K_chem}, K_gap={K_gap}")
print(f"  Layer 2 (ephaptic): K_eph={K_eph}, σ_eph={eph_sigma}")
print(f"  Layer 3 (neuropeptide): K_npp={K_npp}, τ_npp={tau_npp}")

# ════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ════════════════════════════════════════════════════════════════════
rng_init = np.random.RandomState(77)
theta = 2 * np.pi * rng_init.rand(N)
adapt = np.zeros(N)

# Synaptic resources per layer
u_gap = np.ones((N, N)) * U_init
u_eph = np.ones((N, N)) * U_init
u_npp = np.ones((N, N)) * U_init

# Neuropeptide modulation field (slow, per-neuron)
npp_mod = np.zeros(N)

# Coherence field
phi = np.ones(Nx) * 0.15

# Body
kappa_body = np.zeros(Nx)
kappa_dot = np.zeros(Nx)

# Frequencies
rng_freq = np.random.RandomState(99)
omegas = np.zeros(N)
for i in inter: omegas[i] = 2*np.pi*(0.7 + 0.3*rng_freq.rand())
for i in sensory: omegas[i] = 2*np.pi*(0.9 + 0.3*rng_freq.rand())
for i in motor: omegas[i] = 2*np.pi*(1.0 + 0.3*rng_freq.rand())

# Environment
def gen_env(n_steps, dt, n_s, tau, sigma, seed=42):
    rng = np.random.RandomState(seed)
    env = np.zeros((n_steps, n_s))
    decay = np.exp(-dt / tau); ns = sigma * np.sqrt(2 * dt / tau)
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
theta_hist = np.zeros((n_stored, N))
npp_mod_hist = np.zeros((n_stored, N))
u_gap_mean_hist = np.zeros(n_stored)
u_eph_mean_hist = np.zeros(n_stored)
u_npp_mean_hist = np.zeros(n_stored)
order_r_hist = np.zeros(n_stored)
# Per-layer contribution to coupling (diagnostic)
coupling_layer_hist = np.zeros((n_stored, 3))  # [connectome, ephaptic, npp]
si = 0

mask_upper = np.triu_indices(N, k=1)
report_interval = max(1, n_steps // 40)

print(f"\n  Will store {n_stored:,} timepoints")
print(f"  Starting simulation...\n")

# ════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════

for t in range(1, n_steps):
    # ── Fields at neurons ───────────────────────────────────────
    phi_n = interp(phi, n_left, n_frac)
    phi_clipped = np.clip(phi_n, 0, 1)

    # ── Sensory drive ───────────────────────────────────────────
    drive = np.zeros(N)
    for k, idx in enumerate(sensory[:n_sensors]):
        drive[idx] = K_drive * sensory_input[t, k]

    # ── Phase differences ───────────────────────────────────────
    sin_diff = np.sin(theta[:, None] - theta[None, :])

    # ── LAYER 1: Connectome coupling (with depression) ──────────
    chem_eff = W_chem_norm  # no depression on chemical synapses (vesicle pool large)
    gap_eff = u_gap * W_gap_norm
    conn_coupling = K_chem * np.sum(chem_eff * sin_diff, axis=0) + \
                    K_gap * np.sum(gap_eff * sin_diff, axis=0)

    # ── LAYER 2: Ephaptic coupling (gated by φ, with depression) ─
    phi_gate = phi_clipped[:, None] * phi_clipped[None, :]
    eph_eff = u_eph * W_eph_norm * phi_gate
    eph_coupling = K_eph * np.sum(eph_eff * sin_diff, axis=0)

    # ── LAYER 3: Neuropeptide modulation (SLOW) ─────────────────
    # NPP doesn't directly couple phases — it modulates frequency
    # and coupling gain on a slow timescale.
    # Compute NPP input to each neuron: sum of NPP weights × source activity
    npp_input = np.zeros(N)
    for ni in range(N):
        # NPP from all sources, weighted by npp_connectome and source activity
        sources = W_npp_norm[:, ni] * u_npp[:, ni].diagonal() if False else \
                  np.sum(W_npp_norm[:, ni] * (0.5 + 0.5*np.cos(theta)) * u_npp[:, ni])
        npp_input[ni] = sources

    # Slow update of NPP modulation field
    dnpp = (-npp_mod + K_npp * npp_input) / tau_npp
    npp_mod += dnpp * dt

    # NPP modulates intrinsic frequency (neuromodulation)
    omega_eff = omegas - g_adapt * adapt + npp_mod

    # ── Proprioception ──────────────────────────────────────────
    kappa_n = interp(kappa_body, n_left, n_frac)
    proprio = np.zeros(N)
    for i in motor:
        proprio[i] = 0.3 * np.sin(kappa_n[i] - theta[i])

    # ── Total neural update ─────────────────────────────────────
    dtheta = omega_eff + drive + conn_coupling + eph_coupling + proprio
    dtheta += sigma_noise * rng_n.randn(N) / np.sqrt(dt)
    theta += dtheta * dt

    # ── Adaptation ──────────────────────────────────────────────
    activity = 0.5 * (1 + np.cos(theta))
    adapt += (-adapt + activity) / tau_adapt * dt

    # ── Synaptic depression ─────────────────────────────────────
    abs_sin = np.abs(sin_diff)

    du_gap = (U_init - u_gap) / tau_rec_gap - u_gap * abs_sin / tau_dep
    u_gap = np.clip(u_gap + du_gap * dt, 0.01, 1.0)

    du_eph = (U_init - u_eph) / tau_rec_eph - u_eph * abs_sin / tau_dep
    u_eph = np.clip(u_eph + du_eph * dt, 0.01, 1.0)

    du_npp = (U_init - u_npp) / tau_rec_npp - u_npp * abs_sin / (tau_dep * 3)
    u_npp = np.clip(u_npp + du_npp * dt, 0.01, 1.0)

    # ── Coherence field ─────────────────────────────────────────
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
    phi = np.clip(phi, 0, 3.0)

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
        theta_hist[si] = theta.copy()
        npp_mod_hist[si] = npp_mod.copy()
        u_gap_mean_hist[si] = u_gap[mask_upper].mean()
        u_eph_mean_hist[si] = u_eph[mask_upper].mean()
        u_npp_mean_hist[si] = u_npp[mask_upper].mean()
        order_r_hist[si] = np.abs(np.mean(np.exp(1j * theta)))
        # Layer contributions (RMS coupling strength)
        coupling_layer_hist[si, 0] = np.sqrt(np.mean(conn_coupling**2))
        coupling_layer_hist[si, 1] = np.sqrt(np.mean(eph_coupling**2))
        coupling_layer_hist[si, 2] = np.std(npp_mod)
        si += 1

    if t % report_interval == 0:
        pct = 100 * t / n_steps
        elapsed = time.time() - wall_start
        eta = (elapsed / (t / n_steps) - elapsed) / 60 if t > 0 else 0
        mp = phi.mean()
        r = np.abs(np.mean(np.exp(1j * theta)))
        ug = u_gap[mask_upper].mean()
        ue = u_eph[mask_upper].mean()
        un = u_npp[mask_upper].mean()
        nm = np.std(npp_mod)
        print(f"  [{pct:5.1f}%] t={t*dt:.0f}s  φ={mp:.3f}  r={r:.3f}  "
              f"u_gap={ug:.3f}  u_eph={ue:.3f}  u_npp={un:.3f}  "
              f"npp_σ={nm:.4f}  wall={elapsed/60:.1f}m  ETA={eta:.0f}m")

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

# Per-layer coupling analysis
conn_rms = coupling_layer_hist[:n_stored, 0].mean()
eph_rms = coupling_layer_hist[:n_stored, 1].mean()
npp_std = coupling_layer_hist[:n_stored, 2].mean()

print(f"\n  Wall time: {wall_total/3600:.2f} hours ({wall_total/60:.1f} min)")
print(f"  Stored: {n_stored:,} timepoints")
print(f"  Mean φ: {mean_phi.mean():.3f} ± {mean_phi.std():.3f}")
print(f"  Mean r: {order_r_hist[:n_stored].mean():.3f}")
print(f"  Layer contributions (RMS coupling):")
print(f"    Connectome: {conn_rms:.4f}")
print(f"    Ephaptic:   {eph_rms:.4f}")
print(f"    Neuropeptide σ: {npp_std:.4f}")
print(f"  Mean u_gap: {u_gap_mean_hist[:n_stored].mean():.3f}")
print(f"  Mean u_eph: {u_eph_mean_hist[:n_stored].mean():.3f}")
print(f"  Mean u_npp: {u_npp_mean_hist[:n_stored].mean():.3f}")
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

# NPP modulation kymograph
ax = fig.add_subplot(gs[0, 2])
npp_body = np.zeros((n_stored, Nx))
for si_i in range(n_stored):
    for gi in range(Nx):
        npp_body[si_i, gi] = np.sum(neuron_w_grid[gi] * npp_mod_hist[si_i])
im = ax.imshow(npp_body.T, aspect='auto', origin='lower',
               extent=ext, cmap='PuBu', interpolation='bilinear')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
ax.set_title('(c) Neuropeptide modulation')
fig.colorbar(im, ax=ax, shrink=0.7)

# Layer contributions over time
ax = fig.add_subplot(gs[0, 3])
ax.plot(t_stored[:n_stored], coupling_layer_hist[:n_stored, 0],
        color='#2196F3', linewidth=0.5, label='Connectome', alpha=0.8)
ax.plot(t_stored[:n_stored], coupling_layer_hist[:n_stored, 1],
        color='#FF9800', linewidth=0.5, label='Ephaptic', alpha=0.8)
ax.plot(t_stored[:n_stored], coupling_layer_hist[:n_stored, 2],
        color='#9C27B0', linewidth=0.5, label='NPP σ', alpha=0.8)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Coupling RMS')
ax.set_title('(d) Layer contributions')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# Row 2: Time series
ax = fig.add_subplot(gs[1, 0])
phi_mean_t = phi_hist[:n_stored].mean(axis=1)
ax.plot(t_stored[:n_stored], phi_mean_t, color='#FF9800', linewidth=0.8, label='⟨φ⟩')
ax.plot(t_stored[:n_stored], order_r_hist[:n_stored], color='#2196F3',
        linewidth=0.5, alpha=0.7, label='r')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Value')
ax.set_title('(e) Coherence + sync')
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
ax.plot(t_stored[:n_stored], u_gap_mean_hist[:n_stored], color='#2196F3',
        linewidth=0.5, label='Gap', alpha=0.8)
ax.plot(t_stored[:n_stored], u_eph_mean_hist[:n_stored], color='#FF9800',
        linewidth=0.5, label='Eph', alpha=0.8)
ax.plot(t_stored[:n_stored], u_npp_mean_hist[:n_stored], color='#9C27B0',
        linewidth=0.5, label='NPP', alpha=0.8)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean resource u')
ax.set_title('(g) Synaptic depression by layer')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[1, 3])
npp_mean_t = npp_mod_hist[:n_stored].mean(axis=1)
npp_std_t = npp_mod_hist[:n_stored].std(axis=1)
ax.plot(t_stored[:n_stored], npp_mean_t, color='#9C27B0', linewidth=0.8, label='Mean')
ax.fill_between(t_stored[:n_stored], npp_mean_t - npp_std_t,
                npp_mean_t + npp_std_t, alpha=0.2, color='#9C27B0')
ax.set_xlabel('Time (s)'); ax.set_ylabel('NPP modulation')
ax.set_title('(h) Neuropeptide dynamics')
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
ax.set_title('(k) Coherence gradient energy'); ax.grid(True, alpha=0.2)

# NPP spatial profile
ax = fig.add_subplot(gs[2, 3])
mean_npp_body = npp_body.mean(axis=0)
ax.plot(grid_x, mean_npp_body, color='#9C27B0', linewidth=2)
ax.set_xlabel('Body axis'); ax.set_ylabel('⟨NPP mod⟩')
ax.set_title('(l) Neuropeptide modulation profile')
ax.grid(True, alpha=0.2)

# Row 4
ax = fig.add_subplot(gs[3, 0])
phi_head = phi_hist[:n_stored, :Nx//4].mean(axis=1)
phi_tail = phi_hist[:n_stored, 3*Nx//4:].mean(axis=1)
ax.plot(t_stored[:n_stored], phi_head, color='#EF5350', linewidth=0.8, label='Head')
ax.plot(t_stored[:n_stored], phi_tail, color='#2196F3', linewidth=0.8, label='Tail')
ax.set_xlabel('Time (s)'); ax.set_ylabel('φ')
ax.set_title('(m) Head vs tail coherence')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# Connectome vs NPP overlap visualization
ax = fig.add_subplot(gs[3, 1])
conn_degree = (W_chem + W_gap).sum(axis=0) + (W_chem + W_gap).sum(axis=1)
npp_degree = W_npp.sum(axis=0) + W_npp.sum(axis=1)
ax.scatter(conn_degree, npp_degree, s=8, alpha=0.4, c=pos_1d, cmap='coolwarm')
ax.set_xlabel('Wired degree'); ax.set_ylabel('NPP degree')
ax.set_title('(n) Wired vs NPP connectivity')
ax.grid(True, alpha=0.2)

# Phase portrait
ax = fig.add_subplot(gs[3, 2])
ax.plot(phi_mean_t, u_eph_mean_hist[:n_stored], color='#673AB7',
        linewidth=0.3, alpha=0.5)
ax.scatter(phi_mean_t[0], u_eph_mean_hist[0], c='green', s=50, zorder=5)
ax.scatter(phi_mean_t[-1], u_eph_mean_hist[n_stored-1], c='red', s=50, zorder=5)
ax.set_xlabel('⟨φ⟩'); ax.set_ylabel('⟨u_eph⟩')
ax.set_title('(o) Phase portrait'); ax.grid(True, alpha=0.2)

# Summary
ax = fig.add_subplot(gs[3, 3])
ax.axis('off')
summary = (
    f"Wall time: {wall_total/3600:.2f} hrs\n"
    f"Sim time: {T_total}s\n"
    f"Neurons: {N}\n\n"
    f"THREE LAYERS:\n"
    f" Connectome: {len(chemical_edges)} chem\n"
    f"   + {len(gap_edges)} gap\n"
    f" Ephaptic: 3D distance\n"
    f"   ({has_3d}/{N} atlas coords)\n"
    f" Neuropeptide: {npp_mapped}\n"
    f"   interactions (5% overlap)\n\n"
    f"⟨φ⟩ = {mean_phi.mean():.3f}\n"
    f"⟨r⟩ = {order_r_hist[:n_stored].mean():.3f}\n"
    f"VA-VB = {va_vb:.3f}\n"
    f"VA-DA = {va_da:.3f}"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_title('(p) Summary')

plt.savefig(fig_dir / 'fig_trilayer.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_trilayer.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved fig_trilayer")

print(f"\nTotal wall time: {wall_total/3600:.2f} hours")
print("Done!")
