#!/usr/bin/env python3
"""
C. elegans: Low-D (connectome-only) vs High-D (connectome + fields + broadcast).

Three models of increasing dimensionality on the real connectome:

Model A — Connectome only:
  Chemical synapses (directed, weighted) + gap junctions (static conductances).
  This is approximately what OpenWorm/c302 does. The wiring diagram IS the model.

Model B — Connectome + ephaptic field:
  Same connectome PLUS distance-dependent ephaptic coupling between ALL neurons
  based on their physical positions. Neurons that are spatially close influence
  each other even without synaptic connections. This adds a continuous, high-D
  coupling channel that the connectome doesn't capture.

Model C — Connectome + ephaptic + neuropeptide broadcast + proprioception:
  Same as B PLUS a slow, global neuropeptide-like modulatory field (broadcast
  coupling that affects all neurons) PLUS proprioceptive feedback from motor
  output back to sensory neurons (the organism senses its own movement).
  This is the high-D model: the worm as a coupled oscillatory field, not
  just a circuit.

All three models use coupled phase oscillators with frequency hierarchy
(interneurons slow, sensory moderate, motor faster). The ONLY difference
is which coupling channels are active.

Metrics:
- Future mutual information I(X_int(t); Y_env(t+Δt))
- Effective dimensionality D_eff
- Motor coordination: cross-correlation between ventral motor neuron classes
  (VA/VB/DA/DB should show travelling-wave-like coordination for locomotion)

Data:
- herm_full_edgelist.csv — OpenWorm connectome (448 neurons, 7379 edges)
- neuron_positions.csv — 2D neuron positions from White et al. / Choe lab

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

# ════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════
print("Loading data...")

# ── Connectome ──────────────────────────────────────────────────────
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

# ── Neuron positions ────────────────────────────────────────────────
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

# ── Build neuron list (intersection of connectome and positions) ────
# Use all connectome neurons; assign random positions to those without
neurons = sorted(neurons_set)
N = len(neurons)
neuron_idx = {n: i for i, n in enumerate(neurons)}

pos_array = np.zeros((N, 2))
has_position = 0
for i, n in enumerate(neurons):
    # Try exact match, then common prefixes
    if n in positions:
        pos_array[i] = positions[n]
        has_position += 1
    else:
        # Assign random position near center
        pos_array[i] = np.random.randn(2) * 0.3

print(f"  {N} neurons, {has_position} with known positions")
print(f"  {len(chemical_edges)} chemical, {len(gap_edges)} gap junctions")

# ── Adjacency matrices ──────────────────────────────────────────────
W_chem = np.zeros((N, N))
W_gap = np.zeros((N, N))

for src, tgt, w in chemical_edges:
    W_chem[neuron_idx[src], neuron_idx[tgt]] += w
for src, tgt, w in gap_edges:
    i, j = neuron_idx[src], neuron_idx[tgt]
    W_gap[i, j] += w
    W_gap[j, i] += w

# Row-normalise
W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

# ── Distance matrix for ephaptic coupling ───────────────────────────
dist_matrix = squareform(pdist(pos_array))
# Ephaptic coupling: Gaussian falloff with distance, sigma = 0.15 (body units)
ephaptic_sigma = 0.15
W_ephaptic = np.exp(-dist_matrix**2 / (2 * ephaptic_sigma**2))
np.fill_diagonal(W_ephaptic, 0)
# Row-normalise
W_ephaptic_norm = W_ephaptic / np.maximum(W_ephaptic.sum(axis=1, keepdims=True), 1)

print(f"  Ephaptic connections (>0.1 strength): {(W_ephaptic > 0.1).sum()}")

# ── Neuron classification ───────────────────────────────────────────
sensory_prefixes = ('ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB',
                    'AWC','AFD','BAG','IL1','IL2','OLL','OLQ','PHA','PHB','PLM',
                    'ALM','AVM','PVD','FLP')
motor_prefixes = ('VA','VB','VD','DA','DB','DD','AS','RMD','RME','SMD','SMB')

sensory = [i for i, n in enumerate(neurons) if n.startswith(sensory_prefixes)]
motor = [i for i, n in enumerate(neurons) if n.startswith(motor_prefixes)]
inter = [i for i in range(N) if i not in sensory and i not in motor]

# Motor neuron subclasses for coordination analysis
VA = [i for i, n in enumerate(neurons) if n.startswith('VA')]
VB = [i for i, n in enumerate(neurons) if n.startswith('VB')]
DA = [i for i, n in enumerate(neurons) if n.startswith('DA')]
DB = [i for i, n in enumerate(neurons) if n.startswith('DB')]
VD = [i for i, n in enumerate(neurons) if n.startswith('VD')]
DD = [i for i, n in enumerate(neurons) if n.startswith('DD')]

print(f"  Sensory: {len(sensory)}, Inter: {len(inter)}, Motor: {len(motor)}")
print(f"  Motor subclasses: VA={len(VA)}, VB={len(VB)}, DA={len(DA)}, DB={len(DB)}, VD={len(VD)}, DD={len(DD)}")

# ════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ════════════════════════════════════════════════════════════════════
dt = 0.001
T_total = 50.0
T_transient = 10.0
n_steps = int(T_total / dt)
n_transient = int(T_transient / dt)
subsample = 10
dt_stored = dt * subsample

# Coupling strengths — tuned to critical regime
# K_gap=1.2 puts connectome-only model at r~0.3 (edge of sync transition)
# Extra channels should push it deeper into sync and increase MI
K_chem = 0.3      # chemical synapse phase coupling
K_gap = 1.2       # gap junction — just below critical transition
K_ephaptic = 0.5  # ephaptic field coupling
K_peptide = 0.2   # neuropeptide broadcast strength
K_proprio = 0.5   # proprioceptive feedback strength
K_drive = 1.5     # sensory drive strength

sigma_noise = 0.1
n_sensors = min(len(sensory), 50)

# Environment
tau_env = 2.0
sigma_env = 0.5

np.random.seed(42)

delta_ts = np.linspace(0.05, 4.0, 25)

# ════════════════════════════════════════════════════════════════════
# SHARED COMPONENTS
# ════════════════════════════════════════════════════════════════════

def gen_env(n_steps, dt, n_s, tau, sigma, seed=42):
    rng = np.random.RandomState(seed)
    env = np.zeros((n_steps, n_s))
    decay = np.exp(-dt / tau)
    ns = sigma * np.sqrt(2 * dt / tau)
    for t in range(1, n_steps):
        env[t] = env[t-1] * decay + ns * rng.randn(n_s)
    return env

# Fixed across all models: same environment, same frequency assignment, same ICs
sensory_input = gen_env(n_steps, dt, n_sensors, tau_env, sigma_env, seed=42)
env_stored = sensory_input[n_transient::subsample]

# Frequency hierarchy (fixed) — narrow spread for criticality
# C. elegans neurons are non-spiking graded potential neurons
# All operate at low frequencies; hierarchy is subtle
rng_freq = np.random.RandomState(99)
omegas = np.zeros(N)
for i in inter:
    omegas[i] = 2 * np.pi * (0.7 + 0.3 * rng_freq.rand())   # 0.7-1.0 Hz
for i in sensory:
    omegas[i] = 2 * np.pi * (0.9 + 0.3 * rng_freq.rand())   # 0.9-1.2 Hz
for i in motor:
    omegas[i] = 2 * np.pi * (1.0 + 0.3 * rng_freq.rand())   # 1.0-1.3 Hz

# Initial phases (fixed)
theta_init = 2 * np.pi * np.random.RandomState(77).rand(N)

# Noise stream (fixed)
noise_seed = 123

# ════════════════════════════════════════════════════════════════════
# THE THREE MODELS
# ════════════════════════════════════════════════════════════════════

def run_model(mode='connectome_only'):
    """
    Run one of three models:
    - 'connectome_only': chemical + gap junction coupling (Model A)
    - 'connectome_ephaptic': + ephaptic field coupling (Model B)
    - 'full_highD': + ephaptic + neuropeptide broadcast + proprioception (Model C)
    """
    theta = theta_init.copy()
    rng_noise = np.random.RandomState(noise_seed)

    n_stored = (n_steps - n_transient) // subsample
    states = np.zeros((n_stored, N * 2))
    order_param = np.zeros(n_stored)  # global order parameter r
    si = 0

    # Neuropeptide field (slow, global modulation) — Model C only
    peptide_field = np.zeros(N)
    tau_peptide = 1.0  # slow dynamics (1s timescale)

    for t in range(1, n_steps):
        # ── Sensory drive (same for all models) ─────────────────
        drive = np.zeros(N)
        for k, idx in enumerate(sensory[:n_sensors]):
            drive[idx] = K_drive * sensory_input[t, k]

        # ── Chemical synapse coupling (all models) ──────────────
        sin_diff = np.sin(theta[:, None] - theta[None, :])
        chem = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)

        # ── Gap junction coupling (all models) ──────────────────
        gap = K_gap * np.sum(W_gap_norm * sin_diff, axis=0)

        # ── Ephaptic field coupling (Models B and C) ────────────
        if mode in ('connectome_ephaptic', 'full_highD'):
            ephaptic = K_ephaptic * np.sum(W_ephaptic_norm * sin_diff, axis=0)
        else:
            ephaptic = 0.0

        # ── Neuropeptide broadcast (Model C only) ───────────────
        if mode == 'full_highD':
            # Global mean field with slow dynamics
            mean_activity = np.mean(np.sin(theta))
            peptide_field += (-peptide_field / tau_peptide + mean_activity) * dt
            peptide = K_peptide * peptide_field
        else:
            peptide = 0.0

        # ── Proprioceptive feedback (Model C only) ──────────────
        if mode == 'full_highD':
            # Motor output feeds back to sensory neurons
            motor_mean = np.mean(np.cos(theta[motor])) if len(motor) > 0 else 0.0
            proprio = np.zeros(N)
            for idx in sensory[:n_sensors]:
                proprio[idx] = K_proprio * np.sin(motor_mean - theta[idx])
        else:
            proprio = 0.0

        # ── Phase update ────────────────────────────────────────
        dtheta = omegas + drive + chem + gap + ephaptic + peptide + proprio
        dtheta += sigma_noise * rng_noise.randn(N) / np.sqrt(dt)
        theta += dtheta * dt

        # ── Store ───────────────────────────────────────────────
        if t >= n_transient and (t - n_transient) % subsample == 0 and si < n_stored:
            states[si, :N] = np.cos(theta)
            states[si, N:] = np.sin(theta)
            # Order parameter
            order_param[si] = np.abs(np.mean(np.exp(1j * theta)))
            si += 1

    return states[:si], np.mean(order_param[:si])

# ════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════

def gaussian_mi(X, Y):
    n = X.shape[0]
    dX, dY = X.shape[1], Y.shape[1]
    a = 0.01
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    C_XX = Xc.T@Xc/n + a*np.eye(dX)
    C_YY = Yc.T@Yc/n + a*np.eye(dY)
    XY = np.hstack([Xc,Yc])
    C_XY = XY.T@XY/n + a*np.eye(dX+dY)
    s1,l1 = np.linalg.slogdet(C_XX)
    s2,l2 = np.linalg.slogdet(C_YY)
    s3,l3 = np.linalg.slogdet(C_XY)
    if s1<=0 or s2<=0 or s3<=0: return 0.0
    return max(0.0, 0.5*(l1+l2-l3))

def mi_profile(states, env_sig, delta_ts, dt_stored, n_pcs=50):
    n_t = states.shape[0]
    X = states - states.mean(0)
    if n_pcs < X.shape[1]:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        X = X @ Vt[:n_pcs].T
    E = env_sig - env_sig.mean(0)
    if E.shape[1] > 20:
        _, _, Ve = np.linalg.svd(E, full_matrices=False)
        E = E @ Ve[:20].T
    I = np.zeros(len(delta_ts))
    for i, lag in enumerate(delta_ts):
        ls = max(1, int(lag/dt_stored))
        if ls >= n_t-50: continue
        Xn = X[:-ls]; Ef = E[ls:]
        ml = min(len(Xn),len(Ef))
        if ml > 3000:
            idx = np.random.choice(ml,3000,replace=False)
            Xn=Xn[idx]; Ef=Ef[idx]
        else:
            Xn=Xn[:ml]; Ef=Ef[:ml]
        I[i] = gaussian_mi(Xn, Ef)
    return I

def deff(states):
    X = states - states.mean(0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s**2/len(X); ev = ev[ev>1e-10]
    return (ev.sum())**2 / (ev**2).sum()

def motor_coordination(states, class_A, class_B, dt_stored):
    """Cross-correlation between two motor neuron classes at zero lag."""
    if len(class_A) == 0 or len(class_B) == 0:
        return 0.0
    a = np.mean(states[:, class_A], axis=1)
    b = np.mean(states[:, class_B], axis=1)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    rho, _ = pearsonr(a, b)
    return rho

# ════════════════════════════════════════════════════════════════════
# RUN ALL THREE MODELS
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C. elegans: Connectome vs High-D Models")
print("="*60)

models = {
    'A: Connectome only': 'connectome_only',
    'B: + Ephaptic field': 'connectome_ephaptic',
    'C: + Broadcast + Proprio': 'full_highD',
}

results = {}
for label, mode in models.items():
    print(f"\nRunning {label}...")
    states, order_r = run_model(mode)
    d = deff(states)
    print(f"  Order parameter r: {order_r:.3f}")
    mi = mi_profile(states, env_stored, delta_ts, dt_stored)
    mi_short = np.mean(mi[:5])

    # Motor coordination
    va_vb = motor_coordination(states, VA, VB, dt_stored)
    da_db = motor_coordination(states, DA, DB, dt_stored)
    vd_dd = motor_coordination(states, VD, DD, dt_stored)
    va_da = motor_coordination(states, VA, DA, dt_stored)

    results[label] = {
        'states': states, 'deff': d, 'mi_profile': mi, 'mi_short': mi_short,
        'va_vb': va_vb, 'da_db': da_db, 'vd_dd': vd_dd, 'va_da': va_da,
        'mode': mode,
    }
    print(f"  D_eff: {d:.1f}")
    print(f"  MI (short lag): {mi_short:.4f}")
    print(f"  Motor coord: VA-VB={va_vb:.3f}, DA-DB={da_db:.3f}, VA-DA={va_da:.3f}")

# ════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")
labels = list(results.keys())
short_labels = ['Connectome\nonly', '+ Ephaptic\nfield', '+ Broadcast\n+ Proprio']
colors = ['#2196F3', '#FF9800', '#EF5350']

fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 3, hspace=0.35, wspace=0.4)

# (a) MI profiles
ax = fig.add_subplot(gs[0, 0])
for label, color in zip(labels, colors):
    r = results[label]
    ax.plot(delta_ts, r['mi_profile'], color=color, linewidth=2, label=label[:15])
ax.axvline(tau_env, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel(r'Prediction lag $\Delta t$ (s)')
ax.set_ylabel(r'$I_{\mathrm{pred}}$ (nats)')
ax.set_title('(a) Future mutual information')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.2)

# (b) D_eff comparison
ax = fig.add_subplot(gs[0, 1])
deffs = [results[l]['deff'] for l in labels]
bars = ax.bar(short_labels, deffs, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, deffs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}', ha='center', fontsize=9)
ax.set_ylabel(r'$D_{\mathrm{eff}}$')
ax.set_title('(b) Effective dimensionality')
ax.grid(True, alpha=0.2, axis='y')

# (c) MI comparison
ax = fig.add_subplot(gs[0, 2])
mis = [results[l]['mi_short'] for l in labels]
bars = ax.bar(short_labels, mis, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, mis):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mis)*0.03,
            f'{val:.3f}', ha='center', fontsize=9)
ax.set_ylabel(r'$\langle I_{\mathrm{pred}} \rangle$ (nats)')
ax.set_title('(c) Predictive content')
ax.grid(True, alpha=0.2, axis='y')

# (d) Motor coordination
ax = fig.add_subplot(gs[1, 0])
coord_labels = ['VA-VB', 'DA-DB', 'VD-DD', 'VA-DA']
x = np.arange(len(coord_labels))
w = 0.25
for i, (label, color) in enumerate(zip(labels, colors)):
    r = results[label]
    vals = [r['va_vb'], r['da_db'], r['vd_dd'], r['va_da']]
    ax.bar(x + i*w, vals, w, color=color, edgecolor='black', linewidth=0.3,
           label=label[:15])
ax.set_xticks(x + w)
ax.set_xticklabels(coord_labels)
ax.set_ylabel('Cross-correlation')
ax.set_title('(d) Motor neuron coordination')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(0, color='black', linewidth=0.5)

# (e) Motor neuron time series — Model A vs C
ax = fig.add_subplot(gs[1, 1])
states_A = results[labels[0]]['states']
n_show_t = min(800, len(states_A))
t_plot = np.arange(n_show_t) * dt_stored
# Show VA class
for j in VA[:5]:
    ax.plot(t_plot, states_A[:n_show_t, j], linewidth=0.5, alpha=0.6, color='#2196F3')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\cos\theta$ (VA neurons)')
ax.set_title('(e) Motor: connectome only')
ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[1, 2])
states_C = results[labels[2]]['states']
for j in VA[:5]:
    ax.plot(t_plot[:min(n_show_t, len(states_C))],
            states_C[:n_show_t, j], linewidth=0.5, alpha=0.6, color='#EF5350')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\cos\theta$ (VA neurons)')
ax.set_title('(f) Motor: full high-D')
ax.grid(True, alpha=0.2)

plt.savefig(fig_dir / 'fig_celegans_highD.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_celegans_highD.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig_celegans_highD")

# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Connectome: {N} neurons")
print(f"Neuron positions: {has_position} with known coordinates")
print(f"Ephaptic connections (>0.1): {(W_ephaptic > 0.1).sum()}")
print()
for label in labels:
    r = results[label]
    print(f"{label}:")
    print(f"  D_eff = {r['deff']:.1f}")
    print(f"  MI = {r['mi_short']:.4f} nats")
    print(f"  Motor coord: VA-VB={r['va_vb']:.3f}, VA-DA={r['va_da']:.3f}")
print()

# Key comparisons
mi_A = results[labels[0]]['mi_short']
mi_C = results[labels[2]]['mi_short']
deff_A = results[labels[0]]['deff']
deff_C = results[labels[2]]['deff']
print(f"High-D advantage:")
print(f"  MI: {mi_C/(mi_A+1e-10):.1f}x (Model C vs A)")
print(f"  D_eff: {deff_C:.0f} vs {deff_A:.0f}")
print("\nDone!")
