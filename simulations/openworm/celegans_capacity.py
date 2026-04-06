#!/usr/bin/env python3
"""
C. elegans: Environmental capacity test.

Sweep environmental complexity (number of independent spatiotemporal
components) and measure how much each model can track.

The prediction: the graph model (connectome only) saturates at low
environmental complexity because it has limited information channels.
The field model scales further because the continuous field provides
additional coupling dimensions.

Environment: sum of K independent OU processes, each with its own
timescale and spatial structure, driving different subsets of sensory
neurons. K swept from 1 (trivial) to 50 (complex).

Two models (from celegans_highD.py critical regime):
- Graph: connectome only (chemical + gap junctions)
- Field: connectome + ephaptic proximity + neuropeptide broadcast + proprioception

Metric: total MI with the full environment vector at each complexity level.

Author: Ian Todd
"""

import numpy as np
import csv
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm', 'axes.linewidth': 0.8,
})

fig_dir = Path(__file__).parent.parent / "figures"
fig_dir.mkdir(exist_ok=True)
data_dir = Path(__file__).parent

# ── Load data (same as celegans_highD.py) ───────────────────────────
neurons_set = set()
chemical_edges = []; gap_edges = []
with open(data_dir / "herm_full_edgelist.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row['Source'].strip(); tgt = row['Target'].strip()
        w = int(row['Weight'].strip()); typ = row['Type'].strip()
        neurons_set.add(src); neurons_set.add(tgt)
        if typ == 'chemical': chemical_edges.append((src,tgt,w))
        elif typ == 'electrical': gap_edges.append((src,tgt,w))

positions = {}
with open(data_dir / "neuron_positions.csv") as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            try: positions[parts[0].strip()] = float(parts[1])
            except ValueError: continue

neurons = sorted(neurons_set); N = len(neurons)
neuron_idx = {n:i for i,n in enumerate(neurons)}

pos_array = np.zeros((N, 2))
for i, n in enumerate(neurons):
    if n in positions:
        pos_array[i, 0] = positions[n]
    else:
        pos_array[i, 0] = np.random.randn() * 0.3

W_chem = np.zeros((N,N)); W_gap = np.zeros((N,N))
for s,t,w in chemical_edges: W_chem[neuron_idx[s],neuron_idx[t]] += w
for s,t,w in gap_edges:
    i,j = neuron_idx[s],neuron_idx[t]; W_gap[i,j]+=w; W_gap[j,i]+=w
W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0,keepdims=True),1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0,keepdims=True),1)

dist_matrix = squareform(pdist(pos_array))
W_ephaptic = np.exp(-dist_matrix**2 / (2 * 0.15**2))
np.fill_diagonal(W_ephaptic, 0)
W_ephaptic_norm = W_ephaptic / np.maximum(W_ephaptic.sum(axis=1,keepdims=True),1)

sensory_pfx = ('ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB',
               'AWC','AFD','BAG','IL1','IL2','OLL','OLQ','PHA','PHB','PLM',
               'ALM','AVM','PVD','FLP')
motor_pfx = ('VA','VB','VD','DA','DB','DD','AS','RMD','RME','SMD','SMB')
sensory = [i for i,n in enumerate(neurons) if n.startswith(sensory_pfx)]
motor = [i for i,n in enumerate(neurons) if n.startswith(motor_pfx)]
inter = [i for i in range(N) if i not in sensory and i not in motor]
VA = [i for i,n in enumerate(neurons) if n.startswith('VA')]
VB = [i for i,n in enumerate(neurons) if n.startswith('VB')]

print(f"Loaded: {N} neurons, {len(sensory)} sensory, {len(motor)} motor")

# ── Parameters ──────────────────────────────────────────────────────
dt = 0.001
T_total = 40.0
T_transient = 10.0
n_steps = int(T_total / dt)
n_transient = int(T_transient / dt)
subsample = 10
dt_stored = dt * subsample

# Near-critical coupling (from the sweep that showed r~0.3)
K_chem = 0.3; K_gap = 1.2
K_ephaptic = 0.5; K_peptide = 0.2; K_proprio = 0.5
K_drive = 1.5
sigma_noise = 0.1

# Frequencies (narrow, near-critical)
rng_freq = np.random.RandomState(99)
omegas = np.zeros(N)
for i in inter: omegas[i] = 2*np.pi*(0.7 + 0.3*rng_freq.rand())
for i in sensory: omegas[i] = 2*np.pi*(0.9 + 0.3*rng_freq.rand())
for i in motor: omegas[i] = 2*np.pi*(1.0 + 0.3*rng_freq.rand())

theta_init = 2*np.pi*np.random.RandomState(77).rand(N)
noise_seed = 123

# Environmental complexity levels
env_complexities = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50]

# ── Multi-scale environment generator ───────────────────────────────
def gen_complex_env(n_steps, dt, n_components, n_sensors, seed=42):
    """
    Generate n_components independent OU processes with different timescales.
    Each component has a random timescale (0.5-5s) and drives a random
    subset of sensory neurons. Returns the full environment matrix and
    the sensory drive matrix.
    """
    rng = np.random.RandomState(seed)

    # Each component: random timescale, random spatial pattern
    tau_components = 0.5 + 4.5 * rng.rand(n_components)  # 0.5-5s
    sigma_components = 0.3 + 0.4 * rng.rand(n_components)

    # Environment signals
    env = np.zeros((n_steps, n_components))
    for k in range(n_components):
        decay = np.exp(-dt / tau_components[k])
        ns = sigma_components[k] * np.sqrt(2 * dt / tau_components[k])
        for t in range(1, n_steps):
            env[t, k] = env[t-1, k] * decay + ns * rng.randn()

    # Mixing matrix: each component drives a random subset of sensors
    # with random weights
    mix = np.zeros((n_components, n_sensors))
    for k in range(n_components):
        # Each component drives ~30% of sensors
        active = rng.rand(n_sensors) < 0.3
        mix[k, active] = rng.randn(active.sum()) * 0.5

    # Sensory drive = env @ mix → (n_steps, n_sensors)
    sensory_drive = env @ mix

    return env, sensory_drive

# ── Model runner ────────────────────────────────────────────────────
def run_model(sensory_drive, mode='graph'):
    theta = theta_init.copy()
    rng = np.random.RandomState(noise_seed)
    n_stored = (n_steps - n_transient) // subsample
    states = np.zeros((n_stored, N*2))
    si = 0
    n_sensors = min(sensory_drive.shape[1], len(sensory))
    peptide_field = np.zeros(N)

    for t in range(1, n_steps):
        drive = np.zeros(N)
        for k in range(n_sensors):
            drive[sensory[k]] = K_drive * sensory_drive[t, k]

        sin_diff = np.sin(theta[:,None] - theta[None,:])
        chem = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)
        gap = K_gap * np.sum(W_gap_norm * sin_diff, axis=0)

        if mode == 'field':
            ephaptic = K_ephaptic * np.sum(W_ephaptic_norm * sin_diff, axis=0)
            mean_act = np.mean(np.sin(theta))
            peptide_field += (-peptide_field / 1.0 + mean_act) * dt
            peptide = K_peptide * peptide_field
            motor_mean = np.mean(np.cos(theta[motor])) if len(motor) > 0 else 0.0
            proprio = np.zeros(N)
            for idx in sensory[:n_sensors]:
                proprio[idx] = K_proprio * np.sin(motor_mean - theta[idx])
        else:
            ephaptic = 0.0; peptide = 0.0; proprio = 0.0

        dtheta = omegas + drive + chem + gap + ephaptic + peptide + proprio
        dtheta += sigma_noise * rng.randn(N) / np.sqrt(dt)
        theta += dtheta * dt

        if t >= n_transient and (t - n_transient) % subsample == 0 and si < n_stored:
            states[si, :N] = np.cos(theta)
            states[si, N:] = np.sin(theta)
            si += 1

    return states[:si]

# ── MI computation ──────────────────────────────────────────────────
def gaussian_mi(X, Y):
    n = X.shape[0]; dX = X.shape[1]; dY = Y.shape[1]
    a = 0.01
    Xc = X-X.mean(0); Yc = Y-Y.mean(0)
    C_XX = Xc.T@Xc/n + a*np.eye(dX)
    C_YY = Yc.T@Yc/n + a*np.eye(dY)
    XY = np.hstack([Xc,Yc]); C_XY = XY.T@XY/n + a*np.eye(dX+dY)
    s1,l1 = np.linalg.slogdet(C_XX)
    s2,l2 = np.linalg.slogdet(C_YY)
    s3,l3 = np.linalg.slogdet(C_XY)
    if s1<=0 or s2<=0 or s3<=0: return 0.0
    return max(0.0, 0.5*(l1+l2-l3))

def compute_total_mi(states, env_full, dt_stored, lag=0.5):
    """Total MI between internal state and full environment at fixed lag."""
    n_t = states.shape[0]
    lag_steps = max(1, int(lag / dt_stored))
    if lag_steps >= n_t - 50:
        return 0.0

    # PCA on internal state
    X = states - states.mean(0)
    n_pcs = min(50, X.shape[1])
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    X = X @ Vt[:n_pcs].T

    # Environment (already low-D, just center)
    E = env_full - env_full.mean(0)

    Xn = X[:-lag_steps]; Ef = E[lag_steps:]
    ml = min(len(Xn), len(Ef))
    if ml > 4000:
        idx = np.random.choice(ml, 4000, replace=False)
        Xn = Xn[idx]; Ef = Ef[idx]
    else:
        Xn = Xn[:ml]; Ef = Ef[:ml]

    return gaussian_mi(Xn, Ef)

def motor_coord(states, cA, cB):
    if len(cA)==0 or len(cB)==0: return 0.0
    a = np.mean(states[:,cA], axis=1)
    b = np.mean(states[:,cB], axis=1)
    if np.std(a)<1e-10 or np.std(b)<1e-10: return 0.0
    from scipy.stats import pearsonr
    return pearsonr(a,b)[0]

# ════════════════════════════════════════════════════════════════════
# RUN COMPLEXITY SWEEP
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C. elegans: Environmental Capacity Test")
print("="*60)
print(f"Complexity levels: {env_complexities}")

n_sensors = min(len(sensory), 50)
mi_graph = np.zeros(len(env_complexities))
mi_field = np.zeros(len(env_complexities))
coord_graph = np.zeros(len(env_complexities))
coord_field = np.zeros(len(env_complexities))

for ci, n_comp in enumerate(env_complexities):
    print(f"\n  Complexity = {n_comp} components")

    # Generate environment
    env_full, sensory_drive = gen_complex_env(n_steps, dt, n_comp, n_sensors, seed=42+ci)
    env_stored = env_full[n_transient::subsample]

    # Graph model
    print(f"    Graph model...")
    states_g = run_model(sensory_drive, mode='graph')
    mi_graph[ci] = compute_total_mi(states_g, env_stored, dt_stored)
    coord_graph[ci] = motor_coord(states_g, VA, VB)

    # Field model
    print(f"    Field model...")
    states_f = run_model(sensory_drive, mode='field')
    mi_field[ci] = compute_total_mi(states_f, env_stored, dt_stored)
    coord_field[ci] = motor_coord(states_f, VA, VB)

    print(f"    MI: graph={mi_graph[ci]:.3f}, field={mi_field[ci]:.3f} "
          f"(ratio={mi_field[ci]/(mi_graph[ci]+1e-10):.2f}x)")
    print(f"    Coord VA-VB: graph={coord_graph[ci]:.3f}, field={coord_field[ci]:.3f}")

# ════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# (a) MI vs environmental complexity
ax1.plot(env_complexities, mi_graph, 'b-o', markersize=6, linewidth=2,
         label='Graph (connectome only)')
ax1.plot(env_complexities, mi_field, 'r-s', markersize=6, linewidth=2,
         label='Field (connectome + field)')
ax1.set_xlabel('Environmental complexity (independent components)')
ax1.set_ylabel(r'$I(X_{\mathrm{int}}; Y_{\mathrm{env}})$ (nats)')
ax1.set_title('(a) Information capacity vs environmental complexity')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)

# (b) Motor coordination vs environmental complexity
ax2.plot(env_complexities, coord_graph, 'b-o', markersize=6, linewidth=2,
         label='Graph (connectome only)')
ax2.plot(env_complexities, coord_field, 'r-s', markersize=6, linewidth=2,
         label='Field (connectome + field)')
ax2.set_xlabel('Environmental complexity (independent components)')
ax2.set_ylabel('VA-VB motor coordination')
ax2.set_title('(b) Motor coordination vs environmental complexity')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(fig_dir / 'fig_celegans_capacity.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_celegans_capacity.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig_celegans_capacity")

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Summary")
print("="*60)
for ci, nc in enumerate(env_complexities):
    ratio = mi_field[ci] / (mi_graph[ci] + 1e-10)
    print(f"  K={nc:2d}: MI graph={mi_graph[ci]:.3f} field={mi_field[ci]:.3f} "
          f"({ratio:.2f}x)  coord graph={coord_graph[ci]:.3f} field={coord_field[ci]:.3f}")

# Does graph saturate while field continues to scale?
print(f"\nGraph MI at K=1: {mi_graph[0]:.3f}, at K=50: {mi_graph[-1]:.3f} "
      f"(growth: {mi_graph[-1]/mi_graph[0]:.1f}x)")
print(f"Field MI at K=1: {mi_field[0]:.3f}, at K=50: {mi_field[-1]:.3f} "
      f"(growth: {mi_field[-1]/mi_field[0]:.1f}x)")
print("\nDone!")
