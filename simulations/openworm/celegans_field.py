#!/usr/bin/env python3
"""
C. elegans as a bioelectric FIELD, not a graph.

The worm's body is modelled as a 1D continuum (body axis, 0 to 1).
448 neurons are embedded at their known positions along this axis.
Each neuron is a phase oscillator coupled to its neighbours through:

1. The CONNECTOME (chemical synapses + gap junctions) — sparse, directed/bidirectional
2. The BIOELECTRIC FIELD — a continuous 1D diffusion field along the body axis
   that every neuron both contributes to and is influenced by

The bioelectric field is the key: it provides continuous, spatially
graded coupling between ALL neurons based on proximity, not just
those with direct synaptic connections. It's the continuous coupling
channel that the connectome projection misses.

Two simulations:
- GRAPH model: connectome only, no field
- FIELD model: connectome + bioelectric field

The field model should show:
- Higher predictive content (more coupling channels = more correlations)
- More coordinated motor output (field enables long-range coordination)
- Higher effective dimensionality at the critical regime

The field is solved with a simple 1D diffusion PDE:
  ∂u/∂t = D ∂²u/∂x² - u/τ_field + Σ_i δ(x - x_i) * sin(θ_i)

where u(x,t) is the field value, D is the diffusion constant,
τ_field is the decay time, and neurons act as point sources.

Author: Ian Todd
"""

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

# ════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════
print("Loading data...")

# Connectome
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

# Positions (use x-coordinate as body axis position, normalise to [0,1])
positions = {}
with open(data_dir / "neuron_positions.csv") as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            try:
                positions[parts[0].strip()] = float(parts[1])
            except ValueError:
                continue

neurons = sorted(neurons_set)
N = len(neurons)
neuron_idx = {n: i for i, n in enumerate(neurons)}

# Neuron body-axis positions (normalised 0 to 1)
body_pos = np.zeros(N)
pos_min = min(positions.values()) if positions else 0
pos_max = max(positions.values()) if positions else 1
pos_range = pos_max - pos_min if pos_max > pos_min else 1
for i, n in enumerate(neurons):
    if n in positions:
        body_pos[i] = (positions[n] - pos_min) / pos_range
    else:
        body_pos[i] = np.random.rand()  # random position for unknowns

# Adjacency matrices
W_chem = np.zeros((N, N)); W_gap = np.zeros((N, N))
for s,t,w in chemical_edges: W_chem[neuron_idx[s],neuron_idx[t]] += w
for s,t,w in gap_edges:
    i,j = neuron_idx[s],neuron_idx[t]; W_gap[i,j]+=w; W_gap[j,i]+=w
W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

# Neuron classes
sensory = [i for i,n in enumerate(neurons) if n.startswith((
    'ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC',
    'AFD','BAG','IL1','IL2','OLL','OLQ','PHA','PHB','PLM','ALM','AVM'))]
motor = [i for i,n in enumerate(neurons) if n.startswith((
    'VA','VB','VD','DA','DB','DD','AS','RMD','RME','SMD','SMB'))]
inter = [i for i in range(N) if i not in sensory and i not in motor]
VA = [i for i,n in enumerate(neurons) if n.startswith('VA')]
VB = [i for i,n in enumerate(neurons) if n.startswith('VB')]
DA = [i for i,n in enumerate(neurons) if n.startswith('DA')]
DB = [i for i,n in enumerate(neurons) if n.startswith('DB')]

print(f"  {N} neurons, {len(chemical_edges)} chem, {len(gap_edges)} gap")
print(f"  Body positions: {(body_pos > 0).sum()} mapped")

# ════════════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════════════
dt = 0.0005        # 0.5ms timesteps (fine-grained)
T_total = 60.0     # 60 seconds
T_transient = 15.0
n_steps = int(T_total / dt)
n_transient = int(T_transient / dt)
subsample = 20
dt_stored = dt * subsample

# Field grid
N_grid = 200       # spatial resolution of bioelectric field
dx = 1.0 / N_grid
field_x = np.linspace(0, 1, N_grid)

# Field parameters
D_field = 0.02     # diffusion constant (body units²/s)
tau_field = 0.5    # field decay time (s)
K_field = 1.0      # neuron→field source strength
K_field_to_neuron = 0.8  # field→neuron coupling

# Connectome coupling (tuned near critical)
K_chem = 0.3
K_gap = 1.0

# Other
K_drive = 1.5
sigma_noise = 0.1
n_sensors = min(len(sensory), 50)
tau_env = 2.0; sigma_env = 0.5

np.random.seed(42)
delta_ts = np.linspace(0.05, 5.0, 30)

# Precompute: which grid cell is each neuron nearest to?
neuron_grid_idx = np.clip((body_pos * N_grid).astype(int), 0, N_grid - 1)

# Shared: frequencies, ICs, environment, noise
rng_freq = np.random.RandomState(99)
omegas = 2 * np.pi * (0.8 + 0.4 * rng_freq.rand(N))  # 0.8-1.2 Hz
theta_init = 2 * np.pi * np.random.RandomState(77).rand(N)
noise_seed = 123

def gen_env(n_steps, dt, n_s, tau, sigma, seed=42):
    rng = np.random.RandomState(seed)
    env = np.zeros((n_steps, n_s))
    decay = np.exp(-dt / tau)
    ns = sigma * np.sqrt(2 * dt / tau)
    for t in range(1, n_steps):
        env[t] = env[t-1] * decay + ns * rng.randn(n_s)
    return env

sensory_input = gen_env(n_steps, dt, n_sensors, tau_env, sigma_env)
env_stored = sensory_input[n_transient::subsample]

# ════════════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════════════

def run_graph_model():
    """Connectome only — no bioelectric field."""
    theta = theta_init.copy()
    rng = np.random.RandomState(noise_seed)
    n_stored = (n_steps - n_transient) // subsample
    states = np.zeros((n_stored, N * 2))
    r_samples = []
    si = 0

    for t in range(1, n_steps):
        drive = np.zeros(N)
        for k, idx in enumerate(sensory[:n_sensors]):
            drive[idx] = K_drive * sensory_input[t, k]

        sin_diff = np.sin(theta[:, None] - theta[None, :])
        chem = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)
        gap = K_gap * np.sum(W_gap_norm * sin_diff, axis=0)

        dtheta = omegas + drive + chem + gap
        dtheta += sigma_noise * rng.randn(N) / np.sqrt(dt)
        theta += dtheta * dt

        if t >= n_transient and (t - n_transient) % subsample == 0 and si < n_stored:
            states[si, :N] = np.cos(theta)
            states[si, N:] = np.sin(theta)
            r_samples.append(np.abs(np.mean(np.exp(1j * theta))))
            si += 1

    return states[:si], np.mean(r_samples)

def run_field_model():
    """Connectome + continuous bioelectric field along body axis."""
    theta = theta_init.copy()
    rng = np.random.RandomState(noise_seed)
    n_stored = (n_steps - n_transient) // subsample
    states = np.zeros((n_stored, N * 2))
    field_snapshots = np.zeros((n_stored, N_grid))
    r_samples = []
    si = 0

    # Bioelectric field (1D)
    u = np.zeros(N_grid)

    # Diffusion coefficient for stability: D * dt / dx² < 0.5
    diff_coeff = D_field * dt / dx**2
    if diff_coeff > 0.4:
        print(f"  WARNING: diffusion CFL = {diff_coeff:.3f} (should be < 0.5)")

    for t in range(1, n_steps):
        # ── Update field: diffusion + decay + neuron sources ────
        # Laplacian (periodic boundary)
        laplacian = (np.roll(u, 1) + np.roll(u, -1) - 2 * u) / dx**2

        # Neuron sources: each neuron injects sin(θ) at its grid position
        sources = np.zeros(N_grid)
        for i in range(N):
            sources[neuron_grid_idx[i]] += K_field * np.sin(theta[i])

        u += (D_field * laplacian - u / tau_field + sources) * dt

        # ── Neuron dynamics ─────────────────────────────────────
        drive = np.zeros(N)
        for k, idx in enumerate(sensory[:n_sensors]):
            drive[idx] = K_drive * sensory_input[t, k]

        sin_diff = np.sin(theta[:, None] - theta[None, :])
        chem = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)
        gap = K_gap * np.sum(W_gap_norm * sin_diff, axis=0)

        # Field → neuron coupling: each neuron reads the field at its position
        field_at_neuron = u[neuron_grid_idx]
        field_coupling = K_field_to_neuron * np.sin(field_at_neuron - theta)

        dtheta = omegas + drive + chem + gap + field_coupling
        dtheta += sigma_noise * rng.randn(N) / np.sqrt(dt)
        theta += dtheta * dt

        # ── Store ───────────────────────────────────────────────
        if t >= n_transient and (t - n_transient) % subsample == 0 and si < n_stored:
            states[si, :N] = np.cos(theta)
            states[si, N:] = np.sin(theta)
            field_snapshots[si] = u.copy()
            r_samples.append(np.abs(np.mean(np.exp(1j * theta))))
            si += 1

    return states[:si], np.mean(r_samples), field_snapshots[:si]

# ════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════

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
        if ml > 4000:
            idx = np.random.choice(ml,4000,replace=False)
            Xn=Xn[idx]; Ef=Ef[idx]
        else: Xn=Xn[:ml]; Ef=Ef[:ml]
        I[i] = gaussian_mi(Xn, Ef)
    return I

def deff(states):
    X = states - states.mean(0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s**2/len(X); ev = ev[ev>1e-10]
    return (ev.sum())**2 / (ev**2).sum()

def motor_coord(states, classA, classB):
    if len(classA)==0 or len(classB)==0: return 0.0
    a = np.mean(states[:,classA], axis=1)
    b = np.mean(states[:,classB], axis=1)
    if np.std(a)<1e-10 or np.std(b)<1e-10: return 0.0
    return pearsonr(a,b)[0]

# ════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C. elegans: Graph vs Field")
print("="*60)

print("\nRunning GRAPH model (connectome only)...")
states_graph, r_graph = run_graph_model()
d_graph = deff(states_graph)
mi_graph = mi_profile(states_graph, env_stored, delta_ts, dt_stored)
mi_short_graph = np.mean(mi_graph[:5])
coord_graph = {
    'VA-VB': motor_coord(states_graph, VA, VB),
    'DA-DB': motor_coord(states_graph, DA, DB),
    'VA-DA': motor_coord(states_graph, VA, DA),
}
print(f"  r = {r_graph:.3f}, D_eff = {d_graph:.1f}, MI = {mi_short_graph:.4f}")
print(f"  Motor: {coord_graph}")

print("\nRunning FIELD model (connectome + bioelectric field)...")
states_field, r_field, field_snaps = run_field_model()
d_field = deff(states_field)
mi_field = mi_profile(states_field, env_stored, delta_ts, dt_stored)
mi_short_field = np.mean(mi_field[:5])
coord_field = {
    'VA-VB': motor_coord(states_field, VA, VB),
    'DA-DB': motor_coord(states_field, DA, DB),
    'VA-DA': motor_coord(states_field, VA, DA),
}
print(f"  r = {r_field:.3f}, D_eff = {d_field:.1f}, MI = {mi_short_field:.4f}")
print(f"  Motor: {coord_field}")

print(f"\nField advantage: MI = {mi_short_field/(mi_short_graph+1e-10):.2f}x, "
      f"D_eff = {d_field:.0f} vs {d_graph:.0f}")

# ════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 3, hspace=0.35, wspace=0.4)

# (a) MI profiles
ax = fig.add_subplot(gs[0, 0])
ax.plot(delta_ts, mi_field, 'r-', linewidth=2, label='Field model')
ax.plot(delta_ts, mi_graph, 'b--', linewidth=2, label='Graph model')
ax.axvline(tau_env, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel(r'Prediction lag $\Delta t$ (s)')
ax.set_ylabel(r'$I_{\mathrm{pred}}$ (nats)')
ax.set_title('(a) Future mutual information')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# (b) D_eff + MI bars
ax = fig.add_subplot(gs[0, 1])
x = np.arange(2); w = 0.35
bars = ax.bar(x, [mi_short_graph, mi_short_field], 0.6,
              color=['#2196F3', '#EF5350'], edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, [mi_short_graph, mi_short_field]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(mi_short_graph,mi_short_field)*0.03,
            f'{val:.3f}', ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(['Graph\n(connectome)', 'Field\n(connectome + field)'])
ax.set_ylabel(r'$\langle I_{\mathrm{pred}} \rangle$ (nats)')
ax.set_title('(b) Predictive content')
ax.grid(True, alpha=0.2, axis='y')

# (c) Motor coordination
ax = fig.add_subplot(gs[0, 2])
coord_keys = ['VA-VB', 'DA-DB', 'VA-DA']
x = np.arange(len(coord_keys)); w = 0.3
ax.bar(x-w/2, [coord_graph[k] for k in coord_keys], w,
       color='#2196F3', edgecolor='black', linewidth=0.3, label='Graph')
ax.bar(x+w/2, [coord_field[k] for k in coord_keys], w,
       color='#EF5350', edgecolor='black', linewidth=0.3, label='Field')
ax.set_xticks(x); ax.set_xticklabels(coord_keys)
ax.set_ylabel('Cross-correlation')
ax.set_title('(c) Motor coordination')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(0, color='black', linewidth=0.5)

# (d) Bioelectric field kymograph
ax = fig.add_subplot(gs[1, 0])
# Subsample field snapshots for plotting
n_field_show = min(500, len(field_snaps))
field_show = field_snaps[:n_field_show]
t_field = np.arange(n_field_show) * dt_stored
im = ax.pcolormesh(field_x, t_field, field_show, cmap='RdBu_r', shading='auto',
                    rasterized=True)
ax.set_xlabel('Body position')
ax.set_ylabel('Time (s)')
ax.set_title('(d) Bioelectric field kymograph')
fig.colorbar(im, ax=ax, label='Field u(x,t)', fraction=0.046, pad=0.04)
# Mark neuron positions
for i in range(0, N, 20):
    ax.axvline(body_pos[i], color='white', alpha=0.1, linewidth=0.3)

# (e) Motor output — graph
ax = fig.add_subplot(gs[1, 1])
n_show_t = min(600, len(states_graph))
t_plot = np.arange(n_show_t) * dt_stored
for j in VA[:5]:
    ax.plot(t_plot, states_graph[:n_show_t, j], linewidth=0.5, alpha=0.6, color='#2196F3')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\cos\theta$ (VA neurons)')
ax.set_title('(e) Motor: graph model')
ax.grid(True, alpha=0.2)

# (f) Motor output — field
ax = fig.add_subplot(gs[1, 2])
for j in VA[:5]:
    ax.plot(t_plot[:min(n_show_t, len(states_field))],
            states_field[:n_show_t, j], linewidth=0.5, alpha=0.6, color='#EF5350')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\cos\theta$ (VA neurons)')
ax.set_title('(f) Motor: field model')
ax.grid(True, alpha=0.2)

plt.savefig(fig_dir / 'fig_celegans_field.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_celegans_field.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig_celegans_field")

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Graph model: r={r_graph:.3f}, D_eff={d_graph:.1f}, MI={mi_short_graph:.4f}")
print(f"Field model: r={r_field:.3f}, D_eff={d_field:.1f}, MI={mi_short_field:.4f}")
print(f"Field advantage: {mi_short_field/(mi_short_graph+1e-10):.2f}x MI")
print(f"Grid resolution: {N_grid} points, dt={dt*1000:.1f}ms, T={T_total}s")
print("\nDone!")
