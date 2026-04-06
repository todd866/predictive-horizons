#!/usr/bin/env python3
"""
C. elegans: Motor coordination requires field coupling, not just the connectome.

Definitive simulation for the paper. Three models × three coupling regimes ×
multiple seeds. The headline result: MI is similar across models, but motor
coordination diverges dramatically — and only at criticality.

Models:
  A — Connectome only (chemical synapses + gap junctions)
  B — + Ephaptic field (distance-dependent coupling)
  C — + Ephaptic + neuropeptide broadcast + proprioceptive feedback

Coupling regimes:
  Subcritical  — K_gap = 0.4 (desynchronized, r < 0.1)
  Critical     — K_gap = 1.2 (edge of sync transition, r ~ 0.2-0.3)
  Supercritical — K_gap = 3.0 (over-synchronized, r > 0.6)

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
print("Loading connectome data...")

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

# Position array (random for unknown positions, deterministic seed)
rng_pos = np.random.RandomState(0)
pos_array = np.zeros((N, 2))
has_position = 0
for i, n in enumerate(neurons):
    if n in positions:
        pos_array[i] = positions[n]
        has_position += 1
    else:
        pos_array[i] = rng_pos.randn(2) * 0.3

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

# Ephaptic coupling matrix
dist_matrix = squareform(pdist(pos_array))
ephaptic_sigma = 0.15
W_ephaptic = np.exp(-dist_matrix**2 / (2 * ephaptic_sigma**2))
np.fill_diagonal(W_ephaptic, 0)
W_ephaptic_norm = W_ephaptic / np.maximum(W_ephaptic.sum(axis=1, keepdims=True), 1)

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
VD = [i for i, n in enumerate(neurons) if n.startswith('VD')]
DD = [i for i, n in enumerate(neurons) if n.startswith('DD')]

print(f"  {N} neurons ({has_position} with positions)")
print(f"  {len(chemical_edges)} chemical, {len(gap_edges)} gap junctions")
print(f"  Sensory: {len(sensory)}, Inter: {len(inter)}, Motor: {len(motor)}")

# ════════════════════════════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════════════════════════════

dt = 0.001
T_total = 50.0
T_transient = 10.0
n_steps = int(T_total / dt)
n_transient = int(T_transient / dt)
subsample = 10
dt_stored = dt * subsample

# Fixed coupling strengths (except K_gap which sweeps)
K_chem = 0.3
K_ephaptic = 0.5
K_peptide = 0.2
K_proprio = 0.5
K_drive = 1.5
sigma_noise = 0.1
n_sensors = min(len(sensory), 50)
tau_env = 2.0
sigma_env = 0.5
delta_ts = np.linspace(0.05, 4.0, 25)

# Coupling regimes
regimes = {
    'Subcritical': 0.4,
    'Critical': 1.2,
    'Supercritical': 3.0,
}

N_SEEDS = 5


def gen_env(n_steps, dt, n_s, tau, sigma, seed):
    rng = np.random.RandomState(seed)
    env = np.zeros((n_steps, n_s))
    decay = np.exp(-dt / tau)
    ns = sigma * np.sqrt(2 * dt / tau)
    for t in range(1, n_steps):
        env[t] = env[t-1] * decay + ns * rng.randn(n_s)
    return env


def make_freq_assignment(seed):
    """Fixed frequency hierarchy per seed."""
    rng = np.random.RandomState(seed)
    omegas = np.zeros(N)
    for i in inter:
        omegas[i] = 2 * np.pi * (0.7 + 0.3 * rng.rand())
    for i in sensory:
        omegas[i] = 2 * np.pi * (0.9 + 0.3 * rng.rand())
    for i in motor:
        omegas[i] = 2 * np.pi * (1.0 + 0.3 * rng.rand())
    return omegas


def run_model(mode, K_gap, seed):
    """Run one model at one coupling strength with one seed."""
    # Deterministic components from seed
    omegas = make_freq_assignment(seed * 100 + 99)
    theta_init = 2 * np.pi * np.random.RandomState(seed * 100 + 77).rand(N)
    sensory_input = gen_env(n_steps, dt, n_sensors, tau_env, sigma_env, seed * 100 + 42)

    theta = theta_init.copy()
    rng_noise = np.random.RandomState(seed * 100 + 123)

    n_stored = (n_steps - n_transient) // subsample
    states = np.zeros((n_stored, N * 2))
    env_stored = sensory_input[n_transient::subsample]
    order_param = np.zeros(n_stored)
    si = 0

    peptide_field = np.zeros(N)
    tau_peptide = 1.0

    for t in range(1, n_steps):
        drive = np.zeros(N)
        for k, idx in enumerate(sensory[:n_sensors]):
            drive[idx] = K_drive * sensory_input[t, k]

        sin_diff = np.sin(theta[:, None] - theta[None, :])
        chem = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)
        gap = K_gap * np.sum(W_gap_norm * sin_diff, axis=0)

        if mode in ('ephaptic', 'full'):
            ephaptic = K_ephaptic * np.sum(W_ephaptic_norm * sin_diff, axis=0)
        else:
            ephaptic = 0.0

        if mode == 'full':
            mean_activity = np.mean(np.sin(theta))
            peptide_field += (-peptide_field / tau_peptide + mean_activity) * dt
            peptide = K_peptide * peptide_field
            motor_mean = np.mean(np.cos(theta[motor])) if len(motor) > 0 else 0.0
            proprio = np.zeros(N)
            for idx in sensory[:n_sensors]:
                proprio[idx] = K_proprio * np.sin(motor_mean - theta[idx])
        else:
            peptide = 0.0
            proprio = 0.0

        dtheta = omegas + drive + chem + gap + ephaptic + peptide + proprio
        dtheta += sigma_noise * rng_noise.randn(N) / np.sqrt(dt)
        theta += dtheta * dt

        if t >= n_transient and (t - n_transient) % subsample == 0 and si < n_stored:
            states[si, :N] = np.cos(theta)
            states[si, N:] = np.sin(theta)
            order_param[si] = np.abs(np.mean(np.exp(1j * theta)))
            si += 1

    return states[:si], env_stored[:si], np.mean(order_param[:si])


def gaussian_mi(X, Y):
    n = X.shape[0]
    dX, dY = X.shape[1], Y.shape[1]
    a = 0.01
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    C_XX = Xc.T@Xc/n + a*np.eye(dX)
    C_YY = Yc.T@Yc/n + a*np.eye(dY)
    XY = np.hstack([Xc, Yc])
    C_XY = XY.T@XY/n + a*np.eye(dX+dY)
    s1, l1 = np.linalg.slogdet(C_XX)
    s2, l2 = np.linalg.slogdet(C_YY)
    s3, l3 = np.linalg.slogdet(C_XY)
    if s1 <= 0 or s2 <= 0 or s3 <= 0:
        return 0.0
    return max(0.0, 0.5*(l1+l2-l3))


def compute_mi_short(states, env_sig, n_pcs=50):
    """Short-lag MI (mean over first 5 lags)."""
    n_t = states.shape[0]
    X = states - states.mean(0)
    if n_pcs < X.shape[1]:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        X = X @ Vt[:n_pcs].T
    E = env_sig - env_sig.mean(0)
    if E.shape[1] > 20:
        _, _, Ve = np.linalg.svd(E, full_matrices=False)
        E = E @ Ve[:20].T

    mis = []
    for lag in delta_ts[:5]:
        ls = max(1, int(lag / dt_stored))
        if ls >= n_t - 50:
            continue
        Xn = X[:-ls]; Ef = E[ls:]
        ml = min(len(Xn), len(Ef))
        if ml > 3000:
            idx = np.random.choice(ml, 3000, replace=False)
            Xn = Xn[idx]; Ef = Ef[idx]
        else:
            Xn = Xn[:ml]; Ef = Ef[:ml]
        mis.append(gaussian_mi(Xn, Ef))
    return np.mean(mis) if mis else 0.0


def motor_coordination(states, class_A, class_B):
    if len(class_A) == 0 or len(class_B) == 0:
        return 0.0
    a = np.mean(states[:, class_A], axis=1)
    b = np.mean(states[:, class_B], axis=1)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    rho, _ = pearsonr(a, b)
    return rho


def deff(states):
    X = states - states.mean(0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s**2/len(X); ev = ev[ev > 1e-10]
    return (ev.sum())**2 / (ev**2).sum()


# ════════════════════════════════════════════════════════════════════
# RUN ALL CONDITIONS
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C. elegans: Motor Coordination Requires Field Coupling")
print("="*60)

model_modes = {
    'A: Connectome only': 'connectome',
    'B: + Ephaptic': 'ephaptic',
    'C: + Full field': 'full',
}
model_keys = list(model_modes.keys())

# Results storage: results[regime][model] = {mi: [...], va_vb: [...], ...}
all_results = {}

for regime_name, K_gap in regimes.items():
    print(f"\n{'='*40}")
    print(f"Regime: {regime_name} (K_gap = {K_gap})")
    print(f"{'='*40}")
    all_results[regime_name] = {}

    for model_name, mode in model_modes.items():
        mis, va_vbs, va_das, da_dbs, deffs, rs = [], [], [], [], [], []
        states_last = None
        env_last = None

        for seed in range(N_SEEDS):
            print(f"  {model_name}, seed {seed}...", end=" ", flush=True)
            states, env_s, r = run_model(mode, K_gap, seed)

            mi = compute_mi_short(states, env_s)
            va_vb = motor_coordination(states, VA, VB)
            va_da = motor_coordination(states, VA, DA)
            da_db = motor_coordination(states, DA, DB)
            d = deff(states)

            mis.append(mi)
            va_vbs.append(va_vb)
            va_das.append(va_da)
            da_dbs.append(da_db)
            deffs.append(d)
            rs.append(r)
            states_last = states
            env_last = env_s
            print(f"r={r:.2f}, MI={mi:.3f}, VA-VB={va_vb:.3f}")

        all_results[regime_name][model_name] = {
            'mi': np.array(mis),
            'va_vb': np.array(va_vbs),
            'va_da': np.array(va_das),
            'da_db': np.array(da_dbs),
            'deff': np.array(deffs),
            'r': np.array(rs),
            'states_last': states_last,
            'env_last': env_last,
        }

# ════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for regime_name in regimes:
    print(f"\n--- {regime_name} ---")
    for model_name in model_keys:
        r = all_results[regime_name][model_name]
        print(f"  {model_name}:")
        print(f"    r = {r['r'].mean():.3f} ± {r['r'].std():.3f}")
        print(f"    MI = {r['mi'].mean():.4f} ± {r['mi'].std():.4f}")
        print(f"    VA-VB = {r['va_vb'].mean():.3f} ± {r['va_vb'].std():.3f}")
        print(f"    VA-DA = {r['va_da'].mean():.3f} ± {r['va_da'].std():.3f}")
        print(f"    D_eff = {r['deff'].mean():.1f} ± {r['deff'].std():.1f}")

# ════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

colors = ['#2196F3', '#FF9800', '#EF5350']
short_labels = ['Connectome\nonly', '+ Ephaptic\nfield', '+ Full\nfield']
regime_list = list(regimes.keys())

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, hspace=0.38, wspace=0.38)

# ── (a) Motor coordination at criticality (THE headline) ──────────
ax = fig.add_subplot(gs[0, 0:2])
coord_pairs = [('VA-VB', 'va_vb'), ('VA-DA', 'va_da'), ('DA-DB', 'da_db')]
x = np.arange(len(coord_pairs))
w = 0.25

crit = all_results['Critical']
for i, (model_name, color, slabel) in enumerate(zip(model_keys, colors, short_labels)):
    r = crit[model_name]
    means = [r[metric].mean() for _, metric in coord_pairs]
    sems = [r[metric].std() / np.sqrt(N_SEEDS) for _, metric in coord_pairs]
    bars = ax.bar(x + i*w, means, w, yerr=sems, color=color, edgecolor='black',
                  linewidth=0.3, label=slabel.replace('\n', ' '), capsize=3)

ax.set_xticks(x + w)
ax.set_xticklabels([p[0] for p in coord_pairs])
ax.set_ylabel('Cross-correlation')
ax.set_title('(a) Motor neuron coordination at criticality', fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(0, color='black', linewidth=0.5)

# ── (b) MI comparison at criticality ──────────────────────────────
ax = fig.add_subplot(gs[0, 2])
mis_mean = [crit[m]['mi'].mean() for m in model_keys]
mis_sem = [crit[m]['mi'].std() / np.sqrt(N_SEEDS) for m in model_keys]
bars = ax.bar(short_labels, mis_mean, yerr=mis_sem, color=colors,
              edgecolor='black', linewidth=0.5, capsize=3)
for bar, val in zip(bars, mis_mean):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mis_mean)*0.05,
            f'{val:.3f}', ha='center', fontsize=9)
ax.set_ylabel(r'$\langle I_{\mathrm{pred}} \rangle$ (nats)')
ax.set_title('(b) Predictive content (similar)')
ax.grid(True, alpha=0.2, axis='y')

# ── (c) Criticality dependence: VA-VB across regimes ─────────────
ax = fig.add_subplot(gs[1, 0])
x_reg = np.arange(len(regime_list))

for i, (model_name, color, slabel) in enumerate(zip(model_keys, colors, short_labels)):
    means = [all_results[reg][model_name]['va_vb'].mean() for reg in regime_list]
    sems = [all_results[reg][model_name]['va_vb'].std() / np.sqrt(N_SEEDS) for reg in regime_list]
    ax.bar(x_reg + i*w, means, w, yerr=sems, color=color, edgecolor='black',
           linewidth=0.3, label=slabel.replace('\n', ' '), capsize=3)

ax.set_xticks(x_reg + w)
ax.set_xticklabels(regime_list, fontsize=9)
ax.set_ylabel('VA-VB correlation')
ax.set_title('(c) Coordination requires criticality')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(0, color='black', linewidth=0.5)

# ── (d) Motor time series: connectome only ────────────────────────
ax = fig.add_subplot(gs[1, 1])
states_A = all_results['Critical'][model_keys[0]]['states_last']
n_show = min(800, len(states_A))
t_plot = np.arange(n_show) * dt_stored
for j in VA[:5]:
    ax.plot(t_plot, states_A[:n_show, j], linewidth=0.5, alpha=0.6, color='#2196F3')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\cos\theta$ (VA neurons)')
ax.set_title('(d) Motor output: connectome only')
ax.grid(True, alpha=0.2)

# ── (e) Motor time series: full field model ───────────────────────
ax = fig.add_subplot(gs[1, 2])
states_C = all_results['Critical'][model_keys[2]]['states_last']
n_show_c = min(n_show, len(states_C))
for j in VA[:5]:
    ax.plot(t_plot[:n_show_c], states_C[:n_show_c, j], linewidth=0.5, alpha=0.6, color='#EF5350')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\cos\theta$ (VA neurons)')
ax.set_title('(e) Motor output: full field model')
ax.grid(True, alpha=0.2)

plt.savefig(fig_dir / 'fig_coordination.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_coordination.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig_coordination")
print("\nDone!")
