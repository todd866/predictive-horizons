#!/usr/bin/env python3
"""
C. elegans: The Prediction Test.

Does the trilayer coupling architecture let the worm predict its
environment? Does each coupling layer extend the prediction horizon?

EXPERIMENT:
  Run four models on the SAME structured environment:
    Model 1: Connectome only (OpenWorm baseline)
    Model 2: Connectome + ephaptic
    Model 3: Connectome + neuropeptide
    Model 4: Connectome + ephaptic + neuropeptide (full trilayer)

  For each, measure:
    I_pred(Δt) = I(X_internal(t); X_env(t+Δt))

  Environment has three timescales:
    Slow (τ=10s): predictable → should appear as MI at long lags
    Medium (τ=2s): local fluctuations → MI at medium lags
    Fast (τ=0.3s): noise → should NOT produce MI

  Prediction: τ_pred is LONGER for models with more coupling layers.

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

dt = 0.001
T_total = 120.0
T_trans = 20.0
n_steps = int(T_total / dt)
n_trans = int(T_trans / dt)
subsample = max(1, int(0.01 / dt))
dt_stored = dt * subsample

print("="*70)
print("C. elegans: The Prediction Test")
print("="*70)
print(f"  dt={dt}, T={T_total}s, steps={n_steps:,}")

# ════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════
print("\nLoading data...")

neurons_set = set()
chemical_edges = []; gap_edges = []
with open(data_dir / "herm_full_edgelist.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row['Source'].strip(); tgt = row['Target'].strip()
        w = int(row['Weight'].strip()); typ = row['Type'].strip()
        neurons_set.add(src); neurons_set.add(tgt)
        if typ == 'chemical': chemical_edges.append((src, tgt, w))
        elif typ == 'electrical': gap_edges.append((src, tgt, w))

npp_df = pd.read_csv(data_dir / "data" / "neuropeptide_connectome_LR.csv", index_col=0)
npp_neurons = list(npp_df.columns)

with open(data_dir / "data" / "neuron_positions_3d.txt") as f:
    header = f.readline().strip().lstrip('#')
    pos3d_names = header.split()
    pos3d_coords = np.array([[float(x) for x in line.strip().split()]
                             for line in f if len(line.strip().split()) == 3])
pos3d_map = {name: pos3d_coords[i] for i, name in enumerate(pos3d_names) if i < len(pos3d_coords)}

neurons = sorted(neurons_set); N = len(neurons)
neuron_idx = {n: i for i, n in enumerate(neurons)}

W_chem = np.zeros((N, N)); W_gap = np.zeros((N, N))
for src, tgt, w in chemical_edges: W_chem[neuron_idx[src], neuron_idx[tgt]] += w
for src, tgt, w in gap_edges:
    i, j = neuron_idx[src], neuron_idx[tgt]; W_gap[i,j] += w; W_gap[j,i] += w
W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

W_npp = np.zeros((N, N))
for sn in npp_neurons:
    if sn not in neuron_idx: continue
    si = neuron_idx[sn]
    for tn in npp_neurons:
        if tn not in neuron_idx: continue
        ti = neuron_idx[tn]
        if sn in npp_df.index and tn in npp_df.columns:
            v = npp_df.loc[sn, tn]
            if v > 0: W_npp[si, ti] = v
W_npp_norm = W_npp / max(W_npp.max(), 1)

rng_pos = np.random.RandomState(0)
pos_3d = np.zeros((N, 3))
for i, n in enumerate(neurons): pos_3d[i] = pos3d_map.get(n, rng_pos.randn(3)*0.3)
dist_3d = squareform(pdist(pos_3d))
W_eph = np.exp(-dist_3d**2 / (2*0.8**2))
np.fill_diagonal(W_eph, 0)
W_eph_norm = W_eph / np.maximum(W_eph.sum(axis=1, keepdims=True), 1)

sensory_pfx = ('ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB',
               'AWC','AFD','BAG','IL1','IL2','OLL','OLQ','PHA','PHB','PLM',
               'ALM','AVM','PVD','FLP')
motor_pfx = ('VA','VB','VD','DA','DB','DD','AS','RMD','RME','SMD','SMB')
sensory = [i for i, n in enumerate(neurons) if n.startswith(sensory_pfx)]
motor = [i for i, n in enumerate(neurons) if n.startswith(motor_pfx)]
VA = [i for i, n in enumerate(neurons) if n.startswith('VA')]
VB = [i for i, n in enumerate(neurons) if n.startswith('VB')]
n_sensors = min(len(sensory), 50)

rng_freq = np.random.RandomState(99)
omegas = np.zeros(N)
inter = [i for i in range(N) if i not in sensory and i not in motor]
for i in inter: omegas[i] = 2*np.pi*(0.7+0.3*rng_freq.rand())
for i in sensory: omegas[i] = 2*np.pi*(0.9+0.3*rng_freq.rand())
for i in motor: omegas[i] = 2*np.pi*(1.0+0.3*rng_freq.rand())

print(f"  {N} neurons, {(W_npp>0).sum()} NPP interactions")

# ════════════════════════════════════════════════════════════════════
# MULTI-SCALE ENVIRONMENT
# ════════════════════════════════════════════════════════════════════
print("  Generating multi-scale environment...")

def gen_multiscale_env(n_steps, dt, n_ch, seed=42):
    rng = np.random.RandomState(seed)
    taus = [10.0, 2.0, 0.3]
    sigmas = [0.6, 0.4, 0.3]
    comps = []
    for tau, sigma in zip(taus, sigmas):
        c = np.zeros((n_steps, n_ch))
        decay = np.exp(-dt/tau); ns = sigma*np.sqrt(2*dt/tau)
        for t in range(1, n_steps):
            c[t] = c[t-1]*decay + ns*rng.randn(n_ch)
        comps.append(c)
    return sum(comps), comps

env_total, env_comps = gen_multiscale_env(n_steps, dt, n_sensors)
env_stored = env_total[n_trans::subsample]
env_slow = env_comps[0][n_trans::subsample]
env_med = env_comps[1][n_trans::subsample]

K_chem = 0.4; K_gap = 0.8; K_eph = 1.0; K_npp = 0.5
tau_npp = 2.0; K_drive = 1.0; sigma_noise = 0.12
delta_ts = np.linspace(0.05, 10.0, 40)

# ════════════════════════════════════════════════════════════════════
# MODEL RUNNER
# ════════════════════════════════════════════════════════════════════

def run_model(label, use_eph=False, use_npp=False, seed=77):
    print(f"\n  Running: {label}...")
    t0 = time.time()
    rng = np.random.RandomState(seed)
    theta = 2*np.pi*rng.rand(N)
    npp_mod = np.zeros(N)
    rng_n = np.random.RandomState(seed+100)
    n_stored_max = (n_steps - n_trans) // subsample
    states = np.zeros((n_stored_max, N*2))
    si = 0
    for t in range(1, n_steps):
        drive = np.zeros(N)
        for k, idx in enumerate(sensory[:n_sensors]):
            drive[idx] = K_drive * env_total[t, k]
        sin_diff = np.sin(theta[:, None] - theta[None, :])
        coupling = K_chem*np.sum(W_chem_norm*sin_diff, axis=0) + \
                   K_gap*np.sum(W_gap_norm*sin_diff, axis=0)
        if use_eph:
            coupling += K_eph*np.sum(W_eph_norm*sin_diff, axis=0)
        if use_npp:
            npp_in = np.zeros(N)
            for ni in range(N):
                npp_in[ni] = np.sum(W_npp_norm[:, ni]*(0.5+0.5*np.cos(theta)))
            npp_mod += (-npp_mod + K_npp*npp_in)/tau_npp * dt
            omega_mod = omegas + npp_mod
        else:
            omega_mod = omegas
        dtheta = omega_mod + drive + coupling
        dtheta += sigma_noise*rng_n.randn(N)/np.sqrt(dt)
        theta += dtheta*dt
        if t >= n_trans and (t - n_trans) % subsample == 0 and si < n_stored_max:
            states[si, :N] = np.cos(theta)
            states[si, N:] = np.sin(theta)
            si += 1
    print(f"    Done in {time.time()-t0:.1f}s")
    return states[:si]

def gaussian_mi(X, Y):
    n = X.shape[0]; dX, dY = X.shape[1], Y.shape[1]; a = 0.01
    Xc = X-X.mean(0); Yc = Y-Y.mean(0)
    C_XX = Xc.T@Xc/n+a*np.eye(dX); C_YY = Yc.T@Yc/n+a*np.eye(dY)
    XY = np.hstack([Xc,Yc]); C_XY = XY.T@XY/n+a*np.eye(dX+dY)
    s1,l1 = np.linalg.slogdet(C_XX); s2,l2 = np.linalg.slogdet(C_YY); s3,l3 = np.linalg.slogdet(C_XY)
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
        ml = min(len(Xn), len(Ef))
        if ml > 3000:
            idx = np.random.choice(ml, 3000, replace=False)
            Xn = Xn[idx]; Ef = Ef[idx]
        else: Xn = Xn[:ml]; Ef = Ef[:ml]
        I[i] = gaussian_mi(Xn, Ef)
    return I

def tau_pred(mi, lags, frac=0.1):
    peak = mi.max()
    if peak <= 0: return 0.0
    above = np.where(mi > frac*peak)[0]
    return lags[above[-1]] if len(above) > 0 else 0.0

# ════════════════════════════════════════════════════════════════════
# RUN ALL FOUR MODELS
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70)

models = [
    ("1: Connectome only",           False, False),
    ("2: + Ephaptic",                True,  False),
    ("3: + Neuropeptide",            False, True),
    ("4: Full trilayer",             True,  True),
]

results = {}
for label, eph, npp in models:
    states = run_model(label, use_eph=eph, use_npp=npp)
    mi_tot = mi_profile(states, env_stored, delta_ts, dt_stored)
    mi_slow = mi_profile(states, env_slow, delta_ts, dt_stored)
    mi_med = mi_profile(states, env_med, delta_ts, dt_stored)
    tp_tot = tau_pred(mi_tot, delta_ts)
    tp_slow = tau_pred(mi_slow, delta_ts)

    va_vb = 0.0
    if VA and VB:
        a = np.mean(states[:, VA], axis=1); b = np.mean(states[:, VB], axis=1)
        if np.std(a)>1e-10 and np.std(b)>1e-10: va_vb = pearsonr(a, b)[0]

    X = states - states.mean(0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s**2/len(X); ev = ev[ev>1e-10]
    deff = (ev.sum())**2/(ev**2).sum()

    results[label] = dict(mi_tot=mi_tot, mi_slow=mi_slow, mi_med=mi_med,
                          tp_tot=tp_tot, tp_slow=tp_slow, va_vb=va_vb, deff=deff)
    print(f"    MI peak={mi_tot.max():.4f}  τ_pred={tp_tot:.1f}s  τ_pred(slow)={tp_slow:.1f}s  VA-VB={va_vb:.3f}")

wall_total = time.time() - wall_start

print(f"\n{'='*70}")
print(f"SUMMARY (wall time: {wall_total/60:.1f} min)")
print(f"{'='*70}")
print(f"{'Model':<30} {'MI peak':>8} {'τ_pred':>7} {'τ_slow':>7} {'VA-VB':>7} {'D_eff':>7}")
print("-"*70)
for label, _, _ in models:
    r = results[label]
    print(f"{label:<30} {r['mi_tot'].max():>8.4f} {r['tp_tot']:>6.1f}s {r['tp_slow']:>6.1f}s {r['va_vb']:>7.3f} {r['deff']:>7.1f}")

# ════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

colors = ['#2196F3', '#FF9800', '#9C27B0', '#EF5350']
model_labels = [m[0] for m in models]
short_labels = ['Conn\nonly', '+Eph', '+NPP', 'Full\ntrilayer']

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 4, hspace=0.4, wspace=0.4)

# (a) MI with total environment
ax = fig.add_subplot(gs[0, :2])
for label, color in zip(model_labels, colors):
    ax.plot(delta_ts, results[label]['mi_tot'], color=color, linewidth=2,
            label=label.split(':')[1].strip())
ax.set_xlabel(r'Prediction lag $\Delta t$ (s)')
ax.set_ylabel(r'$I(X_{\mathrm{int}}(t);\, X_{\mathrm{env}}(t{+}\Delta t))$ (nats)')
ax.set_title('(a) Future mutual information', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# (b) MI with slow component
ax = fig.add_subplot(gs[0, 2:])
for label, color in zip(model_labels, colors):
    ax.plot(delta_ts, results[label]['mi_slow'], color=color, linewidth=2,
            label=label.split(':')[1].strip())
ax.set_xlabel(r'Prediction lag $\Delta t$ (s)')
ax.set_ylabel(r'MI with slow environment (nats)')
ax.set_title('(b) Tracking the predictable component (τ_env = 10s)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# (c) Prediction horizon
ax = fig.add_subplot(gs[1, 0])
tps = [results[l]['tp_tot'] for l in model_labels]
bars = ax.bar(short_labels, tps, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, tps):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{val:.1f}s',
            ha='center', fontsize=9)
ax.set_ylabel(r'$\tau_{\mathrm{pred}}$ (s)')
ax.set_title('(c) Prediction horizon'); ax.grid(True, alpha=0.2, axis='y')

# (d) τ_pred for slow component
ax = fig.add_subplot(gs[1, 1])
tps_slow = [results[l]['tp_slow'] for l in model_labels]
bars = ax.bar(short_labels, tps_slow, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, tps_slow):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{val:.1f}s',
            ha='center', fontsize=9)
ax.set_ylabel(r'$\tau_{\mathrm{pred}}^{\mathrm{slow}}$ (s)')
ax.set_title('(d) Slow-component horizon'); ax.grid(True, alpha=0.2, axis='y')

# (e) MI at specific lags
ax = fig.add_subplot(gs[1, 2])
lags_show = [0.5, 2.0, 5.0]
x = np.arange(len(lags_show)); w = 0.2
for i, (label, color) in enumerate(zip(model_labels, colors)):
    vals = [results[label]['mi_tot'][np.argmin(np.abs(delta_ts-l))] for l in lags_show]
    ax.bar(x+i*w, vals, w, color=color, edgecolor='black', linewidth=0.3,
           label=label.split(':')[1].strip()[:8])
ax.set_xticks(x+1.5*w); ax.set_xticklabels([f'{l}s' for l in lags_show])
ax.set_ylabel('MI (nats)'); ax.set_title('(e) MI at specific lags')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='y')

# (f) D_eff + VA-VB
ax = fig.add_subplot(gs[1, 3])
deffs = [results[l]['deff'] for l in model_labels]
ax.bar(short_labels, deffs, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel(r'$D_{\mathrm{eff}}$'); ax.set_title('(f) Effective dimensionality')
ax.grid(True, alpha=0.2, axis='y')

plt.savefig(fig_dir / 'fig_prediction_test.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_prediction_test.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved fig_prediction_test")
print(f"\nTotal: {wall_total/60:.1f} min")
print("Done!")
