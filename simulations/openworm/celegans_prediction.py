#!/usr/bin/env python3
"""
C. elegans: Discrete vs Oscillatory Simulation.

Uses the real OpenWorm connectome (448 neurons, 4681 chemical synapses,
2698 gap junctions).

Two models:
1. DISCRETE: LIF neurons with gap junctions as ohmic conductances.
   Chemical synapses as current injections. Standard computational
   neuroscience approach (approximates OpenWorm/c302).

2. OSCILLATORY: Coupled phase oscillators with frequency hierarchy.
   Gap junctions provide continuous bidirectional phase coupling.
   Chemical synapses provide directed phase coupling.
   Intrinsic frequencies assigned by neuron class:
   - Sensory: moderate (responsive to input)
   - Interneurons: slow (integrative, high-D)
   - Motor: faster (output generation)

Sweep coupling strength to find the regime where the oscillatory model
simultaneously maximises D_eff and predictive content.

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

# ── Load connectome ─────────────────────────────────────────────────
print("Loading C. elegans connectome...")
data_file = Path(__file__).parent / "herm_full_edgelist.csv"

neurons_set = set()
chemical_edges = []
gap_edges = []

with open(data_file) as f:
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

neurons = sorted(neurons_set)
N = len(neurons)
neuron_idx = {n: i for i, n in enumerate(neurons)}

W_chem = np.zeros((N, N))
W_gap = np.zeros((N, N))

for src, tgt, w in chemical_edges:
    W_chem[neuron_idx[src], neuron_idx[tgt]] += w
for src, tgt, w in gap_edges:
    i, j = neuron_idx[src], neuron_idx[tgt]
    W_gap[i, j] += w
    W_gap[j, i] += w

# Row-normalise for stability
chem_row_sum = W_chem.sum(axis=0, keepdims=True)
W_chem_norm = W_chem / np.maximum(chem_row_sum, 1)
gap_row_sum = W_gap.sum(axis=0, keepdims=True)
W_gap_norm = W_gap / np.maximum(gap_row_sum, 1)

# Neuron classification
sensory = [i for i, n in enumerate(neurons) if n.startswith((
    'ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC',
    'AFD','BAG','IL1','IL2','OLL','OLQ','PHA','PHB','PLM','ALM','AVM'))]
motor = [i for i, n in enumerate(neurons) if n.startswith((
    'VA','VB','VD','DA','DB','DD','AS','RMD','RME','SMD','SMB'))]
inter = [i for i in range(N) if i not in sensory and i not in motor]

print(f"  {N} neurons ({len(sensory)} sensory, {len(inter)} inter, {len(motor)} motor)")
print(f"  {len(chemical_edges)} chemical, {len(gap_edges)} gap junctions")

# ── Parameters ──────────────────────────────────────────────────────
dt = 0.001
T_total = 40.0
T_transient = 8.0
n_steps = int(T_total / dt)
n_transient = int(T_transient / dt)

tau_env = 2.0
sigma_env = 0.5
sigma_noise = 0.08
n_sensors = min(len(sensory), 50)

delta_ts = np.linspace(0.05, 4.0, 25)
dt_stored = dt * 10
subsample = 10

np.random.seed(42)

# ── Sensory input ───────────────────────────────────────────────────
def gen_env(n_steps, dt, n_s, tau, sigma):
    env = np.zeros((n_steps, n_s))
    decay = np.exp(-dt / tau)
    ns = sigma * np.sqrt(2 * dt / tau)
    for t in range(1, n_steps):
        env[t] = env[t-1] * decay + ns * np.random.randn(n_s)
    return env

# ── Model 1: Discrete LIF ──────────────────────────────────────────
def run_discrete(sensory_input, K_chem=0.5, K_gap_conductance=0.3):
    tau_m = 0.02
    V = np.zeros(N)
    n_stored = (n_steps - n_transient) // subsample
    states = np.zeros((n_stored, N))
    si = 0

    for t in range(1, n_steps):
        # Sensory drive
        drive = np.zeros(N)
        for k, idx in enumerate(sensory[:n_sensors]):
            drive[idx] = sensory_input[t, k]

        # Chemical synapses: weighted presynaptic activity
        pre_activity = np.tanh(V)
        syn = K_chem * (W_chem_norm.T @ pre_activity)

        # Gap junctions as ohmic conductances: g_ij * (V_j - V_i)
        gap = K_gap_conductance * (W_gap_norm @ V - V * W_gap_norm.sum(axis=0))

        dV = (-V / tau_m + drive + syn + gap) * dt
        dV += sigma_noise * np.random.randn(N) * np.sqrt(dt)
        V += dV
        V = np.clip(V, -3, 3)

        if t >= n_transient and (t - n_transient) % subsample == 0 and si < n_stored:
            states[si] = V.copy()
            si += 1
    return states[:si]

# ── Model 2: Oscillatory ───────────────────────────────────────────
def run_oscillatory(sensory_input, K_chem=0.5, K_gap=1.5, K_drive=2.0):
    # Frequency hierarchy: interneurons slow, sensory medium, motor faster
    omegas = np.zeros(N)
    for i in inter:
        omegas[i] = 2 * np.pi * (0.3 + 0.4 * np.random.rand())  # 0.3-0.7 Hz
    for i in sensory:
        omegas[i] = 2 * np.pi * (0.8 + 0.6 * np.random.rand())  # 0.8-1.4 Hz
    for i in motor:
        omegas[i] = 2 * np.pi * (1.5 + 1.0 * np.random.rand())  # 1.5-2.5 Hz

    theta = 2 * np.pi * np.random.rand(N)
    n_stored = (n_steps - n_transient) // subsample
    states = np.zeros((n_stored, N * 2))
    si = 0

    for t in range(1, n_steps):
        # Sensory drive
        drive = np.zeros(N)
        for k, idx in enumerate(sensory[:n_sensors]):
            drive[idx] = K_drive * sensory_input[t, k]

        # Vectorised coupling
        sin_diff = np.sin(theta[:, None] - theta[None, :])
        chem_coupling = K_chem * np.sum(W_chem_norm * sin_diff, axis=0)
        gap_coupling = K_gap * np.sum(W_gap_norm * sin_diff, axis=0)

        dtheta = omegas + drive + chem_coupling + gap_coupling
        dtheta += sigma_noise * np.random.randn(N) / np.sqrt(dt)
        theta += dtheta * dt

        if t >= n_transient and (t - n_transient) % subsample == 0 and si < n_stored:
            states[si, :N] = np.cos(theta)
            states[si, N:] = np.sin(theta)
            si += 1
    return states[:si]

# ── Metrics ─────────────────────────────────────────────────────────
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

def mi_profile(internal, env_sig, delta_ts, dt_stored, n_pcs=50):
    n_t = internal.shape[0]
    X = internal - internal.mean(0)
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

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C. elegans: Discrete vs Oscillatory")
print("="*60)

sensory_input = gen_env(n_steps, dt, n_sensors, tau_env, sigma_env)
env_stored = sensory_input[n_transient::subsample]

# ── Run discrete model ──────────────────────────────────────────────
print("\nRunning discrete (LIF) model...")
states_d = run_discrete(sensory_input)
deff_d = deff(states_d)
mi_d = mi_profile(states_d, env_stored, delta_ts, dt_stored)
mi_short_d = np.mean(mi_d[:5])
print(f"  D_eff: {deff_d:.1f}, MI: {mi_short_d:.4f}")

# ── Sweep oscillatory model parameters ──────────────────────────────
print("\nSweeping oscillatory model (K_gap)...")
K_gap_values = [0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0]
sweep_deff = []
sweep_mi = []
sweep_states = {}

for kg in K_gap_values:
    print(f"  K_gap = {kg:.1f}")
    st = run_oscillatory(sensory_input, K_gap=kg, K_drive=2.0)
    d = deff(st)
    m = mi_profile(st, env_stored, delta_ts, dt_stored)
    ms = np.mean(m[:5])
    sweep_deff.append(d)
    sweep_mi.append(ms)
    sweep_states[kg] = (st, m, d, ms)
    print(f"    D_eff: {d:.1f}, MI: {ms:.4f}")

# Find best: maximise MI * log(D_eff) — balances both
scores = [mi * np.log(d+1) for mi, d in zip(sweep_mi, sweep_deff)]
best_idx = np.argmax(scores)
best_kg = K_gap_values[best_idx]
best_st, best_mi_profile, best_deff, best_mi_short = sweep_states[best_kg]
print(f"\nBest regime: K_gap={best_kg}, D_eff={best_deff:.1f}, MI={best_mi_short:.4f}")
print(f"vs Discrete: D_eff={deff_d:.1f}, MI={mi_short_d:.4f}")
print(f"MI advantage: {best_mi_short/(mi_short_d+1e-10):.1f}x")

# ── Figure ──────────────────────────────────────────────────────────
print("\nGenerating figure...")
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 3, hspace=0.35, wspace=0.35)

# (a) MI profiles: best oscillatory vs discrete
ax = fig.add_subplot(gs[0, 0])
ax.plot(delta_ts, best_mi_profile, 'r-', linewidth=2,
        label=f'Oscillatory ($K_{{gap}}={best_kg}$)')
ax.plot(delta_ts, mi_d, 'b--', linewidth=2, label='Discrete (LIF)')
ax.axvline(tau_env, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel(r'Prediction lag $\Delta t$ (s)')
ax.set_ylabel(r'$I_{\mathrm{pred}}$ (nats)')
ax.set_title('(a) Future mutual information')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (b) Coupling sweep: D_eff and MI
ax = fig.add_subplot(gs[0, 1])
ax2 = ax.twinx()
l1 = ax.plot(K_gap_values, sweep_deff, 'ko-', markersize=5, linewidth=1.2, label=r'$D_{\mathrm{eff}}$')
l2 = ax2.plot(K_gap_values, sweep_mi, 'rs-', markersize=5, linewidth=1.2, label=r'$I_{\mathrm{pred}}$')
ax.axhline(deff_d, color='blue', linestyle='--', alpha=0.4, linewidth=0.8)
ax2.axhline(mi_short_d, color='blue', linestyle=':', alpha=0.4, linewidth=0.8)
ax.axvline(best_kg, color='green', linestyle='--', alpha=0.3)
ax.set_xlabel(r'Gap junction coupling $K_{\mathrm{gap}}$')
ax.set_ylabel(r'$D_{\mathrm{eff}}$')
ax2.set_ylabel(r'$\langle I_{\mathrm{pred}} \rangle$ (nats)', color='red')
ax.set_title('(b) Coupling sweep')
lines = l1 + l2
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')
ax.grid(True, alpha=0.2)

# (c) Bar comparison at best regime
ax = fig.add_subplot(gs[0, 2])
x = np.arange(2)
w = 0.35
bars1 = ax.bar(x - w/2, [mi_short_d, best_mi_short], w,
               label=r'$I_{\mathrm{pred}}$', color=['#2196F3', '#EF5350'],
               edgecolor='black', linewidth=0.5)
ax_r = ax.twinx()
bars2 = ax_r.bar(x + w/2, [deff_d, best_deff], w,
                 label=r'$D_{\mathrm{eff}}$', color=['#90CAF9', '#EF9A9A'],
                 edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(['Discrete\n(LIF)', 'Oscillatory\n(best)'])
ax.set_ylabel(r'$I_{\mathrm{pred}}$ (nats)')
ax_r.set_ylabel(r'$D_{\mathrm{eff}}$')
ax.set_title(f'(c) Best oscillatory ($K_{{gap}}={best_kg}$)')
ax.legend(loc='upper left', fontsize=8)
ax_r.legend(loc='upper right', fontsize=8)

# (d) Motor output — discrete
ax = fig.add_subplot(gs[1, 0])
t_plot = np.arange(min(800, len(states_d))) * dt_stored
motor_d = states_d[:800, motor[:min(8, len(motor))]]
for j in range(motor_d.shape[1]):
    ax.plot(t_plot[:len(motor_d)], motor_d[:, j], linewidth=0.5, alpha=0.6)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Motor neuron V')
ax.set_title('(d) Motor output: discrete')
ax.grid(True, alpha=0.2)

# (e) Motor output — oscillatory (best)
ax = fig.add_subplot(gs[1, 1])
motor_o = best_st[:800, [m for m in motor[:min(8, len(motor))]]]
for j in range(motor_o.shape[1]):
    ax.plot(t_plot[:len(motor_o)], motor_o[:, j], linewidth=0.5, alpha=0.6)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Motor neuron $\cos\theta$')
ax.set_title('(e) Motor output: oscillatory')
ax.grid(True, alpha=0.2)

# (f) Eigenspectra
ax = fig.add_subplot(gs[1, 2])
Xd = states_d - states_d.mean(0)
_, sd, _ = np.linalg.svd(Xd, full_matrices=False)
sd_n = sd**2/len(Xd); sd_n /= sd_n.sum()

Xo = best_st - best_st.mean(0)
_, so, _ = np.linalg.svd(Xo, full_matrices=False)
so_n = so**2/len(Xo); so_n /= so_n.sum()

n_show = min(80, len(sd_n), len(so_n))
ax.semilogy(range(1, n_show+1), sd_n[:n_show], 'b--', linewidth=1.5, label='Discrete')
ax.semilogy(range(1, n_show+1), so_n[:n_show], 'r-', linewidth=1.5, label='Oscillatory')
ax.set_xlabel('Principal component')
ax.set_ylabel('Variance fraction')
ax.set_title('(f) Eigenspectrum')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.savefig(fig_dir / 'fig_celegans_comparison.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_celegans_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig_celegans_comparison")

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Connectome: {N} neurons")
print(f"Best oscillatory regime: K_gap = {best_kg}")
print(f"  D_eff: {best_deff:.1f} (discrete: {deff_d:.1f})")
print(f"  MI: {best_mi_short:.4f} (discrete: {mi_short_d:.4f})")
print(f"  MI advantage: {best_mi_short/(mi_short_d+1e-10):.1f}x")
print(f"  D_eff ratio: {best_deff/deff_d:.2f}")
print("\nDone!")
