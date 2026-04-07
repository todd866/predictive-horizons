#!/usr/bin/env python3
"""
Nonergodic defense simulation — multi-mode environment + metabolic budget.

Composes the celegans package modules into the full nonergodic defense
simulation.  Runs four model configurations on a multi-mode environment
with metabolic budget, and generates three publication-quality figures:

1. fig_deff_trajectories  — D_eff(t) for all 4 models + summary bar chart
2. fig_spectral_bandwidth — power spectra comparison + overlap summary
3. fig_metabolic_tradeoff — budget over time + D_eff vs overlap + cost of complexity

Model configurations:
- Connectome only (use_eph=False, use_npp=False)
- + Ephaptic        (use_eph=True,  use_npp=False)
- + Neuropeptide    (use_eph=False, use_npp=True)
- Full trilayer     (use_eph=True,  use_npp=True)

Author: Ian Todd
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from celegans import (
    load_worm_data, build_coupling_matrices, build_grid_mapping,
    WormState, WormParams, assign_frequencies, step,
    generate_multimode_env, default_taus, default_amplitudes,
    d_eff, d_eff_timeseries, spectral_overlap, motor_correlation,
    classify_trajectory, mi_profile,
    MetabolicBudget, UnlimitedBudget,
)

# ── CLI arguments ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--T', type=float, default=1000.0)
parser.add_argument('--K', type=int, default=8)
parser.add_argument('--no-budget', action='store_true')
parser.add_argument('--parallel', action='store_true', default=True,
                    help='Run models in parallel on multiple cores (default: on)')
parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                    help='Run models sequentially')
args, _ = parser.parse_known_args()

# ── Simulation parameters ─────────────────────────────────────────────
dt = 0.001
T_total = args.T
T_trans = min(50.0, 0.05 * T_total)
Nx = 100                                  # coherence field grid
n_steps = int(T_total / dt)
n_trans = int(T_trans / dt)
subsample = max(1, int(0.02 / dt))        # store every 20 ms
K_env = args.K                             # number of environment modes

BUDGET_PARAMS = dict(
    R_max=100.0, E_in=2.0, cost_coupling=2.0,
    cost_coherence=1.0, cost_dimensional=0.2,
)
# E_in=2.0 sits between connectome-only expenditure (~0.9/s) and
# trilayer expenditure (~3.6/s), so the connectome coasts while
# the trilayer must manage its budget.

# ── Figure styling ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm', 'axes.linewidth': 0.8,
})
colors = ['#2196F3', '#FF9800', '#9C27B0', '#EF5350']

DATA_DIR = Path(__file__).parent / "openworm"
FIG_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1. Load data and build coupling matrices
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Nonergodic Defense Simulation")
print(f"  T = {T_total:.0f} s, K = {K_env} modes, dt = {dt}")
print(f"  Budget: {'OFF' if args.no_budget else 'ON'}")
print("=" * 60)

print("\n[1/5] Loading connectome data...")
wd = load_worm_data(DATA_DIR)
cm = build_coupling_matrices(wd)
gm = build_grid_mapping(wd.pos_1d, Nx)
omegas = assign_frequencies(wd.N, wd.sensory, wd.motor, wd.inter)
n_sensors = min(len(wd.sensory), 50)

print(f"  N = {wd.N} neurons, {len(wd.sensory)} sensory, "
      f"{len(wd.motor)} motor, {len(wd.inter)} inter")
print(f"  Grid: Nx = {Nx}, n_sensors = {n_sensors}")

# ══════════════════════════════════════════════════════════════════════
# 2. Generate multi-mode environment
# ══════════════════════════════════════════════════════════════════════
print("\n[2/5] Generating multi-mode environment...")
taus = default_taus(K_env)
amplitudes = default_amplitudes(taus)

env_total, env_modes, taus_used = generate_multimode_env(
    n_steps, dt, n_sensors, taus, amplitudes=amplitudes, seed=42,
)
print(f"  Modes: {K_env}, taus = [{taus[0]:.2f} .. {taus[-1]:.1f}] s")
print(f"  Amplitudes: [{amplitudes[0]:.3f} .. {amplitudes[-1]:.3f}]")
print(f"  Env shape: {env_total.shape}")


# ══════════════════════════════════════════════════════════════════════
# 3. Run four model configurations
# ══════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = [
    ("Connectome",    False, False),
    ("+ Ephaptic",    True,  False),
    ("+ Neuropeptide", False, True),
    ("Full trilayer", True,  True),
]


def run_model(label, use_eph, use_npp, seed=77):
    """Run one model configuration and return all stored arrays."""
    print(f"\n  --- {label} (eph={use_eph}, npp={use_npp}) ---")
    t0 = time.time()

    state = WormState(wd.N, Nx, seed=seed)
    params = WormParams()

    if args.no_budget:
        budget = UnlimitedBudget()
    else:
        budget = MetabolicBudget(**BUDGET_PARAMS)

    # Pre-allocate storage
    n_stored = (n_steps - n_trans) // subsample + 1
    theta_hist = np.zeros((n_stored, wd.N))
    phi_hist = np.zeros((n_stored, Nx))
    diag_hist = np.zeros((n_stored, 5))   # conn_rms, eph_rms, npp_std, order_r, mean_phi
    budget_hist = np.zeros(n_stored)
    env_stored = np.zeros((n_stored, n_sensors))

    store_idx = 0
    report_interval = max(1, n_steps // 20)  # every 5%

    for t in range(n_steps):
        env_signal = env_total[t, :n_sensors]
        b_scale = budget.budget_scale

        diag = step(
            state, params, omegas, cm, gm,
            wd.sensory, wd.motor, n_sensors,
            env_signal, dt,
            use_eph=use_eph, use_npp=use_npp,
            budget_scale=b_scale,
        )

        budget.update(dt, **diag)

        # Store after transient
        if t >= n_trans and (t - n_trans) % subsample == 0:
            if store_idx < n_stored:
                theta_hist[store_idx] = state.theta.copy()
                phi_hist[store_idx] = state.phi.copy()
                diag_hist[store_idx] = [
                    diag['conn_rms'], diag['eph_rms'], diag['npp_std'],
                    diag['order_r'], diag['mean_phi'],
                ]
                budget_hist[store_idx] = b_scale
                env_stored[store_idx] = env_signal
                store_idx += 1

        if t > 0 and t % report_interval == 0:
            pct = 100 * t / n_steps
            print(f"    {pct:5.1f}%  budget={b_scale:.3f}  "
                  f"order_r={diag['order_r']:.3f}  phi={diag['mean_phi']:.3f}")

    wall = time.time() - t0
    # Trim
    theta_hist = theta_hist[:store_idx]
    phi_hist = phi_hist[:store_idx]
    diag_hist = diag_hist[:store_idx]
    budget_hist = budget_hist[:store_idx]
    env_stored = env_stored[:store_idx]

    print(f"    Done: {store_idx} stored points, {wall:.1f}s wall time")

    return dict(
        label=label,
        theta_hist=theta_hist,
        phi_hist=phi_hist,
        diag_hist=diag_hist,
        budget_hist=budget_hist,
        env_stored=env_stored,
        wall_time=wall,
        use_eph=use_eph,
        use_npp=use_npp,
    )


print("\n[3/5] Running models...")

def _run_one(args):
    """Wrapper for parallel execution."""
    label, eph, npp = args
    return run_model(label, eph, npp, seed=77)

if args.parallel and len(MODEL_CONFIGS) > 1:
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    n_workers = min(len(MODEL_CONFIGS), mp.cpu_count())
    print(f"  Running {len(MODEL_CONFIGS)} models on {n_workers} cores...")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_run_one, MODEL_CONFIGS)
else:
    results = []
    for label, eph, npp in MODEL_CONFIGS:
        r = run_model(label, eph, npp, seed=77)
        results.append(r)

# ══════════════════════════════════════════════════════════════════════
# 4. Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n[4/5] Analysing results...")

dt_stored = dt * subsample
window_sec = 20.0                         # 20-second windows for D_eff timeseries
window_samples = max(10, int(window_sec / dt_stored))

for r in results:
    theta = r['theta_hist']
    N = theta.shape[1]

    # State matrix for D_eff and spectral analysis
    states = np.column_stack([np.cos(theta), np.sin(theta)])
    r['states'] = states

    # D_eff overall
    r['d_eff_overall'] = d_eff(states)

    # D_eff timeseries
    r['d_eff_ts'] = d_eff_timeseries(states, window_samples)

    # Trajectory classification
    r['traj_class'] = classify_trajectory(r['d_eff_ts'])

    # Spectral overlap (internal cos/sin channels vs environment)
    sp = spectral_overlap(states[:, :N], r['env_stored'], dt_stored)
    r['spectral'] = sp
    r['total_overlap'] = sp['total_overlap']

    # Motor coordination (VA-VB)
    r['va_vb_corr'] = motor_correlation(theta, wd.VA, wd.VB)

    # MI profile at 30 lags from 0.1s to 20s
    delta_ts_mi = np.linspace(0.1, 20.0, 30)
    r['mi_lags'] = delta_ts_mi
    r['mi_vals'] = mi_profile(states, r['env_stored'], delta_ts_mi, dt_stored,
                              n_pcs=50, max_samples=3000)
    r['mi_peak'] = float(r['mi_vals'].max()) if len(r['mi_vals']) else 0.0

    # Summary stats
    r['mean_budget'] = float(r['budget_hist'].mean())
    r['mean_phi'] = float(r['diag_hist'][:, 4].mean())
    r['mean_order_r'] = float(r['diag_hist'][:, 3].mean())

    print(f"  {r['label']:20s}  D_eff={r['d_eff_overall']:.2f}  "
          f"class={r['traj_class']:10s}  overlap={r['total_overlap']:.4f}  "
          f"VA-VB={r['va_vb_corr']:.3f}  budget={r['mean_budget']:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 5. Figures
# ══════════════════════════════════════════════════════════════════════
print("\n[5/5] Generating figures...")

# ── Figure 1: D_eff trajectories ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i, r in enumerate(results):
    ts = r['d_eff_ts']
    t_axis = np.arange(len(ts)) * window_sec + T_trans
    lbl = f"{r['label']} ({r['traj_class']})"
    ax1.plot(t_axis, ts, color=colors[i], linewidth=1.3, label=lbl)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel(r'$D_{\mathrm{eff}}$')
ax1.set_title(r'(a) $D_{\mathrm{eff}}(t)$ trajectories')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.2)

# Bar chart
labels = [r['label'] for r in results]
d_effs = [r['d_eff_overall'] for r in results]
classes = [r['traj_class'] for r in results]
x = np.arange(len(labels))
bars = ax2.bar(x, d_effs, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
ax2.set_ylabel(r'$D_{\mathrm{eff}}$ (overall)')
ax2.set_title(r'(b) Overall $D_{\mathrm{eff}}$')
ax2.grid(True, alpha=0.2, axis='y')
for xi, cls in zip(x, classes):
    ax2.text(xi, d_effs[int(xi)] * 1.02, cls, ha='center', va='bottom',
             fontsize=7, style='italic')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_deff_trajectories.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig_deff_trajectories.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig_deff_trajectories")

# ── Figure 2: Spectral bandwidth ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (a) Log-scale PSD
sp_env = results[0]['spectral']  # same env for all
freqs = sp_env['freqs']
mask = freqs > 0
ax1.fill_between(freqs[mask], sp_env['psd_env'][mask] / sp_env['psd_env'].max(),
                 alpha=0.15, color='gray', label='Environment')
for i, r in enumerate(results):
    psd = r['spectral']['psd_internal']
    psd_norm = psd / psd.max() if psd.max() > 0 else psd
    ax1.plot(freqs[mask], psd_norm[mask], color=colors[i],
             linewidth=1.0, alpha=0.8, label=r['label'])

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Normalised PSD')
ax1.set_title('(a) Power spectra')
ax1.legend(fontsize=7, loc='best')
ax1.grid(True, alpha=0.2, which='both')

# (b) Spectral overlap bar chart
overlaps = [r['total_overlap'] for r in results]
bars = ax2.bar(x, overlaps, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
ax2.set_ylabel('Spectral overlap (env-weighted)')
ax2.set_title('(b) Total spectral overlap')
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_spectral_bandwidth.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig_spectral_bandwidth.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig_spectral_bandwidth")

# ── Figure 3: Metabolic tradeoff ─────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# (a) Budget scale over time
for i, r in enumerate(results):
    bh = r['budget_hist']
    t_axis = np.linspace(T_trans, T_total, len(bh))
    ax1.plot(t_axis, bh, color=colors[i], linewidth=1.0, label=r['label'])

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Budget scale $R/R_{\\mathrm{max}}$')
ax1.set_title('(a) Metabolic budget over time')
ax1.legend(fontsize=7, loc='best')
ax1.set_ylim(-0.05, 1.1)
ax1.grid(True, alpha=0.2)

# (b) D_eff vs spectral overlap scatter
for i, r in enumerate(results):
    ax2.scatter(r['total_overlap'], r['d_eff_overall'],
                color=colors[i], s=80, zorder=5, edgecolors='black', linewidths=0.5)
    ax2.annotate(r['label'], (r['total_overlap'], r['d_eff_overall']),
                 textcoords='offset points', xytext=(8, 4), fontsize=7)

ax2.set_xlabel('Spectral overlap')
ax2.set_ylabel(r'$D_{\mathrm{eff}}$')
ax2.set_title(r'(b) $D_{\mathrm{eff}}$ vs spectral overlap')
ax2.grid(True, alpha=0.2)

# (c) Mean budget vs D_eff scatter
for i, r in enumerate(results):
    ax3.scatter(r['mean_budget'], r['d_eff_overall'],
                color=colors[i], s=80, zorder=5, edgecolors='black', linewidths=0.5)
    ax3.annotate(r['label'], (r['mean_budget'], r['d_eff_overall']),
                 textcoords='offset points', xytext=(8, 4), fontsize=7)

ax3.set_xlabel('Mean budget scale')
ax3.set_ylabel(r'$D_{\mathrm{eff}}$')
ax3.set_title(r'(c) Cost of complexity')
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_metabolic_tradeoff.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig_metabolic_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig_metabolic_tradeoff")

# ══════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print(f"{'Model':20s} {'D_eff':>8s} {'Class':>10s} {'Overlap':>8s} "
      f"{'VA-VB':>7s} {'Budget':>7s} {'Phi':>6s} {'Order':>6s} {'Wall(s)':>8s}")
print("-" * 90)
for r in results:
    print(f"{r['label']:20s} {r['d_eff_overall']:8.2f} {r['traj_class']:>10s} "
          f"{r['total_overlap']:8.4f} {r['va_vb_corr']:7.3f} "
          f"{r['mean_budget']:7.3f} {r['mean_phi']:6.3f} "
          f"{r['mean_order_r']:6.3f} {r['wall_time']:8.1f}")
print("=" * 90)


def _tag(value):
    return f"{value:g}".replace('.', 'p')


summary_path = RESULTS_DIR / (
    f"nonergodic_summary_T{_tag(T_total)}_K{K_env}"
    f"{'_nobudget' if args.no_budget else ''}.json"
)
summary_payload = {
    "simulation": {
        "T_total_s": float(T_total),
        "T_trans_s": float(T_trans),
        "dt_s": float(dt),
        "subsample": int(subsample),
        "K_env": int(K_env),
        "budget_enabled": not args.no_budget,
        "taus_s": [float(x) for x in taus_used],
        "amplitudes": [float(x) for x in amplitudes],
    },
    "models": [
        {
            "label": r["label"],
            "d_eff": float(r["d_eff_overall"]),
            "trajectory": r["traj_class"],
            "spectral_overlap": float(r["total_overlap"]),
            "mi_peak_nats": float(r["mi_peak"]),
            "mean_budget": float(r["mean_budget"]),
            "va_vb_corr": float(r["va_vb_corr"]),
            "mean_phi": float(r["mean_phi"]),
            "mean_order_r": float(r["mean_order_r"]),
            "wall_time_s": float(r["wall_time"]),
            "table_row": (
                f"{r['label']} & {r['d_eff_overall']:.2f} & "
                f"{r['total_overlap']:.4f} & {r['traj_class']} & "
                f"{r['mi_peak']:.3f} & {r['mean_budget']:.3f} \\\\"
            ),
        }
        for r in results
    ],
}
with open(summary_path, "w") as f:
    json.dump(summary_payload, f, indent=2)
print(f"\nSaved summary: {summary_path}")
print("\nDone!")
