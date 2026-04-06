#!/usr/bin/env python3
"""
C. elegans: Full field-theoretic simulation.

This is the computationally intensive version. Each neuron is
multi-compartmental, embedded in a 2D bioelectric field on the
body surface, with neuropeptide diffusion, body mechanics, and
multi-scale coherence dynamics.

State variables (~10,000 total):
  - 2,240 neural oscillators (448 neurons × 5 compartments), each with
    phase θ, amplitude r, adaptation a → 6,720 neural variables
  - 2D bioelectric field E(x,y,t) on body surface (100×20 = 2,000 grid)
  - 3 neuropeptide species P_k(x,t) on body axis (200 points × 3)
  - Body curvature field κ(x,t) on body axis (200 points)
  - Coherence field φ(x,t) on body axis (200 points)
  - Metabolic field m(x,t) on body axis (200 points)

Physics:
  - Compartmental coupling: soma ↔ processes with cable equation-like dynamics
  - 2D bioelectric field: ∂E/∂t = D_E ∇²E + sources(neural) - decay
  - Neuropeptide diffusion: ∂P_k/∂t = D_P ∇²P_k + release - degradation
  - Body mechanics: ∂²κ/∂t² = -ε·κ - γ·∂κ/∂t + F_muscle + η (damped oscillator)
  - Coherence: Landau-Ginzburg as before but coupled to all fields
  - Metabolism: depletion from coherence + neural activity, slow recovery

Expected runtime: 4-8 hours on Apple M5.

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
import time

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
dt = 0.0002             # timestep (200μs — resolves fast ion dynamics)
T_total = 500.0         # 500 seconds total
T_transient = 50.0      # 50s transient
n_steps = int(T_total / dt)
n_trans = int(T_transient / dt)
subsample = max(1, int(0.05 / dt))  # store every 50ms
dt_stored = dt * subsample

# Spatial grids
Nx_body = 200           # 1D body axis grid
Nx_surface = 100        # 2D surface: axial
Ny_surface = 20         # 2D surface: circumferential
dx_body = 1.0 / Nx_body
dx_surf = 1.0 / Nx_surface
dy_surf = 1.0 / Ny_surface

# Compartments per neuron
N_comp = 5              # soma + 4 processes

print("="*70)
print("C. elegans: Full Field-Theoretic Simulation")
print("="*70)
print(f"  dt = {dt}s, T = {T_total}s, steps = {n_steps:,}")
print(f"  Body axis: {Nx_body} points")
print(f"  Body surface: {Nx_surface}×{Ny_surface} = {Nx_surface*Ny_surface} points")
print(f"  Compartments per neuron: {N_comp}")
print(f"  Storage interval: {dt_stored}s")
print(f"  Expected stored timepoints: {(n_steps - n_trans) // subsample:,}")

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
N_neurons = len(neurons)
N_total = N_neurons * N_comp  # total oscillators
neuron_idx = {n: i for i, n in enumerate(neurons)}

rng_pos = np.random.RandomState(0)
pos_2d = np.zeros((N_neurons, 2))
for i, n in enumerate(neurons):
    pos_2d[i] = positions.get(n, rng_pos.randn(2) * 0.3)

# 1D body axis position [0, 1]
x_coords = pos_2d[:, 0]
pos_1d = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)

# Map neurons to 2D surface grid (axial position + random circumferential)
rng_circ = np.random.RandomState(11)
neuron_surf_x = (pos_1d * (Nx_surface - 1)).astype(int).clip(0, Nx_surface - 1)
neuron_surf_y = rng_circ.randint(0, Ny_surface, N_neurons)

# Map to 1D body axis grid
grid_x = np.linspace(0, 1, Nx_body)
neuron_body_idx = np.array([np.argmin(np.abs(grid_x - p)) for p in pos_1d])

# Adjacency matrices (neuron-level, not compartment-level)
W_chem = np.zeros((N_neurons, N_neurons))
W_gap = np.zeros((N_neurons, N_neurons))
for src, tgt, w in chemical_edges:
    W_chem[neuron_idx[src], neuron_idx[tgt]] += w
for src, tgt, w in gap_edges:
    i, j = neuron_idx[src], neuron_idx[tgt]
    W_gap[i, j] += w; W_gap[j, i] += w
W_chem_norm = W_chem / np.maximum(W_chem.sum(axis=0, keepdims=True), 1)
W_gap_norm = W_gap / np.maximum(W_gap.sum(axis=0, keepdims=True), 1)

# Neuron classification
sensory_prefixes = ('ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB',
                    'AWC','AFD','BAG','IL1','IL2','OLL','OLQ','PHA','PHB','PLM',
                    'ALM','AVM','PVD','FLP')
motor_prefixes = ('VA','VB','VD','DA','DB','DD','AS','RMD','RME','SMD','SMB')
sensory = [i for i, n in enumerate(neurons) if n.startswith(sensory_prefixes)]
motor = [i for i, n in enumerate(neurons) if n.startswith(motor_prefixes)]
inter = [i for i in range(N_neurons) if i not in sensory and i not in motor]
VA = [i for i, n in enumerate(neurons) if n.startswith('VA')]
VB = [i for i, n in enumerate(neurons) if n.startswith('VB')]
DA = [i for i, n in enumerate(neurons) if n.startswith('DA')]

print(f"  {N_neurons} neurons × {N_comp} compartments = {N_total} oscillators")
print(f"  {len(chemical_edges)} chemical, {len(gap_edges)} gap junctions")
print(f"  Sensory: {len(sensory)}, Inter: {len(inter)}, Motor: {len(motor)}")

# ════════════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════════════

# Neural dynamics (phase-amplitude-adaptation oscillators)
K_internal = 3.0     # intra-neuron compartment coupling (strong)
K_chem = 0.3         # inter-neuron chemical synapse
K_gap = 0.6          # inter-neuron gap junction
K_field_e = 1.0      # bioelectric field → neuron coupling
K_drive = 0.8        # sensory drive
sigma_noise = 0.12   # neural noise
tau_adapt = 2.0      # adaptation timescale (seconds)
adapt_strength = 0.3 # adaptation → frequency modulation

# Bioelectric field (2D PDE on body surface)
D_E = 0.5            # electric field diffusion coefficient
tau_E = 0.1          # field decay timescale
sigma_E = 0.02       # field noise

# Neuropeptide species
N_peptides = 3
D_P = 0.01           # peptide diffusion (slow — 100× slower than E)
tau_P = 5.0          # peptide degradation timescale
K_peptide = 0.3      # peptide → neuron coupling
release_rate = 0.1   # neural activity → peptide release

# Body mechanics (curvature)
epsilon_body = 4.0   # elastic restoring force (spring constant)
gamma_body = 2.0     # viscous damping
K_muscle = 1.0       # motor neuron → muscle force
sigma_body = 0.01    # mechanical noise

# Coherence field (Landau-Ginzburg)
a_LG = 0.5
b_LG = 2.0
kappa_phi = 0.02
sigma_phi = 0.35
gamma_pump = 0.7
tau_phi = 0.5
K_phi_neural = 1.2   # coherence → neural coupling gating

# Metabolic field
m_0 = 1.0
tau_m = 4.0
c_depletion_phi = 0.1   # coherence costs energy
c_depletion_neural = 0.05  # neural activity costs energy
sigma_m = 0.03

# Environment
n_sensors = min(len(sensory), 50)
tau_env = 2.0
sigma_env = 0.5

np.random.seed(42)

print(f"\n  Total state variables: ~{N_total*3 + Nx_surface*Ny_surface + N_peptides*Nx_body + 3*Nx_body:,}")

# ════════════════════════════════════════════════════════════════════
# PRECOMPUTE
# ════════════════════════════════════════════════════════════════════

# Compartment layout: [soma, axon_ant, axon_post, dendrite_1, dendrite_2]
# Internal coupling: soma ↔ all others (star topology)
# comp_idx(neuron_i, comp_j) = neuron_i * N_comp + comp_j
def comp_idx(ni, ci):
    return ni * N_comp + ci

# Frequency assignment — per compartment
rng_freq = np.random.RandomState(99)
omegas = np.zeros(N_total)
for ni in range(N_neurons):
    if ni in sensory:
        base = 2 * np.pi * (0.9 + 0.3 * rng_freq.rand())
    elif ni in motor:
        base = 2 * np.pi * (1.0 + 0.3 * rng_freq.rand())
    else:
        base = 2 * np.pi * (0.7 + 0.3 * rng_freq.rand())
    for ci in range(N_comp):
        # Compartments have slightly different frequencies (±10%)
        omegas[comp_idx(ni, ci)] = base * (1.0 + 0.1 * (rng_freq.rand() - 0.5))

# Spatial distance for field coupling (neuron-neuron on body surface)
dist_1d = np.abs(pos_1d[:, None] - pos_1d[None, :])
W_spatial = np.exp(-dist_1d**2 / (2 * 0.1**2))
np.fill_diagonal(W_spatial, 0)
W_spatial /= np.maximum(W_spatial.sum(axis=1, keepdims=True), 1)

# Neuron-to-body-grid weights
sigma_sp = 0.08
neuron_w_body = np.zeros((Nx_body, N_neurons))
for gi in range(Nx_body):
    for ni in range(N_neurons):
        neuron_w_body[gi, ni] = np.exp(-(grid_x[gi] - pos_1d[ni])**2 / (2*sigma_sp**2))
    s = neuron_w_body[gi].sum()
    if s > 0: neuron_w_body[gi] /= s

# Interpolation for fields at neuron positions
n_body_left = np.clip(np.floor(pos_1d * (Nx_body - 1)).astype(int), 0, Nx_body - 2)
n_body_frac = pos_1d * (Nx_body - 1) - n_body_left

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

# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def interp_1d(field, left, frac):
    return field[left] * (1 - frac) + field[left + 1] * frac

def laplacian_1d(f, dx):
    lap = np.zeros_like(f)
    lap[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    lap[0] = (f[1] - f[0]) / dx**2
    lap[-1] = (f[-2] - f[-1]) / dx**2
    return lap

def laplacian_2d(f, dx, dy):
    """2D Laplacian with Neumann boundaries on a (Nx, Ny) grid."""
    lap = np.zeros_like(f)
    # Interior
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2 +
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
    )
    # Edges (one-sided in the boundary direction)
    lap[0, 1:-1] = (f[1, 1:-1] - f[0, 1:-1]) / dx**2 + (f[0, 2:] - 2*f[0, 1:-1] + f[0, :-2]) / dy**2
    lap[-1, 1:-1] = (f[-2, 1:-1] - f[-1, 1:-1]) / dx**2 + (f[-1, 2:] - 2*f[-1, 1:-1] + f[-1, :-2]) / dy**2
    lap[1:-1, 0] = (f[2:, 0] - 2*f[1:-1, 0] + f[:-2, 0]) / dx**2 + (f[1:-1, 1] - f[1:-1, 0]) / dy**2
    lap[1:-1, -1] = (f[2:, -1] - 2*f[1:-1, -1] + f[:-2, -1]) / dx**2 + (f[1:-1, -2] - f[1:-1, -1]) / dy**2
    return lap

def soma_phases(theta):
    """Extract soma phases (compartment 0) for all neurons."""
    return theta[::N_comp]

def soma_cos(theta):
    return np.cos(theta[::N_comp])

# ════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ════════════════════════════════════════════════════════════════════
print("\nInitializing state...")

rng_init = np.random.RandomState(77)

# Neural state: phase, amplitude, adaptation
theta = 2 * np.pi * rng_init.rand(N_total)      # phases
amp = np.ones(N_total) * 0.5 + 0.1 * rng_init.randn(N_total)  # amplitudes
adapt = np.zeros(N_total)                         # adaptation (starts at 0)

# Bioelectric field (2D surface)
E_field = np.zeros((Nx_surface, Ny_surface))

# Neuropeptide fields (1D body axis × 3 species)
P_fields = np.zeros((N_peptides, Nx_body))

# Body curvature and velocity
kappa = np.zeros(Nx_body)
kappa_dot = np.zeros(Nx_body)

# Coherence field
phi = np.ones(Nx_body) * 0.05
phi[Nx_body//4:3*Nx_body//4] = 0.15  # seed mid-body

# Metabolic field
m_field = np.ones(Nx_body) * m_0

# Random number generators
rng_n = np.random.RandomState(123)
rng_e = np.random.RandomState(234)
rng_p = np.random.RandomState(345)
rng_phi = np.random.RandomState(456)
rng_m = np.random.RandomState(567)
rng_body = np.random.RandomState(678)

# ════════════════════════════════════════════════════════════════════
# STORAGE
# ════════════════════════════════════════════════════════════════════
n_stored = (n_steps - n_trans) // subsample
print(f"  Will store {n_stored:,} timepoints")

# Store compressed summaries, not full state (too large)
phi_hist = np.zeros((n_stored, Nx_body))
kappa_hist = np.zeros((n_stored, Nx_body))
m_hist = np.zeros((n_stored, Nx_body))
E_mean_hist = np.zeros((n_stored, Nx_surface))  # circumferential average
P_hist = np.zeros((n_stored, N_peptides, Nx_body))
soma_phase_hist = np.zeros((n_stored, N_neurons))
soma_amp_hist = np.zeros((n_stored, N_neurons))
order_r_hist = np.zeros(n_stored)
si = 0

report_interval = max(1, n_steps // 50)

# ════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ════════════════════════════════════════════════════════════════════
print("\nRunning simulation...")
print(f"  Estimated runtime: 4-8 hours")
print(f"  Progress reports every {report_interval*dt:.1f}s of sim time\n")

for t in range(1, n_steps):
    # ── FIELDS AT NEURON POSITIONS ──────────────────────────────
    phi_at_n = interp_1d(phi, n_body_left, n_body_frac)
    phi_clipped = np.clip(phi_at_n, 0, 1)
    m_at_n = interp_1d(m_field, n_body_left, n_body_frac)

    # E field at neuron positions (from 2D surface)
    E_at_n = np.zeros(N_neurons)
    for ni in range(N_neurons):
        E_at_n[ni] = E_field[neuron_surf_x[ni], neuron_surf_y[ni]]

    # Peptide fields at neuron positions
    P_at_n = np.zeros((N_peptides, N_neurons))
    for pk in range(N_peptides):
        P_at_n[pk] = interp_1d(P_fields[pk], n_body_left, n_body_frac)

    # Body curvature at neuron positions
    kappa_at_n = interp_1d(kappa, n_body_left, n_body_frac)

    # ── SENSORY DRIVE ───────────────────────────────────────────
    drive_n = np.zeros(N_neurons)
    for k, idx in enumerate(sensory[:n_sensors]):
        drive_n[idx] = K_drive * sensory_input[t, k]

    # ── SOMA-SOMA COUPLING (connectome) ─────────────────────────
    # Only soma compartments (comp 0) couple between neurons
    soma_th = theta[::N_comp]  # shape (N_neurons,)
    sin_diff_soma = np.sin(soma_th[:, None] - soma_th[None, :])

    chem_n = K_chem * np.sum(W_chem_norm * sin_diff_soma, axis=0)
    gap_n = K_gap * np.sum(W_gap_norm * sin_diff_soma, axis=0)

    # ── FIELD-MEDIATED COUPLING (gated by φ) ────────────────────
    phi_gate = phi_clipped[:, None] * phi_clipped[None, :]
    field_n = K_phi_neural * np.sum(W_spatial * phi_gate * sin_diff_soma, axis=0)

    # ── BIOELECTRIC FIELD COUPLING ──────────────────────────────
    # E field provides additional phase forcing
    e_coupling_n = K_field_e * E_at_n * np.cos(soma_th)

    # ── NEUROPEPTIDE MODULATION ─────────────────────────────────
    # Peptides modulate intrinsic frequency (slow neuromodulation)
    peptide_mod_n = np.zeros(N_neurons)
    for pk in range(N_peptides):
        peptide_mod_n += K_peptide * P_at_n[pk]

    # ── COMPARTMENTAL DYNAMICS ──────────────────────────────────
    dtheta = np.zeros(N_total)
    damp = np.zeros(N_total)
    dadapt = np.zeros(N_total)

    for ni in range(N_neurons):
        base = ni * N_comp
        soma = base  # compartment 0

        # Intrinsic frequency (modulated by adaptation and peptides)
        for ci in range(N_comp):
            idx = base + ci
            omega_mod = omegas[idx] - adapt_strength * adapt[idx] + peptide_mod_n[ni]
            dtheta[idx] = omega_mod

            # Internal coupling: each process couples to soma
            if ci > 0:
                dtheta[idx] += K_internal * amp[soma] * np.sin(theta[soma] - theta[idx])
                dtheta[soma] += K_internal * amp[idx] * np.sin(theta[idx] - theta[soma]) / N_comp

            # Amplitude dynamics: driven by coupling strength
            damp[idx] = (0.5 - amp[idx]) / 0.5  # relax to 0.5

            # Adaptation: slow, activity-dependent
            dadapt[idx] = (-adapt[idx] + amp[idx] * (1 + np.cos(theta[idx]))) / tau_adapt

        # External couplings (soma only)
        dtheta[soma] += drive_n[ni] + chem_n[ni] + gap_n[ni] + field_n[ni] + e_coupling_n[ni]

        # Proprioceptive feedback: motor neurons sense body curvature
        if ni in motor:
            dtheta[soma] += 0.3 * np.sin(kappa_at_n[ni] - theta[soma])

    # Neural noise
    dtheta += sigma_noise * rng_n.randn(N_total) / np.sqrt(dt)

    # Update neural state
    theta += dtheta * dt
    amp += damp * dt
    amp = np.clip(amp, 0.01, 2.0)
    adapt += dadapt * dt

    # ── BIOELECTRIC FIELD UPDATE (2D PDE) ───────────────────────
    # Sources: neural activity creates local field
    E_source = np.zeros((Nx_surface, Ny_surface))
    for ni in range(N_neurons):
        sx, sy = neuron_surf_x[ni], neuron_surf_y[ni]
        E_source[sx, sy] += amp[ni * N_comp] * np.cos(theta[ni * N_comp])

    lap_E = laplacian_2d(E_field, dx_surf, dy_surf)
    dE = D_E * lap_E - E_field / tau_E + E_source
    dE = np.clip(dE, -20, 20)
    E_field += dE * dt + sigma_E * rng_e.randn(Nx_surface, Ny_surface) * np.sqrt(dt)
    E_field = np.clip(E_field, -5, 5)

    # ── NEUROPEPTIDE FIELD UPDATE (1D PDE × 3 species) ──────────
    for pk in range(N_peptides):
        # Release: specific neuron classes release specific peptides
        release = np.zeros(Nx_body)
        if pk == 0:  # peptide 0: released by sensory neurons
            for ni in sensory:
                gi = neuron_body_idx[ni]
                release[gi] += release_rate * (1 + np.cos(theta[ni * N_comp]))
        elif pk == 1:  # peptide 1: released by interneurons
            for ni in inter[:50]:  # subset
                gi = neuron_body_idx[ni]
                release[gi] += release_rate * (1 + np.cos(theta[ni * N_comp]))
        else:  # peptide 2: released by motor neurons
            for ni in motor:
                gi = neuron_body_idx[ni]
                release[gi] += release_rate * (1 + np.cos(theta[ni * N_comp]))

        lap_P = laplacian_1d(P_fields[pk], dx_body)
        dP = D_P * lap_P + release - P_fields[pk] / tau_P
        dP = np.clip(dP, -5, 5)
        P_fields[pk] += dP * dt
        P_fields[pk] = np.clip(P_fields[pk], 0, 10)

    # ── BODY MECHANICS (damped oscillator for curvature) ────────
    # Muscle force from motor neurons
    F_muscle = np.zeros(Nx_body)
    for ni in motor:
        gi = neuron_body_idx[ni]
        F_muscle[gi] += K_muscle * amp[ni * N_comp] * np.sin(theta[ni * N_comp])

    # Curvature as damped driven oscillator: κ'' = -εκ - γκ' + F + noise
    dkappa_dot = -epsilon_body * kappa - gamma_body * kappa_dot + F_muscle
    dkappa_dot += sigma_body * rng_body.randn(Nx_body) / np.sqrt(dt)
    kappa_dot += dkappa_dot * dt
    kappa += kappa_dot * dt
    kappa = np.clip(kappa, -3, 3)

    # ── COHERENCE FIELD UPDATE ──────────────────────────────────
    dVdphi = -a_LG * phi + b_LG * phi**3
    lap_phi = laplacian_1d(phi, dx_body)

    # Source: local neural synchrony (from soma phases)
    source_phi = np.zeros(Nx_body)
    for gi in range(Nx_body):
        w = neuron_w_body[gi]
        mask = w > 1e-6
        if mask.sum() < 2: continue
        wm = w[mask]
        r_local = np.abs(np.sum(wm * np.exp(1j * soma_th[mask])) / wm.sum())
        source_phi[gi] = gamma_pump * r_local

    # Metabolic gating
    source_phi *= m_field

    dphi_det = (-dVdphi + kappa_phi * lap_phi + source_phi) / tau_phi
    dphi_det = np.clip(dphi_det, -10, 10)
    phi += dphi_det * dt + (sigma_phi / np.sqrt(tau_phi)) * rng_phi.randn(Nx_body) * np.sqrt(dt)
    phi = np.clip(phi, 0, 1.5)

    # ── METABOLIC FIELD UPDATE ──────────────────────────────────
    # Depletion from coherence + neural activity
    neural_activity = np.zeros(Nx_body)
    for ni in range(N_neurons):
        gi = neuron_body_idx[ni]
        neural_activity[gi] += amp[ni * N_comp]**2

    dm = (m_0 - m_field) / tau_m - c_depletion_phi * phi**2 - c_depletion_neural * neural_activity / N_neurons
    m_field += dm * dt + sigma_m * rng_m.randn(Nx_body) * np.sqrt(dt)
    m_field = np.clip(m_field, 0.01, m_0 * 1.2)

    # ── STORE ───────────────────────────────────────────────────
    if t >= n_trans and (t - n_trans) % subsample == 0 and si < n_stored:
        phi_hist[si] = phi.copy()
        kappa_hist[si] = kappa.copy()
        m_hist[si] = m_field.copy()
        E_mean_hist[si] = E_field.mean(axis=1)  # circumferential average
        P_hist[si] = P_fields.copy()
        soma_phase_hist[si] = soma_th.copy()
        soma_amp_hist[si] = amp[::N_comp].copy()
        order_r_hist[si] = np.abs(np.mean(np.exp(1j * soma_th)))
        si += 1

    # ── PROGRESS REPORT ─────────────────────────────────────────
    if t % report_interval == 0:
        pct = 100 * t / n_steps
        elapsed = time.time() - wall_start
        eta = elapsed / (t / n_steps) - elapsed if t > 0 else 0
        mp = phi.mean()
        r = np.abs(np.mean(np.exp(1j * soma_th)))
        mm = m_field.mean()
        mk = np.std(kappa)
        print(f"  [{pct:5.1f}%] t={t*dt:.0f}s  φ={mp:.3f}  r={r:.3f}  m={mm:.3f}  "
              f"κ_std={mk:.3f}  wall={elapsed/60:.1f}m  ETA={eta/60:.0f}m")

n_stored = si

# ════════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

t_stored = np.arange(n_stored) * dt_stored
wall_total = time.time() - wall_start

# Motor coordination
def mcorr(phases, A, B):
    if not A or not B: return 0.0
    a = np.mean(np.cos(phases[:, A]), axis=1)
    b = np.mean(np.cos(phases[:, B]), axis=1)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10: return 0.0
    return pearsonr(a, b)[0]

va_vb = mcorr(soma_phase_hist[:n_stored], VA, VB)
va_da = mcorr(soma_phase_hist[:n_stored], VA, DA)

# Mean profiles
mean_phi = phi_hist[:n_stored].mean(axis=0)
mean_m = m_hist[:n_stored].mean(axis=0)
mean_kappa_std = np.std(kappa_hist[:n_stored], axis=0)

print(f"\n  Wall time: {wall_total/3600:.2f} hours ({wall_total/60:.1f} min)")
print(f"  Stored timepoints: {n_stored:,}")
print(f"  Mean coherence <φ>: {mean_phi.mean():.3f} ± {mean_phi.std():.3f}")
print(f"  Mean order parameter <r>: {order_r_hist[:n_stored].mean():.3f}")
print(f"  Mean metabolism <m>: {mean_m.mean():.3f} ± {mean_m.std():.3f}")
print(f"  Body curvature σ(κ): {mean_kappa_std.mean():.3f}")
print(f"  Motor coordination: VA-VB={va_vb:.3f}, VA-DA={va_da:.3f}")
pep_str = ", ".join(f"{P_fields[k].mean():.3f}" for k in range(N_peptides))
print(f"  Neuropeptide levels: [{pep_str}]")

# ════════════════════════════════════════════════════════════════════
# FIGURE (12-panel comprehensive)
# ════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 4, hspace=0.45, wspace=0.4)

extent_xt = [0, t_stored[-1] if n_stored > 1 else 1, 0, 1]

# Row 1: Kymographs
# (a) Coherence field
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(phi_hist[:n_stored].T, aspect='auto', origin='lower',
               extent=extent_xt, cmap='inferno', interpolation='bilinear')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
ax.set_title('(a) Coherence φ(x,t)', fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.7)

# (b) Body curvature
ax = fig.add_subplot(gs[0, 1])
kmax = max(0.01, np.abs(kappa_hist[:n_stored]).max() * 0.8)
im = ax.imshow(kappa_hist[:n_stored].T, aspect='auto', origin='lower',
               extent=extent_xt, cmap='RdBu_r', vmin=-kmax, vmax=kmax,
               interpolation='bilinear')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
ax.set_title('(b) Body curvature κ(x,t)')
fig.colorbar(im, ax=ax, shrink=0.7)

# (c) Bioelectric field (circumferential average)
ax = fig.add_subplot(gs[0, 2])
extent_es = [0, t_stored[-1] if n_stored > 1 else 1, 0, 1]
im = ax.imshow(E_mean_hist[:n_stored].T, aspect='auto', origin='lower',
               extent=extent_es, cmap='PiYG', interpolation='bilinear')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
ax.set_title('(c) Bioelectric field ⟨E⟩(x,t)')
fig.colorbar(im, ax=ax, shrink=0.7)

# (d) Metabolic field
ax = fig.add_subplot(gs[0, 3])
im = ax.imshow(m_hist[:n_stored].T, aspect='auto', origin='lower',
               extent=extent_xt, cmap='YlGn', interpolation='bilinear')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
ax.set_title('(d) Metabolism m(x,t)')
fig.colorbar(im, ax=ax, shrink=0.7)

# Row 2: Neuropeptides + profiles
# (e-g) Neuropeptide kymographs
pep_names = ['Sensory peptide', 'Inter peptide', 'Motor peptide']
pep_cmaps = ['Purples', 'Blues', 'Oranges']
for pk in range(N_peptides):
    ax = fig.add_subplot(gs[1, pk])
    im = ax.imshow(P_hist[:n_stored, pk, :].T, aspect='auto', origin='lower',
                   extent=extent_xt, cmap=pep_cmaps[pk], interpolation='bilinear')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Body axis')
    ax.set_title(f'({"efg"[pk]}) {pep_names[pk]}')
    fig.colorbar(im, ax=ax, shrink=0.7)

# (h) Mean coherence + metabolism profiles
ax = fig.add_subplot(gs[1, 3])
ax.plot(grid_x, mean_phi, color='#FF9800', linewidth=2, label='⟨φ⟩')
ax.set_ylabel('⟨φ⟩', color='#FF9800')
ax.set_xlabel('Body axis')
ax2 = ax.twinx()
ax2.plot(grid_x, mean_m, color='#4CAF50', linewidth=2, label='⟨m⟩')
ax2.set_ylabel('⟨m⟩', color='#4CAF50')
ax.set_title('(h) Coherence vs metabolism')
ax.grid(True, alpha=0.2)

# Row 3: Time series
# (i) Order parameter + mean coherence
ax = fig.add_subplot(gs[2, 0])
ax.plot(t_stored[:n_stored], order_r_hist[:n_stored], color='#2196F3', linewidth=0.5, alpha=0.7, label='r')
phi_mean_t = phi_hist[:n_stored].mean(axis=1)
ax.plot(t_stored[:n_stored], phi_mean_t, color='#FF9800', linewidth=0.8, label='⟨φ⟩')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Order param')
ax.set_title('(i) Sync tracks coherence')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# (j) Motor coordination
ax = fig.add_subplot(gs[2, 1])
window = min(200, n_stored // 5)
if window > 10 and VA and VB:
    va_sig = np.mean(np.cos(soma_phase_hist[:n_stored, VA]), axis=1)
    vb_sig = np.mean(np.cos(soma_phase_hist[:n_stored, VB]), axis=1)
    rc = np.zeros(n_stored - window)
    for i in range(len(rc)):
        a, b = va_sig[i:i+window], vb_sig[i:i+window]
        if np.std(a) > 1e-10 and np.std(b) > 1e-10:
            rc[i] = pearsonr(a, b)[0]
    ax.plot(t_stored[window//2:window//2+len(rc)], rc, color='#EF5350', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Time (s)'); ax.set_ylabel('VA-VB corr')
ax.set_title('(j) Motor coordination'); ax.grid(True, alpha=0.2)

# (k) Metabolism over time
ax = fig.add_subplot(gs[2, 2])
m_mean_t = m_hist[:n_stored].mean(axis=1)
ax.plot(t_stored[:n_stored], m_mean_t, color='#4CAF50', linewidth=0.8)
ax.axhline(m_0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (s)'); ax.set_ylabel('⟨m⟩')
ax.set_title('(k) Metabolic state'); ax.grid(True, alpha=0.2)

# (l) Mean soma amplitude
ax = fig.add_subplot(gs[2, 3])
amp_mean_t = soma_amp_hist[:n_stored].mean(axis=1)
ax.plot(t_stored[:n_stored], amp_mean_t, color='#9C27B0', linewidth=0.5)
ax.set_xlabel('Time (s)'); ax.set_ylabel('⟨amplitude⟩')
ax.set_title('(l) Neural amplitude'); ax.grid(True, alpha=0.2)

# Row 4: Summary panels
# (m) Coherence vs D_eff
ax = fig.add_subplot(gs[3, 0])
# Compute local D_eff at a few positions
for gi in range(0, Nx_body, Nx_body // 10):
    mask = np.abs(pos_1d - grid_x[gi]) < 0.08
    if mask.sum() < 3: continue
    local = np.column_stack([np.cos(soma_phase_hist[:n_stored, mask]),
                             np.sin(soma_phase_hist[:n_stored, mask])])
    X = local - local.mean(0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s**2 / len(X); ev = ev[ev > 1e-10]
    deff = (ev.sum())**2 / (ev**2).sum() if len(ev) > 0 else 0
    ax.scatter(mean_phi[gi], deff, c='#2196F3', s=30, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Local ⟨φ⟩'); ax.set_ylabel('Local D_eff')
ax.set_title('(m) Coherence → low D_eff'); ax.grid(True, alpha=0.2)

# (n) Coherence vs coordination
ax = fig.add_subplot(gs[3, 1])
ax.scatter(phi_mean_t[::10], order_r_hist[:n_stored:10],
           c=t_stored[:n_stored:10], cmap='viridis', s=5, alpha=0.5)
ax.set_xlabel('⟨φ⟩'); ax.set_ylabel('r')
ax.set_title('(n) φ predicts sync'); ax.grid(True, alpha=0.2)

# (o) Surface tension profile
ax = fig.add_subplot(gs[3, 2])
grad_phi_sq = np.zeros(Nx_body)
for s_i in range(n_stored):
    gp = np.gradient(phi_hist[s_i], dx_body)
    grad_phi_sq += gp**2
grad_phi_sq /= n_stored
ax.fill_between(grid_x, 0, grad_phi_sq, alpha=0.4, color='#E91E63')
ax.plot(grid_x, grad_phi_sq, color='#E91E63', linewidth=1.5)
ax.set_xlabel('Body axis'); ax.set_ylabel('|∇φ|²')
ax.set_title('(o) Surface tension'); ax.grid(True, alpha=0.2)

# (p) Summary stats text
ax = fig.add_subplot(gs[3, 3])
ax.axis('off')
summary = (
    f"Wall time: {wall_total/3600:.2f} hrs\n"
    f"Sim time: {T_total}s\n"
    f"Oscillators: {N_total}\n"
    f"Grid: {Nx_body} (body) + {Nx_surface}×{Ny_surface} (surface)\n\n"
    f"⟨φ⟩ = {mean_phi.mean():.3f} ± {mean_phi.std():.3f}\n"
    f"⟨r⟩ = {order_r_hist[:n_stored].mean():.3f}\n"
    f"⟨m⟩ = {mean_m.mean():.3f}\n"
    f"VA-VB = {va_vb:.3f}\n"
    f"VA-DA = {va_da:.3f}\n"
    f"σ(κ) = {mean_kappa_std.mean():.3f}"
)
ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_title('(p) Summary')

plt.savefig(fig_dir / 'fig_full_field.pdf', bbox_inches='tight')
plt.savefig(fig_dir / 'fig_full_field.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved fig_full_field")

print(f"\nTotal wall time: {wall_total/3600:.2f} hours")
print("Done!")
