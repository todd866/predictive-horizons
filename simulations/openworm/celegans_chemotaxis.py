#!/usr/bin/env python3
"""
Chemotaxis comparison: C. elegans coupling architectures.

Compares navigation performance across 4 coupling architectures
(connectome-only, +ephaptic, +neuropeptide, trilayer) using pirouette
navigation with either kinematic or embodied locomotion.

Two locomotion modes:
  KINEMATIC (default): Fixed speed + heading noise. Body curvature is
    computed but does not drive locomotion. Isolates the navigation
    question from locomotion quality.

  EMBODIED (--embodied): Neural dynamics drive body curvature → D-V
    muscle model → RFT locomotion. Speed is ~0.04 mm/s (vs biological
    0.15-0.25), so sensory gain is higher to compensate for slower
    gradient sampling.

In both modes, navigation comes from reversal modulation: the
NavigationState reads dC/dt from the OdorCircuit and AVA activity
from the neural dynamics to modulate reversal timing (pirouette).

Usage:
    python3 openworm/celegans_chemotaxis.py [--quick] [--embodied]
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from celegans import (
    load_worm_data, build_coupling_matrices, build_grid_mapping,
    WormState, WormParams, assign_frequencies, step,
    MetabolicBudget, build_frustration_matrix,
    NavigationState, WeathervaneCircuit, reversal_turn_angle,
)
from celegans.arena import Arena
from celegans.body import ArticulatedBody
from celegans.sensory import OdorCircuit
from celegans.behavior import TrajectoryRecorder

DATA_DIR = Path(__file__).parent

# ── Arena geometry ─────────────────────────────────────────────────
PLATE_RADIUS = 50.0      # mm
FOOD_X, FOOD_Y = 5.0, 0.0   # food source position
START_X, START_Y = -5.0, 0.0  # worm start (10mm separation)
FOOD_SIGMA = 5.0          # sharp gradient
FOOD_STRENGTH = 1.0

HEADINGS_FULL = np.linspace(0, 2 * np.pi, 8, endpoint=False)
SEEDS_PER_HEADING = 5

# ── Simulation parameters ─────────────────────────────────────────
DT = 0.005             # 5ms timestep (safe for Kuramoto at 1Hz, 5x faster)
T_TOTAL = 300.0
NX = 50
KAPPA_SCALE = 100.0        # body curvature scaling (dynamics → physical mm^-1)
                            # 100 gives stable heading (κAC~0.57); 200 causes wandering
C_N_AGAR = 5.0            # agar surface anisotropy (Fang-Yen et al. 2010)

# ── Kinematic locomotion ──────────────────────────────────────────
CRAWL_SPEED = 0.15         # mm/s (biological: 0.15-0.25)
HEADING_NOISE = 0.1        # rad/sqrt(s) heading diffusion (lower = straighter runs)

# ── Embodied locomotion ──────────────────────────────────────────
# D-V muscle model → RFT. Speed ~0.04 mm/s (limited by Kuramoto wave quality).
EMBODIED_K_MUSCLE = 2.5     # higher than default 0.8 for stronger curvature
EMBODIED_PROPRIO_GAIN = 3.0 # higher than default 1.5 for better wave propagation
EMBODIED_PROPRIO_DS = 0.1   # anterior offset in body-lengths
EMBODIED_KAPPA_SCALE = 150  # higher than kinematic 100 for more physical curvature
EMBODIED_C_N = 10.0         # higher drag anisotropy for more RFT thrust

# ── Sakaguchi-Kuramoto frustration ────────────────────────────────
FRUSTRATION_ALPHA = 0.0    # disabled — doesn't improve wave (spatial coherence from proprio)

# ── Navigation parameters ─────────────────────────────────────────
NAV_BASE_REV_RATE = 2.0 / 60   # 2 reversals/min baseline
NAV_SENSORY_GAIN = 70.0        # exponential gain on dC/dt (kinematic)
                                # gives ~4:1 reversal ratio at model dCdt ≈ 0.01
NAV_SENSORY_GAIN_EMBODIED = 250.0  # higher for slower embodied speed
                                    # (~4x slower → dCdt ~4x smaller)
NAV_TAU_DCDT = 3.0             # dC/dt smoothing timescale (s)
NAV_MIN_STATE_DUR = 1.0        # min dwell in each state (s)
NAV_MEAN_BACK_DUR = 2.0        # mean backward duration (s)
WEATHERVANE_GAIN = 0.0          # disabled — SMD bilateral signal doesn't
                                # track gradient through Kuramoto network

# ── Model configurations ──────────────────────────────────────────
MODELS = [
    ('connectome_only', False, False),
    ('plus_eph',        True,  False),
    ('plus_npp',        False, True),
    ('trilayer',        True,  True),
]


def wave_diagnostics(state, wd):
    """Compute motor phase wave quality and spatial structure metrics."""
    motor_pos = wd.pos_1d[wd.motor]
    motor_sort = np.argsort(motor_pos)
    motor_phases = state.theta[wd.motor][motor_sort]
    motor_positions = motor_pos[motor_sort]

    phases_unwrapped = np.unwrap(motor_phases)
    coeffs = np.polyfit(motor_positions, phases_unwrapped, 1)
    fit = np.polyval(coeffs, motor_positions)
    residual = float(np.std(phases_unwrapped - fit))
    gradient = float(coeffs[0])

    k = state.kappa
    if len(k) > 1 and k.std() > 1e-12:
        kappa_autocorr = float(np.corrcoef(k[:-1], k[1:])[0, 1])
    else:
        kappa_autocorr = 0.0

    if len(k) > 4 and k.std() > 1e-12:
        fft = np.abs(np.fft.rfft(k - k.mean()))
        freqs = np.fft.rfftfreq(len(k), d=1.0 / len(k))
        if len(fft) > 1:
            peak_idx = np.argmax(fft[1:]) + 1
            dominant_wavelength = float(1.0 / freqs[peak_idx]) if freqs[peak_idx] > 0 else 0.0
        else:
            dominant_wavelength = 0.0
    else:
        dominant_wavelength = 0.0

    return {
        'phase_gradient': gradient,
        'phase_residual': residual,
        'kappa_autocorr': kappa_autocorr,
        'dominant_wavelength': dominant_wavelength,
    }


def run_single(model_name, use_eph, use_npp, heading, seed, t_total=T_TOTAL,
               embodied=False):
    """Run one chemotaxis simulation with navigation circuits."""
    wd = load_worm_data(DATA_DIR)
    cm = build_coupling_matrices(wd)
    gm = build_grid_mapping(wd.pos_1d, Nx=NX)
    if embodied:
        params = WormParams(K_muscle=EMBODIED_K_MUSCLE,
                            proprio_gain=EMBODIED_PROPRIO_GAIN,
                            proprio_delta_s=EMBODIED_PROPRIO_DS)
    else:
        params = WormParams()
    omegas = assign_frequencies(wd.N, wd.sensory, wd.motor, wd.inter)
    state = WormState(wd.N, NX, seed=seed)
    n_sensors = min(len(wd.sensory), 50)
    budget = MetabolicBudget()

    motor_classes = {
        'VA': wd.VA, 'VB': wd.VB, 'DA': wd.DA, 'DB': wd.DB,
        'VD': wd.VD, 'DD': wd.DD,
    }

    # Sakaguchi-Kuramoto frustration matrix (motor-motor connections)
    if FRUSTRATION_ALPHA > 0:
        frust_matrix = build_frustration_matrix(
            wd.pos_1d, wd.motor, alpha=FRUSTRATION_ALPHA,
            W_chem=wd.W_chem, W_gap=wd.W_gap)
    else:
        frust_matrix = None

    # Navigation circuits
    sensory_gain = NAV_SENSORY_GAIN_EMBODIED if embodied else NAV_SENSORY_GAIN
    nav = NavigationState(
        base_rev_rate=NAV_BASE_REV_RATE,
        sensory_gain=sensory_gain,
        tau_dCdt=NAV_TAU_DCDT,
        min_state_duration=NAV_MIN_STATE_DUR,
        mean_back_duration=NAV_MEAN_BACK_DUR,
        seed=seed + 10000,  # decouple from dynamics RNG
    )
    weathervane = WeathervaneCircuit(gain=WEATHERVANE_GAIN)

    # Neuron indices for activity readout
    aval_idx = wd.neuron_idx.get('AVAL')
    avar_idx = wd.neuron_idx.get('AVAR')
    smdvl_idx = wd.neuron_idx.get('SMDVL')
    smdvr_idx = wd.neuron_idx.get('SMDVR')

    arena = Arena(
        radius_mm=PLATE_RADIUS,
        sources=[{'x': FOOD_X, 'y': FOOD_Y,
                  'strength': FOOD_STRENGTH, 'sigma': FOOD_SIGMA}]
    )
    body = ArticulatedBody(
        n_segments=NX, body_length_mm=1.0,
        x0=START_X, y0=START_Y, heading0=float(heading),
        kappa_scale=EMBODIED_KAPPA_SCALE if embodied else KAPPA_SCALE,
        C_n=EMBODIED_C_N if embodied else C_N_AGAR,
    )
    circuit = OdorCircuit(wd.neuron_idx, wd.N)
    rec = TrajectoryRecorder()
    dummy_env = np.zeros(n_sensors)
    body_rng = np.random.RandomState(seed + 20000)

    n_steps = int(t_total / DT)
    record_every = max(1, int(0.01 / DT))  # 100 Hz recording
    wave_samples = []
    n_reversals = 0

    t0 = time.time()
    for i in range(n_steps):
        ext_drive = np.zeros(wd.N)

        # ── Odor sensory drive ────────────────────────────────────
        hx, hy = body.head_position()
        lx, ly, rx, ry = body.bilateral_head_points()
        C_left = arena.concentration(lx, ly)
        C_right = arena.concentration(rx, ry)
        sensory_drive = circuit.transduce(C_left, C_right, DT)
        ext_drive += sensory_drive

        # ── Neural dynamics ───────────────────────────────────────
        diag = step(state, params, omegas, cm, gm,
                    wd.sensory, wd.motor, n_sensors, dummy_env, DT,
                    use_eph=use_eph, use_npp=use_npp,
                    budget_scale=budget.budget_scale,
                    external_drive=ext_drive,
                    frustration_matrix=frust_matrix,
                    motor_classes=motor_classes,
                    pos_1d=wd.pos_1d)

        # ── Read circuit neuron activity ──────────────────────────
        activity = 0.5 + 0.5 * np.cos(state.theta)
        ava_act = 0.5 * (activity[aval_idx] + activity[avar_idx])

        # ── Navigation state machine ─────────────────────────────
        nav_state, did_reverse = nav.step(ava_act, circuit.last_dCdt, DT)

        if did_reverse:
            turn = reversal_turn_angle(nav.dCdt_smooth, nav.rng)
            body.reverse(turn)
            n_reversals += 1

        # ── Locomotion ────────────────────────────────────────────
        if embodied:
            body.step(state.kappa, gm['grid_x'], DT)
        else:
            body.kinematic_step(state.kappa, gm['grid_x'], DT,
                                speed=CRAWL_SPEED, heading_noise_std=HEADING_NOISE,
                                rng=body_rng)
        body.enforce_bounds(arena)
        budget.update(DT, **diag)

        # ── Record ────────────────────────────────────────────────
        if i % record_every == 0:
            rec.record(i * DT, body.x_cm, body.y_cm, body.heading,
                       body.velocity_along_body(), body.angular_velocity(),
                       0.5 * (C_left + C_right))

        if i > 2000 and i % 1000 == 0:
            wave_samples.append(wave_diagnostics(state, wd))

    wall_time = time.time() - t0
    record_dt = DT * record_every

    # ── Metrics ───────────────────────────────────────────────────
    ci = rec.chemotaxis_index(FOOD_X, FOOD_Y)
    pe = rec.path_efficiency(FOOD_X, FOOD_Y)
    rr = rec.reversal_rate(record_dt)
    rtg = rec.reversal_triggered_by_gradient(FOOD_X, FOOD_Y, record_dt)
    bearings = rec.bearing_to_food(FOOD_X, FOOD_Y)
    speeds = np.array(rec.velocities)

    if wave_samples:
        mean_wave = {k: float(np.mean([s[k] for s in wave_samples]))
                     for k in wave_samples[0]}
    else:
        mean_wave = {'phase_gradient': 0, 'phase_residual': 0,
                     'kappa_autocorr': 0, 'dominant_wavelength': 0}

    # Displacement-based speed (net displacement / time)
    displacement = np.sqrt((body.x_cm - START_X)**2 + (body.y_cm - START_Y)**2)
    disp_speed = displacement / t_total

    return {
        'model': model_name,
        'heading': float(heading),
        'seed': seed,
        'chemotaxis_index': ci,
        'path_efficiency': pe,
        'reversal_rate': float(n_reversals / (t_total / 60)),
        'reversal_gradient_fraction': rtg,
        'mean_bearing': float(bearings.mean()) if len(bearings) > 0 else 0.0,
        'mean_speed': float(np.abs(speeds).mean()) if len(speeds) > 0 else 0.0,
        'displacement_speed': float(disp_speed),
        'final_x': body.x_cm,
        'final_y': body.y_cm,
        'final_budget': budget.budget_scale,
        'wall_time_s': wall_time,
        'n_reversals': n_reversals,
        **mean_wave,
    }


def main():
    parser = argparse.ArgumentParser(description='C. elegans chemotaxis simulation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 4 headings x 5 seeds per model, 120s')
    parser.add_argument('--embodied', action='store_true',
                        help='Use embodied RFT locomotion (D-V muscle model)')
    args = parser.parse_args()

    sensory_gain_used = NAV_SENSORY_GAIN_EMBODIED if args.embodied else NAV_SENSORY_GAIN

    if args.quick:
        headings = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        seeds_per = 5
        t_total = 120.0
        print("Quick mode: 4 headings, 5 seeds, 120s per model")
    else:
        headings = HEADINGS_FULL
        seeds_per = SEEDS_PER_HEADING
        t_total = T_TOTAL
        total_runs = len(MODELS) * len(headings) * seeds_per
        print(f"Full experiment: {total_runs} runs "
              f"({len(MODELS)} models x {len(headings)} headings x {seeds_per} seeds)")

    mode_str = "EMBODIED (D-V RFT)" if args.embodied else "KINEMATIC"
    print(f"Locomotion: {mode_str}")
    print(f"Arena: food at ({FOOD_X},{FOOD_Y}), start at ({START_X},{START_Y}), "
          f"separation={np.sqrt((FOOD_X-START_X)**2+(FOOD_Y-START_Y)**2):.0f}mm")
    if args.embodied:
        print(f"D-V params: K_muscle={EMBODIED_K_MUSCLE}, proprio_gain={EMBODIED_PROPRIO_GAIN}, "
              f"kappa_scale={EMBODIED_KAPPA_SCALE}, C_n={EMBODIED_C_N}")
    else:
        print(f"Kinematic: speed={CRAWL_SPEED}mm/s, heading_noise={HEADING_NOISE}")
    print(f"Navigation: rev_rate={NAV_BASE_REV_RATE:.3f}/s, "
          f"sensory_gain={sensory_gain_used}, weathervane={WEATHERVANE_GAIN}")

    results = []
    for model_name, use_eph, use_npp in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        for hi, heading in enumerate(headings):
            for seed in range(seeds_per):
                idx = hi * seeds_per + seed + 1
                total = len(headings) * seeds_per
                print(f"  h={heading:.2f} s={seed} [{idx}/{total}]",
                      end=' ... ', flush=True)
                result = run_single(model_name, use_eph, use_npp,
                                    heading, seed, t_total,
                                    embodied=args.embodied)
                print(f"CI={result['chemotaxis_index']:+.4f} "
                      f"v={result['mean_speed']:.5f} "
                      f"rev={result['n_reversals']} "
                      f"κAC={result['kappa_autocorr']:.3f} "
                      f"({result['wall_time_s']:.0f}s)")
                results.append(result)

    # ── Save results with metadata ──────────────────────────────────
    import subprocess
    git_sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
    loco_mode = 'embodied' if args.embodied else 'kinematic'
    meta = {
        'git_sha': git_sha,
        'dt': DT,
        't_total': t_total,
        'n_headings': len(headings),
        'n_seeds': seeds_per,
        'n_models': len(MODELS),
        'total_runs': len(results),
        'locomotion_mode': loco_mode,
        'food_x': FOOD_X,
        'food_y': FOOD_Y,
        'start_x': START_X,
        'start_y': START_Y,
        'food_sigma': FOOD_SIGMA,
        'sensory_gain': sensory_gain_used,
        'base_rev_rate': NAV_BASE_REV_RATE,
        'weathervane_gain': WEATHERVANE_GAIN,
        'frustration_alpha': FRUSTRATION_ALPHA,
    }
    if loco_mode == 'kinematic':
        meta['crawl_speed_mm_s'] = CRAWL_SPEED
        meta['heading_noise_rad_sqrt_s'] = HEADING_NOISE
    else:
        meta['K_muscle'] = EMBODIED_K_MUSCLE
        meta['proprio_gain'] = EMBODIED_PROPRIO_GAIN
        meta['proprio_delta_s'] = EMBODIED_PROPRIO_DS
        meta['kappa_scale'] = EMBODIED_KAPPA_SCALE
        meta['C_n'] = EMBODIED_C_N
    output = {
        'metadata': meta,
        'runs': results,
    }
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    suffix = '_embodied' if args.embodied else ''
    out_path = out_dir / f'chemotaxis_results{suffix}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'Model':<18} {'CI':>7} {'Speed':>9} {'κ AC':>7} "
          f"{'Resid':>7} {'Rev/min':>8} {'Budget':>7}")
    print('-' * 70)
    for model_name, _, _ in MODELS:
        mr = [r for r in results if r['model'] == model_name]
        if mr:
            cis = [r['chemotaxis_index'] for r in mr]
            se = float(np.std(cis) / np.sqrt(len(cis)))
            pos = sum(1 for c in cis if c > 0)
            print(f"{model_name:<18} "
                  f"{np.mean(cis):>+7.4f}±{se:.3f} "
                  f"[{pos}/{len(cis)}+] "
                  f"{np.mean([r['reversal_rate'] for r in mr]):>6.1f}r/m "
                  f"{np.mean([r['kappa_autocorr'] for r in mr]):>6.3f}κ")


if __name__ == '__main__':
    main()
