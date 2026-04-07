#!/usr/bin/env python3
"""
Chemotaxis simulation: C. elegans navigating an odor gradient.

Runs 4 model configurations x N_headings x N_seeds on a reduced-distance
chemotaxis assay. Compares navigation performance across coupling
architectures using directional proprioception on the axial locomotor chain.

Locomotor scaffold:
  - Directional proprio on VB/DB/VA/DA (body-wall locomotor chain)
  - Cancels built-in local proprio on those neurons, replaces with
    anterior-shifted proprio through external_drive
  - C_n = 5.0 (agar surface anisotropy)
  - No frequency gradient (tested, does not improve propagation)

Limitations:
  - The model produces locally correlated undulation, not a true
    head-to-tail traveling wave (confirmed by propagation diagnostics)
  - Behavioral differences emerge in a sub-wave locomotor regime

Usage:
    python3 openworm/celegans_chemotaxis.py [--quick]
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
    MetabolicBudget,
)
from celegans.arena import Arena
from celegans.body import ArticulatedBody
from celegans.sensory import OdorCircuit
from celegans.behavior import TrajectoryRecorder

DATA_DIR = Path(__file__).parent

# ── Arena geometry (reduced separation) ───────────────────────────
PLATE_RADIUS = 50.0      # mm
FOOD_X, FOOD_Y = 7.5, 0.0   # food source position
START_X, START_Y = -7.5, 0.0  # worm start (15mm separation)
FOOD_SIGMA = 15.0
FOOD_STRENGTH = 1.0

HEADINGS_FULL = np.linspace(0, 2 * np.pi, 8, endpoint=False)
SEEDS_PER_HEADING = 5

# ── Simulation parameters ─────────────────────────────────────────
DT = 0.001
T_TOTAL = 300.0
NX = 50
KAPPA_SCALE = 100.0
C_N_AGAR = 5.0            # agar surface anisotropy (Fang-Yen et al. 2010)

# ── Directional proprioception ────────────────────────────────────
PROPRIO_GAIN = 1.5         # anterior-shifted proprioceptive gain
PROPRIO_DELTA_S = 0.08     # anterior offset in body-lengths

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


def run_single(model_name, use_eph, use_npp, heading, seed, t_total=T_TOTAL):
    """Run one chemotaxis simulation with directional proprioception."""
    wd = load_worm_data(DATA_DIR)
    cm = build_coupling_matrices(wd)
    gm = build_grid_mapping(wd.pos_1d, Nx=NX)
    params = WormParams()
    omegas = assign_frequencies(wd.N, wd.sensory, wd.motor, wd.inter)
    state = WormState(wd.N, NX, seed=seed)
    n_sensors = min(len(wd.sensory), 50)
    budget = MetabolicBudget()

    # Axial locomotor chain: VB + DB + VA + DA
    loco_chain = np.concatenate([
        np.array(wd.VB), np.array(wd.DB),
        np.array(wd.VA), np.array(wd.DA),
    ])

    arena = Arena(
        radius_mm=PLATE_RADIUS,
        sources=[{'x': FOOD_X, 'y': FOOD_Y,
                  'strength': FOOD_STRENGTH, 'sigma': FOOD_SIGMA}]
    )
    body = ArticulatedBody(
        n_segments=NX, body_length_mm=1.0,
        x0=START_X, y0=START_Y, heading0=float(heading),
        kappa_scale=KAPPA_SCALE, C_n=C_N_AGAR,
    )
    circuit = OdorCircuit(wd.neuron_idx, wd.N)
    rec = TrajectoryRecorder()
    dummy_env = np.zeros(n_sensors)

    n_steps = int(t_total / DT)
    record_every = max(1, int(0.01 / DT))  # 100 Hz recording
    wave_samples = []

    t0 = time.time()
    for i in range(n_steps):
        # ── Directional proprioception on locomotor chain ─────────
        # Cancel built-in local proprio, add anterior-shifted proprio
        ext_drive = np.zeros(wd.N)
        kappa_local = np.interp(
            wd.pos_1d[loco_chain], gm['grid_x'], state.kappa)
        kappa_ant = np.interp(
            np.clip(wd.pos_1d[loco_chain] - PROPRIO_DELTA_S, 0.0, 1.0),
            gm['grid_x'], state.kappa)
        local_prop = 0.3 * np.sin(kappa_local - state.theta[loco_chain])
        dir_prop = PROPRIO_GAIN * np.sin(kappa_ant - state.theta[loco_chain])
        ext_drive[loco_chain] += dir_prop - local_prop

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
                    external_drive=ext_drive)

        # ── Body + arena + budget ─────────────────────────────────
        body.step(state.kappa, gm['grid_x'], DT)
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

    return {
        'model': model_name,
        'heading': float(heading),
        'seed': seed,
        'chemotaxis_index': ci,
        'path_efficiency': pe,
        'reversal_rate': rr,
        'reversal_gradient_fraction': rtg,
        'mean_bearing': float(bearings.mean()) if len(bearings) > 0 else 0.0,
        'mean_speed': float(np.abs(speeds).mean()) if len(speeds) > 0 else 0.0,
        'final_x': body.x_cm,
        'final_y': body.y_cm,
        'final_budget': budget.budget_scale,
        'wall_time_s': wall_time,
        **mean_wave,
    }


def main():
    parser = argparse.ArgumentParser(description='C. elegans chemotaxis simulation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 2 headings x 1 seed per model, 60s')
    args = parser.parse_args()

    if args.quick:
        headings = [0.0, np.pi]
        seeds_per = 1
        t_total = 60.0
        print("Quick mode: 2 headings, 1 seed, 60s per model")
    else:
        headings = HEADINGS_FULL
        seeds_per = SEEDS_PER_HEADING
        t_total = T_TOTAL
        total_runs = len(MODELS) * len(headings) * seeds_per
        print(f"Full experiment: {total_runs} runs "
              f"({len(MODELS)} models x {len(headings)} headings x {seeds_per} seeds)")

    print(f"Arena: food at ({FOOD_X},{FOOD_Y}), start at ({START_X},{START_Y}), "
          f"separation={np.sqrt((FOOD_X-START_X)**2+(FOOD_Y-START_Y)**2):.0f}mm")
    print(f"Proprio: gain={PROPRIO_GAIN}, delta_s={PROPRIO_DELTA_S}, C_n={C_N_AGAR}")

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
                                    heading, seed, t_total)
                print(f"CI={result['chemotaxis_index']:+.4f} "
                      f"v={result['mean_speed']:.5f} "
                      f"κAC={result['kappa_autocorr']:.3f} "
                      f"({result['wall_time_s']:.0f}s)")
                results.append(result)

    # ── Save results ──────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'chemotaxis_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'Model':<18} {'CI':>7} {'Speed':>9} {'κ AC':>7} "
          f"{'Resid':>7} {'RevRate':>8} {'Budget':>7}")
    print('-' * 70)
    for model_name, _, _ in MODELS:
        mr = [r for r in results if r['model'] == model_name]
        if mr:
            print(f"{model_name:<18} "
                  f"{np.mean([r['chemotaxis_index'] for r in mr]):>+7.4f} "
                  f"{np.mean([r['mean_speed'] for r in mr]):>9.5f} "
                  f"{np.mean([r['kappa_autocorr'] for r in mr]):>7.3f} "
                  f"{np.mean([r['phase_residual'] for r in mr]):>7.2f} "
                  f"{np.mean([r['reversal_rate'] for r in mr]):>8.1f} "
                  f"{np.mean([r['final_budget'] for r in mr]):>7.3f}")


if __name__ == '__main__':
    main()
