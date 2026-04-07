"""Smoke tests for the celegans package. Run with: python3 -m pytest tests/ -v"""

import numpy as np
import sys
from pathlib import Path

# Add simulations dir to path
sim_dir = Path(__file__).parent.parent
sys.path.insert(0, str(sim_dir))

DATA_DIR = sim_dir / "openworm"


def test_data_loading():
    from celegans.data import load_worm_data
    wd = load_worm_data(DATA_DIR)
    assert wd.N == 448
    assert len(wd.sensory) > 50
    assert len(wd.motor) > 50
    assert len(wd.VA) > 0
    assert wd.W_chem.shape == (448, 448)
    assert (wd.W_npp > 0).sum() > 1000


def test_coupling_matrices():
    from celegans.data import load_worm_data
    from celegans.coupling import build_coupling_matrices, build_grid_mapping
    wd = load_worm_data(DATA_DIR)
    cm = build_coupling_matrices(wd)
    assert cm.W_chem_norm.shape == (448, 448)
    assert cm.W_eph_norm.max() <= 1.0
    assert cm.W_npp_norm.max() <= 1.0 + 1e-10

    gm = build_grid_mapping(wd.pos_1d, Nx=100)
    assert gm['grid_x'].shape == (100,)
    assert gm['neuron_w_grid'].shape == (100, 448)


def test_environment_generation():
    from celegans.environment import (
        generate_multimode_env, default_taus, default_amplitudes
    )
    taus = default_taus(K=5)
    assert len(taus) == 5
    amps = default_amplitudes(taus)
    assert np.all(np.diff(amps) > 0)

    env, modes, _ = generate_multimode_env(
        5000, 0.01, 4, np.array([0.1, 1.0, 10.0]), seed=42
    )
    assert env.shape == (5000, 4)
    assert len(modes) == 3
    # Slow modes should have larger variance when amplitudes increase with tau.
    assert modes[0].std() < modes[-1].std()


def test_dynamics_100_steps():
    from celegans.data import load_worm_data
    from celegans.coupling import build_coupling_matrices, build_grid_mapping
    from celegans.dynamics import WormState, WormParams, assign_frequencies, step

    wd = load_worm_data(DATA_DIR)
    cm = build_coupling_matrices(wd)
    gm = build_grid_mapping(wd.pos_1d, Nx=50)
    params = WormParams()
    omegas = assign_frequencies(wd.N, wd.sensory, wd.motor, wd.inter)
    state = WormState(wd.N, 50)
    n_sensors = min(len(wd.sensory), 50)

    rng = np.random.RandomState(42)
    initial_theta = state.theta.copy()
    for t in range(100):
        env_sig = 0.5 * rng.randn(n_sensors)
        diag = step(state, params, omegas, cm, gm,
                    wd.sensory, wd.motor, n_sensors, env_sig, dt=0.001)

    assert diag['order_r'] > 0
    assert not np.allclose(state.theta, initial_theta)


def test_analysis_d_eff():
    from celegans.analysis import d_eff, classify_trajectory

    # Random data: D_eff ~ D
    X = np.random.randn(500, 10)
    assert abs(d_eff(X) - 10) < 2

    # Low-rank data: D_eff ~ rank
    X_low = np.random.randn(500, 3) @ np.random.randn(3, 10)
    assert abs(d_eff(X_low) - 3) < 1.5

    # Trajectory classification
    assert classify_trajectory(np.array([10, 9, 8, 9, 10, 9.5])) == 'resilient'
    assert classify_trajectory(np.array([10, 8, 6, 4, 2, 1])) == 'terminal'


def test_budget():
    from celegans.budget import MetabolicBudget, UnlimitedBudget

    b = MetabolicBudget(R_max=100, E_in=5.0)
    assert b.budget_scale == 1.0
    assert not b.depleted

    u = UnlimitedBudget()
    assert u.budget_scale == 1.0
    assert u.update(0.01) == 1.0


def test_full_integration():
    """All modules working together."""
    from celegans import (
        load_worm_data, build_coupling_matrices, build_grid_mapping,
        WormState, WormParams, assign_frequencies, step,
        generate_multimode_env, default_taus,
        d_eff, MetabolicBudget,
    )

    wd = load_worm_data(DATA_DIR)
    cm = build_coupling_matrices(wd)
    gm = build_grid_mapping(wd.pos_1d, Nx=50)
    params = WormParams()
    omegas = assign_frequencies(wd.N, wd.sensory, wd.motor, wd.inter)
    state = WormState(wd.N, 50)
    budget = MetabolicBudget()
    n_sensors = min(len(wd.sensory), 50)

    taus = default_taus(K=3)
    env, _, _ = generate_multimode_env(300, 0.001, n_sensors, taus)

    for t in range(200):
        diag = step(state, params, omegas, cm, gm,
                    wd.sensory, wd.motor, n_sensors, env[t], dt=0.001,
                    budget_scale=budget.budget_scale)
        budget.update(0.001, **diag)

    assert diag['order_r'] > 0
    assert diag['mean_phi'] > 0
    assert budget.budget_scale > 0


def test_arena_concentration():
    from celegans.arena import Arena
    arena = Arena(
        radius_mm=50.0,
        sources=[{'x': 40.0, 'y': 0.0, 'strength': 1.0, 'sigma': 15.0}]
    )
    c_at_source = arena.concentration(40.0, 0.0)
    c_far = arena.concentration(-40.0, 0.0)
    assert c_at_source > 0.9
    assert c_far < 0.1
    gx, gy = arena.gradient(0.0, 0.0)
    assert gx > 0


def test_arena_bounds():
    from celegans.arena import Arena
    arena = Arena(radius_mm=50.0, sources=[])
    assert arena.in_bounds(0.0, 0.0)
    assert arena.in_bounds(49.0, 0.0)
    assert not arena.in_bounds(51.0, 0.0)
    assert not arena.in_bounds(35.0, 36.0)


def test_body_straight():
    """A worm with zero curvature should be a straight line."""
    from celegans.body import ArticulatedBody
    body = ArticulatedBody(
        n_segments=50, body_length_mm=1.0,
        x0=0.0, y0=0.0, heading0=0.0
    )
    kappa_grid = np.zeros(50)
    grid_x = np.linspace(0, 1, 50)
    body.set_curvature(kappa_grid, grid_x)
    pos = body.segment_positions()
    assert pos.shape == (50, 2)
    assert np.allclose(pos[:, 1], 0.0, atol=1e-10)
    assert abs(pos[-1, 0] - pos[0, 0]) > 0.9


def test_body_curved():
    """Constant positive curvature should produce a circular arc."""
    from celegans.body import ArticulatedBody
    body = ArticulatedBody(
        n_segments=50, body_length_mm=1.0,
        x0=0.0, y0=0.0, heading0=0.0, kappa_scale=1.0
    )
    kappa_grid = np.full(50, 2 * np.pi)
    grid_x = np.linspace(0, 1, 50)
    body.set_curvature(kappa_grid, grid_x)
    pos = body.segment_positions()
    dist = np.sqrt((pos[-1, 0] - pos[0, 0])**2 + (pos[-1, 1] - pos[0, 1])**2)
    assert dist < 0.15


def test_body_head_position():
    from celegans.body import ArticulatedBody
    body = ArticulatedBody(
        n_segments=50, body_length_mm=1.0,
        x0=10.0, y0=5.0, heading0=np.pi / 4
    )
    kappa_grid = np.zeros(50)
    grid_x = np.linspace(0, 1, 50)
    body.set_curvature(kappa_grid, grid_x)
    hx, hy = body.head_position()
    assert hx > 10.0
    assert hy > 5.0


def test_body_bilateral_points():
    from celegans.body import ArticulatedBody
    body = ArticulatedBody(
        n_segments=50, body_length_mm=1.0,
        x0=0.0, y0=0.0, heading0=0.0
    )
    kappa_grid = np.zeros(50)
    grid_x = np.linspace(0, 1, 50)
    body.set_curvature(kappa_grid, grid_x)
    lx, ly, rx, ry = body.bilateral_head_points()
    hx, hy = body.head_position()
    assert abs(lx - rx) < 1e-10
    assert ly > hy
    assert ry < hy
    assert abs(ly - hy - 0.05) < 1e-10


def test_body_no_shape_change_no_movement():
    """Static body shape should produce zero velocity."""
    from celegans.body import ArticulatedBody
    body = ArticulatedBody(n_segments=50, body_length_mm=1.0,
                           x0=0.0, y0=0.0, heading0=0.0)
    kappa = np.zeros(50)
    grid_x = np.linspace(0, 1, 50)
    body.step(kappa, grid_x, dt=0.001)
    x0, y0_pos = body.x_cm, body.y_cm
    body.step(kappa, grid_x, dt=0.001)
    assert abs(body.x_cm - x0) < 1e-10
    assert abs(body.y_cm - y0_pos) < 1e-10


def test_body_traveling_wave_produces_forward_motion():
    """A traveling curvature wave should propel the worm forward."""
    from celegans.body import ArticulatedBody
    body = ArticulatedBody(n_segments=50, body_length_mm=1.0,
                           x0=0.0, y0=0.0, heading0=0.0,
                           C_t=1.0, C_n=1.5, kappa_scale=1.0)
    grid_x = np.linspace(0, 1, 50)
    dt = 0.001
    freq = 1.0
    wavelength = 1.5
    amplitude = 3.0

    for i in range(2000):
        t = i * dt
        s = np.linspace(0, 1, 50)
        kappa = amplitude * np.sin(2 * np.pi * (s * body.L / wavelength - freq * t))
        body.step(kappa, grid_x, dt)

    assert body.x_cm > 0.05


def test_body_anisotropy_matters():
    """With C_n = C_t (isotropic drag), traveling wave produces no net motion."""
    from celegans.body import ArticulatedBody
    body_iso = ArticulatedBody(n_segments=50, body_length_mm=1.0,
                               x0=0.0, y0=0.0, heading0=0.0,
                               C_t=1.0, C_n=1.0, kappa_scale=1.0)
    body_aniso = ArticulatedBody(n_segments=50, body_length_mm=1.0,
                                 x0=0.0, y0=0.0, heading0=0.0,
                                 C_t=1.0, C_n=1.5, kappa_scale=1.0)
    grid_x = np.linspace(0, 1, 50)
    dt = 0.001
    freq = 1.0
    wavelength = 1.5
    amplitude = 3.0

    for i in range(2000):
        t = i * dt
        s = np.linspace(0, 1, 50)
        kappa = amplitude * np.sin(2 * np.pi * (s * 1.0 / wavelength - freq * t))
        body_iso.step(kappa, grid_x, dt)
        body_aniso.step(kappa, grid_x, dt)

    iso_dist = np.sqrt(body_iso.x_cm**2 + body_iso.y_cm**2)
    aniso_dist = np.sqrt(body_aniso.x_cm**2 + body_aniso.y_cm**2)
    assert aniso_dist > 10 * iso_dist


def test_odor_circuit_neuron_lookup():
    from celegans.data import load_worm_data
    from celegans.sensory import OdorCircuit
    wd = load_worm_data(DATA_DIR)
    circuit = OdorCircuit(wd.neuron_idx, wd.N)
    assert circuit.awcl == 76
    assert circuit.awcr == 77
    assert circuit.awal == 72
    assert circuit.awar == 73


def test_odor_circuit_increasing_concentration():
    from celegans.sensory import OdorCircuit
    idx = {'AWCL': 0, 'AWCR': 1, 'AWAL': 2, 'AWAR': 3}
    circuit = OdorCircuit(idx, N=4)
    dt = 0.001
    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        drive = circuit.transduce(c, c, dt)
    assert drive[2] > 0
    assert drive[3] > 0
    assert drive[0] <= 0
    assert drive[1] <= 0


def test_odor_circuit_decreasing_concentration():
    from celegans.sensory import OdorCircuit
    idx = {'AWCL': 0, 'AWCR': 1, 'AWAL': 2, 'AWAR': 3}
    circuit = OdorCircuit(idx, N=4)
    dt = 0.001
    for c in [0.5, 0.5, 0.5, 0.5, 0.5]:
        circuit.transduce(c, c, dt)
    for c in [0.4, 0.3, 0.2, 0.1]:
        drive = circuit.transduce(c, c, dt)
    assert drive[0] > 0
    assert drive[1] > 0


def test_odor_circuit_bilateral_asymmetry():
    from celegans.sensory import OdorCircuit
    idx = {'AWCL': 0, 'AWCR': 1, 'AWAL': 2, 'AWAR': 3}
    circuit = OdorCircuit(idx, N=4)
    dt = 0.001
    for _ in range(10):
        circuit.transduce(0.5, 0.5, dt)
    drive = circuit.transduce(0.6, 0.4, dt)
    assert drive[2] > drive[3]


def test_trajectory_recorder_chemotaxis_index():
    from celegans.behavior import TrajectoryRecorder
    rec = TrajectoryRecorder()
    for i in range(100):
        x = i * 0.1
        rec.record(t=i * 0.1, x=x, y=0.0, heading=0.0,
                   v_forward=0.1, omega=0.0, C_head=0.0)
    ci = rec.chemotaxis_index(10.0, 0.0)
    assert ci > 0.9


def test_trajectory_recorder_path_efficiency():
    from celegans.behavior import TrajectoryRecorder
    rec = TrajectoryRecorder()
    for i in range(100):
        rec.record(t=i * 0.1, x=i * 0.1, y=0.0, heading=0.0,
                   v_forward=0.1, omega=0.0, C_head=0.0)
    eff = rec.path_efficiency(10.0, 0.0)
    assert eff > 0.95


def test_classify_behavior_forward():
    from celegans.behavior import TrajectoryRecorder
    rec = TrajectoryRecorder()
    for i in range(2000):
        rec.record(t=i * 0.001, x=i * 0.001 * 0.2, y=0.0, heading=0.0,
                   v_forward=0.2, omega=0.0, C_head=0.5)
    states = rec.classify_behavior(dt=0.001, smoothing_window=0.1)
    n_forward = sum(1 for s in states if s == 'forward')
    assert n_forward / len(states) > 0.8


def test_classify_behavior_reversal():
    from celegans.behavior import TrajectoryRecorder
    rec = TrajectoryRecorder()
    for i in range(1000):
        rec.record(t=i * 0.001, x=0.0, y=0.0, heading=0.0,
                   v_forward=0.2, omega=0.0, C_head=0.5)
    for i in range(1000):
        rec.record(t=(1000 + i) * 0.001, x=0.0, y=0.0, heading=0.0,
                   v_forward=-0.2, omega=0.0, C_head=0.5)
    states = rec.classify_behavior(dt=0.001, smoothing_window=0.1)
    assert 'reverse' in states
    assert 'forward' in states


def test_external_drive():
    """external_drive should override the default sensory drive."""
    from celegans import (
        load_worm_data, build_coupling_matrices, build_grid_mapping,
        WormState, WormParams, assign_frequencies, step,
    )
    wd = load_worm_data(DATA_DIR)
    cm = build_coupling_matrices(wd)
    gm = build_grid_mapping(wd.pos_1d, Nx=50)
    params = WormParams()
    omegas = assign_frequencies(wd.N, wd.sensory, wd.motor, wd.inter)
    n_sensors = min(len(wd.sensory), 50)

    # Run with env_signal
    state1 = WormState(wd.N, 50, seed=42)
    env_sig = np.zeros(n_sensors)
    env_sig[0] = 5.0
    for _ in range(10):
        step(state1, params, omegas, cm, gm,
             wd.sensory, wd.motor, n_sensors, env_sig, dt=0.001)

    # Run with external_drive targeting same neuron
    state2 = WormState(wd.N, 50, seed=42)
    ext_drive = np.zeros(wd.N)
    ext_drive[wd.sensory[0]] = 5.0 * params.K_drive
    dummy_env = np.zeros(n_sensors)
    for _ in range(10):
        step(state2, params, omegas, cm, gm,
             wd.sensory, wd.motor, n_sensors, dummy_env, dt=0.001,
             external_drive=ext_drive)

    assert np.allclose(state1.theta, state2.theta, atol=1e-8)


def test_frustration_matrix_shape():
    """build_frustration_matrix produces correct shape and motor-only entries."""
    from celegans.data import load_worm_data
    from celegans.coupling import build_frustration_matrix
    wd = load_worm_data(DATA_DIR)
    frust = build_frustration_matrix(wd.pos_1d, wd.motor, wd.W_gap, alpha=1.0)
    assert frust.shape == (448, 448)
    # Non-motor entries should be zero
    non_motor = [i for i in range(448) if i not in wd.motor]
    assert np.allclose(frust[non_motor, :], 0.0)
    assert np.allclose(frust[:, non_motor], 0.0)
    # Should have some nonzero entries where motor gap junctions exist
    assert np.abs(frust).sum() > 0
    # Anti-symmetric: frust[i,j] = -frust[j,i]
    assert np.allclose(frust, -frust.T, atol=1e-10)


def test_frustration_zero_alpha():
    """alpha=0 should produce zero frustration matrix."""
    from celegans.data import load_worm_data
    from celegans.coupling import build_frustration_matrix
    wd = load_worm_data(DATA_DIR)
    frust = build_frustration_matrix(wd.pos_1d, wd.motor, wd.W_gap, alpha=0.0)
    assert np.allclose(frust, 0.0)


def test_dynamics_with_frustration():
    """step() with frustration_matrix should run without error."""
    from celegans import (
        load_worm_data, build_coupling_matrices, build_grid_mapping,
        WormState, WormParams, assign_frequencies, step,
        build_frustration_matrix,
    )
    wd = load_worm_data(DATA_DIR)
    cm = build_coupling_matrices(wd)
    gm = build_grid_mapping(wd.pos_1d, Nx=50)
    params = WormParams()
    omegas = assign_frequencies(wd.N, wd.sensory, wd.motor, wd.inter)
    state = WormState(wd.N, 50, seed=42)
    n_sensors = min(len(wd.sensory), 50)
    frust = build_frustration_matrix(wd.pos_1d, wd.motor, wd.W_gap, alpha=1.0)

    env = np.zeros(n_sensors)
    for _ in range(50):
        diag = step(state, params, omegas, cm, gm,
                    wd.sensory, wd.motor, n_sensors, env, dt=0.001,
                    frustration_matrix=frust)
    assert diag['order_r'] > 0


def test_navigation_state_forward_default():
    """NavigationState starts in FORWARD state."""
    from celegans.navigation import NavigationState
    nav = NavigationState(seed=42)
    assert nav.state == NavigationState.FORWARD
    # With zero dCdt and zero AVA activity, should mostly stay forward
    n_reversals = 0
    for _ in range(1000):
        _, did_rev = nav.step(0.0, 0.0, dt=0.01)
        if did_rev:
            n_reversals += 1
    # Base rate 2/min = 0.033/s, over 10s expect ~0.33 reversals
    assert n_reversals < 5


def test_navigation_more_reversals_down_gradient():
    """Negative dCdt should produce more reversals than positive."""
    from celegans.navigation import NavigationState

    # Stronger signal, longer run for statistical separation
    # Down-gradient: negative dCdt
    nav_down = NavigationState(seed=42, sensory_gain=10.0)
    revs_down = 0
    for _ in range(50000):
        _, did_rev = nav_down.step(0.5, -0.05, dt=0.01)
        if did_rev:
            revs_down += 1

    # Up-gradient: positive dCdt
    nav_up = NavigationState(seed=43, sensory_gain=10.0)
    revs_up = 0
    for _ in range(50000):
        _, did_rev = nav_up.step(0.5, 0.05, dt=0.01)
        if did_rev:
            revs_up += 1

    assert revs_down > revs_up, f"down={revs_down}, up={revs_up}"


def test_reversal_turn_angle():
    """Turn angles should be centered near pi."""
    from celegans.navigation import reversal_turn_angle
    rng = np.random.RandomState(42)
    angles = [reversal_turn_angle(0.0, rng) for _ in range(1000)]
    mean_angle = np.mean(angles)
    assert abs(mean_angle - np.pi) < 0.2


def test_weathervane_symmetric():
    """Equal bilateral activity should produce zero torque."""
    from celegans.navigation import WeathervaneCircuit
    wv = WeathervaneCircuit(gain=0.5)
    torque = wv.heading_torque(0.5, 0.5)
    assert abs(torque) < 1e-10


def test_weathervane_asymmetric():
    """Higher left activity should produce positive (CCW) torque."""
    from celegans.navigation import WeathervaneCircuit
    wv = WeathervaneCircuit(gain=0.5)
    torque = wv.heading_torque(0.8, 0.2)
    assert torque > 0
    torque_rev = wv.heading_torque(0.2, 0.8)
    assert torque_rev < 0


def test_body_reverse():
    """Body reverse should flip heading by approximately pi."""
    from celegans.body import ArticulatedBody
    body = ArticulatedBody(n_segments=50, body_length_mm=1.0,
                           x0=0.0, y0=0.0, heading0=0.0)
    kappa = np.zeros(50)
    grid_x = np.linspace(0, 1, 50)
    body.set_curvature(kappa, grid_x)
    body.segment_positions()
    body.reverse(np.pi)
    assert abs(body.heading - np.pi) < 1e-10


def test_body_apply_torque():
    """apply_torque should change heading proportionally."""
    from celegans.body import ArticulatedBody
    body = ArticulatedBody(n_segments=50, body_length_mm=1.0,
                           x0=0.0, y0=0.0, heading0=0.0)
    body.apply_torque(1.0, dt=0.1)  # 1 rad/s * 0.1s = 0.1 rad
    assert abs(body.heading - 0.1) < 1e-10


def test_odor_circuit_exposes_dCdt():
    """OdorCircuit.last_dCdt should track concentration change rate."""
    from celegans.sensory import OdorCircuit
    idx = {'AWCL': 0, 'AWCR': 1, 'AWAL': 2, 'AWAR': 3}
    circuit = OdorCircuit(idx, N=4)
    circuit.transduce(0.5, 0.5, dt=0.001)
    assert circuit.last_dCdt == 0.0
    circuit.transduce(0.6, 0.6, dt=0.001)
    assert circuit.last_dCdt > 0


def test_dynamics_drives_body():
    """Dynamics engine kappa output should produce body movement via RFT.

    The dynamics engine produces small kappa amplitudes (~0.05-0.2 sim units)
    due to heavy spring damping. With kappa_scale=10.0, physical curvature
    reaches ~0.5-2.0 mm^-1, enough for measurable RFT locomotion over
    several undulation cycles.
    """
    from celegans import (
        load_worm_data, build_coupling_matrices, build_grid_mapping,
        WormState, WormParams, assign_frequencies, step,
    )
    from celegans.body import ArticulatedBody

    wd = load_worm_data(DATA_DIR)
    cm = build_coupling_matrices(wd)
    Nx = 50
    gm = build_grid_mapping(wd.pos_1d, Nx=Nx)
    params = WormParams()
    omegas = assign_frequencies(wd.N, wd.sensory, wd.motor, wd.inter)
    state = WormState(wd.N, Nx)
    n_sensors = min(len(wd.sensory), 50)

    # Higher kappa_scale bridges the gap between dynamics sim units
    # and physical curvature needed for locomotion
    body = ArticulatedBody(n_segments=Nx, body_length_mm=1.0,
                           x0=0.0, y0=0.0, heading0=0.0,
                           kappa_scale=10.0)

    rng = np.random.RandomState(42)
    for t in range(5000):
        env_sig = 0.3 * rng.randn(n_sensors)
        step(state, params, omegas, cm, gm,
             wd.sensory, wd.motor, n_sensors, env_sig, dt=0.001)
        body.step(state.kappa, gm['grid_x'], dt=0.001)

    # Dynamics kappa is small (~0.02 std) due to heavy spring damping.
    # The pipeline works but locomotion speed is tiny without calibration.
    # This test verifies the plumbing: dynamics kappa → body shape → RFT → nonzero velocity.
    assert body._v_cm[0] != 0.0 or body._v_cm[1] != 0.0, \
        "RFT produced zero velocity — pipeline broken"
    dist = np.sqrt(body.x_cm**2 + body.y_cm**2)
    assert dist > 0, f"Worm position unchanged from origin"
