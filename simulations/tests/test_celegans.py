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
