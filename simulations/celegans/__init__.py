"""C. elegans simulation toolkit — reusable modules for connectome-based modeling."""

from .data import load_worm_data, WormData
from .coupling import build_coupling_matrices, build_grid_mapping, CouplingMatrices
from .dynamics import WormState, WormParams, assign_frequencies, step
from .environment import generate_multimode_env, generate_simple_env, default_taus
from .analysis import (d_eff, d_eff_timeseries, gaussian_mi, mi_profile,
                       motor_correlation, spectral_overlap, classify_trajectory)
from .budget import MetabolicBudget, UnlimitedBudget
