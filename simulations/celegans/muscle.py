"""Dorsal-ventral muscle activation model for C. elegans.

Converts motor neuron activity into two opposing muscle activation
fields (dorsal, ventral) on the body-axis grid.  Curvature arises
from the difference: kappa = K * (m_dorsal - m_ventral).

Motor classes and their roles:
  VA/VB -> ventral excitatory
  DA/DB -> dorsal excitatory
  VD    -> ventral inhibitory (cross-coupled from dorsal)
  DD    -> dorsal inhibitory (cross-coupled from ventral)
"""

import numpy as np


def muscle_drive(dorsal_exc, ventral_exc, dorsal_inh, ventral_inh,
                 grid_x, sigma=0.08):
    """Compute raw dorsal and ventral muscle drive from neuron activations.

    Parameters
    ----------
    dorsal_exc : list of (position, activity) tuples
        DA/DB excitatory neurons.
    ventral_exc : list of (position, activity) tuples
        VA/VB excitatory neurons.
    dorsal_inh : list of (position, activity) tuples
        DD inhibitory neurons (suppress dorsal).
    ventral_inh : list of (position, activity) tuples
        VD inhibitory neurons (suppress ventral).
    grid_x : ndarray, shape (Nx,)
        Body-axis grid positions in [0, 1].
    sigma : float
        Gaussian spread of each neuron's contribution to the grid.

    Returns
    -------
    F_dorsal, F_ventral : ndarray, shape (Nx,)
        Raw muscle drive fields (before filtering).
    """
    Nx = len(grid_x)
    F_d = np.zeros(Nx)
    F_v = np.zeros(Nx)

    for pos, act in dorsal_exc:
        w = np.exp(-(grid_x - pos)**2 / (2 * sigma**2))
        F_d += act * w
    for pos, act in dorsal_inh:
        w = np.exp(-(grid_x - pos)**2 / (2 * sigma**2))
        F_d -= act * w

    for pos, act in ventral_exc:
        w = np.exp(-(grid_x - pos)**2 / (2 * sigma**2))
        F_v += act * w
    for pos, act in ventral_inh:
        w = np.exp(-(grid_x - pos)**2 / (2 * sigma**2))
        F_v -= act * w

    return np.clip(F_d, 0, None), np.clip(F_v, 0, None)


class MuscleState:
    """Low-pass filtered muscle activation fields.

    Parameters
    ----------
    Nx : int
        Number of body-axis grid points.
    tau : float
        Muscle activation time constant (seconds).  ~50 ms.
    """

    def __init__(self, Nx, tau=0.05):
        self.m_dorsal = np.zeros(Nx)
        self.m_ventral = np.zeros(Nx)
        self.tau = tau

    def step(self, F_dorsal, F_ventral, dt):
        """Advance muscle activation by one timestep."""
        alpha = dt / (self.tau + dt)
        self.m_dorsal += alpha * (F_dorsal - self.m_dorsal)
        self.m_ventral += alpha * (F_ventral - self.m_ventral)

    def kappa(self, K_muscle=1.0):
        """Net curvature: dorsal - ventral."""
        return K_muscle * (self.m_dorsal - self.m_ventral)
