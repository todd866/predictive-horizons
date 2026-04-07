"""Odor chemosensory circuit for C. elegans chemotaxis.

Maps odor concentration at the worm's head to drive signals on identified
AWC and AWA neurons. AWC are ON cells (suppressed by odor increase, activated
by decrease). AWA are activated by odor presence.

Reference: Bargmann 2006, Chalasani et al. 2007.
"""

import numpy as np


class OdorCircuit:
    """Odor transduction for AWC and AWA chemosensory neurons.

    Parameters
    ----------
    neuron_idx : dict
        Neuron name -> index mapping from WormData.
    N : int
        Total number of neurons.
    awc_gain : float
        Gain for AWC response to dC/dt.
    awa_gain : float
        Gain for AWA response to concentration.
    tau_smooth : float
        Exponential smoothing timescale for dC/dt (seconds).
    """

    def __init__(self, neuron_idx, N, awc_gain=1.0, awa_gain=1.0,
                 tau_smooth=0.5):
        self.N = N
        self.awcl = neuron_idx.get('AWCL')
        self.awcr = neuron_idx.get('AWCR')
        self.awal = neuron_idx.get('AWAL')
        self.awar = neuron_idx.get('AWAR')
        self.awc_gain = awc_gain
        self.awa_gain = awa_gain
        self.tau_smooth = tau_smooth
        self.C_smooth = None
        self.last_dCdt = 0.0

    def transduce(self, C_left, C_right, dt):
        """Convert bilateral head concentration to per-neuron drive.

        Parameters
        ----------
        C_left, C_right : float
            Odor concentration at left and right sides of the head.
        dt : float
            Timestep in seconds.

        Returns
        -------
        drive : ndarray, shape (N,)
            Per-neuron drive signal.
        """
        C_mean = 0.5 * (C_left + C_right)
        drive = np.zeros(self.N)

        if self.C_smooth is None:
            self.C_smooth = C_mean
            dCdt = 0.0
        else:
            alpha = dt / (self.tau_smooth + dt)
            new_smooth = self.C_smooth + alpha * (C_mean - self.C_smooth)
            dCdt = (new_smooth - self.C_smooth) / dt
            self.C_smooth = new_smooth
        self.last_dCdt = dCdt

        awc_drive = -self.awc_gain * dCdt
        awa_drive_left = self.awa_gain * C_left
        awa_drive_right = self.awa_gain * C_right

        if self.awcl is not None:
            drive[self.awcl] = awc_drive
        if self.awcr is not None:
            drive[self.awcr] = awc_drive
        if self.awal is not None:
            drive[self.awal] = awa_drive_left
        if self.awar is not None:
            drive[self.awar] = awa_drive_right

        return drive
