"""Metabolic budget tracker for nonergodic defense simulation.

Provides a scalar resource R(t) that constrains coupling strengths.
When expenditure exceeds regeneration, R drops, coupling degrades,
coherence erodes, and D_eff declines — modelling progressive collapse
(aging / death).
"""

import numpy as np


class MetabolicBudget:
    """Scalar metabolic resource that constrains network dynamics.

    Parameters
    ----------
    R_max : float
        Maximum (and initial) resource level.
    E_in : float
        Regeneration rate per second.
    cost_coupling : float
        Cost coefficient for coupling-related activity
        (conn_rms + eph_rms + npp_std).
    cost_coherence : float
        Cost coefficient for coherence maintenance (mean_phi).
    cost_dimensional : float
        Cost coefficient for dimensional order (order_r).
    """

    def __init__(
        self,
        R_max: float = 100.0,
        E_in: float = 5.0,
        cost_coupling: float = 1.0,
        cost_coherence: float = 0.5,
        cost_dimensional: float = 0.1,
    ):
        self.R_max = R_max
        self.R = R_max
        self.E_in = E_in
        self.cost_coupling = cost_coupling
        self.cost_coherence = cost_coherence
        self.cost_dimensional = cost_dimensional

    def update(
        self,
        dt: float,
        conn_rms: float = 0.0,
        eph_rms: float = 0.0,
        npp_std: float = 0.0,
        mean_phi: float = 0.0,
        order_r: float = 0.0,
    ) -> float:
        """Advance the resource by one timestep.

        Parameters
        ----------
        dt : float
            Timestep in seconds.
        conn_rms, eph_rms, npp_std : float
            RMS coupling strengths (connectome, ephaptic, neuromodulator).
        mean_phi : float
            Mean phase coherence across oscillators.
        order_r : float
            Kuramoto order parameter (global synchrony).

        Returns
        -------
        float
            budget_scale = R / R_max, in [0, 1].
        """
        expenditure = (
            self.cost_coupling * (conn_rms + eph_rms + npp_std)
            + self.cost_coherence * mean_phi
            + self.cost_dimensional * order_r * 10.0
        )
        dR = (self.E_in - expenditure) * dt
        self.R = float(np.clip(self.R + dR, 0.0, self.R_max))
        return self.budget_scale

    @property
    def budget_scale(self) -> float:
        """Current resource fraction, in [0, 1]."""
        return self.R / self.R_max

    @property
    def depleted(self) -> bool:
        """True when resource falls below 10% of maximum."""
        return self.R < 0.1 * self.R_max


class UnlimitedBudget:
    """No-op budget for unconstrained comparison runs.

    Always returns full budget scale (1.0) and never reports depletion.
    """

    def update(self, dt: float, **kwargs) -> float:
        """No-op update; always returns 1.0."""
        return 1.0

    @property
    def budget_scale(self) -> float:
        return 1.0

    @property
    def depleted(self) -> bool:
        return False
