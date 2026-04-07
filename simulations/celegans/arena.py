"""2D circular arena with Gaussian odor concentration fields."""

import numpy as np


class Arena:
    """Circular arena with chemical concentration sources.

    Parameters
    ----------
    radius_mm : float
        Radius of the circular plate (standard NGM plate: 50 mm).
    sources : list of dict
        Each dict: {'x': float, 'y': float, 'strength': float, 'sigma': float}.
        Concentration = sum of Gaussians.
    """

    def __init__(self, radius_mm, sources):
        self.radius = radius_mm
        self.sources = sources

    def concentration(self, x, y):
        """Scalar odor concentration at (x, y) in mm."""
        c = 0.0
        for s in self.sources:
            r2 = (x - s['x'])**2 + (y - s['y'])**2
            c += s['strength'] * np.exp(-r2 / (2.0 * s['sigma']**2))
        return c

    def gradient(self, x, y):
        """Analytical gradient (dC/dx, dC/dy). For validation, not used by worm."""
        gx, gy = 0.0, 0.0
        for s in self.sources:
            dx = x - s['x']
            dy = y - s['y']
            r2 = dx**2 + dy**2
            g = s['strength'] * np.exp(-r2 / (2.0 * s['sigma']**2))
            inv_s2 = 1.0 / s['sigma']**2
            gx += g * (-dx * inv_s2)
            gy += g * (-dy * inv_s2)
        return gx, gy

    def in_bounds(self, x, y):
        """True if (x, y) is within the circular plate."""
        return x**2 + y**2 <= self.radius**2
