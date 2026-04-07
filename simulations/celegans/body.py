"""Articulated worm body with resistive force locomotion.

Converts body-axis curvature from the dynamics engine into a 2D body shape,
then uses resistive force theory to compute locomotion velocity from
time-varying shape changes.
"""

import numpy as np


class ArticulatedBody:
    """Segmented worm body on a 2D agar surface.

    Parameters
    ----------
    n_segments : int
        Number of body segments.
    body_length_mm : float
        Total body length in mm.
    x0, y0 : float
        Initial center-of-mass position in mm.
    heading0 : float
        Initial heading angle in radians (0 = +x direction).
    C_t : float
        Tangential drag coefficient.
    C_n : float
        Normal drag coefficient. C_n > C_t produces forward thrust from undulation.
    kappa_scale : float
        Converts simulation kappa units to physical curvature (mm^-1).
    bilateral_offset : float
        Lateral offset (mm) for bilateral head sampling points.
    """

    def __init__(self, n_segments=50, body_length_mm=1.0,
                 x0=0.0, y0=0.0, heading0=0.0,
                 C_t=1.0, C_n=1.5, kappa_scale=1.5,
                 bilateral_offset=0.05):
        self.n_seg = n_segments
        self.L = body_length_mm
        self.ds = body_length_mm / n_segments
        self.x_cm = x0
        self.y_cm = y0
        self.heading = heading0
        self.C_t = C_t
        self.C_n = C_n
        self.kappa_scale = kappa_scale
        self.bilateral_offset = bilateral_offset

        self.seg_angles = np.full(n_segments, heading0)
        self._positions = None
        self.prev_positions = None

        self._v_cm = np.zeros(2)
        self._omega = 0.0

    def set_curvature(self, kappa_grid, grid_x):
        """Set segment angles from body-axis curvature field.

        Parameters
        ----------
        kappa_grid : ndarray, shape (Nx,)
            Curvature field on the dynamics body-axis grid (simulation units).
        grid_x : ndarray, shape (Nx,)
            Grid positions in [0, 1].
        """
        seg_s = np.linspace(0.5 / self.n_seg, 1.0 - 0.5 / self.n_seg, self.n_seg)
        kappa_seg = np.interp(seg_s, grid_x, kappa_grid) * self.kappa_scale

        self.seg_angles[0] = self.heading
        for i in range(1, self.n_seg):
            self.seg_angles[i] = self.seg_angles[i - 1] + kappa_seg[i - 1] * self.ds

    def segment_positions(self):
        """Compute lab-frame (x, y) for each segment.

        Head is segment 0. Positions computed by integrating segment
        angles, then translating so CM matches (x_cm, y_cm).

        Returns
        -------
        positions : ndarray, shape (n_segments, 2)
        """
        pos = np.zeros((self.n_seg, 2))
        for i in range(1, self.n_seg):
            pos[i, 0] = pos[i - 1, 0] - self.ds * np.cos(self.seg_angles[i - 1])
            pos[i, 1] = pos[i - 1, 1] - self.ds * np.sin(self.seg_angles[i - 1])

        cm = pos.mean(axis=0)
        pos[:, 0] += self.x_cm - cm[0]
        pos[:, 1] += self.y_cm - cm[1]

        self._positions = pos
        return pos

    def head_position(self):
        """Return (x, y) of the head (segment 0)."""
        if self._positions is None:
            self.segment_positions()
        return float(self._positions[0, 0]), float(self._positions[0, 1])

    def bilateral_head_points(self):
        """Return (x_left, y_left, x_right, y_right) for bilateral sampling.

        Offset perpendicular to head tangent by bilateral_offset mm.
        """
        if self._positions is None:
            self.segment_positions()
        hx, hy = self._positions[0]
        theta_head = self.seg_angles[0]
        nx = -np.sin(theta_head) * self.bilateral_offset
        ny = np.cos(theta_head) * self.bilateral_offset
        return (hx + nx, hy + ny, hx - nx, hy - ny)
