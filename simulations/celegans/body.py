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

        self.seg_angles = np.full(n_segments, float(heading0))
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

    # ------------------------------------------------------------------
    # RFT locomotion
    # ------------------------------------------------------------------

    def compute_velocity(self, dt):
        """Solve for (v_cm_x, v_cm_y, omega) using resistive force theory.

        Uses force-free and torque-free conditions at low Reynolds number.
        The deformation velocity (shape change between timesteps) provides
        the known forcing that drives locomotion.

        Returns
        -------
        vx, vy, omega : float
        """
        pos = self._positions  # shape (n_seg, 2), CM-centred in lab frame
        cm = np.array([self.x_cm, self.y_cm])
        r = pos - cm  # position relative to CM

        # Deformation velocity: how each segment moved relative to CM
        # due to shape change between timesteps.
        if self.prev_positions is not None:
            prev_cm = self.prev_positions.mean(axis=0)
            r_prev = self.prev_positions - prev_cm
            v_deform = (r - r_prev) / dt
        else:
            v_deform = np.zeros_like(r)

        # Build 3x3 linear system: A @ [vx, vy, omega] = b
        # From sum of forces = 0 and sum of torques = 0.
        #
        # For segment i with tangent t_i and normal n_i:
        #   v_i = v_cm + omega * (-r_iy, r_ix) + v_deform_i
        #   f_i = -C_t * (v_i . t_i) * t_i - C_n * (v_i . n_i) * n_i
        #
        # The unknowns (v_cm, omega) contribute to A.
        # The v_deform terms contribute to b (RHS).

        A = np.zeros((3, 3))
        b = np.zeros(3)

        cos_a = np.cos(self.seg_angles)
        sin_a = np.sin(self.seg_angles)

        for i in range(self.n_seg):
            tx, ty = cos_a[i], sin_a[i]
            nx, ny = -sin_a[i], cos_a[i]
            rx, ry = r[i, 0], r[i, 1]

            # Rotation velocity contribution: omega * (-ry, rx)
            rot_x, rot_y = -ry, rx

            # --- Contributions of unknowns to v_i . t_i and v_i . n_i ---
            # v_i = (vx + omega*rot_x + vd_x, vy + omega*rot_y + vd_y)
            # v_i . t = vx*tx + vy*ty + omega*(rot_x*tx + rot_y*ty) + vd.t
            # v_i . n = vx*nx + vy*ny + omega*(rot_x*nx + rot_y*ny) + vd.n

            rot_dot_t = rot_x * tx + rot_y * ty
            rot_dot_n = rot_x * nx + rot_y * ny
            vd_dot_t = v_deform[i, 0] * tx + v_deform[i, 1] * ty
            vd_dot_n = v_deform[i, 0] * nx + v_deform[i, 1] * ny

            # f_i = -C_t*(v.t)*t - C_n*(v.n)*n
            # f_ix = -C_t*(v.t)*tx - C_n*(v.n)*nx
            # f_iy = -C_t*(v.t)*ty - C_n*(v.n)*ny
            #
            # Contribution to force from unknowns [vx, vy, omega]:
            # d(v.t)/d(vx) = tx, d(v.t)/d(vy) = ty, d(v.t)/d(omega) = rot_dot_t
            # d(v.n)/d(vx) = nx, d(v.n)/d(vy) = ny, d(v.n)/d(omega) = rot_dot_n
            #
            # df_ix/d(vx) = -C_t*tx*tx - C_n*nx*nx
            # df_ix/d(vy) = -C_t*tx*ty - C_n*nx*ny
            # ...etc

            # Force equation rows (sum f_ix = 0, sum f_iy = 0)
            # A_row * [vx, vy, omega] = -b_force
            # where b_force is the contribution from v_deform

            # Row 0: sum f_ix = 0 -> sum (coeff * unknowns) = -sum(deform contrib)
            # Row 1: sum f_iy = 0
            # Row 2: sum torque_z = sum (rx * f_iy - ry * f_ix) = 0

            ct, cn = self.C_t, self.C_n

            # Coefficients of [vx, vy, omega] in f_ix
            a_fx_vx = ct * tx * tx + cn * nx * nx
            a_fx_vy = ct * tx * ty + cn * nx * ny
            a_fx_om = ct * tx * rot_dot_t + cn * nx * rot_dot_n

            # Coefficients of [vx, vy, omega] in f_iy
            a_fy_vx = ct * ty * tx + cn * ny * nx
            a_fy_vy = ct * ty * ty + cn * ny * ny
            a_fy_om = ct * ty * rot_dot_t + cn * ny * rot_dot_n

            # Force balance: sum(-C_t*(v.t)*t - C_n*(v.n)*n) = 0
            # => sum(C_t*(v.t)*t + C_n*(v.n)*n) = 0  (move sign)
            # Split: A @ unknowns = -deform_contribution
            A[0, 0] += a_fx_vx
            A[0, 1] += a_fx_vy
            A[0, 2] += a_fx_om
            A[1, 0] += a_fy_vx
            A[1, 1] += a_fy_vy
            A[1, 2] += a_fy_om

            # RHS: negative of deformation contribution
            b[0] -= ct * vd_dot_t * tx + cn * vd_dot_n * nx
            b[1] -= ct * vd_dot_t * ty + cn * vd_dot_n * ny

            # Torque: tau_i = rx * f_iy - ry * f_ix
            # Coefficients in torque from unknowns:
            A[2, 0] += rx * a_fy_vx - ry * a_fx_vx
            A[2, 1] += rx * a_fy_vy - ry * a_fx_vy
            A[2, 2] += rx * a_fy_om - ry * a_fx_om

            # Torque RHS
            deform_fx = ct * vd_dot_t * tx + cn * vd_dot_n * nx
            deform_fy = ct * vd_dot_t * ty + cn * vd_dot_n * ny
            b[2] -= rx * deform_fy - ry * deform_fx

        sol = np.linalg.solve(A, b)
        return sol[0], sol[1], sol[2]

    def step(self, kappa_grid, grid_x, dt):
        """Advance the body by one timestep.

        1. Save current positions as prev_positions.
        2. Update curvature -> seg_angles (new shape, current heading).
        3. Recompute segment_positions (new shape, old CM/heading).
        4. Solve RFT for (vx, vy, omega).
        5. Update x_cm, y_cm, heading.
        6. Adjust seg_angles by the heading change.
        7. Recompute segment_positions with updated CM/heading.
        """
        # 1. Save previous positions
        if self._positions is not None:
            self.prev_positions = self._positions.copy()
        else:
            # First call: establish shape, then save
            self.set_curvature(kappa_grid, grid_x)
            self.segment_positions()
            self.prev_positions = self._positions.copy()

        # 2-3. Update curvature and compute new shape at old CM/heading
        self.set_curvature(kappa_grid, grid_x)
        self.segment_positions()

        # 4. Solve RFT
        vx, vy, omega = self.compute_velocity(dt)
        self._v_cm = np.array([vx, vy])
        self._omega = omega

        # 5. Update state
        delta_heading = omega * dt
        self.x_cm += vx * dt
        self.y_cm += vy * dt
        self.heading += delta_heading

        # 6. Adjust all segment angles by heading change
        self.seg_angles += delta_heading

        # 7. Recompute positions with new CM/heading
        self.segment_positions()

    def velocity_along_body(self):
        """Return translational velocity projected onto heading direction."""
        return (self._v_cm[0] * np.cos(self.heading)
                + self._v_cm[1] * np.sin(self.heading))

    def angular_velocity(self):
        """Return angular velocity (rad/s)."""
        return self._omega

    def enforce_bounds(self, arena):
        """If worm CM is outside arena, push back inside and flip heading."""
        r2 = self.x_cm**2 + self.y_cm**2
        if r2 > arena.radius**2:
            r = np.sqrt(r2)
            # Push CM just inside the boundary
            scale = (arena.radius * 0.99) / r
            self.x_cm *= scale
            self.y_cm *= scale
            # Flip heading (reverse direction)
            self.heading += np.pi
            # Update segment angles to reflect the flip
            self.seg_angles += np.pi
            # Recompute positions
            self.segment_positions()
