"""Behavioral classification and chemotaxis metrics for C. elegans.

Records worm trajectories and provides cycle-averaged behavioral
classification (forward/reverse/omega turn/pause) plus standard
chemotaxis metrics.
"""

import numpy as np


class TrajectoryRecorder:
    """Records and analyzes worm trajectory during chemotaxis."""

    def __init__(self):
        self.times = []
        self.positions = []
        self.headings = []
        self.velocities = []
        self.angular_vels = []
        self.concentrations = []

    def record(self, t, x, y, heading, v_forward, omega, C_head):
        """Record one timestep of trajectory data."""
        self.times.append(t)
        self.positions.append((x, y))
        self.headings.append(heading)
        self.velocities.append(v_forward)
        self.angular_vels.append(omega)
        self.concentrations.append(C_head)

    def classify_behavior(self, dt, v_threshold=0.02, omega_threshold=2.0,
                          smoothing_window=1.0, min_duration=0.5):
        """Classify each timestep as forward/reverse/omega_turn/pause.

        Uses cycle-averaged velocity to avoid aliasing the gait cycle.
        """
        v = np.array(self.velocities)
        w = np.array(self.angular_vels)

        kernel_size = max(1, int(smoothing_window / dt))
        kernel = np.ones(kernel_size) / kernel_size
        v_smooth = np.convolve(v, kernel, mode='same')
        w_smooth = np.convolve(np.abs(w), kernel, mode='same')

        states = []
        for i in range(len(v_smooth)):
            if w_smooth[i] > omega_threshold:
                states.append('omega_turn')
            elif v_smooth[i] > v_threshold:
                states.append('forward')
            elif v_smooth[i] < -v_threshold:
                states.append('reverse')
            else:
                states.append('pause')

        min_steps = max(1, int(min_duration / dt))
        filtered = states.copy()
        i = 0
        while i < len(filtered):
            j = i + 1
            while j < len(filtered) and filtered[j] == filtered[i]:
                j += 1
            run_len = j - i
            if run_len < min_steps and i > 0:
                for k in range(i, j):
                    filtered[k] = filtered[i - 1]
            i = j

        return filtered

    def chemotaxis_index(self, food_x, food_y):
        """CI = (d_start - d_end) / d_start. Range [-1, 1]."""
        if len(self.positions) < 2:
            return 0.0
        x0, y0 = self.positions[0]
        xf, yf = self.positions[-1]
        d_start = np.sqrt((x0 - food_x)**2 + (y0 - food_y)**2)
        d_end = np.sqrt((xf - food_x)**2 + (yf - food_y)**2)
        if d_start < 1e-10:
            return 0.0
        return (d_start - d_end) / d_start

    def path_efficiency(self, food_x, food_y):
        """Straight-line distance / actual path length. Range [0, 1]."""
        if len(self.positions) < 2:
            return 0.0
        pts = np.array(self.positions)
        diffs = np.diff(pts, axis=0)
        path_len = np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2))
        x0, y0 = pts[0]
        xf, yf = pts[-1]
        straight = np.sqrt((xf - x0)**2 + (yf - y0)**2)
        if path_len < 1e-10:
            return 0.0
        return straight / path_len

    def run_length_distribution(self, dt):
        """Duration of each consecutive forward run (seconds)."""
        states = self.classify_behavior(dt)
        runs = []
        current_len = 0
        for s in states:
            if s == 'forward':
                current_len += 1
            else:
                if current_len > 0:
                    runs.append(current_len * dt)
                current_len = 0
        if current_len > 0:
            runs.append(current_len * dt)
        return runs

    def bearing_to_food(self, food_x, food_y):
        """Angle between heading and direction to food. Range [0, pi]."""
        bearings = []
        for (x, y), h in zip(self.positions, self.headings):
            dx = food_x - x
            dy = food_y - y
            food_angle = np.arctan2(dy, dx)
            diff = abs(food_angle - h)
            diff = diff % (2 * np.pi)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            bearings.append(diff)
        return np.array(bearings)

    def reversal_rate(self, dt):
        """Reversals per minute."""
        states = self.classify_behavior(dt)
        reversals = 0
        for i in range(1, len(states)):
            if states[i] == 'reverse' and states[i - 1] != 'reverse':
                reversals += 1
        duration_min = len(states) * dt / 60.0
        if duration_min < 1e-10:
            return 0.0
        return reversals / duration_min

    def reversal_triggered_by_gradient(self, food_x, food_y, dt):
        """Fraction of reversals occurring while moving down-gradient."""
        states = self.classify_behavior(dt)
        bearings = self.bearing_to_food(food_x, food_y)
        total_reversals = 0
        down_gradient_reversals = 0
        for i in range(1, len(states)):
            if states[i] == 'reverse' and states[i - 1] != 'reverse':
                total_reversals += 1
                if bearings[i] > np.pi / 2:
                    down_gradient_reversals += 1
        if total_reversals == 0:
            return 0.0
        return down_gradient_reversals / total_reversals
