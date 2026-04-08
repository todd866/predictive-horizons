"""Navigation circuits for C. elegans chemotaxis.

Implements pirouette (reversal modulation) and weathervane (heading bias)
strategies, reading neural activity from the phase oscillator model.

References:
    Pierce-Shimomura et al. 1999 -- pirouette frequency modulation
    Iino & Yoshida 2009 -- weathervane mechanism
    Kato et al. 2015 -- global brain state dynamics
"""

import numpy as np


class NavigationState:
    """Forward/backward locomotion state with sensory modulation.

    Two-state Markov process where forward->backward transition rate
    is modulated by dC/dt (concentration temporal derivative) and
    AVA command interneuron activity.

    Biological basis: C. elegans suppresses reversals when moving
    up-gradient (dC/dt > 0) and enhances them when moving down-gradient
    (dC/dt < 0).  Baseline reversal rate ~2/min, drops to ~0.5-1/min
    up-gradient (Pierce-Shimomura 1999).

    Parameters
    ----------
    base_rev_rate : float
        Baseline reversal rate in Hz (default 2/60 = 0.033).
    sensory_gain : float
        Exponential gain on dC/dt modulation.
    tau_dCdt : float
        Smoothing timescale for dC/dt (seconds).  ~3-5 s in biology.
    min_state_duration : float
        Minimum dwell time in each state (seconds).
    mean_back_duration : float
        Mean duration of backward state (seconds).
    seed : int
        RNG seed for stochastic transitions.
    """

    FORWARD = 0
    BACKWARD = 1

    def __init__(self, base_rev_rate=2.0 / 60, sensory_gain=5.0,
                 tau_dCdt=3.0, min_state_duration=1.0,
                 mean_back_duration=2.0, seed=0):
        self.base_rev_rate = base_rev_rate
        self.sensory_gain = sensory_gain
        self.tau_dCdt = tau_dCdt
        self.min_state_duration = min_state_duration
        self.mean_back_duration = mean_back_duration

        self.state = self.FORWARD
        self.state_timer = 0.0
        self.dCdt_smooth = 0.0
        self.rng = np.random.RandomState(seed)
        # Pre-draw waiting times from exponential distribution
        # (avoids numerical issues with rate * dt << 1 per-step checks)
        self._next_event_time = self._draw_wait(self.base_rev_rate * 1.5)

    def _draw_wait(self, rate):
        """Draw next event time from exponential distribution."""
        rate = max(rate, 1e-6)
        return -np.log(self.rng.random()) / rate

    def step(self, ava_activity, dCdt, dt):
        """Advance navigation state by one timestep.

        Uses pre-drawn exponential waiting times rather than per-step
        Bernoulli trials, which avoids numerical issues when dt is small.

        Parameters
        ----------
        ava_activity : float
            Mean activity of AVA command interneurons, in [0, 1].
        dCdt : float
            Instantaneous rate of concentration change at the head.
        dt : float
            Timestep in seconds.

        Returns
        -------
        state : int
            FORWARD (0) or BACKWARD (1).
        did_reverse : bool
            True if a forward->backward transition happened this step.
        did_resume : bool
            True if a backward->forward transition happened this step.
            This is where the pirouette turn should be applied.
        """
        self.state_timer += dt

        # Exponential smoothing of dC/dt
        alpha = dt / (self.tau_dCdt + dt)
        self.dCdt_smooth += alpha * (dCdt - self.dCdt_smooth)

        did_reverse = False
        did_resume = False

        if self.state == self.FORWARD:
            if self.state_timer >= self.min_state_duration:
                # Current reversal rate
                rate = self.base_rev_rate * np.exp(
                    -self.sensory_gain * self.dCdt_smooth)
                rate *= (1.0 + ava_activity)
                rate = np.clip(rate, 1e-6, 10.0)
                # Countdown scaled by current rate vs rate when drawn
                self._next_event_time -= dt * rate / max(self.base_rev_rate * 1.5, 1e-6)
                if self._next_event_time <= 0:
                    self.state = self.BACKWARD
                    self.state_timer = 0.0
                    did_reverse = True
                    # Draw next backward->forward wait
                    self._next_event_time = self._draw_wait(
                        1.0 / self.mean_back_duration)

        elif self.state == self.BACKWARD:
            if self.state_timer >= self.min_state_duration:
                self._next_event_time -= dt
                if self._next_event_time <= 0:
                    self.state = self.FORWARD
                    self.state_timer = 0.0
                    did_resume = True
                    # Draw next forward->backward wait
                    rate = self.base_rev_rate * np.exp(
                        -self.sensory_gain * self.dCdt_smooth)
                    rate *= (1.0 + ava_activity)
                    rate = np.clip(rate, 1e-6, 10.0)
                    self._next_event_time = self._draw_wait(rate)

        return self.state, did_reverse, did_resume


def reversal_turn_angle(dCdt_smooth, rng, base_angle=np.pi / 2,
                        dCdt_gain=2.0):
    """Sample turn angle for a pirouette (klinokinesis).

    Larger turns when moving down-gradient, smaller when up-gradient.

    Parameters
    ----------
    dCdt_smooth : float
        Smoothed dC/dt at reversal time.
    rng : numpy.random.RandomState
        RNG for sampling.
    base_angle : float
        Mean angular spread (radians).
    dCdt_gain : float
        Exponential gain on dCdt modulation.

    Returns
    -------
    angle : float
        Total heading change (radians).  Centered near pi (reversal).
    """
    # Wider turns when dCdt < 0 (moving down-gradient)
    scale = base_angle * np.exp(-dCdt_gain * dCdt_smooth)
    scale = np.clip(scale, 0.1, np.pi)
    offset = rng.normal(0, scale)
    return np.pi + offset


class WeathervaneCircuit:
    """Heading bias during forward locomotion from bilateral asymmetry.

    Reads bilateral SMD motor neuron activity and produces a heading
    torque that curves the trajectory toward higher concentration.

    Parameters
    ----------
    gain : float
        Torque per unit activity difference (rad/s).
    """

    def __init__(self, gain=0.5):
        self.gain = gain

    def heading_torque(self, smd_left_activity, smd_right_activity):
        """Compute heading torque from SMD bilateral difference.

        Parameters
        ----------
        smd_left_activity, smd_right_activity : float
            Activity [0, 1] of left and right SMD neurons.

        Returns
        -------
        torque : float
            Angular velocity contribution (rad/s).
            Positive = counterclockwise (toward left).
        """
        return self.gain * (smd_left_activity - smd_right_activity)
