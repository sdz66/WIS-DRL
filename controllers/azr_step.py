"""
AZR Step Controller - Step-based Autonomous Zero Radius Rotation
Provides single-step commands for true in-place rotation.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class AZRStep:
    """
    Step-based AZR (Autonomous Zero Radius) Controller

    Emits holonomic body commands so the simulated robot can rotate in place
    instead of approximating the motion with a bicycle model.
    """

    def __init__(self, map_type='map_c', dt=0.1, rotation_speed=1.5):
        """
        Initialize AZR step controller

        Args:
            map_type: Map type (for compatibility)
            dt: Time step (seconds)
            rotation_speed: Maximum rotation speed (rad/s)
        """
        self.map_type = map_type
        self.dt = dt
        self.rotation_speed = rotation_speed
        self.heading_tolerance = np.deg2rad(2.5)
        self.kp_heading = 2.2
        self.angular_deadband = np.deg2rad(1.5)
        self.omega_rate_limit = 0.25
        self.reset()

    def reset(self):
        """Reset controller state."""
        self.target_heading = None
        self.start_heading = None
        self.is_rotating = False
        self.filtered_omega = 0.0

    def set_rotation_target(self, current_heading, target_heading=None, target_delta=np.pi):
        """
        Set rotation target.

        Args:
            current_heading: Current vehicle heading (rad)
            target_heading: Optional absolute target heading
            target_delta: Desired relative heading change if target_heading is None
        """
        self.start_heading = float(current_heading)
        if target_heading is None:
            target_heading = current_heading + target_delta
        self.target_heading = self._normalize_angle(target_heading)
        self.is_rotating = True

    def is_complete(self, state):
        """Check whether the active rotation target has been reached."""
        if not self.is_rotating or self.target_heading is None:
            return True
        heading_error = self._normalize_angle(self.target_heading - state[2])
        return abs(heading_error) < self.heading_tolerance

    def step(self, state):
        """
        Execute one control step and return a holonomic command.

        Args:
            state: Current vehicle state [x, y, psi, v]

        Returns:
            np.ndarray: Body-frame command [vx, vy, omega]
        """
        psi = state[2]

        if self.target_heading is None or not self.is_rotating:
            self.set_rotation_target(psi, target_delta=np.pi)

        heading_error = self._normalize_angle(self.target_heading - psi)

        if self.is_complete(state):
            self.is_rotating = False
            self.filtered_omega = 0.0
            return np.array([0.0, 0.0, 0.0])

        target_omega = np.clip(
            self.kp_heading * heading_error,
            -self.rotation_speed,
            self.rotation_speed
        )
        delta = np.clip(
            target_omega - self.filtered_omega,
            -self.omega_rate_limit,
            self.omega_rate_limit
        )
        self.filtered_omega += delta

        if abs(heading_error) < self.angular_deadband:
            self.filtered_omega *= abs(heading_error) / max(self.angular_deadband, 1e-6)

        if abs(self.filtered_omega) < 0.02:
            self.filtered_omega = 0.0

        return np.array([0.0, 0.0, self.filtered_omega])

    @staticmethod
    def _normalize_angle(angle):
        """
        Normalize angle to [-pi, pi]

        Args:
            angle: Input angle (rad)

        Returns:
            float: Normalized angle
        """
        return np.arctan2(np.sin(angle), np.cos(angle))


if __name__ == "__main__":
    # Test AZR step controller
    print("Testing AZRStep controller...")

    azr = AZRStep(map_type='map_c')

    # Test state: vehicle at position facing right (0 rad)
    test_state = np.array([5.0, 0.0, 0.0, 0.0])

    print(f"Initial state: {test_state}")
    print(f"Initial heading: {np.degrees(test_state[2]):.1f} deg")

    # Set rotation target: rotate 180 degrees
    azr.set_rotation_target(current_heading=test_state[2], target_delta=np.pi)

    print(f"Target heading: {np.degrees(azr.target_heading):.1f} deg")

    # Simulate rotation steps
    for i in range(30):
        u = azr.step(test_state)

        # Holonomic update for testing
        x, y, psi, _ = test_state
        vx, vy, omega = u
        dt = 0.1
        x_new = x + (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        y_new = y + (vx * np.sin(psi) + vy * np.cos(psi)) * dt
        psi_new = azr._normalize_angle(psi + omega * dt)
        v_new = np.hypot(vx, vy)
        test_state = np.array([x_new, y_new, psi_new, v_new])

        heading_error_deg = np.degrees(azr._normalize_angle(azr.target_heading - test_state[2]))

        print(f"Step {i+1}: u={[f'{val:.3f}' for val in u]}, "
              f"heading={np.degrees(test_state[2]):6.1f} deg, "
              f"error={heading_error_deg:+6.1f} deg, "
              f"v={test_state[3]:.2f}")

        if abs(heading_error_deg) < 5:
            print(f"Rotation completed at step {i+1}!")
            break

    print("Test completed!")
