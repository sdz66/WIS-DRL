"""
APT Step Controller - Step-based Autonomous Parallel Translation
Provides single-step commands for continuous lateral correction.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class APTStep:
    """
    Step-based APT (Autonomous Parallel Translation) Controller

    The RL policy only selects the mode. This controller therefore keeps its own
    translation target and emits smooth body-frame commands that use the wider
    holonomic envelope to correct both lateral and small longitudinal drift.
    """

    def __init__(self, map_type='map_b', dt=0.1, translation_speed=0.5):
        """
        Initialize APT step controller

        Args:
            map_type: Map type (for compatibility)
            dt: Time step (seconds)
            translation_speed: Maximum lateral translation speed (m/s)
        """
        self.map_type = map_type
        self.dt = dt
        self.translation_speed = translation_speed

        # Control gains and limits. APT is treated as a continuous correction
        # primitive: it keeps the entry heading as a reference while allowing a
        # slightly larger holonomic envelope to clean up lateral offset and a
        # small amount of longitudinal drift.
        self.kp_longitudinal = 0.35
        self.kp_lateral = 1.0
        self.max_longitudinal_speed = 0.30
        self.max_lateral_speed = translation_speed
        self.max_yaw_rate = 0.0
        self.position_tolerance = 0.04
        self.longitudinal_tolerance = 0.10
        self.heading_tolerance = np.deg2rad(4.0)
        self.command_rate_limit = 0.05
        self.command_deadband = 0.001
        self.filtered_command = np.zeros(3, dtype=float)

        self.reset()

    def reset(self):
        """Reset controller state."""
        self.target_lateral_offset = 0.0
        self.current_lateral_offset = 0.0
        self.target_position = None
        self.translation_direction = 1.0  # +1 left, -1 right in vehicle frame
        self.start_position = None
        self.reference_heading = None
        self.active = False
        self.filtered_command = np.zeros(3, dtype=float)

    def set_translation_target(self, state, direction='left', distance=1.0, heading_hold=None):
        """
        Set a new lateral translation target from the current state.

        Args:
            state: Current vehicle state [x, y, psi, v]
            direction: 'left' or 'right'
            distance: Translation distance in meters
            heading_hold: Optional heading to hold during translation
        """
        self.start_position = np.array(state[:2], dtype=float)
        self.reference_heading = float(state[2] if heading_hold is None else heading_hold)
        self.translation_direction = 1.0 if direction == 'left' else -1.0
        distance = max(0.0, float(distance))
        self.target_lateral_offset = self.translation_direction * distance
        left_unit = np.array([
            -np.sin(self.reference_heading),
            np.cos(self.reference_heading)
        ], dtype=float)
        self.target_position = self.start_position + left_unit * self.target_lateral_offset
        self.current_lateral_offset = 0.0
        self.active = abs(self.target_lateral_offset) > self.position_tolerance
        self.filtered_command = np.zeros(3, dtype=float)

    def is_complete(self, state):
        """Check whether the active translation target has been reached."""
        if (
            not self.active or
            self.start_position is None or
            self.reference_heading is None or
            self.target_position is None
        ):
            return True

        local_x, local_y = self._project_to_reference_frame(state)
        self.current_lateral_offset = local_y

        remaining = self.target_lateral_offset - self.current_lateral_offset
        longitudinal_drift = abs(local_x)
        heading_error = abs(self._normalize_angle(float(state[2]) - self.reference_heading))

        return (
            abs(remaining) <= self.position_tolerance and
            longitudinal_drift <= self.longitudinal_tolerance and
            heading_error <= self.heading_tolerance
        )

    def step(self, state):
        """
        Execute one control step and return a holonomic command.

        Args:
            state: Current vehicle state [x, y, psi, v]

        Returns:
            np.ndarray: Body-frame command [vx, vy, omega]
        """
        if not self.active or self.start_position is None or self.reference_heading is None:
            return np.array([0.0, 0.0, 0.0])

        local_x, local_y = self._project_to_reference_frame(state)
        self.current_lateral_offset = local_y

        lateral_error = self.target_lateral_offset - local_y
        longitudinal_error = -local_x

        if self.is_complete(state):
            self.active = False
            self.filtered_command = np.zeros(3, dtype=float)
            return np.array([0.0, 0.0, 0.0])

        # Keep the reference frame aligned with the mode-entry heading and
        # compute a smooth correction command in that frame before rotating it
        # into the current body frame.
        ref_vx = np.clip(
            self.kp_longitudinal * longitudinal_error,
            -self.max_longitudinal_speed,
            self.max_longitudinal_speed
        )
        ref_vy = np.clip(
            self.kp_lateral * lateral_error,
            -self.max_lateral_speed,
            self.max_lateral_speed
        )

        c_ref = np.cos(self.reference_heading)
        s_ref = np.sin(self.reference_heading)
        world_vx = c_ref * ref_vx - s_ref * ref_vy
        world_vy = s_ref * ref_vx + c_ref * ref_vy

        psi = float(state[2])
        c = np.cos(psi)
        s = np.sin(psi)
        body_vx = c * world_vx + s * world_vy
        body_vy = -s * world_vx + c * world_vy
        target_command = np.array([body_vx, body_vy, 0.0], dtype=float)
        delta = np.clip(
            target_command - self.filtered_command,
            -self.command_rate_limit,
            self.command_rate_limit
        )
        self.filtered_command = self.filtered_command + delta

        for i in range(3):
            if abs(self.filtered_command[i]) < self.command_deadband:
                self.filtered_command[i] = 0.0

        return self.filtered_command.copy()

    def _project_to_reference_frame(self, state):
        """Project current position into the mode-entry local frame."""
        delta_pos = np.array(state[:2], dtype=float) - self.start_position
        c = np.cos(self.reference_heading)
        s = np.sin(self.reference_heading)
        local_x = c * delta_pos[0] + s * delta_pos[1]
        local_y = -s * delta_pos[0] + c * delta_pos[1]
        return local_x, local_y

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))


if __name__ == "__main__":
    # Test APT step controller
    print("Testing APTStep controller...")
    
    apt = APTStep(map_type='map_b')
    
    # Set translation target: move left by 1 meter
    test_state = np.array([5.0, 0.0, 0.0, 0.0])
    apt.set_translation_target(test_state, direction='left', distance=1.0)
    
    print(f"Initial state: {test_state}")
    print(f"Translation target: left, 1.0m")
    
    # Simulate several steps
    for i in range(20):
        u = apt.step(test_state)
        
        # Holonomic update for testing
        x, y, psi, _ = test_state
        vx, vy, omega = u
        dt = 0.1
        x_new = x + (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        y_new = y + (vx * np.sin(psi) + vy * np.cos(psi)) * dt
        psi_new = apt._normalize_angle(psi + omega * dt)
        v_new = np.hypot(vx, vy)
        test_state = np.array([x_new, y_new, psi_new, v_new])
        
        print(f"Step {i+1}: u={[f'{val:.3f}' for val in u]}, "
              f"state=[{test_state[0]:.2f}, {test_state[1]:.2f}, "
              f"{np.degrees(test_state[2]):.1f}°, {test_state[3]:.2f}]")
    
    print("Test completed!")
