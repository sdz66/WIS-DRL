"""
AZR (Autonomous Zero Radius) module
For vehicle in-place direction reversal
"""
import numpy as np
import csv
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from map_manager import MapManager
from controllers.casadi_nmpc_robust import CasADiNMPCRobust
from controllers.azr_step import AZRStep


class AZR:
    """
    Autonomous Zero Radius class
    For vehicle in-place direction reversal
    """
    
    def __init__(self, map_type='map_c'):
        """
        Initialize AZR
        
        Args:
            map_type: Map type to use for movement
        """
        self.map_type = map_type
        self.manager = MapManager()
        self.env = self.manager.create_map(map_type)
        self.nmpc = CasADiNMPCRobust(dt=0.1, horizon=5)
        self.step_controller = AZRStep(map_type=map_type, dt=0.1)
        self.dt = 0.1
        self.heading_tolerance = np.deg2rad(4.0)
        self.last_states = np.empty((0, 4))
        self.last_controls = np.empty((0, 3))
        self.last_time = 0.0
    
    def wrap_angle(self, ang):
        """
        Normalize angle to [-π, π]
        """
        return np.arctan2(np.sin(ang), np.cos(ang))
    
    def reverse_direction(self, position, start_heading, return_trajectory=False, max_steps=120):
        """
        Reverse vehicle direction in-place - direct angle reversal
        
        Args:
            position: Vehicle position (x, y)
            start_heading: Start heading (rad)
        
        Returns:
            final_position: Final position (x, y)
            final_heading: Final heading (rad)
        """
        # Calculate target heading (opposite direction) - direct angle reversal
        target_heading = start_heading + np.pi
        target_heading = np.arctan2(np.sin(target_heading), np.cos(target_heading))  # Normalize to [-pi, pi]
        
        # Update map with current position and headings
        self.env.set_start_point(position)
        self.env.set_start_heading(start_heading)
        self.env.set_end_point(position)
        self.env.set_end_heading(target_heading)
        
        print(f"Starting AZR direction reversal - {self.map_type}")
        print(f"Direct angle reversal: {np.degrees(start_heading):.2f} deg -> {np.degrees(target_heading):.2f} deg")

        state = np.array([position[0], position[1], start_heading, 0.0], dtype=float)
        self.step_controller.reset()
        self.step_controller.set_rotation_target(current_heading=start_heading, target_heading=target_heading)

        states = [state.copy()]
        controls = []
        for _ in range(max_steps):
            if self.step_controller.is_complete(state):
                break
            u = self.step_controller.step(state)
            state = self._integrate_state(state, u)
            states.append(state.copy())
            controls.append(u.copy())

        self.last_states = np.array(states)
        self.last_controls = np.array(controls) if controls else np.empty((0, 3))
        self.last_time = max(0.0, (len(states) - 1) * self.dt)

        final_state = self.last_states[-1]
        rmse = float(np.linalg.norm(final_state[:2] - np.array(position, dtype=float)))
        heading_error = float(np.degrees(abs(self.wrap_angle(target_heading - final_state[2]))))
        success_rate = 100.0 if rmse < 0.05 and heading_error < 5.0 else 0.0

        positions = [s[:2] for s in self.last_states]
        headings = [s[2] for s in self.last_states]
        
        # Save positions and headings to CSV file
        csv_path = f'outputs/{self.map_type}-azr_positions.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'heading'])
            for pos, heading in zip(positions, headings):
                writer.writerow([pos[0], pos[1], heading])
        
        print(f"\n{self.map_type} AZR results:")
        print(f"RMSE: {rmse:.4f} m")
        print(f"Heading error: {heading_error:.4f} deg")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Completion time: {self.last_time:.2f} s")
        print(f"Positions saved to: {csv_path}")
        
        # Return final position and heading
        result = (final_state[:2].copy(), final_state[2])
        if return_trajectory:
            return result[0], result[1], self.last_states.copy()
        return result

    def _integrate_state(self, state, command):
        x, y, psi, _ = state
        vx, vy, omega = command

        x_new = x + (vx * np.cos(psi) - vy * np.sin(psi)) * self.dt
        y_new = y + (vx * np.sin(psi) + vy * np.cos(psi)) * self.dt
        psi_new = self.wrap_angle(psi + omega * self.dt)
        speed = np.hypot(vx, vy)
        return np.array([x_new, y_new, psi_new, speed], dtype=float)


if __name__ == "__main__":
    """
    Example usage of AZR
    """
    # Create AZR instance
    azr = AZR(map_type='map_c')
    
    # Define position and start heading
    position = (5.0, 0.0)
    start_heading = 0.0  # Facing right
    
    # Reverse direction
    final_position, final_heading = azr.reverse_direction(position, start_heading)
    
    print(f"\nAZR direction reversal completed!")
    print(f"Final position: {final_position}")
    print(f"Final heading: {np.degrees(final_heading):.2f} deg")
