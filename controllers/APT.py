"""
APT (Autonomous Parallel Translation) module
For vehicle left-right parallel translation
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from map_manager import MapManager
from controllers.casadi_nmpc_robust import CasADiNMPCRobust


class APT:
    """
    Autonomous Parallel Translation class
    For vehicle left-right parallel translation
    """
    
    def __init__(self, map_type='map_b'):
        """
        Initialize APT
        
        Args:
            map_type: Map type to use for movement
        """
        self.map_type = map_type
        self.manager = MapManager()
        self.env = self.manager.create_map(map_type)
        self.nmpc = CasADiNMPCRobust(dt=0.1, horizon=5)
        self.dt = 0.1
        self.kp_position = 0.9
        self.max_linear_speed = 0.5
        self.position_tolerance = 0.05
        self.command_smoothing = 0.6
        self.filtered_command = np.zeros(3, dtype=float)
        self.heading_tolerance = np.deg2rad(4.0)
        self.last_states = np.empty((0, 4))
        self.last_controls = np.empty((0, 3))
        self.last_time = 0.0
    
    def translate(self, start, end, direction='left', return_trajectory=False, max_steps=300):
        """
        Translate vehicle from start to end in left or right direction
        
        Args:
            start: Start position and heading (x, y, psi)
            end: End position and heading (x, y, psi)
            direction: Translation direction ('left' or 'right')
        
        Returns:
            final_position: Final position (x, y)
            final_heading: Final heading (rad)
        """
        # Update map with new start and end
        self.env.set_start_point(start[:2])
        self.env.set_start_heading(start[2])
        self.env.set_end_point(end[:2])
        self.env.set_end_heading(end[2])
        
        print(f"Starting APT translation - {self.map_type} ({direction})")

        state = np.array([start[0], start[1], start[2], 0.0], dtype=float)
        target_position = np.array(end[:2], dtype=float)
        target_heading = float(start[2])
        self.filtered_command = np.zeros(3, dtype=float)

        states = [state.copy()]
        controls = []
        for _ in range(max_steps):
            position_error = target_position - state[:2]
            distance = np.linalg.norm(position_error)
            heading_error = self._wrap_angle(target_heading - state[2])
            if distance < self.position_tolerance and abs(heading_error) < self.heading_tolerance:
                break

            local_error = self._project_to_body_frame(state, target_position)
            vx_target = np.clip(self.kp_position * local_error[0], -self.max_linear_speed, self.max_linear_speed)
            vy_target = np.clip(self.kp_position * local_error[1], -self.max_linear_speed, self.max_linear_speed)
            command = np.array([vx_target, vy_target, 0.0], dtype=float)
            self.filtered_command = (
                self.command_smoothing * self.filtered_command +
                (1.0 - self.command_smoothing) * command
            )
            self.filtered_command[2] = 0.0

            command = self.filtered_command.copy()
            state = self._integrate_state(state, command)
            states.append(state.copy())
            controls.append(command)

        self.last_states = np.array(states)
        self.last_controls = np.array(controls) if controls else np.empty((0, 3))
        self.last_time = max(0.0, (len(states) - 1) * self.dt)

        final_state = self.last_states[-1]
        rmse = float(np.linalg.norm(final_state[:2] - target_position))
        heading_error = float(np.degrees(abs(self._wrap_angle(final_state[2] - target_heading))))
        success_rate = 100.0 if rmse < 0.2 and heading_error < 5.0 else 0.0

        positions = [s[:2] for s in self.last_states]

        # Save positions to CSV file
        csv_path = f'outputs/{self.map_type}-apt_positions.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y'])
            for pos in positions:
                writer.writerow([pos[0], pos[1]])
        
        print(f"\n{self.map_type} APT results:")
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
    
    def _generate_parallel_path(self, start, end, direction):
        """
        Generate parallel translation path
        
        Args:
            start: Start position and heading (x, y, psi)
            end: End position and heading (x, y, psi)
            direction: Translation direction ('left' or 'right')
        
        Returns:
            np.ndarray: Reference path for parallel translation
        """
        # Calculate translation distance
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate translation direction vector
        if distance > 0:
            dir_x = dx / distance
            dir_y = dy / distance
        else:
            dir_x, dir_y = 1.0, 0.0
        
        # Calculate perpendicular direction for parallel translation
        if direction == 'left':
            perp_x = -dir_y
            perp_y = dir_x
        else:  # right
            perp_x = dir_y
            perp_y = -dir_x
        
        # Generate path points
        interval = 0.1
        num_points = int(distance / interval) + 1
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            
            # Base position along the line from start to end
            x = start[0] + t * dx
            y = start[1] + t * dy
            
            # Add perpendicular offset for parallel translation
            offset = 0.5  # 0.5 meters offset
            x += offset * perp_x
            y += offset * perp_y
            
            # Calculate heading (same as start heading)
            heading = start[2]
            
            points.append([x, y, heading])
        
        return np.array(points)

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _project_to_body_frame(state, point):
        dx = point[0] - state[0]
        dy = point[1] - state[1]
        psi = state[2]
        local_x = np.cos(psi) * dx + np.sin(psi) * dy
        local_y = -np.sin(psi) * dx + np.cos(psi) * dy
        return np.array([local_x, local_y])

    def _integrate_state(self, state, command):
        x, y, psi, _ = state
        vx, vy, omega = command
        x_new = x + (vx * np.cos(psi) - vy * np.sin(psi)) * self.dt
        y_new = y + (vx * np.sin(psi) + vy * np.cos(psi)) * self.dt
        psi_new = self._wrap_angle(psi + omega * self.dt)
        speed = np.hypot(vx, vy)
        return np.array([x_new, y_new, psi_new, speed], dtype=float)


if __name__ == "__main__":
    """
    Example usage of APT
    """
    # Create APT instance
    apt = APT(map_type='map_b')
    
    # Define start and end
    start = (1.0, 0.0, 0.0)  # (x, y, psi)
    end = (9.0, 0.0, 0.0)   # (x, y, psi)
    
    # Translate from start to end (left direction)
    final_position, final_heading = apt.translate(start, end, direction='left')
    
    print(f"\nAPT translation completed!")
    print(f"Final position: {final_position}")
    print(f"Final heading: {np.degrees(final_heading):.2f} deg")
