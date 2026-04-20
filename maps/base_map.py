"""
Base map environment class
Provides functionality shared by all maps
"""
import os
from typing import Tuple, Optional

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_CACHE_ROOT = os.path.join(_PROJECT_ROOT, '.cache')
os.makedirs(os.path.join(_CACHE_ROOT, 'matplotlib'), exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_CACHE_ROOT, 'matplotlib'))

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BaseMapEnv:
    """
    Base map environment class
    Provides functionality shared by all maps
    """
    
    def __init__(self):
        # Robot parameters
        self.robot_length = 1.65
        self.robot_width = 0.84
        self.wheelbase = 1.0
        self.track_width = 0.69
        self.max_speed = 5.0
        self.max_steering_angle = np.deg2rad(30)
        
        # Map range
        self.x_range = (0.0, 20.0)
        self.y_range = (0.0, 20.0)
        
        # Initial state
        self.initial_state = {
            'x': 0.0,
            'y': 0.0,
            'psi': 0.0
        }
        
        # End point
        self.end_point = (0.0, 0.0)
        self.end_heading = 0.0
        
        # State
        self.state = None
        self.reset()
    
    def reset(self) -> dict:
        """
        Reset environment to initial state
        Returns:
            dict: Initial state
        """
        self.state = {
            'x': self.initial_state['x'],
            'y': self.initial_state['y'],
            'psi': self.initial_state['psi'],
            'vx': 0.0,
            'vy': 0.0,
            'omega': 0.0
        }
        return self.state
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation
        Returns:
            np.ndarray: Observation array
        """
        return np.array([
            self.state['x'],
            self.state['y'],
            self.state['psi'],
            self.state['vx'],
            self.state['vy'],
            self.state['omega']
        ])
    
    def step(self, vx: float, vy: float, omega: float, dt: float = 0.02) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Environment step function
        Args:
            vx: Longitudinal velocity (m/s)
            vy: Lateral velocity (m/s)
            omega: Angular velocity (rad/s)
            dt: Time step (s)
        Returns:
            Tuple[np.ndarray, float, bool, dict]: (observation, reward, done, info dictionary)
        """
        # Limit speed
        speed = np.sqrt(vx**2 + vy**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
        
        # Update state
        self.state['x'] += (vx * np.cos(self.state['psi']) - vy * np.sin(self.state['psi'])) * dt
        self.state['y'] += (vx * np.sin(self.state['psi']) + vy * np.cos(self.state['psi'])) * dt
        self.state['psi'] += omega * dt
        self.state['psi'] = np.arctan2(np.sin(self.state['psi']), np.cos(self.state['psi']))
        
        self.state['vx'] = vx
        self.state['vy'] = vy
        self.state['omega'] = omega
        
        # Simple reward calculation
        done = False
        reward = 0.0
        info = {}
        
        return self._get_observation(), reward, done, info
    
    def _draw_vehicle_at(self, ax: plt.Axes, x: float, y: float, psi: float, color: str = 'red', label: str = 'Vehicle'):
        """
        Draw vehicle outline and direction at specified position
        
        Args:
            ax: matplotlib axis object
            x: Vehicle x coordinate
            y: Vehicle y coordinate
            psi: Vehicle heading (rad)
            color: Vehicle color
            label: Label
        """
        # Unified vehicle dimensions
        vehicle_length = 1.2  # Unified length
        vehicle_width = 0.8   # Unified width
        half_length = vehicle_length / 2
        half_width = vehicle_width / 2
        
        # Vehicle four corners
        corners = np.array([
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ])
        
        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi), np.cos(psi)]
        ])
        
        # Rotate and translate
        rotated_corners = np.dot(corners, rotation_matrix.T)
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += y
        
        # Draw vehicle outline
        ax.plot(
            [*rotated_corners[:, 0], rotated_corners[0, 0]],
            [*rotated_corners[:, 1], rotated_corners[0, 1]],
            'black', linewidth=1.5
        )
        
        # Fill vehicle
        ax.fill(
            rotated_corners[:, 0],
            rotated_corners[:, 1],
            color=color, alpha=0.7
        )
        
        # Draw direction arrow
        arrow_length = half_length * 1.5
        arrow_dx = arrow_length * np.cos(psi)
        arrow_dy = arrow_length * np.sin(psi)
        ax.arrow(x, y, arrow_dx, arrow_dy, 
                 head_width=0.2, head_length=0.25, 
                 fc='yellow', ec='black', 
                 linewidth=1.5, label=label)
    
    def draw_track(self, ax: plt.Axes):
        """
        Draw the track
        Args:
            ax: matplotlib axis object
        """
        # Draw map range
        ax.set_xlim(self.x_range[0] - 1, self.x_range[1] + 1)
        ax.set_ylim(self.y_range[0] - 1, self.y_range[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        # Draw start and end points
        ax.plot(self.initial_state['x'], self.initial_state['y'], 
                'ro', markersize=10, label='Start Point')
        ax.plot(self.end_point[0], self.end_point[1], 
                'gs', markersize=10, label='End Point')
        
        # Draw start vehicle outline and direction
        self._draw_vehicle_at(ax, 
                             self.initial_state['x'], 
                             self.initial_state['y'], 
                             self.initial_state['psi'], 
                             color='red', 
                             label='Start Vehicle')
        
        # Draw end vehicle outline and direction
        self._draw_vehicle_at(ax, 
                             self.end_point[0], 
                             self.end_point[1], 
                             self.end_heading, 
                             color='green', 
                             label='End Vehicle')
    
    def draw_path(self, ax: plt.Axes, path: Optional[np.ndarray], label: str = 'Path'):
        """
        Draw planned path
        Args:
            ax: matplotlib axis object
            path: Path point array, shape (N, 2)
            label: Path label
        """
        if path is not None and len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label=label)
            ax.plot(path[:, 0], path[:, 1], 'ro', markersize=3, alpha=0.6)
    
    def draw_robot(self, ax: plt.Axes, color: str = 'red'):
        """
        Draw robot
        Args:
            ax: matplotlib axis object
            color: Robot color
        """
        self._draw_vehicle_at(ax, 
                             self.state['x'], 
                             self.state['y'], 
                             self.state['psi'], 
                             color=color, 
                             label='Robot')
    
    def render(self, mode: str = 'human', path: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Render environment
        Args:
            mode: Render mode
            path: Planned path point array, shape (N, 2)
        Returns:
            Optional[np.ndarray]: If mode='rgb_array' returns image array
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.set_xlim(self.x_range[0] - 1, self.x_range[1] + 1)
        ax.set_ylim(self.y_range[0] - 1, self.y_range[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Map Environment', fontsize=14)
        
        self.draw_track(ax)
        self.draw_path(ax, path)
        self.draw_robot(ax)
        
        ax.legend(loc='upper right', fontsize=10)
        
        if mode == 'rgb_array':
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
        else:
            plt.show()
            return None
    
    def set_start_point(self, point: Tuple[float, float]):
        """
        Set start point position
        
        Args:
            point: (x, y) coordinates
        """
        self.initial_state['x'] = point[0]
        self.initial_state['y'] = point[1]
        self.reset()
    
    def set_end_point(self, point: Tuple[float, float]):
        """
        Set end point position
        
        Args:
            point: (x, y) coordinates
        """
        self.end_point = point
    
    def set_start_heading(self, heading: float):
        """
        Set start heading
        
        Args:
            heading: Heading angle (rad)
        """
        self.initial_state['psi'] = heading
        self.reset()
    
    def set_end_heading(self, heading: float):
        """
        Set end heading
        
        Args:
            heading: Heading angle (rad)
        """
        self.end_heading = heading
    
    def swap_start_end(self):
        """
        Swap start and end points
        """
        if hasattr(self, 'end_point'):
            temp_point = (self.initial_state['x'], self.initial_state['y'])
            temp_heading = self.initial_state['psi']
            
            self.set_start_point(self.end_point)
            if hasattr(self, 'end_heading'):
                self.set_start_heading(self.end_heading)
            
            self.set_end_point(temp_point)
            self.end_heading = temp_heading
