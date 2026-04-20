"""
Map B: narrow-channel lateral alignment scene for APT.
"""
import os

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_CACHE_ROOT = os.path.join(_PROJECT_ROOT, '.cache')
os.makedirs(os.path.join(_CACHE_ROOT, 'matplotlib'), exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_CACHE_ROOT, 'matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .base_map import BaseMapEnv


class APTAlignmentEnv(BaseMapEnv):
    """
    Narrow alignment corridor. Heading stays nearly fixed while the vehicle
    must move laterally into the upper lane.
    """

    def __init__(
        self,
        channel_width=1.2,
        entry_x=1.0,
        transfer_x=5.8,
        target_y=1.9,
        exit_x=12.0,
        initial_lateral_offset=0.0,
        initial_heading_offset=0.0,
    ):
        super().__init__()
        self.x_range = (0.0, 14.0)
        self.y_range = (-3.5, 3.5)
        self.channel_width = float(channel_width)
        self.entry_x = float(entry_x)
        self.transfer_x = float(transfer_x)
        self.target_y = float(target_y)
        self.exit_x = float(exit_x)
        self.afm_tracking_bias = 1
        self.afm_curvature_lookahead = 25
        self.afm_control_rate_weight = 0.70
        self.afm_lateral_velocity_weight = 0.14
        self.afm_yaw_rate_weight = 0.08
        self.afm_stage_longitudinal_weight = 1.0
        self.afm_stage_lateral_weight = 9.2
        self.afm_stage_heading_weight = 5.0
        self.afm_stage_speed_weight = 4.0
        self.afm_path_lateral_deadband = 0.05
        self.afm_path_heading_deadband = np.deg2rad(2.0)
        self.manual_apt_switch_x = self.transfer_x - 0.15
        self.manual_apt_resume_x = self.transfer_x + 0.05
        half_width = self.channel_width / 2.0
        lower_center = 0.0

        self.drivable_areas = {
            'lower_entry': {
                'type': 'rectangle',
                'x_min': self.entry_x,
                'y_min': lower_center - half_width,
                'x_max': self.transfer_x,
                'y_max': lower_center + half_width
            },
            'transfer_column': {
                'type': 'rectangle',
                'x_min': max(self.entry_x, self.transfer_x - 0.35),
                'y_min': lower_center - half_width,
                'x_max': self.transfer_x + 0.35,
                'y_max': self.target_y + half_width
            },
            'upper_corridor': {
                'type': 'rectangle',
                'x_min': self.transfer_x + 0.35,
                'y_min': self.target_y - half_width,
                'x_max': self.exit_x,
                'y_max': self.target_y + half_width
            },
        }

        self.initial_state = {
            'x': self.entry_x,
            'y': lower_center + float(initial_lateral_offset),
            'psi': float(initial_heading_offset)
        }
        self.end_point = (self.exit_x, self.target_y)
        self.end_heading = 0.0
        self.reference_path = self._generate_reference_path()
        self.reset()

    def _generate_reference_path(self) -> np.ndarray:
        points = []

        def append_segment(start, end, heading=0.0, interval=0.1):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = float(np.hypot(dx, dy))
            num = max(2, int(length / interval) + 1)
            for i in range(num):
                t = i / (num - 1)
                x = start[0] + t * dx
                y = start[1] + t * dy
                points.append([x, y, heading])

        append_segment((self.entry_x, 0.0), (self.transfer_x, 0.0), heading=0.0)
        append_segment((self.transfer_x, 0.0), (self.transfer_x, self.target_y), heading=0.0)
        append_segment((self.transfer_x, self.target_y), (self.exit_x, self.target_y), heading=0.0)

        return np.asarray(points, dtype=float)

    def draw_track(self, ax: plt.Axes):
        ax.set_xlim(self.x_range[0] - 1, self.x_range[1] + 1)
        ax.set_ylim(self.y_range[0] - 1, self.y_range[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.fill_between([self.x_range[0], self.x_range[1]], self.y_range[0], self.y_range[1], color='gray', alpha=1.0)

        for name, area in self.drivable_areas.items():
            ax.fill_between([area['x_min'], area['x_max']], area['y_min'], area['y_max'], color='white', alpha=1.0, label=name.replace('_', ' ').title())
            ax.plot([area['x_min'], area['x_max'], area['x_max'], area['x_min'], area['x_min']], [area['y_min'], area['y_min'], area['y_max'], area['y_max'], area['y_min']], color='r', linewidth=2)

        ax.plot(self.reference_path[:, 0], self.reference_path[:, 1], 'g--', linewidth=1.6, label='Reference Path')
        ax.plot(self.initial_state['x'], self.initial_state['y'], 'ro', markersize=10, label='Start Point')
        ax.plot(self.end_point[0], self.end_point[1], 'gs', markersize=10, label='End Point')
        self._draw_vehicle_at(ax, self.initial_state['x'], self.initial_state['y'], self.initial_state['psi'], color='red', label='Start Vehicle')
        self._draw_vehicle_at(ax, self.end_point[0], self.end_point[1], self.end_heading, color='green', label='End Vehicle')
