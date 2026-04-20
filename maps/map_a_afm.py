"""
Map A: open continuous tracking scene for AFM.
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
from matplotlib import patches
from matplotlib.path import Path
from .base_map import BaseMapEnv


class AFMOpenTrackEnv(BaseMapEnv):
    """
    Wide S-curve corridor designed to make AFM the natural optimal mode.
    """

    def __init__(
        self,
        amplitude=0.4,
        corridor_width=2.8,
        path_length=8.0,
        curve_start=3.0,
        initial_lateral_offset=0.0,
        initial_heading_offset=0.0,
        reference_speed=1.0,
    ):
        super().__init__()
        self.s_curve_start = float(curve_start)
        self.path_length = float(path_length)
        self.s_curve_end = self.s_curve_start + self.path_length
        self.x_range = (0.0, max(14.0, self.s_curve_end + 2.0))
        self.y_range = (-4.0, 4.0)
        self.amplitude = float(amplitude)
        self.corridor_width = float(corridor_width)
        self.reference_speed = float(reference_speed)
        self.afm_tracking_bias = 3
        self.afm_curvature_lookahead = 45
        self.afm_control_rate_weight = 0.55
        self.afm_lateral_velocity_weight = 0.12
        self.afm_yaw_rate_weight = 0.06
        self.afm_stage_longitudinal_weight = 1.1
        self.afm_stage_lateral_weight = 13.0
        self.afm_stage_heading_weight = 6.5
        self.afm_stage_speed_weight = 4.0
        self.afm_path_lateral_deadband = 0.008
        self.afm_path_heading_deadband = np.deg2rad(0.45)
        self.afm_reference_speed_cap = 1.1
        self.afm_reference_speed_floor = 0.55
        self.afm_terminal_slowdown_distance = 0.08

        half_width = self.corridor_width / 2.0
        self.drivable_areas = {
            'entry_open': {'type': 'rectangle', 'x_min': 1.0, 'y_min': -half_width, 'x_max': self.s_curve_start, 'y_max': half_width},
            's_open': {
                'type': 'rectangle',
                'x_min': self.s_curve_start,
                'y_min': -half_width - abs(self.amplitude),
                'x_max': self.s_curve_end,
                'y_max': half_width + abs(self.amplitude)
            },
            'exit_open': {'type': 'rectangle', 'x_min': self.s_curve_end, 'y_min': -half_width, 'x_max': self.s_curve_end + 2.0, 'y_max': half_width},
        }

        self.initial_state = {
            'x': 2.0,
            'y': float(initial_lateral_offset),
            'psi': float(initial_heading_offset)
        }
        self.end_point = (self.s_curve_end + 1.0, 0.0)
        self.end_heading = 0.0
        self.reference_path = self._generate_reference_path()
        self.reset()

    def _generate_reference_path(self) -> np.ndarray:
        dense_x = np.linspace(1.0, self.s_curve_end + 2.0, 1000)
        dense_points = []
        for x in dense_x:
            if x < self.s_curve_start:
                y = 0.0
            elif x < self.s_curve_end:
                y = self.amplitude * np.sin(2.0 * np.pi * (x - self.s_curve_start) / (self.s_curve_end - self.s_curve_start))
            else:
                y = 0.0
            dense_points.append([x, y])
        dense_points = np.asarray(dense_points, dtype=float)

        distances = np.zeros(len(dense_points), dtype=float)
        for i in range(1, len(dense_points)):
            distances[i] = distances[i - 1] + float(np.linalg.norm(dense_points[i] - dense_points[i - 1]))

        sampled = []
        current_distance = 0.0
        while current_distance <= distances[-1]:
            idx = int(np.argmin(np.abs(distances - current_distance)))
            x, y = dense_points[idx]
            if x < self.s_curve_start or x >= self.s_curve_end:
                heading = 0.0
            else:
                dydx = self.amplitude * (2.0 * np.pi / (self.s_curve_end - self.s_curve_start)) * np.cos(
                    2.0 * np.pi * (x - self.s_curve_start) / (self.s_curve_end - self.s_curve_start)
                )
                heading = float(np.arctan2(dydx, 1.0))
            sampled.append([x, y, heading])
            current_distance += 0.1

        if sampled[-1][0] < self.end_point[0] - 0.1:
            sampled.append([self.end_point[0], self.end_point[1], self.end_heading])

        return np.asarray(sampled, dtype=float)

    def draw_track(self, ax: plt.Axes):
        ax.set_xlim(self.x_range[0] - 1, self.x_range[1] + 1)
        ax.set_ylim(self.y_range[0] - 1, self.y_range[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.fill_between([self.x_range[0], self.x_range[1]], self.y_range[0], self.y_range[1], color='gray', alpha=1.0)

        entry = self.drivable_areas['entry_open']
        exit_area = self.drivable_areas['exit_open']
        ax.fill_between([entry['x_min'], entry['x_max']], entry['y_min'], entry['y_max'], color='white', alpha=1.0, label='Entry')
        ax.fill_between([exit_area['x_min'], exit_area['x_max']], exit_area['y_min'], exit_area['y_max'], color='white', alpha=1.0, label='Exit')

        s_curve_x = np.linspace(self.s_curve_start, self.s_curve_end, 300)
        s_curve_y = self.amplitude * np.sin(2.0 * np.pi * (s_curve_x - self.s_curve_start) / (self.s_curve_end - self.s_curve_start))
        s_curve_y_upper = s_curve_y + self.corridor_width / 2.0
        s_curve_y_lower = s_curve_y - self.corridor_width / 2.0
        curve_path = Path([
            *zip(s_curve_x, s_curve_y_lower),
            *zip(reversed(s_curve_x), reversed(s_curve_y_upper)),
        ])
        ax.add_patch(patches.PathPatch(curve_path, facecolor='white', edgecolor='red', linewidth=2))

        ax.plot([entry['x_min'], entry['x_max']], [entry['y_min'], entry['y_min']], 'r-', linewidth=2)
        ax.plot([entry['x_min'], entry['x_max']], [entry['y_max'], entry['y_max']], 'r-', linewidth=2)
        ax.plot([exit_area['x_min'], exit_area['x_max']], [exit_area['y_min'], exit_area['y_min']], 'r-', linewidth=2)
        ax.plot([exit_area['x_min'], exit_area['x_max']], [exit_area['y_max'], exit_area['y_max']], 'r-', linewidth=2)

        ax.plot(self.reference_path[:, 0], self.reference_path[:, 1], 'g--', linewidth=1.6, label='Reference Path')
        ax.plot(self.initial_state['x'], self.initial_state['y'], 'ro', markersize=10, label='Start Point')
        ax.plot(self.end_point[0], self.end_point[1], 'gs', markersize=10, label='End Point')
        self._draw_vehicle_at(ax, self.initial_state['x'], self.initial_state['y'], self.initial_state['psi'], color='red', label='Start Vehicle')
        self._draw_vehicle_at(ax, self.end_point[0], self.end_point[1], self.end_heading, color='green', label='End Vehicle')
