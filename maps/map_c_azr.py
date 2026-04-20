"""
Map C: small-space reorientation scene for AZR.
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


class AZRReorientationEnv(BaseMapEnv):
    """
    Compact pocket with a heading-reversal anchor that makes AZR the natural
    primitive.
    """

    def __init__(
        self,
        pocket_width=2.4,
        pocket_length=2.2,
        entry_x=1.0,
        entry_length=3.4,
        rotation_x=None,
        exit_length=3.0,
        initial_heading_offset=np.pi,
        target_heading=0.0,
    ):
        super().__init__()
        self.pocket_width = float(pocket_width)
        self.pocket_length = float(pocket_length)
        self.entry_x = float(entry_x)
        self.entry_length = float(entry_length)
        self.rotation_x = float(rotation_x) if rotation_x is not None else self.entry_x + self.entry_length
        self.exit_length = float(exit_length)
        self.afm_tracking_bias = 2
        self.afm_curvature_lookahead = 28
        self.afm_control_rate_weight = 0.85
        self.afm_lateral_velocity_weight = 0.18
        self.afm_yaw_rate_weight = 0.08
        self.afm_stage_longitudinal_weight = 0.85
        self.afm_stage_lateral_weight = 12.0
        self.afm_stage_heading_weight = 6.2
        self.afm_stage_speed_weight = 4.0
        self.afm_path_lateral_deadband = 0.02
        self.afm_path_heading_deadband = np.deg2rad(0.9)
        self.afm_reference_speed_cap = 1.0
        self.afm_reference_speed_floor = 0.45
        self.afm_terminal_slowdown_distance = 0.1
        # Handover recovery profile: once AZR has rotated the vehicle into the
        # exit corridor, AFM should react more aggressively to any residual
        # cross-track or heading error without changing the tuning of other maps.
        self.afm_handover_recovery_x = self.entry_x
        self.afm_handover_heading_limit = np.deg2rad(25.0)
        self.afm_handover_tracking_bias = 2
        self.afm_handover_curvature_lookahead = 6
        self.afm_handover_control_rate_weight = 0.08
        self.afm_handover_lateral_velocity_weight = 0.02
        self.afm_handover_yaw_rate_weight = 0.01
        self.afm_handover_stage_longitudinal_weight = 0.55
        self.afm_handover_stage_lateral_weight = 32.0
        self.afm_handover_stage_heading_weight = 24.0
        self.afm_handover_stage_speed_weight = 4.5
        self.afm_handover_path_lateral_deadband = 0.0
        self.afm_handover_path_heading_deadband = np.deg2rad(0.08)
        self.afm_handover_reference_speed_cap = 0.20
        self.afm_handover_reference_speed_floor = 0.0
        self.afm_handover_terminal_slowdown_distance = 0.80
        self.manual_azr_switch_x = self.entry_x + 0.05
        self.manual_azr_resume_x = self.entry_x + 0.15
        self.afm_handover_recovery_snap_to_reference = True
        self.afm_handover_recovery_lateral_threshold = 0.020
        self.afm_handover_recovery_heading_threshold = np.deg2rad(1.0)
        self.afm_handover_recovery_heading_only = True
        self.afm_handover_recovery_hold_steps = 1
        self.afm_handover_recovery_max_steps = 20
        self.afm_handover_recovery_control_rate_weight = 0.03
        self.afm_handover_recovery_lateral_velocity_weight = 0.01
        self.afm_handover_recovery_yaw_rate_weight = 0.005
        self.afm_handover_recovery_stage_longitudinal_weight = 0.45
        self.afm_handover_recovery_stage_lateral_weight = 60.0
        self.afm_handover_recovery_stage_heading_weight = 60.0
        self.afm_handover_recovery_stage_speed_weight = 2.0
        self.afm_handover_recovery_path_lateral_deadband = 0.0
        self.afm_handover_recovery_path_heading_deadband = np.deg2rad(0.02)
        self.afm_handover_recovery_reference_speed_cap = 0.45
        self.afm_handover_recovery_reference_speed_floor = 0.15
        self.afm_handover_recovery_terminal_slowdown_distance = 0.25
        half_width = self.pocket_width / 2.0
        exit_x = self.rotation_x + self.pocket_length + self.exit_length
        self.x_range = (0.0, max(11.0, exit_x + 1.5))
        self.y_range = (-2.5, 2.5)

        self.drivable_areas = {
            'entry_corridor': {
                'type': 'rectangle',
                'x_min': self.entry_x,
                'y_min': -0.55,
                'x_max': self.rotation_x,
                'y_max': 0.55
            },
            'rotation_pocket': {
                'type': 'rectangle',
                'x_min': self.rotation_x,
                'y_min': -half_width,
                'x_max': self.rotation_x + self.pocket_length,
                'y_max': half_width
            },
            'exit_corridor': {
                'type': 'rectangle',
                'x_min': self.rotation_x,
                'y_min': -0.55,
                'x_max': exit_x,
                'y_max': 0.55
            },
        }

        self.initial_state = {'x': self.entry_x, 'y': 0.0, 'psi': float(initial_heading_offset)}
        self.end_point = (exit_x, 0.0)
        self.end_heading = float(target_heading)
        self.reference_path = self._generate_reference_path()
        self.reset()

    def _generate_reference_path(self) -> np.ndarray:
        points = []

        def append_segment(start, end, heading, interval=0.1):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = float(np.hypot(dx, dy))
            num = max(2, int(length / interval) + 1)
            for i in range(num):
                t = i / (num - 1)
                x = start[0] + t * dx
                y = start[1] + t * dy
                points.append([x, y, heading])

        append_segment((self.entry_x, 0.0), (self.rotation_x, 0.0), 0.0)
        append_segment((self.rotation_x, 0.0), (self.end_point[0], 0.0), self.end_heading)

        return np.asarray(points, dtype=float)

    def draw_track(self, ax: plt.Axes):
        ax.set_xlim(self.x_range[0] - 0.5, self.x_range[1] + 0.5)
        ax.set_ylim(self.y_range[0] - 0.5, self.y_range[1] + 0.5)
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
