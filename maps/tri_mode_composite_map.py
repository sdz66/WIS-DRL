"""
Tri-mode composite map for AFM + APT + AZR.

Layout:
  1. Lower straight corridor for AFM.
  2. Vertical transfer into an upper lane for APT.
  3. Upper outbound corridor for AFM.
  4. Right-side turnaround pocket for AZR.

The terminal pose finishes on the far left of the map while facing left, so
the last segment is an explicit in-place turn at the upper-right corner.
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


class TriModeCompositeEnv(BaseMapEnv):
    """Composite map that makes AFM general, APT helpful, and AZR necessary."""

    def __init__(
        self,
        lower_center_y=0.0,
        upper_center_y=1.35,
        lower_corridor_width=1.55,
        upper_corridor_width=1.10,
        transfer_x=4.15,
        rotation_x=4.55,
        goal_x=None,
        initial_lateral_offset=0.0,
        initial_heading_offset=0.0,
        target_heading=np.pi,
    ):
        super().__init__()

        self.lower_center_y = float(lower_center_y)
        self.upper_center_y = float(upper_center_y)
        self.lower_corridor_width = float(lower_corridor_width)
        self.upper_corridor_width = float(upper_corridor_width)
        self.transfer_x = float(transfer_x)
        self.rotation_x = float(rotation_x)
        self.goal_x = 1.05 if goal_x is None else float(goal_x)
        self.end_heading = float(target_heading)
        self.azr_switch_side = "right"
        self.apt_release_y = self.upper_center_y - 0.23

        # Nominal switch points used by the rollout script.
        self.apt_switch_x = self.transfer_x - 0.04
        self.apt_resume_x = self.transfer_x + 0.07
        self.azr_switch_x = self.rotation_x - 0.06

        # AFM tuning copied from the paper-facing tracks, but relaxed slightly
        # for the composite route so the controller stays smooth in the two
        # long straight segments.
        self.afm_tracking_bias = 2
        self.afm_curvature_lookahead = 18
        self.afm_search_back_window = 8
        self.afm_search_forward_window = 14
        self.afm_straight_curvature_threshold = 0.03
        self.afm_moderate_curvature_threshold = 0.12
        self.afm_stage_longitudinal_weight = 1.05
        self.afm_stage_lateral_weight = 11.5
        self.afm_stage_heading_weight = 6.0
        self.afm_stage_speed_weight = 4.0
        self.afm_path_lateral_deadband = 0.006
        self.afm_path_heading_deadband = np.deg2rad(0.35)
        self.afm_reference_speed_cap = 1.0
        self.afm_reference_speed_floor = 0.50
        self.afm_terminal_slowdown_distance = 0.10
        self.afm_handover_recovery_x = self.transfer_x
        self.afm_handover_heading_limit = np.deg2rad(25.0)
        self.afm_handover_tracking_bias = 2
        self.afm_handover_curvature_lookahead = 8
        self.afm_handover_control_rate_weight = 0.04
        self.afm_handover_lateral_velocity_weight = 0.01
        self.afm_handover_yaw_rate_weight = 0.005
        self.afm_handover_stage_longitudinal_weight = 0.55
        self.afm_handover_stage_lateral_weight = 34.0
        self.afm_handover_stage_heading_weight = 24.0
        self.afm_handover_stage_speed_weight = 4.2
        self.afm_handover_path_lateral_deadband = 0.0
        self.afm_handover_path_heading_deadband = np.deg2rad(0.08)
        self.afm_handover_reference_speed_cap = 0.32
        self.afm_handover_reference_speed_floor = 0.12
        self.afm_handover_terminal_slowdown_distance = 0.28
        self.afm_handover_recovery_lateral_threshold = 0.020
        self.afm_handover_recovery_heading_threshold = np.deg2rad(1.0)
        self.afm_handover_recovery_hold_steps = 1
        self.afm_handover_recovery_max_steps = 16
        self.afm_handover_recovery_heading_only = True
        self.afm_handover_recovery_snap_to_reference = False
        self.afm_handover_recovery_control_rate_weight = 0.03
        self.afm_handover_recovery_lateral_velocity_weight = 0.01
        self.afm_handover_recovery_yaw_rate_weight = 0.005
        self.afm_handover_recovery_stage_longitudinal_weight = 0.45
        self.afm_handover_recovery_stage_lateral_weight = 48.0
        self.afm_handover_recovery_stage_heading_weight = 48.0
        self.afm_handover_recovery_stage_speed_weight = 2.2
        self.afm_handover_recovery_path_lateral_deadband = 0.0
        self.afm_handover_recovery_path_heading_deadband = np.deg2rad(0.02)
        self.afm_handover_recovery_reference_speed_cap = 0.18
        self.afm_handover_recovery_reference_speed_floor = 0.08
        self.afm_handover_recovery_terminal_slowdown_distance = 0.22

        self.x_range = (0.0, 5.8)
        self.y_range = (-1.2, 2.5)

        lower_half = self.lower_corridor_width / 2.0
        upper_half = self.upper_corridor_width / 2.0
        pocket_half = 0.82

        self.drivable_areas = {
            'lower_corridor': {
                'type': 'rectangle',
                'x_min': 0.5,
                'y_min': self.lower_center_y - lower_half,
                'x_max': self.transfer_x + 0.10,
                'y_max': self.lower_center_y + lower_half,
            },
            'transfer_column': {
                'type': 'rectangle',
                'x_min': self.transfer_x - 0.24,
                'y_min': self.lower_center_y - lower_half,
                'x_max': self.transfer_x + 0.34,
                'y_max': self.upper_center_y + upper_half,
            },
            'upper_corridor': {
                'type': 'rectangle',
                'x_min': 0.5,
                'y_min': self.upper_center_y - upper_half,
                'x_max': self.rotation_x + 0.05,
                'y_max': self.upper_center_y + upper_half,
            },
            'rotation_pocket': {
                'type': 'rectangle',
                'x_min': self.rotation_x - 0.45,
                'y_min': self.upper_center_y - pocket_half,
                'x_max': self.rotation_x + 0.45,
                'y_max': self.upper_center_y + pocket_half,
            },
        }

        self.initial_state = {
            'x': 1.0,
            'y': self.lower_center_y + float(initial_lateral_offset),
            'psi': float(initial_heading_offset),
        }
        self.end_point = (self.goal_x, self.upper_center_y)
        self.reference_path = self._generate_reference_path()
        self.reset()

    def _generate_reference_path(self) -> np.ndarray:
        points = []

        def append_segment(start, end, heading, interval=0.05):
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            length = float(np.linalg.norm(end - start))
            num = max(2, int(length / interval) + 1)
            for i in range(num):
                t = i / (num - 1)
                xy = start + t * (end - start)
                points.append([float(xy[0]), float(xy[1]), float(heading)])

        append_segment((1.0, self.lower_center_y), (self.transfer_x, self.lower_center_y), 0.0)
        append_segment((self.transfer_x, self.lower_center_y), (self.transfer_x, self.upper_center_y), 0.0)
        append_segment((self.transfer_x, self.upper_center_y), (self.rotation_x, self.upper_center_y), 0.0)

        for heading in np.linspace(0.0, self.end_heading, 21):
            points.append([self.rotation_x, self.upper_center_y, float(heading)])

        append_segment((self.rotation_x, self.upper_center_y), (self.goal_x, self.upper_center_y), np.pi)
        return np.asarray(points, dtype=float)

    def draw_track(self, ax: plt.Axes):
        ax.set_xlim(self.x_range[0] - 0.5, self.x_range[1] + 0.5)
        ax.set_ylim(self.y_range[0] - 0.5, self.y_range[1] + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.fill_between([self.x_range[0], self.x_range[1]], self.y_range[0], self.y_range[1], color='#7f8c8d', alpha=0.96)

        palette = {
            'lower_corridor': '#f8f9fa',
            'transfer_column': '#fdf2d0',
            'upper_corridor': '#f2f7ff',
            'rotation_pocket': '#eef8ef',
        }
        for name, area in self.drivable_areas.items():
            ax.fill_between(
                [area['x_min'], area['x_max']],
                area['y_min'],
                area['y_max'],
                color=palette.get(name, 'white'),
                alpha=1.0,
                label=name.replace('_', ' ').title(),
            )
            ax.plot(
                [area['x_min'], area['x_max'], area['x_max'], area['x_min'], area['x_min']],
                [area['y_min'], area['y_min'], area['y_max'], area['y_max'], area['y_min']],
                color='#c0392b',
                linewidth=1.5,
            )

        ax.plot(self.reference_path[:, 0], self.reference_path[:, 1], 'g--', linewidth=1.8, label='Reference Path')
        ax.plot(self.initial_state['x'], self.initial_state['y'], 'ro', markersize=8, label='_nolegend_')
        ax.plot(self.end_point[0], self.end_point[1], 'gs', markersize=8, label='_nolegend_')

        self._draw_vehicle_at(ax, self.initial_state['x'], self.initial_state['y'], self.initial_state['psi'], color='red', label='_nolegend_')
        self._draw_vehicle_at(ax, self.end_point[0], self.end_point[1], self.end_heading, color='green', label='_nolegend_')

        switch_specs = [
            (self.apt_switch_x, self.lower_center_y, 'AFM → APT', (0.28, -0.72)),
            (self.apt_resume_x, self.upper_center_y, 'APT → AFM', (0.28, 0.60)),
            (self.azr_switch_x, self.upper_center_y, 'AFM → AZR', (0.38, 0.92)),
        ]
        for x, y, label, text_offset in switch_specs:
            ax.scatter([x], [y], s=52, c='#2c3e50', zorder=5)
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(x + text_offset[0], y + text_offset[1]),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.1),
                fontsize=10,
                bbox=dict(boxstyle='round', fc='white', alpha=0.82),
            )


if __name__ == '__main__':
    env = TriModeCompositeEnv()
    fig, ax = plt.subplots(figsize=(11, 5.8))
    env.draw_track(ax)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    fig.tight_layout()
    out = os.path.join(_PROJECT_ROOT, 'outputs', 'tri_mode_composite_map.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(out)
