"""Benchmark the pure AFM/NMPC module on the paper maps."""

import csv
import json
import os
from datetime import datetime

import numpy as np

_CACHE_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.cache')
os.makedirs(os.path.join(_CACHE_ROOT, 'matplotlib'), exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_CACHE_ROOT, 'matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from map_manager import MapManager
from controllers.AFM import AFM
from controllers.casadi_nmpc_robust import CasADiNMPCRobust


plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


def _now_stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _compute_metrics(states, reference_path):
    helper = CasADiNMPCRobust(verbose=False)
    ref = helper.build_reference(reference_path)
    position_errors = []
    lateral_errors = []
    heading_errors = []
    for state in states:
        idx = helper.nearest_index(state, ref)
        ref_state = ref[idx]
        dx = float(state[0] - ref_state[0])
        dy = float(state[1] - ref_state[1])
        path_heading = float(ref_state[2])
        lateral_error = -dx * np.sin(path_heading) + dy * np.cos(path_heading)
        heading_error = helper.wrap_angle(float(state[2] - ref_state[2]))
        position_errors.append(float(np.hypot(dx, dy)))
        lateral_errors.append(float(lateral_error))
        heading_errors.append(float(np.degrees(heading_error)))
    rmse = float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else float('nan')
    heading_mean = float(np.mean(np.abs(heading_errors))) if heading_errors else float('nan')
    final_lateral = float(abs(lateral_errors[-1])) if lateral_errors else float('nan')
    final_heading = float(abs(heading_errors[-1])) if heading_errors else float('nan')
    return {
        'rmse_m': round(rmse, 4),
        'heading_error_mean_deg': round(heading_mean, 4),
        'final_lateral_error_m': round(final_lateral, 4),
        'final_heading_error_deg': round(final_heading, 4),
        'lateral_tracking_rmse_m': round(float(np.sqrt(np.mean(np.square(lateral_errors)))) if lateral_errors else float('nan'), 4),
    }


def _write_trajectory_csv(states, map_type, output_path, dt=0.1):
    rows = []
    for idx, state in enumerate(np.asarray(states, dtype=float)):
        rows.append({
            'map': map_type,
            'method': 'afm_module',
            'step': idx,
            'time_s': round(float(idx * dt), 4),
            'x': round(float(state[0]), 4),
            'y': round(float(state[1]), 4),
            'heading_rad': round(float(state[2]), 4),
            'heading_deg': round(float(np.degrees(state[2])), 2),
        })
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['map', 'method', 'step', 'time_s', 'x', 'y', 'heading_rad', 'heading_deg'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _draw_background(ax, env_map):
    x_min, x_max = env_map.x_range
    y_min, y_max = env_map.y_range
    x_pad = max(0.5, 0.05 * (x_max - x_min))
    y_pad = max(0.5, 0.05 * (y_max - y_min))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.22)
    ax.set_facecolor('#7f8c8d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.fill_between([x_min, x_max], y_min, y_max, color='#7f8c8d', alpha=0.96, zorder=0)

    for area in getattr(env_map, 'drivable_areas', {}).values():
        if area.get('type') != 'rectangle':
            continue
        ax.fill_between([area['x_min'], area['x_max']], area['y_min'], area['y_max'], color='white', alpha=1.0, zorder=1)
        ax.plot(
            [area['x_min'], area['x_max'], area['x_max'], area['x_min'], area['x_min']],
            [area['y_min'], area['y_min'], area['y_max'], area['y_max'], area['y_min']],
            color='#2c3e50',
            linewidth=1.0,
            alpha=0.45,
            zorder=2
        )


def _plot_trajectory(map_type, states, output_path, map_kwargs=None):
    env_map = MapManager().create_map(map_type, **(map_kwargs or {}))
    reference_path = np.asarray(env_map.reference_path, dtype=float)
    actual_x = [float(s[0]) for s in states]
    actual_y = [float(s[1]) for s in states]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    _draw_background(ax, env_map)
    ax.plot(
        reference_path[:, 0],
        reference_path[:, 1],
        color='#e67e22',
        linewidth=3.6,
        linestyle='-',
        solid_capstyle='round',
        zorder=3,
        label='Reference path'
    )
    ax.plot(
        actual_x,
        actual_y,
        color='#1f77b4',
        linewidth=1.35,
        linestyle='-',
        solid_capstyle='round',
        zorder=4,
        label='Actual path'
    )
    ax.set_title(f'{map_type} - AFM Module', fontweight='bold', pad=14)
    ax.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _run_afm_module(map_type, map_kwargs=None, max_time=60.0):
    afm = AFM(map_type=map_type, dt=0.02, horizon=50, map_kwargs=map_kwargs)
    env = afm.env

    initial_state = np.array([
        env.initial_state['x'],
        env.initial_state['y'],
        env.initial_state['psi'],
        0.0,
    ], dtype=float)
    reference_path = np.asarray(env.reference_path, dtype=float)
    states, controls, rmse, heading_error, time_s, success_rate = afm.track_path(
        initial_state,
        reference_path,
        max_time=max_time,
        goal=env.end_point,
        goal_heading=env.end_heading,
    )
    return env, states, controls, rmse, heading_error, time_s, success_rate


def benchmark_afm_maps(map_types=None, output_dir=None):
    map_types = map_types or ['map_a', 'map_b', 'map_c', 'tri_mode_composite']
    run_stamp = _now_stamp()
    output_dir = output_dir or os.path.join('./benchmark_results', f'afm_module_{run_stamp}')
    os.makedirs(output_dir, exist_ok=True)
    trajectory_dir = os.path.join(output_dir, 'trajectory_csvs')
    os.makedirs(trajectory_dir, exist_ok=True)

    rows = []
    for map_type in map_types:
        env, states, controls, rmse, heading_error, time_s, success_rate = _run_afm_module(map_type)
        metrics = _compute_metrics(states, env.reference_path)
        final_distance = float(np.linalg.norm(np.asarray(states[-1][:2], dtype=float) - np.asarray(env.end_point, dtype=float))) if len(states) else float('nan')
        trajectory_csv = os.path.join(trajectory_dir, f'{map_type}_afm.csv')
        _write_trajectory_csv(states, map_type, trajectory_csv, dt=0.02)
        _plot_trajectory(map_type, states, os.path.join(output_dir, f'trajectory_{map_type}.png'))
        rows.append({
            'map': map_type,
            'method': 'afm_module',
            'rmse_m': round(float(rmse), 4),
            'heading_error_deg': round(float(heading_error), 4),
            'success_rate_pct': round(float(success_rate), 1),
            'completion_time_s': round(float(time_s), 2),
            'final_distance_m': round(float(final_distance), 4),
            'failure_reason': 'success' if success_rate > 0 else 'failed',
            'trajectory_csv': trajectory_csv,
            **metrics,
        })

    csv_path = os.path.join(output_dir, 'comparison.csv')
    md_path = os.path.join(output_dir, 'comparison.md')
    json_path = os.path.join(output_dir, 'comparison.json')

    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'map', 'method', 'rmse_m', 'heading_error_deg', 'success_rate_pct',
            'completion_time_s', 'final_distance_m', 'failure_reason', 'trajectory_csv',
            'lateral_tracking_rmse_m', 'heading_error_mean_deg',
            'final_lateral_error_m', 'final_heading_error_deg'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        '# AFM Module Benchmark',
        '',
        '| Map | RMSE (m) | Heading Error (deg) | Success (%) | Time (s) | Final Distance (m) | Lateral RMSE (m) | Final Lateral (m) | Final Heading (deg) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f"| {row['map']} | {row['rmse_m']:.4f} | {row['heading_error_deg']:.4f} | {row['success_rate_pct']:.1f} | "
            f"{row['completion_time_s']:.2f} | {row['final_distance_m']:.4f} | {row['lateral_tracking_rmse_m']:.4f} | "
            f"{row['final_lateral_error_m']:.4f} | {row['final_heading_error_deg']:.4f} |"
        )

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(json_path, 'w') as f:
        json.dump({'run_dir': output_dir, 'rows': rows}, f, indent=2, ensure_ascii=False)

    print(json.dumps({'run_dir': output_dir, 'rows': rows}, indent=2, ensure_ascii=False))


def main():
    benchmark_afm_maps()


if __name__ == '__main__':
    main()
