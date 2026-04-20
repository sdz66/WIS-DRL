"""Benchmark a trained PPO mode-switching policy against pure MPC on the paper maps."""
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

_CACHE_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.cache')
os.makedirs(os.path.join(_CACHE_ROOT, 'matplotlib'), exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_CACHE_ROOT, 'matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stable_baselines3 import PPO

from controllers.AFM import AFM
from controllers.casadi_nmpc_robust import CasADiNMPCRobust
from env.e2e_continuous_env import EndToEndContinuousEnv
from env.mode_env import ModeEnv
from map_manager import MapManager


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


METHOD_DISPLAY_NAMES = {
    'mpc': 'Pure MPC',
    'rule_based': 'Rule-Based Switching',
    'ppo': 'PPO Policy',
    'continuous_rl': 'End-to-End Continuous RL',
}
METHOD_ORDER = ['mpc', 'rule_based', 'ppo', 'continuous_rl']
MAP_TYPES = ['map_a', 'map_b', 'map_c', 'tri_mode_composite']

# Match the discrete PPO / AFM forward-speed envelope during continuous
# evaluation. With dt=0.02 and steps_per_action=5, the discrete acceleration
# ceiling of 1.5 m/s^2 becomes a 0.15 m/s per-action speed delta.
_DISCRETE_EQUIVALENT_CONTINUOUS_LIMITS = {
    'speed_rate_limit': 0.15,
    'speed_rate_limit_up': 0.15,
    'speed_rate_limit_down': 0.15,
    'wheel_speed_min': 0.0,
    'wheel_speed_max': 5.0,
}


def _compute_path_metrics(states, reference_path):
    helper = CasADiNMPCRobust(verbose=False)
    ref = helper.build_reference(reference_path)

    position_errors = []
    heading_errors = []
    for state in states:
        idx = helper.nearest_index(state, ref)
        ref_state = ref[idx]
        position_errors.append(np.linalg.norm(state[:2] - ref_state[:2]))
        heading_errors.append(abs(helper.wrap_angle(state[2] - ref_state[2])))

    rmse = float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else float('nan')
    heading_error_deg = float(np.degrees(np.mean(heading_errors))) if heading_errors else float('nan')
    return rmse, heading_error_deg


class RuleBasedSwitchingPolicy:
    """Threshold-based AFM/APT/AZR selector used as a classical baseline."""

    def __init__(
        self,
        apt_lateral_threshold=0.14,
        apt_lateral_exit=0.05,
        apt_heading_cap_deg=36.0,
        azr_heading_threshold_deg=30.0,
        azr_heading_exit_deg=10.0,
        near_goal_distance=1.30,
        near_goal_heading_deg=12.0,
        forward_blocked_clearance=0.70,
    ):
        self.apt_lateral_threshold = float(apt_lateral_threshold)
        self.apt_lateral_exit = float(apt_lateral_exit)
        self.apt_heading_cap = float(np.deg2rad(apt_heading_cap_deg))
        self.azr_heading_threshold = float(np.deg2rad(azr_heading_threshold_deg))
        self.azr_heading_exit = float(np.deg2rad(azr_heading_exit_deg))
        self.near_goal_distance = float(near_goal_distance)
        self.near_goal_heading = float(np.deg2rad(near_goal_heading_deg))
        self.forward_blocked_clearance = float(forward_blocked_clearance)
        self.last_action = 0

    def reset(self):
        self.last_action = 0

    def decide(self, env):
        """Select 0=AFM, 1=APT or 2=AZR from the current geometry."""
        ctx = env._get_mode_context()
        heading_error = abs(float(ctx['current_heading_error']))
        lateral_error = abs(float(ctx['current_lateral_error']))
        forward_clearance = float(ctx['forward_clearance'])
        distance_to_goal = float(ctx['distance_to_goal'])

        # Simple hysteresis to avoid rapid AFM/APT/AZR flapping, while still
        # keeping the special modes intentionally sticky.
        if self.last_action == 2 and heading_error > self.azr_heading_exit:
            action = 2
        elif self.last_action == 1 and lateral_error > self.apt_lateral_exit and heading_error <= self.apt_heading_cap + np.deg2rad(5.0):
            action = 1
        elif distance_to_goal < self.near_goal_distance and heading_error > self.near_goal_heading:
            action = 2
        elif heading_error >= self.azr_heading_threshold:
            action = 2
        elif forward_clearance < self.forward_blocked_clearance and heading_error > np.deg2rad(25.0):
            action = 2
        elif lateral_error >= self.apt_lateral_threshold and heading_error <= self.apt_heading_cap:
            action = 1
        else:
            action = 0

        self.last_action = action
        return action


def _trajectory_rows_from_states(states, map_type, method, dt):
    """Convert a state rollout into CSV-ready trajectory rows."""
    rows = []
    if states is None:
        return rows

    states = np.asarray(states, dtype=float)
    for idx, state in enumerate(states):
        rows.append({
            'map': map_type,
            'method': method,
            'step': idx,
            'time_s': round(float(idx * dt), 4),
            'x': round(float(state[0]), 4),
            'y': round(float(state[1]), 4),
            'heading_rad': round(float(state[2]), 4),
            'heading_deg': round(float(np.degrees(state[2])), 2),
        })
    return rows


def _method_label(method):
    return METHOD_DISPLAY_NAMES.get(method, method.replace('_', ' ').title())


def _fmt_optional_pct(value):
    if value is None:
        return 'N/A'
    if isinstance(value, (float, int, np.floating, np.integer)) and np.isnan(value):
        return 'N/A'
    return f'{float(value):.1f}'


def _write_trajectory_csv(rows, output_path):
    fieldnames = ['map', 'method', 'step', 'time_s', 'x', 'y', 'heading_rad', 'heading_deg']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_trajectory_csv(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'step': int(row['step']),
                'time_s': float(row['time_s']),
                'x': float(row['x']),
                'y': float(row['y']),
                'heading_rad': float(row['heading_rad']),
            })
    return rows


def _draw_map_background(ax, env_map):
    """Draw the drivable background without the built-in reference path."""
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

    # Non-drivable background
    ax.fill_between([x_min, x_max], y_min, y_max, color='#7f8c8d', alpha=0.96, zorder=0)

    # Drivable rectangles. All of the current paper maps are piecewise-rectangular,
    # so we can reconstruct the corridor layout directly from the map definition.
    if hasattr(env_map, 'drivable_areas'):
        for area in env_map.drivable_areas.values():
            if area.get('type') != 'rectangle':
                continue
            ax.fill_between(
                [area['x_min'], area['x_max']],
                area['y_min'],
                area['y_max'],
                color='white',
                alpha=1.0,
                zorder=1
            )
            ax.plot(
                [area['x_min'], area['x_max'], area['x_max'], area['x_min'], area['x_min']],
                [area['y_min'], area['y_min'], area['y_max'], area['y_max'], area['y_min']],
                color='#2c3e50',
                linewidth=1.2,
                alpha=0.45,
                zorder=2
            )


def _plot_trajectory_from_csv(map_type, csv_path, output_path, method_label):
    """Render one trajectory plot from a saved CSV file."""
    trajectory = _load_trajectory_csv(csv_path)
    if not trajectory:
        return

    env_map = MapManager().create_map(map_type)
    reference_path = np.asarray(env_map.reference_path, dtype=float)
    actual_x = [row['x'] for row in trajectory]
    actual_y = [row['y'] for row in trajectory]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    _draw_map_background(ax, env_map)

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

    ax.set_title(f'{map_type} - {method_label}', fontweight='bold', pad=14)
    ax.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_policy_on_map(model, map_type, trajectory_dir, max_steps=600):
    env = ModeEnv(
        map_type=map_type,
        max_time=50.0,
        dt=0.02,
        steps_per_action=5,
        randomize=False,
        log_dir='./logs/benchmark'
    )
    obs, info = env.reset()

    done = False
    step_count = 0
    final_info = {}
    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, final_info = env.step(int(action))
        done = terminated or truncated
        step_count += 1

    states = np.array(env.state_history)
    rmse, heading_error_deg = _compute_path_metrics(states, env.reference_path)
    trajectory_rows = _trajectory_rows_from_states(states, map_type, 'ppo', env.dt)
    trajectory_csv = os.path.join(trajectory_dir, f'{map_type}_ppo.csv')
    _write_trajectory_csv(trajectory_rows, trajectory_csv)

    total_actions = max(1, sum(env.mode_counts.values()))
    result = {
        'map': map_type,
        'method': 'ppo',
        'rmse_m': round(rmse, 4),
        'heading_error_deg': round(heading_error_deg, 4),
        'success_rate_pct': 100.0 if final_info.get('is_success', False) else 0.0,
        'completion_time_s': round(float(final_info.get('current_time', env.current_time)), 2),
        'final_distance_m': round(float(final_info.get('distance_to_goal', np.linalg.norm(env.state[:2] - env.goal_position))), 4),
        'mode_switches': int(final_info.get('mode_switch_count', env.mode_switch_count)),
        'afm_pct': round(100 * env.mode_counts[0] / total_actions, 1),
        'apt_pct': round(100 * env.mode_counts[1] / total_actions, 1),
        'azr_pct': round(100 * env.mode_counts[2] / total_actions, 1),
        'failure_reason': final_info.get('failure_reason', 'unknown'),
        'trajectory_csv': trajectory_csv
    }
    return result


def evaluate_continuous_policy_on_map(model, map_type, trajectory_dir, max_steps=600):
    env = EndToEndContinuousEnv(
        map_type=map_type,
        max_time=50.0,
        dt=0.02,
        steps_per_action=5,
        randomize=False,
        log_dir='./logs/benchmark',
        **_DISCRETE_EQUIVALENT_CONTINUOUS_LIMITS,
    )
    obs, info = env.reset()

    def _trajectory_state_from_info(info, fallback_state):
        position = info.get('position') if isinstance(info, dict) else None
        heading = info.get('heading_rad') if isinstance(info, dict) else None
        if position is not None and heading is not None:
            return np.array([float(position[0]), float(position[1]), float(heading), 0.0], dtype=float)
        return np.array(fallback_state, dtype=float)

    done = False
    step_count = 0
    final_info = {}
    trajectory_states = [np.array(env.state, dtype=float)]
    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, final_info = env.step(action)
        done = terminated or truncated
        step_count += 1
        fallback_state = env.state if hasattr(env, 'state') else trajectory_states[-1]
        trajectory_states.append(_trajectory_state_from_info(final_info, fallback_state))

    states = np.array(trajectory_states, dtype=float)
    rmse, heading_error_deg = _compute_path_metrics(states, env.reference_path)
    trajectory_rows = _trajectory_rows_from_states(states, map_type, 'continuous_rl', env.dt)
    trajectory_csv = os.path.join(trajectory_dir, f'{map_type}_continuous_rl.csv')
    _write_trajectory_csv(trajectory_rows, trajectory_csv)

    result = {
        'map': map_type,
        'method': 'continuous_rl',
        'rmse_m': round(rmse, 4),
        'heading_error_deg': round(heading_error_deg, 4),
        'success_rate_pct': 100.0 if final_info.get('is_success', False) else 0.0,
        'completion_time_s': round(float(final_info.get('current_time', env.current_time)), 2),
        'final_distance_m': round(float(final_info.get('distance_to_goal', np.linalg.norm(env.state[:2] - env.goal_position))), 4),
        'mode_switches': 0,
        'afm_pct': None,
        'apt_pct': None,
        'azr_pct': None,
        'failure_reason': final_info.get('failure_reason', 'unknown'),
        'trajectory_csv': trajectory_csv,
        'wheel_slip_residual_rms': round(float(final_info.get('wheel_slip_residual_rms', getattr(env, 'last_slip_residual', 0.0))), 4),
        'control_smoothness': round(float(final_info.get('control_smoothness', 0.0)), 4),
    }
    return result


def evaluate_rule_based_on_map(policy, map_type, trajectory_dir, max_steps=600):
    env = ModeEnv(
        map_type=map_type,
        max_time=50.0,
        dt=0.02,
        steps_per_action=5,
        randomize=False,
        log_dir='./logs/benchmark',
        enable_internal_mode_override=False
    )
    obs, info = env.reset()
    policy.reset()

    done = False
    step_count = 0
    final_info = {}
    while not done and step_count < max_steps:
        action = int(policy.decide(env))
        obs, reward, terminated, truncated, final_info = env.step(action)
        done = terminated or truncated
        step_count += 1

    states = np.array(env.state_history)
    rmse, heading_error_deg = _compute_path_metrics(states, env.reference_path)
    trajectory_rows = _trajectory_rows_from_states(states, map_type, 'rule_based', env.dt)
    trajectory_csv = os.path.join(trajectory_dir, f'{map_type}_rule_based.csv')
    _write_trajectory_csv(trajectory_rows, trajectory_csv)

    total_actions = max(1, sum(env.mode_counts.values()))
    result = {
        'map': map_type,
        'method': 'rule_based',
        'rmse_m': round(rmse, 4),
        'heading_error_deg': round(heading_error_deg, 4),
        'success_rate_pct': 100.0 if final_info.get('is_success', False) else 0.0,
        'completion_time_s': round(float(final_info.get('current_time', env.current_time)), 2),
        'final_distance_m': round(float(final_info.get('distance_to_goal', np.linalg.norm(env.state[:2] - env.goal_position))), 4),
        'mode_switches': int(final_info.get('mode_switch_count', env.mode_switch_count)),
        'afm_pct': round(100 * env.mode_counts[0] / total_actions, 1),
        'apt_pct': round(100 * env.mode_counts[1] / total_actions, 1),
        'azr_pct': round(100 * env.mode_counts[2] / total_actions, 1),
        'failure_reason': final_info.get('failure_reason', 'unknown'),
        'trajectory_csv': trajectory_csv
    }
    return result


def evaluate_mpc_on_map(map_type, trajectory_dir, map_kwargs=None):
    afm = AFM(map_type=map_type, dt=0.02, horizon=20, map_kwargs=map_kwargs)
    env = afm.env
    initial_state = np.array([
        env.initial_state['x'],
        env.initial_state['y'],
        env.initial_state['psi'],
        0.0
    ], dtype=float)
    reference_path = np.asarray(env.reference_path, dtype=float)
    states, controls, rmse, heading_error, time_s, success_rate = afm.track_path(
        initial_state,
        reference_path,
        max_time=60.0,
        goal=env.end_point,
        goal_heading=env.end_heading,
    )

    final_distance = np.linalg.norm(states[-1][:2] - np.array(env.end_point)) if len(states) > 0 else np.nan
    trajectory_rows = _trajectory_rows_from_states(states, map_type, 'mpc', afm.dt)
    trajectory_csv = os.path.join(trajectory_dir, f'{map_type}_mpc.csv')
    _write_trajectory_csv(trajectory_rows, trajectory_csv)

    return {
        'map': map_type,
        'method': 'mpc',
        'rmse_m': round(float(rmse), 4),
        'heading_error_deg': round(float(heading_error), 4),
        'success_rate_pct': round(float(success_rate), 1),
        'completion_time_s': round(float(time_s), 2),
        'final_distance_m': round(float(final_distance), 4),
        'mode_switches': 0,
        'afm_pct': 100.0,
        'apt_pct': 0.0,
        'azr_pct': 0.0,
        'failure_reason': 'success' if success_rate > 0 else 'failed',
        'trajectory_csv': trajectory_csv
    }


def _write_csv(rows, output_path):
    fieldnames = [
        'map', 'method', 'rmse_m', 'heading_error_deg', 'success_rate_pct',
        'completion_time_s', 'final_distance_m', 'mode_switches',
        'afm_pct', 'apt_pct', 'azr_pct', 'failure_reason', 'trajectory_csv',
        'wheel_slip_residual_rms', 'control_smoothness'
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _generate_trajectory_plots(rows, output_dir):
    """Generate one trajectory image per map/method from saved CSV files."""
    for row in rows:
        csv_path = row.get('trajectory_csv')
        if not csv_path or not os.path.exists(csv_path):
            continue

        method_label = _method_label(row['method'])
        figure_name = f"trajectory_{row['map']}_{row['method']}.png"
        _plot_trajectory_from_csv(
            row['map'],
            csv_path,
            os.path.join(output_dir, figure_name),
            method_label
        )


def _write_markdown(rows, output_path):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['map'], {})[row['method']] = row

    lines = [
        '# PPO vs Pure MPC vs Rule-Based Switching vs End-to-End Continuous RL Benchmark',
        '',
        '| Map | Method | RMSE (m) | Heading Error (deg) | Success (%) | Time (s) | Final Distance (m) | AFM % | APT % | AZR % | Failure |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for map_name in MAP_TYPES:
        for method in METHOD_ORDER:
            row = grouped.get(map_name, {}).get(method)
            if row is None:
                continue
            lines.append(
                f"| {map_name} | {_method_label(method)} | {row['rmse_m']:.4f} | "
                f"{row['heading_error_deg']:.4f} | {row['success_rate_pct']:.1f} | "
                f"{row['completion_time_s']:.2f} | {row['final_distance_m']:.4f} | "
                f"{_fmt_optional_pct(row['afm_pct'])} | {_fmt_optional_pct(row['apt_pct'])} | {_fmt_optional_pct(row['azr_pct'])} | "
                f"{row['failure_reason']} |"
            )

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _plot_metric(rows, output_dir, metric, ylabel, filename, lower_is_better=True):
    maps = MAP_TYPES
    color_map = {
        'mpc': '#95a5a6',
        'rule_based': '#e67e22',
        'ppo': '#2980b9',
        'continuous_rl': '#8e44ad',
    }
    present_methods = [method for method in METHOD_ORDER if any(row['method'] == method for row in rows)]
    method_specs = [(method, _method_label(method), color_map.get(method, '#34495e')) for method in present_methods]
    method_values = {method: [] for method in present_methods}
    for map_name in maps:
        grouped = [row for row in rows if row['map'] == map_name]
        for method, _, _ in method_specs:
            matching = next((row for row in grouped if row['method'] == method), None)
            if matching is None:
                continue
            method_values[method].append(matching[metric])

    x = np.arange(len(maps))
    width = 0.8 / max(1, len(method_specs))
    center_offset = (len(method_specs) - 1) / 2.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, (method, label, color) in enumerate(method_specs):
        ax.bar(x + (idx - center_offset) * width, method_values[method], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(maps)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} Comparison')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.25)
    if metric == 'success_rate_pct':
        ax.set_ylim(0, 110)

    note = 'Lower is better' if lower_is_better else 'Higher is better'
    ax.text(
        0.98,
        0.98,
        note,
        transform=ax.transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def run_benchmark(model_path, output_dir='./benchmark_results/', rule_policy_kwargs=None, continuous_model_path=None):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f'ppo_vs_mpc_multi_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    trajectory_dir = os.path.join(run_dir, 'trajectory_csvs')
    os.makedirs(trajectory_dir, exist_ok=True)

    model = PPO.load(model_path)
    rule_policy = RuleBasedSwitchingPolicy(**(rule_policy_kwargs or {}))
    maps = MAP_TYPES
    rows = []
    for map_name in maps:
        rows.append(evaluate_mpc_on_map(map_name, trajectory_dir))
        rows.append(evaluate_rule_based_on_map(rule_policy, map_name, trajectory_dir))
        rows.append(evaluate_policy_on_map(model, map_name, trajectory_dir))

    if continuous_model_path:
        continuous_model = PPO.load(continuous_model_path)
        for map_name in maps:
            rows.append(evaluate_continuous_policy_on_map(continuous_model, map_name, trajectory_dir))

    csv_path = os.path.join(run_dir, 'comparison.csv')
    md_path = os.path.join(run_dir, 'comparison.md')
    json_path = os.path.join(run_dir, 'comparison.json')

    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    _generate_trajectory_plots(rows, run_dir)

    _plot_metric(rows, run_dir, 'rmse_m', 'RMSE (m)', 'rmse_comparison.png', lower_is_better=True)
    _plot_metric(rows, run_dir, 'heading_error_deg', 'Heading Error (deg)', 'heading_error_comparison.png', lower_is_better=True)
    _plot_metric(rows, run_dir, 'success_rate_pct', 'Success Rate (%)', 'success_rate_comparison.png', lower_is_better=False)
    _plot_metric(rows, run_dir, 'completion_time_s', 'Completion Time (s)', 'time_comparison.png', lower_is_better=True)

    print(json.dumps({'run_dir': run_dir, 'rows': rows}, indent=2))
    return run_dir, rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark PPO policy against pure MPC, rule-based switching, and continuous RL')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained PPO model')
    parser.add_argument('--continuous-model-path', type=str, default=None,
                        help='Path to trained end-to-end continuous PPO model')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results/', help='Directory for outputs')
    parser.add_argument('--rule-apt-lateral-threshold', type=float, default=0.14,
                        help='APT entry lateral-error threshold in meters')
    parser.add_argument('--rule-apt-lateral-exit', type=float, default=0.05,
                        help='APT exit lateral-error threshold in meters')
    parser.add_argument('--rule-apt-heading-cap-deg', type=float, default=36.0,
                        help='Maximum heading error allowed for APT (deg)')
    parser.add_argument('--rule-azr-heading-threshold-deg', type=float, default=30.0,
                        help='AZR entry heading-error threshold in degrees')
    parser.add_argument('--rule-azr-heading-exit-deg', type=float, default=10.0,
                        help='AZR exit heading-error threshold in degrees')
    parser.add_argument('--rule-near-goal-distance', type=float, default=1.30,
                        help='Distance-to-goal threshold for near-goal AZR')
    parser.add_argument('--rule-near-goal-heading-deg', type=float, default=12.0,
                        help='Near-goal heading-error threshold in degrees')
    parser.add_argument('--rule-forward-blocked-clearance', type=float, default=0.70,
                        help='Forward-clearance threshold that encourages AZR')
    args = parser.parse_args()

    run_benchmark(
        args.model_path,
        args.output_dir,
        continuous_model_path=args.continuous_model_path,
        rule_policy_kwargs=dict(
            apt_lateral_threshold=args.rule_apt_lateral_threshold,
            apt_lateral_exit=args.rule_apt_lateral_exit,
            apt_heading_cap_deg=args.rule_apt_heading_cap_deg,
            azr_heading_threshold_deg=args.rule_azr_heading_threshold_deg,
            azr_heading_exit_deg=args.rule_azr_heading_exit_deg,
            near_goal_distance=args.rule_near_goal_distance,
            near_goal_heading_deg=args.rule_near_goal_heading_deg,
            forward_blocked_clearance=args.rule_forward_blocked_clearance,
        )
    )


if __name__ == '__main__':
    main()
