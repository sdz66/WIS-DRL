"""
Test Script for Trained PPO Mode Selection Model (Task 6)
Comprehensive evaluation with detailed metrics and statistics

Usage:
    python test.py --model-path models/xxx/best_model.zip
    python test.py --model-path models/xxx/mode_selector_final.zip --episodes 50
"""

import numpy as np
import os
import sys
import json
import csv
from datetime import datetime

# Keep matplotlib/fontconfig caches inside the writable workspace to avoid
# startup warnings and repeated cache rebuilds during evaluation runs.
_CACHE_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.cache')
os.makedirs(os.path.join(_CACHE_ROOT, 'matplotlib'), exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_CACHE_ROOT, 'matplotlib'))

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Error: stable_baselines3 not installed!")
    sys.exit(1)

from env.mode_env import ModeEnv
from map_manager import MapManager


def _normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _reference_heading(reference_path, index):
    if reference_path.shape[1] > 2:
        return float(reference_path[index, 2])

    if index < len(reference_path) - 1:
        delta = reference_path[index + 1, :2] - reference_path[index, :2]
    else:
        delta = reference_path[index, :2] - reference_path[index - 1, :2]
    return float(np.arctan2(delta[1], delta[0]))


def _step_path_metrics(state, reference_path):
    state = np.asarray(state, dtype=float)
    distances = np.linalg.norm(reference_path[:, :2] - state[:2], axis=1)
    reference_index = int(np.argmin(distances))
    ref_point = reference_path[reference_index]
    ref_heading = _reference_heading(reference_path, reference_index)

    error_vec = ref_point[:2] - state[:2]
    lateral_error = float(
        error_vec[0] * (-np.sin(ref_heading)) +
        error_vec[1] * np.cos(ref_heading)
    )
    heading_error = float(_normalize_angle(ref_heading - state[2]))
    position_error = float(np.linalg.norm(state[:2] - ref_point[:2]))

    return {
        'reference_index': reference_index,
        'reference_x': float(ref_point[0]),
        'reference_y': float(ref_point[1]),
        'reference_heading_rad': ref_heading,
        'position_error_m': position_error,
        'lateral_error_m': lateral_error,
        'heading_error_rad': heading_error
    }


def _rmse(values):
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))


def test_trained_model(model_path, map_type='tri_mode_composite', num_episodes=20,
                       output_dir='./test_results/', randomize=True,
                       max_time=50.0):
    """
    Test trained model with comprehensive evaluation (Task 6)
    
    Runs multiple episodes and collects:
      - Success rate
      - Average completion time
      - Average reward
      - Mode usage distribution (AFM/APT/AZR percentages)
      - Mode switch frequency
      - Per-episode detailed logs
    
    Args:
        model_path: Path to trained model (.zip file)
        map_type: Map type for testing
        num_episodes: Number of test episodes (default: 20)
        output_dir: Directory for saving results
        randomize: Whether to use randomized environments
    
    Returns:
        dict: Comprehensive test results
    """
    print("\n" + "="*80)
    print(" MODEL EVALUATION - PPO MODE SELECTOR")
    print("="*80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'test_{map_type}_{timestamp}')
    os.makedirs(output_path, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    
    model = PPO.load(model_path)
    print(f"Model loaded successfully!")
    
    # Create test environment
    print(f"Creating test environment (map={map_type}, episodes={num_episodes})")
    env = ModeEnv(
        map_type=map_type,
        max_time=max_time,
        dt=0.02,
        steps_per_action=5,
        randomize=randomize,
        log_dir=output_path
    )
    
    # Storage for results
    all_results = []
    step_trace = []
    episode_rewards = []
    episode_times = []
    episode_steps = []
    success_count = 0
    failure_reason_counts = {}
    
    # Aggregated mode usage stats
    total_afm = 0
    total_apt = 0
    total_azr = 0
    total_switches = 0
    
    # Run test episodes
    print(f"\n{'─'*80}")
    print(f" RUNNING {num_episodes} TEST EPISODES")
    print(f"{'─'*80}\n")
    
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        
        done = False
        episode_reward = 0.0
        step_count = 0
        episode_position_errors = []
        episode_lateral_errors = []
        episode_heading_errors_deg = []
        last_path_metrics = None
        
        # Episode-specific mode tracking
        ep_requested_mode_counts = {0: 0, 1: 0, 2: 0}
        ep_executed_mode_counts = {0: 0, 1: 0, 2: 0}
        ep_switches = 0
        prev_mode = 0
        
        max_steps = int(np.ceil(max_time / env.decision_dt))
        while not done and step_count < max_steps:  # Max steps protection based on time window
            # Use trained policy (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            
            # Convert action to Python integer (handles numpy array case from SB3)
            action = int(action)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track requested and executed mode usage
            ep_requested_mode_counts[action] += 1
            executed_mode = int(info.get('executed_mode', action))
            ep_executed_mode_counts[executed_mode] += 1
            if action != prev_mode and step_count > 0:
                ep_switches += 1
            prev_mode = action
            
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            current_state = env.state.copy()
            last_path_metrics = _step_path_metrics(current_state, env.reference_path)
            episode_position_errors.append(last_path_metrics['position_error_m'])
            episode_lateral_errors.append(abs(last_path_metrics['lateral_error_m']))
            episode_heading_errors_deg.append(abs(np.degrees(last_path_metrics['heading_error_rad'])))
            current_path_rmse = _rmse(episode_position_errors)

            requested_mode = int(info.get('requested_mode', info.get('current_mode', action)))
            executed_mode = int(info.get('executed_mode', requested_mode))
            step_trace.append({
                'map_type': info.get('map_type', map_type),
                'episode': ep,
                'step': step_count,
                'time_s': round(float(info.get('current_time', env.current_time)), 2),
                'x': round(float(info.get('position', [0.0, 0.0])[0]), 4),
                'y': round(float(info.get('position', [0.0, 0.0])[1]), 4),
                'heading_rad': round(float(info.get('heading_rad', 0.0)), 4),
                'heading_deg': round(float(np.degrees(info.get('heading_rad', 0.0))), 2),
                'mode': requested_mode,
                'executed_mode': executed_mode,
                'active_primitive_mode': int(info.get('active_primitive_mode', 0)),
                'suggested_mode': int(info.get('suggested_mode', 0)),
                'is_apt_candidate': bool(info.get('is_apt_candidate', False)),
                'is_azr_candidate': bool(info.get('is_azr_candidate', False)),
                'reward': round(float(reward), 4),
                'total_reward': round(float(info.get('total_reward', episode_reward)), 4),
                'distance_to_goal': round(float(info.get('distance_to_goal', 0.0)), 4),
                'reference_index': last_path_metrics['reference_index'],
                'reference_x': round(float(last_path_metrics['reference_x']), 4),
                'reference_y': round(float(last_path_metrics['reference_y']), 4),
                'reference_heading_rad': round(float(last_path_metrics['reference_heading_rad']), 4),
                'reference_heading_deg': round(float(np.degrees(last_path_metrics['reference_heading_rad'])), 2),
                'position_error_m': round(float(last_path_metrics['position_error_m']), 4),
                'lateral_error_m': round(float(last_path_metrics['lateral_error_m']), 4),
                'heading_error_rad': round(float(last_path_metrics['heading_error_rad']), 4),
                'heading_error_deg': round(float(np.degrees(last_path_metrics['heading_error_rad'])), 2),
                'path_rmse_m': round(float(current_path_rmse), 4),
                'forward_clearance': round(float(info.get('forward_clearance', 0.0)), 4),
                'left_clearance': round(float(info.get('left_clearance', 0.0)), 4),
                'right_clearance': round(float(info.get('right_clearance', 0.0)), 4),
                'path_progress': round(float(info.get('path_progress', 0.0)), 4),
                'path_heading_change_rad': round(float(info.get('path_heading_change', 0.0)), 4),
                'path_heading_change_deg': round(float(np.degrees(info.get('path_heading_change', 0.0))), 2),
                'current_speed_mps': round(float(env.state[3]), 4),
                'step_displacement': round(float(info.get('step_displacement', 0.0)), 4),
                'step_longitudinal': round(float(info.get('step_longitudinal', 0.0)), 4),
                'step_lateral': round(float(info.get('step_lateral', 0.0)), 4),
                'step_heading_change_rad': round(float(info.get('step_heading_change', 0.0)), 4),
                'step_heading_change_deg': round(float(np.degrees(info.get('step_heading_change', 0.0))), 2),
                'apt_direction_hint': info.get('apt_direction_hint', 'unknown'),
                'apt_distance_hint': round(float(info.get('apt_distance_hint', 0.0)), 4),
                'failure_reason': info.get('failure_reason', 'unknown'),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'is_success': bool(info.get('is_success', False)),
            })
        
        # Record episode results
        is_success = info.get('is_success', False)
        if is_success:
            success_count += 1
        final_path_rmse = _rmse(episode_position_errors)
        
        # Store results
        result = {
            'episode': ep,
            'map_type': info.get('map_type', map_type),
            'success': is_success,
            'reward': round(episode_reward, 2),
            'steps': step_count,
            'time_s': round(info.get('current_time', 0), 2),
            'final_distance_m': round(info.get('distance_to_goal', 0), 3),
            'final_path_rmse_m': round(final_path_rmse, 4),
            'mean_abs_position_error_m': round(float(np.mean(episode_position_errors)) if episode_position_errors else 0.0, 4),
            'mean_abs_lateral_error_m': round(float(np.mean(episode_lateral_errors)) if episode_lateral_errors else 0.0, 4),
            'mean_abs_heading_error_deg': round(float(np.mean(episode_heading_errors_deg)) if episode_heading_errors_deg else 0.0, 4),
            'max_abs_lateral_error_m': round(float(np.max(episode_lateral_errors)) if episode_lateral_errors else 0.0, 4),
            'max_position_error_m': round(float(np.max(episode_position_errors)) if episode_position_errors else 0.0, 4),
            'final_heading_error_deg': round(float(abs(np.degrees(last_path_metrics['heading_error_rad']))) if last_path_metrics else 0.0, 4),
            'afm_count': ep_executed_mode_counts[0],
            'apt_count': ep_executed_mode_counts[1],
            'azr_count': ep_executed_mode_counts[2],
            'requested_afm_count': ep_requested_mode_counts[0],
            'requested_apt_count': ep_requested_mode_counts[1],
            'requested_azr_count': ep_requested_mode_counts[2],
            'mode_switches': ep_switches,
            'total_actions': sum(ep_executed_mode_counts.values()),
            'failure_reason': info.get('failure_reason', 'unknown'),
            'apt_candidate_steps': info.get('apt_candidate_steps', 0),
            'azr_candidate_steps': info.get('azr_candidate_steps', 0),
            'blocked_steps': info.get('blocked_steps', 0)
        }
        
        all_results.append(result)
        episode_rewards.append(episode_reward)
        episode_times.append(info.get('current_time', 0))
        episode_steps.append(step_count)
        
        # Aggregate mode stats
        total_afm += ep_executed_mode_counts[0]
        total_apt += ep_executed_mode_counts[1]
        total_azr += ep_executed_mode_counts[2]
        total_switches += ep_switches
        failure_reason = result['failure_reason']
        failure_reason_counts[failure_reason] = failure_reason_counts.get(failure_reason, 0) + 1
        
        # Save per-episode stats to environment CSV
        env.save_episode_stats(ep, is_success)
        
        # Print progress
        status = "✓ SUCCESS" if is_success else "✗ FAILED"
        print(f"Episode {ep:2d}/{num_episodes}: {status} | "
              f"reward={episode_reward:+8.2f} | "
              f"time={info.get('current_time', 0):5.1f}s | "
              f"steps={step_count:4d} | "
              f"modes=[AFM:{ep_executed_mode_counts[0]:3d}, APT:{ep_executed_mode_counts[1]:3d}, AZR:{ep_executed_mode_counts[2]:3d}] | "
              f"switches={ep_switches:2d}")
    
    env.close()
    
    # Calculate aggregate statistics
    total_actions = total_afm + total_apt + total_azr or 1
    
    summary = {
        'test_configuration': {
            'model_path': model_path,
            'map_type': map_type,
            'num_episodes': num_episodes,
            'randomize': randomize,
            'timestamp': timestamp
        },
        'performance_metrics': {
            'success_rate_pct': round(100 * success_count / num_episodes, 1),
            'avg_reward': round(np.mean(episode_rewards), 2),
            'std_reward': round(np.std(episode_rewards), 2),
            'min_reward': round(min(episode_rewards), 2),
            'max_reward': round(max(episode_rewards), 2),
            'median_reward': round(np.median(episode_rewards), 2),
            'avg_completion_time_s': round(np.mean(episode_times), 2),
            'std_time_s': round(np.std(episode_times), 2),
            'avg_steps': round(np.mean(episode_steps), 1),
            'avg_final_path_rmse_m': round(np.mean([r['final_path_rmse_m'] for r in all_results]), 4),
            'std_final_path_rmse_m': round(np.std([r['final_path_rmse_m'] for r in all_results]), 4),
            'avg_mean_abs_lateral_error_m': round(np.mean([r['mean_abs_lateral_error_m'] for r in all_results]), 4),
            'avg_mean_abs_heading_error_deg': round(np.mean([r['mean_abs_heading_error_deg'] for r in all_results]), 4),
            'success_count': success_count,
            'failure_count': num_episodes - success_count
        },
        'mode_usage_statistics': {
            'total_afm_actions': total_afm,
            'total_apt_actions': total_apt,
            'total_azr_actions': total_azr,
            'afm_usage_pct': round(100 * total_afm / total_actions, 1),
            'apt_usage_pct': round(100 * total_apt / total_actions, 1),
            'azr_usage_pct': round(100 * total_azr / total_actions, 1),
            'total_switches': total_switches,
            'avg_switches_per_episode': round(total_switches / num_episodes, 1)
        },
        'failure_reason_statistics': failure_reason_counts
    }

    apt_candidate_total = sum(1 for row in step_trace if row['is_apt_candidate'])
    azr_candidate_total = sum(1 for row in step_trace if row['is_azr_candidate'])
    apt_selected_on_candidate = sum(
        1 for row in step_trace
        if row['is_apt_candidate'] and row['executed_mode'] == 1
    )
    azr_selected_on_candidate = sum(
        1 for row in step_trace
        if row['is_azr_candidate'] and row['executed_mode'] == 2
    )
    summary['mode_diagnostics'] = {
        'apt_candidate_steps': apt_candidate_total,
        'azr_candidate_steps': azr_candidate_total,
        'apt_selected_on_candidate_steps': apt_selected_on_candidate,
        'azr_selected_on_candidate_steps': azr_selected_on_candidate,
        'apt_selection_rate_on_candidate_pct': round(
            100 * apt_selected_on_candidate / max(1, apt_candidate_total), 1
        ),
        'azr_selection_rate_on_candidate_pct': round(
            100 * azr_selected_on_candidate / max(1, azr_candidate_total), 1
        )
    }
    
    # Save comprehensive results
    results_file = os.path.join(output_path, 'test_summary.json')
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed per-episode CSV
    csv_file = os.path.join(output_path, 'episode_details.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    trajectory_trace_file = os.path.join(output_path, 'trajectory_trace.csv')
    step_trace_file = os.path.join(output_path, 'step_trace.csv')
    for trace_file in (trajectory_trace_file, step_trace_file):
        with open(trace_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=step_trace[0].keys())
            writer.writeheader()
            writer.writerows(step_trace)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f" TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Success Rate:     {summary['performance_metrics']['success_rate_pct']:.1f}% "
          f"({summary['performance_metrics']['success_count']}/{num_episodes})")
    print(f"   Average Reward:   {summary['performance_metrics']['avg_reward']:.2f} "
          f"(±{summary['performance_metrics']['std_reward']:.2f})")
    print(f"   Median Reward:    {summary['performance_metrics']['median_reward']:.2f}")
    print(f"   Range:            [{summary['performance_metrics']['min_reward']:.2f}, "
          f"{summary['performance_metrics']['max_reward']:.2f}]")
    print(f"   Avg Time:         {summary['performance_metrics']['avg_completion_time_s']:.2f}s "
          f"(±{summary['performance_metrics']['std_time_s']:.2f}s)")
    print(f"   Avg Steps:        {summary['performance_metrics']['avg_steps']:.1f}")
    print(f"   Final Path RMSE:  {summary['performance_metrics']['avg_final_path_rmse_m']:.4f}m "
          f"(±{summary['performance_metrics']['std_final_path_rmse_m']:.4f}m)")
    print(f"   Mean Lat Error:   {summary['performance_metrics']['avg_mean_abs_lateral_error_m']:.4f}m")
    print(f"   Mean Head Error:  {summary['performance_metrics']['avg_mean_abs_heading_error_deg']:.4f}°")
    
    print(f"\n🎮 MODE USAGE DISTRIBUTION:")
    print(f"   AFM (Path Track): {summary['mode_usage_statistics']['afm_usage_pct']:5.1f}% "
          f"({summary['mode_usage_statistics']['total_afm_actions']} actions)")
    print(f"   APT (Translate):  {summary['mode_usage_statistics']['apt_usage_pct']:5.1f}% "
          f"({summary['mode_usage_statistics']['total_apt_actions']} actions)")
    print(f"   AZR (Rotate):     {summary['mode_usage_statistics']['azr_usage_pct']:5.1f}% "
          f"({summary['mode_usage_statistics']['total_azr_actions']} actions)")
    print(f"   Avg Switches/Ep:  {summary['mode_usage_statistics']['avg_switches_per_episode']:.1f}")
    print(f"   APT Candidate:    {summary['mode_diagnostics']['apt_candidate_steps']} steps | "
          f"chosen {summary['mode_diagnostics']['apt_selection_rate_on_candidate_pct']:.1f}%")
    print(f"   AZR Candidate:    {summary['mode_diagnostics']['azr_candidate_steps']} steps | "
          f"chosen {summary['mode_diagnostics']['azr_selection_rate_on_candidate_pct']:.1f}%")

    print(f"\n🧭 FAILURE REASONS:")
    for reason, count in sorted(summary['failure_reason_statistics'].items()):
        print(f"   {reason:16s} {count:3d}")
    
    print(f"\n💾 OUTPUT FILES:")
    print(f"   Summary JSON:     {results_file}")
    print(f"   Episode CSV:      {csv_file}")
    print(f"   Trajectory CSV:   {trajectory_trace_file}")
    print(f"   Step Trace CSV:   {step_trace_file}")
    print(f"   Action Stats:     {output_path}/action_stats.csv")
    print(f"   Output Directory: {output_path}/\n")
    
    return summary


def main():
    """Command-line interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test trained PPO mode selector model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test (20 episodes)
  python test.py --model-path models/ppo_obstacle_xxx/best_model.zip

  # Extended test (100 episodes)
  python test.py --model-path models/xxx/mode_selector_final.zip --episodes 100

  # Different map
  python test.py --model-path models/xxx/best_model.zip --map map_a --episodes 50

  # Deterministic testing (no randomization)
  python test.py --model-path models/xxx/best_model.zip --no-randomize
        """
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--map', type=str, default='tri_mode_composite',
                       choices=MapManager.get_available_maps(),
                       help='Map type for testing (default: tri_mode_composite)')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of test episodes (default: 20)')
    parser.add_argument('--output-dir', type=str, default='./test_results/',
                       help='Output directory for results')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Disable randomization for deterministic testing')
    parser.add_argument('--max-time', type=float, default=50.0,
                       help='Maximum episode time in seconds (default: 50.0)')
    
    args = parser.parse_args()
    
    # Run tests
    results = test_trained_model(
        model_path=args.model_path,
        map_type=args.map,
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        randomize=not args.no_randomize,
        max_time=args.max_time
    )
    
    if results:
        print("Testing completed successfully!")
        return 0
    else:
        print("Testing failed!")
        return 1


if __name__ == "__main__":
    exit(main())
