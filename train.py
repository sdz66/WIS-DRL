"""
PPO Training Script for Mode Switching Policy (Enhanced Version)
Trains reinforcement learning agent with:
  - TensorBoard logging (Task 1)
  - EvalCallback for automatic evaluation and best model saving (Task 2)
  - Comprehensive hyperparameter configuration
  - Mode usage statistics tracking
  - Randomized training environment
"""
import numpy as np
import os
import sys
import time
import json
from datetime import datetime

# Keep matplotlib/fontconfig caches inside the writable workspace to avoid
# startup warnings and repeated cache rebuilds during training runs.
_CACHE_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.cache')
os.makedirs(os.path.join(_CACHE_ROOT, 'matplotlib'), exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_CACHE_ROOT, 'matplotlib'))

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback, CallbackList, BaseCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import ConstantSchedule
except ImportError:
    print("Error: stable-baselines3 not installed!")
    print("Please install with: pip install stable-baselines3")
    sys.exit(1)

from env.mode_env import ModeEnv
from map_manager import MapManager


class TensorBoardLoggingCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard (Task 1)
    
    Logs:
      - reward/episode_reward: Episode total reward
      - success_rate: Running success rate
      - episode_length: Episode length in steps
      - mode_usage/afm, apt, azr: Mode usage percentages
      - mode_switches: Number of mode switches per episode
    """
    
    def __init__(self, log_freq=1000, verbose=0):
        super(TensorBoardLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.mode_stats = []  # Store mode usage per episode
        
    def _on_step(self) -> bool:
        # Check if we have episode info
        if len(self.locals.get('infos', [])) > 0:
            last_info = self.locals['infos'][-1]
            
            if 'episode' in last_info:
                ep_info = last_info['episode']
                
                # Record basic metrics
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
                # Try to get success info (from custom env)
                success = last_info.get('is_success', False)
                self.successes.append(success)
                
                # Log to TensorBoard at specified frequency
                if self.num_timesteps % self.log_freq == 0:
                    self._log_to_tensorboard()
        
        return True
    
    def _log_to_tensorboard(self):
        """Log aggregated metrics to TensorBoard"""
        if len(self.episode_rewards) == 0:
            return
            
        # Calculate running averages (last 100 episodes)
        window = min(100, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-window:])
        avg_length = np.mean(self.episode_lengths[-window:])
        success_rate = np.mean(self.successes[-window:]) * 100
        
        # Log scalar metrics
        self.logger.record('reward/episode_reward', avg_reward)
        self.logger.record('success_rate', success_rate)
        self.logger.record('episode_length', avg_length)
        
        # Print progress
        if self.verbose > 0:
            print(f"\n[TensorBoard] Step {self.num_timesteps:,}")
            print(f"  Avg Reward (last {window}): {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Avg Length: {avg_length:.1f}")


def create_env(map_type=None, max_time=50.0, reward_weights=None,
               randomize=True, log_dir='./logs/'):
    """
    Create training environment with all enhancements
    
    Args:
        map_type: Map type for training. Use None to run the built-in
                  curriculum over map_a, map_b, and map_c.
        max_time: Maximum episode time
        reward_weights: Dict of reward weights w1-w4
        randomize: Enable/disable randomization (Task 3)
        log_dir: Directory for statistics logs
    
    Returns:
        Monitor-wrapped environment
    """
    env = ModeEnv(
        map_type=map_type,
        max_time=max_time,
        dt=0.02,              # 50 Hz low-level control
        steps_per_action=5,    # 5 steps * 0.02s = 0.1s (10 Hz high-level)
        reward_weights=reward_weights,
        randomize=randomize,  # Task 3: Enable randomization
        log_dir=log_dir       # Task 4: Save stats here
    )
    
    # Wrap with Monitor for automatic logging
    env = Monitor(env)
    
    return env


def _build_run_label(map_type):
    """Create a filesystem-friendly label for the current training set."""
    if isinstance(map_type, (list, tuple)):
        return 'multi_' + '_'.join(map(str, map_type))
    return str(map_type)


def train_mode_selector(
    total_timesteps=500000,
    map_type=None,
    learning_rate=1e-5,
    n_steps=2048,
    batch_size=128,
    n_epochs=8,
    gamma=0.998,
    gae_lambda=0.96,
    clip_range=0.15,
    ent_coef=0.0005,
    save_freq=50000,
    eval_freq=10000,         # Task 2: Evaluate every 10000 steps
    n_eval_episodes=10,      # Task 2: Run 10 episodes for evaluation
    log_dir='./logs/',
    model_dir='./models/',
    tensorboard_dir='./tb_logs/',  # Task 1: TensorBoard logs directory
    reward_weights=None,
    randomize=True,
    max_time=50.0,
    resume_model_path=None
):
    """
    Train PPO agent with full evaluation pipeline
    
    Features:
      - TensorBoard logging (Task 1)
      - Automatic evaluation with EvalCallback (Task 2)
      - Best model saving based on mean reward
      - Randomized training environments (Task 3)
      - Mode usage statistics tracking (Task 4)
      - Enhanced reward function (Task 5)
    
    Args:
        total_timesteps: Total training timesteps (default: 500,000)
        map_type: Map type for training
        learning_rate: PPO learning rate (default: 1e-5)
        n_steps: Steps per update (default: 2048)
        batch_size: Mini-batch size (default: 128)
        n_epochs: Epochs per update (default: 8)
        save_freq: Model save frequency (timesteps)
        eval_freq: Evaluation frequency (timesteps) - Task 2
        n_eval_episodes: Number of evaluation episodes - Task 2
        log_dir: Directory for CSV logs
        model_dir: Directory for saved models
        tensorboard_dir: Directory for TensorBoard logs - Task 1
        reward_weights: Dict with w1-w4 weights
    """
    # Build the training map set. The default curriculum interleaves the three
    # paper maps so the policy keeps seeing AFM, APT, and AZR-friendly layouts.
    if map_type is None:
        training_map_types = [
            'map_a',
            'map_b',
            'map_c',
            'map_a',
            'map_b',
            'map_c',
            'map_a',
            'map_b',
            'map_c',
            'map_a',
            'map_b',
            'map_c'
        ]
    elif isinstance(map_type, (list, tuple)):
        training_map_types = list(map_type)
    else:
        training_map_types = [map_type]

    if len(training_map_types) > 1:
        eval_map_type = list(dict.fromkeys(training_map_types))
    else:
        eval_map_type = training_map_types[0]
    run_label = _build_run_label(training_map_types)

    # Default reward weights
    if reward_weights is None:
        reward_weights = {
            'w1': 3.0,   # Progress reward weight
            'w2': 1.6,   # Cross-track penalty weight
            'w3': 0.7,   # Heading penalty weight
            'w4': 0.03   # Mode switch penalty weight
        }
    
    # Create timestamped directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_log_dir = os.path.join(log_dir, f'ppo_{run_label}_{timestamp}')
    run_model_dir = os.path.join(model_dir, f'ppo_{run_label}_{timestamp}')
    tb_run_dir = os.path.join(tensorboard_dir, f'ppo_{run_label}_{timestamp}')
    
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(tb_run_dir, exist_ok=True)
    
    # Print configuration header
    print("\n" + "="*80)
    print(" PPO TRAINING SYSTEM - 4WID-4WIS ROBOT MODE SWITCHING")
    print("="*80)
    print(f"\n{'─'*80}")
    print(f" CONFIGURATION")
    print(f"{'─'*80}")
    print(f"  Algorithm:            PPO (Proximal Policy Optimization)")
    print(f"  Total Timesteps:      {total_timesteps:,}")
    print(f"  Map Type:             {map_type}")
    print(f"  Training Maps:        {', '.join(training_map_types)}")
    print(f"{'─'*80}")
    print(f" NETWORK ARCHITECTURE:")
    print(f"  Policy Network:       MLP [256, 256] with ReLU")
    print(f"  Value Network:        MLP [256, 256] with ReLU")
    print(f"{'─'*80}")
    print(f" TRAINING HYPERPARAMETERS:")
    print(f"  Learning Rate:        {learning_rate} (Adam optimizer)")
    print(f"  Discount Factor γ:    {gamma}")
    print(f"  GAE Parameter λ:     {gae_lambda}")
    print(f"  PPO Clipping ε:      {clip_range}")
    print(f"  Entropy Coef:        {ent_coef}")
    print(f"  Batch Size:           {batch_size}")
    print(f"  Steps per Update:     {n_steps:,}")
    print(f"  Epochs per Update:    {n_epochs}")
    print(f"{'─'*80}")
    print(f" TIMING CONFIGURATION:")
    print(f"  High-Level Freq:      10 Hz (decision every 0.1s)")
    print(f"  Low-Level Freq:       50 Hz (control cycle = 0.02s)")
    print(f"{'─'*80}")
    print(f" EVALUATION CONFIGURATION (Task 2):")
    print(f"  Evaluation Frequency: Every {eval_freq:,} steps")
    print(f"  Episodes per Eval:   {n_eval_episodes}")
    print(f"  Best Model Saving:   Enabled (based on mean reward)")
    if isinstance(eval_map_type, (list, tuple)):
        print(f"  Evaluation Maps:     {', '.join(map(str, eval_map_type))}")
    else:
        print(f"  Evaluation Map:      {eval_map_type}")
    print(f"{'─'*80}")
    print(f" REWARD WEIGHTS (w₁-w₄):")
    print(f"  w₁ (progress):        {reward_weights['w1']}")
    print(f"  w₂ (cross_track):     {reward_weights['w2']}")
    print(f"  w₃ (heading):         {reward_weights['w3']}")
    print(f"  w₄ (mode_switch):     {reward_weights['w4']}")
    print(f"{'─'*80}")
    print(f" RANDOMIZATION (Task 3): {'ENABLED' if randomize else 'DISABLED'}")
    if randomize:
        print(f"  Initial Position:     ±0.5m")
        print(f"  Initial Heading:      ±15°")
        print(f"  Goal Position:        ±0.3m")
        print(f"  Goal Heading:         ±10°")
    print(f"{'─'*80}")
    print(f"\n OUTPUT DIRECTORIES:")
    print(f"  Models:       {run_model_dir}")
    print(f"  CSV Logs:     {run_log_dir}")
    print(f"  TensorBoard:  {tb_run_dir}")
    print(f"{'─'*80}\n")
    
    # Save configuration to JSON for reproducibility
    config = {
        'algorithm': 'PPO',
        'total_timesteps': total_timesteps,
        'map_type': map_type,
        'training_map_types': training_map_types,
        'evaluation_map_type': eval_map_type,
        'resume_model_path': resume_model_path,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'net_arch': [256, 256]
        },
        'timing': {
            'high_level_freq_hz': 10,
            'low_level_freq_hz': 50,
            'dt_seconds': 0.02,
            'steps_per_action': 5,
            'max_time_seconds': max_time
        },
        'evaluation': {
            'eval_freq': eval_freq,
            'n_eval_episodes': n_eval_episodes
        },
        'reward_weights': reward_weights,
        'randomization': {
            'enabled': randomize,
            'position_range_m': 0.5,
            'heading_range_deg': 15,
            'goal_position_range_m': 0.3,
            'goal_heading_range_deg': 10
        },
        'timestamp': timestamp
    }
    
    config_path = os.path.join(run_model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Create training environment
    print(f"\nCreating training environment...")
    env = create_env(
        map_type=training_map_types,
        max_time=max_time,
        reward_weights=reward_weights,
        randomize=randomize,
        log_dir=run_log_dir
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    # Create separate evaluation environment
    print(f"Creating evaluation environment...")
    eval_env = create_env(
        map_type=eval_map_type,
        max_time=max_time,
        reward_weights=reward_weights,
        randomize=False,
        log_dir=os.path.join(run_log_dir, 'eval')
    )
    
    # Initialize PPO model with MLP [256, 256]
    print(f"\nInitializing PPO model...")
    import torch.nn as nn
    
    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=nn.ReLU
    )
    
    if resume_model_path:
        print(f"Resuming from checkpoint: {resume_model_path}")
        if not os.path.exists(resume_model_path):
            raise FileNotFoundError(f"Resume model not found: {resume_model_path}")
        model = PPO.load(resume_model_path, env=env, print_system_info=False)
        requested_n_steps = n_steps
        if n_steps != model.n_steps:
            print(
                f"  Requested rollout steps {n_steps} do not match the checkpoint "
                f"rollout steps {model.n_steps}; keeping the checkpoint value to "
                f"avoid rollout buffer mismatch."
            )
            n_steps = model.n_steps
        model.learning_rate = learning_rate
        model.lr_schedule = ConstantSchedule(learning_rate)
        model.batch_size = batch_size
        model.n_epochs = n_epochs
        model.gamma = gamma
        model.gae_lambda = gae_lambda
        model.clip_range = ConstantSchedule(clip_range)
        model.ent_coef = ent_coef
        model.tensorboard_log = tb_run_dir
        if hasattr(model.policy, "optimizer"):
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = learning_rate
        config['hyperparameters']['n_steps'] = model.n_steps
        config['hyperparameters']['batch_size'] = batch_size
        config['hyperparameters']['n_epochs'] = n_epochs
        config['hyperparameters']['gamma'] = gamma
        config['hyperparameters']['gae_lambda'] = gae_lambda
        config['hyperparameters']['clip_range'] = clip_range
        config['hyperparameters']['ent_coef'] = ent_coef
        config['resume_training'] = {
            'requested_n_steps': requested_n_steps,
            'effective_n_steps': model.n_steps,
            'requested_batch_size': batch_size,
            'effective_batch_size': batch_size
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Effective rollout steps: {model.n_steps}")
        print(f"  Effective batch size:    {model.batch_size}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_run_dir  # Task 1: Enable TensorBoard logging
        )
    
    # Print model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # === TASK 2: Setup EvalCallback ===
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=eval_freq,           # Evaluate every 10000 steps
        n_eval_episodes=n_eval_episodes,  # Run 10 episodes per evaluation
        deterministic=True,
        render=False
    )
    print(f"Evaluation callback configured:")
    print(f"  → Evaluate every {eval_freq:,} steps")
    print(f"  → {n_eval_episodes} episodes per evaluation")
    print(f"  → Best model saves to: {run_model_dir}/best_model.zip")
    
    # === TASK 1: Setup TensorBoard Logging Callback ===
    tb_callback = TensorBoardLoggingCallback(log_freq=5000, verbose=1)
    print(f"TensorBoard callback configured:")
    print(f"  → Logs to: {tb_run_dir}")
    print(f"  → Metrics: reward, success_rate, episode_length")
    
    # Combine callbacks
    callbacks = CallbackList([eval_callback, tb_callback])
    
    # Start training
    print(f"\n{'*'*80}")
    print(f" STARTING TRAINING FOR {total_timesteps:,} TIMESTEPS")
    print(f"{'*'*80}")
    print(f"\n[INFO] Monitor training with:")
    print(f"  → tensorboard --logdir {tb_run_dir}")
    print(f"  → Stats CSV: {run_log_dir}/action_stats.csv\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(resume_model_path)
        )
    except KeyboardInterrupt:
        print("\n\n*** Training interrupted by user! ***")
        print("Saving current model...")
    
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(run_model_dir, 'mode_selector_final')
    model.save(final_model_path)
    
    # Print completion summary
    print(f"\n{'='*80}")
    print(f" TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nTraining Statistics:")
    print(f"  Total Time:           {training_time/60:.1f} minutes")
    print(f"  Total Timesteps:      {total_timesteps:,}")
    print(f"\nOutput Files:")
    print(f"  Final Model:          {final_model_path}.zip")
    print(f"  Best Model:          {run_model_dir}/best_model.zip (auto-saved by EvalCallback)")
    print(f"  Configuration:       {config_path}")
    print(f"  TensorBoard Logs:    {tb_run_dir}/")
    print(f"  Action Statistics:   {run_log_dir}/action_stats.csv")
    print(f"  Eval Results:        {run_log_dir}/results.csv")
    print(f"\n{'─'*80}")
    print(f" NEXT STEPS:")
    print(f"  1. View TensorBoard:  tensorboard --logdir {tb_run_dir}")
    print(f"  2. Test model:        python test.py --model-path {final_model_path}.zip")
    print(f"  3. Plot results:      python plot_results.py --log-dir {run_log_dir}")
    print(f"{'─'*80}\n")
    
    return model, run_model_dir, run_log_dir


def main():
    """Main function with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train PPO mode selector for 4WID-4WIS robot (Enhanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training (500k steps, built-in curriculum)
  python train.py

  # Custom parameters
  python train.py --timesteps 1000000 --lr 1e-4 --map map_a

  # Custom reward weights
      python train.py --w1 2.4 --w2 1.8 --w3 0.45 --w4 0.05

  # Test existing model
  python train.py --test-only --model-path models/xxx/best_model.zip

TensorBoard Monitoring:
  tensorboard --logdir ./tb_logs/
        """
    )
    
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps (default: 500,000)')
    parser.add_argument('--map', type=str, default=None,
                       choices=MapManager.get_available_maps(),
                       help='Map type for training; omit to use the built-in curriculum over map_a, map_b, and map_c')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Rollout steps per update (default: 2048)')
    parser.add_argument('--n-epochs', type=int, default=8,
                       help='Epochs per update (default: 8)')
    parser.add_argument('--gamma', type=float, default=0.998,
                       help='Discount factor gamma (default: 0.998)')
    parser.add_argument('--gae-lambda', type=float, default=0.96,
                       help='GAE lambda (default: 0.96)')
    parser.add_argument('--ent-coef', type=float, default=0.0005,
                       help='Entropy coefficient (default: 0.0005)')
    parser.add_argument('--clip-range', type=float, default=0.15,
                       help='PPO clipping range (default: 0.15)')
    
    # Reward weights
    parser.add_argument('--w1', type=float, default=3.0,
                       help='Progress reward weight w1 (default: 3.0)')
    parser.add_argument('--w2', type=float, default=1.6,
                       help='Cross-track penalty weight w2 (default: 1.6)')
    parser.add_argument('--w3', type=float, default=0.7,
                       help='Heading penalty weight w3 (default: 0.7)')
    parser.add_argument('--w4', type=float, default=0.03,
                       help='Mode switch penalty weight w4 (default: 0.03)')
    parser.add_argument('--max-time', type=float, default=90.0,
                       help='Maximum episode time in seconds (default: 90.0)')
    
    # Evaluation settings (Task 2)
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency in steps (default: 10000)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Episodes per evaluation (default: 10)')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Disable reset randomization during training and evaluation')
    parser.add_argument('--resume-model-path', type=str, default=None,
                       help='Resume training from an existing PPO checkpoint')
    
    # Testing mode
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for testing')
    
    args = parser.parse_args()
    
    if args.test_only:
        if args.model_path is None:
            print("Error: --model-path required when using --test-only")
            return
        # Import and run test function
        from test import test_trained_model
        test_trained_model(
            args.model_path,
            args.map,
            num_episodes=20,
            randomize=not args.no_randomize
        )
    else:
        # Build reward weights
        reward_weights = {
            'w1': args.w1,
            'w2': args.w2,
            'w3': args.w3,
            'w4': args.w4
        }
        
        # Start training
        train_mode_selector(
            total_timesteps=args.timesteps,
            map_type=args.map,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            reward_weights=reward_weights,
            randomize=not args.no_randomize,
            max_time=args.max_time,
            resume_model_path=args.resume_model_path
        )


if __name__ == "__main__":
    main()
