"""
Training script for the end-to-end continuous RL baseline.

This baseline uses PPO to output 8-D wheel-level continuous commands:
    [v_FL, v_FR, v_RL, v_RR, delta_FL, delta_FR, delta_RL, delta_RR]

The goal is to provide a direct-control baseline for comparing against the
paper's hierarchical discrete mode-switching policy.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np

_CACHE_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.cache')
os.makedirs(os.path.join(_CACHE_ROOT, 'matplotlib'), exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_CACHE_ROOT, 'matplotlib'))

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import ConstantSchedule
except ImportError:
    print("Error: stable-baselines3 not installed!")
    print("Please install with: pip install stable-baselines3")
    sys.exit(1)

from map_manager import MapManager
from env.e2e_continuous_env import EndToEndContinuousEnv


class TensorBoardLoggingCallback(BaseCallback):
    """Lightweight TensorBoard logger for the continuous baseline."""

    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []

    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            last_info = self.locals['infos'][-1]
            if 'episode' in last_info:
                ep_info = last_info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.successes.append(last_info.get('is_success', False))
                if self.num_timesteps % self.log_freq == 0:
                    self._log_to_tensorboard()
        return True

    def _log_to_tensorboard(self):
        if len(self.episode_rewards) == 0:
            return

        window = min(100, len(self.episode_rewards))
        avg_reward = float(np.mean(self.episode_rewards[-window:]))
        avg_length = float(np.mean(self.episode_lengths[-window:]))
        success_rate = float(np.mean(self.successes[-window:]) * 100.0)

        self.logger.record('reward/episode_reward', avg_reward)
        self.logger.record('success_rate', success_rate)
        self.logger.record('episode_length', avg_length)

        if self.verbose > 0:
            print(f"\n[TensorBoard] Step {self.num_timesteps:,}")
            print(f"  Avg Reward (last {window}): {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Avg Length: {avg_length:.1f}")


def _build_run_label(map_type):
    if isinstance(map_type, (list, tuple)):
        return 'multi_' + '_'.join(map(str, map_type))
    return str(map_type)


def create_env(
    map_type=None,
    max_time=50.0,
    reward_weights=None,
    randomize=True,
    log_dir='./logs/'
):
    if map_type is None:
        map_type = [
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
            'map_c',
        ]
    env = EndToEndContinuousEnv(
        map_type=map_type,
        max_time=max_time,
        dt=0.02,
        steps_per_action=5,
        reward_weights=reward_weights,
        randomize=randomize,
        log_dir=log_dir,
    )
    return Monitor(env)


def train_continuous_policy(
    total_timesteps=800000,
    map_type=None,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    save_freq=50000,
    eval_freq=10000,
    n_eval_episodes=10,
    log_dir='./logs/',
    model_dir='./models/',
    tensorboard_dir='./tb_logs/',
    reward_weights=None,
    randomize=True,
    max_time=50.0,
    resume_model_path=None,
):
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
        print(f"Training curriculum: {training_map_types}")
    else:
        print(f"Training on single map: {training_map_types[0]}")

    run_label = _build_run_label(training_map_types)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'e2e_continuous_{run_label}_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    model_save_dir = os.path.join(model_dir, model_name)
    tensorboard_run_dir = os.path.join(tensorboard_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tensorboard_run_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(" END-TO-END CONTINUOUS RL TRAINING SYSTEM - 4WID-4WIS ROBOT")
    print("=" * 80)
    print(f"  Algorithm:            PPO")
    print(f"  Action space:         8-D wheel commands")
    print(f"  Save directory:       {model_save_dir}")
    print(f"  TensorBoard directory:{tensorboard_run_dir}")
    print(f"  Learning rate:        {learning_rate}")
    print(f"  Rollout steps:        {n_steps}")
    print(f"  Batch size:           {batch_size}")
    print(f"  Epochs/update:        {n_epochs}")
    print(f"  Gamma:                {gamma}")
    print(f"  GAE lambda:           {gae_lambda}")
    print(f"  Clip range:           {clip_range}")
    print(f"  Entropy coef:         {ent_coef}")
    print("=" * 80)

    config = {
        'algorithm': 'PPO',
        'policy_type': 'MlpPolicy',
        'model_name': model_name,
        'map_type': map_type,
        'training_map_types': training_map_types,
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'save_freq': save_freq,
        'eval_freq': eval_freq,
        'n_eval_episodes': n_eval_episodes,
        'randomize': randomize,
        'max_time': max_time,
        'resume_model_path': resume_model_path,
        'action_space': '8D wheel commands',
        'observation_shape': 37,
    }
    with open(os.path.join(model_save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    training_env = create_env(
        map_type=training_map_types,
        max_time=max_time,
        reward_weights=reward_weights,
        randomize=randomize,
        log_dir=log_dir
    )
    eval_env = create_env(
        map_type='tri_mode_composite',
        max_time=max_time,
        reward_weights=reward_weights,
        randomize=False,
        log_dir=os.path.join(log_dir, 'eval')
    )

    print(f"\nTraining environment observation space: {training_env.observation_space.shape}")
    print(f"Training environment action space: {training_env.action_space}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_dir,
        log_path=model_save_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_save_dir,
        name_prefix='checkpoint'
    )
    tb_callback = TensorBoardLoggingCallback(log_freq=1000, verbose=1)
    callback = CallbackList([tb_callback, eval_callback, checkpoint_callback])

    if resume_model_path and os.path.exists(resume_model_path):
        print(f"\nResuming training from {resume_model_path}")
        model = PPO.load(resume_model_path, env=training_env, print_system_info=False)
    else:
        print("\nInitializing PPO model...")
        model = PPO(
            'MlpPolicy',
            training_env,
            verbose=1,
            tensorboard_log=tensorboard_run_dir,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    if isinstance(model.learning_rate, ConstantSchedule):
        lr_value = model.learning_rate(1.0)
    else:
        lr_value = model.learning_rate
    print(f"Model initialized. Current learning rate: {lr_value}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=resume_model_path is None
    )

    final_path = os.path.join(model_save_dir, 'final_model')
    best_path = os.path.join(model_save_dir, 'best_model')
    model.save(final_path)
    if os.path.exists(best_path + '.zip'):
        print(f"Best model saved to {best_path}.zip")
    print(f"Final model saved to {final_path}.zip")

    training_env.close()
    eval_env.close()

    return model, model_save_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train end-to-end continuous RL baseline for 4WID-4WIS robot')
    parser.add_argument('--total-timesteps', type=int, default=800000)
    parser.add_argument(
        '--map-type',
        type=str,
        default=None,
        choices=MapManager.get_available_maps(),
        help='Map type for training; omit to use the built-in curriculum over map_a, map_b, and map_c',
    )
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--ent-coef', type=float, default=0.001)
    parser.add_argument('--save-freq', type=int, default=50000)
    parser.add_argument('--eval-freq', type=int, default=10000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--log-dir', type=str, default='./logs/')
    parser.add_argument('--model-dir', type=str, default='./models/')
    parser.add_argument('--tensorboard-dir', type=str, default='./tb_logs/')
    parser.add_argument('--randomize', dest='randomize', action='store_true', default=True)
    parser.add_argument('--no-randomize', dest='randomize', action='store_false')
    parser.add_argument('--max-time', type=float, default=50.0)
    parser.add_argument('--resume-model-path', type=str, default=None)
    parser.add_argument('--steer-rate-limit-deg', type=float, default=18.0)
    parser.add_argument('--speed-rate-limit', type=float, default=1.5)
    parser.add_argument('--residual-penalty-weight', type=float, default=0.15)
    args = parser.parse_args()

    reward_weights = {
        'w1': 3.0,
        'w2': 1.6,
        'w3': 0.7,
        'w4': 0.04,
    }

    train_continuous_policy(
        total_timesteps=args.total_timesteps,
        map_type=args.map_type,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        tensorboard_dir=args.tensorboard_dir,
        reward_weights=reward_weights,
        randomize=args.randomize,
        max_time=args.max_time,
        resume_model_path=args.resume_model_path,
    )


if __name__ == '__main__':
    main()
