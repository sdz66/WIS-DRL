"""
End-to-End Continuous RL environment for 4WID-4WIS wheel-level control.

This environment keeps the same map, observation, and reward scaffolding as
ModeEnv, but replaces the discrete AFM/APT/AZR decision with direct 8-D wheel
commands:

    [v_FL, v_FR, v_RL, v_RR, delta_FL, delta_FR, delta_RL, delta_RR]

The first four entries are wheel speeds, the last four are steering angles.
The action is normalized to [-1, 1] and then mapped to physical bounds.
"""

import os
import sys

import numpy as np
from gymnasium import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.mode_env import ModeEnv


class EndToEndContinuousEnv(ModeEnv):
    """
    PPO environment for end-to-end continuous wheel-level control.

    Compared with the hierarchical mode selector, this baseline exposes an
    8-D continuous action space and does not use the internal AFM/APT/AZR
    override logic. It is intended to validate whether direct wheel-level
    control can match the structured discrete-mode policy.
    """

    def __init__(
        self,
        map_type=None,
        max_time=50.0,
        dt=0.02,
        steps_per_action=5,
        reward_weights=None,
        randomize=True,
        log_dir='./logs/',
        steer_rate_limit_deg=18.0,
        speed_rate_limit=1.5,
        residual_penalty_weight=0.15,
    ):
        continuous_reward_weights = reward_weights if reward_weights else {
            'w1': 3.0,
            'w2': 1.6,
            'w3': 0.7,
            'w4': 0.04,  # control-smoothness penalty weight in the continuous baseline
        }

        super().__init__(
            map_type=map_type,
            max_time=max_time,
            dt=dt,
            steps_per_action=steps_per_action,
            reward_weights=continuous_reward_weights,
            randomize=randomize,
            log_dir=log_dir,
            enable_internal_mode_override=False,
        )

        self.control_mode = 'continuous_rl'
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        self.wheel_speed_max = float(self.max_speed)
        self.wheel_steer_max = float(getattr(self.env_map, 'max_steering_angle', np.deg2rad(30.0)))
        self.wheel_speed_rate_limit = float(speed_rate_limit)
        self.wheel_steer_rate_limit = float(np.deg2rad(steer_rate_limit_deg))
        self.residual_penalty_weight = float(residual_penalty_weight)

        self._rebuild_wheel_geometry()
        self.prev_wheel_command = np.zeros(8, dtype=np.float32)
        self.last_wheel_command = np.zeros(8, dtype=np.float32)
        self.last_slip_residual = 0.0

    def _rebuild_wheel_geometry(self):
        """Cache the wheel positions used by the least-squares twist fit."""
        lf = float(getattr(self.env_map, 'wheelbase', 1.0)) / 2.0
        lr = lf
        lw = float(getattr(self.env_map, 'track_width', 0.69)) / 2.0

        # Wheel order matches the paper's wheel-level command vector:
        # FL, FR, RL, RR.
        self.wheel_positions = np.array([
            [lf, lw],
            [lf, -lw],
            [-lr, lw],
            [-lr, -lw],
        ], dtype=float)

    def reset(self, seed=None):
        """Reset the underlying map state and the wheel-level command memory."""
        obs, info = super().reset(seed=seed)
        self.prev_wheel_command = np.zeros(8, dtype=np.float32)
        self.last_wheel_command = np.zeros(8, dtype=np.float32)
        self.last_slip_residual = 0.0
        self.control_mode = 'continuous_rl'

        # Keep the observation shape identical to the hierarchical baseline,
        # but blank out the discrete-mode hint/history channels so the policy
        # does not receive a mode-selection oracle.
        obs = self.get_obs()
        return obs, info

    def get_obs(self):
        """
        Return the same geometric/body-state features as ModeEnv, but remove
        the discrete mode hint and action history channels.
        """
        obs = super().get_obs().astype(np.float32)
        path_dim = len(self.path_lookahead_offsets) * 3
        body_dim = 8
        mode_hint_start = path_dim + body_dim
        mode_hint_end = mode_hint_start + self.mode_hint_len + self.action_history_len * 3
        obs[mode_hint_start:mode_hint_end] = 0.0
        return obs

    def _normalize_action(self, action):
        """Map a PPO action in [-1, 1] to wheel speeds and steering angles."""
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != 8:
            raise ValueError(f"Continuous policy must output 8 values, got shape {action.shape}")

        wheel_speeds_target = np.clip(action[:4], -1.0, 1.0) * self.wheel_speed_max
        wheel_steers_target = np.clip(action[4:], -1.0, 1.0) * self.wheel_steer_max

        prev_speeds = self.prev_wheel_command[:4]
        prev_steers = self.prev_wheel_command[4:]

        wheel_speeds = prev_speeds + np.clip(
            wheel_speeds_target - prev_speeds,
            -self.wheel_speed_rate_limit,
            self.wheel_speed_rate_limit
        )
        wheel_steers = prev_steers + np.clip(
            wheel_steers_target - prev_steers,
            -self.wheel_steer_rate_limit,
            self.wheel_steer_rate_limit
        )

        wheel_speeds = np.clip(wheel_speeds, -self.wheel_speed_max, self.wheel_speed_max)
        wheel_steers = np.clip(wheel_steers, -self.wheel_steer_max, self.wheel_steer_max)

        wheel_command = np.concatenate([wheel_speeds, wheel_steers]).astype(np.float32)
        return wheel_command

    def _wheel_commands_to_body_twist(self, wheel_command):
        """
        Approximate the body-frame twist induced by a set of wheel-level
        steering angles and wheel speeds via least squares.
        """
        wheel_speeds = np.asarray(wheel_command[:4], dtype=float)
        wheel_steers = np.asarray(wheel_command[4:], dtype=float)

        rows = []
        targets = []
        for (x_i, y_i), speed_i, steer_i in zip(self.wheel_positions, wheel_speeds, wheel_steers):
            target_x = speed_i * np.cos(steer_i)
            target_y = speed_i * np.sin(steer_i)
            rows.append([1.0, 0.0, -y_i])
            targets.append(target_x)
            rows.append([0.0, 1.0, x_i])
            targets.append(target_y)

        a_mat = np.asarray(rows, dtype=float)
        b_vec = np.asarray(targets, dtype=float)
        twist, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        residual = a_mat @ twist - b_vec
        residual_rms = float(np.sqrt(np.mean(np.square(residual))))

        twist = np.asarray(twist, dtype=float)
        twist[0] = float(np.clip(twist[0], -self.max_speed, self.max_speed))
        twist[1] = float(np.clip(twist[1], -self.max_speed, self.max_speed))
        twist[2] = float(np.clip(twist[2], -self.max_yaw_rate, self.max_yaw_rate))
        return twist, residual_rms

    def compute_reward(self, control_penalty=0.0, residual_rms=0.0):
        """
        Continuous-control reward.

        We reuse the path-tracking / progress shaping from ModeEnv and add
        wheel-level smoothness and feasibility regularizers so the PPO policy
        learns to emit physically coherent wheel commands.
        """
        base_reward, info = super().compute_reward(mode_switch_penalty=0.0, action=None)

        prev_cmd = self.prev_wheel_command
        current_cmd = self.last_wheel_command

        steer_delta = np.abs(current_cmd[4:] - prev_cmd[4:]) / max(self.wheel_steer_max, 1e-6)
        speed_delta = np.abs(current_cmd[:4] - prev_cmd[:4]) / max(self.wheel_speed_max, 1e-6)

        control_smoothness = -0.5 * float(np.mean(steer_delta) + 0.6 * np.mean(speed_delta))
        control_smoothness *= float(self.reward_weights.get('w4', 0.04))
        feasibility_penalty = -self.residual_penalty_weight * float(min(residual_rms / 0.25, 1.0))

        reward = base_reward + control_smoothness + feasibility_penalty - float(control_penalty)

        info.update({
            'control_mode': self.control_mode,
            'control_smoothness': control_smoothness,
            'feasibility_penalty': feasibility_penalty,
            'wheel_slip_residual_rms': residual_rms,
            'wheel_speed_cmds': current_cmd[:4].copy(),
            'wheel_steer_cmds': current_cmd[4:].copy(),
        })

        return reward, info

    def step(self, action):
        """
        Execute one continuous-control step.

        The policy outputs 8 normalized wheel commands, which are mapped to
        physical wheel speeds and steering angles before being converted to an
        approximate body twist through least-squares inverse kinematics.
        """
        wheel_command = self._normalize_action(action)
        body_twist, residual_rms = self._wheel_commands_to_body_twist(wheel_command)

        start_state = self.state.copy()
        self.last_wheel_command = wheel_command.copy()
        self.last_slip_residual = residual_rms

        for _ in range(self.steps_per_action):
            self._update_state_holonomic(body_twist)
            self.current_time += self.dt
            self.state_history.append(self.state.copy())

            if (
                self._check_goal_reached() or
                self._check_collision() or
                self._out_of_bounds() or
                self.current_time >= self.max_time
            ):
                break

        self.last_step_motion = self._compute_step_motion(start_state, self.state)
        if (
            self.last_step_motion['displacement'] < 0.015 and
            abs(self.last_step_motion['heading_change']) < np.deg2rad(1.0)
        ):
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        observation = self.get_obs()
        reward, info = self.compute_reward(control_penalty=0.0, residual_rms=residual_rms)

        terminated = (
            self._check_goal_reached() or
            self._check_collision() or
            self._out_of_bounds() or
            self.no_progress_steps >= self.no_progress_limit
        )
        truncated = self.current_time >= self.max_time and not terminated
        self.last_failure_reason = self._get_failure_reason()

        self.episode_steps += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        self.prev_wheel_command = wheel_command.copy()

        info.update({
            'map_type': self.map_type,
            'episode_step': self.episode_steps,
            'current_time': self.current_time,
            'current_mode': -1,
            'requested_mode': -1,
            'executed_mode': -1,
            'active_primitive_mode': 0,
            'position': self.state[:2].copy(),
            'heading_rad': self.state[2],
            'total_reward': self.total_reward,
            'step_displacement': self.last_step_motion['displacement'],
            'step_longitudinal': self.last_step_motion['longitudinal'],
            'step_lateral': self.last_step_motion['lateral'],
            'step_heading_change': self.last_step_motion['heading_change'],
            'failure_reason': self.last_failure_reason,
            'afm_count': 0,
            'apt_count': 0,
            'azr_count': 0,
            'requested_afm_count': 0,
            'requested_apt_count': 0,
            'requested_azr_count': 0,
            'mode_switch_count': 0,
            'apt_candidate_steps': 0,
            'azr_candidate_steps': 0,
            'blocked_steps': 0,
            'stuck_steps': self.no_progress_steps,
            'total_actions': self.episode_steps,
            'is_success': self._check_goal_reached(),
            'distance_to_goal': np.linalg.norm(self.state[:2] - self.goal_position),
            'primitive_aborted': False,
            'primitive_active': False,
            'control_mode': self.control_mode,
            'wheel_command': wheel_command.copy(),
            'body_twist': body_twist.copy(),
            'wheel_slip_residual_rms': residual_rms
        })

        return observation, reward, terminated, truncated, info
