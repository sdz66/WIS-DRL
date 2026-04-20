"""
Mode Switching Environment for Reinforcement Learning (Enhanced Version)
Gym-style environment for training PPO agent to select AFM/APT/AZR modes

Features:
  - Randomized initialization (Task 3)
  - Mode usage statistics tracking (Task 4)  
  - Optimized reward function with smooth/efficiency/progress rewards (Task 5)
  - TensorBoard-compatible logging
  - Evaluation-ready interface
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import csv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from map_manager import MapManager
from controllers.afm_step import AFMStep
from controllers.apt_step import APTStep
from controllers.azr_step import AZRStep


class ModeEnv(gym.Env):
    """
    Enhanced Mode Switching Environment for 4WID-4WIS Robot
    
    Action space: Discrete(3)
        0 = AFM (Autonomous Free Movement - Path tracking)
        1 = APT (Autonomous Parallel Translation)
        2 = AZR (Autonomous Zero Radius - In-place rotation)
    
    Observation space: Box(37)
        [local_path(4x3), body_state(8), mode_hint(3), action_history(3x3), env_clearance(5)]
    
    Enhanced Features:
        - Randomized initial state for generalization
        - Comprehensive mode usage statistics
        - Multi-component reward shaping
        - CSV logging for analysis
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, map_type=None, max_time=50.0, dt=0.02, steps_per_action=5,
                 reward_weights=None, randomize=True, log_dir='./logs/',
                 enable_internal_mode_override=True):
        """
        Initialize Enhanced ModeEnv
        
        Args:
            map_type: Type of map to use. Can be a single map name, a list of
                     map names for multi-map training, or None to use the
                     built-in curriculum over map_a/map_b/map_c.
            max_time: Maximum episode time (seconds)
            dt: Time step for low-level control (seconds) - 0.02s for 50Hz
            steps_per_action: Number of low-level steps per RL action (5 for 10Hz high-level)
            reward_weights: Dict with keys w1-w4 for reward shaping
                           w1: progress reward weight (default: 1.5)
                           w2: cross-track penalty weight (default: 2.1)
                           w3: heading penalty weight (default: 0.55)
                           w4: mode_switch penalty weight (default: 0.08)
            randomize: Whether to randomize initial state (Task 3)
            log_dir: Directory for saving statistics logs
            enable_internal_mode_override: Whether to allow the environment to
                override AFM requests with built-in APT/AZR heuristics.
        """
        super(ModeEnv, self).__init__()
        
        # Environment parameters - 10Hz/50Hz configuration
        self.map_choices = self._normalize_map_choices(map_type)
        self.multi_map_training = len(self.map_choices) > 1
        self.map_cycle_index = 0
        self.map_type = None
        self.max_time = max_time
        self.dt = dt
        self.steps_per_action = steps_per_action
        self.decision_dt = self.dt * self.steps_per_action
        self.randomize = randomize
        self.log_dir = log_dir
        self.enable_internal_mode_override = enable_internal_mode_override
        
        # Reward weights configuration (w1, w2, w3, w4)
        default_weights = {
            'w1': 3.0,      # Progress reward weight
            'w2': 1.6,      # Cross-track penalty weight
            'w3': 0.7,      # Heading penalty weight
            'w4': 0.03      # Mode switch penalty weight
        }
        self.reward_weights = reward_weights if reward_weights else default_weights
        
        self.manager = MapManager()

        # Observation scaling and history parameters
        self.path_lookahead_offsets = (0, 5, 10, 20)
        self.action_history_len = 3
        self.mode_hint_len = 3
        self.max_clearance_distance = 3.0
        self.max_speed = 5.0
        self.max_yaw_rate = 1.5
        self.position_scale = 1.0
        self.e_perp_max = 0.75
        self.e_theta_max = np.deg2rad(35.0)
        self.terminal_rewards = {
            'success': 120.0,
            'collision': -80.0,
            'timeout': -50.0,
            'stuck': -50.0
        }
        # Small auxiliary shaping term to help PPO discover the rare APT/AZR
        # transitions on the composite training map without overwhelming the
        # path-tracking and safety terms. The reward is intentionally modest so
        # the policy cannot "farm" alignment by staying in a primitive.
        self.mode_alignment_reward = 0.06
        self.mode_alignment_penalty = 0.12
        self.active_primitive_mode = 0
        
        # Current state [x, y, psi, v]
        self.state = None
        self.current_time = 0.0
        self.current_mode = 0  # Default to AFM
        self.previous_mode = 0
        
        # Episode tracking
        self.episode_steps = 0
        self.total_reward = 0.0
        
        # History for visualization and analysis
        self.state_history = []
        self.mode_history = []
        self.reward_history = []
        
        # Task 4: Mode usage statistics
        # `requested_mode_counts` tracks PPO requests, while `mode_counts`
        # tracks the executed mode after any safety override.
        self.requested_mode_counts = {0: 0, 1: 0, 2: 0}
        self.mode_counts = {0: 0, 1: 0, 2: 0}  # AFM, APT, AZR executed counts
        self.mode_switch_count = 0
        self.episode_stats = []  # Store per-episode stats
        self.apt_active_steps = 0
        self.azr_active_steps = 0
        self.apt_stall_steps = 0
        self.azr_stall_steps = 0
        self.max_apt_dwell_steps = 40
        self.max_azr_dwell_steps = 30
        self.apt_stall_limit = 6
        self.azr_stall_limit = 6
        self.apt_cooldown_steps = 0
        self.azr_cooldown_steps = 0
        self.cooldown_after_apt = 18
        self.cooldown_after_azr = 12
        
        # Task 5: Reward tracking for progress bonus
        self.prev_distance_to_goal = None
        self.prev_path_progress = 0.0
        self.prev_lateral_error = None
        self.prev_heading_error = None
        self.last_step_motion = {
            'displacement': 0.0,
            'longitudinal': 0.0,
            'lateral': 0.0,
            'heading_change': 0.0
        }
        self.last_kinematics = {'vx': 0.0, 'vy': 0.0, 'omega': 0.0}
        self.no_progress_steps = 0
        self.no_progress_limit = 1000
        self.last_failure_reason = 'running'
        self.apt_candidate_steps = 0
        self.azr_candidate_steps = 0
        self.blocked_steps = 0
        self.mode_action_history = [0] * self.action_history_len
        
        # Define action space: 0=AFM, 1=APT, 2=AZR
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        obs_dim = (
            len(self.path_lookahead_offsets) * 3 +
            8 +
            self.mode_hint_len +
            self.action_history_len * 3 +
            5
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Goal threshold
        self.goal_threshold = 0.35  # meters
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Load the initial map configuration before the first reset.
        self._load_map(self.map_choices[0])
        
        map_label = self.map_type if not self.multi_map_training else self.map_choices
        print(f"ModeEnv initialized: map={map_label}, max_time={max_time}s")
        print(f"Randomization: {'Enabled' if randomize else 'Disabled'}")
        print(f"Base initial state: {self.base_initial_state}")
        print(f"Base goal position: {self.goal_position}")
        
    def reset(self, seed=None):
        """
        Reset environment to (possibly randomized) initial state (Task 3)
        
        Randomization parameters:
            - Initial position: ±0.5m in x and y
            - Initial heading: ±15 degrees (±0.262 rad)
            - Goal position: ±0.3m perturbation
            - Goal heading: ±10 degrees (±0.175 rad)
        
        Returns:
            observation: Initial observation
            info: Additional info dict
        """
        super().reset(seed=seed)

        # Cycle through the multi-map training set if enabled. For single-map
        # training, this is a no-op.
        selected_map = self._select_map_for_reset()
        if selected_map != self.map_type:
            self._load_map(selected_map)

        # Task 3: Randomize initial state and goal, but reject samples that
        # start outside the safe/drivable region because they poison training.
        self.state, self.goal_position, self.goal_heading = self._sample_reset_state()
        
        # Reset timing and mode
        self.current_time = 0.0
        self.current_mode = 0
        self.previous_mode = 0
        
        # Reset episode tracking
        self.episode_steps = 0
        self.total_reward = 0.0
        
        # Reset mode statistics for new episode
        self.requested_mode_counts = {0: 0, 1: 0, 2: 0}
        self.mode_counts = {0: 0, 1: 0, 2: 0}
        self.mode_switch_count = 0
        self.apt_candidate_steps = 0
        self.azr_candidate_steps = 0
        self.blocked_steps = 0
        self.no_progress_steps = 0
        self.last_failure_reason = 'running'
        self.mode_action_history = [0] * self.action_history_len
        self.active_primitive_mode = 0
        self.apt_active_steps = 0
        self.azr_active_steps = 0
        self.apt_stall_steps = 0
        self.azr_stall_steps = 0
        self.apt_cooldown_steps = 0
        self.azr_cooldown_steps = 0
        
        # Initialize previous distance for progress tracking
        self.prev_distance_to_goal = np.linalg.norm(self.state[:2] - self.goal_position)
        self.prev_path_progress = 0.0
        self.prev_lateral_error = None
        self.prev_heading_error = None
        self.last_step_motion = {
            'displacement': 0.0,
            'longitudinal': 0.0,
            'lateral': 0.0,
            'heading_change': 0.0
        }
        self.last_kinematics = {'vx': 0.0, 'vy': 0.0, 'omega': 0.0}
        
        # Reset controller internal state so mode-specific targets do not leak
        self.afm_controller.reset()
        self.apt_controller.reset()
        self.azr_controller.reset()
        
        # Clear history
        self.state_history = [self.state.copy()]
        self.mode_history = [self.current_mode]
        self.reward_history = []
        
        return self.get_obs(), {'map_type': self.map_type}

    def _sample_reset_state(self, max_attempts=100):
        """Sample a valid initial state and goal configuration."""
        if not self.randomize:
            return (
                self.base_initial_state.copy(),
                self.base_goal_position.copy(),
                self.base_goal_heading
            )

        for _ in range(max_attempts):
            # Random position offset (±0.5m)
            pos_noise = np.random.uniform(-0.5, 0.5, size=2)

            # Random heading offset (±15° = ±0.262 rad)
            heading_noise = np.random.uniform(-0.262, 0.262)

            randomized_init = self.base_initial_state.copy()
            randomized_init[0] += pos_noise[0]
            randomized_init[1] += pos_noise[1]
            randomized_init[2] = np.arctan2(
                np.sin(randomized_init[2] + heading_noise),
                np.cos(randomized_init[2] + heading_noise)
            )

            # Random goal position perturbation (±0.3m)
            goal_position = self.base_goal_position + np.random.uniform(-0.3, 0.3, size=2)

            # Random goal heading perturbation (±10° = ±0.175 rad)
            goal_heading_noise = np.random.uniform(-0.175, 0.175)
            goal_heading = np.arctan2(
                np.sin(self.base_goal_heading + goal_heading_noise),
                np.cos(self.base_goal_heading + goal_heading_noise)
            )

            self.state = randomized_init.copy()
            self.goal_position = goal_position.copy()
            self.goal_heading = goal_heading

            if self._check_collision() or self._out_of_bounds():
                continue
            if not self._is_drivable_point(goal_position[0], goal_position[1]):
                continue

            return randomized_init, goal_position, goal_heading

        # Fall back to the deterministic reset if randomized sampling cannot
        # produce a valid configuration after repeated attempts.
        deterministic_state = self.base_initial_state.copy()
        deterministic_goal = self.base_goal_position.copy()
        deterministic_heading = self.base_goal_heading
        self.state = deterministic_state.copy()
        self.goal_position = deterministic_goal.copy()
        self.goal_heading = deterministic_heading
        return deterministic_state, deterministic_goal, deterministic_heading

    def _normalize_map_choices(self, map_type):
        """Normalize a single map name or iterable of map names."""
        if map_type is None:
            choices = ['map_a', 'map_b', 'map_c']
        elif isinstance(map_type, str):
            choices = [map_type]
        else:
            try:
                choices = list(map_type)
            except TypeError:
                choices = [map_type]

        normalized = [MapManager._normalize_map_type(choice) for choice in choices]

        if not normalized:
            raise ValueError("At least one map type must be provided")
        return normalized

    def _select_map_for_reset(self):
        """Choose the next map for this episode."""
        if not self.multi_map_training:
            return self.map_choices[0]

        selected = self.map_choices[self.map_cycle_index % len(self.map_choices)]
        self.map_cycle_index += 1
        return selected

    def _load_map(self, map_type):
        """Load a map and rebuild all map-dependent state."""
        canonical = MapManager._normalize_map_type(map_type)
        self.map_type = canonical
        self.env_map = self.manager.create_map(canonical)
        self.reference_path = self.env_map.reference_path.copy()

        self.base_initial_state = np.array([
            self.env_map.initial_state['x'],
            self.env_map.initial_state['y'],
            self.env_map.initial_state['psi'],
            0.0
        ])
        self.base_goal_position = np.array(self.env_map.end_point)
        self.base_goal_heading = self.env_map.end_heading
        self.goal_position = self.base_goal_position.copy()
        self.goal_heading = self.base_goal_heading

        self.max_speed = getattr(self.env_map, 'max_speed', 5.0)
        self.position_scale = max(
            self.env_map.x_range[1] - self.env_map.x_range[0],
            self.env_map.y_range[1] - self.env_map.y_range[0],
            1.0
        )
        self.e_perp_max = max(0.6, 0.08 * self.position_scale)
        self.e_theta_max = np.deg2rad(35.0)

        # Rebuild the controllers so their internal compatibility map matches
        # the currently active map.
        self.afm_controller = AFMStep(map_type=canonical)
        self.apt_controller = APTStep(map_type=canonical)
        self.azr_controller = AZRStep(map_type=canonical)
        self.reference_path_segment_lengths = np.linalg.norm(
            np.diff(self.reference_path[:, :2], axis=0),
            axis=1
        )
        self.reference_path_cumulative_lengths = np.concatenate(
            ([0.0], np.cumsum(self.reference_path_segment_lengths))
        )
        self.reference_path_total_length = float(
            self.reference_path_cumulative_lengths[-1]
            if len(self.reference_path_cumulative_lengths) > 0 else 1.0
        )

        return canonical

    def _compute_path_progress(self, nearest_idx=None):
        """Compute continuous progress along the reference path."""
        if self.reference_path is None or len(self.reference_path) <= 1:
            return 0.0

        if nearest_idx is None:
            nearest_idx = self._find_nearest_path_index()

        nearest_idx = int(np.clip(nearest_idx, 0, len(self.reference_path) - 1))
        total_length = max(getattr(self, 'reference_path_total_length', 1.0), 1e-6)

        if nearest_idx >= len(self.reference_path) - 1:
            return 1.0

        p0 = self.reference_path[nearest_idx, :2]
        p1 = self.reference_path[nearest_idx + 1, :2]
        segment = p1 - p0
        segment_length = float(np.linalg.norm(segment))

        if segment_length < 1e-9:
            progress = self.reference_path_cumulative_lengths[nearest_idx] / total_length
            return float(np.clip(progress, 0.0, 1.0))

        projection = float(
            np.dot(self.state[:2] - p0, segment) / max(segment_length ** 2, 1e-9)
        )
        projection = float(np.clip(projection, 0.0, 1.0))
        progress = (
            self.reference_path_cumulative_lengths[nearest_idx] +
            projection * segment_length
        ) / total_length
        return float(np.clip(progress, 0.0, 1.0))
    
    def step(self, action):
        """
        Execute one RL step with enhanced reward calculation (Task 5)
        
        Args:
            action: Mode selection (0=AFM, 1=APT, 2=AZR) - can be numpy array or scalar
        
        Returns:
            observation: Next observation
            reward: Enhanced reward value
            terminated: Whether episode is finished (natural termination)
            truncated: Whether episode was truncated (timeout etc.)
            info: Additional information dict with mode stats
        """
        # Convert action to Python integer (handles numpy array case from SB3)
        requested_action = int(action)

        # Update requested-mode tracking. The controller may keep executing an
        # in-progress primitive until it completes.
        self.previous_mode = self.current_mode
        self.current_mode = requested_action

        self.apt_cooldown_steps = max(0, self.apt_cooldown_steps - 1)
        self.azr_cooldown_steps = max(0, self.azr_cooldown_steps - 1)

        # Task 4: Track requested mode usage
        self.requested_mode_counts[requested_action] += 1

        # Compute context before executing the mode. This is also used to
        # configure APT/AZR targets when the policy switches modes.
        mode_context = self._get_mode_context()
        if mode_context['apt_candidate']:
            self.apt_candidate_steps += 1
        if mode_context['azr_candidate']:
            self.azr_candidate_steps += 1
        if mode_context['forward_blocked']:
            self.blocked_steps += 1

        self._update_action_history(requested_action)

        # Check for mode switch (Task 5: smooth reward)
        mode_switch_penalty = 0.0
        if self.previous_mode != self.current_mode:
            mode_switch_penalty = 1.0
            self.mode_switch_count += 1
            self._configure_mode_transition(requested_action, mode_context)

        # If requested, let the environment's built-in heuristic override an
        # AFM request when the local geometry clearly calls for a primitive.
        # This stays enabled for learned policies, but rule-based baselines can
        # disable it to keep the comparison pure.
        execution_action = requested_action
        if self.enable_internal_mode_override and requested_action == 0:
            if mode_context['azr_candidate'] and (
                (
                    abs(mode_context['path_heading_change']) > np.deg2rad(55) and
                    abs(mode_context['current_heading_error']) > np.deg2rad(50)
                ) or
                (
                    mode_context['forward_blocked'] and
                    abs(mode_context['current_heading_error']) > np.deg2rad(60)
                ) or
                (
                    self.map_type != 'map_a' and
                    mode_context['distance_to_goal'] < 1.1 and
                    abs(mode_context['current_heading_error']) > np.deg2rad(18)
                )
            ):
                execution_action = 2
            elif mode_context['apt_candidate'] and (
                abs(mode_context['current_lateral_error']) > 0.15 or
                abs(mode_context['path_heading_change']) > np.deg2rad(14) or
                mode_context['forward_blocked']
            ):
                execution_action = 1

        # Prevent a single primitive from monopolizing the episode. The
        # policy can still re-enter APT/AZR after this cap, but the cap keeps
        # the controller from staying in one mode long enough to stall the
        # episode on obstacle-heavy maps.
        if self.active_primitive_mode == 1:
            self.apt_active_steps += 1
            if self.apt_active_steps > self.max_apt_dwell_steps:
                self.apt_controller.reset()
                self.active_primitive_mode = 0
                self.apt_cooldown_steps = self.cooldown_after_apt
                execution_action = 0
        elif self.active_primitive_mode == 2:
            self.azr_active_steps += 1
            if self.azr_active_steps > self.max_azr_dwell_steps:
                self.azr_controller.reset()
                self.active_primitive_mode = 0
                self.azr_cooldown_steps = self.cooldown_after_azr
                execution_action = 0

        # Start or refresh a primitive whenever the executed mode is a
        # primitive. The primitive then stays active until it actually
        # finishes.
        if execution_action == 1 and (
            self.active_primitive_mode != 1 or
            not self.apt_controller.active or
            self.apt_controller.is_complete(self.state)
        ):
            self._configure_mode_transition(execution_action, mode_context)
            self.active_primitive_mode = 1
            self.apt_active_steps = 0
        elif execution_action == 2 and (
            self.active_primitive_mode != 2 or
            not self.azr_controller.is_rotating or
            self.azr_controller.is_complete(self.state)
        ):
            self._configure_mode_transition(execution_action, mode_context)
            self.active_primitive_mode = 2
            self.azr_active_steps = 0

        # Keep active motion primitives running until they truly finish.
        if self.active_primitive_mode == 1:
            if self.apt_controller.active and not self.apt_controller.is_complete(self.state):
                execution_action = 1
            else:
                self.active_primitive_mode = 0
        elif self.active_primitive_mode == 2:
            if self.azr_controller.is_rotating and not self.azr_controller.is_complete(self.state):
                execution_action = 2
            else:
                self.active_primitive_mode = 0

        # If a sharp turn appears while APT is active, hand control back so
        # the generalized mode logic can switch to AZR instead of locking the
        # robot into a long lateral slide.
        if self.active_primitive_mode == 1 and mode_context['azr_candidate']:
            remaining_translation = abs(
                self.apt_controller.target_lateral_offset -
                self.apt_controller.current_lateral_offset
            )
            if remaining_translation < 0.2 or mode_context['forward_clearance'] < 0.55:
                self.apt_controller.reset()
                self.active_primitive_mode = 0
                execution_action = 2

        # Track the actual mode that will be executed this high-level step.
        self.mode_counts[execution_action] += 1

        # Execute multiple low-level control steps
        start_state = self.state.copy()
        for _ in range(self.steps_per_action):
            if execution_action == 0:
                u = self.afm_controller.step(self.state, self.reference_path)
                self._update_state_afm(u)
            elif execution_action == 1:
                cmd = self.apt_controller.step(self.state)
                self._update_state_holonomic(cmd)
            else:  # action == 2
                cmd = self.azr_controller.step(self.state)
                self._update_state_holonomic(cmd)

            self.current_time += self.dt
            self.state_history.append(self.state.copy())
            self.mode_history.append(execution_action)

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

        post_mode_context = self._get_mode_context()
        observation = self.get_obs()
        reward, info = self.compute_reward(mode_switch_penalty, action=requested_action)

        primitive_aborted = False
        if execution_action == 1:
            lateral_progress = max(0.0, info.get('delta_lateral_error', 0.0))
            if lateral_progress < 0.001 and abs(self.last_step_motion['displacement']) < 0.01:
                self.apt_stall_steps += 1
            else:
                self.apt_stall_steps = 0
            if self.apt_stall_steps >= self.apt_stall_limit:
                self.apt_controller.reset()
                self.active_primitive_mode = 0
                self.apt_active_steps = 0
                self.apt_stall_steps = 0
                self.apt_cooldown_steps = self.cooldown_after_apt
                primitive_aborted = True
        elif execution_action == 2:
            heading_progress = max(0.0, info.get('delta_heading_error', 0.0))
            if heading_progress < np.deg2rad(0.25) and abs(self.last_step_motion['heading_change']) < np.deg2rad(0.25):
                self.azr_stall_steps += 1
            else:
                self.azr_stall_steps = 0
            if self.azr_stall_steps >= self.azr_stall_limit:
                self.azr_controller.reset()
                self.active_primitive_mode = 0
                self.azr_active_steps = 0
                self.azr_stall_steps = 0
                self.azr_cooldown_steps = self.cooldown_after_azr
                primitive_aborted = True

        terminated = (
            self._check_goal_reached() or
            self._check_collision() or
            self._out_of_bounds() or
            self.no_progress_steps >= self.no_progress_limit
        )
        truncated = self.current_time >= self.max_time and not terminated
        self.last_failure_reason = self._get_failure_reason()

        if execution_action == 1 and self.apt_controller.is_complete(self.state):
            self.active_primitive_mode = 0
            self.apt_active_steps = 0
            self.apt_cooldown_steps = self.cooldown_after_apt
        elif execution_action == 2 and self.azr_controller.is_complete(self.state):
            self.active_primitive_mode = 0
            self.azr_active_steps = 0
            self.azr_cooldown_steps = self.cooldown_after_azr

        self.episode_steps += 1
        self.total_reward += reward
        self.reward_history.append(reward)

        info.update({
            'map_type': self.map_type,
            'episode_step': self.episode_steps,
            'current_time': self.current_time,
            'current_mode': self.current_mode,
            'requested_mode': requested_action,
            'executed_mode': execution_action,
            'active_primitive_mode': self.active_primitive_mode,
            'position': self.state[:2].copy(),
            'heading_rad': self.state[2],
            'total_reward': self.total_reward,
            'forward_clearance': post_mode_context['forward_clearance'],
            'left_clearance': post_mode_context['left_clearance'],
            'right_clearance': post_mode_context['right_clearance'],
            'suggested_mode': post_mode_context['suggested_mode'],
            'apt_direction_hint': post_mode_context['apt_direction'],
            'apt_distance_hint': post_mode_context['apt_distance'],
            'path_heading_change': post_mode_context['path_heading_change'],
            'path_progress': post_mode_context['path_progress'],
            'current_heading_error': post_mode_context['current_heading_error'],
            'current_lateral_error': post_mode_context['current_lateral_error'],
            'is_apt_candidate': post_mode_context['apt_candidate'],
            'is_azr_candidate': post_mode_context['azr_candidate'],
            'forward_blocked': post_mode_context['forward_blocked'],
            'step_displacement': self.last_step_motion['displacement'],
            'step_longitudinal': self.last_step_motion['longitudinal'],
            'step_lateral': self.last_step_motion['lateral'],
            'step_heading_change': self.last_step_motion['heading_change'],
            'failure_reason': self.last_failure_reason,
            'afm_count': self.mode_counts[0],
            'apt_count': self.mode_counts[1],
            'azr_count': self.mode_counts[2],
            'requested_afm_count': self.requested_mode_counts[0],
            'requested_apt_count': self.requested_mode_counts[1],
            'requested_azr_count': self.requested_mode_counts[2],
            'mode_switch_count': self.mode_switch_count,
            'apt_candidate_steps': self.apt_candidate_steps,
            'azr_candidate_steps': self.azr_candidate_steps,
            'blocked_steps': self.blocked_steps,
            'stuck_steps': self.no_progress_steps,
            'total_actions': sum(self.mode_counts.values()),
            'is_success': self._check_goal_reached(),
            'distance_to_goal': np.linalg.norm(self.state[:2] - self.goal_position),
            'primitive_aborted': primitive_aborted,
            'primitive_active': self.active_primitive_mode != 0
        })

        return observation, reward, terminated, truncated, info
    
    def get_obs(self):
        """
        Generate paper-aligned observation vector.
        
        Returns:
            np.ndarray: Observation vector
        """
        nearest_idx = self._find_nearest_path_index()
        path_features = self._get_local_path_features(nearest_idx)
        ref_point = self.reference_path[nearest_idx]

        lateral_error = self._compute_lateral_error(ref_point) / max(self.e_perp_max, 1e-6)
        heading_error = self._compute_heading_error(
            ref_point[2] if ref_point.shape[0] > 2 else self.state[2]
        ) / np.pi

        vx = self.last_kinematics['vx'] / max(self.max_speed, 1e-6)
        vy = self.last_kinematics['vy'] / max(self.max_speed, 1e-6)
        omega = self.last_kinematics['omega'] / max(self.max_yaw_rate, 1e-6)
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal_position) / max(self.position_scale, 1e-6)
        progress = self._compute_path_progress(nearest_idx)
        time_remaining = max(0.0, (self.max_time - self.current_time) / max(self.max_time, 1e-6))

        body_state = np.array([
            lateral_error,
            heading_error,
            vx,
            vy,
            omega,
            distance_to_goal,
            progress,
            time_remaining
        ], dtype=np.float32)

        mode_context = self._get_mode_context()
        mode_hint = np.zeros(self.mode_hint_len, dtype=np.float32)
        mode_hint[int(np.clip(mode_context['suggested_mode'], 0, self.mode_hint_len - 1))] = 1.0

        action_history = self._encode_action_history()
        clearances = self._compute_local_clearances()
        env_state = np.array([
            clearances['forward'] / max(self.max_clearance_distance, 1e-6),
            clearances['rear'] / max(self.max_clearance_distance, 1e-6),
            clearances['left'] / max(self.max_clearance_distance, 1e-6),
            clearances['right'] / max(self.max_clearance_distance, 1e-6),
            clearances['free_width'] / max(2.0 * self.max_clearance_distance, 1e-6)
        ], dtype=np.float32)

        obs = np.concatenate([path_features, body_state, mode_hint, action_history, env_state])
        return obs.astype(np.float32)
    
    def compute_reward(self, mode_switch_penalty=0.0, action=None):
        """
        Paper-aligned reward function.
        
        r_t = w1 * r_goal + w2 * r_cp + w3 * r_yaw + w4 * r_switch + r_terminal
        
        Args:
            mode_switch_penalty: Binary indicator of mode switch (0 or 1)
        
        Returns:
            float: Total reward value
            dict: Detailed reward breakdown
        """
        w1 = self.reward_weights['w1']
        w2 = self.reward_weights['w2']
        w3 = self.reward_weights['w3']
        w4 = self.reward_weights['w4']

        mode_context = self._get_mode_context()
        nearest_idx = mode_context['nearest_idx']
        ref_point = self.reference_path[nearest_idx]

        lateral_error = abs(self._compute_lateral_error(ref_point))
        heading_error = abs(self._compute_heading_error(ref_point[2] if ref_point.shape[0] > 2 else 0.0))
        current_distance = np.linalg.norm(self.state[:2] - self.goal_position)
        prev_distance = self.prev_distance_to_goal
        if prev_distance is not None:
            r_goal = np.clip(
                (prev_distance - current_distance) /
                max(self.decision_dt * self.max_speed, 1e-6),
                -1.0,
                1.0
            )
        else:
            r_goal = 0.0
        path_progress = mode_context['path_progress']
        r_path = 10.0 * (path_progress - self.prev_path_progress)

        r_cp = -min(lateral_error / max(self.e_perp_max, 1e-6), 1.0)
        r_yaw = -min(heading_error / max(self.e_theta_max, 1e-6), 1.0)
        r_switch = -float(mode_switch_penalty)
        r_time = -0.02

        prev_lateral_error = self.prev_lateral_error
        prev_heading_error = self.prev_heading_error
        delta_lateral_error = 0.0 if prev_lateral_error is None else prev_lateral_error - lateral_error
        delta_heading_error = 0.0 if prev_heading_error is None else prev_heading_error - heading_error

        r_mode_align = 0.0
        executed_mode = self.mode_history[-1] if len(self.mode_history) > 0 else self.active_primitive_mode
        if action is not None and not mode_context['primitive_active']:
            if action == mode_context['suggested_mode']:
                if action == 0:
                    r_mode_align = self.mode_alignment_reward
                else:
                    r_mode_align = self.mode_alignment_reward * 0.75
            else:
                r_mode_align = -self.mode_alignment_penalty

        r_primitive_progress = 0.0
        r_apt_stall = 0.0
        r_azr_stall = 0.0
        if executed_mode == 1:
            lateral_improvement = max(0.0, delta_lateral_error)
            forward_progress = max(0.0, self.last_step_motion['longitudinal'])
            heading_change = abs(self.last_step_motion['heading_change'])
            r_primitive_progress = (
                1.5 * min(lateral_improvement / max(self.decision_dt * 0.22, 1e-6), 1.0) +
                0.9 * min(forward_progress / 0.03, 1.0) -
                0.08 * min(abs(self.last_step_motion['lateral']) / 0.02, 1.0) -
                0.06 * min(heading_change / np.deg2rad(4.0), 1.0)
            )
            if current_distance > self.goal_threshold * 2 and lateral_improvement < 0.001 and forward_progress < 0.005:
                r_apt_stall = -0.22
        elif executed_mode == 2:
            heading_improvement = max(0.0, delta_heading_error)
            displacement = self.last_step_motion['displacement']
            lateral_drift = abs(self.last_step_motion['lateral'])
            r_primitive_progress = (
                2.5 * min(heading_improvement / max(np.deg2rad(10.0), 1e-6), 1.0) -
                0.18 * min(displacement / 0.02, 1.0) -
                0.10 * min(lateral_drift / 0.02, 1.0)
            )
            if current_distance > self.goal_threshold * 2 and heading_improvement < np.deg2rad(0.25):
                r_azr_stall = -0.22
        else:
            forward_progress = max(0.0, self.last_step_motion['longitudinal'])
            r_primitive_progress = 0.35 * min(forward_progress / 0.05, 1.0)

        self.prev_distance_to_goal = current_distance
        self.prev_path_progress = path_progress
        self.prev_lateral_error = lateral_error
        self.prev_heading_error = heading_error

        failure_reason = self._get_failure_reason()
        if failure_reason == 'success':
            r_terminal = self.terminal_rewards['success']
        elif failure_reason in ('collision', 'out_of_bounds'):
            r_terminal = self.terminal_rewards['collision']
        elif failure_reason in ('timeout', 'stuck'):
            r_terminal = self.terminal_rewards['timeout']
        else:
            r_terminal = 0.0

        reward = (
            w1 * r_goal +
            r_path +
            w2 * r_cp +
            w3 * r_yaw +
            w4 * r_switch +
            r_time +
            r_terminal +
            r_mode_align +
            r_primitive_progress +
            r_apt_stall +
            r_azr_stall
        )

        r_terminal_brake = 0.0
        if current_distance < 1.0:
            current_speed = np.hypot(self.last_kinematics['vx'], self.last_kinematics['vy'])
            speed_norm = min(current_speed / max(self.max_speed, 1e-6), 1.0)
            r_terminal_brake = 0.15 * (1.0 - speed_norm)
            reward += r_terminal_brake

        info = {
            'lateral_error': lateral_error,
            'heading_error': heading_error,
            'distance_to_goal': current_distance,
            'r_goal': r_goal,
            'r_path': r_path,
            'r_cp': r_cp,
            'r_yaw': r_yaw,
            'r_switch': r_switch,
            'r_time': r_time,
            'r_terminal': r_terminal,
            'r_mode_align': r_mode_align,
            'r_primitive_progress': r_primitive_progress,
            'r_apt_stall': r_apt_stall,
            'r_azr_stall': r_azr_stall,
            'r_terminal_brake': r_terminal_brake,
            'delta_lateral_error': delta_lateral_error,
            'delta_heading_error': delta_heading_error,
            'primitive_active': mode_context['primitive_active'],
            'active_mode': mode_context['active_mode'],
            'mode_switch_penalty': w4 * mode_switch_penalty,
            'goal_reached': False,
            'collision': False,
            'forward_clearance': mode_context['forward_clearance'],
            'failure_reason': failure_reason
        }

        if failure_reason == 'success':
            info['goal_reached'] = True
            info['goal_reward'] = r_terminal
            print("*** GOAL REACHED! ***")
            print(f"    Time: {self.current_time:.2f}s")
            print(f"    Total reward: {self.total_reward + reward:.2f}")
        if failure_reason == 'collision':
            info['collision'] = True
            print(f"*** COLLISION! Position: ({self.state[0]:.2f}, {self.state[1]:.2f}) ***")
        if failure_reason == 'out_of_bounds':
            info['out_of_bounds'] = True
        else:
            info['out_of_bounds'] = False
        info['stuck'] = failure_reason == 'stuck'

        return reward, info
    
    def check_done(self):
        """Check termination conditions"""
        if self._check_goal_reached():
            return True
        
        if self.current_time >= self.max_time:
            return True
        
        if self._check_collision():
            return True
        
        if self._out_of_bounds():
            return True

        if self.no_progress_steps >= self.no_progress_limit:
            return True
        
        return False
    
    def save_episode_stats(self, episode_num, success):
        """
        Save episode statistics to CSV file (Task 4)
        
        Args:
            episode_num: Episode number
            success: Whether episode was successful
        """
        csv_path = os.path.join(self.log_dir, 'action_stats.csv')
        
        # Calculate mode percentages
        total_actions = sum(self.mode_counts.values()) or 1
        afm_pct = 100 * self.mode_counts[0] / total_actions
        apt_pct = 100 * self.mode_counts[1] / total_actions
        azr_pct = 100 * self.mode_counts[2] / total_actions
        
        # Write header if file doesn't exist
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow([
                    'episode', 'map_type', 'success', 'total_reward', 'steps', 
                    'time(s)', 'final_distance',
                    'afm_count', 'apt_count', 'azr_count',
                    'afm_pct', 'apt_pct', 'azr_pct',
                    'mode_switches', 'failure_reason',
                    'apt_candidate_steps', 'azr_candidate_steps', 'blocked_steps'
                ])
            
            writer.writerow([
                episode_num,
                self.map_type,
                success,
                round(self.total_reward, 2),
                self.episode_steps,
                round(self.current_time, 2),
                round(np.linalg.norm(self.state[:2] - self.goal_position), 3),
                self.mode_counts[0],
                self.mode_counts[1],
                self.mode_counts[2],
                round(afm_pct, 1),
                round(apt_pct, 1),
                round(azr_pct, 1),
                self.mode_switch_count,
                self.last_failure_reason,
                self.apt_candidate_steps,
                self.azr_candidate_steps,
                self.blocked_steps
            ])
    
    def render(self, mode='human'):
        """Optional visualization placeholder"""
        pass
    
    def close(self):
        """Clean up resources"""
        pass
    
    def _update_state_afm(self, u):
        """Update vehicle state using the bicycle model for AFM."""
        x, y, psi, v = self.state
        a, delta = u

        L = 1.0

        x_new = x + v * np.cos(psi) * self.dt
        y_new = y + v * np.sin(psi) * self.dt
        psi_new = psi + (v / L) * np.tan(delta) * self.dt
        v_new = np.clip(v + a * self.dt, 0.0, self.max_speed)
        omega = (v / L) * np.tan(delta)

        psi_new = np.arctan2(np.sin(psi_new), np.cos(psi_new))

        self.state = np.array([x_new, y_new, psi_new, v_new])
        self.last_kinematics = {
            'vx': float(v_new),
            'vy': 0.0,
            'omega': float(omega)
        }

    def _update_action_history(self, action):
        """Append the latest high-level action to the fixed-length history."""
        self.mode_action_history.append(int(action))
        if len(self.mode_action_history) > self.action_history_len:
            self.mode_action_history = self.mode_action_history[-self.action_history_len:]

    def _encode_action_history(self):
        """Encode the recent action history as flattened one-hot vectors."""
        history = np.zeros(self.action_history_len * 3, dtype=np.float32)
        for i, action in enumerate(self.mode_action_history[-self.action_history_len:]):
            action = int(np.clip(action, 0, 2))
            history[i * 3 + action] = 1.0
        return history

    def _get_local_path_features(self, nearest_idx):
        """Encode a short local path preview in the robot frame."""
        features = []
        last_idx = len(self.reference_path) - 1
        for offset in self.path_lookahead_offsets:
            idx = min(nearest_idx + offset, last_idx)
            point = self.reference_path[idx]
            local_xy = self._project_point_to_body_frame(point[:2])
            if point.shape[0] > 2:
                heading_error = self._normalize_angle(point[2] - self.state[2])
            else:
                heading_error = 0.0
            features.extend([
                local_xy[0] / max(self.position_scale, 1e-6),
                local_xy[1] / max(self.position_scale, 1e-6),
                heading_error / np.pi
            ])
        return np.array(features, dtype=np.float32)

    def _update_state_holonomic(self, command):
        """Update vehicle state with body-frame [vx, vy, omega] commands."""
        x, y, psi, _ = self.state
        vx, vy, omega = command

        x_new = x + (vx * np.cos(psi) - vy * np.sin(psi)) * self.dt
        y_new = y + (vx * np.sin(psi) + vy * np.cos(psi)) * self.dt
        psi_new = np.arctan2(np.sin(psi + omega * self.dt), np.cos(psi + omega * self.dt))
        speed = np.hypot(vx, vy)

        self.state = np.array([x_new, y_new, psi_new, speed])
        self.last_kinematics = {
            'vx': float(vx),
            'vy': float(vy),
            'omega': float(omega)
        }

    def _configure_mode_transition(self, action, mode_context=None):
        """Configure internal controller targets when switching modes."""
        if mode_context is None:
            mode_context = self._get_mode_context()

        if action == 1:
            self.apt_controller.set_translation_target(
                self.state,
                direction=mode_context['apt_direction'],
                distance=mode_context['apt_distance'],
                heading_hold=mode_context.get('target_heading', self.state[2])
            )
        elif action == 2:
            self.azr_controller.set_rotation_target(
                current_heading=self.state[2],
                target_heading=mode_context['target_heading']
            )

    def _compute_step_motion(self, start_state, end_state):
        """Measure the macro-step motion in the start-state local frame."""
        delta_pos = end_state[:2] - start_state[:2]
        heading = start_state[2]
        longitudinal = np.cos(heading) * delta_pos[0] + np.sin(heading) * delta_pos[1]
        lateral = -np.sin(heading) * delta_pos[0] + np.cos(heading) * delta_pos[1]
        heading_change = np.arctan2(
            np.sin(end_state[2] - start_state[2]),
            np.cos(end_state[2] - start_state[2])
        )
        return {
            'displacement': float(np.linalg.norm(delta_pos)),
            'longitudinal': float(longitudinal),
            'lateral': float(lateral),
            'heading_change': float(heading_change)
        }
    
    def _find_nearest_path_index(self):
        """Find index of nearest point on reference path"""
        distances = np.linalg.norm(self.reference_path[:, :2] - self.state[:2], axis=1)
        return int(np.argmin(distances))
    
    def _compute_lateral_error(self, ref_point):
        """Compute lateral (cross-track) error"""
        error_vec = ref_point[:2] - self.state[:2]
        
        if ref_point.shape[0] > 2:
            ref_heading = ref_point[2]
        else:
            idx = self._find_nearest_path_index()
            if idx < len(self.reference_path) - 1:
                ref_heading = np.arctan2(
                    self.reference_path[idx+1, 1] - self.reference_path[idx, 1],
                    self.reference_path[idx+1, 0] - self.reference_path[idx, 0]
                )
            else:
                ref_heading = self.state[2]
        
        return (
            error_vec[0] * (-np.sin(ref_heading)) +
            error_vec[1] * np.cos(ref_heading)
        )
    
    def _compute_heading_error(self, ref_heading):
        """Compute heading error normalized to [-pi, pi]"""
        error = ref_heading - self.state[2]
        return np.arctan2(np.sin(error), np.cos(error))

    def _compute_obstacle_distance(self):
        """Compute forward clearance to the nearest obstacle/boundary."""
        return self._compute_local_clearances()['forward']
    
    def _check_goal_reached(self):
        """Check if vehicle reached goal within threshold"""
        distance = np.linalg.norm(self.state[:2] - self.goal_position)
        heading_error = abs(np.arctan2(
            np.sin(self.goal_heading - self.state[2]),
            np.cos(self.goal_heading - self.state[2])
        ))
        
        return distance < self.goal_threshold and heading_error < np.deg2rad(45)
    
    def _check_collision(self):
        """Check collision with obstacles/boundaries"""
        x, y = self.state[:2]
        
        if hasattr(self.env_map, 'x_range') and hasattr(self.env_map, 'y_range'):
            x_min, x_max = self.env_map.x_range
            y_min, y_max = self.env_map.y_range
            
            if x < x_min or x > x_max or y < y_min or y > y_max:
                return True
        
        if not self._is_drivable_point(x, y):
            return True
        
        return False
    
    def _out_of_bounds(self):
        """Check if vehicle is out of safe operational bounds"""
        x, y = self.state[:2]
        
        if hasattr(self.env_map, 'x_range') and hasattr(self.env_map, 'y_range'):
            x_min, x_max = self.env_map.x_range
            y_min, y_max = self.env_map.y_range
            
            margin = 1.0
            return (x < x_min + margin or x > x_max - margin or
                    y < y_min + margin or y > y_max - margin)
        
        return False

    def _get_mode_context(self):
        """Summarize local geometry and upcoming path demand for mode decisions."""
        nearest_idx = self._find_nearest_path_index()
        ref_point = self.reference_path[nearest_idx]
        clearances = self._compute_local_clearances()
        path_progress = self._compute_path_progress(nearest_idx)
        path_heading_change = self._compute_path_heading_change(nearest_idx)
        local_goal = self._project_point_to_body_frame(self._get_lookahead_point(nearest_idx, lookahead=20))
        current_lateral_error = self._compute_lateral_error(ref_point)
        current_heading_error = self._compute_heading_error(
            ref_point[2] if ref_point.shape[0] > 2 else self.state[2]
        )
        forward_blocked = clearances['forward'] < 1.2
        lateral_need = local_goal[1]
        lateral_room = max(clearances['left'], clearances['right'])
        curvature_demand = max(abs(path_heading_change), abs(current_heading_error))
        heading_aligned_for_translation = abs(current_heading_error) < np.deg2rad(40)
        apt_candidate = (
            heading_aligned_for_translation and lateral_room > 0.75 and (
                (
                    path_progress > 0.15 and
                    abs(current_lateral_error) > 0.18 and
                    curvature_demand <= np.deg2rad(55)
                ) or
                (
                    curvature_demand > np.deg2rad(12) and
                    curvature_demand <= np.deg2rad(55)
                ) or
                (
                    forward_blocked and
                    abs(lateral_need) > 0.15
                )
            )
        )
        azr_candidate = (
            abs(current_heading_error) > np.deg2rad(70) or
            abs(path_heading_change) > np.deg2rad(60) or
            (
                abs(path_heading_change) > np.deg2rad(40) and
                path_progress > 0.35 and
                forward_blocked and
                abs(current_heading_error) > np.deg2rad(40)
            )
        )

        if abs(lateral_need) > 0.2:
            apt_direction = 'left' if lateral_need > 0 else 'right'
        else:
            apt_direction = 'left' if clearances['left'] >= clearances['right'] else 'right'

        suggested_distance = np.clip(max(abs(current_lateral_error), abs(lateral_need), 0.18), 0.18, 1.10)
        future_idx = min(nearest_idx + 25, len(self.reference_path) - 1)
        future_heading = self.reference_path[future_idx][2] if self.reference_path.shape[1] > 2 else self.state[2]
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal_position)
        near_goal = distance_to_goal < 1.1
        target_heading = (
            self.goal_heading
            if near_goal
            else (future_heading if abs(path_heading_change) > np.deg2rad(35) else ref_point[2])
        )

        if azr_candidate and (
            abs(path_heading_change) > np.deg2rad(50) or
            abs(current_heading_error) > np.deg2rad(50) or
            (near_goal and abs(current_heading_error) > np.deg2rad(18))
        ):
            suggested_mode = 2
        elif apt_candidate:
            suggested_mode = 1
        else:
            suggested_mode = 0

        near_goal_requires_realign = (
            self.map_type != 'map_a' and
            near_goal and
            abs(current_heading_error) > np.deg2rad(18)
        )

        if near_goal_requires_realign:
            azr_candidate = True
            suggested_mode = 2

        if self.apt_cooldown_steps > 0:
            apt_candidate = False
            if suggested_mode == 1:
                suggested_mode = 0
        if self.azr_cooldown_steps > 0:
            azr_candidate = False
            if suggested_mode == 2:
                suggested_mode = 0

        active_mode = self.active_primitive_mode
        primitive_active = False
        if active_mode == 1:
            primitive_active = bool(self.apt_controller.active and not self.apt_controller.is_complete(self.state))
        elif active_mode == 2:
            primitive_active = bool(self.azr_controller.is_rotating and not self.azr_controller.is_complete(self.state))

        return {
            'nearest_idx': nearest_idx,
            'path_progress': path_progress,
            'path_heading_change': path_heading_change,
            'curvature_demand': curvature_demand,
            'current_heading_error': current_heading_error,
            'current_lateral_error': current_lateral_error,
            'distance_to_goal': distance_to_goal,
            'forward_clearance': clearances['forward'],
            'left_clearance': clearances['left'],
            'right_clearance': clearances['right'],
            'forward_blocked': forward_blocked,
            'apt_candidate': apt_candidate,
            'azr_candidate': azr_candidate,
            'apt_direction': apt_direction,
            'apt_distance': float(suggested_distance),
            'target_heading': target_heading,
            'suggested_mode': suggested_mode,
            'active_mode': active_mode,
            'primitive_active': primitive_active
        }

    def _compute_local_clearances(self, max_distance=3.0, step=0.05):
        """Ray-cast clearances in the forward/left/right directions."""
        psi = self.state[2]
        rear_angle = psi + np.pi
        forward = self._raycast_clearance(psi, max_distance=max_distance, step=step)
        rear = self._raycast_clearance(rear_angle, max_distance=max_distance, step=step)
        left = self._raycast_clearance(psi + np.pi / 2, max_distance=max_distance, step=step)
        right = self._raycast_clearance(psi - np.pi / 2, max_distance=max_distance, step=step)
        return {
            'forward': forward,
            'rear': rear,
            'left': left,
            'right': right,
            'free_width': left + right
        }

    def _raycast_clearance(self, angle, max_distance=3.0, step=0.05):
        """Approximate free space along a ray until a boundary or obstacle is hit."""
        x0, y0 = self.state[:2]
        distance = 0.0
        while distance <= max_distance:
            probe_x = x0 + distance * np.cos(angle)
            probe_y = y0 + distance * np.sin(angle)
            if not self._is_drivable_point(probe_x, probe_y):
                return max(0.0, distance - step)
            distance += step
        return max_distance

    def _is_drivable_point(self, x, y):
        """Check whether a point lies inside any drivable region."""
        if hasattr(self.env_map, 'drivable_areas'):
            for area_def in self.env_map.drivable_areas.values():
                if area_def['type'] != 'rectangle':
                    continue
                if (
                    area_def['x_min'] <= x <= area_def['x_max'] and
                    area_def['y_min'] <= y <= area_def['y_max']
                ):
                    return True
            return False

        if hasattr(self.env_map, 'x_range') and hasattr(self.env_map, 'y_range'):
            x_min, x_max = self.env_map.x_range
            y_min, y_max = self.env_map.y_range
            return x_min <= x <= x_max and y_min <= y <= y_max

        return True

    def _get_lookahead_point(self, nearest_idx, lookahead=25):
        """Get a path point ahead of the current position."""
        target_idx = min(nearest_idx + lookahead, len(self.reference_path) - 1)
        return self.reference_path[target_idx][:2]

    def _project_point_to_body_frame(self, point):
        """Project a world-frame point into the current body frame."""
        dx = point[0] - self.state[0]
        dy = point[1] - self.state[1]
        psi = self.state[2]
        local_x = np.cos(psi) * dx + np.sin(psi) * dy
        local_y = -np.sin(psi) * dx + np.cos(psi) * dy
        return np.array([local_x, local_y])

    def _compute_path_heading_change(self, nearest_idx, lookahead=25):
        """Estimate the upcoming change in reference heading."""
        current_heading = self.reference_path[nearest_idx][2] if self.reference_path.shape[1] > 2 else self.state[2]
        future_idx = min(nearest_idx + lookahead, len(self.reference_path) - 1)
        future_heading = self.reference_path[future_idx][2] if self.reference_path.shape[1] > 2 else current_heading
        return self._normalize_angle(future_heading - current_heading)

    def _get_failure_reason(self):
        """Classify the current terminal condition for analysis."""
        if self._check_goal_reached():
            return 'success'
        if self._check_collision():
            return 'collision'
        if self._out_of_bounds():
            return 'out_of_bounds'
        if self.no_progress_steps >= self.no_progress_limit:
            return 'stuck'
        if self.current_time >= self.max_time:
            return 'timeout'
        return 'running'

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))


if __name__ == "__main__":
    # Test the enhanced environment
    env = ModeEnv(map_type=('map_a', 'map_b', 'map_c'), randomize=True)
    
    print("\n=== Testing Enhanced ModeEnv ===")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Test multiple episodes with randomization
    print("\n=== Running test episodes with randomization ===")
    for ep in range(3):
        obs, info = env.reset()
        print(f"\nEpisode {ep+1}:")
        print(f"  Initial obs dim: {obs.shape[0]}")
        print(f"  Initial obs sample: [{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}, {obs[3]:.3f}, ...]")
        
        done = False
        step_count = 0
        while not done and step_count < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            if step_count % 25 == 0:
                print(f"  Step {step_count}: mode={info['current_mode']}, "
                      f"reward={reward:.2f}, dist={info['distance_to_goal']:.2f}m")
        
        # Save episode stats
        env.save_episode_stats(ep+1, info['is_success'])
        
        print(f"  Finished: steps={step_count}, success={info['is_success']}")
        print(f"  Mode usage: AFM={info['afm_count']}, APT={info['apt_count']}, AZR={info['azr_count']}")
        print(f"  Switches: {info['mode_switch_count']}")
    
    env.close()
    print(f"\n=== Stats saved to {env.log_dir}/action_stats.csv ===")
