"""
Robust CasADi-based NMPC controller
For path tracking
"""
import numpy as np
import casadi as ca
from typing import Tuple, List, Optional


class CasADiNMPCRobust:
    """
    Robust CasADi-based NMPC controller class
    Uses nonlinear model for path tracking with fallback control strategy
    """
    
    def __init__(self, dt: float = 0.1, horizon: int = 5, verbose: bool = False):
        """
        Initialize NMPC controller
        
        Args:
            dt: Time step (s)
            horizon: Prediction horizon
        """
        self.dt = dt
        self.N = horizon
        self.verbose = verbose
        
        # Vehicle parameters
        # In the paper-aligned AFM model, the virtual-wheel equations use the
        # centroid-to-axle distance L in the denominator 2L. We keep the same
        # symbol here for consistency with the manuscript.
        self.L = 1.0
        self.min_turning_radius = 1.2  # Minimum turning radius (m)
        
        # Control constraints
        # AFM is treated as a forward-tracking mode in the paper-aligned
        # implementation, so we keep the longitudinal speed non-negative to
        # avoid reverse-driving local minima on straight corridors.
        self.v_min = 0.0   # Minimum speed (m/s)
        self.v_max = 8.0  # Maximum speed (m/s)
        self.a_min = -3.0  # Minimum acceleration (m/s²)
        self.a_max = 2.0  # Maximum acceleration (m/s²)
        # The paper's virtual-wheel geometry uses a 2L wheelbase term, so the
        # equivalent steering bound is derived from the full wheelbase.
        self.delta_max = np.arctan((2.0 * self.L) / self.min_turning_radius)  # Maximum steering angle (rad)
        
        # Weights
        self.Q = np.diag([10, 10, 5, 10])  # State weights, keep pose tracking dominant
        # Control weights for [a, delta_F, delta_R]
        self.R = np.diag([0.5, 2.2, 2.2])
        self.steering_balance_weight = 0.2  # Encourage AFM-like opposite steering
        # Soft robustness constraints: smooth control updates, suppress
        # lateral crab motion, and damp heading chatter on straight exits.
        self.control_rate_weight = 0.75
        self.lateral_velocity_weight = 0.25
        self.yaw_rate_weight = 0.12
        self.afm_tracking_bias = 0
        self.afm_curvature_lookahead = 20
        self.afm_search_back_window = 4
        self.afm_search_forward_window = 24
        self.afm_straight_curvature_threshold = 0.035
        self.afm_moderate_curvature_threshold = 0.12
        # Stage-cost weights in the path frame. These are deliberately
        # asymmetric so small cross-track deviations do not trigger overly
        # aggressive micro-corrections.
        self.stage_longitudinal_weight = 1.5
        self.stage_lateral_weight = 11.0
        self.stage_heading_weight = 6.0
        self.stage_speed_weight = 4.0
        self.path_lateral_deadband = 0.05
        self.path_heading_deadband = np.deg2rad(2.0)
        self.reference_speed_cap = 1.2
        self.reference_speed_floor = 0.6
        self.terminal_slowdown_distance = 0.15
        
        # Initialize CasADi optimization problem
        self._initialize_optimization()

    def _paper_body_velocity_np(self, v: float, delta_f: float, delta_r: float):
        """
        Paper-aligned virtual-wheel body-frame velocity mapping.

        Returns:
            vx_body, vy_body, omega
        """
        vx_body = v * np.cos(delta_f)
        vy_body = v * (np.sin(delta_f) + np.cos(delta_f) * np.tan(delta_r)) / 2.0
        omega = v * (np.sin(delta_f) - np.cos(delta_f) * np.tan(delta_r)) / (2.0 * self.L)
        return vx_body, vy_body, omega

    def configure_from_map(self, env_map):
        """
        Apply optional map-specific AFM robustness settings.

        The map classes can expose these values to bias AFM toward slightly
        different robustness trade-offs without changing the controller API.
        """
        if env_map is None:
            return

        self.control_rate_weight = float(getattr(env_map, 'afm_control_rate_weight', self.control_rate_weight))
        self.lateral_velocity_weight = float(getattr(env_map, 'afm_lateral_velocity_weight', self.lateral_velocity_weight))
        self.yaw_rate_weight = float(getattr(env_map, 'afm_yaw_rate_weight', self.yaw_rate_weight))
        self.afm_tracking_bias = int(getattr(env_map, 'afm_tracking_bias', self.afm_tracking_bias))
        self.afm_curvature_lookahead = int(getattr(env_map, 'afm_curvature_lookahead', self.afm_curvature_lookahead))
        self.afm_search_back_window = int(getattr(env_map, 'afm_search_back_window', self.afm_search_back_window))
        self.afm_search_forward_window = int(getattr(env_map, 'afm_search_forward_window', self.afm_search_forward_window))
        self.afm_straight_curvature_threshold = float(getattr(env_map, 'afm_straight_curvature_threshold', self.afm_straight_curvature_threshold))
        self.afm_moderate_curvature_threshold = float(getattr(env_map, 'afm_moderate_curvature_threshold', self.afm_moderate_curvature_threshold))
        self.stage_longitudinal_weight = float(getattr(env_map, 'afm_stage_longitudinal_weight', self.stage_longitudinal_weight))
        self.stage_lateral_weight = float(getattr(env_map, 'afm_stage_lateral_weight', self.stage_lateral_weight))
        self.stage_heading_weight = float(getattr(env_map, 'afm_stage_heading_weight', self.stage_heading_weight))
        self.stage_speed_weight = float(getattr(env_map, 'afm_stage_speed_weight', self.stage_speed_weight))
        self.path_lateral_deadband = float(getattr(env_map, 'afm_path_lateral_deadband', self.path_lateral_deadband))
        self.path_heading_deadband = float(getattr(env_map, 'afm_path_heading_deadband', self.path_heading_deadband))
        self.reference_speed_cap = float(getattr(env_map, 'afm_reference_speed_cap', self.reference_speed_cap))
        self.reference_speed_floor = float(getattr(env_map, 'afm_reference_speed_floor', self.reference_speed_floor))
        self.terminal_slowdown_distance = float(getattr(env_map, 'afm_terminal_slowdown_distance', self.terminal_slowdown_distance))

    @staticmethod
    def _soft_deadband_symbolic(value, deadband):
        """Smoothly suppress tiny errors while preserving large deviations."""
        if deadband <= 0.0:
            return value
        return ca.sqrt(value * value + deadband * deadband) - deadband
    
    def _initialize_optimization(self):
        """
        Initialize CasADi optimization problem
        """
        # State variables: [x, y, psi, v]
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        psi = ca.MX.sym('psi')
        v = ca.MX.sym('v')
        states = ca.vertcat(x, y, psi, v)
        nx = states.size()[0]
        
        # Control variables: [a, delta_F, delta_R]
        a = ca.MX.sym('a')
        delta_f = ca.MX.sym('delta_f')
        delta_r = ca.MX.sym('delta_r')
        controls = ca.vertcat(a, delta_f, delta_r)
        nu = controls.size()[0]
        
        # State update function (paper-aligned virtual-wheel model)
        vx_body = v * ca.cos(delta_f)
        vy_body = v * (ca.sin(delta_f) + ca.cos(delta_f) * ca.tan(delta_r)) / 2.0
        omega = v * (ca.sin(delta_f) - ca.cos(delta_f) * ca.tan(delta_r)) / (2.0 * self.L)
        f = ca.Function('f', [states, controls], [ca.vertcat(
            vx_body * ca.cos(psi) - vy_body * ca.sin(psi),
            vx_body * ca.sin(psi) + vy_body * ca.cos(psi),
            omega,
            a
        )])
        
        # States and controls over prediction horizon
        X = ca.MX.sym('X', nx, self.N+1)
        U = ca.MX.sym('U', nu, self.N)
        P = ca.MX.sym('P', nx + nx * (self.N+1) + nx + nu)  # Initial state + reference trajectory + goal state + previous control
        
        # Cost function
        cost = 0
        g = []  # Constraints
        
        # Initial state constraint
        g.append(X[:, 0] - P[:nx])
        
        # Extract goal state from parameters
        goal_state = P[nx + nx*(self.N+1): nx + nx*(self.N+1) + nx]
        prev_control = P[nx + nx*(self.N+1) + nx:]
        U_prev = prev_control
        
        # Traverse prediction horizon
        for k in range(self.N):
            # State update
            Xk = X[:, k]
            Uk = U[:, k]
            Xk1 = X[:, k+1]
            
            # Reference state
            ref = P[nx + k*nx : nx + (k+1)*nx]
            
            # Track the path in the local path frame so tiny deviations do not
            # cause excessive corrective steering. This soft deadband is the
            # main damping mechanism for AFM drift/jitter.
            dx = ref[0] - Xk[0]
            dy = ref[1] - Xk[1]
            path_heading = ref[2]
            longitudinal_error = dx * ca.cos(path_heading) + dy * ca.sin(path_heading)
            lateral_error = -dx * ca.sin(path_heading) + dy * ca.cos(path_heading)
            heading_error = ca.atan2(ca.sin(ref[2] - Xk[2]), ca.cos(ref[2] - Xk[2]))
            speed_error = ref[3] - Xk[3]

            lateral_soft = self._soft_deadband_symbolic(lateral_error, self.path_lateral_deadband)
            heading_soft = self._soft_deadband_symbolic(heading_error, self.path_heading_deadband)

            cost += self.stage_longitudinal_weight * longitudinal_error**2
            cost += self.stage_lateral_weight * lateral_soft**2
            cost += self.stage_heading_weight * heading_soft**2
            cost += self.stage_speed_weight * speed_error**2
            
            # Control cost
            cost += ca.mtimes(ca.mtimes(Uk.T, self.R), Uk)

            # Smooth control transitions to avoid steering chatter and
            # acceleration spikes when AFM is tracking near-straight segments
            # or exiting a reorientation pocket.
            if k == 0:
                control_delta = Uk - prev_control
            else:
                control_delta = Uk - U_prev
            cost += self.control_rate_weight * ca.dot(control_delta, control_delta)
            U_prev = Uk

            # Keep AFM close to a forward-turning motion pattern rather than
            # drifting into lateral-crab behavior. When delta_F + delta_R ≈ 0,
            # the virtual wheels produce a more conventional turning motion.
            steering_balance = Uk[1] + Uk[2]
            cost += self.steering_balance_weight * steering_balance**2

            # Lateral body velocity and yaw-rate regularization. These are the
            # main robustness constraints that help remove the small lagging
            # and jittering behaviors we observed on map A / map C.
            vy_body_k = Xk[3] * (ca.sin(Uk[1]) + ca.cos(Uk[1]) * ca.tan(Uk[2])) / 2.0
            omega_k = Xk[3] * (ca.sin(Uk[1]) - ca.cos(Uk[1]) * ca.tan(Uk[2])) / (2.0 * self.L)
            cost += self.lateral_velocity_weight * vy_body_k**2
            cost += self.yaw_rate_weight * omega_k**2
            
            # State update constraint (using Euler integration)
            Xk1_pred = Xk + self.dt * f(Xk, Uk)
            g.append(Xk1 - Xk1_pred)
        
        # Terminal cost
        goal_error_final = ca.vertcat(
            goal_state[0] - X[0, self.N],
            goal_state[1] - X[1, self.N],
            ca.atan2(ca.sin(goal_state[2] - X[2, self.N]), ca.cos(goal_state[2] - X[2, self.N])),
            goal_state[3] - X[3, self.N]
        )
        cost += ca.mtimes(ca.mtimes(goal_error_final.T, self.Q), goal_error_final)
        
        # Optimization variables
        opt_vars = ca.vertcat(
            ca.reshape(X, nx*(self.N+1), 1),
            ca.reshape(U, nu*self.N, 1)
        )
        
        # Constraints
        constraints = ca.vertcat(*g)
        
        # Create optimization problem
        nlp = {'x': opt_vars, 'f': cost, 'g': constraints, 'p': P}
        
        # Solver options
        solver_options = {
            'ipopt': {
                'max_iter': 200,
                'print_level': 0,
                'acceptable_tol': 1e-2,
                'acceptable_obj_change_tol': 1e-2,
                'tol': 1e-2,
                'constr_viol_tol': 1e-2
            },
            'print_time': False
        }
        
        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, solver_options)
        
        # Store variable dimensions
        self.nx = nx
        self.nu = nu
        
        # Calculate constraint count
        # Each entry in g is an nx-sized vector constraint, so we need the
        # flattened constraint dimension instead of the Python list length.
        self.n_constraints = int(constraints.size1())
    
    def wrap_angle(self, ang):
        """
        Normalize angle to [-π, π]
        """
        return np.arctan2(np.sin(ang), np.cos(ang))
    
    def nearest_index(self, state, path):
        """
        Find nearest point on path to current state
        """
        d = np.linalg.norm(path[:, :2] - state[:2], axis=1)
        nearest_idx = int(np.argmin(d))
        
        return nearest_idx
    
    def build_reference(self, path):
        """
        Build reference trajectory
        """
        ref = []
        if len(path) > 1:
            segment_lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
            remaining_distances = np.zeros(len(path), dtype=float)
            remaining_distances[:-1] = np.flip(np.cumsum(np.flip(segment_lengths)))
        else:
            remaining_distances = np.zeros(len(path), dtype=float)

        for i in range(len(path)):
            if path.shape[1] == 3:
                # Reference path contains heading information
                x, y, psi = path[i]
            else:
                # Reference path doesn't contain heading information, need to calculate
                x, y = path[i]
                if i < len(path) - 1:
                    dx = path[i + 1, 0] - path[i, 0]
                    dy = path[i + 1, 1] - path[i, 1]
                else:
                    dx = path[i, 0] - path[i - 1, 0]
                    dy = path[i, 1] - path[i - 1, 1]
                psi = np.arctan2(dy, dx)

            # Conservative speed planning:
            # - slow down before sharp turns
            # - taper to zero near the final point so the vehicle stops cleanly
            remaining_distance = float(remaining_distances[i])
            if path.shape[1] == 3 and len(path) > 2:
                prev_idx = max(i - 1, 0)
                next_idx = min(i + 1, len(path) - 1)
                while prev_idx > 0 and np.linalg.norm(path[prev_idx, :2] - path[prev_idx - 1, :2]) < 1e-6:
                    prev_idx -= 1
                while next_idx < len(path) - 1 and np.linalg.norm(path[next_idx + 1, :2] - path[next_idx, :2]) < 1e-6:
                    next_idx += 1
                heading_change = abs(
                    self.wrap_angle(path[next_idx, 2] - path[prev_idx, 2])
                )
                corner_scale = 1.0 / (1.0 + 1.5 * heading_change / (np.pi / 2))
            else:
                corner_scale = 1.0

            speed_target = min(self.reference_speed_cap, 0.35 * remaining_distance + 0.55)
            vref = max(self.reference_speed_floor, speed_target) * corner_scale
            if remaining_distance < self.terminal_slowdown_distance:
                vref = 0.0
            ref.append([x, y, psi, vref])
        
        return np.array(ref)
    
    def solve_nmpc(self, state, ref_traj, goal=None, curvature=0, corner_detected=False, corner_distance=0, prev_control=None):
        """
        Solve NMPC optimization problem
        """
        # Initial state
        x0 = state
        
        # Reference trajectory
        ref = ref_traj.flatten()
        
        # Build parameters
        if goal is not None:
            # Use provided goal
            goal_state = goal
        else:
            # Use last point of reference trajectory as goal
            goal_state = ref_traj[-1]

        if prev_control is None:
            prev_control = np.zeros(self.nu, dtype=float)
        prev_control = np.asarray(prev_control, dtype=float).reshape(-1)
        if prev_control.size != self.nu:
            raise ValueError(f"prev_control must have size {self.nu}, got {prev_control.shape}")

        params = np.concatenate([x0, ref, goal_state, prev_control])
        
        # Initial guess
        x_guess = np.zeros(self.nx * (self.N + 1))
        u_guess = np.zeros(self.nu * self.N)
        
        # Simple initial guess
        for k in range(self.N + 1):
            x_guess[k*self.nx : (k+1)*self.nx] = ref_traj[k]
            if k < self.N:
                u_idx = k * self.nu
                heading_step = self.wrap_angle(ref_traj[min(k + 1, self.N), 2] - ref_traj[k, 2])
                steer_seed = np.clip(0.5 * heading_step, -self.delta_max / 2.0, self.delta_max / 2.0)
                if k == 0:
                    u_guess[u_idx:u_idx + self.nu] = prev_control
                else:
                    u_guess[u_idx] = 0.0
                    u_guess[u_idx + 1] = steer_seed
                    u_guess[u_idx + 2] = -steer_seed
        
        opt_guess = np.concatenate([x_guess, u_guess])
        
        # Constraint lower and upper bounds
        lbg = np.zeros(self.n_constraints)
        ubg = np.zeros(self.n_constraints)
        
        # Variable lower and upper bounds
        lbx = -np.inf * np.ones(self.nx*(self.N+1) + self.nu*self.N)
        ubx = np.inf * np.ones(self.nx*(self.N+1) + self.nu*self.N)
        
        # Control constraints
        for k in range(self.N):
            u_idx = self.nx*(self.N+1) + k*self.nu
            lbx[u_idx] = self.a_min
            ubx[u_idx] = self.a_max
            lbx[u_idx+1] = -self.delta_max
            ubx[u_idx+1] = self.delta_max
            lbx[u_idx+2] = -self.delta_max
            ubx[u_idx+2] = self.delta_max
        
        # Speed constraints
        for k in range(self.N+1):
            lbx[k*self.nx + 3] = self.v_min
            ubx[k*self.nx + 3] = self.v_max
        
        # Solve
        try:
            sol = self.solver(x0=opt_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=params)
            
            # Extract control input
            u_opt = sol['x'][self.nx*(self.N+1):self.nx*(self.N+1)+self.nu]
            return np.array(u_opt).flatten(), True
        except Exception as e:
            # Solve failed, use fallback control
            return self._fallback_control(state, ref_traj[0], curvature, corner_detected, corner_distance), False
    
    def _fallback_control(self, state, ref_state, curvature=0, corner_detected=False, corner_distance=0, turn_radius=2.5):
        """
        Fallback control strategy
        """
        # Calculate position error
        pos_error = ref_state[:2] - state[:2]
        distance = np.linalg.norm(pos_error)
        
        # Calculate angle error
        desired_heading = ref_state[2]
        heading_error = self.wrap_angle(desired_heading - state[2])
        
        # Calculate position error direction
        if distance > 0:
            pos_error_direction = np.arctan2(pos_error[1], pos_error[0])
            # Calculate difference between position error direction and current heading
            pos_heading_error = self.wrap_angle(pos_error_direction - state[2])
        else:
            pos_heading_error = 0
        
        # Use the reference speed profile as the baseline so the fallback
        # controller also slows down near the end of the path.
        baseline_speed = float(ref_state[3]) if len(ref_state) > 3 else 0.8

        # Adjust control parameters based on distance and curvature
        if distance > 1.0:
            # Far from reference path, prioritize position error direction
            kp_pos = 1.0
            kp_heading = 2.0
            kp_pos_heading = 3.0  # Reduce position direction gain for smoother steering
            desired_speed = min(1.5, max(baseline_speed, distance * kp_pos))
        else:
            # Close to reference path, prioritize heading error for stability
            kp_pos = 0.4
            kp_heading = 1.4  # Reduce heading gain for smoother straight line driving
            kp_pos_heading = 0.25  # Reduce position direction gain for smoother steering
            desired_speed = min(
                baseline_speed,
                0.7 if state[3] < 0.1 else min(0.7, distance * kp_pos + 0.3)
            )

        if distance < 0.25:
            desired_speed = 0.0
        elif distance < 0.9:
            desired_speed = min(desired_speed, max(0.0, distance * 0.45))
        
        # Adjust steering in advance when approaching a corner
        if curvature > 0.5:  # Curvature threshold, can be adjusted based on actual situation
            # Calculate desired steering direction
            # Simplified processing, assuming reference path heading already considers the corner
            # Increase heading error weight to make vehicle turn in advance
            kp_heading *= 1.5
            if self.verbose:
                print(f"Corner detected, turning in advance: curvature={curvature:.4f}")
        
        # Start turning in advance at one turning radius before the corner
        if corner_detected and corner_distance < turn_radius * 1.5 and corner_distance > turn_radius * 0.5:
            # Calculate corner direction
            corner_heading = ref_state[2]
            corner_heading_error = self.wrap_angle(corner_heading - state[2])
            
            # Increase heading error weight to make vehicle turn in advance
            kp_heading *= 2.0
            # Reduce speed for better turning stability
            desired_speed = min(desired_speed, 0.6)
            
            if self.verbose:
                print(f"Early turn: distance to corner={corner_distance:.2f}m, turning radius={turn_radius:.2f}m")
        
        # Calculate acceleration with smooth speed transition
        speed_error = desired_speed - state[3]
        # Add a smoothing factor to reduce acceleration changes
        a = np.clip(speed_error * 0.5, self.a_min, self.a_max)
        
        # Calculate steering angle
        delta = np.clip((pos_heading_error * kp_pos_heading + heading_error * kp_heading), -self.delta_max, self.delta_max)
        delta_f = delta
        delta_r = -delta
        
        # Print debug information
        if self.verbose:
            print(
                "Fallback control: "
                f"position error={distance:.2f}, "
                f"heading error={np.degrees(heading_error):.2f}, "
                f"position direction error={np.degrees(pos_heading_error):.2f}, "
                f"acceleration={a:.2f}, steering_F={np.degrees(delta_f):.2f}, "
                f"steering_R={np.degrees(delta_r):.2f}, "
                f"curvature={curvature:.4f}, corner detected={corner_detected}, "
                f"corner distance={corner_distance:.2f}"
            )
        
        return np.array([a, delta_f, delta_r])

    def step(self, state, control):
        """
        State update
        """
        x, y, psi, v = state
        a, delta_f, delta_r = control
        
        # Paper-aligned virtual-wheel body-frame update
        vx_body, vy_body, omega = self._paper_body_velocity_np(v, delta_f, delta_r)
        x_dot = vx_body * np.cos(psi) - vy_body * np.sin(psi)
        y_dot = vx_body * np.sin(psi) + vy_body * np.cos(psi)

        x_new = x + x_dot * self.dt
        y_new = y + y_dot * self.dt
        psi_new = psi + omega * self.dt
        v_new = v + a * self.dt
        
        # Angle normalization
        psi_new = self.wrap_angle(psi_new)
        
        # Speed limit
        v_new = np.clip(v_new, self.v_min, self.v_max)
        
        return np.array([x_new, y_new, psi_new, v_new])
    
    def track_path(self, initial_state, path, max_time=100.0, goal=None, goal_heading=None):
        """
        Track path
        """
        ref_path = self.build_reference(path)

        goal_position_tolerance = float(getattr(self, 'goal_position_tolerance', 0.2))
        goal_heading_tolerance = float(getattr(self, 'goal_heading_tolerance', np.deg2rad(60)))
        
        state = initial_state[:4].copy()
        states = [state.copy()]
        controls = []
        prev_control = np.zeros(self.nu, dtype=float)
        
        t = 0.0
        success = False
        
        # Record passed path point index
        passed_idx = 0
        
        while t < max_time:
            # Check if reached goal
            if goal is not None and goal_heading is not None:
                # Use provided goal and heading
                position_error = np.linalg.norm(np.array(goal) - state[:2])
                heading_error = abs(self.wrap_angle(goal_heading - state[2]))
            else:
                # Use last point of reference path
                goal = ref_path[-1]
                position_error = np.linalg.norm(goal[:2] - state[:2])
                heading_error = abs(self.wrap_angle(goal[2] - state[2]))
            
            # Print debug information
            if self.verbose and t % 1.0 < 0.1:
                print(f"Debug info: position error={position_error:.2f}, heading error={np.degrees(heading_error):.2f}")
            
            # todo 跟踪判定条件
            if position_error < goal_position_tolerance and heading_error < goal_heading_tolerance:
                if self.verbose:
                    print(f"Reached goal: position error={position_error:.2f}, heading error={np.degrees(heading_error):.2f}")
                success = True
                break
            
            path_len = len(ref_path)

            # Calculate nearest point on reference path
            search_back = max(0, int(self.afm_search_back_window))
            search_forward = max(1, int(self.afm_search_forward_window))
            search_start = max(0, passed_idx - search_back)
            search_end = min(path_len, passed_idx + search_forward + 1)
            if search_end > search_start:
                local_window = ref_path[search_start:search_end]
                idx = search_start + self.nearest_index(state, local_window)
            else:
                idx = self.nearest_index(state, ref_path)
            
            # Update passed path point index. The tracking bias is applied
            # later, so we keep the monotonic progression conservative here.
            passed_idx = max(passed_idx, idx - 5)  # Allow 5 point error

            if path_len > 20:
                # If reference path is long enough, use middle point as nearest point
                idx = max(idx, passed_idx)  # At least use passed point as nearest point

            tracking_bias = max(0, int(self.afm_tracking_bias))
            tracking_idx_probe = min(idx + tracking_bias, path_len - 1)

            # Analyze reference path curvature, detect upcoming corners
            look_ahead = min(max(0, int(self.afm_curvature_lookahead)), path_len - tracking_idx_probe - 1)
            curvature = 0
            corner_detected = False
            corner_distance = 0
            
            # Turning radius (m)
            turn_radius = 2.5
            
            # Detect corners
            if look_ahead > 3:
                for i in range(tracking_idx_probe, min(tracking_idx_probe + look_ahead - 3, path_len - 3)):
                    # Calculate angle change between three consecutive points
                    p1 = ref_path[i][:2]
                    p2 = ref_path[i+1][:2]
                    p3 = ref_path[i+2][:2]
                    
                    # Calculate vectors
                    v1 = p2 - p1
                    v2 = p3 - p2
                    
                    # Calculate angle change
                    angle1 = np.arctan2(v1[1], v1[0])
                    angle2 = np.arctan2(v2[1], v2[0])
                    angle_change = abs(self.wrap_angle(angle2 - angle1))
                    
                    # If angle change exceeds threshold, corner detected
                    if angle_change > np.deg2rad(30):  # 30 degree threshold
                        corner_detected = True
                        # Calculate distance from current position to corner
                        current_pos = state[:2]
                        corner_pos = p2
                        corner_distance = np.linalg.norm(current_pos - corner_pos)
                        break
            
            # Calculate curvature
            if look_ahead > 2:
                # Calculate curvature between current point, middle point, and look ahead point
                current_point = ref_path[tracking_idx_probe][:2]
                middle_point = ref_path[tracking_idx_probe + look_ahead//2][:2]
                look_ahead_point = ref_path[tracking_idx_probe + look_ahead][:2]
                
                # Calculate vectors
                v1 = middle_point - current_point
                v2 = look_ahead_point - middle_point
                
                # Calculate angle change
                angle1 = np.arctan2(v1[1], v1[0])
                angle2 = np.arctan2(v2[1], v2[0])
                angle_change = abs(self.wrap_angle(angle2 - angle1))
                
                # Calculate curvature
                curvature = angle_change / (look_ahead * 0.1)  # 0.1 is path point interval

            if corner_detected or curvature >= self.afm_moderate_curvature_threshold:
                tracking_bias = max(1, tracking_bias)
            elif curvature <= self.afm_straight_curvature_threshold:
                tracking_bias = 0
            else:
                tracking_bias = min(tracking_bias, 1)

            tracking_idx = min(idx + tracking_bias, path_len - 1)
            passed_idx = max(passed_idx, tracking_idx - 5)
            
            # Get reference trajectory for prediction window
            ref_traj = []

            # Use points on reference path to ensure correct trajectory
            for k in range(self.N + 1):
                # Calculate index on reference path
                j = min(tracking_idx + k, path_len - 1)
                # Add point from reference path
                ref_traj.append(ref_path[j])
            
            ref_traj = np.array(ref_traj)
            
            # Prepare goal state for NMPC
            if goal is not None and goal_heading is not None:
                # Use provided goal and heading
                goal_state = np.array([goal[0], goal[1], goal_heading, 0.0])
            else:
                # Use last point of reference path
                goal_state = ref_path[-1]
            
            # Solve NMPC
            u, ok = self.solve_nmpc(
                state,
                ref_traj,
                goal_state,
                curvature,
                corner_detected,
                corner_distance,
                prev_control=prev_control
            )
            
            # Print debug information
            if self.verbose and t % 1.0 < 0.1:
                print(f"State: x={state[0]:.2f}, y={state[1]:.2f}, psi={np.degrees(state[2]):.2f}, v={state[3]:.2f}")
                print(
                    f"Control: a={u[0]:.2f}, "
                    f"delta_F={np.degrees(u[1]):.2f}, "
                    f"delta_R={np.degrees(u[2]):.2f}"
                )
                print(f"Nearest point index: {idx}, tracking index: {tracking_idx}, reference path length: {path_len}")
                print(f"Reference trajectory first point: {ref_traj[0]}")
                print(f"Reference trajectory last point: {ref_traj[-1]}")
                print(f"Path curvature: {curvature:.4f}, look ahead points: {look_ahead}")
            
            # State update
            state = self.step(state, u)
            prev_control = np.asarray(u, dtype=float).copy()
            
            states.append(state.copy())
            controls.append(u.copy())
            
            t += self.dt
        
        states = np.array(states)
        controls = np.array(controls)
        
        # Evaluation metrics
        err = []
        heading_err = []
        
        for s in states:
            idx = self.nearest_index(s, ref_path)
            ref = ref_path[idx]
            
            err.append(np.linalg.norm(s[:2] - ref[:2]))
            heading_err.append(abs(self.wrap_angle(s[2] - ref[2])))
        
        rmse = np.sqrt(np.mean(np.square(err)))
        heading_error = np.degrees(np.mean(heading_err))
        success_rate = 100.0 if success else 0.0
        
        return states, controls, rmse, heading_error, t, success_rate
