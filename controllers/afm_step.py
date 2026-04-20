"""
AFM Step Controller - Step-based Autonomous Free Movement
Wraps NMPC controller for single-step execution compatible with RL environment
"""
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.AFM import AFM


class AFMStep:
    """
    Step-based AFM (Autonomous Free Movement) Controller
    
    Provides single-step control for path tracking using MPC.
    Each step() call returns control input for one time step.
    """
    
    def __init__(self, map_type='map_a', dt=0.02, horizon=20, map_kwargs=None):
        """
        Initialize AFM step controller
        
        Args:
            map_type: Map type for reference path
            dt: Time step (seconds)
            horizon: MPC prediction horizon
        """
        self.map_type = map_type
        self.dt = dt
        self.horizon = horizon
        
        # Reuse the unified AFM wrapper so pure MPC and training share the
        # same low-level controller implementation.
        self.nmpc = AFM(map_type=map_type, dt=dt, horizon=horizon, map_kwargs=map_kwargs)
        self.env_map = self.nmpc.env
        
        # Reference path will be set externally or from map
        self.reference_path = None
        
        # Internal state tracking
        self.passed_idx = 0  # Track progress along path
        
    def reset(self):
        """Reset controller internal state"""
        self.passed_idx = 0
        self._prev_control = np.zeros(3, dtype=float)
    
    def step(self, state, ref_path=None):
        """
        Execute one control step and return control input
        
        Args:
            state: Current vehicle state [x, y, psi, v]
            ref_path: Reference path (Nx3 or Nx2 array). If None, uses stored path.
        
        Returns:
            np.ndarray: Control input [acceleration, front_steer, rear_steer]
        """
        if ref_path is not None:
            self.reference_path = ref_path
        
        if self.reference_path is None:
            raise ValueError("No reference path provided")
        
        # Keep the live state around so curvature/corner heuristics can reason
        # about how far the vehicle really is from the upcoming corner.
        self._current_state = np.array(state, dtype=float)
        
        # Build reference trajectory with heading info
        ref_traj = self.nmpc.build_reference(self.reference_path)
        
        # Find nearest point on reference path
        idx = self.nmpc.nearest_index(state, ref_traj)
        
        # Update passed index to prevent backtracking
        self.passed_idx = max(self.passed_idx, idx - 5)
        idx = max(idx, self.passed_idx)
        
        # Get reference trajectory segment for prediction window
        path_len = len(ref_traj)
        ref_segment = []
        
        for k in range(self.horizon + 1):
            j = min(idx + k, path_len - 1)
            ref_segment.append(ref_traj[j])
        
        ref_segment = np.array(ref_segment)
        
        # Use the local prediction-window endpoint as the rolling goal. Using
        # the full-path terminal state makes AFM cut across mode-transition
        # segments (for example before an intended APT/AZR maneuver).
        goal_state = ref_segment[-1]
        
        # Analyze path curvature for corner detection
        curvature, corner_detected, corner_distance = self._analyze_curvature(
            ref_traj, idx, path_len, state
        )
        
        # Solve MPC optimization problem
        u, success = self.nmpc.solve_nmpc(
            state,
            ref_segment,
            goal=goal_state,
            curvature=curvature,
            corner_detected=corner_detected,
            corner_distance=corner_distance,
            prev_control=getattr(self, '_prev_control', np.zeros(3, dtype=float))
        )
        self._prev_control = np.asarray(u, dtype=float).copy()
        
        return u
    
    def _analyze_curvature(self, ref_traj, idx, path_len, current_state):
        """
        Analyze reference path curvature and detect corners
        
        Returns:
            float: Curvature value
            bool: Whether corner detected
            float: Distance to corner
        """
        look_ahead = min(25, path_len - idx - 1)
        curvature = 0.0
        corner_detected = False
        corner_distance = 0.0
        
        # Detect corners
        if look_ahead > 3:
            for i in range(idx, min(idx + look_ahead - 3, path_len - 3)):
                p1 = ref_traj[i][:2]
                p2 = ref_traj[i+1][:2]
                p3 = ref_traj[i+2][:2]
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                angle1 = np.arctan2(v1[1], v1[0])
                angle2 = np.arctan2(v2[1], v2[0])
                angle_change = abs(self.nmpc.wrap_angle(angle2 - angle1))
                
                if angle_change > np.deg2rad(30):
                    corner_detected = True
                    current_pos = np.array(current_state[:2], dtype=float)
                    corner_distance = float(np.linalg.norm(current_pos - p2))
                    break
        
        # Calculate curvature
        if look_ahead > 2:
            current_point = ref_traj[idx][:2]
            middle_point = ref_traj[idx + look_ahead//2][:2]
            look_ahead_point = ref_traj[idx + look_ahead][:2]
            
            v1 = middle_point - current_point
            v2 = look_ahead_point - middle_point
            
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            angle_change = abs(self.nmpc.wrap_angle(angle2 - angle1))
            
            curvature = angle_change / (look_ahead * self.dt)
        
        return curvature, corner_detected, corner_distance
    
    def _get_current_position(self):
        """Get current position placeholder"""
        if hasattr(self, '_current_state'):
            return np.array(self._current_state[:2], dtype=float)
        return np.array([0.0, 0.0])


if __name__ == "__main__":
    # Test AFM step controller
    print("Testing AFMStep controller...")
    
    afm = AFMStep(map_type='map_a')
    
    # Test state
    test_state = np.array([1.0, 0.0, 0.0, 0.5])
    
    # Get control input
    u = afm.step(test_state)
    
    print(f"Test state: {test_state}")
    print(f"Control output: {u}")
    print(f"Acceleration: {u[0]:.3f} m/s²")
    print(f"Front steering: {np.degrees(u[1]):.2f} deg")
    print(f"Rear steering: {np.degrees(u[2]):.2f} deg")
