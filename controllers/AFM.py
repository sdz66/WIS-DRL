"""
AFM (Autonomous Free Movement) module
For point-to-point autonomous movement along reference path
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from map_manager import MapManager
from controllers.casadi_nmpc_robust import CasADiNMPCRobust


class AFM(CasADiNMPCRobust):
    """
    Autonomous Free Movement class
    For point-to-point movement along reference path
    Inherits all functionality from CasADiNMPCRobust
    """
    
    def __init__(self, map_type='map_a', dt: float = 0.02, horizon: int = 20, map_kwargs=None):
        """
        Initialize AFM
        
        Args:
            map_type: Map type to use for movement
            dt: Time step (s)
            horizon: Prediction horizon
        """
        # Initialize parent class (CasADiNMPCRobust)
        super().__init__(dt=dt, horizon=horizon)
        
        # AFM specific initialization
        self.map_type = map_type
        self.manager = MapManager()
        self.env = self.manager.create_map(map_type, **(map_kwargs or {}))
        # CasADiNMPCRobust is the parent controller itself, so configure the
        # inherited NMPC directly instead of expecting an extra wrapper member.
        self.configure_from_map(self.env)
        self.goal_position_tolerance = float(getattr(self.env, 'goal_position_tolerance', 0.2))
        self.goal_heading_tolerance = float(getattr(self.env, 'goal_heading_tolerance', np.deg2rad(60)))
    
    def move(self, start, goal):
        """
        Move from start to goal along reference path
        
        Args:
            start: Start position and heading (x, y, psi)
            goal: Goal position and heading (x, y, psi)
        
        Returns:
            final_position: Final position (x, y)
            final_heading: Final heading (rad)
        """
        # Update map with new start and goal
        self.env.set_start_point(start[:2])
        self.env.set_start_heading(start[2])
        self.env.set_end_point(goal[:2])
        self.env.set_end_heading(goal[2])
        
        # Get reference path
        if hasattr(self.env, '_generate_reference_path'):
            # Regenerate reference path based on new start and goal
            reference_path = self.env._generate_reference_path()
        else:
            # Use existing reference path
            reference_path = self.env.reference_path
        
        # Initial state
        initial_state = np.array([
            start[0],
            start[1],
            start[2],
            0.0  # Initial speed
        ])
        
        # Track path - using inherited track_path method from CasADiNMPCRobust
        print(f"Starting AFM movement - {self.map_type}")
        states, controls, rmse, heading_error, time, success_rate = self.track_path(
            initial_state,
            reference_path,
            max_time=100.0,
            goal=goal[:2],
            goal_heading=goal[2]
        )
        
        # Extract positions at 0.1 second intervals
        positions = []
        dt = 0.1
        current_time = 0.0
        state_index = 0
        
        while current_time < time and state_index < len(states):
            # Find the state closest to current_time
            while state_index < len(states) - 1 and (state_index + 1) * 0.1 <= current_time:
                state_index += 1
            
            positions.append(states[state_index][:2])
            current_time += dt
        
        # Add the final position
        if len(states) > 0:
            positions.append(states[-1][:2])
        
        print(f"\n{self.map_type} AFM results:")
        print(f"RMSE: {rmse:.4f} m")
        print(f"Heading error: {heading_error:.4f} deg")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Completion time: {time:.2f} s")
        
        # Save positions to CSV file
        csv_path = f'outputs/{self.map_type}-afm_positions.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y'])
            for pos in positions:
                writer.writerow([pos[0], pos[1]])
        
        print(f"Positions saved to: {csv_path}")
        
        # Return final position and heading
        if len(states) > 0:
            final_state = states[-1]
            return final_state[:2], final_state[2]
        else:
            return start[:2], start[2]


if __name__ == "__main__":
    """
    Example usage of AFM
    """
    # Create AFM instance
    afm = AFM(map_type='map_a')
    
    # Define start and goal
    start = (1.0, 0.0, 0.0)  # (x, y, psi)
    goal = (9.0, -2.0, 0.0)   # (x, y, psi)
    
    # Move from start to goal
    final_position, final_heading = afm.move(start, goal)
    
    print(f"\nAFM movement completed!")
    print(f"Final position: {final_position}")
    print(f"Final heading: {np.degrees(final_heading):.2f} deg")
