"""
Mode Switch Controller
Program to call main controller with custom mode switch segments
"""
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_controller import run_main_controller
from map_manager import MapManager


def define_mode_segments(map_type):
    """
    Define mode switch segments for different maps
    
    Args:
        map_type: Map type to define segments for
    
    Returns:
        List of waypoints with mode change information
    """
    if map_type == 'map_a':
        return None

    env = MapManager().create_map(map_type)

    if map_type == 'map_b':
        return [
            {'position': (env.transfer_x - 0.10, 0.0), 'mode': 'afm', 'heading': 0.0},
            {'position': (env.transfer_x, env.target_y), 'mode': 'apt', 'direction': 'left', 'heading': 0.0},
            {'position': env.end_point, 'mode': 'afm', 'heading': env.end_heading},
        ]
    elif map_type == 'map_c':
        return [
            {'position': (env.rotation_x, env.initial_state['y']), 'mode': 'afm', 'heading': env.initial_state['psi']},
            {'position': (env.rotation_x, env.initial_state['y']), 'mode': 'azr'},
            {'position': env.end_point, 'mode': 'afm', 'heading': env.end_heading},
        ]
    elif map_type == 'tri_mode_composite':
        return [
            {'position': (env.apt_switch_x, env.lower_center_y), 'mode': 'afm', 'heading': 0.0},
            {'position': (env.apt_resume_x, env.upper_center_y), 'mode': 'apt', 'direction': 'left', 'heading': 0.0},
            {'position': (env.rotation_x, env.upper_center_y), 'mode': 'afm', 'heading': 0.0},
            {'position': (env.rotation_x, env.upper_center_y), 'mode': 'azr'},
            {'position': env.end_point, 'mode': 'afm', 'heading': np.pi},
        ]

    return None


def run_mode_switch():
    """
    Run mode switch controller for all maps
    """
    map_types = ['map_a', 'map_b', 'map_c', 'tri_mode_composite']  # Process all maps
    results = []
    
    for map_type in map_types:
        print(f"\n=== Processing map: {map_type} ===")
        
        # Define mode switch segments
        waypoints = define_mode_segments(map_type)
        
        # Run main controller with mode switches
        positions, rmse, heading_error, success_rate, time = run_main_controller(map_type, waypoints)
        results.append((map_type, rmse, heading_error, success_rate, time))
    
    # Print summary results
    print("\n=== Summary Results ===")
    print("Map Type\tRMSE(m)\tHeading Error(deg)\tSuccess Rate(%)\tCompletion Time(s)")
    for result in results:
        print(f"{result[0]}\t{result[1]:.4f}\t{result[2]:.4f}\t\t{result[3]:.2f}\t\t{result[4]:.2f}")


if __name__ == "__main__":
    run_mode_switch()
