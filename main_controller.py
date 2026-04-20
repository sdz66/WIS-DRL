"""
Main Controller
Coordinates AFM, AZR, and APT modules to complete path tracking
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from map_manager import MapManager
from controllers.AFM import AFM
from controllers.AZR import AZR
from controllers.APT import APT


def run_main_controller(map_type, waypoints=None):
    """
    Run main controller path tracking
    
    Args:
        map_type: Map type to use
        waypoints: List of waypoints with mode change information
                  Each waypoint is a dict: {'position': (x, y), 'mode': 'afm|azr|apt', 'direction': 'left|right'} (direction only for apt)
    
    Returns:
        positions: List of positions at 0.1 second intervals
        rmse: Root mean square error
        heading_error: Heading error in degrees
        success_rate: Success rate in percentage
        time: Completion time in seconds
    """
    # Create map
    manager = MapManager()
    env = manager.create_map(map_type)
    
    # Get initial state and reference path
    initial_state = np.array([
        env.initial_state['x'],
        env.initial_state['y'],
        env.initial_state['psi'],
        0.0  # Initial speed
    ])
    
    reference_path = env.reference_path

    if waypoints is None and map_type == 'tri_mode_composite':
        waypoints = [
            {
                'position': (env.apt_switch_x, env.lower_center_y),
                'mode': 'afm',
                'heading': 0.0,
            },
            {
                'position': (env.apt_resume_x, env.upper_center_y),
                'mode': 'apt',
                'direction': 'left',
                'heading': 0.0,
            },
            {
                'position': (env.rotation_x, env.upper_center_y),
                'mode': 'afm',
                'heading': 0.0,
            },
            {
                'position': (env.rotation_x, env.upper_center_y),
                'mode': 'azr',
            },
            {
                'position': env.end_point,
                'mode': 'afm',
                'heading': np.pi,
            },
        ]
    
    # Print reference path information
    print(f"Reference path length: {len(reference_path)}")
    print(f"Reference path start: {reference_path[0]}")
    print(f"Reference path end: {reference_path[-1]}")
    print(f"Reference path middle: {reference_path[len(reference_path)//2]}")
    
    # Create controllers
    afm = AFM(map_type=map_type)
    azr = AZR(map_type=map_type)
    apt = APT(map_type=map_type)
    
    # Track path
    print(f"Starting main controller path tracking - {map_type}")
    
    # Default to AFM if no waypoints provided
    if waypoints is None:
        # Use AFM to track the full reference path
        states, controls, rmse, heading_error, time, success_rate = afm.nmpc.track_path(
            initial_state, reference_path, max_time=100.0, goal=env.end_point, goal_heading=env.end_heading
        )
    else:
        # Use waypoints with mode changes
        current_position = (env.initial_state['x'], env.initial_state['y'])
        current_heading = env.initial_state['psi']
        all_states = []
        total_time = 0.0
        
        # Process waypoints
        for i, waypoint in enumerate(waypoints):
            waypoint_pos = waypoint['position']
            mode = waypoint['mode']
            direction = waypoint.get('direction', 'left')
            
            print(f"\n=== Processing waypoint {i+1}: {waypoint_pos} with {mode} mode ===")
            
            if mode == 'afm':
                # Use AFM to move to waypoint
                wp_heading = waypoint.get('heading', current_heading)
                
                # Get reference path from environment
                if hasattr(env, '_generate_reference_path'):
                    reference_path = env._generate_reference_path()
                else:
                    reference_path = env.reference_path
                
                # Track path using full reference path
                states, _, _, _, waypoint_time, _ = afm.track_path(
                    np.array([current_position[0], current_position[1], current_heading, 0.0]),
                    reference_path,
                    max_time=50.0, goal=waypoint_pos, goal_heading=wp_heading
                )
                all_states.extend(states)
                total_time += waypoint_time
                
            elif mode == 'azr':
                # Use AZR to reverse direction at current position - stop any previous movement
                print("Stopping AFM movement and reversing direction in-place...")
                # Reverse direction in-place
                final_pos, final_heading, azr_states = azr.reverse_direction(
                    current_position,
                    current_heading,
                    return_trajectory=True
                )
                if len(azr_states) > 0:
                    all_states.extend(azr_states)
                    total_time += max(0.0, (len(azr_states) - 1) * 0.1)
                current_position = tuple(final_pos)
                current_heading = final_heading
                
            elif mode == 'apt':
                # Use APT to translate to waypoint (pure translation) - stop any previous movement
                wp_heading = waypoint.get('heading', current_heading)
                
                final_pos, final_heading, apt_states = apt.translate(
                    (current_position[0], current_position[1], current_heading),
                    (waypoint_pos[0], waypoint_pos[1], wp_heading),
                    direction=direction,
                    return_trajectory=True
                )
                if len(apt_states) > 0:
                    all_states.extend(apt_states)
                waypoint_time = max(0.0, (len(apt_states) - 1) * 0.1)
                total_time += waypoint_time
                current_position = final_pos
                current_heading = final_heading
            
            # Update current state (already updated for APT mode)
            if mode == 'afm' and 'states' in locals() and len(states) > 0:
                current_position = (states[-1][0], states[-1][1])
                current_heading = states[-1][2]
        
        # Process final leg to end
        print(f"\n=== Processing final leg to end: {env.end_point} ===")
        
        # Get reference path from environment
        if hasattr(env, '_generate_reference_path'):
            reference_path = env._generate_reference_path()
        else:
            reference_path = env.reference_path
        
        # Track path using full reference path
        states, controls, rmse, heading_error, final_time, success_rate = afm.track_path(
            np.array([current_position[0], current_position[1], current_heading, 0.0]),
            reference_path,
            max_time=50.0, goal=env.end_point, goal_heading=env.end_heading
        )
        all_states.extend(states)
        total_time += final_time
        
        states = np.array(all_states)
        time = total_time
    
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
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw map
    env.draw_track(ax)
    
    # Draw tracking path
    if len(states) > 0:
        ax.plot(states[:, 0], states[:, 1], 'b-', label='Main Controller Tracking Path')
        
        # Draw vehicle at final position
        final_state = states[-1]
        env._draw_vehicle_at(ax, final_state[0], final_state[1], final_state[2], color='green')
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{map_type} - Main Controller Path Tracking')
    ax.legend()
    ax.grid(True)
    
    # Save image
    output_path = f'outputs/{map_type}-main_controller.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{map_type} tracking results:")
    print(f"RMSE: {rmse:.4f} m")
    print(f"Heading error: {heading_error:.4f} deg")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Completion time: {time:.2f} s")
    print(f"Image saved to: {output_path}")
    
    return positions, rmse, heading_error, success_rate, time


def main():
    """
    Main function
    """
    map_types = ['map_a', 'map_b', 'map_c', 'tri_mode_composite']  # Process all maps
    results = []
    all_positions = {}
    
    for map_type in map_types:
        print(f"\n=== Processing map: {map_type} ===")
        positions, rmse, heading_error, success_rate, time = run_main_controller(map_type)
        results.append((map_type, rmse, heading_error, success_rate, time))
        all_positions[map_type] = positions
    
    # Save positions to CSV file
    csv_path = 'outputs/main_controller_vehicle_positions.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = []
        for map_type in map_types:
            header.extend([f'{map_type}_x', f'{map_type}_y'])
        writer.writerow(header)
        
        # Find maximum number of positions
        max_len = max(len(positions) for positions in all_positions.values())
        
        # Write data rows
        for i in range(max_len):
            row = []
            for map_type in map_types:
                if i < len(all_positions[map_type]):
                    x, y = all_positions[map_type][i]
                    row.extend([x, y])
                else:
                    row.extend(['', ''])
            writer.writerow(row)
    
    print(f"\nPositions saved to: {csv_path}")
    
    # Print summary results
    print("\n=== Summary Results ===")
    print("Map Type\tRMSE(m)\tHeading Error(deg)\tSuccess Rate(%)\tCompletion Time(s)")
    for result in results:
        print(f"{result[0]}\t{result[1]:.4f}\t{result[2]:.4f}\t\t{result[3]:.2f}\t\t{result[4]:.2f}")


if __name__ == "__main__":
    main()
