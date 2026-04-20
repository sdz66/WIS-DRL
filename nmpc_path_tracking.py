"""
Path tracking using CasADi NMPC
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from map_manager import MapManager
from controllers.casadi_nmpc_robust import CasADiNMPCRobust


def run_nmpc_tracking(map_type):
    """
    Run NMPC path tracking
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
    
    # Print reference path information
    print(f"Reference path length: {len(reference_path)}")
    print(f"Reference path start: {reference_path[0]}")
    print(f"Reference path end: {reference_path[-1]}")
    print(f"Reference path middle: {reference_path[len(reference_path)//2]}")
    
    # Create NMPC controller
    nmpc = CasADiNMPCRobust(dt=0.1, horizon=5)
    
    # Track path
    print(f"Starting NMPC path tracking - {map_type}")
    states, controls, rmse, heading_error, time, success_rate = nmpc.track_path(
        initial_state, reference_path, max_time=100.0, goal=env.end_point, goal_heading=env.end_heading
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
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw map
    env.draw_track(ax)
    
    # Draw NMPC tracking path
    if len(states) > 0:
        ax.plot(states[:, 0], states[:, 1], 'b-', label='NMPC Tracking Path')
        
        # Draw vehicle at final position
        final_state = states[-1]
        env._draw_vehicle_at(ax, final_state[0], final_state[1], final_state[2], color='green')
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{map_type} - NMPC Path Tracking')
    ax.legend()
    ax.grid(True)
    
    # Save image
    output_path = f'outputs/{map_type}-nmpc.png'
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
        positions, rmse, heading_error, success_rate, time = run_nmpc_tracking(map_type)
        results.append((map_type, rmse, heading_error, success_rate, time))
        all_positions[map_type] = positions
    
    # Save positions to CSV file
    csv_path = 'outputs/nmpc_vehicle_positions.csv'
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
