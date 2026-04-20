"""Map visualization script for the tri-mode composite map."""
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maps.tri_mode_composite_map import TriModeCompositeEnv


def draw_composite_map():
    """Draw the tri-mode composite map and save it to the figures directory."""
    print("\n=== Drawing TriModeCompositeEnv ===")
    
    # Create map instance
    map_env = TriModeCompositeEnv()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw the track
    map_env.draw_track(ax)
    
    # Add title and legend
    ax.set_title('Tri-Mode Composite Map', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'tri_mode_composite_map.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Also show the plot
    plt.show()
    
    print(f"\n✅ Map saved to: {output_path}")
    print("✅ Map visualization completed!")
    
    # Print map details
    print("\n=== Map Details ===")
    print(f"Map range: x={map_env.x_range}, y={map_env.y_range}")
    print(f"Initial state: {map_env.initial_state}")
    print(f"Goal: {map_env.end_point} (heading: {map_env.end_heading:.2f} rad)")
    print(f"Reference path points: {len(map_env.reference_path)}")
    
    # Print drivable areas
    print("\nDrivable areas:")
    for name, area in map_env.drivable_areas.items():
        print(f"  - {name}: x[{area['x_min']:.1f}, {area['x_max']:.1f}], y[{area['y_min']:.1f}, {area['y_max']:.1f}]")


if __name__ == "__main__":
    draw_composite_map()
