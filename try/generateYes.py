# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import splprep, splev
from PIL import Image
import gc
import os
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# %%
def generate_smooth_mouse_heatmap_custom_color(num_images=2000, width=1920, height=1080, num_paths=3, points_per_path=100, output_dir="heatmaps_yes"):
    """
    Generate multiple heatmaps of smooth mouse movements concentrated in specific areas and save them.

    :param num_images: Number of images to generate (default: 10000)
    :param width: Width of the website/webpage (default: 1920)
    :param height: Height of the website/webpage (default: 1080)
    :param num_paths: Number of smooth mouse paths to simulate (default: 3)
    :param points_per_path: Number of points in each path (default: 100)
    :param output_dir: Directory to save the generated heatmaps (default: "heatmaps_yes")
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define specific hot zones (x_start, y_start, width, height)
    hot_zones = [
        (100, 200, 400, 300),  # Example zone 1
        (800, 400, 300, 300),  # Example zone 2
        (1500, 100, 300, 600), # Example zone 3
    ]

    # Main progress bar for overall image generation
    for img_index in tqdm(range(num_images), desc="Generating heatmaps", unit="image"):
        mouse_data = []

        # Generate multiple smooth mouse movement paths
        for _ in range(num_paths):
            # Choose a random hot zone
            zone = hot_zones[np.random.randint(len(hot_zones))]
            x_start = np.random.randint(zone[0], zone[0] + zone[2])
            y_start = np.random.randint(zone[1], zone[1] + zone[3])
            x_end = np.random.randint(zone[0], zone[0] + zone[2])
            y_end = np.random.randint(zone[1], zone[1] + zone[3])

            # Control points for the curve (Bezier-like curve)
            control_x = np.random.randint(zone[0], zone[0] + zone[2], 4)  # Random intermediate control points
            control_y = np.random.randint(zone[1], zone[1] + zone[3], 4)

            # Ensure control points are distinct
            if len(set(control_x)) < len(control_x) or len(set(control_y)) < len(control_y):
                continue

            # Add start and end points to control points
            x_points = np.concatenate([[x_start], control_x, [x_end]])
            y_points = np.concatenate([[y_start], control_y, [y_end]])

            # Create smooth curve using splines
            try:
                tck, u = splprep([x_points, y_points], s=0)
                u_new = np.linspace(u.min(), u.max(), points_per_path)
                smooth_path = splev(u_new, tck)
            except ValueError as e:
                print(f"Skipping path due to error: {e}")
                continue

            # Extract the smooth path points and add to the mouse data
            x_smooth, y_smooth = smooth_path
            mouse_data.extend(list(zip(x_smooth, y_smooth)))

        # Create 2D histogram for heatmap
        heatmap, xedges, yedges = np.histogram2d(
            [x for x, y in mouse_data],
            [y for x, y in mouse_data],
            bins=[np.arange(0, width, 10), np.arange(0, height, 10)]
        )

        # Normalize the heatmap for better visibility
        heatmap = np.clip(heatmap.T, 0, 255)

        # Plotting the heatmap using seaborn with a custom colormap
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap, cmap='coolwarm', cbar=False)  # 'coolwarm' closely matches the red-yellow-blue gradient

        # Remove axes for a clean look
        plt.axis('off')

        # Save the generated heatmap
        output_filename = os.path.join(output_dir, f"smooth_mouse_heatmap_yes_1_{img_index + 1}.png")
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        
        # Clear the plot to free up memory - close it properly
        plt.clf()
        plt.close('all')

        # Explicitly call the garbage collector every 100 images
        if (img_index + 1) % 100 == 0:
            gc.collect()
            
    # Final status message
    print(f"Generation complete. {num_images} images saved to {os.path.abspath(output_dir)}")

# %%
# Example usage to generate multiple heatmaps - with a clearer message
print("Starting heatmap generation...")
generate_smooth_mouse_heatmap_custom_color(num_images=2000)