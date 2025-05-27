# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import splprep, splev
from PIL import Image
import gc
import os
from tqdm import tqdm


# %%
# Function to generate smooth mouse movement heatmap with scattered appearance
def generate_smooth_mouse_heatmap_scattered(num_images=10000, width=1920, height=1080, 
                                           num_paths=3, points_per_path=100, 
                                           output_dir="heatmaps_no"):
    """
    Generate multiple heatmaps of smooth mouse movements with a scattered appearance and save them.

    :param num_images: Number of images to generate (default: 10000)
    :param width: Width of the website/webpage (default: 1920)
    :param height: Height of the website/webpage (default: 1080)
    :param num_paths: Number of smooth mouse paths to simulate (default: 3)
    :param points_per_path: Number of points in each path (default: 100)
    :param output_dir: Directory to save the generated heatmaps (default: "heatmaps_no")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Main progress bar for overall image generation
    for img_index in tqdm(range(num_images), desc="Generating scattered heatmaps", unit="image"):
        mouse_data = []

        # Generate multiple smooth mouse movement paths
        for _ in range(num_paths):
            # Start and end points for the path with more randomness
            start_x, start_y = np.random.randint(0, width), np.random.randint(0, height)
            end_x, end_y = np.random.randint(0, width), np.random.randint(0, height)

            # Generate more control points for smoother paths
            control_x = np.random.randint(0, width, 6)  # 6 random intermediate control points
            control_y = np.random.randint(0, height, 6)

            # Add start and end points to control points
            x_points = np.concatenate([[start_x], control_x, [end_x]])
            y_points = np.concatenate([[start_y], control_y, [end_y]])

            # Create smooth curve using splines with error handling
            try:
                tck, u = splprep([x_points, y_points], s=0)
                u_new = np.linspace(u.min(), u.max(), points_per_path)
                smooth_path = splev(u_new, tck)
                
                # Extract the smooth path points and add to the mouse data
                x_smooth, y_smooth = smooth_path
                mouse_data.extend(list(zip(x_smooth, y_smooth)))
            except ValueError as e:
                print(f"Skipping path due to error: {e}")
                continue

        # Create 2D histogram for heatmap with adjusted bins for scattering
        heatmap, xedges, yedges = np.histogram2d(
            [x for x, y in mouse_data],
            [y for x, y in mouse_data],
            bins=[np.arange(0, width, 15), np.arange(0, height, 15)]  # Larger bin sizes for more scatter
        )

        # Normalize the heatmap for better visibility
        heatmap = np.clip(heatmap.T, 0, 255)

        # Plotting the heatmap using seaborn with a custom colormap (blue -> green -> yellow -> red)
        plt.figure(figsize=(12, 6), facecolor='white')  # Set background to white
        sns.heatmap(heatmap, cmap='coolwarm', cbar=False)  # Using 'coolwarm' colormap

        # Remove axes for a clean look
        plt.axis('off')

        # Save the generated heatmap to the output directory
        output_filename = os.path.join(output_dir, f"smooth_mouse_heatmap_scattered_no_0_{img_index + 1}.png")
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        
        # Clear the plot to free up memory - close it properly
        plt.clf()
        plt.close('all')

        # Explicitly call the garbage collector every 100 images for memory efficiency
        if (img_index + 1) % 100 == 0:
            gc.collect()
    
    # Final status message
    print(f"Generation complete. {num_images} images saved to {os.path.abspath(output_dir)}")

# %%
# Example usage to generate multiple heatmaps
print("Starting scattered heatmap generation...")
generate_smooth_mouse_heatmap_scattered(
    num_images=10000, 
    output_dir="heatmaps_no"
)