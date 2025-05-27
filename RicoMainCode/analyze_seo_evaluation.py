import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Rectangle

# Set paths
IMAGE_PATH = "seo full form - Google Search.jpeg"
JSON_PATH = "./result/screenshot_eval/seo full form - Google Search.json"
OUTPUT_DIR = "./result/seo_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load JSON and image data"""
    # Load JSON
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # Load image
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return data, image

def analyze_element_types(data):
    """Analyze UI element types and distribution"""
    elements = data['children']
    
    # Count element types
    element_types = {}
    for element in elements:
        element_type = element.get('type', 'unknown')
        if element_type in element_types:
            element_types[element_type] += 1
        else:
            element_types[element_type] = 1
    
    # Generate a bar chart of element types
    plt.figure(figsize=(10, 6))
    types = list(element_types.keys())
    counts = list(element_types.values())
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    sorted_types = [types[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    plt.bar(sorted_types, sorted_counts)
    plt.title('UI Element Types in Google Search Page', fontsize=16)
    plt.xlabel('Element Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/element_types.png", dpi=300)
    plt.close()
    
    return element_types

def analyze_clickable_elements(data):
    """Analyze clickable elements"""
    elements = data['children']
    
    # Count clickable vs non-clickable
    clickable = sum(1 for e in elements if e.get('clickable', False))
    non_clickable = len(elements) - clickable
    
    # Generate a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie([clickable, non_clickable], labels=['Clickable', 'Non-clickable'], 
            autopct='%1.1f%%', colors=['#66b3ff', '#99ff99'])
    plt.title('Clickable vs Non-clickable Elements', fontsize=16)
    plt.savefig(f"{OUTPUT_DIR}/clickable_elements.png", dpi=300)
    plt.close()
    
    return clickable, non_clickable

def analyze_element_distribution(data, image):
    """Analyze spatial distribution of UI elements"""
    elements = data['children']
    height, width = image.shape[:2]
    
    # Create a heatmap of element positions
    heatmap = np.zeros((height, width))
    
    for element in elements:
        if 'bounds' in element and len(element['bounds']) == 4:
            x1, y1, x2, y2 = element['bounds']
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width-1, x2), min(height-1, y2)
            heatmap[y1:y2, x1:x2] += 1
    
    # Normalize heatmap
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Plot heatmap
    plt.figure(figsize=(12, 16))
    plt.imshow(image, alpha=0.7)
    plt.imshow(heatmap, cmap='hot', alpha=0.3)
    plt.title('UI Element Density Heatmap', fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/element_heatmap.png", dpi=300)
    plt.close()
    
    return heatmap

def visualize_elements_by_type(data, image):
    """Visualize elements by type with color coding"""
    elements = data['children']
    
    # Define colors for different element types
    color_map = {
        'Text': 'red',
        'Image': 'blue',
        'Icon': 'green',
        'Button': 'orange',
        'Text Button': 'purple',
        'Toolbar': 'cyan',
        'Drawer': 'magenta',
        'unknown': 'yellow'
    }
    
    # Clone image for drawing
    plt.figure(figsize=(12, 16))
    plt.imshow(image)
    
    # Draw bounding boxes
    for element in elements:
        if 'bounds' in element and len(element['bounds']) == 4:
            x1, y1, x2, y2 = element['bounds']
            # Get element type and color
            element_type = element.get('type', 'unknown')
            color = color_map.get(element_type, 'yellow')
            
            # Draw rectangle
            width = x2 - x1
            height = y2 - y1
            rect = Rectangle((x1, y1), width, height, 
                           linewidth=1, edgecolor=color, facecolor='none', alpha=0.7)
            plt.gca().add_patch(rect)
    
    # Add color legend
    handles = [Rectangle((0, 0), 1, 1, color=color) for color in color_map.values()]
    plt.legend(handles, color_map.keys(), loc='upper right')
    
    plt.title('UI Elements by Type', fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/elements_by_type.png", dpi=300)
    plt.close()

def generate_analysis_report(data, image, element_types, clickable_stats):
    """Generate a comprehensive analysis report"""
    elements = data['children']
    height, width = image.shape[:2]
    clickable, non_clickable = clickable_stats
    
    # Calculate element density
    total_pixels = width * height
    element_pixels = sum((e['bounds'][2] - e['bounds'][0]) * (e['bounds'][3] - e['bounds'][1]) 
                        for e in elements if 'bounds' in e and len(e['bounds']) == 4)
    screen_coverage = (element_pixels / total_pixels) * 100
    
    # Write report
    report_path = f"{OUTPUT_DIR}/analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("===== UI ANALYSIS REPORT: GOOGLE SEARCH PAGE =====\n\n")
        f.write(f"Image: {IMAGE_PATH}\n")
        f.write(f"Resolution: {width}x{height} pixels\n\n")
        
        f.write("===== ELEMENT STATISTICS =====\n")
        f.write(f"Total UI Elements: {len(elements)}\n")
        f.write(f"Clickable Elements: {clickable} ({clickable/len(elements)*100:.1f}%)\n")
        f.write(f"Non-clickable Elements: {non_clickable} ({non_clickable/len(elements)*100:.1f}%)\n")
        f.write(f"Screen Coverage: {screen_coverage:.1f}%\n\n")
        
        f.write("===== ELEMENT TYPE DISTRIBUTION =====\n")
        for element_type, count in sorted(element_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{element_type}: {count} ({count/len(elements)*100:.1f}%)\n")
        
        f.write("\n===== USABILITY INSIGHTS =====\n")
        # Add some automatic insights based on the analysis
        if clickable/len(elements) < 0.2:
            f.write("- Low proportion of clickable elements, potentially limiting user interaction\n")
        if screen_coverage > 70:
            f.write("- High screen coverage may lead to visual clutter\n")
        if screen_coverage < 30:
            f.write("- Low screen coverage might indicate excessive white space\n")
        if len(element_types) < 3:
            f.write("- Limited variety of UI element types might reduce visual hierarchy\n")
        
        f.write("\n===== GENERATED VISUALIZATIONS =====\n")
        f.write("1. element_types.png - Distribution of UI element types\n")
        f.write("2. clickable_elements.png - Proportion of clickable vs non-clickable elements\n")
        f.write("3. element_heatmap.png - Heatmap showing element density\n")
        f.write("4. elements_by_type.png - Visualization of elements by type\n")
    
    print(f"Analysis report saved to {report_path}")
    return report_path

def main():
    # Load data
    data, image = load_data()
    
    # Run analysis
    element_types = analyze_element_types(data)
    clickable_stats = analyze_clickable_elements(data)
    analyze_element_distribution(data, image)
    visualize_elements_by_type(data, image)
    
    # Generate report
    report_path = generate_analysis_report(data, image, element_types, clickable_stats)
    
    print("\n===== UI ANALYSIS COMPLETE =====")
    print(f"Analysis results saved to {OUTPUT_DIR}/")
    print("Generated files:")
    print("1. element_types.png - Distribution of UI element types")
    print("2. clickable_elements.png - Proportion of clickable vs non-clickable elements")
    print("3. element_heatmap.png - Heatmap showing element density")
    print("4. elements_by_type.png - Visualization of elements by type")
    print(f"5. {report_path} - Comprehensive analysis report")

if __name__ == "__main__":
    main() 