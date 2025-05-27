import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
OUTPUT_DIR = "./result/nielsen_weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_nielsen_piechart():
    """Generate a pie chart showing the weightage of Nielsen's 10 heuristics"""
    
    # Nielsen's heuristics and their weights
    heuristics = [
        "1. Visibility of system status",
        "2. Match between system and real world",
        "3. User control and freedom",
        "4. Consistency and standards",
        "5. Error prevention",
        "6. Recognition rather than recall",
        "7. Flexibility and efficiency of use",
        "8. Aesthetic and minimalist design",
        "9. Help users recognize/recover from errors",
        "10. Help and documentation"
    ]
    
    weights = [15, 10, 12, 14, 8, 12, 7, 13, 6, 3]  # in percentage
    
    # Colors for the pie chart (using a color palette that is visually distinct)
    colors = plt.cm.tab10(np.arange(len(heuristics)))
    
    # Create pie chart
    plt.figure(figsize=(12, 10))
    
    # Create a slightly exploded pie to highlight the most important heuristics
    explode = [0.1 if w >= 14 else 0.05 if w >= 12 else 0 for w in weights]
    
    # Generate the pie chart with percentage labels
    patches, texts, autotexts = plt.pie(
        weights, 
        explode=explode,
        labels=None,  # We'll add a custom legend instead
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        pctdistance=0.85
    )
    
    # Make the percentage text more readable
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Add a circle at the center to make it look like a donut chart (optional)
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    plt.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Add title with custom styling
    plt.title('Weightage of Nielsen\'s 10 Usability Heuristics', fontsize=18, pad=20)
    
    # Create a custom legend with numbered heuristics
    legend_labels = [f"{h} ({w}%)" for h, w in zip(heuristics, weights)]
    plt.legend(
        patches,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(-0.15, 0.5),
        fontsize=10
    )
    
    # Add a text explaining the weightage system
    plt.figtext(
        0.5, 0.01,
        "Weights are based on detectability, user impact, modern UI trends, and objective measurability",
        ha='center',
        fontsize=10,
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    output_path = f"{OUTPUT_DIR}/nielsen_heuristics_weights.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Nielsen's heuristics weightage pie chart to {output_path}")
    
    # Create a second version with a different visualization style (horizontal bar chart)
    plt.figure(figsize=(12, 8))
    
    # Sort the heuristics by weight
    sorted_indices = np.argsort(weights)
    sorted_heuristics = [heuristics[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    bars = plt.barh(
        sorted_heuristics,
        sorted_weights,
        color=sorted_colors,
        alpha=0.8
    )
    
    # Add weight labels
    for bar, weight in zip(bars, sorted_weights):
        plt.text(
            weight + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{weight}%",
            va='center',
            fontweight='bold'
        )
    
    # Add labels and title
    plt.xlabel('Weight (%)', fontsize=12)
    plt.title('Nielsen\'s Heuristics by Weight', fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the bar chart
    bar_output_path = f"{OUTPUT_DIR}/nielsen_heuristics_barchart.png"
    plt.savefig(bar_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Nielsen's heuristics weightage bar chart to {bar_output_path}")
    
    return output_path, bar_output_path

if __name__ == "__main__":
    pie_path, bar_path = generate_nielsen_piechart()
    
    print("\n===== NIELSEN'S HEURISTICS WEIGHTAGE =====")
    print("Weights are assigned based on:")
    print("1. Detectability through automated analysis")
    print("2. Impact on core user experience")
    print("3. Alignment with modern UI design trends")
    print("4. Objective measurability in interface elements")
    
    print("\nWeightage distribution:")
    heuristics = [
        "Visibility of system status",
        "Match between system and real world",
        "User control and freedom",
        "Consistency and standards",
        "Error prevention",
        "Recognition rather than recall",
        "Flexibility and efficiency of use",
        "Aesthetic and minimalist design",
        "Help users recognize/recover from errors",
        "Help and documentation"
    ]
    weights = [15, 10, 12, 14, 8, 12, 7, 13, 6, 3]
    
    for h, w in zip(heuristics, weights):
        print(f"  - {h}: {w}%") 