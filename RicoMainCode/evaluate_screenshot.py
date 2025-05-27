#!/usr/bin/env python3
import os
import argparse
import sys
from detect_ui_elements import extract_ui_elements_from_image
from run_evaluation import evaluate_ui

def main():
    """
    Main function to detect UI elements from a screenshot and run evaluation
    """
    parser = argparse.ArgumentParser(description="Evaluate UI from a screenshot")
    parser.add_argument("image_path", help="Path to the screenshot image")
    parser.add_argument("--output_dir", help="Directory to save results", default="./result/screenshot_eval")
    parser.add_argument("--skip_detection", action="store_true", help="Skip UI detection if JSON already exists")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image path and prepare JSON path
    image_path = args.image_path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(args.output_dir, f"{base_name}.json")
    
    # Check if we should skip detection
    if not args.skip_detection or not os.path.exists(json_path):
        print("\n===== STEP 1: DETECTING UI ELEMENTS =====")
        # Extract UI elements from image
        elements, complexity, json_path = extract_ui_elements_from_image(image_path, json_path)
        
        if elements is None:
            print("Failed to detect UI elements. Exiting.")
            return 1
        
        print(f"Successfully detected {len(elements)} UI elements")
    else:
        print(f"\nUsing existing JSON file: {json_path}")
    
    # Run UI evaluation
    print("\n===== STEP 2: EVALUATING UI =====")
    result = evaluate_ui(image_path, json_path)
    
    if result is None:
        print("Evaluation failed. Please check the image and JSON files.")
        return 1
    
    score, ui_elements = result
    
    print(f"\n===== UI EVALUATION RESULTS =====")
    print(f"Overall Usability Score: {score:.1f} / 10\n")
    print(f"Detected {len(ui_elements)} UI elements")
    
    # Count elements by type
    element_types = {}
    for element in ui_elements:
        element_type = element['type']
        if element_type in element_types:
            element_types[element_type] += 1
        else:
            element_types[element_type] = 1
    
    print("\nUI Element Types:")
    for element_type, count in element_types.items():
        print(f"  - {element_type}: {count}")
    
    print("\nVisualization will be saved to the results directory")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 