import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from preprocess import process_image, extract_ui_elements, extract_features

# Load the model with custom objects
def load_ui_model():
    MODEL_PATH = "./models/ui_evaluation_model.h5"
    custom_objects = {
        'mse': tf.keras.metrics.MeanSquaredError(),
        'mae': tf.keras.metrics.MeanAbsoluteError()
    }
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print(f"Model loaded from {MODEL_PATH}")
    
    # Check if it's a hybrid model
    is_hybrid_model = isinstance(model.input, list)
    print(f"Using {'hybrid' if is_hybrid_model else 'standard'} model")
    return model, is_hybrid_model

# Main evaluation function
def evaluate_ui(image_path, json_path):
    # Load model
    model, is_hybrid_model = load_ui_model()
    
    # Process image
    image = process_image(image_path)
    if image is None:
        print(f"Error: Unable to process image at {image_path}")
        return None
    
    # Extract UI elements
    ui_elements, complexity = extract_ui_elements(json_path)
    if not ui_elements:
        print(f"Error: Unable to extract UI elements from {json_path}")
        return None
    
    # Prepare input for model
    if is_hybrid_model:
        # For hybrid model, need both image and features
        features = extract_features(ui_elements, complexity)
        image_batch = np.expand_dims(image, axis=0)
        features_batch = np.expand_dims(features, axis=0)
        
        # Make prediction with hybrid model
        prediction = model.predict([image_batch, features_batch])
    else:
        # For standard model, only need the image
        image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction with standard model
        prediction = model.predict(image_batch)
    
    score = float(prediction[0][0])
    # Limit score to 0-10 range
    score = max(0, min(10, score))
    
    return score, ui_elements

# Main execution
if __name__ == "__main__":
    # Example usage with test files
    test_image = "test.png"
    test_json = "test.json"
    
    # Check if files exist
    if not os.path.exists(test_image) or not os.path.exists(test_json):
        test_image = "174.png"
        test_json = "174.json"
    
    # Run evaluation
    score, ui_elements = evaluate_ui(test_image, test_json)
    
    if score is not None:
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
        
        # Visualize the UI with bounding boxes
        OUTPUT_DIR = "./result/evaluation_simple"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load and display the image
        import cv2
        original_img = cv2.imread(test_image)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(original_img)
        
        # Add bounding boxes for UI elements
        from matplotlib.patches import Rectangle
        for element in ui_elements:
            if 'size' in element and len(element['size']) == 4:
                x, y, width, height = element['size']
                
                # Color code by element type
                color = 'blue'  # default
                if element['type'] == 'Text Button':
                    color = 'green'
                elif element['type'] == 'Text':
                    color = 'red'
                elif element['type'] == 'Image' or element['type'] == 'Icon':
                    color = 'orange'
                elif element['type'] == 'Toolbar' or element['type'] == 'Drawer':
                    color = 'purple'
                
                rect = Rectangle((x, y), width-x, height-y, 
                                linewidth=2, edgecolor=color, facecolor='none')
                plt.gca().add_patch(rect)
                
                # Add element type label
                plt.text(x, y-5, element['type'], color=color, fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        plt.title(f"UI Elements Detection - {len(ui_elements)} elements")
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        base_filename = os.path.splitext(os.path.basename(test_image))[0]
        output_file = f"{OUTPUT_DIR}/{base_filename}_elements.png"
        plt.savefig(output_file)
        plt.close()
        
        print(f"\nVisualization saved to {output_file}")
    else:
        print("Evaluation failed.") 