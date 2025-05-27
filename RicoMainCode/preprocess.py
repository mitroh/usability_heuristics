import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define paths
DATA_PATH = "./data/semantic_annotations"
PROCESSED_PATH = "./preprocessing/"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Function to process UI images
def process_image(image_path):
    """
    Process image for model input - includes resizing and normalization
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    # Resize to a standard size
    image = cv2.resize(image, (224, 224))  # Standard size
    image = image / 255.0  # Normalize
    return image

# Enhanced function to extract UI elements from JSON with detailed properties
def extract_ui_elements(json_path):
    """
    Extract detailed UI elements from a JSON file including hierarchy, text, and positioning
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Recursive function to traverse the JSON structure
        def traverse_children(children, depth=0, parent=None):
            elements = []
            for idx, child in enumerate(children):
                # Extract text content
                text = child.get('text', '')
                
                # Extract component type
                comp_type = child.get('componentLabel', 'unknown')
                
                element = {
                    'type': comp_type,
                    'text': text,
                    'size': child.get('bounds', [0, 0, 0, 0]),
                    'position': child.get('bounds', [0, 0, 0, 0])[:2] if 'bounds' in child else [0, 0],
                    'clickable': child.get('clickable', False),
                    'resource_id': child.get('resource-id', ''),
                    'class': child.get('class', ''),
                    'icon_class': child.get('iconClass', ''),
                    'depth': depth,
                    'index': idx,
                    'parent_type': parent['type'] if parent else None
                }
                elements.append(element)
                # Recursively traverse any children
                if 'children' in child:
                    elements.extend(traverse_children(child['children'], depth + 1, element))
            return elements
        
        # Start traversal from the top-level children
        ui_elements = traverse_children(data.get('children', []))
        
        # Extract screen size
        screen_bounds = data.get('bounds', [0, 0, 1440, 2560])  # Default size if not specified
        
        # Calculate UI density metrics
        element_count = len(ui_elements)
        ui_complexity = {
            'element_count': element_count,
            'clickable_count': sum(1 for e in ui_elements if e['clickable']),
            'text_count': sum(1 for e in ui_elements if e.get('text')),
            'max_depth': max([e['depth'] for e in ui_elements]) if ui_elements else 0,
            'screen_width': max(1, screen_bounds[2] - screen_bounds[0]),  # Ensure not zero
            'screen_height': max(1, screen_bounds[3] - screen_bounds[1])  # Ensure not zero
        }
        
        # Additional features for usability assessment - with error handling
        try:
            screen_area = ui_complexity['screen_width'] * ui_complexity['screen_height']
            if screen_area > 0:
                ui_complexity['density'] = element_count / screen_area * 1000000
            else:
                ui_complexity['density'] = 0  # Default if area is zero
        except ZeroDivisionError:
            ui_complexity['density'] = 0  # Default value if division by zero
        
        return ui_elements, ui_complexity
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return [], {'element_count': 0, 'clickable_count': 0, 'text_count': 0, 'max_depth': 0, 
                   'screen_width': 1440, 'screen_height': 2560, 'density': 0}

# Extract features for ML model
def extract_features(elements, complexity):
    """
    Extract numerical features from UI elements for machine learning model
    """
    # Calculate feature vector
    features = [
        complexity['element_count'],
        complexity['clickable_count'],
        complexity['text_count'],
        complexity['max_depth'],
        complexity['density'],
        len([e for e in elements if e['type'] == 'Text Button']),
        len([e for e in elements if e['type'] == 'Text']),
        len([e for e in elements if e['type'] == 'Image']),
        len([e for e in elements if e['type'] == 'Icon']),
        len([e for e in elements if e['clickable']]),
        # Layout metrics
        len(set([e['position'][0] for e in elements if e['type'] == 'Text'])) if elements else 0,  # Number of unique text alignments
    ]
    return np.array(features)

# Check if preprocessed data exists
if os.path.exists(f"{PROCESSED_PATH}images.npy") and os.path.exists(f"{PROCESSED_PATH}features.npy"):
    print("Preprocessed data found. Skipping preprocessing...")
else:
    images, features, elements_data, complexity_data, file_paths = [], [], [], [], []
    
    # Check if directory exists before processing
    if not os.path.exists(DATA_PATH):
        print(f"Warning: Data path {DATA_PATH} not found. Using test files instead.")
        # Use test files if data directory doesn't exist
        json_files = ["test.json", "174.json"]
        for file in json_files:
            if os.path.exists(file):
                json_path = file
                image_path = file.replace(".json", ".png")
                
                if os.path.exists(image_path):
                    img_array = process_image(image_path)
                    if img_array is not None:
                        ui_elements, complexity = extract_ui_elements(json_path)
                        feature_vector = extract_features(ui_elements, complexity)
                        
                        images.append(img_array)
                        features.append(feature_vector)
                        elements_data.append(ui_elements)
                        complexity_data.append(complexity)
                        file_paths.append(image_path)
    else:
        # Get all JSON files and use only half of them
        all_json_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]
        half_count = len(all_json_files) // 2
        print(f"Using only {half_count} files out of {len(all_json_files)} (50%) to avoid memory issues...")
        
        # Skip files known to cause errors
        error_files = []
        if os.path.exists("error_files.txt"):
            with open("error_files.txt", "r") as f:
                error_files = f.read().strip().split()
        
        selected_files = all_json_files[:half_count]
        for file in tqdm(selected_files):
            if file in error_files:
                continue
                
            json_path = os.path.join(DATA_PATH, file)
            image_path = json_path.replace(".json", ".png")

            if os.path.exists(image_path):
                try:
                    img_array = process_image(image_path)
                    if img_array is not None:
                        ui_elements, complexity = extract_ui_elements(json_path)
                        feature_vector = extract_features(ui_elements, complexity)
                        
                        images.append(img_array)
                        features.append(feature_vector)
                        elements_data.append(ui_elements)
                        complexity_data.append(complexity)
                        file_paths.append(image_path)
                except Exception as e:
                    print(f"Error processing {json_path}: {e}")
                    continue
    
    # Convert to numpy arrays
    images = np.array(images)
    features = np.array(features)
    
    # Save preprocessed data
    np.save(f"{PROCESSED_PATH}images.npy", images)
    np.save(f"{PROCESSED_PATH}features.npy", features)
    
    # Save additional data as pickle files (more complex structures)
    import pickle
    with open(f"{PROCESSED_PATH}elements_data.pkl", 'wb') as f:
        pickle.dump(elements_data, f)
    with open(f"{PROCESSED_PATH}complexity_data.pkl", 'wb') as f:
        pickle.dump(complexity_data, f)
    with open(f"{PROCESSED_PATH}file_paths.pkl", 'wb') as f:
        pickle.dump(file_paths, f)
        
    print(f"Preprocessed data saved! Total samples: {len(images)}")
    
    # Prepare train/test splits
    if len(images) > 0:
        X_img_train, X_img_test, X_feat_train, X_feat_test = train_test_split(
            images, features, test_size=0.2, random_state=42
        )
        
        # Save splits
        np.save(f"{PROCESSED_PATH}X_img_train.npy", X_img_train)
        np.save(f"{PROCESSED_PATH}X_img_test.npy", X_img_test)
        np.save(f"{PROCESSED_PATH}X_feat_train.npy", X_feat_train)
        np.save(f"{PROCESSED_PATH}X_feat_test.npy", X_feat_test)
        
        print(f"Train/test splits saved. Train: {len(X_img_train)}, Test: {len(X_img_test)}")
