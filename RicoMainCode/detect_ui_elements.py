import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import pytesseract
from PIL import Image
import argparse

# Load AI models for detection
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    DETECTRON_AVAILABLE = True
except ImportError:
    DETECTRON_AVAILABLE = False
    print("Detectron2 not available, falling back to basic OpenCV detection")

# Set up constants for output
DEFAULT_OUTPUT_DIR = "./detected_json"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

def setup_detectron_model():
    """Setup Detectron2 model for object detection if available"""
    if not DETECTRON_AVAILABLE:
        return None
        
    # Set up detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for object detection
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    return predictor

def detect_text_regions(image):
    """Detect text regions using Tesseract OCR"""
    # Convert to PIL image for tesseract
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Use pytesseract to get text regions
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
    
    text_elements = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        # Filter out empty text
        if int(data['conf'][i]) > 60 and data['text'][i].strip() != '':
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text = data['text'][i]
            
            # Create text element
            text_element = {
                "type": "Text",
                "text": text,
                "bounds": [x, y, x + w, y + h],
                "clickable": False,
                "class": "android.support.v7.widget.AppCompatTextView",
                "ancestors": ["android.widget.TextView", "android.view.View", "java.lang.Object"]
            }
            text_elements.append(text_element)
    
    return text_elements

def detect_ui_elements_cv(image):
    """Detect UI elements using basic OpenCV methods"""
    elements = []
    height, width = image.shape[:2]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours and convert to UI elements
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out very small contours (likely noise)
        if w < 20 or h < 20:
            continue
            
        # Determine element type based on aspect ratio and size
        aspect_ratio = w / h
        element_type = "Image"  # Default type
        
        if aspect_ratio > 4:  # Very wide
            element_type = "Toolbar"
        elif aspect_ratio < 0.25:  # Very tall
            element_type = "Drawer"
        elif w < 50 and h < 50:
            element_type = "Icon"
        
        # Create UI element
        element = {
            "type": element_type,
            "bounds": [x, y, x+w, y+h],
            "clickable": True if element_type in ["Icon", "Image"] else False,
            "class": "android.widget.ImageView" if element_type in ["Icon", "Image"] else "android.view.ViewGroup"
        }
        
        # Add ancestors
        if element_type in ["Icon", "Image"]:
            element["ancestors"] = ["android.widget.ImageView", "android.view.View", "java.lang.Object"]
        else:
            element["ancestors"] = ["android.view.ViewGroup", "android.view.View", "java.lang.Object"]
        
        elements.append(element)
    
    return elements

def detect_ui_elements_detectron(image, predictor):
    """Detect UI elements using Detectron2 (if available)"""
    if predictor is None:
        return []
        
    # Get predictions
    outputs = predictor(image)
    
    # Get detected instances
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    elements = []
    
    # Map COCO classes to UI element types 
    coco_to_ui = {
        0: "Icon",  # person
        1: "Image", # bicycle
        2: "Image", # car
        # ... map other relevant COCO classes
        62: "Text Button", # tv
        73: "Icon", # book
    }
    
    for box, class_id, score in zip(boxes, classes, scores):
        if score < 0.7:
            continue
            
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Get element type
        element_type = coco_to_ui.get(int(class_id), "Image")
        
        # Create UI element
        element = {
            "type": element_type,
            "bounds": [x1, y1, x2, y2],
            "clickable": element_type in ["Icon", "Text Button"],
            "class": "android.widget.ImageView" if element_type == "Image" else "android.widget.Button" if element_type == "Text Button" else "android.support.v7.widget.AppCompatImageView"
        }
        
        elements.append(element)
    
    return elements

def identify_clickable_elements(image, elements):
    """Attempt to identify which elements are likely clickable"""
    height, width = image.shape[:2]
    
    for element in elements:
        x1, y1, x2, y2 = element["bounds"]
        
        # Extract element image
        element_img = image[y1:y2, x1:x2]
        if element_img.size == 0:
            continue
            
        # Simple heuristic: elements with button-like appearance are clickable
        # Button-like: rounded corners or distinct color
        
        # Check if the element has a distinct color from surroundings
        try:
            element_hsv = cv2.cvtColor(element_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(element_hsv)
            
            # High saturation elements are likely buttons
            if np.mean(s) > 100:
                element["clickable"] = True
                element["type"] = "Text Button" if element["type"] == "Text" else element["type"]
        except:
            pass
            
    return elements

def create_json_structure(image_path, elements):
    """Create the final JSON structure with detected UI elements"""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Create the base JSON structure
    ui_json = {
        "class": "com.android.internal.policy.PhoneWindow$DecorView",
        "bounds": [0, 0, width, height],
        "clickable": False,
        "ancestors": [
            "android.widget.FrameLayout",
            "android.view.ViewGroup", 
            "android.view.View", 
            "java.lang.Object"
        ],
        "children": elements
    }
    
    return ui_json

def extract_ui_elements_from_image(image_path, output_json_path=None):
    """Main function to extract UI elements from an image and save as JSON"""
    print(f"Processing image: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
        
    # Setup detectron model if available
    detectron_predictor = setup_detectron_model() if DETECTRON_AVAILABLE else None
    
    # Detect UI elements using available methods
    elements = []
    
    # Use Detectron2 if available
    if detectron_predictor:
        print("Using Detectron2 for UI element detection")
        elements += detect_ui_elements_detectron(image, detectron_predictor)
    
    # Always run CV detection for additional elements
    print("Using OpenCV for UI element detection")
    elements += detect_ui_elements_cv(image)
    
    # Detect text elements
    print("Detecting text elements")
    elements += detect_text_regions(image)
    
    # Identify clickable elements
    elements = identify_clickable_elements(image, elements)
    
    # Create final JSON structure
    ui_json = create_json_structure(image_path, elements)
    
    # If no output path is specified, create one based on input
    if output_json_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_json_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}.json")
    
    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(ui_json, f, indent=2)
    
    print(f"Detected {len(elements)} UI elements")
    print(f"JSON saved to: {output_json_path}")
    
    # Calculate complexity metrics
    complexity = {
        'element_count': len(elements),
        'clickable_count': sum(1 for e in elements if e.get('clickable', False)),
        'text_count': sum(1 for e in elements if e.get('type') == 'Text'),
        'max_depth': 1,  # Simple detection doesn't handle hierarchy depth
        'screen_width': image.shape[1],
        'screen_height': image.shape[0],
        'density': len(elements) / (image.shape[0] * image.shape[1]) * 1000000
    }
    
    return elements, complexity, output_json_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect UI elements from a screenshot and generate JSON")
    parser.add_argument("image_path", help="Path to the screenshot image")
    parser.add_argument("--output", help="Path for output JSON file (optional)")
    
    args = parser.parse_args()
    
    # Extract UI elements
    elements, complexity, json_path = extract_ui_elements_from_image(args.image_path, args.output)
    
    # Display summary
    print("\n===== UI ELEMENT DETECTION SUMMARY =====")
    print(f"Total elements: {complexity['element_count']}")
    print(f"Clickable elements: {complexity['clickable_count']}")
    print(f"Text elements: {complexity['text_count']}")
    
    # Count elements by type
    element_types = {}
    for element in elements:
        element_type = element['type']
        if element_type in element_types:
            element_types[element_type] += 1
        else:
            element_types[element_type] = 1
    
    print("\nElement types:")
    for element_type, count in element_types.items():
        print(f"  - {element_type}: {count}")
    
    print(f"\nOutput JSON: {json_path}")
    print("You can now use this JSON with run_evaluation.py") 