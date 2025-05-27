import os
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from preprocess import process_image, extract_ui_elements, extract_features
from tensorflow.keras.models import load_model

# Create output directories
OUTPUT_DIR = "./result"
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "evaluation_graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Load the model
try:
    # Try to load the model
    MODEL_PATH = "./models/ui_evaluation_model.h5"
    # Use custom objects for metrics
    custom_objects = {
        'mse': tf.keras.metrics.MeanSquaredError(),
        'mae': tf.keras.metrics.MeanAbsoluteError()
    }
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print(f"Model loaded from {MODEL_PATH}")
    
    # Check if it's the hybrid model by inspecting input shape
    is_hybrid_model = isinstance(model.input, list)
    print(f"Using {'hybrid' if is_hybrid_model else 'standard'} model")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def extract_ui_elements_for_evaluation(json_path):
    """
    Extract UI elements from a JSON file and return a list of elements with their properties.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Recursive function to traverse the JSON structure
        def traverse_children(children):
            elements = []
            for child in children:
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
                    'icon_class': child.get('iconClass', '')
                }
                elements.append(element)
                # Recursively traverse any children
                if 'children' in child:
                    elements.extend(traverse_children(child['children']))
            return elements
        
        # Start traversal from the top-level children
        ui_elements = traverse_children(data.get('children', []))
        return ui_elements
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

def nielsen_heuristic_evaluation(ui_elements):
    """
    Comprehensive evaluation based on Nielsen's 10 Usability Heuristics
    Returns evaluation results and specific issues found for each heuristic
    """
    heuristics_results = {}
    issues_found = {}
    
    # 1. Visibility of System Status
    def check_visibility_of_status(ui_elements):
        """
        Check if system provides appropriate feedback about its status.
        """
        issues = []
        
        # Check for progress indicators, status messages
        has_progress_indicator = any('progress' in element['type'].lower() for element in ui_elements)
        if not has_progress_indicator:
            issues.append("No progress indicators found")
            
        has_status_text = any('status' in element.get('text', '').lower() for element in ui_elements)
        if not has_status_text:
            issues.append("No status messages found")
            
        # Check for loading indicators
        has_loading = any('load' in element.get('text', '').lower() for element in ui_elements)
        if not has_loading and not has_progress_indicator:
            issues.append("No loading indicators found")
        
        # Check for visual feedback elements (like highlighted states)
        visual_feedback_terms = ['selected', 'active', 'current', 'highlighted']
        has_visual_feedback = any(any(term in element.get('text', '').lower() for term in visual_feedback_terms) 
                               for element in ui_elements)
        
        if not has_visual_feedback:
            issues.append("No visual feedback indicators (selected/active states)")
        
        result = has_progress_indicator or has_status_text or has_loading or has_visual_feedback
        return result, issues
    
    # 2. Match Between System and Real World
    def check_system_real_world_match(ui_elements):
        """
        Check if system uses familiar language and follows real-world conventions.
        """
        issues = []
        
        # Check for jargon or technical terms in texts
        jargon_terms = ['backend', 'frontend', 'api', 'json', 'html', 'css', 'runtime', 'compiler', 
                        'exception', 'syntax', 'backend', 'runtime', 'query', 'parameter']
        
        jargon_found = []
        for element in ui_elements:
            if element.get('text'):
                for term in jargon_terms:
                    if term in element['text'].lower():
                        jargon_found.append(term)
        
        if jargon_found:
            issues.append(f"Technical jargon found: {', '.join(set(jargon_found))}")
        
        # Check if icons represent real-world objects
        common_icons = ['home', 'star', 'search', 'user', 'cart', 'heart', 'mail', 'trash', 'settings']
        familiar_icons = [element.get('icon_class', '').lower() for element in ui_elements 
                         if element.get('icon_class', '').lower() in common_icons]
        
        if not familiar_icons:
            issues.append("No familiar icons found (home, search, etc.)")
        
        # Check for natural language in UI text
        has_human_text = False
        for element in ui_elements:
            if element.get('text') and len(element.get('text', '')) > 20:  # Long enough to be a sentence
                has_human_text = True
                break
        
        if not has_human_text:
            issues.append("No natural language text found in UI")
            
        result = (len(jargon_found) == 0 and (len(familiar_icons) > 0 or has_human_text))
        return result, issues
    
    # 3. User Control and Freedom
    def check_user_control(ui_elements):
        """
        Check if users have control and freedom to undo/exit actions.
        """
        issues = []
        
        # Look for back buttons, cancel options, undo features
        has_back = any('back' in element.get('text', '').lower() for element in ui_elements)
        if not has_back:
            issues.append("No back button found")
            
        has_cancel = any('cancel' in element.get('text', '').lower() for element in ui_elements)
        if not has_cancel:
            issues.append("No cancel option found")
            
        has_exit = any('exit' in element.get('text', '').lower() or 'close' in element.get('text', '').lower() 
                     for element in ui_elements)
        if not has_exit:
            issues.append("No exit or close option found")
        
        has_undo = any('undo' in element.get('text', '').lower() for element in ui_elements)
        if not has_undo:
            issues.append("No undo functionality found")
        
        # Check for navigation elements (menus, tabs)
        has_nav = any('nav' in element['type'].lower() for element in ui_elements)
        if not has_nav:
            issues.append("No navigation elements found")
        
        # Check if there's a home button or link
        has_home = any('home' in element.get('text', '').lower() or
                     element.get('icon_class', '').lower() == 'home' for element in ui_elements)
        if not has_home:
            issues.append("No home button/link found")
        
        result = (has_back or has_cancel or has_exit or has_undo) and (has_nav or has_home)
        return result, issues
    
    # 4. Consistency and Standards
    def check_consistency(ui_elements):
        """
        Check if UI elements have consistent sizes, styles, and alignments.
        """
        issues = []
        
        # Check button size consistency
        button_elements = [element for element in ui_elements if element['type'] == 'Text Button']
        if len(button_elements) > 1:
            button_heights = [el['size'][3] - el['size'][1] for el in button_elements]
            height_variation = max(button_heights) - min(button_heights) if button_heights else 0
            
            if height_variation > 10:
                issues.append(f"Inconsistent button heights (variation: {height_variation}px)")
            
            # Check horizontal alignment
            button_x_positions = [el['position'][0] for el in button_elements]
            unique_positions = len(set(button_x_positions))
            
            if unique_positions > 3:
                issues.append(f"Buttons not aligned consistently ({unique_positions} different x-positions)")
        
        # Check text alignment
        text_elements = [element for element in ui_elements if element['type'] == 'Text']
        if len(text_elements) > 1:
            text_x_positions = [el['position'][0] for el in text_elements]
            unique_text_positions = len(set(text_x_positions))
            
            if unique_text_positions > 3:
                issues.append(f"Text elements not aligned consistently ({unique_text_positions} different alignments)")
        
        # Check icon consistency
        icon_elements = [element for element in ui_elements if element['type'] == 'Icon']
        if len(icon_elements) > 1:
            icon_sizes = [(el['size'][2] - el['size'][0]) * (el['size'][3] - el['size'][1]) for el in icon_elements]
            if max(icon_sizes) / max(1, min(icon_sizes)) > 2:
                issues.append("Inconsistent icon sizes (more than 2x variation)")
        
        # Check for UI patterns that follow platform conventions
        has_standard_ui = any(e['type'] in ['Toolbar', 'Drawer', 'Tab', 'Dialog'] for e in ui_elements)
        if not has_standard_ui:
            issues.append("No standard UI components found (toolbar, drawer, tabs, etc.)")
        
        result = len(issues) <= 1 or has_standard_ui  # Allow up to 1 minor issue if standard UI is present
        return result, issues
    
    # 5. Error Prevention
    def check_error_prevention(ui_elements):
        """
        Check for features that prevent errors.
        """
        issues = []
        
        # Look for confirmation dialogs
        has_confirmation = any('confirm' in element.get('text', '').lower() for element in ui_elements)
        if not has_confirmation:
            issues.append("No confirmation dialogs found")
        
        # Check for input validation hints
        has_validation = any('valid' in element.get('text', '').lower() for element in ui_elements)
        validation_related = ['required', 'must be', 'please enter', 'format']
        has_validation_hint = any(any(hint in element.get('text', '').lower() for hint in validation_related) 
                                 for element in ui_elements)
        
        if not has_validation and not has_validation_hint:
            issues.append("No input validation or hints found")
        
        # Check for potentially destructive actions
        destructive_actions = ['delete', 'remove', 'clear']
        has_destructive = any(any(action in element.get('text', '').lower() for action in destructive_actions) 
                             for element in ui_elements)
        
        if has_destructive and not has_confirmation:
            issues.append("Destructive actions without confirmation prompts")
        
        # Look for form elements with constraints/guidance
        has_form_guidance = any('hint' in element.get('text', '').lower() for element in ui_elements)
        
        if not has_form_guidance and any(e['type'] in ['EditText', 'TextBox', 'Input'] for e in ui_elements):
            issues.append("Form fields without guidance")
        
        result = (has_confirmation or has_validation or has_validation_hint or has_form_guidance) and not (has_destructive and not has_confirmation)
        return result, issues
    
    # 6. Recognition Rather Than Recall
    def check_recognition(ui_elements):
        """
        Check if interface elements are visible and don't require users to remember information.
        """
        issues = []
        
        # Check for visible labels on interactive elements
        buttons = [element for element in ui_elements if element['type'] == 'Text Button']
        buttons_with_labels = [element for element in buttons if element.get('text')]
        
        labeled_ratio = 1.0  # Default if no buttons
        if len(buttons) > 0:
            labeled_ratio = len(buttons_with_labels) / len(buttons)
            if labeled_ratio < 0.7:
                issues.append(f"Only {int(labeled_ratio * 100)}% of buttons have text labels")
        
        # Check for tooltips, hints
        has_tooltips = any('tooltip' in element['type'].lower() for element in ui_elements)
        if not has_tooltips:
            issues.append("No tooltips found")
        
        # Check for menu visibility
        menu_items = [element for element in ui_elements if 'menu' in element['type'].lower()]
        if len(menu_items) > 0 and all(not element.get('text') for element in menu_items):
            issues.append("Menu items without text labels")
        
        # Check for visible navigation and hierarchy
        has_clear_nav = any(e['type'] in ['Toolbar', 'Tab', 'Navigation'] for e in ui_elements)
        if not has_clear_nav:
            issues.append("No clear navigation structure")
        
        result = (len(buttons) == 0 or labeled_ratio >= 0.7) and (has_tooltips or has_clear_nav)
        return result, issues
    
    # 7. Flexibility and Efficiency of Use
    def check_flexibility(ui_elements):
        """
        Check if UI caters to both novice and expert users.
        """
        issues = []
        
        # Look for shortcuts, customization options
        has_shortcuts = any('shortcut' in element.get('text', '').lower() for element in ui_elements)
        if not has_shortcuts:
            issues.append("No keyboard shortcuts found")
        
        has_customization = any('custom' in element.get('text', '').lower() or 'setting' in element.get('text', '').lower() 
                               for element in ui_elements)
        if not has_customization:
            issues.append("No customization options found")
        
        # Check for advanced features
        has_advanced_features = any('advanced' in element.get('text', '').lower() for element in ui_elements)
        if not has_advanced_features:
            issues.append("No advanced features for expert users")
        
        # Check for search functionality
        has_search = any('search' in element.get('text', '').lower() or element.get('icon_class', '').lower() == 'search' 
                        for element in ui_elements)
        if not has_search:
            issues.append("No search functionality found")
        
        # Check for personalization options
        has_personalization = any('profile' in element.get('text', '').lower() or 'account' in element.get('text', '').lower() 
                                 for element in ui_elements)
        if not has_personalization:
            issues.append("No user personalization options")
        
        result = has_shortcuts or has_customization or has_advanced_features or has_search or has_personalization
        return result, issues
    
    # 8. Aesthetic and Minimalist Design
    def check_aesthetic_design(ui_elements):
        """
        Check if design is clean and minimalist.
        """
        issues = []
        
        # Count elements to check for clutter
        element_count = len(ui_elements)
        if element_count > 30:
            issues.append(f"Interface may be cluttered ({element_count} elements found)")
        
        # Check for whitespace by analyzing element density
        if len(ui_elements) > 0:
            bounds = [element['size'] for element in ui_elements if 'size' in element and len(element['size']) == 4]
            if bounds:
                # Calculate total screen area and element areas
                screen_area = max([b[2] for b in bounds]) * max([b[3] for b in bounds])
                element_areas = sum([(b[2] - b[0]) * (b[3] - b[1]) for b in bounds])
                density = element_areas / max(1, screen_area)
                
                if density > 0.7:
                    issues.append(f"High element density ({density:.2f}), insufficient whitespace")
        
        # Check for redundant information
        text_contents = [element.get('text', '').lower() for element in ui_elements if element.get('text')]
        duplicate_texts = set([text for text in text_contents if text and text_contents.count(text) > 1])
        if duplicate_texts:
            issues.append(f"Duplicate text content found: {', '.join(list(duplicate_texts)[:3])}...")
        
        # Check text length - too verbose is not minimalist
        long_texts = [t for t in text_contents if len(t) > 100]
        if long_texts:
            issues.append(f"Found {len(long_texts)} overly verbose text elements")
        
        result = element_count <= 30 and (len(bounds) == 0 or density <= 0.7) and not duplicate_texts and not long_texts
        return result, issues
    
    # 9. Help Users Recognize, Diagnose, and Recover from Errors
    def check_error_handling(ui_elements):
        """
        Check for clear error messages and recovery options.
        """
        issues = []
        
        # Look for error messages
        error_texts = ['error', 'warning', 'failed', 'invalid', 'incorrect']
        error_elements = [element for element in ui_elements 
                         if element.get('text') and any(err in element.get('text', '').lower() for err in error_texts)]
        
        # Check if error messages explain the problem
        if error_elements:
            short_errors = [e for e in error_elements if len(e.get('text', '')) < 15]
            if short_errors:
                issues.append("Error messages too brief to be helpful")
            
            # Check for error codes
            has_error_codes = any(any(c.isdigit() for c in e.get('text', '')) and 
                                sum(c.isdigit() for c in e.get('text', '')) > 3 for e in error_elements)
            if has_error_codes:
                issues.append("Error messages contain error codes")
        else:
            issues.append("No error messages found")
        
        # Check for recovery suggestions
        recovery_terms = ['try', 'fix', 'recover', 'solution', 'help']
        has_recovery_options = any(any(term in element.get('text', '').lower() for term in recovery_terms) 
                                 for element in ui_elements)
        if not has_recovery_options:
            issues.append("No error recovery suggestions found")
        
        # Check for clear visual indication of errors
        has_error_visuals = any(element['type'] == 'Icon' and element.get('icon_class', '').lower() in ['error', 'warning'] 
                              for element in ui_elements)
        if not has_error_visuals and error_elements:
            issues.append("No visual indicators for errors (icons, color coding)")
        
        result = ((len(error_elements) > 0 and len(short_errors) == 0 and not has_error_codes) or 
                 has_recovery_options or has_error_visuals)
        return result, issues
    
    # 10. Help and Documentation
    def check_help_documentation(ui_elements):
        """
        Check for help resources and documentation.
        """
        issues = []
        
        # Look for help buttons, documentation links, tutorials
        help_terms = ['help', 'guide', 'tutorial', 'documentation', 'faq', 'support', 'learn', 'how to']
        help_elements = [element for element in ui_elements 
                        if element.get('text') and any(term in element.get('text', '').lower() for term in help_terms)]
        
        if not help_elements:
            issues.append("No help or documentation options found")
        
        # Check for contextual help
        contextual_help = any('?' in element.get('text', '') for element in ui_elements)
        if not contextual_help:
            issues.append("No contextual help indicators (e.g., '?') found")
        
        # Check for guided tutorials or onboarding
        has_tutorial = any('tutorial' in element.get('text', '').lower() or 'guide' in element.get('text', '').lower() 
                          for element in ui_elements)
        if not has_tutorial:
            issues.append("No tutorials or guided help found")
        
        # Check for easy access to help (usually in menu/footer)
        help_in_standard_location = False
        for element in ui_elements:
            if (element.get('text') and any(term in element.get('text', '').lower() for term in help_terms) and
                (element['type'] == 'Text Button' or element.get('parent_type') in ['Menu', 'Footer', 'Navigation'])):
                help_in_standard_location = True
                break
                
        if not help_in_standard_location:
            issues.append("Help not found in standard locations (menu, footer)")
        
        result = len(help_elements) > 0 or contextual_help or has_tutorial
        return result, issues
    
    # Evaluate all 10 heuristics
    visibility_result, visibility_issues = check_visibility_of_status(ui_elements)
    real_world_result, real_world_issues = check_system_real_world_match(ui_elements)
    user_control_result, user_control_issues = check_user_control(ui_elements)
    consistency_result, consistency_issues = check_consistency(ui_elements)
    error_prevention_result, error_prevention_issues = check_error_prevention(ui_elements)
    recognition_result, recognition_issues = check_recognition(ui_elements)
    flexibility_result, flexibility_issues = check_flexibility(ui_elements)
    aesthetic_result, aesthetic_issues = check_aesthetic_design(ui_elements)
    error_handling_result, error_handling_issues = check_error_handling(ui_elements)
    help_result, help_issues = check_help_documentation(ui_elements)
    
    # Store results and issues
    heuristics_results = {
        "1_visibility_of_status": visibility_result,
        "2_system_real_world_match": real_world_result,
        "3_user_control": user_control_result,
        "4_consistency": consistency_result,
        "5_error_prevention": error_prevention_result,
        "6_recognition_not_recall": recognition_result,
        "7_flexibility": flexibility_result,
        "8_aesthetic_design": aesthetic_result,
        "9_error_handling": error_handling_result,
        "10_help_documentation": help_result
    }
    
    issues_found = {
        "1_visibility_of_status": visibility_issues,
        "2_system_real_world_match": real_world_issues,
        "3_user_control": user_control_issues,
        "4_consistency": consistency_issues,
        "5_error_prevention": error_prevention_issues,
        "6_recognition_not_recall": recognition_issues,
        "7_flexibility": flexibility_issues,
        "8_aesthetic_design": aesthetic_issues,
        "9_error_handling": error_handling_issues,
        "10_help_documentation": help_issues
    }
    
    # Calculate overall score
    passed_heuristics = sum(1 for result in heuristics_results.values() if result)
    total_heuristics = len(heuristics_results)
    overall_score = (passed_heuristics / total_heuristics) * 10
    
    return heuristics_results, issues_found, overall_score

def predict_ui(image_path, json_path):
    """
    Runs the trained model on a test UI image and evaluates all 10 Nielsen's heuristics.
    """
    # Process the image
    image = process_image(image_path)
    if image is None:
        print(f"Error: Unable to process image at {image_path}")
        return None, None, None
    
    # Extract UI elements from JSON
    ui_elements, complexity = extract_ui_elements(json_path)
    elements_for_eval = extract_ui_elements_for_evaluation(json_path)
    if not ui_elements:
        print(f"Error: Unable to extract UI elements from {json_path}")
        return None, None, None
    
    # Prepare model input
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
    
    # Run heuristic evaluation
    heuristics_results, issues_found, heuristic_score = nielsen_heuristic_evaluation(elements_for_eval)
    
    # Add UI elements to the results
    heuristics_results['ui_elements'] = elements_for_eval
    
    # Combine model prediction with heuristic score
    if prediction is not None:
        model_score = float(prediction[0][0])
        # Weighted average: 40% model prediction, 60% heuristic evaluation
        final_score = 0.4 * model_score + 0.6 * heuristic_score
    else:
        final_score = heuristic_score
    
    # Limit score to 0-10 range
    final_score = max(0, min(10, final_score))
    
    return final_score, heuristics_results, issues_found

def generate_evaluation_graphs(image_path, score, heuristics, issues, output_dir=GRAPHS_DIR):
    """
    Generate comprehensive visualization graphs for UI evaluation results.
    Shows specific issues identified for each heuristic.
    
    Args:
        image_path: Path to the original UI image
        score: Model prediction score
        heuristics: Dictionary containing heuristic evaluation results
        issues: Dictionary containing identified issues for each heuristic
        output_dir: Directory to save the generated graphs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Set up the style
    plt.style.use('ggplot')
    sns.set_style("whitegrid")
    
    # 1. UI Elements Visualization
    try:
        # Load the original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Error: Unable to load image at {image_path}")
            return
        
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(original_img)
        
        # Add bounding boxes for UI elements
        if 'ui_elements' in heuristics:
            for element in heuristics['ui_elements']:
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
        
        plt.title(f"UI Elements Detection - {len(heuristics.get('ui_elements', []))} elements")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{base_filename}_elements.png")
        plt.close()
        
        # 2. Heuristic Evaluation Bar Chart
        if heuristics:
            # Create a figure with 2 subplots (top: bar chart, bottom: issue details)
            fig = plt.figure(figsize=(14, 10))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
            
            # Bar chart - upper subplot
            ax1 = plt.subplot(gs[0])
            
            # Extract heuristic metrics (excluding ui_elements)
            metrics = {k: v for k, v in heuristics.items() if k != 'ui_elements'}
            
            # Format labels for readability
            formatted_labels = []
            for key in metrics.keys():
                # Remove numbers and underscores, capitalize words
                parts = key.split('_')[1:]  # Skip the leading number
                formatted = ' '.join(part.capitalize() for part in parts)
                formatted_labels.append(formatted)
            
            # Create colormap based on pass/fail
            colors = ['green' if v else 'red' for v in metrics.values()]
            
            # Plot bar chart
            bars = ax1.bar(formatted_labels, [1 if v else 0 for v in metrics.values()], color=colors)
            
            # Add score on each bar
            for bar, val in zip(bars, metrics.values()):
                height = 0.5
                ax1.text(bar.get_x() + bar.get_width()/2., height, 
                        'Pass' if val else 'Fail', ha='center', va='center', 
                        color='white', fontweight='bold')
            
            ax1.set_ylim(0, 1.2)
            ax1.set_title(f"Heuristic Evaluation Results - Overall Score: {score:.1f}/10")
            ax1.set_ylabel("Status")
            ax1.set_xticklabels(formatted_labels, rotation=45, ha='right')
            
            # Issue details - lower subplot
            ax2 = plt.subplot(gs[1])
            ax2.axis('off')
            
            # Create a table showing issues for each heuristic
            table_data = []
            for i, (key, result) in enumerate(metrics.items()):
                # Get corresponding issues
                heuristic_issues = issues.get(key, [])
                
                # Format heuristic name
                parts = key.split('_')[1:]  # Skip the leading number
                heuristic_name = ' '.join(part.capitalize() for part in parts)
                
                status = "✅ PASS" if result else "❌ FAIL"
                issue_text = '\n'.join(heuristic_issues) if heuristic_issues else "No issues found"
                
                # Add a row for this heuristic
                table_data.append([f"{i+1}. {heuristic_name}", status, issue_text])
            
            # Create table
            table = ax2.table(cellText=table_data, 
                            colLabels=["Heuristic", "Status", "Issues Identified"],
                            colWidths=[0.3, 0.1, 0.6],
                            loc='center',
                            cellLoc='left')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Color code the status cells
            for i, (key, result) in enumerate(metrics.items()):
                table[(i+1, 1)].set_facecolor('lightgreen' if result else 'lightcoral')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{base_filename}_heuristics.png")
            plt.close()
            
        # 3. Element Type Distribution Pie Chart
        if 'ui_elements' in heuristics:
            element_types = [element['type'] for element in heuristics['ui_elements']]
            type_counts = {}
            for element_type in element_types:
                if element_type in type_counts:
                    type_counts[element_type] += 1
                else:
                    type_counts[element_type] = 1
            
            # Filter out types with very small counts for clarity
            total_elements = sum(type_counts.values())
            significant_types = {k: v for k, v in type_counts.items() if v / total_elements >= 0.03}
            other_count = sum(v for k, v in type_counts.items() if v / total_elements < 0.03)
            
            if other_count > 0:
                significant_types['Other'] = other_count
            
            plt.figure(figsize=(10, 8))
            plt.pie(significant_types.values(), labels=significant_types.keys(), autopct='%1.1f%%', 
                   shadow=True, startangle=90, explode=[0.05] * len(significant_types))
            plt.axis('equal')
            plt.title(f"UI Element Type Distribution - {len(heuristics['ui_elements'])} total elements")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{base_filename}_element_types.png")
            plt.close()
            
        # 4. Heuristic Radar Chart
        if heuristics:
            metrics = {k: v for k, v in heuristics.items() if k != 'ui_elements'}
            
            # Format labels for readability
            categories = []
            for key in metrics.keys():
                # Extract just the number and name (e.g., "1. Visibility")
                num = key.split('_')[0]
                name = ' '.join(key.split('_')[1:3])  # Take first two parts of the name
                categories.append(f"{num}. {name.capitalize()}")
            
            # Convert boolean values to numeric
            values = [1 if v else 0 for v in metrics.values()]
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of categories
            N = len(categories)
            
            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add values
            values += values[:1]  # Close the loop
            
            # Draw the chart
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            
            # Fix axis to go in the right order and start at 12 o'clock
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw axis lines for each angle and label
            plt.xticks(angles[:-1], categories)
            
            # Draw y-axis labels (0 and 1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Fail', 'Pass'])
            ax.set_ylim(0, 1)
            
            # Add title
            plt.title(f"Nielsen's Heuristics Evaluation - Score: {score:.1f}/10", size=15, y=1.1)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{base_filename}_radar_chart.png")
            plt.close()
            
        # 5. Issue Categories Chart
        if issues:
            # Count issues by heuristic
            issue_counts = {k: len(v) for k, v in issues.items()}
            
            # Format labels
            labels = [f"{k.split('_')[0]}. {' '.join(k.split('_')[1:3])}" for k in issue_counts.keys()]
            counts = list(issue_counts.values())
            
            # Sort by count
            sorted_data = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
            labels = [x[0] for x in sorted_data]
            counts = [x[1] for x in sorted_data]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(labels, counts, color=sns.color_palette("coolwarm", len(labels)))
            
            # Add the number at the top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.title('Number of Issues by Heuristic Category')
            plt.xlabel('Nielsen\'s Heuristic')
            plt.ylabel('Issue Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{base_filename}_issue_counts.png")
            plt.close()
        
        print(f"Evaluation graphs saved to {output_dir}")
        
    except Exception as e:
        print(f"Error generating evaluation graphs: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    import random
    import glob
    
    # Path to semantic annotations directory
    ANNOTATIONS_DIR = "./data/semantic_annotations"
    
    if os.path.exists(ANNOTATIONS_DIR):
        # Get all PNG files in the directory
        png_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.png"))
        
        if png_files:
            # Randomly select one PNG file
            random_png = random.choice(png_files)
            # Get the corresponding JSON file
            random_json = random_png.replace(".png", ".json")
            
            if os.path.exists(random_json):
                test_image = random_png
                test_json = random_json
                print(f"Randomly selected: {os.path.basename(test_image)}")
            else:
                # Fallback to default test files
                test_image = "test.png"
                test_json = "test.json"
                print("Corresponding JSON not found, using default test files.")
        else:
            # No PNG files found, use default test files
            test_image = "test.png"
            test_json = "test.json"
            print("No PNG files found in annotations directory, using default test files.")
    else:
        # Annotations directory doesn't exist, use default test files
        test_image = "test.png"
        test_json = "test.json"
        print(f"Annotations directory {ANNOTATIONS_DIR} not found, using default test files.")
    
    # Check if default files exist
    if (test_image == "test.png" and not os.path.exists(test_image)) or (test_json == "test.json" and not os.path.exists(test_json)):
        test_image = "174.png"
        test_json = "174.json"
    
    # Run evaluation
    score, heuristics, issues = predict_ui(test_image, test_json)
    
    print(f"\n===== UI EVALUATION RESULTS =====")
    print(f"Overall Usability Score: {score:.1f} / 10\n")
    
    print("Nielsen's 10 Usability Heuristics Evaluation:")
    print("---------------------------------------------")
    
    # Format and print results
    heuristic_names = {
        "1_visibility_of_status": "Visibility of System Status",
        "2_system_real_world_match": "Match Between System and Real World",
        "3_user_control": "User Control and Freedom",
        "4_consistency": "Consistency and Standards",
        "5_error_prevention": "Error Prevention",
        "6_recognition_not_recall": "Recognition Rather Than Recall",
        "7_flexibility": "Flexibility and Efficiency of Use",
        "8_aesthetic_design": "Aesthetic and Minimalist Design",
        "9_error_handling": "Help Users Recognize, Diagnose, and Recover from Errors",
        "10_help_documentation": "Help and Documentation"
    }
    
    for key, name in heuristic_names.items():
        if key in heuristics:
            status = "PASS" if heuristics[key] else "FAIL"
            print(f"{key.split('_')[0]}. {name}: {status}")
            
            if key in issues:
                for issue in issues[key]:
                    print(f"   - {issue}")
            print()
    
    # Generate evaluation graphs
    generate_evaluation_graphs(test_image, score, heuristics, issues)
    
    print(f"\nEvaluation complete! Graphs saved to {GRAPHS_DIR}/")
    print(f"Open {GRAPHS_DIR}/{os.path.splitext(os.path.basename(test_image))[0]}_heuristics.png to see detailed results")

