import json

def check_consistency(ui_elements):
    # Example heuristic: Ensure buttons have consistent labels
    button_labels = [e["label"] for e in ui_elements if "BUTTON" in e["label"]]
    return len(set(button_labels)) == len(button_labels)  # Should be unique

def evaluate_ui(json_path):
    ui_elements = extract_ui_elements(json_path)
    results = {
        "consistency": check_consistency(ui_elements),
        # Add more heuristic checks here
    }
    return results
