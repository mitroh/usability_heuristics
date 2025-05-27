# UI Element Detector and Evaluator

This tool allows you to analyze website screenshots, detect UI elements, and evaluate the usability of the UI.

## Overview

The system consists of two main components:

1. **UI Element Detector** - Uses computer vision and OCR to identify UI elements from screenshots
2. **UI Evaluator** - Analyzes detected elements and provides a usability score

## Installation

### Dependencies

The tool requires the following dependencies:

```bash
pip install opencv-python numpy tensorflow pytesseract pillow
```

For OCR functionality, you'll need to install Tesseract:

- **macOS**: `brew install tesseract`
- **Linux**: `apt-get install tesseract-ocr`
- **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

For best results, Detectron2 is recommended but optional:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Usage

### Option 1: Complete Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
python evaluate_screenshot.py path/to/your/screenshot.png
```

This will:
1. Detect UI elements in the screenshot
2. Generate a JSON file with element information
3. Run the evaluation on the detected elements
4. Display the usability score and element summary
5. Save visualization to the results directory

#### Additional Options

- `--output_dir DIR` - Specify a custom directory for results
- `--skip_detection` - Skip UI detection if JSON file already exists

### Option 2: Step-by-Step Approach

#### Step 1: Detect UI Elements

```bash
python detect_ui_elements.py path/to/your/screenshot.png
```

This creates a JSON file in the `detected_json` directory.

#### Step 2: Run Evaluation

```bash
python run_evaluation.py path/to/your/screenshot.png path/to/generated.json
```

## Understanding the Results

The evaluation provides:

1. **Usability Score** - A score from 0-10 based on UI elements and their arrangement
2. **Element Summary** - Breakdown of detected UI element types
3. **Visualization** - Image with bounding boxes showing detected elements

## Example

```bash
python evaluate_screenshot.py website_home.png
```

Output:
```
===== STEP 1: DETECTING UI ELEMENTS =====
Processing image: website_home.png
Using OpenCV for UI element detection
Detecting text elements
Detected 42 UI elements
JSON saved to: ./result/screenshot_eval/website_home.json

===== STEP 2: EVALUATING UI =====
Model loaded from ./models/ui_evaluation_model.h5
Using standard model

===== UI EVALUATION RESULTS =====
Overall Usability Score: 7.5 / 10

Detected 42 UI elements

UI Element Types:
  - Text: 18
  - Image: 12
  - Icon: 8
  - Text Button: 4

Visualization will be saved to the results directory
```

## JSON Format

The generated JSON follows the structure:

```json
{
  "class": "...",
  "bounds": [0, 0, width, height],
  "clickable": false,
  "ancestors": [...],
  "children": [
    {
      "type": "Text",
      "text": "Example Text",
      "bounds": [x, y, width, height],
      "clickable": false,
      "class": "..."
    },
    ...
  ]
}
```

## Limitations

1. Element detection accuracy depends on image quality
2. Hierarchy detection is basic and may not capture all relationships
3. Some specialized UI elements may be misclassified

## Troubleshooting

1. **Poor detection results**: Try improving screenshot quality or resolution
2. **OCR issues**: Ensure Tesseract is properly installed
3. **Missing elements**: Adjust thresholds in `detect_ui_elements.py` 