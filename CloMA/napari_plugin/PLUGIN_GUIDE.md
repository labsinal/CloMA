```
CloMA Napari Plugin: Complete Documentation
============================================

## What This Plugin Does

This napari plugin integrates ALL of your CloMA functionality into napari's interactive image viewer.
Instead of using command-line tools and file I/O, you work with image layers directly.

## Features at a Glance

✓ Image Preprocessing      - CLAHE + background subtraction
✓ Colony Segmentation      - Hybrid threshold + Cellpose method
✓ Border Filtering         - Remove edge colonies automatically  
✓ Feature Extraction       - 20+ morphological features per colony
✓ Complete Pipeline        - All steps in one click
✓ Interactive Controls     - Adjust parameters in real-time
✓ Results Storage          - Features stored in viewer for export

## File Structure

napari-plugin/
├── __init__.py                 # Plugin entry point & initialization
├── _cloma_widget.py            # Basic widget implementation
├── _cloma_advanced_widget.py   # Feature-rich widget (used by default)
├── napari.yaml                 # Napari plugin manifest
├── setup.py                    # Installation configuration (setuptools)
├── pyproject.toml              # Modern Python package config (alternative)
├── install.py                  # Installation helper script
├── README.md                   # Full documentation
├── QUICKSTART.md              # Quick reference guide
├── examples.py                # 5 ready-to-run examples
├── example_usage.py           # More detailed code examples
├── test_cloma_widget.py       # Unit tests
└── example_with_well_detection.py  # (To be created) Well detection example

## Installation

### Quick Install

```bash
cd /path/to/CloMA/napari-plugin
pip install -e .
```

### Verification

```python
# In Python or notebook:
import napari
from CloMA.napari_plugin import CloMAWidget
print("✓ Plugin installed successfully!")
```

### Start napari

```bash
napari
```

Then: File > Open > choose your well image

In the dock panel on the right, you should see "CloMA Tools"

## Key Components Explained

### 1. __init__.py
- Defines plugin entry points for napari
- Imports from _cloma_advanced_widget (with fallback to _cloma_widget)
- Exports CloMAWidget for public use

### 2. _cloma_widget.py
- Basic widget implementation
- Uses magicgui for UI
- Fallback if advanced version fails

### 3. _cloma_advanced_widget.py (RECOMMENDED)
- Enhanced widget with better error handling
- More polished UI with status messages
- Proper formatting and status updates
- This is what loads by default

### 4. napari.yaml & setup.py & pyproject.toml
- Plugin discovery & installation configuration
- napari.yaml: napari manifest format
- setup.py: Traditional setuptools config
- pyproject.toml: Modern PEP 517/518 config

### 5. install.py
- Helper script to verify installation
- Run: python napari-plugin/install.py

### 6. README.md
- Complete documentation
- Features explained in detail
- Parameter descriptions
- Troubleshooting guide

### 7. QUICKSTART.md
- Quick reference guide
- Basic workflow instructions
- Common tasks & solutions

### 8. examples.py
- 5 self-contained examples:
  1. Basic usage
  2. Step-by-step analysis
  3. Programmatic (no GUI)
  4. Load your own image
  5. Batch processing
- Run with: python examples.py

### 9. example_usage.py
- More detailed code examples
- Advanced usage patterns
- Python-only usage

### 10. test_cloma_widget.py
- Unit tests for the widget
- Run with: pytest test_cloma_widget.py

## Usage Workflows

### Workflow 1: Interactive (Most Common)

```
1. Open napari with your image
2. See CloMA Tools in right dock
3. Click tools to process image
4. New layers created automatically
5. Check console for feature results
```

### Workflow 2: Complete Pipeline

```
1. Select your image layer
2. Click "Run Complete Pipeline"
3. Set parameters (radius, shrink)
4. Wait for completion
5. Results in console + viewer.features_table
```

### Workflow 3: Programmatic

```python
from CloMA.napari_plugin import CloMAWidget
from napari.layers import Image
import napari

viewer = napari.Viewer()
image = napari.io.imread('well.tiff')
viewer.add_image(image, name='well')

widget = CloMAWidget(viewer)
widget._run_pipeline(radius_factor=0.5, shrink=0.05)

features = viewer.features_table  # Access results
```

## Key Parameters

### Radius Factor (0.1 - 1.0)
- Controls detection radius: radius = image_height/2 * factor
- 0.5 = 50% of image height
- Higher = larger detection area

### Border Shrink (0.0 - 0.2)  
- Additional shrinking of border exclusion zone
- 0.05 = 5% additional shrink
- Only used if "Filter Border Colonies" is ON

### Filter Border Colonies (ON/OFF)
- Removes colonies touching the image edge
- Recommended: ON for well images
- Uses circular valid area based on radius & shrink

## Output: Feature Table

After extraction, the feature table contains:

**Morphological Features:**
- label, area, perimeter, eccentricity, solidity
- axis_major_length, axis_minor_length
- extent, feret_diameter_max, equivalent_diameter_area
- euler_number, num_pixels
- area_bbox, area_convex, area_filled
- centroid (y, x), orientation

**Intensity Features** (if image provided):
- intensity_mean, intensity_min, intensity_max, intensity_std
- One set per color channel if multi-channel

**Access Results:**

```python
# In napari Python console:
features = viewer.cloma_features  # or viewer.features_table

# Basic stats
print(f"Colonies: {len(features)}")
print(f"Mean area: {features['area'].mean():.0f}")

# Export
features.to_csv('results.csv', index=False)

# Full statistics
print(features.describe())
```

## Common Tasks

### Task: Analyze one well image
1. Open napari: napari
2. File > Open well_image.tiff
3. Run Complete Pipeline
4. Results in console

### Task: Process multiple wells
- Run examples.py > select example 5 (Batch processing)
- Edit image_folder path
- Run script

### Task: Fine-tune parameters
1. Select your image
2. Try different Radius / Shrink values
3. Check preview layers
4. Find best parameters
5. Run full analysis

### Task: Export results for statistics
```python
# In napari console:
features = viewer.cloma_features
features.to_csv('my_results.csv')
features.to_excel('my_results.xlsx')
```

### Task: Compare multiple images
```python
import pandas as pd

results = []
for img_path in [img1, img2, img3, ...]:
    # Process each image
    features = process_image(img_path)
    features['image'] = img_path
    results.append(features)

combined = pd.concat(results)
combined.to_csv('combined_results.csv')
```

## Troubleshooting

### "Plugin not found in napari"
```bash
cd /path/to/napari-plugin
pip install -e .
# Restart napari
```

### "No image layer found"
- File > Open (load an image)
- Click image in layers panel
- Ensure it's an Image layer (not Labels)

### "Module not found: CloMA"
- Ensure main CloMA package is installed
- From project root: pip install -e .

### Slow performance
- Large images take time
- Consider preprocessing first
- Check GPU support: python -c "import torch; print(torch.cuda.is_available())"

### Memory issues  
- Cellpose uses GPU memory
- Clear unused layers
- Reduce image size if needed
- Restart napari if memory accumulates

### Feature table not showing
1. Check console output
2. Segmentation created?
3. Run Extract Features again

## Advanced Usage

### Custom preprocessing
```python
from CloMA.extras.preprocess import preprocess_images
img = load_your_image()
preprocessed = preprocess_images(img)
viewer.add_image(preprocessed)
```

### Custom segmentation
```python
from CloMA.segmentation import segment_well_colonies_hybrid
seg = segment_well_colonies_hybrid(image, radius=200, shrink=0.05)
viewer.add_labels(seg)
```

### Feature extraction on custom segmentation
```python
from CloMA.feature_extraction import extract_features
features = extract_features(your_segmentation, your_image)
features.to_csv('results.csv')
```

## For Developers

### Modifying the widget

Edit _cloma_advanced_widget.py:
- UI code in _build_ui()
- Processing code in _preprocess_image(), etc.
- Changes reload without reinstall

### Running tests

```bash
pytest -v test_cloma_widget.py
```

### Creating custom UI

Add to container in _build_ui():
```python
new_button = PushButton(text="My Tool")
new_button.clicked.connect(my_function)
container.append(new_button)
```

## Summary

This plugin makes CloMA analysis interactive and user-friendly:
- No command-line needed
- Visual feedback with layers
- Parameter adjustment in real-time
- Results immediately accessible
- Export to CSV/Excel

Happy analyzing! 🔬
````
