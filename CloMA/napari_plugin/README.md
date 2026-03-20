# CloMA Napari Plugin

A napari plugin that integrates CloMA (Clonogenic Morphometric Analysis) functionality into napari, enabling interactive image segmentation and feature extraction of clonogenic colonies.

## Features

The plugin provides the following tools integrated into napari:

### 1. **Preprocess Image**
   - Applies CLAHE filtering and background subtraction
   - Takes an image layer and creates a new preprocessed layer
   - Input: Active image layer
   - Output: New preprocessed image layer

### 2. **Segment Colonies**
   - Hybrid segmentation using threshold-based and Cellpose methods
   - Automatically splits large touching colonies
   - Optionally filters border colonies
   - Parameters:
     - **Radius Shrink Factor** (0-1): Controls which colonies to keep at the border
     - **Shrink Percentage** (0-0.3): Percentage to reduce the valid detection radius
     - **Filter Border Colonies**: Remove colonies touching the image border
   - Input: Active image layer
   - Output: New labels layer with segmentation masks

### 3. **Extract Features**
   - Extracts 20+ morphological and intensity features from segmentation
   - Features include: area, perimeter, eccentricity, solidity, intensity statistics, etc.
   - Input: Segmentation (labels layer) and image layer
   - Output: Feature table stored in viewer, printed to console

### 4. **Complete Pipeline**
   - Runs all steps: preprocessing → segmentation → filtering → feature extraction
   - Creates intermediate layers for each step
   - Parameters same as individual tools
   - Input: Active image layer
   - Output: Multiple layers + feature table

## Installation

### Option 1: Install as napari plugin

```bash
# From the plugin directory
cd /path/to/CloMA/napari-plugin
pip install -e .
```

Then access it in napari via Plugins > CloMA > CloMA Tools

### Option 2: Load directly in napari

```python
import napari
from CloMA.napari_plugin import create_cloma_widget

viewer = napari.Viewer()
viewer.window.add_dock_widget(create_cloma_widget(viewer), area='right')
napari.run()
```

## Usage Workflow

### Basic Usage

1. **Open an image** in napari (File > Open)
2. **Select the active layer** by clicking on it
3. **Choose a tool** from the CloMA dock widget:
   - Click "Preprocess Image" to enhance the image
   - Click "Segment Colonies" to detect colonies
   - Click "Extract Features" to analyze detected colonies

### Advanced: Complete Pipeline

1. Open your image
2. Click "Run Complete Pipeline"
3. Set parameters:
   - Adjust "Radius Shrink Factor" to control border exclusion (lower = more border colonies)
   - Adjust "Shrink Percentage" for finer control
4. Results:
   - View preprocessed image in a new layer
   - View segmentation masks in a new labels layer
   - Check console for feature table summary

### Example: Analyzing Multiple Wells

For images with multiple wells, you can:
1. Use "Well Detection" (if implemented) to split the image
2. Run the pipeline on each well separately
3. Export features from the console output

## Feature Table

The extracted features include:

**Morphological Features:**
- `area`: Colony area in pixels
- `perimeter`: Colony perimeter
- `eccentricity`: Elongation (0=circle, 1=line)
- `solidity`: Convexity (ratio of area to convex hull)
- `axis_major_length`, `axis_minor_length`: Principal axes
- `equivalent_diameter_area`: Diameter of equivalent circle
- And more...

**Intensity Features** (if image provided):
- `intensity_mean`, `intensity_std`, `intensity_min`, `intensity_max`: Statistics per channel

The feature table is:
- Printed to console after extraction
- Stored in `viewer.features_table` (accessible via Python console)
- Can be accessed via: `features_df = viewer.features_table`

## Tips & Tricks

- **Adjust radius shrink factor**: Lower values include more colonies at the border; higher values exclude them
- **Filter border colonies**: Toggle on to remove colonies not completely within the circular well
- **Use preprocessing**: Better results by preprocessing first, then segmenting the preprocessed image
- **View intermediate results**: Each step creates a new layer you can inspect
- **Export features**: Access the feature table via Python console: `viewer.features_table.to_csv('features.csv')`

## Troubleshooting

**"No image layer found"**
- Check that you have an image open
- Click on the image layer to make it active
- Ensure it's an Image layer (not Labels)

**"No segmentation layer found"** (when extracting features)
- First run segmentation to create a labels layer
- The plugin looks for Labels layers

**Slow performance**
- Large images may take time for Cellpose segmentation
- Consider preprocessing first for better results
- Reduce image size if needed

## Parameters Explained

### Radius Shrink Factor
- Controls the valid detection circle radius
- Formula: `effective_radius = image_height/2 * shrink_factor`
- **0.5** (default): Uses 50% of the image height as radius
- Useful for excluding edge colonies

### Shrink Percentage
- Reduces the border exclusion radius further
- Used when `Filter Border Colonies` is enabled
- **0.05** (default): Shrinks the valid area by 5%
- Use for fine-tuning border exclusion

## Architecture

```
CloMA/
└── napari-plugin/
    ├── __init__.py                 # Plugin entry points
    ├── _cloma_widget.py            # Main widget implementation
    ├── napari.yaml                 # Napari manifest
    ├── setup.py                    # Installation configuration
    └── README.md                   # This file
```

The plugin uses:
- `magicgui` for UI widgets
- `napari.layers` for image/labels integration
- CloMA's core functions:
  - `CloMA.extras.preprocess.preprocess_images`
  - `CloMA.segmentation.segment_well_colonies_hybrid`
  - `CloMA.extras.filter_border_colonies`
  - `CloMA.feature_extraction.extract_features`

## Future Enhancements

- [ ] Interactive well detection GUI integrated with napari
- [ ] Batch processing multiple images
- [ ] Export feature table to CSV directly from plugin
- [ ] Visualization of feature overlays on segmentation
- [ ] Real-time parameter adjustment with preview
- [ ] Support for multi-channel images

## License

Same as CloMA

## Contributing

Improvements and bug reports welcome!
