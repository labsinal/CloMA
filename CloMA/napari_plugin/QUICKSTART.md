# Quick Start Guide for CloMA Napari Plugin

## Installation

### Method 1: Install from Source (Recommended)

```bash
cd /path/to/CloMA/napari-plugin
pip install -e .
```

This installs the plugin in development mode, so any changes to the code are reflected immediately.

### Method 2: Install with pip (from release)

```bash
pip install cloma-napari
```

### Method 3: Load Directly in Python

```python
import napari
from CloMA.napari_plugin import CloMAWidget

viewer = napari.Viewer()
widget = CloMAWidget(viewer)
viewer.window.add_dock_widget(widget.native, area='right', name='CloMA Tools')
napari.run()
```

## Basic Workflow

### 1. Start napari with your image

```bash
napari /path/to/your/well_image.tiff
```

Or in Python:
```python
import napari
viewer = napari.Viewer()
image = napari.io.imread('/path/to/your/well_image.tiff')
viewer.add_image(image, name='well')
napari.run()
```

### 2. Click "CloMA Tools" dock widget on the right

The widget shows all available tools.

### 3. Process your image

**Option A: Step by step**
1. Click "Preprocess Image" → creates `preprocessed_well` layer
2. Click on preprocessed image to make it active
3. Click "Segment Colonies" → creates `segmentation_well` layer
4. Click "Extract Features" → prints results, stores in `viewer.cloma_features`

**Option B: Complete pipeline**
1. Click "Run Complete Pipeline" → does all steps automatically
2. Check console for results
3. Access feature table: `viewer.cloma_features`

## Advanced Usage

### Access Results Programmatically

```python
# After running extraction
features = viewer.cloma_features
print(f"Found {len(features)} colonies")
print(features.describe())

# Export to CSV
features.to_csv('results.csv', index=False)
```

### Use in Batch Processing

```python
import napari
from CloMA.napari_plugin import CloMAWidget
from pathlib import Path
import cv2

# Process multiple images
for image_path in Path('images/').glob('*.tiff'):
    viewer = napari.Viewer()
    image = cv2.imread(str(image_path))
    viewer.add_image(image, name='image')
    
    widget = CloMAWidget(viewer)
    widget._run_pipeline(radius_factor=0.5, shrink=0.05)
    
    # Save results
    features = viewer.cloma_features
    features.to_csv(f'{image_path.stem}_features.csv')
    
    viewer.close()
```

### Parameter Tuning

**Radius Factor** (0.1 - 1.0)
- Controls the detection radius
- 0.5 = 50% of image height
- Lower values exclude edge colonies

**Border Shrink** (0.0 - 0.2)
- Additional shrinking of border exclusion zone
- 0.05 = 5% shrink
- Higher values = more border exclusion

**Filter Border Colonies**
- Toggle to remove colonies touching image edge
- Recommended: ON for well images

## Troubleshooting

### Plugin doesn't appear in napari

1. Check installation:
   ```bash
   pip list | grep cloma
   ```

2. Reinstall:
   ```bash
   pip install -e /path/to/napari-plugin
   ```

3. Check if napari can find it:
   ```python
   import napari
   napari.plugins.discover()
   ```

### "No image layer found"

- Open an image: File > Open
- Ensure it's in the layers panel
- Click on the image layer to make it active

### Slow performance

- Processing large images takes time
- Consider reducing image size
- Use GPU if available (check CUDA in console)

### Memory issues with Cellpose

- Reduce image size
- Clear layers you don't need
- Restart napari if memory grows

## Features Extracted

The plugin extracts 20+ features for each colony:

**Geometric:**
- area, perimeter, eccentricity, solidity
- axis_major_length, axis_minor_length
- extent, feret_diameter_max
- equivalent_diameter_area

**Connectivity:**
- euler_number, num_pixels
- area_bbox, area_convex, area_filled

**Intensity** (if image provided):
- intensity_mean, intensity_min, intensity_max, intensity_std
- (for each color channel if available)

## Examples

See `example_usage.py` for complete examples:

```bash
cd /path/to/napari-plugin
python example_usage.py
```

## Getting Help

1. Check the README.md for detailed documentation
2. Review example_usage.py for code examples
3. Check console output for error messages
4. Enable debug mode: Set `DEBUG=1` before running

## Contributing

Found a bug or have a feature request?
- Check existing issues on GitHub
- Create a new issue with:
  - Steps to reproduce
  - Expected vs actual behavior
  - Your napari/Python versions
  - Console error messages

