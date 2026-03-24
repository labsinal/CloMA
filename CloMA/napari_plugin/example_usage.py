"""
Example script showing how to use the CloMA napari plugin
"""

import napari
from napari.utils import nbscreenshot
import numpy as np
from pathlib import Path


def example_load_and_analyze():
    """
    Example (Load) an image and run the complete CloMA pipeline
    """
    # Create viewer
    viewer = napari.Viewer()

    # Option 1: Load your own image
    image_path = "path/to/your/well_image.tiff"  # Change this to your image
    if Path(image_path).exists():
        image = napari.io.imread(image_path)
        viewer.add_image(image, name="original")
    else:
        print("⚠️ No image provided. Using synthetic image for demo...")
        # Create synthetic image for demonstration
        from skimage.draw import circle
        img = np.zeros((512, 512), dtype=np.uint8)
        for _ in range(5):
            y, x = np.random.randint(100, 412, 2)
            rr, cc = circle(y, x, np.random.randint(30, 60))
            img[rr, cc] = np.random.randint(100, 255)
        viewer.add_image(img, name="synthetic_well")

    # Add the CloMA widget
    from CloMA.napari_plugin import CloMAWidget
    cloma_widget = CloMAWidget(viewer)
    viewer.window.add_dock_widget(cloma_widget.native, area="right", name="CloMA Tools")

    # Start the viewer
    napari.run()


def example_programmatic_pipeline():
    """
    Example (Run) the pipeline programmatically without GUI
    """
    import cv2
    from CloMA.extras.preprocess import preprocess_images
    from CloMA.segmentation import segment_well_colonies_hybrid
    from CloMA.extras import filter_border_colonies
    from CloMA.feature_extraction import extract_features
    from math import floor

    # Load image
    image_path = "path/to/your/well_image.tiff"  # Change this
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    print("Processing image...")

    # Step 1: Preprocess
    print("1. Preprocessing...")
    preprocessed = preprocess_images(image)

    # Step 2: Segment
    print("2. Segmenting...")
    radius = image.shape[0] / 2
    shrink = 0.05
    segmentation = segment_well_colonies_hybrid(
        image=image,
        radius=radius,
        shrink=shrink
    )

    # Step 3: Filter border colonies
    print("3. Filtering border colonies...")
    segmentation = filter_border_colonies(
        segmentation,
        radius=floor(radius * (1 - shrink))
    )

    # Step 4: Extract features
    print("4. Extracting features...")
    features = extract_features(segmentation=segmentation, image=image)

    print(f"\n✓ Analysis complete!")
    print(f"Found {len(features)} colonies")
    print("\nFeature summary:")
    print(features[["area", "perimeter", "eccentricity", "solidity"]].describe())

    return features


def example_interactive_with_viewer():
    """
    Example (Load) image and use napari + plugin interactively
    """
    import cv2

    # Create viewer
    viewer = napari.Viewer()

    # Load an example image
    image_path = "path/to/your/well_image.tiff"  # Change this
    if Path(image_path).exists():
        image = cv2.imread(image_path)
        viewer.add_image(image, name="well_image")
    else:
        print("Creating demo image...")
        # Demo image
        img = np.random.randint(50, 150, (512, 512, 3), dtype=np.uint8)
        # Add some circles
        from skimage.draw import circle as draw_circle
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for _ in range(3):
            y, x = np.random.randint(100, 412, 2)
            r = np.random.randint(40, 80)
            rr, cc = draw_circle(y, x, r)
            gray[rr, cc] = 255
        viewer.add_image(img, name="well_image")

    # Add CloMA widget
    from CloMA.napari_plugin import CloMAWidget
    cloma_widget = CloMAWidget(viewer)
    viewer.window.add_dock_widget(cloma_widget.native, area="right", name="CloMA Tools")

    # Now you can:
    # 1. Click on the image to make it active
    # 2. Click "Preprocess Image" to preprocess
    # 3. Click "Segment Colonies" to segment
    # 4. Click "Extract Features" to analyze
    # 5. Check the console for feature table

    print("\n" + "="*50)
    print("CloMA Napari Plugin Interactive Example")
    print("="*50)
    print("\nInstructions:")
    print("1. Image loaded in 'well_image' layer")
    print("2. Use the CloMA Tools panel on the right")
    print("3. Click tools in order:")
    print("   - Preprocess Image (optional but recommended)")
    print("   - Segment Colonies (creates segmentation layer)")
    print("   - Extract Features (shows analysis results)")
    print("\nOr use 'Run Complete Pipeline' to do all steps at once!")
    print("\nResults:")
    print("- New layers created for each step")
    print("- Feature table printed to console")
    print("- Features stored in viewer.features_table")
    print("="*50)

    napari.run()


if __name__ == "__main__":
    # Run interactive example
    example_interactive_with_viewer()

    # Uncomment to run other examples:
    # example_programmatic_pipeline()
    # example_load_and_analyze()
