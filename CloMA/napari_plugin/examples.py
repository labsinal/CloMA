"""
Simple beginner examples for CloMA napari plugin
"""

import napari
import numpy as np
from pathlib import Path


# =============================================================================
# EXAMPLE 1: Simplest possible usage
# =============================================================================

def example_1_basic():
    """
    Most basic example - open napari with CloMA tools
    """
    print("Opening napari with CloMA tools...")
    print("\nSteps:")
    print("1. Open an image (File > Open)")
    print("2. Look for 'CloMA Tools' dock on the right")
    print("3. Click buttons to analyze your image")

    viewer = napari.Viewer()

    # Create a synthetic well image for demo
    print("Creating demo image...")
    img = np.zeros((512, 512), dtype=np.uint8)

    # Add some colony-like circles
    from skimage.draw import circle as draw_circle
    np.random.seed(42)
    for i in range(8):
        y = np.random.randint(100, 412)
        x = np.random.randint(100, 412)
        radius = np.random.randint(30, 60)
        rr, cc = draw_circle(y, x, radius)
        img[rr, cc] = np.random.randint(180, 255)

    # Add some noise
    img += np.random.randint(0, 20, img.shape)

    # Add to viewer
    viewer.add_image(img, name='demo_well')

    # Add CloMA widget
    from CloMA.napari_plugin import CloMAWidget
    widget = CloMAWidget(viewer)
    viewer.window.add_dock_widget(widget.native, area='right', name='CloMA Tools')

    print("✓ Demo image loaded!")
    print("\nTry clicking 'Run Complete Pipeline' to analyze it")

    napari.run()


# =============================================================================
# EXAMPLE 2: Step-by-step analysis
# =============================================================================

def example_2_step_by_step():
    """
    Step-by-step analysis with annotations
    """
    print("Step-by-step analysis example\n")

    viewer = napari.Viewer()

    # Create demo image
    print("1. Creating sample image...")
    img = np.random.randint(60, 100, (512, 512), dtype=np.uint8)
    from skimage.draw import circle as draw_circle
    np.random.seed(42)
    for _ in range(5):
        y, x = np.random.randint(80, 432, 2)
        r = np.random.randint(35, 70)
        rr, cc = draw_circle(y, x, r)
        img[rr, cc] = np.random.randint(150, 255)

    viewer.add_image(img, name='Original')
    print("   ✓ Image loaded\n")

    # Add widget
    from CloMA.napari_plugin import CloMAWidget
    widget = CloMAWidget(viewer)
    viewer.window.add_dock_widget(widget.native, area='right', name='CloMA Tools')

    print("2. Widget ready! In napari, you can:")
    print("   a) Click image layer to select it")
    print("   b) Click 'Preprocess Image' to enhance")
    print("   c) Click 'Segment Colonies' to detect")
    print("   d) Click 'Extract Features' to analyze\n")
    print("3. Or click 'Run Complete Pipeline' for all steps at once!\n")
    print("4. Results appear as new layers in the viewer")
    print("5. Feature table printed to console\n")

    napari.run()


# =============================================================================
# EXAMPLE 3: Programmatic usage without GUI
# =============================================================================

def example_3_programmatic():
    """
    Use CloMA functions directly without napari GUI
    """
    print("Programmatic analysis example\n")

    from CloMA.extras.preprocess import preprocess_images
    from CloMA.segmentation import segment_well_colonies_hybrid
    from CloMA.extras import filter_border_colonies
    from CloMA.feature_extraction import extract_features
    from math import floor
    import cv2

    # Create synthetic image
    print("1. Creating sample image...")
    img = np.random.randint(60, 100, (512, 512), dtype=np.uint8)
    from skimage.draw import circle as draw_circle
    np.random.seed(42)
    for _ in range(5):
        y, x = np.random.randint(80, 432, 2)
        r = np.random.randint(35, 70)
        rr, cc = draw_circle(y, x, r)
        img[rr, cc] = np.random.randint(150, 255)

    print("   Image size:", img.shape)
    print("   ✓ Image ready\n")

    # Step 1: Preprocess
    print("2. Preprocessing...")
    preprocessed = preprocess_images(img)
    print("   ✓ Preprocessed\n")

    # Step 2: Segment
    print("3. Segmenting colonies...")
    radius = img.shape[0] / 2
    segmentation = segment_well_colonies_hybrid(
        image=img,
        radius=radius,
        shrink=0.05
    )
    print("   Unique labels:", len(np.unique(segmentation)) - 1, "colonies")
    print("   ✓ Segmented\n")

    # Step 3: Filter borders
    print("4. Filtering border colonies...")
    segmentation = filter_border_colonies(segmentation, radius=floor(radius * 0.95))
    print("   Unique labels after filter:", len(np.unique(segmentation)) - 1)
    print("   ✓ Filtered\n")

    # Step 4: Extract features
    print("5. Extracting features...")
    features = extract_features(segmentation=segmentation, image=img)
    print("   ✓ Features extracted\n")

    # Show results
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Colonies found: {len(features)}")
    print(f"\nFeature statistics:")
    print(features[['area', 'perimeter', 'eccentricity', 'solidity']].describe())
    print("="*60)

    return features


# =============================================================================
# EXAMPLE 4: Load your own image
# =============================================================================

def example_4_your_image():
    """
    Load and analyze your own image
    """
    # Change this to your image path!
    image_path = "path/to/your/well_image.tiff"

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("\nChange this line in the code:")
        print(f'    image_path = "path/to/your/well_image.tiff"')
        return

    print(f"Loading image: {image_path}\n")

    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Could not load image")
        return

    viewer = napari.Viewer()
    viewer.add_image(img, name='Your Image')

    from CloMA.napari_plugin import CloMAWidget
    widget = CloMAWidget(viewer)
    viewer.window.add_dock_widget(widget.native, area='right')

    print("Image loaded! Use the CloMA Tools to analyze it.")
    napari.run()


# =============================================================================
# EXAMPLE 5: Batch processing multiple images
# =============================================================================

def example_5_batch_processing():
    """
    Process multiple images from a folder
    """
    from CloMA.extras.preprocess import preprocess_images
    from CloMA.segmentation import segment_well_colonies_hybrid
    from CloMA.extras import filter_border_colonies
    from CloMA.feature_extraction import extract_features
    from math import floor

    print("Batch processing example\n")

    # Change this folder path!
    image_folder = "path/to/well_images"

    if not Path(image_folder).exists():
        print(f"❌ Folder not found: {image_folder}")
        print("\nTo use this example:")
        print("1. Change image_folder to your folder")
        print("2. Put .tiff or .tif images in that folder")
        print("3. Run this example again")
        return

    import cv2

    image_files = list(Path(image_folder).glob("*.tif*"))
    if not image_files:
        print("❌ No .tif files found in folder")
        return

    print(f"Found {len(image_files)} images to process\n")

    results = {}

    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {image_path.name}...")

        # Load
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Process
        preprocessed = preprocess_images(img)
        radius = img.shape[0] / 2
        seg = segment_well_colonies_hybrid(image=img, radius=radius, shrink=0.05)
        seg = filter_border_colonies(seg, radius=floor(radius * 0.95))
        features = extract_features(segmentation=seg, image=img)

        results[image_path.name] = {
            'features': features,
            'colonies_count': len(features),
            'mean_area': features['area'].mean() if len(features) > 0 else 0,
        }

        print(f"    ✓ Found {len(features)} colonies")

    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    for filename, result in results.items():
        print(f"{filename}: {result['colonies_count']} colonies (mean area: {result['mean_area']:.0f})")

    # Export combined results
    output_file = Path(image_folder) / "batch_results.csv"
    for filename, result in results.items():
        result['features']['image'] = filename
    all_features = pd.concat([r['features'] for r in results.values()], ignore_index=True)
    all_features.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    print("="*60)


# =============================================================================
# Main menu
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CloMA Napari Plugin - Examples")
    print("="*60)
    print("\nChoose an example:")
    print("1. Basic usage (simplest)")
    print("2. Step-by-step with annotations")
    print("3. Programmatic (no GUI)")
    print("4. Load your own image")
    print("5. Batch process multiple images")
    print("\nOr run directly from code:")
    print("  from examples import example_3_programmatic")
    print("  features = example_3_programmatic()")
    print("="*60 + "\n")

    try:
        choice = input("Enter example number (1-5): ").strip()

        if choice == "1":
            example_1_basic()
        elif choice == "2":
            example_2_step_by_step()
        elif choice == "3":
            features = example_3_programmatic()
        elif choice == "4":
            example_4_your_image()
        elif choice == "5":
            example_5_batch_processing()
        else:
            print("Invalid choice!")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
