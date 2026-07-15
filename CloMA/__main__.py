import os
import sys
from pathlib import Path

import typer
from typing import Annotated
import numpy as np
from pandas import DataFrame
from cv2 import imread, imwrite
from os.path import isdir, join, basename

# Ensure the repository root is importable when running this CLI directly.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

app = typer.Typer(
    help="CloMA - Clonogenic Morphometric Analysis, a tool for segmenting and extracting features from clonogenic colonies.",
    add_completion=False,
    no_args_is_help=True,
)

def _read_image(path: str) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return tifffile.imread(path)

    image = imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _make_output_path(output: str, default_name: str) -> str:
    if isdir(output):
        return join(output, default_name)
    return output


def _save_table(table: DataFrame, output: str, default_name: str) -> None:
    output_path = _make_output_path(output, default_name)
    table.to_csv(output_path, index=False)

@app.command()
def well_detection(img: Annotated[str, typer.Option(help="Path to plate image (not used on GUI mode)")] = None,
                   output: Annotated[str, typer.Option(help="Path to save wells (not used on GUI mode)")] = None,
                   sigma: Annotated[float, typer.Option(help="Intensity of gaussian blur (not used on GUI mode)")] = None,
                   radius: Annotated[int, typer.Option(help="Well radius (not used on GUI mode)")] = None,
                   gui: Annotated[bool, typer.Option(help="Flag to activate GUI mode")] = False):
    """
    Detect wells from images interactively.
    """
    from CloMA.extras import WellDetectorGUI, detect_wells
    import tkinter as tk

    if gui:
        root = tk.Tk()
        WellDetectorGUI(root)
        root.mainloop()
        
    else:
        import cv2
        if not os.isdir(output): 
            print("output must be dir.") 
            return            
        crops = detect_wells(img, radius, sigma)
        for i, crop in enumerate(crops):
            cv2.imwrite(_make_output_path(output, f"well_{i+1:04}.tif"), crop)

@app.command()
def preprocess_images(
    img: Annotated[str, typer.Option(help="Path to image to enhance")],
    output: Annotated[str, typer.Option(help="Path to folder or file where image will be saved")],
    invert: Annotated[bool, typer.Option(help="Whether to invert or not the image")] = True
):
    """
    Preprocess an image and save the result.
    """
    from CloMA.extras.preprocess import preprocess_images

    image = _read_image(img)
    processed_img = preprocess_images(image=image, invert=invert)
    output_path = _make_output_path(output, f"preprocessed_{basename(img)}")
    imwrite(output_path, processed_img)


@app.command()
def segment_images(
    img: Annotated[str, typer.Option(help="Path to image to be segmented")],
    output: Annotated[str, typer.Option(help="Path to folder or file where segmentation masks will be saved")],
    treshold: Annotated[int, typer.Option(help="Treshold for segmentation, do not use this for automatic with multiotsu")] = None,
    circle_mask: Annotated[float, typer.Option(help="Value (0 - 1) for circular mask to define border")] = 0.97,
    reference: Annotated[str, typer.Option(help="Path to reference labels, lave blank for automatic separation")] = None,
    remove_border: Annotated[bool, typer.Option(help="Flag to remove border colonies")] = True,
    preprocess: Annotated[bool, typer.Option(help="Flag to apply preprocessing before segmenting")] = False,
    invert: Annotated[bool, typer.Option(help="Flag to invert or not during preprocessing (only used if preprocess activated)")] = True
):
    """
    Segment colonies from an image using automatic or reference mode.
    Run this on the raw images, not preprocessed ones
    """
    import CloMA.segmentation.new_segmentation as seg
    from pathlib import Path

    # open image as array
    image = _read_image(img)

    # check if image is grayscale


    # if preprocess is set to true, run preprocessment
    if preprocess:
        from CloMA.extras import preprocess_images
        image = preprocess_images(image, invert)

    # Calculate treshold
    tresh = seg.multiotsu_tresholding(image) if treshold is None else treshold

    binary = seg.apply_treshold(image, tresh)


    # Create mask
    mask = seg.create_circular_mask(image.shape, radius=image.shape[0] / 2 * circle_mask)
    binary = np.where(mask, binary, 0)


    # Separate masks into labels
    if reference is None:
        labels = seg.automatic_separation(binary)

    else:
        labels = seg.reference_separation(binary, _read_image(reference))
    
    # remove border colonies
    if remove_border:
        final_labels = seg.remove_border_colonies(labels, mask)

    # create output name    
    out_name = _make_output_path(output, f"seg_{Path(img).stem}.tif")
    # export image
    imwrite(out_name, final_labels)


@app.command()
def extract_features(
    seg: Annotated[str, typer.Option(help="Path to segmentation masks to extract features")],
    output: Annotated[str, typer.Option(help="Path to folder or file where image will be saved")],
    img: Annotated[str, typer.Option(help="Path to image for colour related features")] = None,
):
    """
    Extract features from a segmentation mask and optionally an image.
    """
    from CloMA.feature_extraction import extract_features

    seg_array = _read_image(seg)
    image = _read_image(img) if img is not None else None
    table = extract_features(segmentation=seg_array, image=image)
    _save_table(table, output, f"features_{basename(seg)}")


if __name__ == "__main__":
    app()
