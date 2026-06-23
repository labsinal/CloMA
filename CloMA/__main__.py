import os
import sys
from pathlib import Path

import typer
from typing import Annotated
import numpy as np
from pandas import DataFrame
from cv2 import imread, imwrite
from os.path import isdir, join, basename
from os import makedirs
from glob import glob

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
def complete_pipeline(
    img: Annotated[str, typer.Option(help="Path to image to be processed")],
    output: Annotated[str, typer.Option(help="Path to folder where pipeline result will be saved")],
):
    """
    Run the complete CloMA pipeline: well detection, preprocessing, segmentation, and feature extraction.
    """
    from CloMA.extras.well_detection_interactive import WellDetectorGUI
    from CloMA.extras.preprocess import preprocess_images
    from CloMA.segmentation import automatic_segmentation
    from CloMA.feature_extraction import extract_features
    from CloMA.extras import filter_border_colonies

    wells_dir = join(output, "wells")
    makedirs(wells_dir, exist_ok=True)

    import tkinter as tk

    root = tk.Tk()
    WellDetectorGUI(root, image_path=img, output_folder=wells_dir)
    root.mainloop()

    preprocessed_folder = join(output, "preprocessed")
    segmentation_folder = join(output, "segmentations")
    features_folder = join(output, "features")

    makedirs(preprocessed_folder, exist_ok=True)
    makedirs(segmentation_folder, exist_ok=True)
    makedirs(features_folder, exist_ok=True)

    for well_path in glob(join(wells_dir, "*.tiff")):
        image = _read_image(well_path)
        preprocessed_image = preprocess_images(image)
        imwrite(join(preprocessed_folder, f"preprocessed_{basename(well_path)}"), preprocessed_image)

        segmentation = automatic_segmentation(image=image)
        segmentation = filter_border_colonies(segmentation, radius=image.shape[0] // 2)
        seg_out = join(segmentation_folder, f"segmentation_{basename(well_path)}")
        if Path(seg_out).suffix.lower() in {".tif", ".tiff"}:
            import tifffile
            tifffile.imwrite(seg_out, segmentation.astype(np.uint32))
        else:
            # convert to supported type for OpenCV if necessary
            out_seg = segmentation
            if out_seg.dtype == np.uint32:
                out_seg = out_seg.astype(np.uint16)
            imwrite(seg_out, out_seg)

        table = extract_features(segmentation=segmentation, image=image)
        _save_table(table, features_folder, f"features_{basename(well_path)}")


@app.command()
def well_detection():
    """
    Detect wells from images interactively.
    """
    from CloMA.extras.well_detection_interactive import WellDetectorGUI
    import tkinter as tk

    root = tk.Tk()
    WellDetectorGUI(root)
    root.mainloop()


@app.command()
def segment_images(
    img: Annotated[str, typer.Option(help="Path to image to be segmented")],
    output: Annotated[str, typer.Option(help="Path to folder or file where segmentation masks will be saved")],
    mode: Annotated[str, typer.Option(help="Segmentation mode: automatic or reference", case_sensitive=False)] = "automatic",
    reference: Annotated[str, typer.Option(help="Path to reference label image used for reference segmentation")] = None,
    shrink: Annotated[float, typer.Option(help="Shrink fraction for circular masking")] = 0.03,
):
    """
    Segment colonies from an image using automatic or reference mode.
    Run this on the raw images, not preprocessed ones
    """
    from CloMA.segmentation import automatic_segmentation, reference_segmentation

    image = _read_image(img)

    if mode.lower() not in {"automatic", "reference"}:
        raise typer.BadParameter("mode must be 'automatic' or 'reference'")

    if mode.lower() == "reference":
        if reference is None:
            raise typer.BadParameter("reference is required for reference segmentation mode")
        import tifffile

        reference_labels = tifffile.imread(reference)
        segmentation = reference_segmentation(
            image=image,
            reference_labels=reference_labels,
            shrink=shrink,
        )
    else:
        segmentation = automatic_segmentation(image=image, shrink=shrink)

    output_path = _make_output_path(output, f"segmentation_{basename(img)}")
    if Path(output_path).suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        tifffile.imwrite(output_path, segmentation.astype(np.uint32))
    else:
        out_seg = segmentation
        if out_seg.dtype == np.uint32:
            out_seg = out_seg.astype(np.uint16)
        imwrite(output_path, out_seg)


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

    import tifffile

    seg_array = tifffile.imread(seg)
    image = _read_image(img) if img is not None else None
    table = extract_features(segmentation=seg_array, image=image)
    _save_table(table, output, f"features_{basename(seg)}")

@app.command()
def preprocess_images(
    img: Annotated[str, typer.Option(help="Path to image to enhance")],
    output: Annotated[str, typer.Option(help="Path to folder or file where image will be saved")],
):
    """
    Preprocess an image and save the result.
    """
    from CloMA.extras.preprocess import preprocess_images

    image = _read_image(img)
    processed_img = preprocess_images(image=image)
    output_path = _make_output_path(output, f"preprocessed_{basename(img)}")
    imwrite(output_path, processed_img)

if __name__ == "__main__":
    app()
