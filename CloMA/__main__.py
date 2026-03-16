import typer
from typing import Annotated
import numpy as np
from pandas import DataFrame
from cv2 import imread, imwrite
from os.path import isdir, join, basename
from os import makedirs
from glob import glob
from math import floor

app = typer.Typer(help= "ClOMA - Clonogenic Morphometric Analysis, a tool for segmenting and extracting features from clonogenic colonies.", 
                  add_completion = False,
                  no_args_is_help=True)

@app.command()
def complete_pipeline(img : Annotated[str, typer.Option(help="Path to image to be processed")],
                      output : Annotated[str, typer.Option(help="Path to folder where pipeline result will be saved")]):
    """
    CloMA complete pipeline
    """
    from CloMA.extras.well_detection_interactive import WellDetectorGUI
    from CloMA.extras.preprocess import preprocess_images
    from CloMA.segmentation import segment_well_colonies_hybrid
    from CloMA.feature_extraction import extract_features
    from CloMA.extras import filter_border_colonies

    
    # Start with well detection
    wells_dir = join(output, "wells")
    makedirs(wells_dir, exist_ok=True)
    import tkinter as tk
    root = tk.Tk()
    app = WellDetectorGUI(root, image_path=img, output_folder=wells_dir)
    root.mainloop()

    # Create folders names to save paths
    preprocessed_folder = join(output, "preprocessed")
    segmentation_folder = join(output, "segmentations")
    features_folder = join(output, "features")

    # Create folders
    makedirs(preprocessed_folder, exist_ok=True)
    makedirs(segmentation_folder, exist_ok=True)
    makedirs(features_folder, exist_ok=True)

    # start loop for all the wells
    for well in glob(join(wells_dir, "*.tiff")):
        
        # open well image
        well_array : np.ndarray = imread(well)

        # preprocess
        print("preprocessing image...")
        preprocessed_image = preprocess_images(well_array)
        imwrite(join(preprocessed_folder, "preprocessed_" + basename(well)), preprocessed_image)

        # segment
        print("segmenting colonies...")
        segmentation = segment_well_colonies_hybrid(preprocessed_image, radius = well_array.shape[0] / 2, shrink = 0.05)
        # filter border colonies
        segmentation = filter_border_colonies(segmentation, radius = floor(well_array.shape[0] / 2 * ( 1 - 0.05 )))
        imwrite(join(segmentation_folder, "segmentation_" + basename(well)), segmentation)

        # extract features
        print("extracting features...")
        table = extract_features(segmentation=segmentation, image=well_array)
        table.to_csv(join(features_folder, "features_" + basename(well)))

        print("all done!")

@app.command()
def well_detection():
    """
    Detects wells from images interactively
    """
    from CloMA.extras.well_detection_interactive import WellDetectorGUI
    import tkinter as tk
    root = tk.Tk()
    app = WellDetectorGUI(root)
    root.mainloop()

@app.command()
def preprocess_images(img : Annotated[str, typer.Option(help="Path to image to enhance")],
                      output : Annotated[str, typer.Option(help="Path to folder or file where imade will be saved")]):
    """
    Preprocess images
    """
    from CloMA.extras.preprocess import preprocess_images

    img_array : np.ndarray = imread(img)

    processed_img : np.ndarray = preprocess_images(image=img_array)

    if isdir(output):
        imwrite(output + "preprocessed_" + basename(img), processed_img)
    else:
        imwrite(output, processed_img)

@app.command()
def segment_images(img : Annotated[str, typer.Option(help="Path to image be segmented")],
                    output : Annotated[str, typer.Option(help="Path to folder or file where segmentation masks will be saved")],
                    radius : Annotated[int, typer.Option(help="Radius to mask segmentation")] = None,
                    shrink : Annotated[float, typer.Option(help="Percentage to shrink mask")] = 0.1,
                    filter_border : Annotated[bool, typer.Option(help="Remove colonies in the borders")] = True):
    """
    Segment colonies from images
    """
    from CloMA.segmentation import segment_well_colonies_hybrid

    img_array : np.ndarray = imread(img)

    radius : float | None = img_array.shape[0] / 2 if radius == None else radius

    segmentation : np.ndarray = segment_well_colonies_hybrid(image = img_array,
                                          radius = radius,
                                          shrink = shrink)

    if filter_border:
        from CloMA.extras import filter_border_colonies
        segmentation = filter_border_colonies(segmentation, floor(radius))

    if isdir(output):
        imwrite(output + "segmentation_" + basename(img), segmentation)
    else:
        imwrite(output, segmentation)

@app.command()
def extract_features(seg : Annotated[str, typer.Option(help="Path to segmentation masks to extract features")],
                      output : Annotated[str, typer.Option(help="Path to folder or file where imade will be saved")],
                      img : Annotated[str, typer.Option(help="Path to image for colour related features")] = None):
    """
    Extract features from segmentation masks
    """
    from CloMA.feature_extraction import extract_features
    import tifffile

    seg_array : np.ndarray = tifffile.imread(seg)
    img_array : np.ndarray | None = imread(img) if img is not None else None

    table : DataFrame = extract_features(segmentation=seg_array, image= img_array)
    
    if isdir(output):
        table.to_csv(output + "features_" + basename(img))
    else:
        table.to_csv(output)


if __name__ == "__main__":
    app()