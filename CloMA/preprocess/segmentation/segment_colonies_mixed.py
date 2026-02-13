"""
Module that segment colonies from well images
"""

######################################
# imports
from numpy import ndarray, copy, ogrid, where, unique
import numpy as np
from cv2 import imread, imwrite
from cv2 import createCLAHE
from cv2 import cvtColor, COLOR_BGR2GRAY
from skimage.morphology import reconstruction
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from scipy.ndimage import label
from os.path import join, basename, dirname 
from cellpose import models
from torch import cuda

######################################
# Define helper functions

def remove_small_relative(labels, fraction=0.1):
    props = regionprops(labels)
    if len(props) == 0:
        return labels

    areas = np.array([p.area for p in props])
    min_area = fraction * np.mean(areas)

    filtered = np.zeros_like(labels)
    new_id = 1
    for p in props:
        if p.area >= min_area:
            filtered[labels == p.label] = new_id
            new_id += 1

    return filtered, min_area


def segment_well_colonies_hybrid(image: ndarray,
                                 radius: int,
                                 shrink: float = 0,
                                 min_big_area: int = 1500) -> ndarray:
    """
    Hybrid segmentation:
    - Threshold segmentation defines colony masks
    - Cellpose used ONLY to split large touching colonies
    """

    # =========================
    # 1. Threshold segmentation (your good pipeline)
    # =========================

    clahe = createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    gray = clahe.apply(cvtColor(image, COLOR_BGR2GRAY))

    seed = copy(-gray)
    seed[1:-1, 1:-1] = (-gray).min()
    background = reconstruction(seed, -gray, method="dilation")

    colonies = -gray - background

    rows, cols = colonies.shape
    cy, cx = rows // 2, cols // 2
    Y, X = ogrid[:rows, :cols]

    circle_mask = (X - cx)**2 + (Y - cy)**2 <= (radius * (1 - shrink))**2

    colonies = where(circle_mask, colonies, 0)

    thresh = threshold_otsu(colonies[circle_mask])
    binary = colonies > thresh

    dilated_masks = expand_labels(binary, distance=4)
    
    labels_thresh, n_thresh = label(dilated_masks)

    labels_thresh = where(binary.astype(bool), labels_thresh, 0)

    # =========================
    # 2. Cellpose segmentation (topology only)
    # =========================

    model = models.CellposeModel(gpu=cuda.is_available())
    labels_cp, _, _ = model.eval(colonies, diameter=None)

    # =========================
    # 3. Hybrid merging logic
    # =========================

    final_labels = np.zeros_like(labels_thresh)
    current_label = 1

    for t_label in range(1, n_thresh + 1):
        region = labels_thresh == t_label
        area = region.sum()

        # ---- SMALL COLONIES: keep threshold result
        if area < min_big_area:
            final_labels[region] = current_label
            current_label += 1
            continue

        # ---- LARGE COLONIES: check Cellpose splits
        cp_inside = labels_cp * region
        cp_ids = unique(cp_inside)
        cp_ids = cp_ids[cp_ids > 0]

        # If Cellpose does NOT split → keep original
        if len(cp_ids) <= 1:
            final_labels[region] = current_label
            current_label += 1
            continue

        # If Cellpose splits → transfer topology, keep threshold shape
        for cp_id in cp_ids:
            subregion = region & (labels_cp == cp_id)
            if subregion.sum() == 0:
                continue
            final_labels[subregion] = current_label
            current_label += 1

    final_labels, min_area = remove_small_relative(final_labels)

    print(f"Filtering colonies smaller than: {min_area} px")

    return final_labels

######################################
# Define main function

def main() -> None:
    """
    Code's main function
    """
    # import argument parsing library
    from argparse import ArgumentParser

    # create parser instance
    parser = ArgumentParser(description="")

    # add arguments to parser
    parser.add_argument("-i", "--input_image",
                        action="store",
                        required=True,
                        dest="input_path",
                        help="Path to image to be segmented")
    
    parser.add_argument("-d", "--diameter",
                        action="store",
                        required=False,
                        type=int,
                        dest="diameter",
                        help="Diameter for masking")
    
    parser.add_argument("-s", "--shrink",
                        action="store",
                        required=False,
                        type=float,
                        dest="shrink",
                        help="Shring percentage for masking 1 = all masked, 0 = radius mask")
    
    parser.add_argument("-o", "--output_image",
                        action="store",
                        required=False,
                        default=None,
                        dest="output_path",
                        help="Path to save segmentation")
    
    args = parser.parse_args()

    # open image
    image = imread(args.input_path)

    radius = args.diameter / 2 if args.diameter is not None else image.shape[0] / 2
    shrink = args.shrink if args.shrink is not None else 0
    # apply segmentation algorithm
    segmentation = segment_well_colonies_hybrid(image, radius = radius, shrink = shrink)

    # save segmentation
    if args.output_path:
        filename = args.output_path
    else:
       filename = join(dirname(args.input_path), f"seg_{basename(args.input_path).split(".")[0]}.tif") 

    print(f"Saving segmentation as {filename}")
    imwrite(filename, segmentation)

######################################
# Call main function id runned directly
if __name__ == "__main__":
    main()

# end of current module
