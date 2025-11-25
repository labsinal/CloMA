"""
Module that segment colonies from well images
"""

######################################
# imports
from numpy import ndarray, copy, ogrid, where, ones, zeros
from cv2 import imread, imwrite
from cv2 import createCLAHE
from cv2 import cvtColor, COLOR_BGR2GRAY
from skimage.morphology import reconstruction
from skimage.filters import median
from os.path import join, basename, dirname 
from cellpose import models, io
from torch import cuda
import matplotlib.pyplot as plt

######################################
# Define helper functions

def apply_cellpose(image:ndarray) -> ndarray:
    """
    Function that applies cellpose model to segment colonies

    Args:
        image:ndarray | well image as numpy array
    Returns
    ndarray | segmented well image
    """
    use_gpu = cuda.is_available()
    # Create cellpose model instance
    model = models.CellposeModel(gpu=use_gpu)

    # Apply model to image
    masks, flows, _ = model.eval(image, diameter=None)

    return masks, flows

def segment_well_colonies(image:ndarray, output_dir:str,
                          radius:int, shrink:float) -> None:
    """
    Function that segments well colonies photo

    Args:
        image:ndarray | well image as numpy array
    Returns
    None
    """

    # Create CLAHE filter instance
    clahe = createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

    # apply CLAHE to image
    cl1 = clahe.apply(cvtColor(image, COLOR_BGR2GRAY))

    # Create dilation parameters
    seed = copy(-cl1)
    seed[1:-1, 1:-1] = (-cl1).min()
    mask = -cl1

    # Execute dilation reconstruction
    dilated = reconstruction(seed, mask, method='dilation')

    # Create colonies image
    colonies = -cl1 - dilated

    # Create circle mask
    rows, cols = colonies.shape
    cy, cx = rows // 2, cols // 2
    Y, X = ogrid[:rows, :cols]
    circle_mask = (X - cx)**2 + (Y - cy)**2 <= (radius * (1-shrink))**2

    # Mask segmentation
    masked_colonies = where(circle_mask, colonies, 0)

    masks, flows = apply_cellpose(masked_colonies)

    io.save_masks(masked_colonies, masks, flows, output_dir, tif=True, png=False)


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

    output_dir = args.output_path if args.output_path is not None else join(dirname(args.input_path), f"segmented_{basename(args.input_path).split(".")[0]}.tif")

    radius = args.diameter / 2 if args.diameter is not None else image.shape[0] / 2
    shrink = args.shrink if args.shrink is not None else 0

    # apply segmentation algorithm
    segment_well_colonies(image, output_dir=output_dir, radius = radius, shrink = shrink)

######################################
# Call main function id runned directly
if __name__ == "__main__":
    main()

# end of current module
