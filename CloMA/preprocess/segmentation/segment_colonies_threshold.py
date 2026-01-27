"""
Module that segment colonies from well images
"""

######################################
# imports
from numpy import ndarray, copy, ogrid, where
from cv2 import imread, imwrite
from cv2 import createCLAHE
from cv2 import cvtColor, COLOR_BGR2GRAY
from skimage.morphology import reconstruction
from skimage.filters import threshold_otsu
from skimage.segmentation import expand_labels
from scipy.ndimage import label
from os.path import join, basename, dirname 

######################################
# Define helper functions

def segment_well_colonies(image:ndarray, radius:int, shrink = 0) -> ndarray:
    """
    Function that segments well colonies photo

    Args:
        image:ndarray | well image as numpy array
    Returns
    ndarray | segmented well image
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

    # Binarize segmentation
    thresh = threshold_otsu(masked_colonies)
    binary = masked_colonies > thresh
    
    dilated_masks = expand_labels(binary, distance=5)

    # separate masks via touching
    labels = label(dilated_masks)

    masked_labels = where(binary, dilated_masks, 0)

    return masked_labels

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
    segmentation = segment_well_colonies(image, radius = radius, shrink = shrink)

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
