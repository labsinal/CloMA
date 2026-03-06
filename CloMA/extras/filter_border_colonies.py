"""
Module that paints black all colonies touching the border of the well.
"""

######################################
# imports

from numpy import ndarray, zeros_like
from os.path import join, dirname, basename
from cv2 import imwrite, connectedComponents, circle
from tifffile import imread
import numpy as np

######################################
# Define functions

def filter_border_colonies(input_segmentation: ndarray,
                           radius: int) -> ndarray:
    """
    Function that paints black all colonies touching the border of the well.

    Args:
        input_segmentation (ndarray): Unfiltered segmentation.
        radius (int): Radius to consider a colony touching the border.
    Return:
        Segmentation mask with only colonies inside a circle with defined radius and not touching the border,
        preserving original label IDs.
    """
    # Ensure grayscale or single-channel mask
    if input_segmentation.ndim == 3:
        input_segmentation = input_segmentation[..., 0]
    mask = (input_segmentation > 0).astype(np.uint8)

    # Identify connected components (labels start at 1)
    num_labels, labels = connectedComponents(mask)

    # Prepare output mask (same dtype as input for safety)
    filtered_mask = np.zeros_like(input_segmentation)

    # Define circular valid area (inside well)
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    circle_mask = np.zeros_like(mask, dtype=np.uint8)
    circle(circle_mask, center, radius, 1, -1)

    # Keep only colonies completely inside the circle
    for i in range(1, num_labels):
        colony_mask = (labels == i)
        # If any pixel of colony touches or goes outside the valid circle, skip it
        if np.any(colony_mask & (~circle_mask.astype(bool))):
            continue
        # Otherwise, preserve the original pixel values
        filtered_mask[colony_mask] = input_segmentation[colony_mask]

    return filtered_mask


######################################
# Define main function

def main() -> None:
    """
    Code's main function
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Remove colonies touching well border.")
    parser.add_argument("-i", "--input_segmentation",
                        action="store",
                        required=True,
                        dest="input_segmentation",
                        help="Path to the input segmentation.")

    parser.add_argument("-r", "--radius",
                        action="store",
                        required=False,
                        type=int,
                        dest="radius",
                        help="Filter radius")

    parser.add_argument("-o", "--output_segmentation",
                        action="store",
                        required=False,
                        dest="output_segmentation",
                        help="Path to save output segmentation.")

    
    args = parser.parse_args()

    # Default output path
    output_path = args.output_segmentation
    if output_path is None:
        output_path = join(dirname(args.input_segmentation),
                           basename(args.input_segmentation).replace(".png", "_filtered.png"))

    image = imread(args.input_segmentation)
    if image is None:
        raise FileNotFoundError(f"Could not read file {args.input_segmentation}")

    radius = round(args.radius) if args.radius is not None else image.shape[0] // 2

    filtered = filter_border_colonies(image, radius)
    imwrite(output_path, filtered)
    print(f"Filtered segmentation saved to {output_path}")


######################################
# Call main function if run directly
if __name__ == "__main__":
    main()

# end of current module
