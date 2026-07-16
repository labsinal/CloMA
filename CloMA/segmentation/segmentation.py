"""
Module related to segmentation of colonies
"""

# imports
import numpy as np
from skimage.filters import threshold_multiotsu
import skimage as ski
import scipy.ndimage as ndi
import cv2

########################################
# define functions
def apply_treshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Function that returns a binary image given a treshold

    Args:
        image (np.ndarray): preprocessed grayscale image
        threshold (float): treshold value
    Return (np.ndarray):
        binary image comparin each pixel if it is > treshold 
    """
    return (image > threshold)

def multiotsu_tresholding(image:np.ndarray) -> int:
    """
    Function that applies multiotsu tresholding separeting in two classes
    from a grayscale image

    Args:
        image (np.ndarray): preprocessed grayscale image
    Return (int):
        multiotsu treshold
    """
    return threshold_multiotsu(image, 2)[0]

def create_circular_mask(shape: tuple[int, int], center:tuple[int, int] = None, radius:int = None) -> np.ndarray:
    """
    Function that creates a circular mask
    Args:
        shape (tuple[int, int]): the shape of the resulting image
        center (tuple[int, int]): center of the circle mask
        radius (int): radius of the circle mask
    Return (np.ndarray)
        Array of binary image that is true inside the circle defined and false outside it.
    """
    if center is None: # Use the middle of the image
        center = (int(shape[1]/2), int(shape[0]/2))
    if radius is None: # Use the smallest distance to the edge
        radius = min(center[0], center[1], shape[1]-center[1], shape[0]-center[0])

    # Generate coordinate grids
    Y, X = np.ogrid[:shape[0], :shape[1]:]
    
    # Calculate squared distance from center
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)

    # Return boolean mask (True inside circle, False outside)
    return dist_from_center <= radius

def automatic_separation(binary:np.ndarray,
                         min_distance: int = 10,
                         threshold_rel: float = 0.06) -> np.ndarray:
    """
    Function that given a binary image separate its masks
    Args:
        binary (np.ndarray): Binary image
        min_distance (int): Min distance for colony separation
        threshold_rel (float): peak detection parameter
    Return (np.ndarray): Separated labels image
    """
    # Fill holes inside binary objects so watershed works on solid regions
    filled = ndi.binary_fill_holes(binary)

    # Remove small artifacts and smooth object boundaries
    opened = ski.morphology.opening(
        filled,
        ski.morphology.disk(2)
    )
    closed = ski.morphology.closing(
        opened,
        ski.morphology.disk(2)
    )

    # Compute distance transform from object boundaries
    distance = ndi.distance_transform_edt(closed)

    # Detect local maxima in the distance map as watershed markers
    coords = ski.feature.peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        labels=closed,
    )
    markers = np.zeros_like(closed, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # Apply watershed to separate touching objects using the distance peaks
    labels = ski.segmentation.watershed(
        -distance,
        markers,
        mask=closed,
    )

    labels_merged = labels.copy()

    # Merge small enclosed regions back into neighboring labels if needed
    for lab in np.unique(labels_merged):

        if lab == 0:
            continue

        region = labels_merged == lab

        border_region = (
            ski.morphology.dilation(region)
            & ~region
        )

        neighbors = np.unique(
            labels_merged[border_region]
        )

        neighbors = neighbors[
            neighbors != lab
        ]

        # Ignore objects touching background
        if 0 in neighbors:
            continue

        # Only consider regions fully enclosed by exactly one neighbor
        if len(neighbors) != 1:
            continue

        surround_label = neighbors[0]

        area_region = region.sum()

        area_surround = np.sum(
            labels_merged == surround_label
        )

        # Merge very small internal regions into the surrounding object
        if area_region < 0.3 * area_surround:

            labels_merged[
                labels_merged == lab
            ] = surround_label
    
    return labels_merged

def reference_separation(binary:np.ndarray,
                         reference: np.ndarray) -> np.ndarray:
    """
    Function that given a binary mask and reference labels
    adjust the morphology from the reference to the binary mask
    
    Args:
        binary (np.ndarray): binary image
        reference (np.ndarray): reference labels
    Returns:
        Reference labels with adjusted morphology 
    """
    return np.where(binary, reference, 0)

def remove_border_colonies(labels:np.ndarray,
                           mask:np.ndarray) -> np.ndarray:
    """
    Function that given a mask and labels paint black (0) all labels
    that have at least one pixel touching the mask.
    """
    # Identify the border pixels of the mask by XORing the mask with
    # its eroded version. This leaves only the boundary pixels.
    border = mask ^ ski.morphology.erosion(mask)

    # Gather the unique label ids that appear on the border positions.
    border_labels = np.unique(
        labels[border]
    )

    # Exclude the background label (0) from the list of border labels.
    border_labels = border_labels[
        border_labels > 0
    ]

    # Create a copy of the label image to preserve the input array.
    labels_final = labels.copy()

    # Set all border-touching labeled objects to background (0).
    labels_final[
        np.isin(labels_final, border_labels)
    ] = 0

    return labels_final

def run_segmentation(
    image: np.ndarray,
    threshold: int = None,
    circle_mask: float = 0.97,
    reference: np.ndarray = None,
    remove_border: bool= True,
    preprocess: bool = False,
    invert: bool = True
) -> np.ndarray:
    """
    Segment colonies from an image using automatic or reference mode.
    Run this on the raw images, not preprocessed ones
    """
    from pathlib import Path

    # if preprocess is set to true, run preprocessment
    if preprocess:
        from CloMA.extras import preprocess_images
        image = preprocess_images(image, invert)

    # check if image is grayscale
    if (len(image.shape) > 2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate threshold
    tresh = multiotsu_tresholding(image) if threshold is None else threshold

    binary = apply_treshold(image, tresh)

    # Create mask
    mask = create_circular_mask(image.shape, radius=image.shape[0] / 2 * circle_mask)
    binary = np.where(mask, binary, 0)

    # Separate masks into labels
    if reference is None:
        labels = automatic_separation(binary)

    else:
        labels = reference_separation(binary, reference)
    
    # remove border colonies
    if remove_border:
        final_labels = remove_border_colonies(labels, mask)

    # create return labels
    return final_labels    
# end of current module