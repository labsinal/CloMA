"""
Module that segment colonies from well images
"""

######################################
# imports
import numpy as np
from ..extras import preprocess_images
import scipy.ndimage as ndi
import skimage as ski

######################################
# Define helper functions

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # Use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # Use the smallest distance to the edge
        radius = min(center[0], center[1], w-center[0], h-center[1])

    # Generate coordinate grids
    Y, X = np.ogrid[:h, :w]
    
    # Calculate squared distance from center
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Return boolean mask (True inside circle, False outside)
    return dist_from_center <= radius

def remove_border_colonies(labels : np.ndarray,
                           shrink : float) -> np.ndarray:
    """
    Function that removes colonies touching the border 
    of the defined mask
    """
    radius = int((min(labels.shape[0], labels.shape[1]) // 2) * (1 - shrink))
    mask = create_circular_mask(labels.shape[0], labels.shape[1],
                                radius=radius)

    border = mask ^ ski.morphology.erosion(mask)

    border_labels = np.unique(
        labels[border]
    )

    border_labels = border_labels[
        border_labels > 0
    ]

    labels_final = labels.copy()

    labels_final[
        np.isin(labels_final, border_labels)
    ] = 0

    return labels_final

def binary_segmentation(image  : np.ndarray,
                        shrink : float = 0.03,
                        preprocessed: bool = False,
                        threshold_method: str = "otsu",
                        threshold_value: float = None) -> np.ndarray:
    """
    Function that does the binary colonies segmentation.

    Args:
        image: Raw or preprocessed image.
        shrink: Fraction used to compute the circular mask.
        preprocessed: If True, the input image is already preprocessed.
    """
    if preprocessed:
        preprocessed_image = image
    else:
        preprocessed_image = preprocess_images(image)

    # create a circular mask
    radius = int((min(preprocessed_image.shape[0], preprocessed_image.shape[1]) // 2) * (1 - shrink))
    mask = create_circular_mask(preprocessed_image.shape[0], preprocessed_image.shape[1], radius=radius)
    masked = np.zeros_like(preprocessed_image)
    masked[mask] = preprocessed_image[mask]

    # calculate the threshold using only the masked region
    if not np.any(mask):
        threshold = 0
    else:
        if threshold_method == "manual" and threshold_value is not None:
            threshold = float(threshold_value)
        elif threshold_method == "multi":
            # use two classes to obtain a single threshold from multi otsu
            thr = ski.filters.threshold_multiotsu(masked[mask], classes=2)
            threshold = thr[0] if len(thr) > 0 else ski.filters.threshold_otsu(masked[mask])
        else:
            threshold = ski.filters.threshold_otsu(masked[mask])

    # make binary segmentation
    return masked > threshold


def watershed_from_binary(binary: np.ndarray,
                         min_distance: int = 10,
                         threshold_rel: float = 0.06) -> np.ndarray:
    """
    Given a binary mask, run distance-transform based watershed and return labels.
    """
    # prepare for watershed
    filled = ndi.binary_fill_holes(binary)

    opened = ski.morphology.opening(
        filled,
        ski.morphology.disk(2)
    )
    closed = ski.morphology.closing(
        opened,
        ski.morphology.disk(2)
    )

    distance = ndi.distance_transform_edt(closed)

    coords = ski.feature.peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        labels=closed,
    )
    markers = np.zeros_like(closed, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    labels = ski.segmentation.watershed(
        -distance,
        markers,
        mask=closed,
    )

    return labels

def automatic_segmentation( image: np.ndarray,
                            shrink: float = 0.03,
                            threshold_method: str = "otsu",
                            threshold_value: float = None) -> np.ndarray:
    """
    Automatic segmentation using threshhold and
    watershed separation
    """
    binary = binary_segmentation(image, shrink, preprocessed=False,
                                 threshold_method=threshold_method,
                                 threshold_value=threshold_value)

    labels = watershed_from_binary(binary)

    # merge surrounded colonies
    labels_merged = labels.copy()

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

        # Fully enclosed by exactly one colony
        if len(neighbors) != 1:
            continue

        surround_label = neighbors[0]

        area_region = region.sum()

        area_surround = np.sum(
            labels_merged == surround_label
        )

        # optional safety criterion
        if area_region < 0.3 * area_surround:

            labels_merged[
                labels_merged == lab
            ] = surround_label
    
    # remove touching border colonies
    labels_final = remove_border_colonies(labels_merged, shrink)

    return labels_final

def reference_segmentation(image : np.ndarray,
                           reference_labels : np.ndarray,
                           shrink : float = 0.97) -> np.ndarray:
    
    """
    Function that runs segmentation from reference labels.
    The input image is preprocessed before computing the binary mask.
    """

    preprocessed = preprocess_images(image)
    binary = binary_segmentation(image=preprocessed, shrink=shrink, preprocessed=True)

    # Ensure reference labels match binary mask shape. If not, resize using
    # nearest-neighbor interpolation (preserve integer labels).
    ref = reference_labels
    if getattr(ref, "ndim", 0) > 2:
        # If multi-channel, take the first channel which is commonly used for labels
        ref = ref[..., 0]

    # If shapes differ, center-crop or center-pad the reference labels to match
    if ref.shape != binary.shape:
        th, tw = binary.shape
        rh, rw = ref.shape

        # If reference is larger -> crop center
        if rh >= th and rw >= tw:
            start_y = (rh - th) // 2
            start_x = (rw - tw) // 2
            ref_matched = ref[start_y:start_y + th, start_x:start_x + tw]
        else:
            # reference is smaller in at least one dimension -> pad centered
            ref_matched = np.zeros(binary.shape, dtype=ref.dtype)
            start_y = max((th - rh) // 2, 0)
            start_x = max((tw - rw) // 2, 0)
            ref_matched[start_y:start_y + rh, start_x:start_x + rw] = ref

        return np.where(binary, ref_matched, 0)

    return np.where(binary, ref, 0)

# end of current module
