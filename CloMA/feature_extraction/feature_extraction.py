"""
Module for extracting morphological, intensity, and texture features from
clonogenic assay segmentations.
"""

import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops


#################################
# Private helper functions

def _extract_morphology(region, row):
    """
    Extract morphological features from a segmented object.

    Args:
    region : skimage.measure.RegionProperties
        RegionProperties object corresponding to a single segmented object.
    row : dict
        Dictionary where the extracted features will be stored.
    """

    # Basic region properties
    row["label"] = region.label
    row["area"] = region.area
    row["area_bbox"] = region.area_bbox
    row["area_convex"] = region.area_convex
    row["area_filled"] = region.area_filled

    # Ellipse-derived measurements
    row["axis_major_length"] = region.axis_major_length
    row["axis_minor_length"] = region.axis_minor_length

    # Object centroid
    row["y"] = region.centroid[0]
    row["x"] = region.centroid[1]

    # Shape descriptors
    row["eccentricity"] = region.eccentricity
    row["equivalent_diameter_area"] = region.equivalent_diameter_area
    row["euler_number"] = region.euler_number
    row["extent"] = region.extent
    row["feret_diameter_max"] = region.feret_diameter_max
    row["orientation"] = region.orientation
    row["perimeter"] = region.perimeter
    row["perimeter_crofton"] = region.perimeter_crofton
    row["solidity"] = region.solidity


def _extract_rgb(region, row):
    """
    Extract RGB intensity statistics from a segmented object.

    Args:
    region : skimage.measure.RegionProperties
        RegionProperties object containing the cropped RGB image.
    row : dict
        Dictionary where the extracted features will be stored.
    """

    # Binary mask and cropped RGB image
    mask = region.image
    rgb = region.intensity_image

    # Extract object pixels only
    pixels = rgb[mask]

    channels = ("r", "g", "b")

    # Compute statistics for each color channel
    for i, channel in enumerate(channels):

        row[f"intensity_mean-{channel}"] = pixels[:, i].mean()
        row[f"intensity_std-{channel}"] = pixels[:, i].std()
        row[f"intensity_min-{channel}"] = pixels[:, i].min()
        row[f"intensity_max-{channel}"] = pixels[:, i].max()


def _extract_dye(region, grayscale_image, mean_background, row):
    """
    Extract grayscale dye intensity features relative to the image background.

    The object intensity is compared to the average grayscale intensity of the
    image background.

    Args:
    region : skimage.measure.RegionProperties
        RegionProperties object corresponding to one segmented object.
    grayscale_image : ndarray
        Grayscale version of the original image.
    mean_background : float
        Mean grayscale intensity of the image background.
    row : dict
        Dictionary where the extracted features will be stored.
    """

    # Crop grayscale image using the object's bounding box
    minr, minc, maxr, maxc = region.bbox
    crop = grayscale_image[minr:maxr, minc:maxc]

    # Extract only object pixels
    pixels = crop[region.image]

    # Intensity statistics relative to the background
    row["dye_mean"] = abs(pixels.mean() - mean_background)
    row["dye_max"] = abs(pixels.max() - mean_background)
    row["dye_min"] = abs(pixels.min() - mean_background)
    row["dye_std"] = pixels.std()


def _extract_texture(region, grayscale_image, row):
    """
    Extract Gray-Level Co-occurrence Matrix (GLCM) texture features.

    Texture features are computed from the grayscale image cropped to the
    object's bounding box.

    Args:
    region : skimage.measure.RegionProperties
        RegionProperties object corresponding to one segmented object.
    grayscale_image : ndarray
        Grayscale version of the original image.
    row : dict
        Dictionary where the extracted features will be stored.
    """

    # Crop grayscale image to the object's bounding box
    minr, minc, maxr, maxc = region.bbox
    crop = grayscale_image[minr:maxr, minc:maxc].copy()

    # Remove pixels outside the object
    crop[~region.image] = 0

    # Compute Gray-Level Co-occurrence Matrix
    glcm = graycomatrix(
        crop,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    # Texture measurements to extract
    properties = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    )

    # Store each texture feature
    for prop in properties:
        row[f"glcm_{prop}"] = graycoprops(glcm, prop)[0, 0]


#################################
# Public API

def extract_features(segmentation, image=None):
    """
    Extract morphological, intensity, dye, and texture features from a
    labeled segmentation image.

    Args:
    segmentation : ndarray
        Labeled segmentation image where 0 represents the background and each
        positive integer corresponds to a unique object.
    image : ndarray, optional
        Original RGB image. If provided, intensity, dye, and texture features
        are also extracted. Otherwise, only morphological features are
        computed.

    Returns:
    pandas.DataFrame
        DataFrame containing one row per segmented object and one column per
        extracted feature.
    """

    # Store extracted features for every object
    rows = []

    grayscale = None
    mean_background = None

    # Precompute grayscale image and background intensity
    if image is not None:

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mean_background = grayscale[segmentation == 0].mean()

    # Iterate over every segmented object
    for region in regionprops(segmentation, intensity_image=image):

        row = {}

        # Morphological features
        _extract_morphology(region, row)

        if image is not None:

            # Color features
            _extract_rgb(region, row)

            # Dye intensity features
            _extract_dye(
                region,
                grayscale,
                mean_background,
                row,
            )

            # Texture features
            _extract_texture(
                region,
                grayscale,
                row,
            )

        rows.append(row)

    return pd.DataFrame(rows)