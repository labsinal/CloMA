"""
Module that extract features from clonogenic images and segmentation
"""

# Imports
from numpy import ndarray
from pandas import DataFrame
from pandas import concat
from tifffile import imread as read_tiff
from cv2 import imread
from skimage.measure import regionprops_table
from os.path import join, basename, dirname
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis


# Define helper functions


def read_image(image_path: str) -> ndarray:
    """Function that opens a image independently of filetype

    Args:
        image_path (str): Path to image

    Returns:
        np.ndarray: Opened image
    """
    # If it is tiff open with tifffile
    if image_path.endswith(".tif") or image_path.endswith(".tiff"):
        return read_tiff(image_path)

    # Else open with cv2
    return imread(image_path)


def extract_features(segmentation: ndarray, image: ndarray = None) -> DataFrame:
    """Function that extracts features from segmentation and images"""

    properties = [
        "label",
        "area",
        "area_bbox",
        "area_convex",
        "area_filled",
        "axis_major_length",
        "axis_minor_length",
        "centroid",
        "eccentricity",
        "equivalent_diameter_area",
        "euler_number",
        "extent",
        "feret_diameter_max",
        "intensity_max",
        "intensity_mean",
        "intensity_min",
        "intensity_std",
        "num_pixels",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        "solidity",
    ]

    if image is None:
        remove = [
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "intensity_std",
        ]
        properties = [x for x in properties if x not in remove]

    image = rgb2gray(image)

    properties_df = DataFrame(
        regionprops_table(segmentation, image, properties=properties)
    )

    properties_df.rename(
        columns={
            "centroid-0": "y",
            "centroid-1": "x",
            "intensity_max-0": "intensity_max-r",
            "intensity_max-1": "intensity_max-g",
            "intensity_max-2": "intensity_max-b",
            "intensity_mean-0": "intensity_mean-r",
            "intensity_mean-1": "intensity_mean-g",
            "intensity_mean-2": "intensity_mean-b",
            "intensity_min-0": "intensity_min-r",
            "intensity_min-1": "intensity_min-g",
            "intensity_min-2": "intensity_min-b",
            "intensity_std-0": "intensity_std-r",
            "intensity_std-1": "intensity_std-g",
            "intensity_std-2": "intensity_std-b",
        },
        inplace=True,
    )

    # =====================================================
    # ADD NEW FEATURES ONLY IF IMAGE PROVIDED
    # =====================================================

    if image is not None:

        gray_uint8 = (image * 255).astype(np.uint8)
        gradient = sobel(image)

        texture_features = []
        gradient_features = []
        intensity_distribution_features = []

        labels = properties_df["label"].values

        for lab in labels:

            mask = segmentation == lab

            # -----------------------
            # GLCM Texture
            # -----------------------
            region_pixels = gray_uint8 * mask

            glcm = graycomatrix(
                region_pixels,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True,
            )

            contrast = graycoprops(glcm, "contrast")[0, 0]
            dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
            homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
            energy = graycoprops(glcm, "energy")[0, 0]
            correlation = graycoprops(glcm, "correlation")[0, 0]

            texture_features.append(
                [contrast, dissimilarity, homogeneity, energy, correlation]
            )

            # -----------------------
            # Gradient statistics
            # -----------------------
            grad_vals = gradient[mask]

            gradient_features.append(
                [
                    np.mean(grad_vals),
                    np.std(grad_vals),
                    np.max(grad_vals),
                ]
            )

            # -----------------------
            # Intensity distribution
            # -----------------------
            intens_vals = image[mask]

            intensity_distribution_features.append(
                [
                    np.std(intens_vals),
                    skew(intens_vals),
                    kurtosis(intens_vals),
                ]
            )

        texture_df = DataFrame(
            texture_features,
            columns=[
                "glcm_contrast",
                "glcm_dissimilarity",
                "glcm_homogeneity",
                "glcm_energy",
                "glcm_correlation",
            ],
        )

        gradient_df = DataFrame(
            gradient_features,
            columns=[
                "gradient_mean",
                "gradient_std",
                "gradient_max",
            ],
        )

        intensity_dist_df = DataFrame(
            intensity_distribution_features,
            columns=[
                "gray_std",
                "gray_skewness",
                "gray_kurtosis",
            ],
        )

        properties_df = concat(
            [properties_df, texture_df, gradient_df, intensity_dist_df],
            axis=1,
        )

    return properties_df


# Define main function
def main() -> None:
    """
    Main function to extract features from clonogenic images and segmentation
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Module taht extract features from clonogenic images and segmentation"
    )

    parser.add_argument(
        "-s",
        "--segmentation_path",
        action="store",
        dest="segmentation_path",
        required=True,
        help="Path to semgnetation images",
    )

    parser.add_argument(
        "-i",
        "--image_path",
        action="store",
        dest="image_path",
        required=False,
        help="Path to clonogenic image (if not defined only area and shape features will be extracted)",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        action="store",
        dest="output_path",
        required=False,
        help="Path to output csv, if not passed will be saved in the same folder as segmentation_path",
    )

    args = parser.parse_args()

    # open images
    segmentation = read_image(args.segmentation_path)
    image = read_image(args.image_path) if args.image_path is not None else None

    # create featurres table
    features_table = extract_features(segmentation, image=image)

    # Save
    output_path = (
        args.output_path
        if args.output_path is not None
        else join(
            dirname(args.segmentation_path),
            basename(args.segmentation_path).split(".")[0] + ".csv",
        )
    )

    features_table.to_csv(output_path, index=False)


# Call main funtion
if __name__ == "__main__":
    main()