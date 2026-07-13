"""
Module that segment colonies from well images
"""

######################################
# imports
from numpy import ndarray, copy
import cv2
from skimage.morphology import reconstruction

######################################
# Define helper functions

def preprocess_images(image:ndarray, invert:bool) -> ndarray:
    """
    Image preprocessment using clahe and background reconstruction subtraction

    Args:
        image (np.ndarray): Image as numpy array
        invert (bool): wether to invert or not the image
    Returns:
        np.ndarray: Image with filters applied
    """
    # Create cv2 clahe object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    
    # Check if image is grayscale
    if len(image.shape) > 2:
        # convert if it is not
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply clahe filterj
    clahed = clahe.apply(image)

    # invert image
    seed = copy(-clahed) if invert else copy(clahed)
    seed[1:-1, 1:-1] = (-clahed).min()
    
    #  apply reconstruction by dilation 
    background = reconstruction(seed, -clahed, method="dilation")

    # Create colonis image subtracting the background from the original clahed one
    colonies = -clahed - background

    return colonies