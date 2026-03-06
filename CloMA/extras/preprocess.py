"""
Module that segment colonies from well images
"""

######################################
# imports
from numpy import ndarray, copy
from cv2 import createCLAHE
from cv2 import cvtColor, COLOR_BGR2GRAY
from cv2 import imread, imwrite
from skimage.morphology import reconstruction

######################################
# Define helper functions

def preprocess_images(image:ndarray) -> ndarray:
    """
    Image preprocessment using clahe and background reconstruction subtraction

    Args:
        image:np.ndarray
            Image as numpy array
    
    Returns:
        np.ndarray
            Image with filters applied
    """
    # Create cv2 clahe object
    clahe = createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    
    # Check if image is grayscale
    if image.shape[2] > 2:
        # convert if it is not
        image = cvtColor(image, COLOR_BGR2GRAY)

    # apply clahe filter
    clahed = clahe.apply(image)

    # invert image
    seed = copy(-clahed)
    seed[1:-1, 1:-1] = (-clahed).min()
    
    #  apply reconstruction by dilation 
    background = reconstruction(seed, -clahed, method="dilation")

    # Create colonis image subtracting the background from the original clahed one
    colonies = -clahed - background

    return colonies

######################################
# Define main function

def main():
    """
    Code's main  function
    """

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Image pre-processing")
    parser.add_argument("-i", "--input_image",
                        action="store",
                        required=True,
                        dest="input_image",
                        help="Path to the input_image file.")

    parser.add_argument("-o", "--output_image",
                        action="store",
                        required=False,
                        dest="output_image",
                        help="Path to save the processed image")
    
    args = parser.parse_args()

    output_path = args.output_image

    if output_path is None:
        split_input = args.input_image.rsplit(".", 1)
        output_path = split_input[0] + "_preprocessed." + split_input[-1]

    img = imread(args.input_image)

    preprocessed_img = preprocess_images(image=img)

    imwrite(output_path, preprocessed_img)

    print(f"Saving on {output_path}")

    
if __name__ == "__main__":
    main()