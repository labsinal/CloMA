"""
Module that detects wells in a plate photograph
"""

######################################
# imports
from numpy import ndarray
from cv2 import imread, imwrite
from cv2 import cvtColor, COLOR_BGR2GRAY
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from os import makedirs
from os.path import join, dirname, basename
from sys import exit

######################################
# Define helper functions

def detect_wells_interactive(image: ndarray) -> list[ndarray]:
    """
    Function that interactively detects wells in a plate photograph

    Args:
        image (ndarray): Image of plate
        well_radius (int): Radius of wells in pixels
    Return:
        list[ndarray]: List of detected wells as numpy arrays
    """
    import matplotlib.pyplot as plt

    # convert images to grayscale
    image_gray = cvtColor(image, COLOR_BGR2GRAY)

    # Detect edges
    sigma = 3
    edges = canny(image_gray, sigma)
    while True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image[...,::-1])
        ax1.set_title("Original Image")
        ax2.imshow(edges)
        ax2.set_title(f"Edges (sigma={sigma})")
        plt.show()
        input_value = input("Try new sigma value (type 'ok' for last tested value): ")
        if input_value.casefold() == "ok":
            break
        else:
            try:
                sigma = int(input_value)
                edges = canny(image_gray, sigma)
            except:
                print("Invalid sigma value.")
                continue
    
    # Applt hough transform
    radius = 300
    hough_res = hough_circle(edges, radius)
    while True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image[...,::-1])
        ax1.set_title("Original Image")
        ax2.imshow(hough_res[0])
        ax2.set_title(f"Hough transform (radius={radius})")
        plt.show()
        input_value = input("Try new radius value (type 'ok' for last tested value): ")
        if input_value.casefold() == "ok":
            break
        else:
            radius = int(input_value)
            hough_res = hough_circle(edges, radius)

    # Extract circles
    accum, cx, cy, radii = hough_circle_peaks(hough_res, [radius], min_xdistance=radius, min_ydistance=radius)

    # Filter circles that do not overlap and are entirelly in the image
    max_y, max_x, _= image.shape
    circles_info = [{"cx": x, "cy": y, "radii": r} for x, y, r in zip(cx, cy, radii)]
    circles_filter = lambda circle : circle["cx"] - circle["radii"] > 0 and \
                                      circle["cy"] - circle["radii"] > 0 and \
                                      circle["cx"] + circle["radii"] < max_x and \
                                      circle["cy"] + circle["radii"] < max_y
    
    circles_info = list(filter(circles_filter, circles_info))

    circles = [image[circle["cy"] - circle["radii"]:circle["cy"] + circle["radii"], circle["cx"] - circle["radii"]:circle["cx"] + circle["radii"]] for circle in circles_info]

    if len(circles) == 0:
        print("No circles detected with given parameters.")
        exit()

    nrows = 1 if len(circles) <= 3 else len(circles) // 3
    ncols = 3 if len(circles) >= 3 else len(circles)

    fig, axs = plt.subplots(nrows, ncols)

    for i, circle in enumerate(circles):
        if len(circles) <= 3:
            axs[i].imshow(circle)
            axs[i].set_title(f"Well {i}")
        else:
            axs[i//ncols, i % ncols].imshow(circle)
            axs[i//ncols, i % ncols].set_title(f"Well {i}")
    plt.show()

    return circles

def detect_wells(image: ndarray, well_radius: int, sigma:float) -> list[ndarray]:
    """
    Function that detects wells in a plate photograph

    Args:
        image (ndarray): Image of plate
        well_radius (int): Radius of wells in pixels
    Return:
        list[ndarray]: List of detected wells as numpy arrays
    """

    # convert images to grayscale
    image_gray = cvtColor(image, COLOR_BGR2GRAY)

    # Detect edges
    edges = canny(image_gray, sigma=sigma)

    # Apply Hough Transform to detect circles
    hough_res = hough_circle(edges, [well_radius])

    # Extract circles
    accum, cx, cy, radii = hough_circle_peaks(hough_res, [well_radius], min_xdistance=well_radius, min_ydistance=well_radius)

    # Filter circles that do not overlap and are entirelly in the image
    max_y, max_x, _= image.shape
    circles_info = [{"cx": x, "cy": y, "radii": r} for x, y, r in zip(cx, cy, radii)]
    circles_filter = lambda circle : circle["cx"] - circle["radii"] > 0 and \
                                      circle["cy"] - circle["radii"] > 0 and \
                                      circle["cx"] + circle["radii"] < max_x and \
                                      circle["cy"] + circle["radii"] < max_y
    
    circles_info = list(filter(circles_filter, circles_info))

    circles = [image[circle["cy"] - circle["radii"]:circle["cy"] + circle["radii"], circle["cx"] - circle["radii"]:circle["cx"] + circle["radii"]] for circle in circles_info]

    if circles == []:
        print("No circles detected with given parameters.")
        exit()

    return circles


######################################
# Define main function

def main() -> None:
    """
    Code's main function
    """
    # import argument parsing library
    from argparse import ArgumentParser

    # initialize argument parsing object
    parser = ArgumentParser(description="Module that detects wells in a plate photograph")

    # add arguments to parser
    parser.add_argument("-i,", "--input",
                        action="store",
                        required=True,
                        dest="input_path",
                        help="Path to image of plate")

    parser.add_argument("-o", "-output",
                         action="store",
                         required=False,
                         default=None,
                         dest="output_path",
                         help="Path to output folder")

    parser.add_argument("-r", "--radius",
                        action="store",
                        required=False,
                        default=-1,
                        dest="well_radius",
                        type=int,
                        help="Radius of wells in pixels")
    
    parser.add_argument("--sigma",
                        action="store",
                        dest="sigma",
                        default=3,
                        type=float,
                        help="Sigma for gaussiam blur (def=3)")


    parser.add_argument("--interactive",
                        action="store_true",
                        dest="interactive",
                        help="Radius of wells in pixel")
    
    # parse args
    args = parser.parse_args()

    # open image
    image = imread(args.input_path)

    # run main function accordingly with interactive being selected
    if args.interactive:
        wells = detect_wells_interactive(image)
    else:
        if args.well_radius <= 0:
            print("Inform correct well radius or run interactivelly.")
            exit()

        wells = detect_wells(image, args.well_radius, args.sigma)

    output = args.output_path

    # if no output is passed save to input folder
    if output is None:
        output = join(dirname(args.input_path), f"{basename(args.input_path).split(".")[0]}_wells")
        print(f"Saving to {output}")
    
    # Create output dir if it does not exist
    makedirs(output, exist_ok=True)

    # Save wells images
    for i, well in enumerate(wells):
        filename = f"{basename(args.input_path).split(".")[0]}_well_{i:04d}.tiff"
        imwrite(join(output, filename), well)
    
    print("Done!")

######################################
# Call main function id runned directly
if __name__ == "__main__":
    main()

# end of current module