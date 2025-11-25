"""
Run CloMA pipeline interactively
"""

######################################
# imports

from preprocess.detect_well import detect_wells_interactive
from preprocess.segmentation.segment_colonies_cellpose import segment_well_colonies
from preprocess.filter_border_colonies import filter_border_colonies
from os import path
from os import makedirs
from glob import glob
from pandas import DataFrame
from pandas import concat
from cv2 import imread, imwrite
from tifffile import imread as tifread

######################################
# Define helper functions

def cloma_interactive_pipeline(input:str, output:str) -> DataFrame:
    """
    Run CloMA pipelina interactively
    
    :param input: (str) input_folder or file 
    :param multiple_sheets: (bool) wheter to save a single sheet or multiple
    :param output: (str|None) Where to save images, if None, save in input folder
    """

    # get input dir
    input_dir = path.dirname(input)

    # open image
    image = imread(input)

    # run well detector
    print("Starting well detector...")
    wells = detect_wells_interactive(image)

    wells_dir = path.join(output, "wells")
    segmentation_dir = path.join(output, "segmentations")
    makedirs(wells_dir, exist_ok=True)
    print(f"Saving wells to {wells_dir}")

    for i, well in enumerate(wells):
        filename = f"{path.basename(input).split(".")[0]}_well_{i:04d}.tiff"
        imwrite(path.join(output, filename), well)

        # run segmentation
        radius = well.shape[0] / 2
        segment_well_colonies(well, segmentation_dir, radius, 0)
    
    segmentations = list(map(tifread, glob(path.join(segmentation_dir, "*"))))

    filtered_segmentations = list(map(lambda x:filter_border_colonies(x, radius-2), segmentations))

    




######################################
# Define main function

def main() -> None:
    """
    Code's main function
    """

    # print welcome message
    print("Welcome to      ")
    print("""
   █████████  ████           ██████   ██████   █████████  
  ███░░░░░███░░███          ░░██████ ██████   ███░░░░░███ 
 ███     ░░░  ░███   ██████  ░███░█████░███  ░███    ░███ 
░███          ░███  ███░░███ ░███░░███ ░███  ░███████████ 
░███          ░███ ░███ ░███ ░███ ░░░  ░███  ░███░░░░░███ 
░░███     ███ ░███ ░███ ░███ ░███      ░███  ░███    ░███ 
 ░░█████████  █████░░██████  █████     █████ █████   █████
  ░░░░░░░░░  ░░░░░  ░░░░░░  ░░░░░     ░░░░░ ░░░░░   ░░░░░ 
                                                          
                                                          
                                                          """)
    
    # Get user inputs
    input_path = input("Paste here the path to your image or folder containing images:\n")
    print("-" * 30)
    
    while True:
        print("\n" * 3)
        output_path = input("Paste here the path to save the output:\n")
        print("-" * 30)

        # deal with output
        if not path.isdir(output_path):
            print("Output must be a directory!")
        else:
            makedirs(output_path, exists_ok=True)
            break

    return_df = DataFrame()

    if path.isdir(input_path):

        for file in sorted(glob(path.join(input_path, "*"))):
            
            current_df = cloma_interactive_pipeline(file, output_path)

            return_df = concat(return_df, current_df)
    else:
        return_df = cloma_interactive_pipeline(input_path, output_path)
    
    print(f"Saving results in: {output_path}")

    return_df.to_csv(output_path, index=False)

    print("\n" * 3)
    print("Done!")

######################################
# Call main function id runned directly
if __name__ == "__main__":
    main()

# end of current module