"""
Run CloMA pipeline interactively
"""

######################################
# imports

from preprocess.detect_well import detect_wells_interactive
from preprocess.segmentation.segment_colonies_mixed import segment_well_colonies_hybrid
from preprocess.filter_border_colonies import filter_border_colonies
from feature_extraction.extract_features import extract_features
from os import path
from os import makedirs
from glob import glob
from pandas import DataFrame
from pandas import concat
from cv2 import imread, imwrite
from tifffile import imread as tifread
from tifffile import imwrite as tifwrite

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
    makedirs(segmentation_dir, exist_ok=True)
    print(f"Saving wells to {wells_dir}")

    print("\n" * 3)

    print("Saving segmentations: \n")

    # Save wells
    for i, well in enumerate(wells):
        filename = f"{path.basename(input).split(".")[0]}_well_{i:04d}.tiff"
        imwrite(path.join(wells_dir, filename), well)

        # run segmentation
        radius = well.shape[0] // 2
        segmentation = segment_well_colonies_hybrid(well, radius)
        filename = f"{path.basename(input).split(".")[0]}_labels_{i:04d}.tiff"
        imwrite(path.join(segmentation_dir, filename), segmentation)
        
    
    seg_files = glob(path.join(segmentation_dir, "*"))

    segmentations = list(map(tifread, seg_files))

    # Filter segmentations
    filtered_segmentations = list(map(lambda x:filter_border_colonies(x, radius-100), segmentations))

    # overwrite segmentation to filtered ones
    for filtered_seg, filename in zip(filtered_segmentations, seg_files):
        print(filename)
        tifwrite(filename, filtered_seg)

    # Extract features
    complete_df = DataFrame()

    for i, (well, seg) in enumerate(zip(wells, filtered_segmentations)):
        features = extract_features(seg, well)
        features["File"] = input
        features["Well"] = i + 1
    
        complete_df = concat([complete_df, features])
    
    return complete_df

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
            makedirs(output_path, exist_ok=True)
            break

    return_df = DataFrame()

    if path.isdir(input_path):

        for file in sorted(glob(path.join(input_path, "*"))):
            
            curr_output = path.join(output_path, path.basename(file).split(".")[0])

            makedirs(curr_output, exist_ok=True)

            current_df = cloma_interactive_pipeline(file, curr_output)

            return_df = concat([return_df, current_df])
    else:
        return_df = cloma_interactive_pipeline(input_path, output_path)
    
    save_file = path.join(output_path, "CloMA_table.csv")

    print(f"Saving results in: {save_file}")

    return_df.to_csv(save_file, index=False)

    print("\n" * 3)
    print("Done!")

######################################
# Call main function id runned directly
if __name__ == "__main__":
    main()

# end of current module