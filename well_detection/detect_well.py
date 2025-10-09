"""
Module that detects wells in a plate photograph
"""

######################################
# imports

######################################
# Define helper functions

######################################
# Define main function

def main() -> None:
    """
    Code's main function
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")

    parser.add_argument("-i,", "--input",
                        action="store",
                        required=True,
                        dest="input_path",
                        help="Path to image of plate")
    
######################################
# Call main function id runned directly
if __name__ == "__main__":
    main()

# end of current module