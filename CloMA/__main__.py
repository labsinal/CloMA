import typer
from typing import Annotated
import numpy as np

app = typer.Typer(add_completion = False)

@app.command()
def complete_pipeline(seg : Annotated[str, typer.Option(help="Path to segmentation masks to extract features")],
                      output : Annotated[str, typer.Option(help="Path to folder or file where imade will be saved")]):
    """
    Extract features from segmentation masks
    """
    ...


@app.command()
def well_detection(img : Annotated[str, typer.Option(help="Path to image to detect wells")], 
                   output:Annotated[str, typer.Option(help="Path to folder where image paths will ve saved")]):
    """
    Detects wells from images
    """
    ...

@app.command()
def preprocess_images(img : Annotated[str, typer.Option(help="Path to image to enhance")],
                      output : Annotated[str, typer.Option(help="Path to folder or file where imade will be saved")]):
    """
    Preprocess images
    """
    ...

@app.command()
def segment_images(img : Annotated[str, typer.Option(help="Path to image be segmented")],
                      output : Annotated[str, typer.Option(help="Path to folder or file where segmentation masks will be saved")],
                      filter_border_colonies : Annotated[bool, typer.Option(help="Remove colonies in the borders")] = True):
    """
    Segment colonies from images
    """
    ...

@app.command()
def extract_features(seg : Annotated[str, typer.Option(help="Path to segmentation masks to extract features")],
                      output : Annotated[str, typer.Option(help="Path to folder or file where imade will be saved")],
                      img : Annotated[str, typer.Option(help="Path to image for colour related features")] = None):
    """
    Extract features from segmentation masks
    """
    ...

if __name__ == "__main__":
    app()