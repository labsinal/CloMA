"""
Preprocessing tab for the CloMA napari plugin.
"""
# imports
from magicgui import magicgui
from napari.layers import Image

from CloMA.extras import preprocess_images

#################################
# tab function

@magicgui(
    call_button="Run preprocessing",
    invert={"text": "Invert image"},
)
def preprocessing_tab(
    image: Image,
    invert: bool = True,
):
    """
    Preprocess an image layer.

    Parameters
    ----------
    image : napari.layers.Image
        Input image layer.
    invert : bool
        Whether to invert the image before preprocessing.

    Returns
    -------
    tuple
        A new napari image layer.
    """

    processed = preprocess_images(
        image=image.data,
        invert=invert,
    )

    return (
        processed,
        {
            "name": f"{image.name}_preprocessed",
        },
        "image",
    )