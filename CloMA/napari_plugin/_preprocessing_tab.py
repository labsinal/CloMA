"""
Preprocessing tab for the CloMA napari plugin.
"""
# imports
from qtpy.QtWidgets import QWidget, QVBoxLayout
from magicgui import magicgui
from napari.layers import Image

from CloMA.extras import preprocess_images

# Create PreprocessingTab class
class PreprocessingTab(QWidget):

    # Define class constructor
    def __init__(self, napari_viewer):
        super().__init__()
        
        # Get viewer
        self.viewer = napari_viewer

        # Create the magicgui widget
        self.gui = magicgui(
            self.run_preprocessing,
            call_button="Run preprocessing",
            invert={"text": "Invert image"},
        )

        # Build the layout
        layout = QVBoxLayout()
        layout.addWidget(self.gui.native)

        # Add stretch for spacing
        layout.addStretch()

        self.setLayout(layout)

    def run_preprocessing(
        self,
        image: Image,
        invert: bool = True,
    ):
        """
        Called automatically when the Run button is pressed.
        """

        # Create preprocessed image
        processed = preprocess_images(
            image=image.data,
            invert=invert,
        )

        # Add image as a new layer
        self.viewer.add_image(
            processed,
            name=f"{image.name}_preprocessed",
        )