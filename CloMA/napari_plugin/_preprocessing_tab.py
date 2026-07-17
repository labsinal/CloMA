"""
Preprocessing tab for the CloMA napari plugin.
"""
# imports
from qtpy.QtWidgets import QVBoxLayout
from magicgui import magicgui
from napari.layers import Image

from CloMA.napari_plugin._cloma_tab import CloMATab
from CloMA.extras import preprocess_images

# Create PreprocessingTab class
class PreprocessingTab(CloMATab):

    # Define class constructor
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        # Create the magicgui widget
        self.gui = magicgui(
            self.run_preprocessing,

            image={
                "choices": self.get_image_layers,
            },

            call_button="Run preprocessing",

            invert={
                "text": "Invert image",
            },
        )

        self.register_layer_widget(self.gui)

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