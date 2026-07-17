"""
Segmentation tab for the CloMA napari plugin.
"""

# Imports
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
)
from magicgui import magicgui
from napari.layers import Image, Labels
from CloMA.segmentation import (
    run_segmentation as run_seg,
    create_circular_mask,
)

from CloMA.napari_plugin._cloma_tab import CloMATab

class SegmentationTab(CloMATab):
    """
    Segmentation tab of the CloMA plugin.

    This widget combines:
        - a custom threshold selector
        - a magicgui parameter form
        - a live circular mask preview
    """

    def __init__(self, napari_viewer):
        """Create the segmentation tab."""

        super().__init__(napari_viewer)

        # Preview layer (created only once)
        self.preview_layer = None

        ###############################################################
        # Threshold controls (Qt)

        # Label
        threshold_label = QLabel("Threshold")

        # Slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(0)

        # Numeric input
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 255)
        self.threshold_spinbox.setValue(0)

        # Keep slider and spinbox synchronized
        self.threshold_slider.valueChanged.connect(
            self.threshold_spinbox.setValue
        )
        self.threshold_spinbox.valueChanged.connect(
            self.threshold_slider.setValue
        )

        # Button to read the current image contrast
        self.threshold_button = QPushButton("From contrast")
        self.threshold_button.clicked.connect(
            self.threshold_from_contrast
        )

        # Horizontal layout for threshold controls
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)
        threshold_layout.addWidget(self.threshold_button)

        ###############################################################
        # MagicGUI widget

        self.gui = magicgui(
            self.run_segmentation,

            image={
                "choices": self.get_image_layers,
            },

            reference={
                "choices": self.get_label_layers,
            },

            call_button="Run Segmentation",

            circle_mask={
                "widget_type": "FloatSlider",
                "min": 0.50,
                "max": 1.00,
                "step": 0.01,
            },
        )

        self.register_layer_widget(self.gui)

        ###############################################################
        # Live preview connections

        # Update preview whenever the image changes
        self.gui.image.changed.connect(
            self.update_preview
        )

        # Update preview whenever the circle radius changes
        self.gui.circle_mask.changed.connect(
            self.update_preview
        )

        ###############################################################
        # Main layout

        layout = QVBoxLayout()

        layout.addWidget(threshold_label)
        layout.addLayout(threshold_layout)

        layout.addWidget(self.gui.native)

        # Push widgets to the top
        layout.addStretch()

        self.setLayout(layout)

    ##################################################################
    # Threshold helper

    def threshold_from_contrast(self):
        """
        Set the threshold equal to the minimum contrast limit of the
        selected image.
        """

        image = self.gui.image.value

        if image is None:
            return

        self.threshold_slider.setValue(
            int(image.contrast_limits[0])
        )

    ##################################################################
    # Preview

    def update_preview(self, *args):
        """
        Update the circular mask preview layer.
        """

        image = self.gui.image.value

        if image is None:
            return

        radius_factor = self.gui.circle_mask.value

        mask = ~create_circular_mask(
            image.data.shape,
            radius=image.data.shape[0] / 2 * radius_factor,
        )

        # Create preview layer only once
        if self.preview_layer is None:

            self.preview_layer = self.viewer.add_labels(
                mask.astype(int),
                name="Circle mask preview",
                opacity=0.30,
            )

        # Otherwise only replace its data
        else:

            self.preview_layer.data = mask.astype(int)

    def show_preview(self):
        """
        Show preview layer.
        """

        if self.preview_layer is not None:
            self.preview_layer.visible = True

    def hide_preview(self):
        """
        Hide preview layer.
        """

        if self.preview_layer is not None:
            self.preview_layer.visible = False

    ##################################################################
    # Segmentation

    def run_segmentation(
        self,
        image: Image,
        circle_mask: float = 0.97,
        reference: Labels = None,
        remove_border: bool = True,
        preprocess: bool = False,
        invert: bool = True,
    ):
        """
        Run colony segmentation.
        """

        # Read threshold from the custom widget
        threshold = self.threshold_slider.value()

        labels = run_seg(
            image=image.data,
            threshold=None if threshold == 0 else threshold,
            circle_mask=circle_mask,
            reference=(
                reference.data
                if reference is not None
                else None
            ),
            remove_border=remove_border,
            preprocess=preprocess,
            invert=invert,
        )

        # Add labels to the viewer
        self.viewer.add_labels(
            labels,
            name=f"{image.name}_labels",
        )