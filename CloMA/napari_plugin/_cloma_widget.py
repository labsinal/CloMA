"""
Module to create CloMA and Napari integration widget
"""
# Imports
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from ._preprocessing_tab import preprocessing_tab
# from ._segmentation_tab import SegmentationTab
# from ._features_tab import FeatureExtractionTab

# Create widget Class
class CloMAWidget(QWidget):

    # Set __init__ function
    def __init__(self, napari_viewer):
        # Call QWidget contructor
        super().__init__()

        # Get napari viewer
        self.viewer = napari_viewer

        # Create tabs container
        tabs = QTabWidget()

        # Add tabs
        tabs.addTab(
            preprocessing_tab().native,
            "Preprocessing",
        )

        # tabs.addTab(
        #     SegmentationTab(viewer),
        #     "Segmentation",
        # )

        # tabs.addTab(
        #     FeatureExtractionTab(viewer),
        #     "Feature Extraction",
        # )

        # Stack widgets vertically
        layout = QVBoxLayout()

        # Add the tabs to layout
        layout.addWidget(tabs)

        # Assign the layout
        self.setLayout(layout)