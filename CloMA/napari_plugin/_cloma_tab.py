"""
Base class for all CloMA napari plugin tabs.
"""

from qtpy.QtWidgets import QWidget
from napari.layers import Image, Labels


class CloMATab(QWidget):
    """
    Base class for all CloMA plugin tabs.

    Provides:

    - Access to the napari viewer.
    - Automatic updating of Image/Labels selectors.
    - Helper methods for retrieving napari layers.
    """

    def __init__(self, viewer):
        super().__init__()

        # Store napari viewer
        self.viewer = viewer

        # List of magicgui widgets whose layer selectors
        # should be kept synchronized with the viewer.
        self._layer_widgets = []

        # Automatically refresh selectors whenever layers
        # are added or removed.
        self.viewer.layers.events.inserted.connect(
            self._refresh_registered_widgets
        )

        self.viewer.layers.events.removed.connect(
            self._refresh_registered_widgets
        )

    def register_layer_widget(self, widget):
        """
        Register a magicgui widget containing Image and/or Labels
        selectors that should automatically update when napari layers
        change.
        """
        if widget not in self._layer_widgets:
            self._layer_widgets.append(widget)

    def _refresh_registered_widgets(self, event=None):
        """
        Refresh every registered layer selector.

        This method is automatically called whenever a layer is added
        or removed from the napari viewer.
        """
        self.refresh_layer_choices(*self._layer_widgets)

    def refresh_layer_choices(self, *widgets):
        """
        Refresh every magicgui field that supports dynamic choices.
        """

        for widget in widgets:

            for field in widget:

                if hasattr(field, "reset_choices"):
                    field.reset_choices()

    def get_image_layers(self):
        """
        Return all Image layers currently in the viewer.
        """
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, Image)
        ]

    def get_label_layers(self):
        """
        Return all Labels layers currently in the viewer.
        """
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, Labels)
        ]