"""
Features tab for the CloMA napari plugin.
"""
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QTableView,
)
from magicgui import magicgui
from napari.layers import (
    Image,
    Labels,
)
import pandas as pd
from CloMA.napari_plugin._cloma_tab import CloMATab
from CloMA.feature_extraction import extract_features
from .pandas_model import PandasModel


class FeaturesTab(CloMATab):
    """
    Feature extraction tab.

    This widget allows the user to

    • extract colony features
    • visualize the resulting dataframe
    • sort the table
    • click colonies directly from the table
    • export/import CSV files
    """

    def __init__(self, napari_viewer):

        super().__init__(napari_viewer)

        self.df = None
        self.model = None

        self.extract_gui = magicgui(
            self.run_feature_extraction,

            labels={
                "choices": self.get_label_layers,
            },

            image={
                "choices": self.get_image_layers,
            },

            call_button="Extract Features",
        )

        self.register_layer_widget(self.extract_gui)

        ###############################################################
        # Buttons

        self.import_button = QPushButton("Import CSV")
        self.export_button = QPushButton("Export CSV")

        self.export_button.setEnabled(False)

        self.import_button.clicked.connect(
            self.import_csv
        )

        self.export_button.clicked.connect(
            self.export_csv
        )

        ###############################################################
        # Table

        self.table = QTableView()

        self.table.setSortingEnabled(True)

        self.table.setSelectionBehavior(
            QTableView.SelectRows
        )

        self.table.setSelectionMode(
            QTableView.SingleSelection
        )

        self.table.setAlternatingRowColors(False)

        self.table.verticalHeader().setVisible(False)

        ###############################################################
        # Layout

        layout = QVBoxLayout()

        layout.addWidget(self.extract_gui.native)

        buttons = QHBoxLayout()

        buttons.addWidget(self.import_button)
        buttons.addWidget(self.export_button)

        buttons.addStretch()

        layout.addLayout(buttons)

        layout.addWidget(self.table)

        self.setLayout(layout)

    ####################################################################
    # Feature extraction

    def run_feature_extraction(
        self,
        labels: Labels,
        image: Image = None,
    ):
        """
        Extract colony features from the selected labels layer.
        """

        # Extract features
        self.df = extract_features(
            segmentation=labels.data,
            image=None if image is None else image.data,
        )

        # Update table
        self.update_table()

    ####################################################################
    # Table

    def update_table(self):
        """
        Create or update the table model.
        """

        if self.df is None:
            return

        # First time
        if self.model is None:

            self.model = PandasModel(self.df)

            self.table.setModel(self.model)

            # Connect selection AFTER the model exists
            self.table.selectionModel().selectionChanged.connect(
                self.on_selection_changed
            )

        # Existing model
        else:

            self.model.set_dataframe(self.df)

        # Resize columns to fit their contents
        self.table.resizeColumnsToContents()

        # Allow exporting
        self.export_button.setEnabled(True)

    ####################################################################
    # Import / Export

    def import_csv(self):
        """
        Import a previously exported feature table.
        """

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import features",
            "",
            "CSV (*.csv)",
        )

        if not filename:
            return

        self.df = pd.read_csv(filename)

        self.update_table()

    def export_csv(self):
        """
        Export the current feature table.
        """

        if self.df is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export features",
            "features.csv",
            "CSV (*.csv)",
        )

        if not filename:
            return

        self.model.dataframe.to_csv(
            filename,
            index=False,
        )

    ####################################################################
    # Table interaction

    def on_selection_changed(
        self,
        selected,
        deselected,
    ):
        """
        Highlight the selected colony in napari.
        """

        # No selection
        if not selected.indexes():
            return

        # No dataframe
        if self.model is None:
            return

        # No labels layer available
        if self.extract_gui.labels.value is None:
            return

        # Selected row in the current (possibly sorted) table
        row = selected.indexes()[0].row()

        colony = self.model.row(row)

        self.show_colony(colony)

    ####################################################################
    # Viewer

    def show_colony(self, colony):
        """
        Highlight one colony and center the camera.
        """

        ###############################################################
        # Highlight label

        if "label" in colony.index:

            labels = self.extract_gui.labels.value

            if labels is not None:
                labels.selected_label = int(colony["label"])

        ###############################################################
        # Camera center

        if (
            "x" in colony.index
            and "y" in colony.index
        ):

            self.viewer.camera.center = (
                float(colony["y"]),
                float(colony["x"]),
            )

        ###############################################################
        # Camera zoom

        zoom = self.compute_zoom(colony)

        if zoom is not None:

            self.viewer.camera.zoom = zoom

    ####################################################################
    # Zoom

    def compute_zoom(self, colony):
        """
        Compute a reasonable zoom level for one colony.

        Returns
        -------
        float
            Recommended camera zoom.
        """
        image = self.extract_gui.image.value

        if image is None:
            return None

        image_height, image_width = image.data.shape[:2]

        ###############################################################
        # Estimate colony diameter

        diameter = None

        if "feret_diameter_max" in colony.index:

            diameter = colony["feret_diameter_max"]

        elif "axis_major_length" in colony.index:

            diameter = colony["axis_major_length"]

        elif "equivalent_diameter_area" in colony.index:

            diameter = colony["equivalent_diameter_area"]

        elif "area" in colony.index:

            diameter = (4 * colony["area"] / 3.14159) ** 0.5

        if diameter is None or diameter <= 0:
            return None

        ###############################################################
        # Desired field of view

        margin = 6

        zoom = min(
            image_height,
            image_width,
        ) / (diameter * margin)

        ###############################################################
        # Clamp zoom

        zoom = max(1, zoom)
        zoom = min(40, zoom)

        return zoom