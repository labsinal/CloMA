"""
Main CloMA widget for napari integration
"""

from typing import Optional
from math import floor
import numpy as np
import napari

from napari.layers import Image, Labels
from qtpy.QtWidgets import QApplication, QTableWidgetItem, QHeaderView, QSizePolicy
from qtpy.QtCore import QTimer, Qt

class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically"""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()

class CloMAWidget:
    """Main widget for CloMA napari plugin"""

    def __init__(self, napari_viewer: napari.Viewer):
        self.viewer = napari_viewer
        self._build_ui()

    # =========================
    # 🔄 UI HELPERS
    # =========================

    def _set_status(self, msg: str):
        self.status_label.value = msg
        QApplication.processEvents()
        print(msg)

    def _refresh_layer_choices(self):
        image_layers = [l for l in self.viewer.layers if isinstance(l, Image)]
        label_layers = [l for l in self.viewer.layers if isinstance(l, Labels)]

        if hasattr(self, "seg_image_select"):
            self.seg_image_select.choices = image_layers

        if hasattr(self, "feat_image_select"):
            self.feat_image_select.choices = image_layers

        if hasattr(self, "feat_labels_select"):
            self.feat_labels_select.choices = label_layers

        if hasattr(self, "preprocess_image_select"):
            self.preprocess_image_select.choices = image_layers

        if image_layers and hasattr(self, "seg_image_select"):
            self.seg_image_select.value = image_layers[-1]

        if image_layers and hasattr(self, "preprocess_image_select"):
            self.preprocess_image_select.value = image_layers[-1]

        if label_layers and hasattr(self, "feat_labels_select"):
            self.feat_labels_select.value = label_layers[-1]

        QTimer.singleShot(0, self._update_circle_overlay)

    # =========================
    # 📊 TABLE HANDLING (NEW)
    # =========================

    def _populate_table(self, df):
        """Populate feature table cleanly"""
        table = self.features_table_widget
        table.setSortingEnabled(False)

        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.astype(str).tolist())

        for i in range(len(df)):
            for j in range(len(df.columns)):
                val = df.iat[i, j]

                if isinstance(val, float):
                    text = f"{val:.3f}"
                else:
                    text = str(val)

                item = NumericTableWidgetItem(text)

                if isinstance(val, (int, float)):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    # =========================
    # 📤 EXPORT
    # =========================

    def _export_features(self):
        try:
            if not hasattr(self, "_last_features_df"):
                self._set_status("✗ No features to export")
                return

            from qtpy.QtWidgets import QFileDialog

            file_path, _ = QFileDialog.getSaveFileName(
                None, "Save Features Table", "", "CSV (*.csv);;Excel (*.xlsx)"
            )

            if not file_path:
                return

            if file_path.endswith(".xlsx"):
                self._last_features_df.to_excel(file_path, index=False)
            else:
                if not file_path.endswith(".csv"):
                    file_path += ".csv"
                self._last_features_df.to_csv(file_path, index=False)

            self._set_status(f"✓ Saved to {file_path}")

        except Exception as e:
            self._set_status(f"✗ Export error: {e}")
            raise

    # =========================
    # 🖱 TABLE CLICK
    # =========================

    def _on_table_click(self, row, column):
        try:
            df = self._last_features_df

            # ✅ Read label_id from the table itself, not the dataframe,
            # so it works correctly after the user sorts any column
            label_col = df.columns.get_loc("label") if "label" in df.columns else 0
            label_id = int(self.features_table_widget.item(row, label_col).text())

            labels_layer = self.feat_labels_select.value

            if labels_layer is None:
                return

            labels_layer.selected_label = label_id

            coords = np.argwhere(labels_layer.data == label_id)
            if len(coords) == 0:
                return

            cy, cx = coords.mean(axis=0)
            self.viewer.camera.center = (cy, cx)
            self.viewer.camera.zoom = 2

        except Exception as e:
            self._set_status(f"✗ Click error: {e}")

    # =========================
    # 🎨 ROI CIRCLE
    # =========================

    def _update_circle_overlay(self):
        if getattr(self, "_updating_circle", False):
            return
        self._updating_circle = True

        try:
            layer = self.seg_image_select.value
            if layer is None:
                return

            image = layer.data
            h, w = image.shape[:2]

            max_radius = min(h, w) / 2

            radius_value = self.radius_slider.value
            if hasattr(self, "p_slider"):
                pass

            radius = max(2, int(radius_value * max_radius))
            cy, cx = h // 2, w // 2

            circle = [[cy - radius, cx - radius],
                    [cy - radius, cx + radius],
                    [cy + radius, cx + radius],
                    [cy + radius, cx - radius]]

            if hasattr(self, "_circle_layer"):
                self._circle_layer.data = [circle]
            else:
                self._circle_layer = self.viewer.add_shapes(
                    [circle],
                    shape_type="ellipse",
                    edge_color="red",
                    face_color="transparent",
                    edge_width=3,
                    name="ROI Circle"
                )
        finally:
            self._updating_circle = False
    # =========================
    # 🧪 CORE FUNCTIONS
    # =========================

    def _preprocess_image(self):
        """Preprocess the selected image layer"""
        try:
            from CloMA.extras.preprocess import preprocess_images

            self._set_status("⏳ Preprocessing image...")

            layer = self.preprocess_image_select.value

            if layer is None:
                self._set_status("✗ No image selected")
                return

            image = layer.data.astype(np.uint8)

            # Convert to grayscale if needed
            if image.ndim == 3 and image.shape[2] > 2:
                from cv2 import cvtColor, COLOR_BGR2GRAY
                image = cvtColor(image, COLOR_BGR2GRAY)

            # Run preprocessing
            preprocessed = preprocess_images(image)

            # Add result to viewer
            preprocessed_uint8 = np.clip(preprocessed, 0, 255).astype(np.uint8)

            self.viewer.add_image(
                preprocessed_uint8,
                name=f"preprocessed_{layer.name}",
                colormap="viridis"
            )

            self._set_status("✓ Preprocessing complete!")

        except Exception as e:
            self._set_status(f"✗ Error: {str(e)}")
            raise

    def _extract_features(self, labels_layer, image_layer):
        from CloMA.feature_extraction import extract_features

        self._set_status("⏳ Extracting features...")

        segmentation = labels_layer.data.astype(np.uint32)
        image = image_layer.data if image_layer else None

        df = extract_features(segmentation=segmentation, image=image)

        self._last_features_df = df
        labels_layer.features = df

        self._populate_table(df)

        self.export_btn.enabled = True
        self._set_status(f"✓ {len(df)} colonies")

    def _run_segmentation(self, radius_factor=0.5, shrink=0.05):
        from CloMA.segmentation import segment_well_colonies_hybrid
        from CloMA.extras import filter_border_colonies

        self._set_status("⏳ Segmenting...")

        layer = self.seg_image_select.value

        if layer is None:
            self._set_status("✗ No image selected")
            return

        image = layer.data.astype(np.uint8)

        if image.ndim == 3 and image.shape[2] > 2:
            from cv2 import cvtColor, COLOR_BGR2GRAY
            image = cvtColor(image, COLOR_BGR2GRAY)

        h, w = image.shape[:2]
        radius = int(radius_factor * (min(h, w) / 2))

        seg = segment_well_colonies_hybrid(image=image, radius=radius, shrink=shrink)

        if self.seg_filter_borders.value:  # ✅ checkbox
            seg = filter_border_colonies(seg, radius=floor(radius * (1 - shrink)))

        self.viewer.add_labels(seg.astype(np.uint32), name=f"seg_{layer.name}")
        self._set_status("✓ Segmentation complete!")

    def _run_pipeline(self, radius_factor=0.5, shrink=0.05):
        from CloMA.extras.preprocess import preprocess_images
        from CloMA.segmentation import segment_well_colonies_hybrid
        from CloMA.extras import filter_border_colonies
        from CloMA.feature_extraction import extract_features

        self._set_status("⏳ Running pipeline...")

        layer = self.viewer.layers.selection.active
        image = layer.data

        if image.ndim == 3:
            from cv2 import cvtColor, COLOR_BGR2GRAY
            image = cvtColor(image, COLOR_BGR2GRAY)

        # Preprocess
        pre = preprocess_images(image)

        # Segment
        h, w = image.shape[:2]
        radius = int(radius_factor * (min(h, w) / 2))

        seg = segment_well_colonies_hybrid(image=image, radius=radius, shrink=shrink)
        seg = filter_border_colonies(seg, radius=floor(radius * (1 - shrink)))

        seg_layer = self.viewer.add_labels(seg.astype(np.uint32), name="pipeline_seg")

        # Features
        df = extract_features(segmentation=seg, image=image)

        self._last_features_df = df
        seg_layer.features = df

        # 🔥 UPDATE UI
        self.seg_image_select.value = layer
        self.radius_slider.value = radius_factor
        self._update_circle_overlay()

        self._populate_table(df)
        self.export_btn.enabled = True

        self._set_status(f"✓ Pipeline complete ({len(df)} colonies)")

    def _import_features(self):
        try:
            from qtpy.QtWidgets import QFileDialog
            import pandas as pd

            file_path, _ = QFileDialog.getOpenFileName(
                None, "Open Features Table", "", "CSV (*.csv);;Excel (*.xlsx)"
            )

            if not file_path:
                return

            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            labels_layer = self.feat_labels_select.value

            if labels_layer is None:
                self._set_status("✗ No labels layer selected")
                return

            self._last_features_df = df
            labels_layer.features = df

            self._populate_table(df)
            self.export_btn.enabled = True

            self._set_status(f"✓ Imported {len(df)} rows from {file_path.split('/')[-1]}")

        except Exception as e:
            self._set_status(f"✗ Import error: {e}")
            raise

    # =========================
    # 🧱 UI BUILD
    # =========================

    def _build_ui(self):
        from magicgui.widgets import Container, PushButton, FloatSlider, ComboBox, Label, CheckBox
        from qtpy.QtWidgets import QTabWidget, QTableWidget

        container = Container(layout="vertical")
        tabs = QTabWidget()

        # PREPROCESS TAB
        preprocess_tab = Container()

        preprocess_title = Label(value="<b>Preprocessing</b>")

        self.preprocess_image_select = ComboBox(
            label="Input Image",
            choices=[]
        )

        preprocess_btn = PushButton(text="Preprocess Image")
        preprocess_btn.clicked.connect(self._preprocess_image)

        preprocess_tab.extend([
            preprocess_title,
            self.preprocess_image_select,
            preprocess_btn
        ])

        # SEG TAB
        seg_tab = Container()
        self.seg_image_select = ComboBox(label="Image", choices=[])
        self.radius_slider = FloatSlider(min=0, max=1, value=0.9)
        self.radius_slider.changed.connect(lambda v: self._update_circle_overlay())

        self.seg_filter_borders = CheckBox(value=True, text="Filter border colonies")  # ✅

        btn = PushButton(text="Segment")
        btn.clicked.connect(
            lambda: self._run_segmentation(self.radius_slider.value)
        )

        seg_tab.extend([self.seg_image_select, self.radius_slider, self.seg_filter_borders, btn])

        # FEATURES TAB
        feat_tab = Container()
        self.feat_labels_select = ComboBox(label="Labels", choices=[])
        self.feat_image_select = ComboBox(label="Image", choices=[])
        self.import_btn = PushButton(text="Import")
        self.import_btn.clicked.connect(self._import_features)

        extract_btn = PushButton(text="Extract")
        extract_btn.clicked.connect(
            lambda: self._extract_features(
                self.feat_labels_select.value,
                self.feat_image_select.value,
            )
        )

        self.features_table_widget = QTableWidget()
        self.features_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.features_table_widget.verticalHeader().setVisible(False)
        self.features_table_widget.cellClicked.connect(self._on_table_click)
        self.features_table_widget.setSortingEnabled(True) 

        self.export_btn = PushButton(text="Export")
        self.export_btn.clicked.connect(self._export_features)
        self.export_btn.enabled = False

        feat_tab.extend([
            self.feat_labels_select,
            self.feat_image_select,
            extract_btn,
            self.import_btn,
        ])

        feat_tab.native.layout().addWidget(self.features_table_widget)
        feat_tab.native.layout().addWidget(self.export_btn.native)

        # PIPELINE TAB
        pipe_tab = Container()
        self.p_slider = FloatSlider(min=0, max=1, value=0.5)
        self.p_slider.changed.connect(lambda v: self._update_circle_overlay())  
        p_btn = PushButton(text="Run Pipeline")
        p_btn.clicked.connect(lambda: self._run_pipeline(self.p_slider.value))

        pipe_tab.extend([self.p_slider, p_btn])

        # In _build_ui, after both sliders are created:
        self.radius_slider.changed.connect(
            lambda v: setattr(self.p_slider, 'value', v) or self._update_circle_overlay()
        )
        self.p_slider.changed.connect(
            lambda v: setattr(self.radius_slider, 'value', v) or self._update_circle_overlay()
        )

        preprocess_tab.native.layout().setAlignment(Qt.AlignTop)
        seg_tab.native.layout().setAlignment(Qt.AlignTop)
        feat_tab.native.layout().setAlignment(Qt.AlignTop)
        pipe_tab.native.layout().setAlignment(Qt.AlignTop)

        preprocess_tab.native.layout().addStretch()
        seg_tab.native.layout().addStretch()
        feat_tab.native.layout().addStretch()
        pipe_tab.native.layout().addStretch()

        tabs.addTab(preprocess_tab.native, "Preprocess")
        tabs.addTab(seg_tab.native, "Seg")
        tabs.addTab(feat_tab.native, "Features")
        tabs.addTab(pipe_tab.native, "Pipeline")

        self.status_label = Label(value="Ready")

        container.append(self.status_label)
        container.native.layout().addWidget(tabs)

        self.native = container.native

        self.viewer.layers.events.inserted.connect(lambda e: self._refresh_layer_choices())
        QTimer.singleShot(100, self._refresh_layer_choices)

        # Force everything to stay at the top
        container.native.layout().setAlignment(Qt.AlignTop)

        # Reduce spacing
        container.native.layout().setSpacing(6)
        container.native.layout().setContentsMargins(8, 8, 8, 8)


def create_cloma_widget(viewer=None):
    if viewer is None:
        viewer = napari.current_viewer()
    return CloMAWidget(viewer).native