"""
Advanced CloMA widget with better UI using magicgui
This version provides a more polished napari experience
"""

from typing import Optional, Union
from math import floor
import numpy as np
from napari.layers import Image, Labels
import napari
import pandas as pd


class CloMAAdvancedWidget:
    """Advanced CloMA widget with improved UI"""

    def __init__(self, napari_viewer: napari.Viewer):
        self.viewer = napari_viewer
        self.features_table: Optional[pd.DataFrame] = None
        self._build_ui()

    def _build_ui(self):
        """Build the UI with better organization"""
        try:
            from magicgui import magicgui
            from magicgui.widgets import (
                Container, PushButton, FloatSlider, Checkbox, Label, LineEdit
            )

            # Main container
            container = Container(layout="vertical")

            # Header
            title = Label(value="<h2>🔬 CloMA Napari</h2>")
            subtitle = Label(
                value="<small>Clonogenic Morphometric Analysis<br/>Select a layer and choose a tool below</small>"
            )
            container.append(title)
            container.append(subtitle)

            # Separator
            sep1 = Label(value="<hr/>")
            container.append(sep1)

            # ===== PREPROCESSING SECTION =====
            prep_title = Label(value="<b>📊 Preprocessing</b>")
            container.append(prep_title)

            prep_btn = PushButton(text="Preprocess Image", tooltip="Apply CLAHE + background subtraction")
            prep_btn.clicked.connect(self._preprocess_image)
            container.append(prep_btn)

            container.append(Label(value="<hr/>"))

            # ===== SEGMENTATION SECTION =====
            seg_title = Label(value="<b>🎯 Segmentation</b>")
            container.append(seg_title)

            radius_slider = FloatSlider(
                value=0.5,
                min=0.1,
                max=1.0,
                step=0.05,
                label="Radius Factor"
            )

            shrink_slider = FloatSlider(
                value=0.05,
                min=0.0,
                max=0.2,
                step=0.01,
                label="Border Shrink"
            )

            filter_check = Checkbox(value=True, text="Filter Border Colonies")

            seg_btn = PushButton(text="Segment Colonies", tooltip="Hybrid threshold + Cellpose segmentation")
            seg_btn.clicked.connect(
                lambda: self._segment_colonies(
                    radius_factor=radius_slider.value,
                    shrink=shrink_slider.value,
                    filter_border=filter_check.value
                )
            )

            container.append(radius_slider)
            container.append(shrink_slider)
            container.append(filter_check)
            container.append(seg_btn)

            container.append(Label(value="<hr/>"))

            # ===== FEATURE EXTRACTION SECTION =====
            feat_title = Label(value="<b>📈 Analysis</b>")
            container.append(feat_title)

            extract_btn = PushButton(text="Extract Features", tooltip="Extract morphological features from segmentation")
            extract_btn.clicked.connect(self._extract_features)
            container.append(extract_btn)

            container.append(Label(value="<hr/>"))

            # ===== COMPLETE PIPELINE SECTION =====
            pipe_title = Label(value="<b>🚀 Complete Pipeline</b>")
            container.append(pipe_title)

            pipe_radius = FloatSlider(
                value=0.5,
                min=0.1,
                max=1.0,
                step=0.05,
                label="Radius Factor"
            )

            pipe_shrink = FloatSlider(
                value=0.05,
                min=0.0,
                max=0.2,
                step=0.01,
                label="Border Shrink"
            )

            pipe_btn = PushButton(
                text="▶ Run Complete Pipeline",
                tooltip="Preprocess → Segment → Filter → Extract (all in one!)"
            )
            pipe_btn.clicked.connect(
                lambda: self._run_pipeline(
                    radius_factor=pipe_radius.value,
                    shrink=pipe_shrink.value
                )
            )

            container.append(pipe_radius)
            container.append(pipe_shrink)
            container.append(pipe_btn)

            container.append(Label(value="<hr/>"))

            # ===== STATUS SECTION =====
            self.status_label = Label(value="Ready to analyze")
            container.append(self.status_label)

            self.container = container
            self.native = container.native

        except ImportError:
            print("magicgui not installed. Using basic widget instead.")
            self._build_basic_ui()

    def _build_basic_ui(self):
        """Fallback for basic UI without magicgui"""
        from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QCheckBox
        from qtpy.QtCore import Qt

        widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("CloMA Napari Tools")
        layout.addWidget(title)

        # Preprocess button
        prep_btn = QPushButton("Preprocess Image")
        prep_btn.clicked.connect(self._preprocess_image)
        layout.addWidget(prep_btn)

        # Segment button
        seg_btn = QPushButton("Segment Colonies")
        seg_btn.clicked.connect(lambda: self._segment_colonies(0.5, 0.05, True))
        layout.addWidget(seg_btn)

        # Extract button
        extract_btn = QPushButton("Extract Features")
        extract_btn.clicked.connect(self._extract_features)
        layout.addWidget(extract_btn)

        # Pipeline button
        pipeline_btn = QPushButton("Run Complete Pipeline")
        pipeline_btn.clicked.connect(lambda: self._run_pipeline(0.5, 0.05))
        layout.addWidget(pipeline_btn)

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        layout.addStretch()
        widget.setLayout(layout)
        self.native = widget

    def _set_status(self, msg: str, is_error: bool = False):
        """Update status message"""
        prefix = "❌ " if is_error else "✓ "
        status_msg = f"{prefix} {msg}"
        print(status_msg)
        try:
            self.status_label.value = status_msg
        except:
            pass

    def _get_active_image(self) -> Optional[Image]:
        """Get active image layer"""
        if not self.viewer.layers:
            self._set_status("No layers available", is_error=True)
            return None

        # Active layer
        if self.viewer.active_layer and isinstance(self.viewer.active_layer, Image):
            return self.viewer.active_layer

        # First image layer
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                return layer

        self._set_status("No image layer found", is_error=True)
        return None

    def _get_segmentation(self) -> Optional[Labels]:
        """Get most recent segmentation layer"""
        for layer in reversed(self.viewer.layers):
            if isinstance(layer, Labels):
                return layer
        return None

    def _preprocess_image(self):
        """Preprocess selected image"""
        try:
            from CloMA.extras.preprocess import preprocess_images

            self._set_status("Preprocessing image...")

            layer = self._get_active_image()
            if not layer:
                return

            image = layer.data.astype(np.uint8)
            if image.ndim == 3 and image.shape[2] > 2:
                from cv2 import cvtColor, COLOR_BGR2GRAY
                image = cvtColor(image, COLOR_BGR2GRAY)

            processed = preprocess_images(image)
            processed = np.clip(processed, 0, 255).astype(np.uint8)

            self.viewer.add_image(processed, name=f"preprocessed_{layer.name}", colormap="viridis")
            self._set_status(f"Preprocessing complete - created 'preprocessed_{layer.name}'")

        except Exception as e:
            self._set_status(f"Preprocessing failed: {str(e)}", is_error=True)
            raise

    def _segment_colonies(self, radius_factor: float = 0.5, shrink: float = 0.05, filter_border: bool = True):
        """Segment colonies"""
        try:
            from CloMA.segmentation import segment_well_colonies_hybrid
            from CloMA.extras import filter_border_colonies

            self._set_status("Segmenting colonies...")

            layer = self._get_active_image()
            if not layer:
                return

            image = layer.data
            if image.ndim == 3 and image.shape[2] > 2:
                from cv2 import cvtColor, COLOR_BGR2GRAY
                image = cvtColor(image, COLOR_BGR2GRAY)

            radius = image.shape[0] / 2 * radius_factor if radius_factor > 0 else image.shape[0] / 2

            seg = segment_well_colonies_hybrid(image=image, radius=radius, shrink=shrink)

            if filter_border:
                seg = filter_border_colonies(seg, radius=floor(radius * (1 - shrink)))

            self.viewer.add_labels(seg.astype(np.uint32), name=f"segmentation_{layer.name}")
            self._set_status(f"Segmentation complete - created 'segmentation_{layer.name}'")

        except Exception as e:
            self._set_status(f"Segmentation failed: {str(e)}", is_error=True)
            raise

    def _extract_features(self):
        """Extract features from segmentation"""
        try:
            from CloMA.feature_extraction import extract_features

            self._set_status("Extracting features...")

            seg_layer = self._get_segmentation()
            if not seg_layer:
                self._set_status("No segmentation layer found - run Segment Colonies first", is_error=True)
                return

            seg = seg_layer.data

            # Find image
            image = None
            for layer in self.viewer.layers:
                if isinstance(layer, Image):
                    image = layer.data
                    break

            # Extract
            self.features_table = extract_features(segmentation=seg.astype(np.uint32), image=image)

            # Store in viewer
            try:
                self.viewer.window.qt_viewer.viewer.cloma_features = self.features_table
            except:
                pass

            n_colonies = len(self.features_table)
            mean_area = self.features_table["area"].mean()
            mean_perim = self.features_table["perimeter"].mean()

            summary = f"Extracted features from {n_colonies} colonies | Mean area: {mean_area:.0f} | Mean perimeter: {mean_perim:.0f}"
            self._set_status(summary)

            print(f"\n{'='*60}")
            print(f"Feature Extraction Results")
            print(f"{'='*60}")
            print(f"Colonies detected: {n_colonies}")
            print(f"\nStatistics:")
            print(self.features_table[["area", "perimeter", "eccentricity", "solidity"]].describe())
            print(f"\nAccess full table: viewer.cloma_features")
            print(f"{'='*60}\n")

        except Exception as e:
            self._set_status(f"Feature extraction failed: {str(e)}", is_error=True)
            raise

    def _run_pipeline(self, radius_factor: float = 0.5, shrink: float = 0.05):
        """Run complete pipeline"""
        try:
            from CloMA.extras.preprocess import preprocess_images
            from CloMA.segmentation import segment_well_colonies_hybrid
            from CloMA.extras import filter_border_colonies
            from CloMA.feature_extraction import extract_features

            self._set_status("⏳ Pipeline starting... (1/4 preprocessing)")

            layer = self._get_active_image()
            if not layer:
                return

            image = layer.data
            if image.ndim == 3 and image.shape[2] > 2:
                from cv2 import cvtColor, COLOR_BGR2GRAY
                image = cvtColor(image, COLOR_BGR2GRAY)

            # 1. Preprocess
            preprocessed = preprocess_images(image)
            preprocessed = np.clip(preprocessed, 0, 255).astype(np.uint8)
            self.viewer.add_image(preprocessed, name="pipeline_preprocessed", colormap="viridis")

            # 2. Segment
            self._set_status("⏳ Pipeline running... (2/4 segmentation)")
            radius = image.shape[0] / 2 * radius_factor if radius_factor > 0 else image.shape[0] / 2
            seg = segment_well_colonies_hybrid(image=image, radius=radius, shrink=shrink)

            # 3. Filter
            self._set_status("⏳ Pipeline running... (3/4 filtering)")
            seg = filter_border_colonies(seg, radius=floor(radius * (1 - shrink)))
            self.viewer.add_labels(seg.astype(np.uint32), name="pipeline_segmentation")

            # 4. Extract
            self._set_status("⏳ Pipeline running... (4/4 feature extraction)")
            self.features_table = extract_features(segmentation=seg.astype(np.uint32), image=image)

            try:
                self.viewer.window.qt_viewer.viewer.cloma_features = self.features_table
            except:
                pass

            n_colonies = len(self.features_table)
            self._set_status(f"Pipeline complete! Found {n_colonies} colonies")

            print(f"\n{'='*60}")
            print(f"PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Colonies: {n_colonies}")
            print(f"Mean area: {self.features_table['area'].mean():.1f} pixels")
            print(f"Mean perimeter: {self.features_table['perimeter'].mean():.1f} pixels")
            print(f"\nFeature table statistics:")
            print(self.features_table.describe())
            print(f"\nAccess results: viewer.cloma_features")
            print(f"{'='*60}\n")

        except Exception as e:
            self._set_status(f"Pipeline failed: {str(e)}", is_error=True)
            raise
