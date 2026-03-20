"""
Tests for CloMA napari plugin
"""

import pytest
import numpy as np
from napari.layers import Image, Labels
import tempfile
from pathlib import Path


class TestCloMAWidget:
    """Test CloMA widget functionality"""

    @pytest.fixture
    def napari_viewer(self, make_napari_viewer):
        """Create a test napari viewer"""
        return make_napari_viewer()

    @pytest.fixture
    def cloma_widget(self, napari_viewer):
        """Create CloMA widget"""
        from CloMA.napari_plugin import CloMAWidget
        return CloMAWidget(napari_viewer)

    @pytest.fixture
    def sample_image(self):
        """Create sample test image"""
        img = np.random.randint(50, 150, (256, 256), dtype=np.uint8)
        # Add some synthetic colonies
        from skimage.draw import circle
        for _ in range(3):
            y, x = np.random.randint(50, 206, 2)
            r = np.random.randint(20, 40)
            rr, cc = circle(y, x, r)
            img[rr, cc] = 200
        return img

    def test_widget_creation(self, napari_viewer, cloma_widget):
        """Test that widget is created correctly"""
        assert cloma_widget is not None
        assert cloma_widget.viewer == napari_viewer

    def test_get_active_image_layer(self, napari_viewer, cloma_widget, sample_image):
        """Test getting active image layer"""
        napari_viewer.add_image(sample_image, name="test")
        layer = cloma_widget._get_active_image()
        assert layer is not None
        assert isinstance(layer, Image)

    def test_preprocess_image(self, napari_viewer, cloma_widget, sample_image):
        """Test image preprocessing"""
        napari_viewer.add_image(sample_image, name="test")
        cloma_widget._preprocess_image()
        # Should create new layer
        assert len(napari_viewer.layers) > 1

    def test_segment_colonies(self, napari_viewer, cloma_widget, sample_image):
        """Test colony segmentation"""
        napari_viewer.add_image(sample_image, name="test")
        cloma_widget._segment_colonies(radius_factor=0.5, shrink=0.05, filter_border=False)
        # Should create labels layer
        assert any(isinstance(layer, Labels) for layer in napari_viewer.layers)

    def test_extract_features(self, napari_viewer, cloma_widget, sample_image):
        """Test feature extraction"""
        napari_viewer.add_image(sample_image, name="test")
        cloma_widget._segment_colonies(radius_factor=0.5)

        cloma_widget._extract_features()
        assert cloma_widget.features_table is not None
        assert len(cloma_widget.features_table) > 0

    def test_complete_pipeline(self, napari_viewer, cloma_widget, sample_image):
        """Test complete pipeline"""
        napari_viewer.add_image(sample_image, name="test")
        cloma_widget._run_pipeline(radius_factor=0.5, shrink=0.05)

        # Should create multiple layers
        assert len(napari_viewer.layers) > 1
        assert cloma_widget.features_table is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
