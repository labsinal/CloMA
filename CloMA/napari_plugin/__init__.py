"""
CloMA Napari Plugin - Integration of CloMA (Clonogenic Morphometric Analysis) with Napari

This plugin provides napari integration for:
- Image preprocessing (CLAHE + background subtraction)
- Colony segmentation (hybrid threshold + Cellpose)
- Border filtering
- Feature extraction (morphological + intensity)
"""

__version__ = "0.1.0"

# Import both widget versions
try:
    from ._cloma_advanced_widget import CloMAAdvancedWidget as CloMAWidget
except ImportError:
    from ._cloma_widget import CloMAWidget

__all__ = ["CloMAWidget"]


def create_cloma_widget(napari_viewer) -> CloMAWidget:
    """Factory function for napari plugin discovery"""
    return CloMAWidget(napari_viewer)
