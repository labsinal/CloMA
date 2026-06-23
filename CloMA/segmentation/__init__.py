from .segmentation import automatic_segmentation, reference_segmentation, binary_segmentation

# Backwards-compatible aliases
segment_well_colonies_hybrid = automatic_segmentation
segment_well_colonies_reference = reference_segmentation

# Do not import the napari plugin here to avoid pulling a heavy GUI
# dependency when the package is used from CLI or scripts.