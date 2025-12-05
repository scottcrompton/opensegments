"""
Segmentation Module
===================

This package provides a full hierarchical segmentation engine:
- Macro-level clustering (12 clusters)
- Auto-selected micro-clusters within each macro group (4–6)
- Human-readable segment labels
- Production-ready class-based architecture

Public classes exposed:

    - FeatureBuilder
    - MacroSegmentationModel
    - MicroSegmentationModel
    - SegmentationTrainer
    - SegmentationApplier

Usage example:

    from backend.modules.segmentation import SegmentationTrainer

    trainer = SegmentationTrainer()
    trainer.train(df_all_states)
"""

from .config import SegmentationConfig
from .features import FeatureBuilder
from .trainer import SegmentationTrainer
from .labels import SegmentLabelGenerator

# Optional: SegmentationApplier if apply.py exists
try:
    from .apply import SegmentationApplier
    __all__ = [
        "SegmentationConfig",
        "FeatureBuilder",
        "SegmentationTrainer",
        "SegmentationApplier",
        "SegmentLabelGenerator"
    ]
except ImportError:
    __all__ = [
        "SegmentationConfig",
        "FeatureBuilder",
        "SegmentationTrainer",
        "SegmentLabelGenerator"
    ]
