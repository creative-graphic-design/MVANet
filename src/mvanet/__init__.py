"""MVANet: Multi-view Aggregation Network for Dichotomous Image Segmentation."""

# Transformers API (recommended)
from mvanet.transformers import (
    MVANetConfig,
    MVANetForImageSegmentation,
    MVANetImageProcessor,
    MVANetTTAPipeline,
)

# Legacy API (requires mvanet-original dependency group)
try:
    from mvanet_original import MVANetPredictor
except ImportError:
    MVANetPredictor = None  # type: ignore

__version__ = "0.1.0"

__all__ = [
    # Legacy
    "MVANetPredictor",
    # Transformers
    "MVANetConfig",
    "MVANetForImageSegmentation",
    "MVANetImageProcessor",
    "MVANetTTAPipeline",
]
