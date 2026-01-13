"""Transformers-compatible MVANet implementation."""

from mvanet.transformers.configuration_mvanet import MVANetConfig
from mvanet.transformers.image_processing_mvanet import MVANetImageProcessor
from mvanet.transformers.modeling_mvanet import MVANetForImageSegmentation
from mvanet.transformers.tta_pipeline import MVANetTTAPipeline

# Register with Auto* classes for from_pretrained() support
MVANetConfig.register_for_auto_class()
MVANetForImageSegmentation.register_for_auto_class("AutoModel")
MVANetImageProcessor.register_for_auto_class("AutoImageProcessor")

__all__ = [
    "MVANetConfig",
    "MVANetForImageSegmentation",
    "MVANetImageProcessor",
    "MVANetTTAPipeline",
]
