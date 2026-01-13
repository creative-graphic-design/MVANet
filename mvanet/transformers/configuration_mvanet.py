"""MVANet model configuration."""

from typing import List

from transformers import PretrainedConfig


class MVANetConfig(PretrainedConfig):
    """
    Configuration class for MVANet model.

    This is the configuration class to store the configuration of a
    :class:`~mvanet.transformers.MVANetForImageSegmentation`.
    It is used to instantiate a MVANet model according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and
    can be used to control the model outputs. Read the documentation from
    :class:`~transformers.PretrainedConfig` for more information.

    Args:
        embedding_dim (:obj:`int`, `optional`, defaults to 128):
            The embedding dimension used throughout the model.
        backbone_type (:obj:`str`, `optional`, defaults to :obj:`"swinb"`):
            Type of backbone to use. Currently only "swinb" (Swin Transformer Base) is supported.
        backbone_pretrained (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to use pretrained weights for the backbone.
        backbone_out_channels (:obj:`List[int]`, `optional`, defaults to :obj:`[128, 128, 256, 512, 1024]`):
            Output channel dimensions for each backbone level (SwinB specific).
        mclm_num_heads (:obj:`int`, `optional`, defaults to 1):
            Number of attention heads in Multi-field Cross Localization Module (MCLM).
        mclm_pool_ratios (:obj:`List[int]`, `optional`, defaults to :obj:`[1, 4, 8]`):
            Pool ratios for MCLM multi-scale attention.
        mcrm_num_heads (:obj:`int`, `optional`, defaults to 1):
            Number of attention heads in Multi-crop Refinement Module (MCRM).
        mcrm_pool_ratios (:obj:`List[int]`, `optional`, defaults to :obj:`[2, 4, 8]`):
            Pool ratios for MCRM multi-scale attention.
        insmask_hidden_dim (:obj:`int`, `optional`, defaults to 384):
            Hidden dimension in the instance mask head.
        global_view_scale (:obj:`float`, `optional`, defaults to 0.5):
            Scale factor for creating the global view (downsampled version of input).
        num_patches (:obj:`int`, `optional`, defaults to 4):
            Number of local patches (currently only 4 for 2x2 grid is supported).
        image_size (:obj:`int`, `optional`, defaults to 1024):
            Input image size the model was trained on.
        num_channels (:obj:`int`, `optional`, defaults to 3):
            Number of input channels (3 for RGB images).
        num_labels (:obj:`int`, `optional`, defaults to 1):
            Number of output labels (1 for binary segmentation).

    Example::

        >>> from mvanet.transformers import MVANetConfig, MVANetForImageSegmentation

        >>> # Initializing a MVANet configuration
        >>> configuration = MVANetConfig()

        >>> # Initializing a model from the configuration
        >>> model = MVANetForImageSegmentation(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "mvanet"

    def __init__(
        self,
        embedding_dim: int = 128,
        backbone_type: str = "swinb",
        backbone_pretrained: bool = True,
        backbone_out_channels: List[int] | None = None,
        mclm_num_heads: int = 1,
        mclm_pool_ratios: List[int] | None = None,
        mcrm_num_heads: int = 1,
        mcrm_pool_ratios: List[int] | None = None,
        insmask_hidden_dim: int = 384,
        global_view_scale: float = 0.5,
        num_patches: int = 4,
        image_size: int = 1024,
        num_channels: int = 3,
        num_labels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.backbone_type = backbone_type
        self.backbone_pretrained = backbone_pretrained
        # SwinB backbone output channels: [128, 128, 256, 512, 1024]
        self.backbone_out_channels = (
            backbone_out_channels
            if backbone_out_channels is not None
            else [128, 128, 256, 512, 1024]
        )
        self.mclm_num_heads = mclm_num_heads
        self.mclm_pool_ratios = (
            mclm_pool_ratios if mclm_pool_ratios is not None else [1, 4, 8]
        )
        self.mcrm_num_heads = mcrm_num_heads
        self.mcrm_pool_ratios = (
            mcrm_pool_ratios if mcrm_pool_ratios is not None else [2, 4, 8]
        )
        self.insmask_hidden_dim = insmask_hidden_dim
        self.global_view_scale = global_view_scale
        self.num_patches = num_patches
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
