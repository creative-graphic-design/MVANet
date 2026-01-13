"""Image processor for MVANet model."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    ImageInput,
    PILImageResampling,
)
from transformers.utils import TensorType


def to_pil_image(image: Union[np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
    """Convert various image formats to PIL Image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        # (C, H, W) tensor
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            # Grayscale
            return Image.fromarray(image, mode="L")
        elif image.ndim == 3:
            if image.shape[2] == 1:
                return Image.fromarray(image.squeeze(2), mode="L")
            elif image.shape[2] == 3:
                return Image.fromarray(image, mode="RGB")
            elif image.shape[2] == 4:
                return Image.fromarray(image, mode="RGBA")
    raise ValueError(f"Unsupported image type: {type(image)}")


class MVANetImageProcessor(BaseImageProcessor):
    """
    Constructs a MVANet image processor.

    Args:
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the image.
        size (:obj:`Dict[str, int]`, `optional`, defaults to :obj:`{"height": 1024, "width": 1024}`):
            Target size for resizing. MVANet was trained on 1024x1024 images.
        resample (:obj:`PILImageResampling`, `optional`, defaults to :obj:`PILImageResampling.BILINEAR`):
            Resampling filter to use when resizing the image.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to normalize the image.
        image_mean (:obj:`List[float]`, `optional`, defaults to :obj:`[0.485, 0.456, 0.406]`):
            Mean to use for normalization (ImageNet mean).
        image_std (:obj:`List[float]`, `optional`, defaults to :obj:`[0.229, 0.224, 0.225]`):
            Standard deviation to use for normalization (ImageNet std).
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 1024, "width": 1024}
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        )
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]

    def resize(
        self,
        image: Image.Image,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
    ) -> Image.Image:
        """Resize image to target size."""
        target_height = size["height"]
        target_width = size["width"]
        return image.resize((target_width, target_height), resample)

    def normalize(
        self,
        image: np.ndarray,
        mean: List[float],
        std: List[float],
    ) -> np.ndarray:
        """Normalize image with mean and std."""
        image = image.astype(np.float32) / 255.0
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        image = (image - mean) / std
        return image

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images for MVANet.

        Args:
            images (:obj:`ImageInput`):
                Images to preprocess. Can be a single image or a batch of images.
            do_resize (:obj:`bool`, `optional`):
                Whether to resize the image(s). Defaults to :obj:`self.do_resize`.
            size (:obj:`Dict[str, int]`, `optional`):
                Target size for resizing. Defaults to :obj:`self.size`.
            resample (:obj:`PILImageResampling`, `optional`):
                Resampling filter to use. Defaults to :obj:`self.resample`.
            do_normalize (:obj:`bool`, `optional`):
                Whether to normalize the image(s). Defaults to :obj:`self.do_normalize`.
            image_mean (:obj:`List[float]`, `optional`):
                Mean for normalization. Defaults to :obj:`self.image_mean`.
            image_std (:obj:`List[float]`, `optional`):
                Std for normalization. Defaults to :obj:`self.image_std`.
            return_tensors (:obj:`str` or :obj:`TensorType`, `optional`):
                Type of tensors to return. Can be 'pt' for PyTorch.

        Returns:
            :obj:`BatchFeature`: A :obj:`BatchFeature` with the following fields:
                - pixel_values (:obj:`torch.Tensor`): Preprocessed images.
        """
        # Set defaults
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        # Convert to list if single image
        if not isinstance(images, list):
            images = [images]

        # Convert to PIL Images
        pil_images = []
        # original_sizes = []
        for img in images:
            pil_img = to_pil_image(img)
            # Convert to RGB if not already
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            # original_sizes.append(pil_img.size)  # (width, height)
            pil_images.append(pil_img)

        # Resize
        if do_resize:
            pil_images = [self.resize(img, size, resample) for img in pil_images]

        # Convert to numpy arrays (H, W, C)
        np_images = [np.array(img) for img in pil_images]

        # Normalize
        if do_normalize:
            np_images = [
                self.normalize(img, image_mean, image_std) for img in np_images
            ]

        # Convert to (C, H, W) format
        np_images = [img.transpose(2, 0, 1) for img in np_images]

        # Convert to tensors
        if return_tensors == "pt":
            pixel_values = torch.tensor(np.stack(np_images), dtype=torch.float32)
        else:
            pixel_values = np.stack(np_images)

        # Store original sizes as metadata (for post-processing)
        data = {
            "pixel_values": pixel_values,
            # "original_sizes": original_sizes,  # List of (width, height) tuples
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        """
        Post-process model outputs to semantic segmentation masks.

        Args:
            outputs (:obj:`SemanticSegmenterOutput` or :obj:`torch.Tensor`):
                Model outputs containing logits.
            target_sizes (:obj:`List[Tuple[int, int]]`, `optional`):
                List of target sizes (width, height) for each image.
                If not provided, returns masks at model output size.

        Returns:
            :obj:`List[torch.Tensor]`: List of segmentation masks (values in [0, 1]).
        """
        # Extract logits from outputs
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)  # (B, 1, H, W)

        # Resize to target sizes if provided
        if target_sizes is not None:
            masks = []
            for i, (target_w, target_h) in enumerate(target_sizes):
                mask = F.interpolate(
                    probs[i : i + 1],
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                masks.append(mask.squeeze(0).squeeze(0))  # (H, W)
            return masks

        # Return at original size
        return [
            probs[i].squeeze(0) for i in range(probs.shape[0])
        ]  # List of (1, H, W) or (H, W)
