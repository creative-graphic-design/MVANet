"""Test Time Augmentation (TTA) pipeline for MVANet."""

from typing import List, Union

import torch
import ttach as tta
from PIL import Image

from mvanet.transformers.image_processing_mvanet import MVANetImageProcessor
from mvanet.transformers.modeling_mvanet import MVANetForImageSegmentation


class MVANetTTAPipeline:
    """
    Test-Time Augmentation pipeline for MVANet.

    This pipeline applies test-time augmentation (TTA) to improve prediction quality
    by averaging predictions over multiple augmented versions of the input image.

    Args:
        model (:class:`~mvanet.transformers.MVANetForImageSegmentation`):
            The MVANet model to use for predictions.
        processor (:class:`~mvanet.transformers.MVANetImageProcessor`):
            The image processor for preprocessing.
        tta_scales (:obj:`List[float]`, `optional`, defaults to :obj:`[0.75, 1.0, 1.25]`):
            Scales to use for multi-scale TTA.
        tta_horizontal_flip (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to apply horizontal flip augmentation.

    Example::

        >>> from mvanet.transformers import MVANetForImageSegmentation, MVANetImageProcessor, MVANetTTAPipeline
        >>> from PIL import Image

        >>> model = MVANetForImageSegmentation.from_pretrained("creative-graphic-design/mvanet")
        >>> processor = MVANetImageProcessor()
        >>> tta_pipeline = MVANetTTAPipeline(model, processor)

        >>> image = Image.open("image.png")
        >>> masks = tta_pipeline([image])
        >>> mask_pil = transforms.ToPILImage()(masks[0])
    """

    def __init__(
        self,
        model: MVANetForImageSegmentation,
        processor: MVANetImageProcessor,
        tta_scales: List[float] | None = None,
        tta_horizontal_flip: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.tta_scales = tta_scales if tta_scales is not None else [0.75, 1.0, 1.25]
        self.tta_horizontal_flip = tta_horizontal_flip

        # Create TTA transforms
        self._create_tta_transforms()

    def _create_tta_transforms(self):
        """Create TTA transforms using ttach library."""
        transforms_list = []

        if self.tta_horizontal_flip:
            transforms_list.append(tta.HorizontalFlip())

        transforms_list.append(
            tta.Scale(
                scales=self.tta_scales,
                interpolation="bilinear",
                align_corners=False,
            )
        )

        self.tta_transforms = tta.Compose(transforms_list)

    @torch.inference_mode()
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: bool = True,
    ) -> List[torch.Tensor]:
        """
        Run inference with TTA on input images.

        Args:
            images (:obj:`PIL.Image.Image` or :obj:`List[PIL.Image.Image]`):
                Input image(s) to process.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to return tensors or PIL Images.

        Returns:
            :obj:`List[torch.Tensor]`: List of segmentation masks (values in [0, 1]).
                Each mask has shape (H, W) where H, W are the original image dimensions.
        """
        # Convert single image to list
        if not isinstance(images, list):
            images = [images]

        # Store original sizes
        original_sizes = [(img.height, img.width) for img in images]

        # Preprocess images
        inputs = self.processor(images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.device)

        # Apply TTA
        masks_per_image = []
        for tta_transform in self.tta_transforms:
            # Augment input
            augmented = tta_transform.augment_image(pixel_values)

            # Forward pass
            outputs = self.model(pixel_values=augmented)

            # Deaugment output
            deaugmented = tta_transform.deaugment_mask(outputs.logits)
            masks_per_image.append(deaugmented)

        # Average predictions across TTA transforms
        averaged_logits = torch.mean(torch.stack(masks_per_image, dim=0), dim=0)

        # Apply sigmoid to get probabilities
        probs = averaged_logits.sigmoid()

        # Post-process to original sizes
        final_masks = []
        for i, (orig_h, orig_w) in enumerate(original_sizes):
            mask = torch.nn.functional.interpolate(
                probs[i : i + 1],
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            final_masks.append(mask.squeeze(0).squeeze(0))  # (H, W)

        if not return_tensors:
            # Convert to PIL Images
            from torchvision import transforms

            to_pil = transforms.ToPILImage()
            final_masks = [to_pil(mask.cpu()) for mask in final_masks]

        return final_masks

    def predict_single(self, image: Image.Image) -> torch.Tensor:
        """
        Predict on a single image with TTA.

        Args:
            image (:obj:`PIL.Image.Image`): Input image.

        Returns:
            :obj:`torch.Tensor`: Segmentation mask with shape (H, W).
        """
        return self([image])[0]

    def predict_batch(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Predict on a batch of images with TTA.

        Args:
            images (:obj:`List[PIL.Image.Image]`): List of input images.

        Returns:
            :obj:`List[torch.Tensor]`: List of segmentation masks.
        """
        return self(images)
