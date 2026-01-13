"""Tests to verify transformers-compatible implementation matches existing predictor."""

import numpy as np
import pytest
import torch
from PIL import Image

from mvanet_original import MVANetPredictor
from mvanet.transformers import (
    MVANetConfig,
    MVANetForImageSegmentation,
    MVANetImageProcessor,
    MVANetTTAPipeline,
)


def images_are_similar(
    img1: Image.Image, img2: Image.Image, tolerance: float = 5e-3
) -> bool:
    """
    Compare two PIL images for similarity.

    Args:
        img1: First image
        img2: Second image
        tolerance: Maximum allowed relative difference

    Returns:
        True if images are similar within tolerance
    """
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    if arr1.shape != arr2.shape:
        return False

    epsilon = 1e-8
    abs_diff = np.abs(arr1 - arr2)
    rel_diff = abs_diff / (np.maximum(arr1, arr2) + epsilon)

    mean_rel_diff = np.mean(rel_diff)

    # For very small images (< 400x400), use higher tolerance due to precision issues
    # during multiple resize operations
    size = min(arr1.shape[0], arr1.shape[1])
    if size < 400:
        tolerance = tolerance * 3  # Triple tolerance for small images

    return mean_rel_diff < tolerance


@pytest.fixture(scope="module")
def transformers_model():
    """Create transformers model with loaded weights."""
    predictor = MVANetPredictor()
    config = MVANetConfig()
    model = MVANetForImageSegmentation(config)
    # Load weights from predictor's network
    model.load_state_dict(predictor.net.state_dict())
    # Move model to same device as predictor
    model = model.to(predictor.device)
    # Set to eval mode
    model.eval()
    # Manually move positional encoding tensors to the correct device
    for module in model.modules():
        if hasattr(module, "positional_encoding") and hasattr(
            module.positional_encoding, "dim_t"
        ):
            module.positional_encoding.dim_t = module.positional_encoding.dim_t.to(
                predictor.device
            )
    return model


@pytest.fixture(scope="module")
def transformers_processor():
    """Create transformers processor."""
    return MVANetImageProcessor()


class TestOutputCompatibility:
    """Tests to verify new implementation produces same outputs as old."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_single_image_output_matches(
        self,
        sample_image: Image.Image,
        transformers_model: MVANetForImageSegmentation,
        transformers_processor: MVANetImageProcessor,
    ) -> None:
        """Test that transformers model matches predictor output (without TTA)."""
        # Old predictor (without TTA - direct model call)
        predictor = MVANetPredictor()
        original_w, original_h = sample_image.size
        resized_image = sample_image.resize([1024, 1024], Image.Resampling.BILINEAR)
        transformed_image = predictor.image_transform(resized_image)
        transformed_image = transformed_image.unsqueeze(0).to(predictor.device)

        with torch.no_grad():
            old_output = predictor.net(transformed_image)
            old_output = old_output.sigmoid()
            old_mask_pil = predictor.to_pil(old_output.squeeze(0).cpu())
            old_mask_pil = old_mask_pil.resize(
                (original_w, original_h), Image.Resampling.BILINEAR
            )

        # New transformers implementation
        inputs = transformers_processor(sample_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(transformers_model.device)

        with torch.no_grad():
            outputs = transformers_model(pixel_values=pixel_values)
            masks = transformers_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(original_w, original_h)]
            )

        new_mask = masks[0].cpu().numpy()
        new_mask = (new_mask * 255).clip(0, 255).astype(np.uint8)
        new_mask_pil = Image.fromarray(new_mask, mode="L")

        assert images_are_similar(old_mask_pil, new_mask_pil, tolerance=5e-3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_tta_output_matches(
        self,
        sample_image: Image.Image,
        transformers_model: MVANetForImageSegmentation,
        transformers_processor: MVANetImageProcessor,
    ) -> None:
        """Test that TTA pipeline matches predictor output."""
        # Old predictor with TTA (default behavior)
        predictor = MVANetPredictor()
        old_mask_pil = predictor(sample_image, output_type="map")

        # New TTA pipeline
        tta_pipeline = MVANetTTAPipeline(transformers_model, transformers_processor)
        new_masks = tta_pipeline([sample_image], return_tensors=False)
        new_mask_pil = new_masks[0]

        # TTA has more tolerance due to averaging differences
        assert images_are_similar(old_mask_pil, new_mask_pil, tolerance=2e-2)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_batch_output_matches(
        self,
        sample_images_batch: list[Image.Image],
        transformers_model: MVANetForImageSegmentation,
        transformers_processor: MVANetImageProcessor,
    ) -> None:
        """Test that batch processing matches."""
        # Old predictor (without TTA for faster comparison)
        predictor = MVANetPredictor()
        old_masks = []

        for image in sample_images_batch:
            original_w, original_h = image.size
            resized_image = image.resize([1024, 1024], Image.Resampling.BILINEAR)
            transformed_image = predictor.image_transform(resized_image)
            transformed_image = transformed_image.unsqueeze(0).to(predictor.device)

            with torch.no_grad():
                output = predictor.net(transformed_image)
                output = output.sigmoid()
                mask_pil = predictor.to_pil(output.squeeze(0).cpu())
                mask_pil = mask_pil.resize(
                    (original_w, original_h), Image.Resampling.BILINEAR
                )
                old_masks.append(mask_pil)

        # New transformers implementation (batch)
        inputs = transformers_processor(sample_images_batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(transformers_model.device)
        original_sizes = [(img.width, img.height) for img in sample_images_batch]

        with torch.no_grad():
            outputs = transformers_model(pixel_values=pixel_values)
            new_masks = transformers_processor.post_process_semantic_segmentation(
                outputs, target_sizes=original_sizes
            )

        # Compare each image in batch
        for i, (old_mask, new_mask) in enumerate(zip(old_masks, new_masks)):
            new_mask_np = new_mask.cpu().numpy()
            new_mask_np = (new_mask_np * 255).clip(0, 255).astype(np.uint8)
            new_mask_pil = Image.fromarray(new_mask_np, mode="L")

            assert images_are_similar(old_mask, new_mask_pil, tolerance=5e-3), (
                f"Batch image {i} differs"
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_different_sizes(
        self,
        small_image: Image.Image,
        large_image: Image.Image,
        transformers_model: MVANetForImageSegmentation,
        transformers_processor: MVANetImageProcessor,
    ) -> None:
        """Test with different image sizes."""
        for test_image in [small_image, large_image]:
            # Old predictor
            predictor = MVANetPredictor()
            original_w, original_h = test_image.size
            resized_image = test_image.resize([1024, 1024], Image.Resampling.BILINEAR)
            transformed_image = predictor.image_transform(resized_image)
            transformed_image = transformed_image.unsqueeze(0).to(predictor.device)

            with torch.no_grad():
                old_output = predictor.net(transformed_image)
                old_output = old_output.sigmoid()
                old_mask_pil = predictor.to_pil(old_output.squeeze(0).cpu())
                old_mask_pil = old_mask_pil.resize(
                    (original_w, original_h), Image.Resampling.BILINEAR
                )

            # New transformers implementation
            inputs = transformers_processor(test_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(transformers_model.device)

            with torch.no_grad():
                outputs = transformers_model(pixel_values=pixel_values)
                masks = transformers_processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(original_w, original_h)]
                )

            new_mask = masks[0].cpu().numpy()
            new_mask = (new_mask * 255).clip(0, 255).astype(np.uint8)
            new_mask_pil = Image.fromarray(new_mask, mode="L")

            assert images_are_similar(old_mask_pil, new_mask_pil, tolerance=5e-3)


class TestPreprocessing:
    """Tests for preprocessing compatibility."""

    def test_preprocessing_matches(self, sample_image: Image.Image) -> None:
        """Test that preprocessing matches exactly."""
        # Old predictor preprocessing
        predictor = MVANetPredictor()
        resized_image = sample_image.resize([1024, 1024], Image.Resampling.BILINEAR)
        old_transformed = predictor.image_transform(resized_image)

        # New processor preprocessing
        processor = MVANetImageProcessor()
        inputs = processor(sample_image, return_tensors="pt")
        new_transformed = inputs["pixel_values"][0]

        # Should be exactly the same
        assert torch.allclose(old_transformed, new_transformed, atol=1e-6)

    def test_batch_preprocessing(self, sample_images_batch: list[Image.Image]) -> None:
        """Test batch preprocessing."""
        processor = MVANetImageProcessor()
        inputs = processor(sample_images_batch, return_tensors="pt")

        assert inputs["pixel_values"].shape[0] == len(sample_images_batch)
        assert inputs["pixel_values"].shape[1:] == (3, 1024, 1024)


class TestModelAPI:
    """Tests for model API."""

    def test_config_creation(self) -> None:
        """Test config can be created."""
        config = MVANetConfig()
        assert config.embedding_dim == 128
        assert config.image_size == 1024
        assert config.model_type == "mvanet"

    def test_model_creation(self) -> None:
        """Test model can be created."""
        config = MVANetConfig()
        model = MVANetForImageSegmentation(config)
        assert model.config == config

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_model_forward(
        self, transformers_model: MVANetForImageSegmentation
    ) -> None:
        """Test model forward pass."""
        device = transformers_model.device
        x = torch.randn(2, 3, 1024, 1024, device=device)

        with torch.no_grad():
            outputs = transformers_model(x)

        assert outputs.logits.shape == (2, 1, 1024, 1024)
        assert outputs.loss is None

    def test_return_dict(self, transformers_model: MVANetForImageSegmentation) -> None:
        """Test return_dict parameter."""
        device = transformers_model.device
        x = torch.randn(1, 3, 1024, 1024, device=device)

        with torch.no_grad():
            # return_dict=True
            outputs_dict = transformers_model(x, return_dict=True)
            assert hasattr(outputs_dict, "logits")

            # return_dict=False
            outputs_tuple = transformers_model(x, return_dict=False)
            assert isinstance(outputs_tuple, tuple)
