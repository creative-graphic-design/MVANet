"""Tests to verify batch processing produces identical results to single processing."""

import numpy as np
import pytest
import torch
from PIL import Image

from mvanet.predictor import MVANetPredictor


def images_are_similar(
    img1: Image.Image, img2: Image.Image, tolerance: float = 1e-3
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
    # Convert to numpy arrays
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # Check shapes match
    if arr1.shape != arr2.shape:
        return False

    # Calculate relative difference
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    abs_diff = np.abs(arr1 - arr2)
    rel_diff = abs_diff / (np.maximum(arr1, arr2) + epsilon)

    # Check if mean relative difference is within tolerance
    mean_rel_diff = np.mean(rel_diff)

    return mean_rel_diff < tolerance


class TestBatchCorrectness:
    """Tests to verify batch predictions match single predictions."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_batch_vs_single_rgba_small_batch(
        self,
        predictor: MVANetPredictor,
        sample_image: Image.Image,
        small_image: Image.Image,
    ) -> None:
        """Test that batch prediction matches single predictions for 2 images (RGBA output)."""
        images = [sample_image, small_image]

        # Single predictions
        single_results = [predictor(img, output_type="rgba") for img in images]

        # Batch prediction
        batch_results = predictor(images, output_type="rgba")

        # Compare results
        assert len(batch_results) == len(single_results)
        for i, (single, batch) in enumerate(zip(single_results, batch_results)):
            assert single.size == batch.size, f"Image {i}: Size mismatch"
            assert single.mode == batch.mode, f"Image {i}: Mode mismatch"
            assert images_are_similar(single, batch, tolerance=2e-3), (
                f"Image {i}: Predictions differ significantly"
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_batch_vs_single_map_small_batch(
        self,
        predictor: MVANetPredictor,
        sample_image: Image.Image,
        small_image: Image.Image,
    ) -> None:
        """Test that batch prediction matches single predictions for 2 images (map output)."""
        images = [sample_image, small_image]

        # Single predictions
        single_results = [predictor(img, output_type="map") for img in images]

        # Batch prediction
        batch_results = predictor(images, output_type="map")

        # Compare results
        assert len(batch_results) == len(single_results)
        for i, (single, batch) in enumerate(zip(single_results, batch_results)):
            assert single.size == batch.size, f"Image {i}: Size mismatch"
            assert single.mode == batch.mode, f"Image {i}: Mode mismatch"
            assert images_are_similar(single, batch, tolerance=2e-3), (
                f"Image {i}: Predictions differ significantly"
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_batch_vs_single_large_batch(
        self,
        predictor: MVANetPredictor,
        sample_image: Image.Image,
        small_image: Image.Image,
        large_image: Image.Image,
    ) -> None:
        """Test that batch prediction matches single predictions for 3 images."""
        images = [sample_image, small_image, large_image]

        # Single predictions (map output for faster comparison)
        single_results = [predictor(img, output_type="map") for img in images]

        # Batch prediction
        batch_results = predictor(images, output_type="map")

        # Compare results
        assert len(batch_results) == len(single_results)
        for i, (single, batch) in enumerate(zip(single_results, batch_results)):
            assert single.size == batch.size, f"Image {i}: Size mismatch"
            assert single.mode == batch.mode, f"Image {i}: Mode mismatch"
            assert images_are_similar(single, batch, tolerance=2e-3), (
                f"Image {i}: Predictions differ significantly"
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_batch_single_item_matches_direct_single(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test that batch processing with single item matches direct single prediction."""
        # Direct single prediction
        single_result = predictor(sample_image, output_type="map")

        # Batch prediction with single item
        batch_result = predictor([sample_image], output_type="map")[0]

        # Compare results
        assert single_result.size == batch_result.size
        assert single_result.mode == batch_result.mode
        assert images_are_similar(single_result, batch_result, tolerance=2e-3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_numerical_stability_across_batch_sizes(
        self,
        predictor: MVANetPredictor,
        sample_image: Image.Image,
        small_image: Image.Image,
    ) -> None:
        """Test that the same image produces consistent results in different batch positions."""
        # Predict sample_image alone
        single_result = predictor(sample_image, output_type="map")

        # Predict sample_image as first item in batch of 2
        batch_results_first = predictor([sample_image, small_image], output_type="map")

        # Predict sample_image as second item in batch of 2
        batch_results_second = predictor([small_image, sample_image], output_type="map")

        # Compare: single vs first position in batch
        assert images_are_similar(
            single_result, batch_results_first[0], tolerance=2e-3
        ), "Image differs when processed as first item in batch"

        # Compare: single vs second position in batch
        assert images_are_similar(
            single_result, batch_results_second[1], tolerance=2e-3
        ), "Image differs when processed as second item in batch"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_pixel_value_ranges(
        self,
        predictor: MVANetPredictor,
        sample_image: Image.Image,
        small_image: Image.Image,
    ) -> None:
        """Test that batch predictions produce valid pixel value ranges."""
        images = [sample_image, small_image]
        batch_results = predictor(images, output_type="map")

        for i, result in enumerate(batch_results):
            # Convert to array and check range
            arr = np.array(result, dtype=np.uint8)
            assert arr.min() >= 0, f"Image {i}: Pixel values below 0"
            assert arr.max() <= 255, f"Image {i}: Pixel values above 255"
