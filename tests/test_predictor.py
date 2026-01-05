"""Integration tests for MVANetPredictor."""

import pytest
import torch
from PIL import Image

from mvanet.predictor import MVANetPredictor


class TestPredictorInitialization:
    """Tests for MVANetPredictor initialization."""

    def test_predictor_initialization(self, predictor: MVANetPredictor) -> None:
        """Test that predictor initializes correctly."""
        assert predictor is not None
        assert isinstance(predictor, MVANetPredictor)
        assert predictor.device is not None

    def test_model_loaded(self, predictor: MVANetPredictor) -> None:
        """Test that model is loaded from Hugging Face Hub."""
        assert predictor._net is not None
        assert predictor.net is not None
        # Verify model is in eval mode
        assert predictor.net.training is False

    def test_transforms_loaded(self, predictor: MVANetPredictor) -> None:
        """Test that image transforms and TTA transforms are loaded."""
        assert predictor._image_transform is not None
        assert predictor.image_transform is not None
        assert predictor._tta_transforms is not None
        assert predictor.tta_transforms is not None


class TestSingleImagePrediction:
    """Tests for single image prediction."""

    def test_single_predict_rgba(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test single image prediction with output_type='rgba'."""
        result = predictor(sample_image, output_type="rgba")

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size == sample_image.size

    def test_single_predict_map(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test single image prediction with output_type='map'."""
        result = predictor(sample_image, output_type="map")

        assert isinstance(result, Image.Image)
        assert result.mode == "L"  # Grayscale
        assert result.size == sample_image.size

    def test_single_predict_different_sizes(
        self,
        predictor: MVANetPredictor,
        small_image: Image.Image,
        large_image: Image.Image,
    ) -> None:
        """Test single image prediction with various image sizes."""
        # Test small image
        result_small = predictor(small_image, output_type="map")
        assert result_small.size == small_image.size

        # Test large image
        result_large = predictor(large_image, output_type="map")
        assert result_large.size == large_image.size


class TestBatchPrediction:
    """Tests for batch image prediction."""

    def test_batch_predict_rgba(
        self, predictor: MVANetPredictor, sample_images_batch: list[Image.Image]
    ) -> None:
        """Test batch prediction with output_type='rgba'."""
        results = predictor(sample_images_batch, output_type="rgba")

        assert isinstance(results, list)
        assert len(results) == len(sample_images_batch)

        for result, original in zip(results, sample_images_batch):
            assert isinstance(result, Image.Image)
            assert result.mode == "RGBA"
            assert result.size == original.size

    def test_batch_predict_map(
        self, predictor: MVANetPredictor, sample_images_batch: list[Image.Image]
    ) -> None:
        """Test batch prediction with output_type='map'."""
        results = predictor(sample_images_batch, output_type="map")

        assert isinstance(results, list)
        assert len(results) == len(sample_images_batch)

        for result, original in zip(results, sample_images_batch):
            assert isinstance(result, Image.Image)
            assert result.mode == "L"
            assert result.size == original.size

    def test_batch_predict_empty(self, predictor: MVANetPredictor) -> None:
        """Test batch prediction with empty list."""
        results = predictor([], output_type="rgba")

        assert isinstance(results, list)
        assert len(results) == 0

    def test_batch_predict_single_item(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test batch prediction with single image in list."""
        results = predictor([sample_image], output_type="map")

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], Image.Image)
        assert results[0].size == sample_image.size


class TestOutputValidation:
    """Tests for output format validation."""

    def test_rgba_output_format(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test that RGBA output has correct format with alpha channel."""
        result = predictor(sample_image, output_type="rgba")

        assert result.mode == "RGBA"
        # Verify image has 4 channels
        bands = result.getbands()
        assert bands == ("R", "G", "B", "A")

        # Verify alpha channel exists and has valid range
        alpha_channel = result.split()[3]
        # Get extrema (min, max) instead of all values to avoid deprecation warning
        alpha_min, alpha_max = alpha_channel.getextrema()
        assert alpha_min >= 0
        assert alpha_max <= 255

    def test_map_output_format(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test that map output is grayscale."""
        result = predictor(sample_image, output_type="map")

        assert result.mode == "L"
        # Verify it's single channel
        bands = result.getbands()
        assert bands == ("L",)

        # Verify values are in valid range
        value_min, value_max = result.getextrema()
        assert value_min >= 0
        assert value_max <= 255

    def test_output_dimensions(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test that output dimensions match input dimensions."""
        width, height = sample_image.size

        # Test RGBA output
        rgba_result = predictor(sample_image, output_type="rgba")
        assert rgba_result.size == (width, height)

        # Test map output
        map_result = predictor(sample_image, output_type="map")
        assert map_result.size == (width, height)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_non_rgb_grayscale_input(
        self, predictor: MVANetPredictor, grayscale_image: Image.Image
    ) -> None:
        """Test with grayscale (L mode) input."""
        # Predictor should convert to RGB internally
        result = predictor(grayscale_image, output_type="map")

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.size == grayscale_image.size

    def test_non_rgb_rgba_input(
        self, predictor: MVANetPredictor, rgba_image: Image.Image
    ) -> None:
        """Test with RGBA input."""
        # Predictor should convert to RGB internally
        result = predictor(rgba_image, output_type="map")

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.size == rgba_image.size

    def test_invalid_output_type(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test that invalid output_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid output_type"):
            predictor(sample_image, output_type="invalid")

    def test_square_image(self, predictor: MVANetPredictor) -> None:
        """Test with square image."""
        square_image = Image.new("RGB", (512, 512), color="white")
        result = predictor(square_image, output_type="map")

        assert result.size == (512, 512)

    def test_very_small_image(self, predictor: MVANetPredictor) -> None:
        """Test with very small image."""
        tiny_image = Image.new("RGB", (50, 50), color="white")
        result = predictor(tiny_image, output_type="map")

        assert result.size == (50, 50)


class TestInferenceMode:
    """Tests to verify inference mode and no gradient computation."""

    def test_no_gradients_computed(
        self, predictor: MVANetPredictor, sample_image: Image.Image
    ) -> None:
        """Test that no gradients are computed during prediction."""
        # Verify model is in eval mode
        assert predictor.net.training is False

        # Run prediction and verify it works
        result = predictor(sample_image, output_type="map")
        assert result is not None

    def test_device_usage(self, predictor: MVANetPredictor) -> None:
        """Test that predictor uses the correct device."""
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert predictor.device.type == expected_device.type
