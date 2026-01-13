"""Pytest configuration and fixtures for MVANet tests."""

import pytest
import torch
from PIL import Image, ImageDraw

from mvanet.predictor import MVANetPredictor


@pytest.fixture(scope="session")
def predictor() -> MVANetPredictor:
    """Create a predictor instance for testing.

    This fixture is session-scoped to avoid loading the model multiple times,
    which is expensive in terms of time and memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MVANetPredictor(device=device)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a simple test image (800x600 RGB).

    Returns a synthetic image with some colored shapes to test segmentation.
    """
    # Create a white background
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    # Draw some shapes (circle and rectangle)
    draw.ellipse([200, 150, 400, 350], fill="red", outline="black")
    draw.rectangle([450, 200, 650, 400], fill="blue", outline="black")

    return img


@pytest.fixture
def small_image() -> Image.Image:
    """Create a small test image (300x200 RGB)."""
    img = Image.new("RGB", (300, 200), color="lightblue")
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill="yellow")
    return img


@pytest.fixture
def large_image() -> Image.Image:
    """Create a large test image (1920x1080 RGB)."""
    img = Image.new("RGB", (1920, 1080), color="lightgray")
    draw = ImageDraw.Draw(img)
    draw.rectangle([500, 300, 1400, 800], fill="green")
    return img


@pytest.fixture
def sample_images_batch(
    sample_image: Image.Image,
    small_image: Image.Image,
    large_image: Image.Image,
) -> list[Image.Image]:
    """Create a batch of test images with different sizes."""
    return [sample_image, small_image, large_image]


@pytest.fixture
def grayscale_image() -> Image.Image:
    """Create a grayscale test image."""
    img = Image.new("L", (400, 300), color=128)
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 75, 300, 225], fill=255)
    return img


@pytest.fixture
def rgba_image() -> Image.Image:
    """Create an RGBA test image with transparency."""
    img = Image.new("RGBA", (400, 300), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 75, 300, 225], fill=(255, 0, 0, 200))
    return img
