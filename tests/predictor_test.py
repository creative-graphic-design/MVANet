import pathlib
from typing import get_args

import pytest
import torch
from PIL import Image, ImageChops
from PIL.Image import Image as PilImage

from mvanet.predictor import MVANetPredictor, OutputType


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


@pytest.fixture
def test_fixtures_dir(root_dir: pathlib.Path) -> pathlib.Path:
    return root_dir / "test_fixtures"


@pytest.fixture
def test_image(test_fixtures_dir: pathlib.Path) -> PilImage:
    image_path = test_fixtures_dir / "image_generation_ai_kun.png"
    return Image.open(image_path)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    argnames="output_type",
    argvalues=get_args(OutputType),
)
def test_predictor(
    test_image: PilImage,
    output_type: OutputType,
    test_fixtures_dir: pathlib.Path,
    device: torch.device,
):
    predictor = MVANetPredictor(device=device)

    predicted_image = predictor(test_image, output_type=output_type)

    expected_image_path = test_fixtures_dir / f"expected_{output_type}_{device}.png"
    expected_image = Image.open(expected_image_path)

    diff = ImageChops.difference(predicted_image, expected_image)

    assert diff.getbbox() is None
