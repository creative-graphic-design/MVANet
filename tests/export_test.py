from typing import List

import pytest
import torch
import torch.nn as nn
import torch_tensorrt

from mvanet.predictor import MVANetPredictor

# @pytest.fixture(autouse=True)
# def setup():
#     torch._dynamo.config.inline_inbuilt_nn_modules = True


@pytest.fixture(scope="module")
def model() -> nn.Module:
    predictor = MVANetPredictor()
    return predictor.net


@pytest.fixture
def inputs(device: torch.device) -> torch.Tensor:
    return torch.randn((1, 3, 1024, 1024), device=device)


def test_mvanet_raw(model: nn.Module, inputs: torch.Tensor):
    model(inputs)


def test_mvanet_default_export(model: nn.Module, inputs: torch.Tensor):
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=[inputs])
    # torch_tensorrt.save(trt_gm, "trt.ep", inputs=[inputs],)

    exported = torch_tensorrt.dynamo._exporter.export(trt_gm)
    torch.export.save(exported, "trt.ep", pickle_protocol=4)
