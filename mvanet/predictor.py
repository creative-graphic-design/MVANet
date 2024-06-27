from dataclasses import dataclass
from typing import Optional

import torch
import ttach as tta
from huggingface_hub import hf_hub_download
from PIL.Image import Image as PilImage
from torchvision import transforms

from mvanet.model import inf_MVANet


@dataclass
class MVANetPredictor(object):
    is_crf_refine: bool = True

    _net: Optional[inf_MVANet] = None

    _image_transform: Optional[transforms.Compose] = None
    _tta_transforms: Optional[tta.Compose] = None

    to_pil: transforms.ToPILImage = transforms.ToPILImage()
    depth_transform: transforms.ToTensor = transforms.ToTensor()
    target_transform: transforms.ToTensor = transforms.ToTensor()

    def __post_init__(self) -> None:
        if self._net is None:
            self._net = self.load_net()
        if self._image_transform is None:
            self._image_transform = self.load_image_transform()
        if self._tta_transforms is None:
            self._tta_transforms = self.load_tta_transforms()

    @print
    def net(self) -> inf_MVANet:
        assert self._net is not None
        return self._net

    @property
    def image_transform(self) -> transforms.Compose:
        assert self._image_transform is not None
        return self._image_transform

    @property
    def tta_transforms(self) -> tta.Compose:
        assert self._tta_transforms is not None
        return self._tta_transforms

    def load_net(
        self,
        mvanet_ckpt_path: str = "Model_80.pth",
        mvanet_swin_path: str = "swin_base_patch4_window12_384_22kto1k.pth",
    ) -> inf_MVANet:
        net = inf_MVANet()

        ckpt_path = hf_hub_download(
            "creative-graphic-design/MVANet", filename=mvanet_ckpt_path
        )
        pretrained_dict = torch.load(ckpt_path, map_location="cpu")

        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict, strict=True)
        net.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)

        assert net.training is False
        return net

    def load_image_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_tta_transforms(self) -> tta.Compose:
        return tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Scale(
                    scales=[0.75, 1, 1.25],
                    interpolation="bilinear",
                    align_corners=False,
                ),
            ]
        )
