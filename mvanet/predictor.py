from dataclasses import dataclass
from typing import Literal, Optional

import torch
import ttach as tta
from huggingface_hub import hf_hub_download
from PIL import Image
from PIL.Image import Image as PilImage
from torchvision import transforms

from mvanet.model import inf_MVANet

OutputType = Literal["rgba", "map"]


@dataclass
class MVANetPredictor(object):
    is_crf_refine: bool = True

    _net: Optional[inf_MVANet] = None

    _image_transform: Optional[transforms.Compose] = None
    _tta_transforms: Optional[tta.Compose] = None

    to_pil: transforms.ToPILImage = transforms.ToPILImage()
    depth_transform: transforms.ToTensor = transforms.ToTensor()
    target_transform: transforms.ToTensor = transforms.ToTensor()

    repo_id: str = "creative-graphic-design/MVANet-checkpoints"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        if self._net is None:
            self._net = self.load_net()
        if self._image_transform is None:
            self._image_transform = self.load_image_transform()
        if self._tta_transforms is None:
            self._tta_transforms = self.load_tta_transforms()

    @property
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
    ) -> inf_MVANet:
        net = inf_MVANet()

        ckpt_path = hf_hub_download(self.repo_id, filename=mvanet_ckpt_path)
        pretrained_dict = torch.load(ckpt_path, map_location="cpu")

        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict, strict=True)
        net.eval()

        net = net.to(self.device)

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

    def to_rgba(self, image: PilImage, mask: PilImage) -> PilImage:
        rgba_image = image.copy()
        rgba_image.putalpha(mask)
        return rgba_image

    @torch.inference_mode()
    def __call__(self, image: PilImage, output_type: OutputType = "rgba") -> PilImage:
        image = image.convert("RGB") if image.mode != "RGB" else image
        original_w, original_h = image.size

        resized_image = image.resize([1024, 1024], Image.BILINEAR)

        transformed_image = self.image_transform(resized_image)
        assert isinstance(transformed_image, torch.Tensor)

        transformed_image = transformed_image.unsqueeze(0)
        transformed_image = transformed_image.to(self.device)

        mask = []
        for tta_transform in self.tta_transforms:
            rgb_trans = tta_transform.augment_image(transformed_image)
            model_output = self.net(rgb_trans)
            deaug_mask = tta_transform.deaugment_mask(model_output)
            mask.append(deaug_mask)

        predicted_mask_th = torch.mean(torch.stack(mask, dim=0), dim=0)
        predicted_mask_th = predicted_mask_th.sigmoid()
        predicted_mask_pl = self.to_pil(predicted_mask_th.squeeze(0).cpu())
        predicted_mask_pl = predicted_mask_pl.resize(
            (original_w, original_h), Image.BILINEAR
        )

        if output_type == "rgba":
            return self.to_rgba(image, predicted_mask_pl)
        elif output_type == "map":
            return predicted_mask_pl
        else:
            raise ValueError(f"Invalid output_type: {output_type}")
