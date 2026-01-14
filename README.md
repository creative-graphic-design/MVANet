# MVANet

[![CI](https://github.com/creative-graphic-design/MVANet/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/MVANet/actions/workflows/ci.yaml)
[![Release](https://github.com/creative-graphic-design/MVANet/actions/workflows/release.yaml/badge.svg)](https://github.com/creative-graphic-design/MVANet/actions/workflows/release.yaml)
[![Deploy](https://github.com/creative-graphic-design/MVANet/actions/workflows/deploy.yaml/badge.svg)](https://github.com/creative-graphic-design/MVANet/actions/workflows/deploy.yaml)
[![PyPI](https://img.shields.io/pypi/v/mvanet.svg)](https://pypi.python.org/pypi/mvanet)

This is a fork of the original [MVANet](https://github.com/qianyu-dlut/MVANet), with bug fixes, packaging improvements, and transformers-compatible API.

MVANet is a Multi-view Aggregation Network for Dichotomous Image Segmentation, presented at CVPR 2024 (Highlight). It achieves state-of-the-art performance for high-precision object segmentation from high-resolution images.

## Installation

```shell
pip install mvanet
```

## Usage

### Transformers API (Recommended)

The transformers-compatible API provides better integration with the HuggingFace ecosystem:

```python
from PIL import Image
from mvanet.transformers import (
    MVANetConfig,
    MVANetForImageSegmentation,
    MVANetImageProcessor,
)

# Load image
image = Image.open("/path/to/image.png")

# Initialize model and processor
config = MVANetConfig()
model = MVANetForImageSegmentation(config)
processor = MVANetImageProcessor()

# Preprocess
inputs = processor(image, return_tensors="pt")

# Inference
outputs = model(**inputs)

# Post-process
masks = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)
```

### Legacy API (Optional)

The original predictor API is available as an optional dependency:

```shell
# Install with legacy API support
pip install mvanet[original]
# or with uv
uv sync --group original
```

```python
from PIL import Image
from mvanet import MVANetPredictor

test_image = Image.open("/path/to/image.png")

predictor = MVANetPredictor()

# Predict the RGBA image
predicted_image = predictor(test_image, output_type="rgba")
predicted_image.save("rgba.png")

# Predict the mask image
predicted_mask = predictor(test_image, output_type="map")
predicted_mask.save("mask.png")
```

### Test Time Augmentation (TTA)

For higher quality predictions, you can use TTA:

```python
from mvanet.transformers import MVANetTTAPipeline

# Create TTA pipeline
tta_pipeline = MVANetTTAPipeline(model, processor)

# Run inference with TTA
masks = tta_pipeline([image])
mask_pil = masks[0]
mask_pil.save("mask_tta.png")
```

## Configuration

The model behavior can be customized through `MVANetConfig`:

```python
from mvanet.transformers import MVANetConfig

config = MVANetConfig(
    embedding_dim=128,                              # Embedding dimension throughout the model
    backbone_out_channels=[128, 128, 256, 512, 1024],  # Backbone output channels (SwinB)
    mclm_pool_ratios=[1, 4, 8],                    # MCLM multi-scale attention ratios
    mcrm_pool_ratios=[2, 4, 8],                    # MCRM multi-scale attention ratios
    insmask_hidden_dim=384,                        # Instance mask head hidden dimension
    global_view_scale=0.5,                         # Global view downscale factor
    num_patches=4,                                 # Number of local patches (2x2 grid)
    image_size=1024,                               # Input image size
    num_channels=3,                                # Number of input channels (RGB)
    num_labels=1,                                  # Number of output labels (binary segmentation)
)
```

### Key Parameters

- **`mcrm_pool_ratios`**: Controls the pooling ratios in the Multi-crop Refinement Module. Default `[2, 4, 8]` matches the trained model.
- **`global_view_scale`**: Scale factor for creating the global view (downsampled version). Default `0.5` creates a half-resolution global view.
- **`num_patches`**: Number of local patches. Currently only `4` (2x2 grid) is supported.
- **`insmask_hidden_dim`**: Hidden dimension in the instance mask head. Larger values may capture more complex patterns but require more memory.

## Paper and Citation

This implementation is based on the CVPR 2024 paper:

**Multi-view Aggregation Network for Dichotomous Image Segmentation**
Qian Yu, Xiaoqi Zhao, Youwei Pang, Lihe Zhang, Huchuan Lu
[arXiv:2404.07445](https://arxiv.org/abs/2404.07445)

```bibtex
@article{yu2024multi,
  title={Multi-view Aggregation Network for Dichotomous Image Segmentation},
  author={Yu, Qian and Zhao, Xiaoqi and Pang, Youwei and Zhang, Lihe and Lu, Huchuan},
  journal={arXiv preprint arXiv:2404.07445},
  year={2024}
}
```

## Links

- **Original Repository**: [qianyu-dlut/MVANet](https://github.com/qianyu-dlut/MVANet)
- **Paper**: [arXiv:2404.07445](https://arxiv.org/abs/2404.07445)
- **Checkpoints**: [Google Drive](https://drive.google.com/file/d/1_gabQXOF03MfXnf3EWDK1d_8wKiOemOv/view?usp=sharing)

## License

This project follows the license of the original MVANet repository.
