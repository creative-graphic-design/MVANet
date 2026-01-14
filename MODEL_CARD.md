---
library_name: transformers
tags:
  - vision
  - image-segmentation
  - semantic-segmentation
  - dichotomous-image-segmentation
  - pytorch
license: mit
datasets:
  - DIS-5K
metrics:
  - f-measure
  - mae
  - s-measure
  - e-measure
---

# MVANet: Multi-view Aggregation Network for Dichotomous Image Segmentation

MVANet is a state-of-the-art model for high-accuracy dichotomous image segmentation (DIS). It models DIS as a multi-view object perception problem, processing images through both distant views (global context) and close-up views (local details) to achieve accurate segmentation of high-resolution objects.

## Model Details

### Model Description

MVANet introduces a novel multi-view processing paradigm for dichotomous image segmentation. Unlike traditional single-view approaches, MVANet splits high-resolution input images into:

- **1 distant view**: A downsampled global image (0.5× scale) capturing overall context
- **4 close-up views**: Local patches arranged in a 2×2 grid preserving fine-grained details

This multi-view approach balances semantic understanding from large receptive fields with high-precision detail preservation from small receptive fields, achieving state-of-the-art performance on the DIS-5K benchmark.

**Key Features:**

- Single-stream, single-stage architecture (more efficient than multi-stage approaches)
- Multi-field Cross Localization Module (MCLM) for global-local feature fusion
- Multi-crop Refinement Module (MCRM) for boundary detail enhancement
- Swin Transformer Base (Swin-B) backbone for hierarchical feature extraction
- Achieves 4.6 FPS inference speed (2× faster than InSPyReNet)

- **Developed by:** Qian Yu, Xiaoqi Zhao, Youwei Pang, Lihe Zhang, Huchuan Lu
- **Model type:** Semantic Segmentation (Dichotomous Image Segmentation)
- **Language(s):** Python, PyTorch
- **License:** MIT
- **Backbone:** Swin Transformer Base (Swin-B)

### Model Sources

- **Repository:** https://github.com/qianyu-dlut/MVANet
- **Paper:** "MVANet: Multi-view Aggregation Network for Dichotomous Image Segmentation"
- **Transformers Implementation:** https://github.com/creative-graphic-design/MVANet

## Uses

### Direct Use

MVANet is designed for high-accuracy dichotomous image segmentation tasks, including:

- Foreground object segmentation in natural scenes
- High-resolution fine-grained object delineation
- Category-agnostic segmentation with various structural complexities
- AR/VR applications
- Image editing and matting
- 3D shape reconstruction

### Example Usage

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

# Load model and processor
model = AutoModel.from_pretrained("creative-graphic-design/mvanet")
processor = AutoImageProcessor.from_pretrained("creative-graphic-design/mvanet")

# Load and process image
image = Image.open("image.jpg")
inputs = processor(image, return_tensors="pt")

# Generate segmentation mask
outputs = model(**inputs)
masks = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)
mask = masks[0]  # Segmentation mask as tensor
```

### Downstream Use

MVANet can be fine-tuned for:

- Domain-specific segmentation tasks
- Video object segmentation (frame-by-frame processing)
- Interactive segmentation with user guidance
- Multi-object instance segmentation

### Out-of-Scope Use

MVANet is **not suitable** for:

- Real-time video processing (4.6 FPS may be insufficient for some applications)
- Semantic segmentation with multiple classes (designed for binary segmentation)
- Very low-resolution images (optimized for high-resolution inputs ≥1024×1024)
- 3D volumetric segmentation (2D image-only model)

## Bias, Risks, and Limitations

### Technical Limitations

1. **Input Resolution**: Optimized for 1024×1024 pixel images. Smaller images may not benefit from multi-view architecture.
2. **Memory Requirements**: Requires GPU with sufficient memory for high-resolution processing (tested on RTX 3090 with 24GB).
3. **Patch Grid**: Currently hardcoded to 2×2 grid (4 patches). Other configurations not supported.
4. **Binary Segmentation**: Designed for single-object foreground/background separation only.

### Bias Considerations

- **Dataset Bias**: Trained on DIS-5K which may not represent all object types or imaging conditions
- **High-Resolution Bias**: Performance optimized for high-resolution images (2K, 4K+); may underperform on low-resolution inputs
- **Structural Complexity**: May struggle with objects having extremely complex or fractal-like structures not well-represented in training data

### Recommendations

Users should:

- Validate model performance on their specific domain before deployment
- Ensure input images are high-resolution (≥1024×1024 recommended)
- Use GPU acceleration for acceptable inference speed
- Post-process outputs for production use cases requiring guarantees
- Consider fine-tuning on domain-specific data for best results

## How to Get Started with the Model

### Installation

```bash
pip install transformers[torch,torch-vision]
```

### Basic Inference

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch

# Load model and processor
model = AutoModel.from_pretrained("creative-graphic-design/MVANet")
processor = AutoImageProcessor.from_pretrained("creative-graphic-design/MVANet")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load image
image = Image.open("path/to/image.jpg")

# Process image
inputs = processor(image, return_tensors="pt")
inputs = inputs.to(device)

# Generate mask
with torch.no_grad():
    outputs = model(**inputs)
    masks = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )

# Save mask
mask = masks[0].cpu().numpy()
mask_image = Image.fromarray((mask * 255).astype("uint8"))
mask_image.save("mask.png")
```

### Batch Processing

```python
images = [Image.open(f"image_{i}.jpg") for i in range(4)]
inputs = processor(images, return_tensors="pt")
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)
    masks = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(img.height, img.width) for img in images]
    )
```

### Test-Time Augmentation (TTA)

```python
from mvanet.transformers import MVANetTTAPipeline

tta_pipeline = MVANetTTAPipeline(model, processor)
masks = tta_pipeline(images, return_tensors=False)  # List of PIL Images
```

## Training Details

### Training Data

**DIS-5K Dataset:**

- **Training set (DIS-TR)**: 3,000 high-resolution images
- **Validation set (DIS-VD)**: 470 images
- **Test sets (DIS-TE1-4)**: 2,000 images (4 subsets with increasing complexity)
- **Resolution**: 2K, 4K, or larger
- **Categories**: 225 diverse object categories
- **Annotations**: High-precision instance masks

The DIS-5K dataset focuses on challenging high-resolution fine-grained object segmentation with:

- Salient objects
- Camouflaged objects
- Objects with intricate internal details
- Various structural complexities
- Complex lighting conditions and occlusions

### Training Procedure

#### Preprocessing

1. **Resizing**: Original images resized to 1024×1024
2. **Multi-view Creation**:
   - Global view: 0.5× downscaling → 512×512
   - Local patches: 2×2 grid → 4 patches of 512×512
3. **Normalization**: ImageNet mean and std
4. **Data Augmentation**:
   - Random horizontal flipping
   - Random cropping
   - Random rotation

#### Training Hyperparameters

- **Optimizer**: Adam
- **Learning rate**: 1e-5 (0.00001)
- **Batch size**: 1 (due to high memory requirements)
- **Epochs**: 80
- **Loss function**: BCE + weighted IoU loss
- **Backbone initialization**: Swin-B pretrained on ImageNet-22K and fine-tuned on ImageNet-1K
- **Hardware**: Single NVIDIA RTX 3090 (24GB)
- **Training regime**: Mixed precision (fp16)

#### Loss Components

```python
L = l_f + Σ(l_l^i + λ_g * l_g^i + λ_a * l_a^i)
```

Where:

- `l_f`: Final prediction loss
- `l_l^i`: Assembled local representation loss at layer i
- `l_g^i`: Global representation loss at layer i
- `l_a^i`: Token attention map loss at layer i
- `λ_g = 0.3`, `λ_a = 0.3`

#### Training Time

Approximately 48-72 hours on single RTX 3090 GPU for 80 epochs.

## Evaluation

### Testing Data

**DIS-TE (DIS-5K Test Sets):**

- **DIS-TE1**: 500 images (lowest complexity)
- **DIS-TE2**: 500 images (medium-low complexity)
- **DIS-TE3**: 500 images (medium-high complexity)
- **DIS-TE4**: 500 images (highest complexity)
- **Total**: 2,000 diverse high-resolution test images

### Metrics

| Metric      | Description                                   | MVANet Performance |
| ----------- | --------------------------------------------- | ------------------ |
| **F^max_β** | Maximum F-measure (precision-recall balance)  | State-of-the-art   |
| **F^ω_β**   | Weighted F-measure                            | State-of-the-art   |
| **S_m**     | Structural similarity (region + object aware) | State-of-the-art   |
| **E^m_φ**   | Enhanced-alignment measure                    | State-of-the-art   |
| **MAE**     | Mean Absolute Error                           | State-of-the-art   |
| **FPS**     | Frames per second (inference speed)           | 4.6 FPS            |

### Results

MVANet achieves state-of-the-art performance on DIS-5K benchmark:

- **2.5% improvement** in F^max_β over InSPyReNet (previous best)
- **2.1% improvement** in E^m_φ
- **0.5% improvement** in S_m
- **0.4% improvement** in MAE
- **2× faster inference speed** (4.6 FPS vs 2.2 FPS)

#### Comparison with State-of-the-Art Methods

MVANet outperforms 11 well-known methods including:

- F³Net, GCPANet, PFNet, BSANet (CNN-based)
- ISDNet, IFA, IS-Net (early DIS methods)
- FP-DIS, UDUN, PGNet, InSPyNet (recent state-of-the-art)

#### Summary

MVANet represents a significant advancement in dichotomous image segmentation:

- **First single-stream, single-stage architecture** for DIS
- **Parsimonious design** with fewer parameters than multi-stage methods
- **Superior accuracy** across all standard metrics
- **Faster inference** enabling practical applications

The multi-view paradigm proves effective for balancing global context understanding and local detail preservation in high-resolution image segmentation.

## Model Architecture

### Overview

```
Input (1024×1024) → Multi-view Split → Feature Extraction → MCLM → Decoder → Output
                     ├─ Global: 512×512 (0.5×)
                     └─ Local: 4×512×512 (2×2 grid)
```

### Key Components

1. **Swin Transformer Base (Backbone)**

   - Hierarchical architecture with shifted windows
   - 4 stages with [2, 2, 18, 2] blocks
   - Attention heads: [4, 8, 16, 32]
   - Window size: 12
   - Output channels: [128, 128, 256, 512, 1024]

2. **Multi-field Cross Localization Module (MCLM)**

   - Multi-head cross-attention between global and local features
   - Multi-granularity pooling: [1, 4, 8]
   - 1 attention head
   - Enhances object localization

3. **Multi-crop Refinement Module (MCRM)**

   - Token attention for background filtering
   - Multi-scale refinement: [2, 4, 8]
   - 1 attention head
   - Embedded in each decoder stage

4. **Decoder**

   - FPN-like top-down architecture
   - MCRM at each stage for on-the-fly refinement
   - Skip connections from encoder

5. **Instance Mask Head**
   - 3-layer convolutional head
   - Hidden dimension: 384
   - Boundary detail enhancement

### Compute Infrastructure

#### Hardware

- **Training**: Single NVIDIA RTX 3090 GPU (24GB VRAM)
- **Inference**: RTX 3090 or equivalent
- **Minimum GPU memory**: 12GB recommended for 1024×1024 inputs

#### Software

- **Framework**: PyTorch 2.0+
- **Library**: Transformers 4.30+
- **Dependencies**: timm, einops, pillow
- **CUDA**: 11.8+ recommended

## Environmental Impact

Carbon emissions estimated for training on single RTX 3090 GPU:

- **Hardware Type**: NVIDIA RTX 3090 (350W TDP)
- **Hours used**: ~60 hours (80 epochs)
- **Power consumption**: ~21 kWh
- **Carbon Emitted**: ~10 kg CO2eq (estimated, varies by region)

Training MVANet is relatively efficient compared to large-scale vision models due to:

- Single GPU training
- Relatively fast convergence (80 epochs)
- No large-scale pretraining required (uses pretrained Swin-B backbone)

## Citation

**BibTeX:**

```bibtex
@article{yu2023mvanet,
  title={MVANet: Multi-view Aggregation Network for Dichotomous Image Segmentation},
  author={Yu, Qian and Zhao, Xiaoqi and Pang, Youwei and Zhang, Lihe and Lu, Huchuan},
  journal={arXiv preprint arXiv:2404.xxxxx},
  year={2023}
}
```

**APA:**

Yu, Q., Zhao, X., Pang, Y., Zhang, L., & Lu, H. (2023). MVANet: Multi-view Aggregation Network for Dichotomous Image Segmentation. arXiv preprint arXiv:2404.xxxxx.

## Acknowledgements

This work was supported by:

- National Natural Science Foundation of China (Grant 62276046)
- Dalian Science and Technology Innovation Foundation (Grant 2023JJ12GX015)

## License

This model is released under the MIT License. See LICENSE file for details.

## Model Card Authors

- Integration by creative-graphic-design team

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/creative-graphic-design/MVANet).
