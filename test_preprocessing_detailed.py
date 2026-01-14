"""Compare preprocessing in detail."""

import torch
import numpy as np
from PIL import Image

from mvanet.predictor import MVANetPredictor
from mvanet.transformers import MVANetImageProcessor


def main():
    print("=" * 80)
    print("Detailed Preprocessing Comparison")
    print("=" * 80)
    print()

    # Create sample image
    sample_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    print(f"Sample image: {sample_image.size}, mode={sample_image.mode}")
    print()

    # Method 1: MVANetPredictor (torchvision ToTensor + Normalize)
    print("Method 1: MVANetPredictor")
    print("-" * 40)
    predictor = MVANetPredictor()
    resized1 = sample_image.resize([1024, 1024], Image.Resampling.BILINEAR)
    tensor1 = predictor.image_transform(resized1)
    print(f"Output shape: {tensor1.shape}")
    print(f"Output dtype: {tensor1.dtype}")
    print(f"Output device: {tensor1.device}")
    print(f"Output range: [{tensor1.min().item():.6f}, {tensor1.max().item():.6f}]")
    print(f"Output mean: {tensor1.mean().item():.6f}")
    print(f"Sample values (first 5): {tensor1.flatten()[:5]}")
    print()

    # Method 2: MVANetImageProcessor (numpy + torch.tensor)
    print("Method 2: MVANetImageProcessor")
    print("-" * 40)
    processor = MVANetImageProcessor()
    inputs = processor(sample_image, return_tensors="pt")
    tensor2 = inputs["pixel_values"][0]
    print(f"Output shape: {tensor2.shape}")
    print(f"Output dtype: {tensor2.dtype}")
    print(f"Output device: {tensor2.device}")
    print(f"Output range: [{tensor2.min().item():.6f}, {tensor2.max().item():.6f}]")
    print(f"Output mean: {tensor2.mean().item():.6f}")
    print(f"Sample values (first 5): {tensor2.flatten()[:5]}")
    print()

    # Compare
    print("Comparison")
    print("-" * 40)
    exact_match = torch.equal(tensor1, tensor2)
    close_match = torch.allclose(tensor1, tensor2, atol=1e-6)
    max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
    mean_diff = torch.mean(torch.abs(tensor1 - tensor2)).item()

    print(f"Exact match: {exact_match}")
    print(f"Close match (atol=1e-6): {close_match}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print()

    if not exact_match:
        # Find where differences occur
        diff_mask = tensor1 != tensor2
        num_diff = diff_mask.sum().item()
        total = tensor1.numel()
        print(
            f"Number of different values: {num_diff} / {total} ({num_diff / total * 100:.4f}%)"
        )

        if num_diff > 0:
            print("\nFirst 10 different values:")
            t1_vals = tensor1[diff_mask][:10]
            t2_vals = tensor2[diff_mask][:10]
            for i, (v1, v2) in enumerate(zip(t1_vals, t2_vals)):
                diff = (v1 - v2).item()
                print(
                    f"  [{i}] Method1: {v1.item():.10f}, Method2: {v2.item():.10f}, Diff: {diff:.2e}"
                )

    print()

    # Test with real image
    print("=" * 80)
    print("Test with Gradient Image")
    print("=" * 80)
    print()

    # Create gradient image (more interesting pattern)
    gradient = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        gradient[i, :, 0] = int(i / 512 * 255)  # Red gradient
        gradient[:, i, 1] = int(i / 512 * 255)  # Green gradient
    gradient[:, :, 2] = 128  # Constant blue
    gradient_image = Image.fromarray(gradient)

    print("Gradient image created")
    print()

    # Method 1
    print("Method 1: MVANetPredictor")
    print("-" * 40)
    resized1 = gradient_image.resize([1024, 1024], Image.Resampling.BILINEAR)
    tensor1 = predictor.image_transform(resized1)
    print(f"Output range: [{tensor1.min().item():.6f}, {tensor1.max().item():.6f}]")
    print(f"Output mean: {tensor1.mean().item():.6f}")
    print(f"Red channel mean: {tensor1[0].mean().item():.6f}")
    print(f"Green channel mean: {tensor1[1].mean().item():.6f}")
    print(f"Blue channel mean: {tensor1[2].mean().item():.6f}")
    print()

    # Method 2
    print("Method 2: MVANetImageProcessor")
    print("-" * 40)
    inputs = processor(gradient_image, return_tensors="pt")
    tensor2 = inputs["pixel_values"][0]
    print(f"Output range: [{tensor2.min().item():.6f}, {tensor2.max().item():.6f}]")
    print(f"Output mean: {tensor2.mean().item():.6f}")
    print(f"Red channel mean: {tensor2[0].mean().item():.6f}")
    print(f"Green channel mean: {tensor2[1].mean().item():.6f}")
    print(f"Blue channel mean: {tensor2[2].mean().item():.6f}")
    print()

    # Compare
    print("Comparison")
    print("-" * 40)
    exact_match = torch.equal(tensor1, tensor2)
    close_match = torch.allclose(tensor1, tensor2, atol=1e-6)
    max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
    mean_diff = torch.mean(torch.abs(tensor1 - tensor2)).item()

    print(f"Exact match: {exact_match}")
    print(f"Close match (atol=1e-6): {close_match}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")


if __name__ == "__main__":
    main()
