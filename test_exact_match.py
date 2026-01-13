"""Test to check exact match between old and new implementations."""

import torch
from PIL import Image

from mvanet.predictor import MVANetPredictor
from mvanet.transformers import (
    MVANetConfig,
    MVANetForImageSegmentation,
    MVANetImageProcessor,
)


def main():
    # Create sample image (matching pytest fixture)
    from PIL import ImageDraw

    sample_image = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(sample_image)
    draw.ellipse([200, 150, 400, 350], fill="red", outline="black")
    draw.rectangle([450, 200, 650, 400], fill="blue", outline="black")

    # Old implementation
    print("=" * 80)
    print("Testing MVANetPredictor vs MVANetForImageSegmentation")
    print("=" * 80)
    print()

    predictor = MVANetPredictor()
    original_w, original_h = sample_image.size
    resized_image = sample_image.resize([1024, 1024], Image.Resampling.BILINEAR)
    transformed_image = predictor.image_transform(resized_image)
    transformed_image = transformed_image.unsqueeze(0).to(predictor.device)

    # New implementation
    config = MVANetConfig()
    model = MVANetForImageSegmentation(config)
    model.load_state_dict(predictor.net.state_dict())
    model = model.to(predictor.device)
    model.eval()

    # Move positional encoding tensors
    for module in model.modules():
        if hasattr(module, "positional_encoding") and hasattr(
            module.positional_encoding, "dim_t"
        ):
            module.positional_encoding.dim_t = module.positional_encoding.dim_t.to(
                predictor.device
            )

    processor = MVANetImageProcessor()

    # Test 1: Preprocessing exact match
    print("Test 1: Preprocessing")
    print("-" * 40)
    # IMPORTANT: Use the same resized_image for both to avoid double-resizing!
    inputs = processor(resized_image, return_tensors="pt")
    new_transformed = inputs["pixel_values"][0].to(predictor.device)

    preprocessing_match = torch.allclose(
        transformed_image[0], new_transformed, atol=1e-6
    )
    max_diff = torch.max(torch.abs(transformed_image[0] - new_transformed)).item()
    mean_diff = torch.mean(torch.abs(transformed_image[0] - new_transformed)).item()

    print(f"Preprocessing exact match: {preprocessing_match}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print()

    # Test 2: Model output exact match
    print("Test 2: Model Output (Logits)")
    print("-" * 40)

    with torch.no_grad():
        # Use the SAME tensor for both models!
        test_input = transformed_image.clone()

        # Old model output (logits before sigmoid)
        old_logits = predictor.net(test_input)

        # New model output (logits)
        outputs = model(pixel_values=test_input)
        new_logits = outputs.logits

        # Compare logits
        logits_exact_match = torch.allclose(old_logits, new_logits, atol=1e-6)
        logits_max_diff = torch.max(torch.abs(old_logits - new_logits)).item()
        logits_mean_diff = torch.mean(torch.abs(old_logits - new_logits)).item()
        logits_rel_diff = torch.mean(
            torch.abs(old_logits - new_logits) / (torch.abs(old_logits) + 1e-8)
        ).item()

        print(f"Logits exact match (atol=1e-6): {logits_exact_match}")
        print(f"Max absolute difference: {logits_max_diff:.2e}")
        print(f"Mean absolute difference: {logits_mean_diff:.2e}")
        print(
            f"Mean relative difference: {logits_rel_diff:.2e} ({logits_rel_diff * 100:.4f}%)"
        )
        print()

        # Test 3: After sigmoid
        print("Test 3: Model Output (After Sigmoid)")
        print("-" * 40)

        old_probs = old_logits.sigmoid()
        new_probs = new_logits.sigmoid()

        probs_exact_match = torch.allclose(old_probs, new_probs, atol=1e-6)
        probs_max_diff = torch.max(torch.abs(old_probs - new_probs)).item()
        probs_mean_diff = torch.mean(torch.abs(old_probs - new_probs)).item()
        probs_rel_diff = torch.mean(
            torch.abs(old_probs - new_probs) / (old_probs + 1e-8)
        ).item()

        print(f"Probabilities exact match (atol=1e-6): {probs_exact_match}")
        print(f"Max absolute difference: {probs_max_diff:.2e}")
        print(f"Mean absolute difference: {probs_mean_diff:.2e}")
        print(
            f"Mean relative difference: {probs_rel_diff:.2e} ({probs_rel_diff * 100:.4f}%)"
        )
        print()

        # Test 4: Check if outputs are bitwise identical
        print("Test 4: Bitwise Comparison")
        print("-" * 40)

        bitwise_identical = torch.equal(old_logits, new_logits)
        print(f"Bitwise identical: {bitwise_identical}")

        if not bitwise_identical:
            # Find where differences occur
            diff_mask = old_logits != new_logits
            num_diff = diff_mask.sum().item()
            total = old_logits.numel()
            print(
                f"Number of different values: {num_diff} / {total} ({num_diff / total * 100:.4f}%)"
            )

            # Show some different values
            if num_diff > 0:
                old_vals = old_logits[diff_mask][:5]
                new_vals = new_logits[diff_mask][:5]
                print("\nFirst 5 different values:")
                for i, (o, n) in enumerate(zip(old_vals, new_vals)):
                    print(
                        f"  [{i}] Old: {o.item():.10f}, New: {n.item():.10f}, Diff: {(o - n).item():.2e}"
                    )
        print()

        # Test 5: State dict comparison
        print("Test 5: Model State Dict Comparison")
        print("-" * 40)

        old_state = predictor.net.state_dict()
        new_state = model.state_dict()

        # Check if all keys match
        old_keys = set(old_state.keys())
        new_keys = set(new_state.keys())

        if old_keys == new_keys:
            print("✓ All state dict keys match")

            # Check if all values match
            all_match = True
            max_param_diff = 0.0
            diff_params = []

            for key in old_keys:
                if not torch.equal(old_state[key], new_state[key]):
                    all_match = False
                    diff = torch.max(torch.abs(old_state[key] - new_state[key])).item()
                    if diff > max_param_diff:
                        max_param_diff = diff
                    diff_params.append((key, diff))

            if all_match:
                print("✓ All parameter values are bitwise identical")
            else:
                print(f"✗ {len(diff_params)} parameters have different values")
                print(f"  Max parameter difference: {max_param_diff:.2e}")
                print("\n  Top 5 parameters with largest differences:")
                diff_params.sort(key=lambda x: x[1], reverse=True)
                for key, diff in diff_params[:5]:
                    print(f"    {key}: {diff:.2e}")
        else:
            print("✗ State dict keys don't match")
            print(f"  Old-only keys: {old_keys - new_keys}")
            print(f"  New-only keys: {new_keys - old_keys}")
        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        f"Preprocessing:      {'✓ Exact match' if preprocessing_match else f'✗ Diff: {mean_diff:.2e}'}"
    )
    print(
        f"Logits:             {'✓ Exact match' if logits_exact_match else f'✗ Mean diff: {logits_mean_diff:.2e}'}"
    )
    print(
        f"Probabilities:      {'✓ Exact match' if probs_exact_match else f'✗ Mean diff: {probs_mean_diff:.2e}'}"
    )
    print(f"Bitwise identical:  {'✓ Yes' if bitwise_identical else '✗ No'}")
    print()


if __name__ == "__main__":
    main()
