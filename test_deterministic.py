"""Test with deterministic mode to check if non-determinism is the cause."""

import torch
from PIL import Image

from mvanet.predictor import MVANetPredictor
from mvanet.transformers import (
    MVANetConfig,
    MVANetForImageSegmentation,
    MVANetImageProcessor,
)


def set_deterministic(seed=42):
    """Set deterministic mode for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_with_mode(deterministic=False):
    """Test with or without deterministic mode."""
    if deterministic:
        set_deterministic()
        print("🔒 Deterministic mode: ENABLED")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("🔓 Deterministic mode: DISABLED")

    # Create sample image
    sample_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

    # Old implementation
    predictor = MVANetPredictor()
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
    inputs = processor(sample_image, return_tensors="pt")

    # Run multiple times
    print("\nRunning 3 forward passes...")
    results = []

    for i in range(3):
        with torch.no_grad():
            old_logits = predictor.net(transformed_image)
            new_logits = model(
                pixel_values=inputs["pixel_values"].to(model.device)
            ).logits

            diff = torch.mean(torch.abs(old_logits - new_logits)).item()
            bitwise = torch.equal(old_logits, new_logits)
            results.append((diff, bitwise))

            print(
                f"  Pass {i + 1}: Mean diff = {diff:.2e}, Bitwise identical = {bitwise}"
            )

    # Check consistency
    print("\nConsistency check:")
    diffs = [r[0] for r in results]
    all_same_diff = len(set(f"{d:.10e}" for d in diffs)) == 1
    print(f"  All runs have same difference: {all_same_diff}")

    if all_same_diff:
        print(f"  ✓ Consistent difference: {diffs[0]:.2e}")
    else:
        print(f"  ✗ Differences vary: {diffs}")

    return results[0][0], results[0][1]


def main():
    print("=" * 80)
    print("Testing Determinism Impact on Model Output")
    print("=" * 80)
    print()

    # Test without deterministic mode
    print("Test 1: Non-deterministic mode (default)")
    print("-" * 80)
    nondeter_diff, nondeter_bitwise = test_with_mode(deterministic=False)
    print()

    # Test with deterministic mode
    print("Test 2: Deterministic mode")
    print("-" * 80)
    deter_diff, deter_bitwise = test_with_mode(deterministic=True)
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("Non-deterministic mode:")
    print(f"  Mean difference: {nondeter_diff:.2e}")
    print(f"  Bitwise identical: {nondeter_bitwise}")
    print()
    print("Deterministic mode:")
    print(f"  Mean difference: {deter_diff:.2e}")
    print(f"  Bitwise identical: {deter_bitwise}")
    print()

    if deter_bitwise:
        print("✓ Outputs are BITWISE IDENTICAL in deterministic mode!")
        print("  → The difference is caused by GPU non-determinism")
    else:
        print("✗ Outputs still differ in deterministic mode")
        print("  → There may be other causes (implementation differences)")


if __name__ == "__main__":
    main()
