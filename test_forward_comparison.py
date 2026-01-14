"""Compare forward pass step by step to find where differences occur."""

import torch
from PIL import Image

from mvanet.predictor import MVANetPredictor
from mvanet.transformers import (
    MVANetConfig,
    MVANetForImageSegmentation,
    MVANetImageProcessor,
)


def compare_tensors(name, t1, t2):
    """Compare two tensors and print statistics."""
    if t1.shape != t2.shape:
        print(f"  {name}: ✗ SHAPE MISMATCH")
        print(f"    Old shape: {t1.shape}, New shape: {t2.shape}")
        return False

    exact_match = torch.equal(t1, t2)
    close_match = torch.allclose(t1, t2, atol=1e-6)
    max_diff = torch.max(torch.abs(t1 - t2)).item()
    mean_diff = torch.mean(torch.abs(t1 - t2)).item()

    status = "✓" if exact_match else ("~" if close_match else "✗")
    print(f"  {name}: {status} Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    return exact_match


def main():
    print("=" * 80)
    print("Step-by-Step Forward Pass Comparison")
    print("=" * 80)
    print()

    # Create sample image (matching pytest fixture - complex image with shapes)
    from PIL import ImageDraw

    sample_image = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(sample_image)
    draw.ellipse([200, 150, 400, 350], fill="red", outline="black")
    draw.rectangle([450, 200, 650, 400], fill="blue", outline="black")

    # Old implementation
    predictor = MVANetPredictor()
    old_model = predictor.net

    # New implementation
    config = MVANetConfig()
    new_model = MVANetForImageSegmentation(config)
    new_model.load_state_dict(old_model.state_dict())
    new_model = new_model.to(predictor.device)
    new_model.eval()

    # Move positional encoding tensors
    for module in new_model.modules():
        if hasattr(module, "positional_encoding") and hasattr(
            module.positional_encoding, "dim_t"
        ):
            module.positional_encoding.dim_t = module.positional_encoding.dim_t.to(
                predictor.device
            )

    processor = MVANetImageProcessor()

    # Prepare input
    resized_image = sample_image.resize([1024, 1024], Image.Resampling.BILINEAR)
    transformed_image = predictor.image_transform(resized_image)
    x = transformed_image.unsqueeze(0).to(predictor.device)

    print("Input:")
    print(f"  Shape: {x.shape}")
    print()

    # Step-by-step comparison
    with torch.no_grad():
        batch_size = x.shape[0]

        # Step 1: Shallow features
        print("Step 1: Shallow features")
        old_shallow = old_model.shallow(x)
        new_shallow = new_model.shallow(x)
        compare_tensors("shallow", old_shallow, new_shallow)
        print()

        # Step 2: Multi-view input
        print("Step 2: Multi-view input creation")
        from mvanet.transformers.modeling_mvanet import rescale_to, image2patches

        old_glb = rescale_to(x, scale_factor=0.5, interpolation="bilinear")
        new_glb = rescale_to(x, scale_factor=0.5, interpolation="bilinear")
        compare_tensors("glb", old_glb, new_glb)

        old_loc = image2patches(x)
        new_loc = image2patches(x)
        compare_tensors("loc", old_loc, new_loc)

        old_input = torch.cat((old_loc, old_glb), dim=0)
        new_input = torch.cat((new_loc, new_glb), dim=0)
        compare_tensors("input (loc + glb)", old_input, new_input)
        print()

        # Step 3: Backbone features
        print("Step 3: Backbone feature extraction")
        old_feature = old_model.backbone(old_input)
        new_feature = new_model.backbone(new_input)

        for i in range(len(old_feature)):
            compare_tensors(f"feature[{i}]", old_feature[i], new_feature[i])
        print()

        # Step 4: Feature projection
        print("Step 4: Feature projection")
        old_e5 = old_model.output5(old_feature[4])
        new_e5 = new_model.output5(new_feature[4])
        compare_tensors("e5", old_e5, new_e5)

        old_e4 = old_model.output4(old_feature[3])
        new_e4 = new_model.output4(new_feature[3])
        compare_tensors("e4", old_e4, new_e4)

        old_e3 = old_model.output3(old_feature[2])
        new_e3 = new_model.output3(new_feature[2])
        compare_tensors("e3", old_e3, new_e3)

        old_e2 = old_model.output2(old_feature[1])
        new_e2 = new_model.output2(new_feature[1])
        compare_tensors("e2", old_e2, new_e2)

        old_e1 = old_model.output1(old_feature[0])
        new_e1 = new_model.output1(new_feature[0])
        compare_tensors("e1", old_e1, new_e1)
        print()

        # Step 5: MCLM
        print("Step 5: Multi-field Cross Localization Module (MCLM)")
        old_loc_e5, old_glb_e5 = old_e5.split([batch_size * 4, batch_size], dim=0)
        new_loc_e5, new_glb_e5 = new_e5.split([batch_size * 4, batch_size], dim=0)

        compare_tensors("loc_e5 (before MCLM)", old_loc_e5, new_loc_e5)
        compare_tensors("glb_e5 (before MCLM)", old_glb_e5, new_glb_e5)

        old_e5_cat = old_model.multifieldcrossatt(old_loc_e5, old_glb_e5)
        new_e5_cat = new_model.multifieldcrossatt(new_loc_e5, new_glb_e5)
        compare_tensors("e5_cat (after MCLM)", old_e5_cat, new_e5_cat)
        print()

        # Step 6: Decoder
        print("Step 6: Decoder with MCRM")

        from mvanet.transformers.modeling_mvanet import resize_as

        old_e4 = old_model.conv4(
            old_model.dec_blk4(old_e4 + resize_as(old_e5_cat, old_e4))
        )
        new_e4 = new_model.conv4(
            new_model.dec_blk4(new_e4 + resize_as(new_e5_cat, new_e4))
        )
        compare_tensors("e4 (after dec_blk4)", old_e4, new_e4)

        old_e3 = old_model.conv3(old_model.dec_blk3(old_e3 + resize_as(old_e4, old_e3)))
        new_e3 = new_model.conv3(new_model.dec_blk3(new_e3 + resize_as(new_e4, new_e3)))
        compare_tensors("e3 (after dec_blk3)", old_e3, new_e3)

        old_e2 = old_model.conv2(old_model.dec_blk2(old_e2 + resize_as(old_e3, old_e2)))
        new_e2 = new_model.conv2(new_model.dec_blk2(new_e2 + resize_as(new_e3, new_e2)))
        compare_tensors("e2 (after dec_blk2)", old_e2, new_e2)

        old_e1 = old_model.conv1(old_model.dec_blk1(old_e1 + resize_as(old_e2, old_e1)))
        new_e1 = new_model.conv1(new_model.dec_blk1(new_e1 + resize_as(new_e2, new_e1)))
        compare_tensors("e1 (after dec_blk1)", old_e1, new_e1)
        print()

        # Step 7: Final processing
        print("Step 7: Final processing")

        from mvanet.transformers.modeling_mvanet import patches2image

        old_loc_e1, old_glb_e1 = old_e1.split([batch_size * 4, batch_size], dim=0)
        new_loc_e1, new_glb_e1 = new_e1.split([batch_size * 4, batch_size], dim=0)

        old_output1_cat = patches2image(old_loc_e1)
        new_output1_cat = patches2image(new_loc_e1)
        compare_tensors(
            "output1_cat (after patches2image)", old_output1_cat, new_output1_cat
        )

        old_output1_cat = old_output1_cat + resize_as(old_glb_e1, old_output1_cat)
        new_output1_cat = new_output1_cat + resize_as(new_glb_e1, new_output1_cat)
        compare_tensors("output1_cat (after add glb)", old_output1_cat, new_output1_cat)

        old_final = old_model.insmask_head(old_output1_cat)
        new_final = new_model.insmask_head(new_output1_cat)
        compare_tensors("final (after insmask_head)", old_final, new_final)

        old_final = old_final + resize_as(old_shallow, old_final)
        new_final = new_final + resize_as(new_shallow, new_final)
        compare_tensors("final (after add shallow 1)", old_final, new_final)

        old_final = old_model.upsample1(rescale_to(old_final))
        new_final = new_model.upsample1(rescale_to(new_final))
        compare_tensors("final (after upsample1)", old_final, new_final)

        old_final = rescale_to(old_final + resize_as(old_shallow, old_final))
        new_final = rescale_to(new_final + resize_as(new_shallow, new_final))
        compare_tensors("final (after rescale + add shallow 2)", old_final, new_final)

        old_final = old_model.upsample2(old_final)
        new_final = new_model.upsample2(new_final)
        compare_tensors("final (after upsample2)", old_final, new_final)

        old_output = old_model.output(old_final)
        new_output = new_model.output(new_final)
        compare_tensors("output (final)", old_output, new_output)
        print()


if __name__ == "__main__":
    main()
