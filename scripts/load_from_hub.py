import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "creative-graphic-design/MVANet"

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = model.to(device)

    processor = AutoImageProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    filename = "0003.jpg"
    image = Image.open(filename)
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)

    torch.cuda.synchronize()
    start = time.time()

    with torch.inference_mode():
        outputs = model(**inputs)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start
    print(f"[single batch] {elapsed_time} sec.")

    (mask,) = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size]
    )
    mask_np = mask.cpu().numpy()
    mask_np = (mask_np * 255).clip(0, 255).astype("uint8")
    mask_pl = Image.fromarray(mask_np, mode="L")
    mask_pl.save("output_single_remote_code.png")


if __name__ == "__main__":
    main()
