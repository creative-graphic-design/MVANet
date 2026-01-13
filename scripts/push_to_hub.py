"""Script to push MVANet model to Hugging Face Hub.

This script:
1. Loads the pretrained MVANet model with weights
2. Saves the model, config, and processor locally
3. Pushes everything to Hugging Face Hub

Usage:
    python scripts/push_to_hub.py --repo-id <REPO_ID> [--token <HF_TOKEN>]

Example:
    python scripts/push_to_hub.py --repo-id creative-graphic-design/mvanet

Requirements:
    - huggingface_hub library (pip install huggingface_hub)
    - Hugging Face account with write access
    - Login via `huggingface-cli login` or pass token via --token
"""

import argparse

from transformers import AutoConfig, AutoImageProcessor, AutoModel

from mvanet.predictor import MVANetPredictor
from mvanet.transformers import (
    MVANetConfig,
    MVANetForImageSegmentation,
    MVANetImageProcessor,
)


def main():
    parser = argparse.ArgumentParser(
        description="Push MVANet model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="creative-graphic-design/MVANet",
        help="Repository ID on Hugging Face Hub (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional if already logged in)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--model-card",
        type=str,
        default="MODEL_CARD.md",
        help="Path to model card file (default: MODEL_CARD.md)",
    )
    args = parser.parse_args()

    # # Create local directory
    # local_dir = Path(args.local_dir)
    # local_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MVANet Model Hub Upload Script")
    print("=" * 80)
    print(f"\n📦 Repository ID: {args.repo_id}")
    print(f"🔒 Private: {args.private}")
    print()

    # Step 1: Load pretrained model weights
    print("Step 1: Loading pretrained model weights...")
    predictor = MVANetPredictor()
    print(f"✓ Loaded weights from: {predictor.repo_id}")
    print()

    # Step 2: Create transformers-compatible model
    print("Step 2: Creating transformers-compatible model...")
    config = MVANetConfig()
    model = MVANetForImageSegmentation(config)

    # Load weights from predictor
    model.load_state_dict(predictor.net.state_dict())
    print("✓ Converted to transformers format")
    print()

    # Step 3: Save model, config, and processor locally
    print(f"Step 3: Saving model to {args.repo_id}...")
    model.push_to_hub(args.repo_id)
    print(f"✓ Saved model to {args.repo_id}")

    # # config.save_pretrained(local_dir)
    # config.push_to_hub(args.repo_id)
    # print(f"✓ Saved config to {args.repo_id}")

    processor = MVANetImageProcessor()
    # processor.save_pretrained(local_dir)
    processor.push_to_hub(args.repo_id)
    print(f"✓ Saved processor to {args.repo_id}")
    print()

    # # Step 4: Copy model card
    # print("Step 4: Copying model card...")
    # model_card_path = Path(args.model_card)
    # if model_card_path.exists():
    #     readme_path = local_dir / "README.md"
    #     shutil.copy(model_card_path, readme_path)
    #     print(f"✓ Copied {model_card_path} to {readme_path}")
    # else:
    #     print(f"⚠ Warning: Model card not found at {model_card_path}")
    # print()

    # # Step 5: Create repository on Hugging Face Hub
    # print("Step 5: Creating repository on Hugging Face Hub...")
    # try:
    #     create_repo(
    #         repo_id=args.repo_id,
    #         repo_type="model",
    #         private=args.private,
    #         token=args.token,
    #         exist_ok=True,
    #     )
    #     print(f"✓ Repository created/verified: https://huggingface.co/{args.repo_id}")
    # except Exception as e:
    #     print(f"✗ Failed to create repository: {e}")
    #     print("\nMake sure you are logged in via: huggingface-cli login")
    #     return
    # print()

    # # Step 6: Push to hub
    # print("Step 6: Pushing to Hugging Face Hub...")
    # print("This may take a few minutes...")
    # try:
    #     api = HfApi(token=args.token)
    #     api.upload_folder(
    #         repo_id=args.repo_id,
    #         folder_path=local_dir,
    #         repo_type="model",
    #         commit_message="Upload MVANet model",
    #     )
    #     print(f"✓ Successfully pushed to: https://huggingface.co/{args.repo_id}")
    # except Exception as e:
    #     print(f"✗ Failed to push to hub: {e}")
    #     return
    # print()

    # # Success message
    # print("=" * 80)
    # print("🎉 Upload Complete!")
    # print("=" * 80)
    # print("\nYour model is now available at:")
    # print(f"  https://huggingface.co/{args.repo_id}")
    # print("\nYou can now use it with:")
    # print(f'  model = AutoModel.from_pretrained("{args.repo_id}")')
    # print(f'  processor = AutoImageProcessor.from_pretrained("{args.repo_id}")')
    # print()


if __name__ == "__main__":
    main()
