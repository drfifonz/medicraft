import argparse
from pathlib import Path

import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from medicraft.config import WANDB_PRJ_NAME_FID
from medicraft.datasets import EyeScansV2
from medicraft.validation import FIDCalculator

if __name__ == "__main__":
    REAL_DATASET_PATH = Path("data/datasets/ophthal_anonym/v2/real_dataset.csv")
    parser = argparse.ArgumentParser(description="Calculate FID between two image datasets")
    parser.add_argument("--real_dataset", "-r", type=str, default=REAL_DATASET_PATH)
    parser.add_argument("--syntetic_dataset", "-s", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--num_workers", "-w", type=int, default=4)
    parser.add_argument("--wandb_name", "-n", type=str)
    args = parser.parse_args()
    if not Path(args.syntetic_dataset).is_file():
        generated_csv = REAL_DATASET_PATH.parent / args.syntetic_dataset
        if not generated_csv.is_file():
            raise FileNotFoundError(f"Dataset file not found: {args.syntetic_dataset}")
    else:
        generated_csv = Path(args.syntetic_dataset)

    real_csv = Path(args.real_dataset)
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    config = {
        **vars(args),
        "transformations": str(transform),
        "model": {"name": "InceptionV3", "pretrained": True},
    }

    wandb.init(
        project=WANDB_PRJ_NAME_FID,
        config=config,
        name=args.wandb_name,
    )

    real_dm = EyeScansV2(
        batch_size=args.batch_size, dataset_csv_file=real_csv, num_workers=args.num_workers, transforms=transform
    )
    generated_dm = EyeScansV2(
        batch_size=args.batch_size, dataset_csv_file=generated_csv, num_workers=args.num_workers, transforms=transform
    )

    real_dm.setup()
    generated_dm.setup()

    real_loader: DataLoader = real_dm.train_dataloader()
    generated_loader: DataLoader = generated_dm.train_dataloader()

    fid_calculator = FIDCalculator()
    fid_value: float = fid_calculator.calculate_fid(real_loader, generated_loader)
    print("FID:", fid_value)
    wandb.log(
        {
            "FID": fid_value,
            "device": fid_calculator.device,
            "real_dataset": str(real_csv.stem),
            "synthetic_dataset": str(generated_csv.stem),
        }
    )

    wandb.finish()
