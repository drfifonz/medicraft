import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3

from medicraft.datasets import EyeScansV2


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super(InceptionV3FeatureExtractor, self).__init__()
        # Load pretrained InceptionV3 with aux_logits disabled.
        inception = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        inception.eval()
        self.inception = inception

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # InceptionV3 expects 299x299 images.
        # Forward through the network until the final pooling layer to extract features.
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        # Final average pooling and flattening to get the feature vector.
        x = self.inception.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class FIDCalculator:
    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: str = device
        self.model = InceptionV3FeatureExtractor().to(self.device)
        self.model.eval()

    def get_activations(self, dataloader: DataLoader) -> np.ndarray:
        activations = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                features = self.model(images)
                activations.append(features)
        activations = torch.cat(activations, dim=0)
        return activations.cpu().numpy()

    def calculate_fid(self, loader1: DataLoader, loader2: DataLoader) -> float:
        act1: np.ndarray = self.get_activations(loader1)
        act2: np.ndarray = self.get_activations(loader2)

        mu1: np.ndarray = np.mean(act1, axis=0)
        sigma1: np.ndarray = np.cov(act1, rowvar=False)
        mu2: np.ndarray = np.mean(act2, axis=0)
        sigma2: np.ndarray = np.cov(act2, rowvar=False)
        fid: float = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid

    def calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        diff: np.ndarray = mu1 - mu2
        # Compute the square root of the product of covariance matrices.
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid: float = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid


if __name__ == "__main__":
    # Parse command line arguments.
    REAL_DATASET_PATH = Path("data/datasets/ophthal_anonym/v2/real_dataset.csv")

    parser = argparse.ArgumentParser(description="Calculate FID between two image datasets")
    parser.add_argument("--real_dataset", "-r", type=str, default=REAL_DATASET_PATH)
    parser.add_argument("--syntetic_dataset", "-s", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--num_workers", "-w", type=int, default=4)
    args = parser.parse_args()
    if not Path(args.syntetic_dataset).is_file():
        generated_csv = REAL_DATASET_PATH.parent / args.syntetic_dataset
        if not generated_csv.is_file():
            raise FileNotFoundError(f"Dataset file not found: {args.syntetic_dataset}")
    else:
        generated_csv = Path(args.syntetic_dataset)

    real_csv = Path(args.real_dataset)
    # Define image transformations: resize to 299x299, convert to tensor, and normalize as per InceptionV3.
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Convert CSV file paths to Path objects.

    # Instantiate the data modules using EyeScansV2.
    real_dm = EyeScansV2(
        batch_size=args.batch_size, dataset_csv_file=real_csv, num_workers=args.num_workers, transforms=transform
    )
    generated_dm = EyeScansV2(
        batch_size=args.batch_size, dataset_csv_file=generated_csv, num_workers=args.num_workers, transforms=transform
    )

    # Prepare the data modules (this usually sets up the datasets internally).
    real_dm.setup()
    generated_dm.setup()

    # Get the DataLoaders (assuming the train_dataloader() returns the DataLoader we need).
    real_loader: DataLoader = real_dm.train_dataloader()
    generated_loader: DataLoader = generated_dm.train_dataloader()

    # Calculate FID.
    fid_calculator = FIDCalculator()
    fid_value: float = fid_calculator.calculate_fid(real_loader, generated_loader)
    print("FID:", fid_value)
