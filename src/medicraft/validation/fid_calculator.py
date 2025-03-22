from typing import Optional

import numpy as np
import torch
from models import InceptionV3FeatureExtractor
from scipy import linalg
from torch.utils.data import DataLoader


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
