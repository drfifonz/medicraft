import argparse
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from medicraft.datasets import EyeScansV2


def denormalize(tensor: Tensor, mean: list[float], std: list[float]) -> Tensor:
    """
    Undo the normalization for a given image tensor using the specified mean and std.
    """

    for c in range(tensor.shape[0]):
        tensor[c] = (tensor[c] * std[c]) + mean[c]
    return torch.clamp(tensor, 0.0, 1.0)


def generate_smote_images(
    data_loader: DataLoader,
    save_dir: Union[str, Path],
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> pd.DataFrame:
    """
    Generates new images via SMOTE for balancing classes.

    :param data_loader: A torch DataLoader providing (image, label) pairs.
                       The images should already be normalized if you plan to invert them for saving.
    :param save_dir: Directory path where generated images are stored.
    :param mean: Normalization mean used in original dataset.
    :param std: Normalization std used in original dataset.
    :return: A pandas DataFrame with columns ['diagnosis', 'filepath'] for the newly saved images.
    """

    # Ensure save_dir is a Path object
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images and labels from the data_loader
    all_images = []
    all_labels = []

    for batch_images, batch_labels in data_loader:
        # batch_images shape: [batch_size, C, H, W]
        # batch_labels shape: [batch_size]
        # Flatten the images so we can feed them to SMOTE
        batch_size = batch_images.size(0)
        flattened = batch_images.view(batch_size, -1)  # [batch_size, C*H*W]
        all_images.append(flattened)
        all_labels.append(batch_labels)

    # Stack everything into a single array
    images_tensor = torch.cat(all_images, dim=0)  # shape: [N, C*H*W]
    labels_tensor = torch.cat(all_labels, dim=0)  # shape: [N]

    # Convert to NumPy arrays for SMOTE
    X = images_tensor.numpy()
    y = labels_tensor.numpy()

    # Apply SMOTE to oversample
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Original dataset size: {len(X)}, Resampled dataset size: {len(X_resampled)}")

    # Convert the resampled data back to torch Tensors
    X_resampled_tensor = torch.from_numpy(X_resampled)
    y_resampled_tensor = torch.from_numpy(y_resampled)

    # Reshape images back to (C, H, W)
    # We can infer C, H, W from the first batch in our original loader:
    c = data_loader.dataset[0][0].shape[0]  # or from the transform, e.g. 3
    h = data_loader.dataset[0][0].shape[1]  # e.g. 256
    w = data_loader.dataset[0][0].shape[2]  # e.g. 512
    X_resampled_tensor = X_resampled_tensor.view(-1, c, h, w)

    # DataFrame to store the generated sample info
    df = pd.DataFrame(columns=["diagnosis", "filepath"])

    # Save each new synthetic sample as an image
    for idx, (img_tensor, label) in enumerate(zip(X_resampled_tensor, y_resampled_tensor)):
        # Denormalize (if your original data was normalized)
        img_tensor = denormalize(img_tensor, mean, std)

        # Convert to PIL
        pil_image = to_pil_image(img_tensor)

        # Save to disk
        filename = f"smote_{idx}_class_{label.item()}.png"
        filepath = save_dir / filename
        pil_image.save(filepath)

        # Add record to DataFrame
        df.loc[len(df)] = [label.item(), str(filepath)]

    return df


if __name__ == "__main__":
    SAVE_DATASET_DIR = Path(".results/datasets/phd/smote_0000")
    SAVE_CSV_PATH = Path("data/datasets/ophthal_anonym/v2/smote_dataset.csv")

    REAL_DATASET_PATH = Path("data/datasets/ophthal_anonym/v2/real_dataset.csv")

    parser = argparse.ArgumentParser(description="Generate SMOTE images for class balancing")
    parser.add_argument("--real_dataset", "-r", type=str, default=REAL_DATASET_PATH)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--num_workers", "-w", type=int, default=4)
    args = parser.parse_args()

    SAVE_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    real_dm = EyeScansV2(
        batch_size=args.batch_size,
        dataset_csv_file=Path(args.real_dataset),
        num_workers=args.num_workers,
        transforms=transform,
    )
    real_dm.setup()
    real_loader: DataLoader = real_dm.train_dataloader()

    df_smote = generate_smote_images(real_loader, SAVE_DATASET_DIR)

    df_smote.to_csv(SAVE_CSV_PATH, index=False)

    print(f"Saved SMOTE‐generated images to: {SAVE_DATASET_DIR}")
    print(f"DataFrame with new samples saved to: {SAVE_CSV_PATH}")
