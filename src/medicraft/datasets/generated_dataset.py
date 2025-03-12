import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class GeneratedOCTDataset(Dataset):

    def __init__(
        self,
        csv_file: str | Path,
        transforms=None,
        split_type: Literal["train", "val", "test"] | None = None,
    ):

        df = pd.read_csv(csv_file)
        self.df = df[df["split_type"] == "train"] if split_type else pd.read_csv(csv_file)

        self.transforms = transforms

        # Create a mapping from labels to class indices (similar to ImageFolder)
        self.labels = sorted(self.df["diagnosis"].unique())
        self.class_to_idx = {label: i for i, label in enumerate(self.labels)}  # TODO use maping from sklearn

        # Generate samples list in the format (path, class_idx)
        self.samples = []
        for i, row in self.df.iterrows():
            path = row["filepath"]
            class_idx = self.class_to_idx[row["diagnosis"]]
            self.samples.append((path, class_idx))

        # For compatibility with ImageFolder
        self.imgs = self.samples
        self.targets = [sample[1] for sample in self.samples]
        logging.info(f"Dataset {split_type+' '}loaded")

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, target = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.samples)


def show_batch(images, labels, class_names):
    """Helper function to display a batch of images"""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    for i, (img, label) in enumerate(zip(images, labels)):
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        if len(images) > 1:
            ax = axes[i]
        else:
            ax = axes

        ax.imshow(img)
        ax.set_title(f"Class: {class_names[label]}")
        ax.axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Set paths to your real data
    csv_file = "data/datasets/ophthal_anonym/test_stratification.csv"  # CSV with diagnosis and filepath columns

    # Create a transform for the images
    transform = transforms.Compose(
        [
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create the dataset
    print("Loading OCT dataset...")
    dataset = GeneratedOCTDataset(csv_file=csv_file, transforms=transform)

    # Print dataset information
    print(f"Dataset loaded with {len(dataset)} images")
    print(f"Classes: {dataset.labels}")
    print(f"Label mapping: {dataset.class_to_idx}")

    # Check class distribution
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.labels[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images ({count/len(dataset)*100:.1f}%)")

    # Create data loader
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Iterate through a few batches
    print("\nFetching sample batches:")
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Image tensor shape: {images.shape}")
        print(f"  Labels: {labels.numpy()}")

        # Map label indices back to class names
        class_names = [dataset.labels[label] for label in labels]
        print(f"  Classes: {class_names}")

        # Display images if possible
        try:
            fig = show_batch(images, labels, dataset.labels)
            plt.show()
        except Exception as e:
            print(f"  Could not display images: {e}")

        # Only process 3 batches for this test
        if i >= 2:
            break

    # Create train/val split
    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Set fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    print("\nSplit dataset:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")

    # Create data loaders for split datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Check one batch from each loader
    print("\nTraining batch sample:")
    train_images, train_labels = next(iter(train_loader))
    train_classes = [dataset.labels[label] for label in train_labels]
    print(f"  Shape: {train_images.shape}")
    print(f"  Classes: {train_classes}")

    print("\nValidation batch sample:")
    val_images, val_labels = next(iter(val_loader))
    val_classes = [dataset.labels[label] for label in val_labels]
    print(f"  Shape: {val_images.shape}")
    print(f"  Classes: {val_classes}")

    print("\nGeneratedOCTDataset test completed successfully!")
