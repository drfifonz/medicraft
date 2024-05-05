from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import umap
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torchvision import transforms as T
from tqdm import tqdm

from config import DEVICE


class Embeddings:
    def __init__(self, model: nn.Module, transform: T.transforms.Compose):
        self.model = self.__remove_last_layer(model)
        self.model.to(DEVICE)
        self.model.eval()

        self.transform = (
            T.transforms.Compose(
                [
                    T.CenterCrop(256),
                    T.Resize(224),
                    T.ToTensor(),
                ]
            )
            if not transform
            else transform
        )

    def __remove_last_layer(self, model: nn.Module) -> nn.Module:
        """
        Remove last layer from model.
        """
        return nn.Sequential(*list(model.children())[:-1])

    def image_to_embedding(
        self,
        image_path: str | Path,
        model: nn.Module,
        transform: T.transforms.Compose,
        img_type: str = "L",
    ) -> torch.Tensor:
        """
        transform image to  embedding with given model.
        """
        img = Image.open(image_path).convert(img_type)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img).squeeze().numpy()
        return embedding

    def get_images_embeddings(self, image_paths: list[str | Path], diagnosis: str) -> dict[str, torch.Tensor]:
        """
        Get embeddings for list of images.
        """
        embeddings = {}
        for image_path in tqdm(image_paths, desc="Getting embeddings"):
            if not isinstance(image_path, Path):
                image_path = Path(image_path)
            embedding = self.image_to_embedding(image_path, self.model, self.transform)
            embeddings[image_path.name] = {
                "embedding": embedding,
                "diagnosis": diagnosis,
            }
        return embeddings

    def get_images_embeddings_from_dataset(self, dataset_df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """
        Get embeddings for list of images.
        """
        embeddings = {}
        for diagnosis in dataset_df["diagnosis"].unique():
            image_paths = dataset_df[dataset_df["diagnosis"] == diagnosis]["image_path"].tolist()
            embeddings.update(self.get_images_embeddings(image_paths, diagnosis))
        return embeddings

    def save_embeddings(self, embeddings: dict[str, torch.Tensor], save_path: str | Path) -> None:
        """
        Save embeddings to file.
        """
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        torch.save(embeddings, save_path)
        print(f"Embeddings saved to {save_path}")

    def load_embeddings(self, embeddings_path: str | Path) -> dict[str, torch.Tensor]:
        """
        Load embeddings from file.
        """
        if not isinstance(embeddings_path, Path):
            embeddings_path = Path(embeddings_path)
        embeddings = torch.load(embeddings_path)
        return embeddings

    def create_umap_plot(self, embeddings: dict[str, torch.Tensor], save_path: str | Path) -> None:
        """
        Create UMAP plot, save it to file.
        """
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        embeddings = torch.stack([v["embedding"] for v in embeddings.values()])
        embedded_data = embeddings.cpu().numpy()

        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
        umap_embedding = umap_model.fit_transform(embedded_data)

        plt.figure(figsize=(8, 8), dpi=200)
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=10)
        plt.title("UMAP Plot")
        plt.savefig(str(save_path))

    def create_violin_plot(self, embeddings_dataframe: pd.DataFrame, save_path: str | Path) -> None:
        """
        Create violin plot, save it to file.
        """
        if not isinstance(save_path, Path):
            save_path = Path(save_path)

        embeddings_dataframe = embeddings_dataframe[
            embeddings_dataframe["diagnosis1"] == embeddings_dataframe["diagnosis2"]
        ]
        embeddings_dataframe["diagnosis"] = embeddings_dataframe["diagnosis1"]
        embeddings_dataframe.drop(columns=["diagnosis1", "diagnosis2"], inplace=True)

        sns.violinplot(x="diagnosis", y="distance", data=embeddings_dataframe)

    def calculate_embeddings_distances(
        self, embedings1: dict[str, torch.Tensor], embedings2: dict[str, torch.Tensor]
    ) -> pd.DataFrame:
        """
        Create pandas dataframe from embeddings.
        """
        data = []
        pbar = tqdm(total=len(embedings1) * len(embedings2), desc="Calculating distances")
        for embending1 in embedings1.values():
            for embending2 in embedings2.values():
                distance = self.calculate_cosine_similarity(
                    torch.tensor(embending1["embedding"]), torch.tensor(embending2["embedding"])
                )
                data.append(
                    {
                        "image1": embending1["image_path"],
                        "diagnosis1": embending1["diagnosis"],
                        "image2": embending2["image_path"],
                        "diagnosis2": embending2["diagnosis"],
                        "distance": distance,
                    }
                )
                pbar.update(1)

        return pd.DataFrame(data)

    def calculate_cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Calculate cosine distance between two embeddings.
        """
        return torch.nn.functional.cosine_similarity(emb1, emb2).item()
