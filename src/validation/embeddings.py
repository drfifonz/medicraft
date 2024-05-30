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
        """
        Initialize the Embeddings class.

        Args:
            model (nn.Module): The neural network model.
            transform (T.transforms.Compose): The image transformation pipeline.
        """
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
        Remove the last layer from the model.

        :param model: The input model.
        :type model: nn.Module
        :return: The modified model with the last layer removed.
        :rtype: nn.Module
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
        Transform an image to an embedding using the given model.

        :param image_path: The path to the image file.
        :param model: The neural network model used for embedding.
        :param transform: The image transformation pipeline.
        :param img_type: The image type (default: "L").
        :return: The embedding tensor.
        """
        img = Image.open(image_path).convert(img_type)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img).squeeze().numpy()
        return embedding

    def get_images_embeddings(self, image_paths: list[str | Path], diagnosis: str) -> dict[str, torch.Tensor]:
        """
        Get embeddings for list of images.

        :param image_paths: List of image paths.
        :type image_paths: list[str | Path]
        :param diagnosis: Diagnosis for the images.
        :type diagnosis: str
        :return: Dictionary containing image embeddings.
        :rtype: dict[str, torch.Tensor]
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

        :param dataset_df: DataFrame containing image paths and diagnoses.
        :type dataset_df: pd.DataFrame
        :return: Dictionary containing embeddings for each image.
        :rtype: dict[str, torch.Tensor]
        """
        embeddings = {}
        for diagnosis in dataset_df["diagnosis"].unique():
            image_paths = dataset_df[dataset_df["diagnosis"] == diagnosis]["image_path"].tolist()
            embeddings.update(self.get_images_embeddings(image_paths, diagnosis))
        return embeddings

    def save_embeddings(self, embeddings: dict[str, torch.Tensor], save_path: str | Path) -> None:
        """
        Save embeddings to file.

        :param embeddings: A dictionary containing embeddings as values and their corresponding keys.
        :type embeddings: dict[str, torch.Tensor]
        :param save_path: The path where the embeddings will be saved.
        :type save_path: str or Path
        :return: None
        """
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        torch.save(embeddings, save_path)
        print(f"Embeddings saved to {save_path}")

    def load_embeddings(self, embeddings_path: str | Path) -> dict[str, torch.Tensor]:
        """
        Load embeddings from file.

        :param embeddings_path: The path to the embeddings file.
        :type embeddings_path: str | Path
        :return: A dictionary mapping strings to torch.Tensor objects representing the embeddings.
        :rtype: dict[str, torch.Tensor]
        """
        if not isinstance(embeddings_path, Path):
            embeddings_path = Path(embeddings_path)
        embeddings = torch.load(embeddings_path)
        return embeddings

    def create_umap_plot(self, embeddings: dict[str, torch.Tensor], save_path: str | Path) -> None:
        """
        Create UMAP plot, save it to file.

        :param embeddings: A dictionary containing embeddings as values.
        :type embeddings: dict[str, torch.Tensor]
        :param save_path: The path to save the UMAP plot.
        :type save_path: str or Path
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

        :param embeddings_dataframe: DataFrame containing embeddings data.
        :type embeddings_dataframe: pd.DataFrame
        :param save_path: Path to save the violin plot.
        :type save_path: str or Path
        :return: None
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

        :param embedings1: A dictionary containing embeddings for the first set of images.
        :type embedings1: dict[str, torch.Tensor]
        :param embedings2: A dictionary containing embeddings for the second set of images.
        :type embedings2: dict[str, torch.Tensor]
        :return: A pandas DataFrame containing the calculated distances between embeddings.
        :rtype: pd.DataFrame
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

        :param emb1: The first embedding.
        :type emb1: torch.Tensor
        :param emb2: The second embedding.
        :type emb2: torch.Tensor
        :return: The cosine similarity between the two embeddings.
        :rtype: float
        """
        return torch.nn.functional.cosine_similarity(emb1, emb2).item()
