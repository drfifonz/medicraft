from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


class GeneratedDataset:

    def __init__(self, synth_dir_path: str | Path, real_dataset_csv_path: str | Path):
        self.synth_dir_pathpath = Path(synth_dir_path) if isinstance(synth_dir_path, str) else synth_dir_path
        self.real_dataset_csv_path = real_dataset_csv_path

    def get_generated_labels_distibution(self) -> OrderedDict:
        # get dataset distibution by files in directories
        distribution = {dir.name: len(list(dir.iterdir())) for dir in self.synth_dir_pathpath.iterdir() if dir.is_dir()}
        return OrderedDict(sorted(distribution.items()))

    def get_real_labels_distibution(self) -> OrderedDict:
        df = pd.read_csv(self.real_dataset_csv_path)
        df = df[df["image_type"] == "OCT"]

        distribution = {
            "reference": len(df[df["reference_eye"] == True]),  # noqa:E712
        }
        for diagnosis in df["diagnosis"].unique():
            x = df.loc[df["reference_eye"] == False]  # noqa:E712
            distribution[diagnosis] = len(x.loc[df["diagnosis"] == diagnosis])

        return OrderedDict(sorted(distribution.items()))

    def get_max_generated_stratified_distribution(self) -> dict:

        real_distribution = self.get_real_labels_distibution()
        generated_distribution = self.get_generated_labels_distibution()

        assert real_distribution.keys() == generated_distribution.keys()

        multiplier = self.max_scalar(
            np.array(list(real_distribution.values())), np.array(list(generated_distribution.values()))
        )

        return {k: int(v * multiplier) for k, v in real_distribution.items()}

    @staticmethod
    def max_scalar(A, B):
        # where A i real labels and B is generated labels
        mask = A > 0
        alpha = np.min(B[mask] / A[mask]) if np.any(mask) else float("inf")
        return alpha

    @classmethod
    def create_csv_file(cls, distribution: dict):

        # TODO consider if pass both distributions and add extra column in dataset that could be used in loader
        # might be a good idea ^^
        df = pd.DataFrame(columns=["filename", "diagnosis", "reference_eye"])
        for diagnosis, num in distribution.items():
            for i in range(num):
                # get list of files in directory
                files = list(cls.synth_dir_pathpath / diagnosis.iterdir())
                if diagnosis == "reference":
                    df = df.append(
                        {"file_path": str(files[num]), "diagnosis": diagnosis, "reference_eye": True},
                        ignore_index=True,
                    )
                else:
                    df = df.append(
                        {"file_path": str(files[num]), "diagnosis": diagnosis, "reference_eye": False},
                        ignore_index=True,
                    )
        df.to_csv(cls.synth_dir_pathpath / "generated_dataset.csv", index=False)


if __name__ == "__main__":
    dataset = GeneratedDataset(
        synth_dir_path="data/datasets/ophthal_anonym_classed/train",
        real_dataset_csv_path="data/datasets/ophthal_anonym/dataset.csv",
    )
    max_stratification = dataset.get_max_generated_stratified_distribution()
    print(max_stratification)
