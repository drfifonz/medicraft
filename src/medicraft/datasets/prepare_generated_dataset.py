from pathlib import Path

import numpy as np
import pandas as pd


class GeneratedDataset:
    def __init__(self, dir_path: str | Path):
        self.path = Path(dir_path) if isinstance(dir_path, str) else dir_path

    @classmethod
    def get_generated_labels_distibution(cls) -> dict:
        # get num of image files in each subdirectory
        return {dir.name: len(list(dir.iterdir())) for dir in cls.path.iterdir() if dir.is_dir()}

    @classmethod
    def get_real_labels_distibution(self, csv_file_path: str | Path) -> dict:
        df = pd.read_csv(csv_file_path)
        df = df[df["image_type"] == "OCT"]

        distribution = {
            "reference": len(df[df["reference_eye"] == True]),  # noqa:E712
        }
        for diagnosis in df["diagnosis"].unique():
            distribution[diagnosis] = len(df[df["reference_eye"] == True][df["diagnosis"] == diagnosis])  # noqa:E712
        return distribution

    @classmethod
    def get_max_generated_stratified_distribution(cls):
        # get ratio to biggest class
        real_distribution = cls.get_real_labels_distibution()
        generated_distribution = cls.get_generated_labels_distibution()
        # ensure dicts are same key sorted
        assert real_distribution.keys() == generated_distribution.keys()
        real_distribution = dict(sorted(real_distribution.items()))
        generated_distribution = dict(sorted(generated_distribution.items()))
        multiplier = cls.max_scalar(
            np.array(list(real_distribution.values())), np.array(list(generated_distribution.values()))
        )
        return {k: int(v * multiplier) for k, v in generated_distribution.items()}

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
                files = list(cls.path / diagnosis.iterdir())
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
        df.to_csv(cls.path / "generated_dataset.csv", index=False)
