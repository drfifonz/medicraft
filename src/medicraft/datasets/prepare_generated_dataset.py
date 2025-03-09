import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("src")
from medicraft.const import PROJECT_DIR  # noqa: E402


class GeneratedDataset:

    def __init__(self, real_dataset_csv_path: str | Path):
        self.real_dataset_csv_path = (
            Path(real_dataset_csv_path) if isinstance(real_dataset_csv_path, str) else real_dataset_csv_path
        )

    def get_generated_labels_distibution(self, path: str | Path) -> OrderedDict:
        # get dataset distibution by files in directories
        path = Path(path) if isinstance(path, str) else path
        distribution = {dir.name: len(list(dir.iterdir())) for dir in path.iterdir() if dir.is_dir()}
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

    def get_max_generated_stratified_distribution(self, generated_distribution: OrderedDict) -> dict:

        real_distribution = self.get_real_labels_distibution()
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

    def create_csv_file(self, distribution: OrderedDict, file_name: str, dataset_path=str | Path | list[str | Path]):

        # TODO consider if pass both distributions and add extra column in dataset that could be used in loader
        # might be a good idea ^^
        dataset_files_map = self.get_dataset_files_map(dataset_path)

        trimmed_dataset_files_map = {}

        for diagnosis, file_list in dataset_files_map.items():
            if diagnosis in distribution:
                trimmed_dataset_files_map[diagnosis] = file_list[: distribution[diagnosis]]
        df = pd.DataFrame(
            [(k, v) for k, values in trimmed_dataset_files_map.items() for v in values],
            columns=["diagnosis", "filepath"],
        )
        df.to_csv(file_name, index=False)

    def get_dataset_files_map(self, dataset_path: str | Path | list[str | Path]) -> dict:
        if isinstance(dataset_path, (str, Path)):
            dataset_path = [dataset_path]
        dataset_files_map = {}
        for path in dataset_path:
            path = Path(path) if isinstance(path, str) else path
            for dir in path.iterdir():
                if dir.is_dir():
                    if dir.name not in dataset_files_map:
                        dataset_files_map[dir.name] = [
                            str(file.resolve().relative_to(PROJECT_DIR)) for file in dir.iterdir()
                        ]
                    else:
                        dataset_files_map[dir.name] += [
                            str(file.resolve().relative_to(Path(PROJECT_DIR))) for file in dir.iterdir()
                        ]
                else:
                    raise ValueError(f"Path {dir} is not a directory")
        return dataset_files_map

    @staticmethod
    def combine_distributions(distributions: list[OrderedDict]) -> OrderedDict:
        # combine distributions
        combined = {}
        for d in distributions:
            for k, v in d.items():
                if k in combined:
                    combined[k] += v
                else:
                    combined[k] = v
        return OrderedDict(sorted(combined.items()))


if __name__ == "__main__":
    real_dataset_csv_path = Path("data/datasets/ophthal_anonym/dataset.csv")

    distribution_paths = [
        "data/datasets/ophthal_anonym_classed/train",
        "data/datasets/ophthal_anonym_classed/val",
    ]

    dataset = GeneratedDataset(real_dataset_csv_path=real_dataset_csv_path)

    distributions = [dataset.get_generated_labels_distibution(path) for path in distribution_paths]
    combined_dist = dataset.combine_distributions(distributions)
    maximal_stratified_distribution = dataset.get_max_generated_stratified_distribution(combined_dist)

    print("Total combined distribution:\t", dict(combined_dist))
    print("Max stratified distribution:\t", maximal_stratified_distribution)

    dataset.create_csv_file(
        distribution=maximal_stratified_distribution,
        file_name=real_dataset_csv_path.parent / "test_stratification.csv",
        dataset_path=distribution_paths,
    )
    print("Done")
