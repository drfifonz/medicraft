import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

    def create_csv_files(self, distribution: OrderedDict, file_name: str, dataset_path=str | Path | list[str | Path]):

        # TODO consider if pass both distributions and add extra column in dataset that could be used in loader
        # might be a good idea ^^
        dataset_files_map = self.get_dataset_files_map(dataset_path)

        file_name = Path(file_name) if isinstance(file_name, str) else file_name
        all_file_name = file_name.parent / (file_name.stem + "_all.csv")
        trimmed_file_name = file_name.parent / (file_name.stem + "_trimmed.csv")

        df_all = pd.DataFrame(
            [(k, v) for k, values in dataset_files_map.items() for v in values],
            columns=["diagnosis", "filepath"],
        )
        df_all = add_split_type_column(df_all)
        df_all.to_csv(all_file_name, index=False)

        trimmed_dataset_files_map = {}
        for diagnosis, file_list in dataset_files_map.items():
            if diagnosis in distribution:
                trimmed_dataset_files_map[diagnosis] = file_list[: distribution[diagnosis]]
        df_trimmed = pd.DataFrame(
            [(k, v) for k, values in trimmed_dataset_files_map.items() for v in values],
            columns=["diagnosis", "filepath"],
        )
        df_trimmed = add_split_type_column(df_trimmed)
        df_trimmed.to_csv(trimmed_file_name, index=False)

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


def add_split_type_column(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    #TODO consider changing splitting to don not use test_ratio and do 2 splits
    """
    # Check if ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("The sum of train, validation, and test ratios must be 1.0")

    # Create a copy of the dataframe to avoid modifying the original
    df_with_split = df.copy()

    # Calculate relative ratio for first split (train vs. rest)
    rest_ratio = val_ratio + test_ratio
    train_vs_rest_ratio = train_ratio / (train_ratio + rest_ratio)

    # Calculate relative ratio for second split (val vs. test)
    val_vs_test_ratio = val_ratio / rest_ratio
    print("CREATING SPLITTING RATIO")
    print(f"{train_vs_rest_ratio=}, {val_vs_test_ratio=}")
    # Initialize split_type column with 'test' values
    df_with_split["split_type"] = "test"

    # First split: train vs. rest (val+test)
    train_indices, rest_indices = train_test_split(
        np.arange(len(df_with_split)), train_size=train_ratio, random_state=random_state
    )

    # Mark train samples
    df_with_split.loc[train_indices, "split_type"] = "train"

    # Second split: val vs. test
    val_indices, test_indices = train_test_split(
        rest_indices, train_size=val_ratio / (val_ratio + test_ratio), random_state=random_state
    )

    # Mark validation samples
    df_with_split.loc[val_indices, "split_type"] = "val"

    return df_with_split


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
    dataset.create_csv_files(
        distribution=maximal_stratified_distribution,
        file_name=real_dataset_csv_path.parent / "v2" / "test_stratification.csv",
        dataset_path=distribution_paths,
    )
    print("Done")
