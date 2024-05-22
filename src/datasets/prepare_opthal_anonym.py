from pathlib import Path

import pandas as pd
from opthal_anonymized import get_csv_dataset

DATASET_FILE_PATH = Path("data/datasets/ophthal_anonym/dataset-splitted.csv")


def main():
    df: pd.DataFrame = get_csv_dataset(DATASET_FILE_PATH)
    print(df["diagnosis"].value_counts())


if __name__ == "__main__":
    main()
