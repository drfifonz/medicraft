from pathlib import Path

import pandas as pd
from prepare_generated_dataset import add_split_type_column


def create_df_for_real_dataset(real_dataset_csv_path: Path | str, images_dir: str | Path) -> pd.DataFrame:
    data = {}
    images_dir = Path(images_dir) if isinstance(images_dir, str) else images_dir
    df = pd.read_csv(real_dataset_csv_path)
    df = df[df["image_type"] == "OCT"]

    data["reference"] = df[df["reference_eye"] == True]["filename"].tolist()  # noqa:E712
    for diagnosis in df["diagnosis"].unique():
        x = df.loc[df["reference_eye"] == False]  # noqa:E712
        data[diagnosis] = x.loc[df["diagnosis"] == diagnosis]["filename"].tolist()

    print("reference", len(data["reference"]))
    for diagnosis in df["diagnosis"].unique():
        print(diagnosis, len(data[diagnosis]))

    for diagnosis, filenames in data.items():
        data[diagnosis] = [str(images_dir / filename) for filename in filenames]

    return pd.DataFrame(
        [(k, v) for k, values in data.items() for v in values],
        columns=["diagnosis", "filepath"],
    )


if __name__ == "__main__":
    REAL_DATASET_CSV_FILE = Path("data/datasets/ophthal_anonym/dataset.csv")

    SAVE_PATH = Path("data/datasets/ophthal_anonym/v2/real_dataset.csv")

    df = create_df_for_real_dataset(
        real_dataset_csv_path=REAL_DATASET_CSV_FILE,
        images_dir=REAL_DATASET_CSV_FILE.parent / "images",
    )
    min_count = df["diagnosis"].value_counts().min()
    df_balanced = df.groupby("diagnosis", group_keys=False).apply(lambda x: x.sample(min_count, random_state=42))
    df_balanced = df_balanced.reset_index(drop=True)

    df = add_split_type_column(df)
    df_balanced = add_split_type_column(df_balanced)

    df.to_csv(SAVE_PATH.parent / f"{SAVE_PATH.stem}{SAVE_PATH.suffix}", index=False)
    df_balanced.to_csv(SAVE_PATH.parent / f"{SAVE_PATH.stem}_stratified{SAVE_PATH.suffix}", index=False)
    print("Saved stratified and non stratified versions to", SAVE_PATH.parent)
    print("done")
