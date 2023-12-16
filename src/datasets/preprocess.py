from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

DATA_ROOT_PATH = Path("data")
EASY_CASES_PATH = DATA_ROOT_PATH / "easy_cases_with_fluid"
REFERENCE_EYE_DIR_NAME = "healthy_eye"  # TODO rename to reference_eye

IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]


def get_patients_paths(dataset_dir: Path) -> list[Path]:
    """Returns list of paths to patients directories in given directory."""
    return [p for p in dataset_dir.iterdir() if p.is_dir()]


def get_images_paths(images_dir: Path) -> Optional[list[Path]]:
    """Returns list of paths to images in given directory."""
    try:
        return [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_FORMATS]
    except FileNotFoundError:
        return None


def get_lesion_eyes_paths(dataset_dir: Path) -> Optional[list[Path]]:
    """Returns list of paths to all reference (second, healfy) eyes images in dataset directory."""
    lesion_eyes_paths = []

    for patient_dir in get_patients_paths(dataset_dir):
        patient_examinations = [p for p in patient_dir.iterdir() if p.is_dir() and p.name != REFERENCE_EYE_DIR_NAME]
        for examination_dir in patient_examinations:
            lesion_eye_photos = get_images_paths(examination_dir)
            if lesion_eye_photos:
                lesion_eyes_paths.extend(lesion_eye_photos)
    return lesion_eyes_paths


def get_reference_eyes_paths(dataset_dir: Path) -> Optional[list[Path]]:
    """Returns list of images' paths to all eyes with lesion without reference eye."""
    healthy_eyes_paths = []

    for patient_dir in get_patients_paths(dataset_dir):
        healthy_eye_dir = patient_dir / REFERENCE_EYE_DIR_NAME
        healthy_eye_photos = get_images_paths(healthy_eye_dir)
        if healthy_eye_photos:
            healthy_eyes_paths.extend(healthy_eye_photos)
    return healthy_eyes_paths


def resize_images_and_save(
    images_paths: list[Path],
    output_dir_path: str,
    size: tuple[int, int],
) -> None:
    """Resizes images from given paths and saves them to given directory."""
    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images_paths):
        image = Image.open(image_path).resize(size)
        image.save(output_dir_path / image_path.name)


if __name__ == "__main__":
    patiens_paths = get_patients_paths(EASY_CASES_PATH)

    # print(get_healty_eyes_paths(patiens_paths[0]))
    healthy_eyes_paths = get_reference_eyes_paths(EASY_CASES_PATH)

    print(f"Healthy eyes: {len(healthy_eyes_paths)}")
    print(healthy_eyes_paths[0])
    lesion_eyes_paths = get_lesion_eyes_paths(EASY_CASES_PATH)
    print(f"Lesion eyes : {len(lesion_eyes_paths)}")
    print(lesion_eyes_paths[0])

    SIZE = (256, 128)
    OUTPUT_DIR = "data/healthy_eyes_" + str("x".join([str(x) for x in SIZE]))
    print(f"Resize to {SIZE}", OUTPUT_DIR)
    resize_images_and_save(healthy_eyes_paths, OUTPUT_DIR, SIZE)
