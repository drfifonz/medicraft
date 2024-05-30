import logging
from pathlib import Path
from shutil import copyfile

from tqdm import tqdm


def copy_results_directory(src_dir: str, dest_dir: str, absolute_paths: tuple[bool] = (False, False)) -> None:
    """
    Copy the results directory to the destination directory.

    :param src_dir: Source directory.
    :param dest_dir: Destination directory.
    :param absolute_paths: Tuple of two booleans, if True, the source or destination paths will be absolute.
    :return: None
    :raises FileNotFoundError: If the source directory does not exist.
    :raises NotADirectoryError: If the source directory is not a directory.
    """
    src_dir = Path(src_dir) if not absolute_paths[0] else Path("/") / src_dir
    dest_dir = Path(dest_dir) if not absolute_paths[1] else Path("/") / dest_dir

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source directory {src_dir} is not a directory.")

    if not dest_dir.exists():
        logging.info(f"Creating destination directory {dest_dir}")
        dest_dir.mkdir(parents=True)

    total_num_files = sum(1 for _ in src_dir.rglob("*") if _.is_file())

    for src_file in tqdm(src_dir.rglob("**/*"), total=total_num_files, desc="Coping files"):
        if src_file.is_file():
            relative_path = src_file.relative_to(src_dir)
            dest_file = dest_dir / relative_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            tqdm.write(f"Copying {src_file} to {dest_file}")
            copyfile(src_file, dest_file)
