import os


def check_results_dir_exists(results_dir: str) -> bool:
    return os.path.exists(results_dir)
