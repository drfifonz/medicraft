import os


def check_results_dir_exists(results_dir: str) -> bool:
    """
    Check if the results directory exists.

    :param results_dir: The path to the results directory.
    :type results_dir: str
    :return: True if the results directory exists, False otherwise.
    :rtype: bool
    """
    return os.path.exists(results_dir)
