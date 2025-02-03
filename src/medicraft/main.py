import argparse
import logging

from pipeline import Pipeline

parser = argparse.ArgumentParser(description="Run medicraft pipeline.")

parser.add_argument("--file", "-f", type=str, help="Path to the configuration file.", default="config.yml")
parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output.")
parser.add_argument("--step_job_id", type=int, help="Step job id to run.", default=None)


def main():
    """
    Main function to run the medicraft pipeline.

    This function parses the command line arguments, loads the configuration file,
    and runs the pipeline with the specified options.

    Args:
        None

    Returns:
        None
    """
    args = parser.parse_args()
    logging.info(args)

    pipeline = Pipeline()

    pipeline.load_config(args.file)

    raise
    pipeline.run(verbose=args.verbose, step_job_id=args.step_job_id)


if __name__ == "__main__":
    main()
