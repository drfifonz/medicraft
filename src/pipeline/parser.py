import json
import logging
import sys

import yaml
from pydantic import BaseModel, ValidationError

import config as cfg
from pipeline.blocks import ConfigBlocks


def j_print(data, *args, **kwargs):
    """
    Print the given data in JSON format if possible, otherwise print it as is.
    For development purposes only.

    Args:
        data: The data to be printed.
        *args: Additional positional arguments to be passed to the print function.
        **kwargs: Additional keyword arguments to be passed to the print function.
    """

    try:
        print(json.dumps(data, indent=4), *args, **kwargs)
    except Exception:
        print(data, *args, **kwargs)


def read_config_file(config_file: str) -> dict:
    """
    Parse the configuration file

    :param config_file: The path to the configuration file
    :type config_file: str
    :return: The parsed configuration data
    :rtype: dict
    :raises yaml.YAMLError: If there is an error parsing the configuration file
    """
    try:
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
            return data
    except yaml.YAMLError as e:
        logging.error(f"Error parsing the configuration file: {e}")
        raise e


def parse_config(config: dict) -> dict[str, BaseModel]:
    """
    Parse and validate the configuration

    :param config: The configuration dictionary to be parsed and validated
    :type config: dict
    :return: A dictionary containing the parsed and validated configuration blocks
    :rtype: dict[str, BaseModel]
    """
    results = {}

    unique_essential_blocks = [ConfigBlocks.output.name, ConfigBlocks.general.name]

    for block in ConfigBlocks:
        block_config = config.get(block.name.lower())

        if block.name.lower() == ConfigBlocks.experiment.name.lower():
            block_config = {**block_config, **get_experiment_configs(config)}

        # get general configs
        if block.name.lower() not in unique_essential_blocks:
            general_config = config.get(ConfigBlocks.general.name)
            if general_config is None:
                logging.error("General config not found")
                sys.exit("Parsing config failed")

            general_block_config = general_config.get(block.name.lower())

            block_config = {**general_block_config, **block_config} if general_block_config else block_config

        try:
            block_instance: BaseModel = block.value(**block_config)
            results[block.name.lower()] = block_instance

        except ValidationError as e:
            logging.error(f"Error while processing block: {block.name} ")
            for e in e.errors():
                logging.error(
                    f"Error type: \033[1m{e['type']}\033[0m \tfor \033[1m{e['loc']}\033[0m\t| Error message: {e['msg']}"
                )
            sys.exit("Parsing config failed")
        if block.name == ConfigBlocks.experiment.name and cfg.DEV_DEBUG:
            j_print(block_instance.model_dump())
            print("OK", block.name)

    return results


def get_experiment_configs(config: dict) -> dict:
    """
    Get training configuration defined in other blocks.

    :param config: The configuration dictionary.
    :type config: dict
    :return: The experiment configuration dictionary.
    :rtype: dict
    """
    output_config = config.get(ConfigBlocks.output.name.lower())
    general_config = config.get(ConfigBlocks.general.name.lower())
    experiment_config = config.get(ConfigBlocks.experiment.name.lower())

    if output_config is None:
        logging.error("Output config not found")
        sys.exit("Parsing config failed")

    if general_config is None:
        logging.error("General config not found")
        sys.exit("Parsing config failed")

    if experiment_config is None:
        logging.error("Experiment config not found")
        sys.exit("Parsing config failed")

    experiment_config = {
        "total_steps": general_config.get("total_steps", 0),
        "image_size": general_config.get("image_size"),
        "experiment_id": general_config.get("experiment_id"),
        "models": general_config.get("models"),
        "results_dir": output_config.get("results_dir"),
        "copy_results_to": output_config.get("copy_results_to"),
        **experiment_config,
    }

    return experiment_config
