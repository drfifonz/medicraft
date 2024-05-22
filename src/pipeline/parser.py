import logging
import sys

import yaml
from pydantic import BaseModel, ValidationError

from pipeline.blocks import ConfigBlocks


def j_print(data, *args, **kwargs):
    import json

    try:
        print(json.dumps(data, indent=4), *args, **kwargs)
    except Exception as e:
        print(data, *args, **kwargs)


def read_config_file(config_file: str) -> dict:
    """
    Parse the configuration file
    """
    try:
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
            return data
    except yaml.YAMLError as e:
        logging.error(f"Error parsing the configuration file: {e}")
        raise e


def create_pipeline(config: dict) -> None:
    """
    Create the pipeline
    """
    pass


def parse_config(config: dict) -> dict[str, BaseModel]:
    """
    Parse and validate the configuration
    """
    results = {}

    unique_essential_blocks = [ConfigBlocks.output.name, ConfigBlocks.general.name]

    for block in ConfigBlocks:
        block_config = config.get(block.name.lower())

        if block.name.lower() == ConfigBlocks.training.name.lower():
            block_config = {**block_config, **get_training_configs(config)}

        # get general configs
        if block.name.lower() not in unique_essential_blocks:
            general_config = config.get(ConfigBlocks.general.name)
            if general_config is None:
                logging.error("General config not found")
                sys.exit("Parsing config failed")

            general_block_config = general_config.get(block.name.lower())

            # print(block_config)
            # print(general_block_config)
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
        j_print(block_instance.model_dump())
        print("OK", block.name)
        print("\n\n")

    return results


def get_training_configs(config: dict) -> dict:
    """
    Get tranining configuration defined in another blocks
    """
    output_config = config.get(ConfigBlocks.output.name.lower())
    general_config = config.get(ConfigBlocks.general.name.lower())
    training_config = config.get(ConfigBlocks.training.name.lower())

    if output_config is None:
        logging.error("Output config not found")
        sys.exit("Parsing config failed")

    if general_config is None:
        logging.error("General config not found")
        sys.exit("Parsing config failed")

    if training_config is None:
        logging.error("Training config not found")
        sys.exit("Parsing config failed")

    training_config = {
        "total_steps": general_config.get("total_steps"),
        "image_size": general_config.get("image_size"),
        "models": general_config.get("models"),
        **training_config,
    }

    return training_config


if __name__ == "__main__":
    config = read_config_file("mock_config.yml")
    # j_print(config)
    print(config.keys())

    parse_config(config)
    print("FINISHED PARSING")
