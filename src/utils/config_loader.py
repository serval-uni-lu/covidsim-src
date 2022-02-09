import yaml
import sys

DEFAULT_CONFIG = "./config/config.yaml"


def get_config(config_path=None):

    if config_path is None:
        config_path = DEFAULT_CONFIG

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config
